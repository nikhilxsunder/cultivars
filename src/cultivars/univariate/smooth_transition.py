# filepath: /src/cultivars/univariate/smooth_transition.py
#
# Copyright (c) 2026 Nikhil Sunder
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Smooth-transition autoregressive models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import SummaryTable, trailing_lag
from .._internals import (
    _ObservedRegimeResult,
    _SmoothTransitionFit,
    _SmoothTransitionModel,
)
from ..state_space import NonlinearSSM

__all__ = ["ESTAR", "LSTAR", "STARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class STARResult(_ObservedRegimeResult):
    """A fitted smooth-transition autoregression.

    Attributes:
        transition: ``"logistic"`` or ``"exponential"``.
        gamma: Transition speed, in units of standard deviations of the
            transition variable.
        transition_scale: The standard deviation ``gamma`` is expressed
            against, retained so the weight path can be reproduced and so
            ``gamma`` can be read on the original scale.
    """

    transition: str
    gamma: float
    transition_scale: float

    @classmethod
    def _from_fit(
        cls, fit: _SmoothTransitionFit, model: _SmoothTransitionModel[STARResult]
    ) -> STARResult:
        """Assemble the public result from a raw fit and its specification."""
        values = trailing_lag(model.endog, delay=fit.delay, length=fit.nobs)
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            order=model.order,
            delay=fit.delay,
            threshold=fit.threshold,
            threshold_values=values,
            lower_params=fit.lower_params,
            upper_params=fit.upper_params,
            sigma2=fit.sigma2,
            ssr=fit.ssr,
            transition=model.transition,
            gamma=fit.gamma,
            transition_scale=float(np.std(values)),
        )

    @property
    def regime_weight(self) -> npt.NDArray[np.float64]:
        """The transition function evaluated over the sample.

        Reproduces the estimator's own weighting exactly, including the
        clipping of the exponent -- an extreme ``gamma`` saturates the weight
        rather than overflowing ``exp``. The logistic form is monotone in the
        standardized transition variable, so the two regimes are asymmetric;
        the exponential form is symmetric about the threshold, so both tails
        share a regime and the middle is the other.
        """
        u = (self.threshold_values - self.threshold) / self.transition_scale
        if self.transition == "logistic":
            return 1.0 / (1.0 + np.exp(-np.clip(self.gamma * u, -50.0, 50.0)))
        return 1.0 - np.exp(-np.clip(self.gamma * u**2, 0.0, 50.0))

    @property
    def is_effectively_abrupt(self) -> bool:
        """Whether the fitted transition is so fast it is a hard threshold.

        A large ``gamma`` drives the logistic weight to an indicator, at which
        point the smooth model is a :class:`SETAR` with two extra parameters
        and a flat likelihood in ``gamma``. Reported so the flatness is visible
        rather than mistaken for a converged estimate.
        """
        return self.transition == "logistic" and self.gamma > 100.0

    def nonlinear_state_space(self, measurement_variance: float) -> NonlinearSSM:
        """The fitted smooth-transition mean as a noisily observed state space.

        The errors-in-variables reading of the fit: the latent state is
        the lag stack of the *true* series (deep enough to hold both the
        autoregression and the transition lag), the transition blends the
        two regime autoregressions by the fitted weight function --
        reproduced exactly, exponent clipping included -- and the
        observation reads the current level through Gaussian measurement
        error of the stated variance, which is required and strictly
        positive because with none the model is observation-driven and
        filtering it is vacuous. Unlike the hard-threshold emitter this
        transition is smooth, so the extended and unscented filters are
        trustworthy readers alongside the particle filter; the initial
        state is diffuse at the sample mean and variance.

        Args:
            measurement_variance: Observation noise variance the emitted
                system is read through, strictly positive.

        Returns:
            The :class:`_NonlinearStateSpaceModel`.

        Raises:
            SpecificationError: If the variance is not strictly positive.
        """
        lower = np.asarray(self.lower_params, dtype=np.float64)
        upper = np.asarray(self.upper_params, dtype=np.float64)
        order, delay = self.order, self.delay
        threshold, gamma = self.threshold, self.gamma
        scale, kind = self.transition_scale, self.transition
        depth = max(order, delay)

        def mean_map(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            base = np.column_stack([np.ones(states.shape[0]), states[:, :order]])
            u = (states[:, delay - 1] - threshold) / scale
            if kind == "logistic":
                weight = 1.0 / (1.0 + np.exp(-np.clip(gamma * u, -50.0, 50.0)))
            else:
                weight = 1.0 - np.exp(-np.clip(gamma * u**2, 0.0, 50.0))
            return np.asarray(
                (1.0 - weight) * (base @ lower) + weight * (base @ upper),
                dtype=np.float64,
            )

        return NonlinearSSM._lag_stack_state_space(
            mean_map,
            order=depth,
            sigma2=self.sigma2,
            measurement_variance=float(measurement_variance),
            center=float(np.mean(self.endog)),
            spread=float(np.var(self.endog)),
        )

    def _transition_params(self) -> dict[str, float]:
        """The transition speed, inserted between the threshold and the variance."""
        return {"gamma": self.gamma}

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        family = "LSTAR" if self.transition == "logistic" else "ESTAR"
        return f"{family}({self.order}, d={self.delay})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        shape = (
            "monotone in the transition variable"
            if self.transition == "logistic"
            else "symmetric about the threshold"
        )
        notes = [
            self._stationarity_note(),
            f"Transition: {self.transition}, {shape}. "
            f"gamma = {self.gamma:.4f} per s.d. of the transition variable "
            f"(s.d. = {self.transition_scale:.4f}); "
            f"mean upper weight = {self.upper_fraction:.3f}.",
        ]
        if self.is_effectively_abrupt:
            notes.append(
                "gamma is large enough that the transition is effectively a hard "
                "threshold; the likelihood is nearly flat in gamma here, so prefer "
                "SETAR unless the smooth form is required."
            )
        notes.append(
            "The threshold and delay are not identified under linearity, so information "
            "criteria rank specifications but do not test for nonlinearity."
        )
        notes.append("Standard errors are not yet available for this estimator.")
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=self._summary_metadata(),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=tuple(notes),
        )


class LSTAR(_SmoothTransitionModel[STARResult]):
    """Logistic smooth-transition autoregression.

    The weight rises monotonically through the threshold, so the two regimes
    are genuinely different states and the dynamics differ above and below.
    Use this when the asymmetry has a direction -- contractions behaving
    differently from expansions, for instance.

    Args:
        endog: The series.
        order: Autoregressive order within each regime.
        delay: Delay of the transition variable ``y_{t-d}``.
    """

    __slots__ = ()

    def __init__(self, endog: npt.ArrayLike, *, order: int, delay: int = 1) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, order=order, transition="logistic", delay=delay)

    def fit(self) -> STARResult:
        """Estimate the transition by concentrated least squares."""
        return STARResult._from_fit(self._fit_family(), self)


class ESTAR(_SmoothTransitionModel[STARResult]):
    """Exponential smooth-transition autoregression.

    The weight is symmetric about the threshold, so both tails share one regime
    and the middle is the other. This is the natural form for mean reversion
    that strengthens with distance from equilibrium -- the canonical
    application is a real exchange rate under transaction costs, where small
    deviations persist and large ones are arbitraged away.

    Args:
        endog: The series.
        order: Autoregressive order within each regime.
        delay: Delay of the transition variable ``y_{t-d}``.
    """

    __slots__ = ()

    def __init__(self, endog: npt.ArrayLike, *, order: int, delay: int = 1) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, order=order, transition="exponential", delay=delay)

    def fit(self) -> STARResult:
        """Estimate the transition by concentrated least squares."""
        return STARResult._from_fit(self._fit_family(), self)
