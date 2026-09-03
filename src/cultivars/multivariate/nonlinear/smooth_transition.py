# filepath: /src/cultivars/multivariate/nonlinear/smooth_transition.py
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

"""The smooth-transition vector autoregression: regimes as a dial, not a switch.

An STVAR replaces the threshold VAR's indicator with a transition function
taking values in ``[0, 1]``: at every date the system is a convex blend of two
regimes, with the blend weight read off an observable. The canonical modern
use is Auerbach and Gorodnichenko's fiscal multipliers, where the weight is a
smoothed output-growth measure and the finding is that government spending
works differently in recessions than in expansions -- a statement the smooth
form can make with dates that are 70% recession, which no hard split can
represent (Auerbach & Gorodnichenko 2012).

One class carries both transition shapes, because the multivariate literature
names the model STVAR whichever ``G`` it uses: ``transition="logistic"`` is
monotone in the transition variable, so the regimes are directional states --
recession versus expansion -- and ``"exponential"`` is symmetric about the
threshold, so both tails share a regime and the middle is the other. (The
univariate family splits these into LSTAR and ESTAR because those are
established names; LSTVAR/ESTVAR are not.)

Estimation concentrates: conditional on the transition speed and location the
model is linear, so the regime coefficients come from one multivariate
least-squares solve and only ``(gamma, c)`` is searched, derivative-free and
multi-start, on the log-determinant of the residual covariance. The innovation
covariance is common across regimes here, and the fit record says why: smooth
weights never partition the sample, so a per-regime covariance has no
subsample to be estimated from.

Everything the threshold module says about honesty carries over: propagation
is regime-conditional, and the chi-squared linearity test is refused because
``gamma`` and ``c`` are not identified under the null (Davies).

References:
    Auerbach, A. J., & Gorodnichenko, Y. (2012). Measuring the output
        responses to fiscal policy. *American Economic Journal: Economic
        Policy*, 4(2), 1-27.
    Terasvirta, T., & Yang, Y. (2014). Specification, estimation and
        evaluation of vector smooth transition autoregressive models with
        applications. *CREATES Research Paper* 2014-08.
    Granger, C. W. J., & Terasvirta, T. (1993). *Modelling Nonlinear Economic
        Relationships*. Oxford University Press.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import Regime, SummaryTable
from ..._internals import (
    _SmoothTransitionVectorAutoRegressionModel,
    _VectorObservedRegimeResult,
    _VectorSmoothTransitionFit,
)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class STVARResult(_VectorObservedRegimeResult):
    """A fitted smooth-transition vector autoregression.

    Attributes:
        transition: ``"logistic"`` or ``"exponential"``.
        gamma: Transition speed, per standard deviation of the transition
            variable.
        transition_scale: The standard deviation ``gamma`` is expressed
            against, retained so the weight path can be reproduced and so
            ``gamma`` can be read on the original scale.
        sigma_u: Common innovation covariance, dof-corrected. One covariance
            rather than two: smooth weights never partition the sample, so a
            per-regime covariance has no subsample to be estimated from.
    """

    transition: str
    gamma: float
    transition_scale: float
    sigma_u: npt.NDArray[np.float64] = field(repr=False)

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorSmoothTransitionFit,
        model: _SmoothTransitionVectorAutoRegressionModel[STVARResult],
    ) -> STVARResult:
        """Assemble the public result from a raw fit and its specification."""
        return cls(
            endog=model.endog,
            names=model.names,
            order=model.order,
            trend=model.trend,
            delay=fit.delay,
            threshold=fit.threshold,
            threshold_name=model.transition_name,
            threshold_values=fit.threshold_values,
            transition_series=model.transition_series,
            lower_coefficients=fit.lower_coefficients,
            upper_coefficients=fit.upper_coefficients,
            lower_deterministic=fit.lower_deterministic,
            upper_deterministic=fit.upper_deterministic,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            transition=model.transition,
            gamma=fit.gamma,
            transition_scale=fit.transition_scale,
            sigma_u=fit.sigma_u,
        )

    @property
    def regime_weight(self) -> npt.NDArray[np.float64]:
        """The transition function evaluated over the sample.

        Reproduces the estimator's own weighting exactly, including the
        clipping of the exponent -- an extreme ``gamma`` saturates the weight
        rather than overflowing ``exp``. The logistic form is monotone in the
        standardized transition variable, so the two regimes are directional;
        the exponential form is symmetric about the threshold, so both tails
        share a regime and the middle is the other.
        """
        return self._weight_at(self.threshold_values)

    def _weight_at(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The transition function at arbitrary transition values."""
        u = (values - self.threshold) / self.transition_scale
        if self.transition == "logistic":
            return 1.0 / (1.0 + np.exp(-np.clip(self.gamma * u, -50.0, 50.0)))
        return 1.0 - np.exp(-np.clip(self.gamma * u**2, 0.0, 50.0))

    def _regime_sigma(self, regime: Regime) -> npt.NDArray[np.float64]:
        """The common innovation covariance, whichever regime is named."""
        return self.sigma_u

    @property
    def is_effectively_abrupt(self) -> bool:
        """Whether the fitted transition is so fast it is a hard threshold.

        A large ``gamma`` drives the logistic weight to an indicator, at which
        point the smooth model is a :class:`~cultivars.multivariate.nonlinear.TVAR`
        with two extra parameters and a flat likelihood in ``gamma``. Reported
        so the flatness is visible rather than mistaken for a converged
        estimate.
        """
        return self.transition == "logistic" and self.gamma > 100.0

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table."""
        return f"STVAR({self.order}, {self.transition}, d={self.delay})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
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
            "The innovation covariance is common across regimes: smooth "
            "weights never partition the sample, so a per-regime covariance "
            "has no subsample to be estimated from.",
        ]
        if self.is_effectively_abrupt:
            notes.append(
                "gamma is large enough that the transition is effectively a "
                "hard threshold; the likelihood is nearly flat in gamma here, "
                "so prefer TVAR unless the smooth form is required."
            )
        notes.append(
            "The threshold, delay, and transition speed are not identified "
            "under linearity, so information criteria rank specifications but "
            "do not test for nonlinearity."
        )
        notes.append(self._regime_conditional_note())
        notes.append("Standard errors are not yet available for this estimator.")
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=(*self._summary_metadata(), ("Gamma", f"{self.gamma:.4f}")),
            columns=("equation", "lower own L1", "upper own L1"),
            rows=self._own_lag_rows(),
            notes=tuple(notes),
        )


class STVAR(_SmoothTransitionVectorAutoRegressionModel[STVARResult]):
    """Smooth-transition vector autoregression, Auerbach-Gorodnichenko style.

    At every date the system is a convex blend of two regimes, weighted by a
    logistic or exponential function of an observable -- a named column's own
    lag, or an external series such as a smoothed growth measure.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order within each regime.
        transition_variable: A variable name from ``names`` (self-exciting)
            or an aligned external series.
        transition: ``"logistic"`` for directional regimes, ``"exponential"``
            for distance-from-threshold regimes.
        delay: Delay of the transition variable.
        trend: Deterministic terms per regime.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((300, 2))
        >>> for t in range(1, 300):
        ...     g = 1.0 / (1.0 + np.exp(-5.0 * y[t - 1, 0]))
        ...     a = (1.0 - g) * 0.6 + g * -0.3
        ...     y[t] = a * y[t - 1] + rng.standard_normal(2)
        >>> res = STVAR(y, order=1, transition_variable="y1", delay=1).fit()
        >>> 0.0 < res.upper_fraction < 1.0
        True
        >>> res.forecast(3).shape
        (3, 2)
    """

    __slots__ = ()

    def fit(self) -> STVARResult:
        """Estimate the transition by concentrated maximum likelihood."""
        return STVARResult._from_fit(self._fit_regimes(), self)
