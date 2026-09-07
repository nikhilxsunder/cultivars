# filepath: /src/cultivars/univariate/unobserved_components.py
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

"""Structural time series: decomposition with a likelihood.

Harvey's unobserved-components family writes a series as the sum of
interpretable pieces -- a trend, a damped stochastic cycle, a stochastic
seasonal, an irregular -- each with its own law of motion, and estimates
them jointly by exact maximum likelihood through the Kalman filter. That
is the difference between this and an ad-hoc filter: the components come
with a likelihood, information criteria, and full-sample uncertainty
bands, and the end-of-sample estimates degrade honestly (their variances
grow) instead of silently, which is where the HP filter is at its worst.

Three trend specifications are offered, in Harvey's nomenclature: the
*local level* (a random-walk level), the *local linear trend* (level and
slope both stochastic), and the *smooth trend* (an integrated random walk
-- only the slope moves, so the level is twice-integrated noise and the
trend is stiff). The cycle is the damped stochastic cycle, a bivariate
rotation with amplitude decay ``rho`` and frequency ``lambda_c``, whose
implied period ``2 pi / lambda_c`` is estimated, not imposed. The
seasonal is the stochastic trigonometric form, all harmonics sharing one
innovation variance.

One initialization statement rather than a hidden constant: the
nonstationary states (trend, seasonal) start from an approximate-diffuse
prior with variance ``1e6``, the cycle from its exact stationary
covariance. The likelihood is exact Gaussian given that prior; a strict
diffuse treatment differs in the first few observations' contributions.

References:
    Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and
        the Kalman Filter*. Cambridge University Press.
    Clark, P. K. (1987). The cyclical component of U.S. economic
        activity. *Quarterly Journal of Economics*, 102(4), 797-814.
    Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State
        Space Methods* (2nd ed.). Oxford University Press.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core import SummaryTable
from .._internals import (
    _ComparisonMixin,
    _SeriesMixin,
    _structural_matrices,
    _StructuralFit,
    _StructuralParameters,
    _SummaryMixin,
    _UnobservedComponentsModel,
)
from ..exceptions import SpecificationError
from ..state_space import LinearGaussianSSM

__all__ = ["UnobservedComponents", "UnobservedComponentsResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class UnobservedComponentsResult(_SummaryMixin, _SeriesMixin, _ComparisonMixin):
    """A fitted structural decomposition: components, variances, and bands.

    Attributes:
        endog: The observed series.
        trend: The trend specification (``"level"``, ``"lltrend"``, or
            ``"smooth"``).
        has_cycle: Whether a damped stochastic cycle was estimated.
        seasonal_period: The seasonal period, or ``None``.
        sigma2_irregular: Observation noise variance.
        sigma2_level: Level innovation variance, or ``None`` (smooth
            trend).
        sigma2_slope: Slope innovation variance, or ``None``.
        cycle_rho: Cycle damping, or ``None``.
        cycle_freq: Cycle frequency in radians, or ``None``.
        sigma2_cycle: Cycle innovation variance, or ``None``.
        sigma2_seasonal: Seasonal innovation variance, or ``None``.
        llf: Exact Gaussian log-likelihood (approximate-diffuse
            initialization, as documented on the module).
        nobs: Observations.
        n_params: Free parameters the likelihood was maximized over.
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    trend: str
    has_cycle: bool
    seasonal_period: int | None
    sigma2_irregular: float
    sigma2_level: float | None
    sigma2_slope: float | None
    cycle_rho: float | None
    cycle_freq: float | None
    sigma2_cycle: float | None
    sigma2_seasonal: float | None
    llf: float
    nobs: int
    n_params: float
    _smoothed_state: npt.NDArray[np.float64] = field(repr=False)
    _smoothed_state_cov: npt.NDArray[np.float64] = field(repr=False)
    _slices: dict[str, slice] = field(repr=False)

    @classmethod
    def _from_fit(
        cls, fit: _StructuralFit, model: _UnobservedComponentsModel[UnobservedComponentsResult]
    ) -> UnobservedComponentsResult:
        """Assemble the public result from a raw fit and its specification."""
        p = fit.params
        return cls(
            endog=model.endog,
            trend=model.trend,
            has_cycle=model.cycle,
            seasonal_period=model.seasonal,
            sigma2_irregular=p.sigma2_irregular,
            sigma2_level=p.sigma2_level,
            sigma2_slope=p.sigma2_slope,
            cycle_rho=p.cycle_rho,
            cycle_freq=p.cycle_freq,
            sigma2_cycle=p.sigma2_cycle,
            sigma2_seasonal=p.sigma2_seasonal,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=float(fit.n_params),
            _smoothed_state=fit.smoothed_state,
            _smoothed_state_cov=fit.smoothed_state_cov,
            _slices=fit.slices,
        )

    @property
    def _params(self) -> _StructuralParameters:
        """The parameter record, rebuilt for the system builder."""
        return _StructuralParameters(
            sigma2_irregular=self.sigma2_irregular,
            sigma2_level=self.sigma2_level,
            sigma2_slope=self.sigma2_slope,
            cycle_rho=self.cycle_rho,
            cycle_freq=self.cycle_freq,
            sigma2_cycle=self.sigma2_cycle,
            sigma2_seasonal=self.sigma2_seasonal,
        )

    def _component(self, name: str) -> npt.NDArray[np.float64]:
        """One component's smoothed path, or a refusal naming what exists.

        Raises:
            SpecificationError: If the specification lacks the component.
        """
        if name not in self._slices:
            raise SpecificationError(
                f"this specification has no {name} component; it carries {sorted(self._slices)}."
            )
        block = self._slices[name]
        if name == "seasonal":
            design = _structural_matrices(
                self._params,
                trend=self.trend,
                cycle=self.has_cycle,
                seasonal=self.seasonal_period,
            )[0]
            pattern = design[0, block]
            return np.asarray(self._smoothed_state[:, block] @ pattern, dtype=np.float64)
        return np.asarray(self._smoothed_state[:, block.start], dtype=np.float64)

    def _component_std(self, name: str) -> npt.NDArray[np.float64]:
        """One component's smoothed standard deviation path."""
        block = self._slices[name]
        if name == "seasonal":
            design = _structural_matrices(
                self._params,
                trend=self.trend,
                cycle=self.has_cycle,
                seasonal=self.seasonal_period,
            )[0]
            pattern = design[0, block]
            variances = np.einsum(
                "i,tij,j->t",
                pattern,
                self._smoothed_state_cov[:, block, block],
                pattern,
            )
            return np.asarray(np.sqrt(np.maximum(variances, 0.0)), dtype=np.float64)
        return np.asarray(
            np.sqrt(np.maximum(self._smoothed_state_cov[:, block.start, block.start], 0.0)),
            dtype=np.float64,
        )

    @property
    def level(self) -> npt.NDArray[np.float64]:
        """Smoothed trend level, ``(n,)``."""
        return self._component("level")

    @property
    def level_std(self) -> npt.NDArray[np.float64]:
        """Smoothed standard deviation of the level, ``(n,)``."""
        return self._component_std("level")

    @property
    def slope(self) -> npt.NDArray[np.float64]:
        """Smoothed trend slope, ``(n,)``.

        Raises:
            SpecificationError: If the trend has no slope.
        """
        return self._component("slope")

    @property
    def cycle(self) -> npt.NDArray[np.float64]:
        """Smoothed cycle, ``(n,)``.

        Raises:
            SpecificationError: If no cycle was specified.
        """
        return self._component("cycle")

    @property
    def cycle_period(self) -> float:
        """The estimated cycle period ``2 pi / lambda_c`` in observations.

        Raises:
            SpecificationError: If no cycle was specified.
        """
        if self.cycle_freq is None:
            raise SpecificationError("this specification has no cycle component.")
        return float(2.0 * np.pi / self.cycle_freq)

    @property
    def seasonal(self) -> npt.NDArray[np.float64]:
        """Smoothed seasonal, ``(n,)``.

        Raises:
            SpecificationError: If no seasonal was specified.
        """
        return self._component("seasonal")

    @property
    def signal(self) -> npt.NDArray[np.float64]:
        """The systematic part: everything except the irregular."""
        total = self._component("level").copy()
        if "cycle" in self._slices:
            total += self._component("cycle")
        if "seasonal" in self._slices:
            total += self._component("seasonal")
        return total

    @property
    def irregular(self) -> npt.NDArray[np.float64]:
        """The observed series minus the smoothed signal."""
        return np.asarray(self.endog - self.signal, dtype=np.float64)

    @property
    def state_space(self) -> LinearGaussianSSM:
        """The fitted system, re-applicable to data it was not estimated on.

        The exact linear-Gaussian emitter: filtering a new series through
        it reads that series with this decomposition's estimated
        variances, and its ``loglikelihood`` on the estimation sample
        reproduces ``llf``.
        """
        model, _ = LinearGaussianSSM._from_structural_system(
            self._params,
            trend=self.trend,
            cycle=self.has_cycle,
            seasonal=self.seasonal_period,
        )
        return model

    def forecast(self, steps: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Out-of-sample forecasts with honest standard errors.

        Filters the estimation sample to the last state, then propagates
        mean and covariance through the fitted system.

        Args:
            steps: Horizons ahead, at least 1.

        Returns:
            ``(mean, std)`` arrays of length ``steps``; the standard
            errors include state uncertainty and the irregular, but not
            parameter uncertainty.

        Raises:
            SpecificationError: If ``steps`` is not positive.
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        design, transition, selection, state_cov, obs_cov, _, _ = _structural_matrices(
            self._params,
            trend=self.trend,
            cycle=self.has_cycle,
            seasonal=self.seasonal_period,
        )
        forward = self.state_space.filter(self.endog)
        mean = forward.filtered_state[-1].copy()
        cov = forward.filtered_state_cov[-1].copy()
        noise = selection @ state_cov @ selection.T
        out_mean = np.empty(steps)
        out_std = np.empty(steps)
        for step in range(steps):
            mean = transition @ mean
            cov = transition @ cov @ transition.T + noise
            out_mean[step] = float(design[0] @ mean)
            out_std[step] = float(np.sqrt(max(design[0] @ cov @ design[0] + obs_cov[0, 0], 0.0)))
        return out_mean, out_std

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """Aligned per-observation output: the decomposition itself."""
        out: dict[str, npt.NDArray[np.float64]] = {"observed": self.endog}
        out["level"] = self.level
        if "slope" in self._slices:
            out["slope"] = self.slope
        if "cycle" in self._slices:
            out["cycle"] = self.cycle
        if "seasonal" in self._slices:
            out["seasonal"] = self.seasonal
        out["irregular"] = self.irregular
        return out

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        pieces = [self.trend]
        if self.has_cycle:
            pieces.append("cycle")
        if self.seasonal_period is not None:
            pieces.append(f"seasonal({self.seasonal_period})")
        return "UC[" + "+".join(pieces) + "]"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        ic = self.information_criteria
        rows: list[tuple[str, str]] = [("sigma2.irregular", f"{self.sigma2_irregular:.6g}")]
        if self.sigma2_level is not None:
            rows.append(("sigma2.level", f"{self.sigma2_level:.6g}"))
        if self.sigma2_slope is not None:
            rows.append(("sigma2.slope", f"{self.sigma2_slope:.6g}"))
        if self.cycle_rho is not None and self.cycle_freq is not None:
            rows.append(("cycle.rho", f"{self.cycle_rho:.4f}"))
            rows.append(("cycle.frequency", f"{self.cycle_freq:.4f}"))
            assert self.sigma2_cycle is not None
            rows.append(("sigma2.cycle", f"{self.sigma2_cycle:.6g}"))
        if self.sigma2_seasonal is not None:
            rows.append(("sigma2.seasonal", f"{self.sigma2_seasonal:.6g}"))
        notes = [
            "Exact Gaussian likelihood under an approximate-diffuse prior "
            "(variance 1e6) on the nonstationary states; the cycle starts "
            "at its stationary covariance.",
            "Component paths are full-sample (smoothed); their bands widen "
            "at the sample ends, which is the honest behavior an ad-hoc "
            "filter hides.",
        ]
        if self.cycle_freq is not None:
            notes.insert(
                0,
                f"The estimated cycle period is {self.cycle_period:.1f} observations.",
            )
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=(
                ("Trend", self.trend),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("AIC", f"{ic.aic:.3f}"),
                ("Parameters", f"{self.n_params:.0f}"),
                ("BIC", f"{ic.bic:.3f}"),
            ),
            columns=("", "estimate"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class UnobservedComponents(_UnobservedComponentsModel[UnobservedComponentsResult]):
    """Structural time-series decomposition by exact maximum likelihood.

    Args:
        endog: The observed series.
        trend: ``"level"`` for a local level, ``"lltrend"`` for a local
            linear trend, ``"smooth"`` for an integrated random walk.
        cycle: Whether to estimate a damped stochastic cycle (damping,
            frequency, and variance all estimated).
        seasonal: Trigonometric seasonal period, or ``None``.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 300
        >>> level = np.cumsum(0.1 * rng.standard_normal(n))
        >>> y = level + 0.5 * rng.standard_normal(n)
        >>> res = UnobservedComponents(y, trend="level").fit()
        >>> bool(np.mean((res.level - level) ** 2) < np.mean((y - level) ** 2))
        True
    """

    def fit(self) -> UnobservedComponentsResult:
        """Maximize the exact likelihood and smooth every component."""
        return UnobservedComponentsResult._from_fit(self._fit_structural(), self)
