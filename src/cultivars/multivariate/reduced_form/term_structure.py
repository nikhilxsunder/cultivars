# filepath: /src/cultivars/multivariate/reduced_form/term_structure.py
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

"""The dynamic Nelson-Siegel model: the yield curve as three moving factors.

Diebold and Li (2006) turned the Nelson-Siegel curve from a per-date fit
into a time-series model: the curve at every date is a combination of a
level, a slope, and a curvature factor with fixed exponential loadings,
and the factors themselves evolve. Diebold, Rudebusch, and Aruoba (2006)
wrote that as one state space -- factors follow autoregressions, curves
observe them through the loadings with per-maturity measurement noise --
and that is the form estimated here, by exact maximum likelihood through
the Kalman filter in a single step rather than the two-step regression
shortcut (which survives as the optimizer's warm start).

The one-step form earns its keep three ways. The likelihood is exact and
one number, so specifications compare honestly. The measurement errors
are per-maturity, so illiquid points on the curve are down-weighted by
their own fitted noise instead of contaminating the factors. And missing
observations cost nothing: the substrate's filter handles element-wise
missingness, so ragged panels -- maturities that enter and leave the
sample -- estimate without imputation.

Factor dynamics are diagonal autoregressions, Diebold-Li's own
parsimonious choice; the factor innovations are correlated through a full
covariance. The loading decay ``lambda`` is estimated by default and can
be fixed for comparability with published two-step results.

References:
    Nelson, C. R., & Siegel, A. F. (1987). Parsimonious modeling of yield
        curves. *Journal of Business*, 60(4), 473-489.
    Diebold, F. X., & Li, C. (2006). Forecasting the term structure of
        government bond yields. *Journal of Econometrics*, 130(2),
        337-364.
    Diebold, F. X., Rudebusch, G. D., & Aruoba, S. B. (2006). The
        macroeconomy and the yield curve: A dynamic latent factor
        approach. *Journal of Econometrics*, 131(1-2), 309-338.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable, _nelson_siegel_loadings
from ..._internals import (
    _ComparisonMixin,
    _maximize_likelihood,
    _NelsonSiegelFit,
    _NelsonSiegelObjective,
    _NelsonSiegelParameters,
    _SummaryMixin,
)
from ...exceptions import DimensionError, NumericalError, SpecificationError
from ...state_space import LinearGaussianSSM

__all__ = ["DynamicNelsonSiegel", "DynamicNelsonSiegelResult", "TimeVaryingNelsonSiegelResult", "TimeVaryingNelsonSiegel"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class DynamicNelsonSiegelResult(_SummaryMixin, _ComparisonMixin):
    """A fitted dynamic Nelson-Siegel model: factors, decay, and noise.

    Attributes:
        panel: The observed ``(nobs, p)`` yield panel (missing entries
            ``numpy.nan``).
        maturities: The ``(p,)`` maturities.
        decay: The fitted loading decay ``lambda``.
        decay_fixed: Whether the decay was held fixed rather than
            estimated.
        mu: ``(3,)`` factor means (level, slope, curvature).
        ar: ``(3,)`` diagonal factor persistences.
        state_innovation_cov: ``(3, 3)`` factor innovation covariance.
        measurement_var: ``(p,)`` per-maturity measurement variances.
        factors: ``(nobs, 3)`` smoothed factor paths.
        factor_cov: ``(nobs, 3, 3)`` smoothed factor covariances.
        llf: Exact Gaussian log-likelihood.
        nobs: Curve dates.
        n_params: Free parameters the likelihood was maximized over.
    """

    panel: npt.NDArray[np.float64] = field(repr=False)
    maturities: npt.NDArray[np.float64]
    decay: float
    decay_fixed: bool
    mu: npt.NDArray[np.float64]
    ar: npt.NDArray[np.float64]
    state_innovation_cov: npt.NDArray[np.float64] = field(repr=False)
    measurement_var: npt.NDArray[np.float64] = field(repr=False)
    factors: npt.NDArray[np.float64] = field(repr=False)
    factor_cov: npt.NDArray[np.float64] = field(repr=False)
    llf: float
    nobs: int
    n_params: float

    @property
    def n_maturities(self) -> int:
        """Points on the curve."""
        return int(self.maturities.shape[0])

    @property
    def level(self) -> npt.NDArray[np.float64]:
        """The smoothed level factor (the long end), ``(nobs,)``."""
        return np.asarray(self.factors[:, 0], dtype=np.float64)

    @property
    def slope(self) -> npt.NDArray[np.float64]:
        """The smoothed slope factor, ``(nobs,)``."""
        return np.asarray(self.factors[:, 1], dtype=np.float64)

    @property
    def curvature(self) -> npt.NDArray[np.float64]:
        """The smoothed curvature factor, ``(nobs,)``."""
        return np.asarray(self.factors[:, 2], dtype=np.float64)

    @property
    def loadings(self) -> npt.NDArray[np.float64]:
        """The ``(p, 3)`` Nelson-Siegel loadings at the fitted decay."""
        return _nelson_siegel_loadings(self.maturities, self.decay)

    @property
    def _params(self) -> _NelsonSiegelParameters:
        """The parameter record, rebuilt for the system builder."""
        return _NelsonSiegelParameters(
            decay=self.decay,
            mu=self.mu,
            ar=self.ar,
            state_chol=np.linalg.cholesky(self.state_innovation_cov),
            obs_var=self.measurement_var,
        )

    @property
    def state_space(self) -> LinearGaussianSSM:
        """The fitted system, re-applicable to data it was not estimated on.

        The exact linear-Gaussian emitter: its ``loglikelihood`` on the
        estimation panel reproduces ``llf``, and filtering a different
        panel (or the same maturities over new dates) reads it with this
        fit's factor dynamics and noise.
        """
        return LinearGaussianSSM._from_nelson_siegel_system(self._params, self.maturities)

    def fitted_curves(self) -> npt.NDArray[np.float64]:
        """Smoothed fitted yields, ``(nobs, p)``."""
        return np.asarray(self.factors @ self.loadings.T, dtype=np.float64)

    def curve(
        self, date_index: int, maturities: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """The fitted curve at one date, on any maturity grid.

        Args:
            date_index: Row of the panel (negative indexing allowed).
            maturities: Strictly positive maturities to evaluate on;
                defaults to the estimation grid.

        Returns:
            Fitted yields at those maturities.

        Raises:
            SpecificationError: If a requested maturity is not strictly
                positive.
        """
        grid = (
            self.maturities
            if maturities is None
            else np.asarray(maturities, dtype=np.float64).ravel()
        )
        if np.any(grid <= 0.0):
            raise SpecificationError("maturities must be strictly positive.")
        basis = _nelson_siegel_loadings(grid, self.decay)
        return np.asarray(basis @ self.factors[date_index], dtype=np.float64)

    def forecast(self, steps: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Curve forecasts with honest standard errors.

        Filters the estimation panel to the last factor state, then
        propagates mean and covariance through the fitted dynamics.

        Args:
            steps: Horizons ahead, at least 1.

        Returns:
            ``(mean, std)`` arrays of shape ``(steps, p)`` on the
            estimation maturities; the standard errors include factor
            uncertainty and measurement noise, but not parameter
            uncertainty.

        Raises:
            SpecificationError: If ``steps`` is not positive.
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        forward = self.state_space.filter(self.panel)
        mean = forward.filtered_state[-1].copy()
        cov = forward.filtered_state_cov[-1].copy()
        transition = np.diag(self.ar)
        basis = self.loadings
        out_mean = np.empty((steps, self.n_maturities))
        out_std = np.empty((steps, self.n_maturities))
        for step in range(steps):
            mean = self.mu + transition @ (mean - self.mu)
            cov = transition @ cov @ transition.T + self.state_innovation_cov
            out_mean[step] = basis @ mean
            spread = np.einsum("pi,ij,pj->p", basis, cov, basis)
            out_std[step] = np.sqrt(np.maximum(spread + self.measurement_var, 0.0))
        return out_mean, out_std

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        tag = "fixed" if self.decay_fixed else "estimated"
        return f"DNS(lambda {tag})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        ic = self.information_criteria
        rows = (
            (
                "level",
                f"{self.mu[0]:.4f}",
                f"{self.ar[0]:.4f}",
                f"{np.sqrt(self.state_innovation_cov[0, 0]):.4f}",
            ),
            (
                "slope",
                f"{self.mu[1]:.4f}",
                f"{self.ar[1]:.4f}",
                f"{np.sqrt(self.state_innovation_cov[1, 1]):.4f}",
            ),
            (
                "curvature",
                f"{self.mu[2]:.4f}",
                f"{self.ar[2]:.4f}",
                f"{np.sqrt(self.state_innovation_cov[2, 2]):.4f}",
            ),
        )
        notes = [
            f"Loading decay lambda = {self.decay:.4f} "
            f"({'held fixed' if self.decay_fixed else 'estimated'}); the "
            "curvature loading peaks near maturity "
            f"{1.79 / self.decay:.2f}.",
            "Estimated in one step by exact maximum likelihood through the "
            "Kalman filter; the Diebold-Li two-step estimator is the warm "
            "start, not the answer.",
            "Measurement noise is per-maturity, so noisy points on the "
            "curve are down-weighted rather than contaminating the "
            "factors; missing entries are handled element-wise.",
        ]
        return SummaryTable(
            title="Dynamic Nelson-Siegel Results",
            metadata=(
                ("Dates", f"{self.nobs}"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Maturities", f"{self.n_maturities}"),
                ("AIC", f"{ic.aic:.3f}"),
                ("Parameters", f"{self.n_params:.0f}"),
                ("BIC", f"{ic.bic:.3f}"),
            ),
            columns=("factor", "mean", "persistence", "innovation sd"),
            rows=rows,
            notes=tuple(notes),
        )


class DynamicNelsonSiegel:
    """One-step maximum-likelihood dynamic Nelson-Siegel estimation.

    Args:
        panel: The ``(nobs, p)`` yield panel, one row per date and one
            column per maturity; ``numpy.nan`` entries are missing and
            handled element-wise, so ragged panels estimate without
            imputation.
        maturities: The ``(p,)`` strictly positive maturities, in
            whatever time unit the decay should be quoted in.
        decay: A loading decay to hold fixed, or ``None`` (default) to
            estimate it by maximum likelihood.

    Raises:
        DimensionError: If the panel and maturities disagree, or the
            panel is too short.
        SpecificationError: If a maturity or a fixed decay is not
            strictly positive.
        NumericalError: If the panel has rows with no finite entries at
            all.

    Example:
        >>> import numpy as np
        >>> from cultivars._core import _nelson_siegel_loadings
        >>> rng = np.random.default_rng(0)
        >>> taus = np.array([0.25, 1.0, 2.0, 5.0, 10.0])
        >>> basis = _nelson_siegel_loadings(taus, 0.6)
        >>> f = np.zeros((120, 3))
        >>> mu = np.array([5.0, -1.5, 0.5])
        >>> f[0] = mu
        >>> for t in range(1, 120):
        ...     f[t] = mu + 0.9 * (f[t - 1] - mu) + 0.2 * rng.standard_normal(3)
        >>> curves = f @ basis.T + 0.05 * rng.standard_normal((120, 5))
        >>> res = DynamicNelsonSiegel(curves, taus, decay=0.6).fit()
        >>> bool(np.corrcoef(res.level, f[:, 0])[0, 1] > 0.95)
        True
    """

    __slots__ = ("_fixed_decay", "_maturities", "_panel")

    def __init__(
        self,
        panel: npt.ArrayLike,
        maturities: npt.ArrayLike,
        *,
        decay: float | None = None,
    ) -> None:
        """Validate the panel, the maturity grid, and any fixed decay."""
        block = np.asarray(panel, dtype=np.float64)
        if block.ndim != 2:
            raise DimensionError(f"panel must be (nobs, p); got shape {block.shape}.")
        grid = np.asarray(maturities, dtype=np.float64).ravel()
        if grid.shape[0] != block.shape[1]:
            raise DimensionError(
                f"maturities must have one entry per panel column "
                f"({block.shape[1]}); got {grid.shape[0]}."
            )
        if grid.shape[0] < 4:
            raise DimensionError(
                "the three factors need at least 4 maturities to be "
                f"identified with measurement noise; got {grid.shape[0]}."
            )
        if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
            raise SpecificationError("maturities must be finite and strictly positive.")
        if block.shape[0] < 30:
            raise DimensionError(
                f"the factor dynamics need at least 30 dates; got {block.shape[0]}."
            )
        finite = np.isfinite(block)
        if np.any(np.isinf(block)):
            raise NumericalError("panel entries must be finite or NaN.")
        if not np.all(finite.any(axis=1)):
            raise NumericalError(
                "every date must observe at least one maturity; drop all-missing rows."
            )
        if decay is not None and not decay > 0.0:
            raise SpecificationError(f"a fixed decay must be strictly positive; got {decay}.")
        self._panel = block
        self._maturities = grid
        self._fixed_decay = None if decay is None else float(decay)

    def fit(self) -> DynamicNelsonSiegelResult:
        """Maximize the exact likelihood from the two-step warm start."""
        objective = _NelsonSiegelObjective(
            panel=self._panel,
            maturities=self._maturities,
            fixed_decay=self._fixed_decay,
        )
        params, llf = _maximize_likelihood(objective)
        model = LinearGaussianSSM._from_nelson_siegel_system(params, self._maturities)
        smoothed = model.smooth(self._panel)
        fit = _NelsonSiegelFit(
            params=params,
            llf=llf,
            n_params=objective.starts()[0].shape[0],
            nobs=int(self._panel.shape[0]),
            factors=smoothed.smoothed_state,
            factor_cov=smoothed.smoothed_state_cov,
        )
        return DynamicNelsonSiegelResult(
            panel=self._panel,
            maturities=self._maturities,
            decay=fit.params.decay,
            decay_fixed=self._fixed_decay is not None,
            mu=fit.params.mu,
            ar=fit.params.ar,
            state_innovation_cov=fit.params.state_chol @ fit.params.state_chol.T,
            measurement_var=fit.params.obs_var,
            factors=fit.factors,
            factor_cov=fit.factor_cov,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=float(fit.n_params),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class TimeVaryingNelsonSiegelResult(_SummaryMixin, _ComparisonMixin):
    """A fitted Nelson-Siegel model with a time-varying loading decay.

    Attributes:
        panel: The observed ``(nobs, p)`` yield panel.
        maturities: The ``(p,)`` maturities.
        mu: ``(3,)`` factor means (level, slope, curvature).
        ar: ``(3,)`` diagonal factor persistences.
        state_innovation_cov: ``(3, 3)`` factor innovation covariance.
        measurement_var: ``(p,)`` per-maturity measurement variances.
        log_decay_mean: Unconditional mean of the log decay.
        decay_ar: Persistence of the log decay.
        decay_sd: Innovation standard deviation of the log decay.
        factors: ``(nobs, 3)`` smoothed factor paths.
        factor_cov: ``(nobs, 3, 3)`` smoothed factor covariances.
        log_decay: ``(nobs,)`` smoothed log-decay path.
        log_decay_std: ``(nobs,)`` smoothed log-decay standard deviations.
        llf: The *approximate* Gaussian log-likelihood of the chosen
            filter. Not the exact likelihood: information criteria are
            comparable against the constant-decay model only as a
            heuristic, and the summary says so.
        filter: ``"extended"`` or ``"unscented"``.
        nobs: Curve dates.
        n_params: Free parameters the likelihood was maximized over.
    """

    panel: npt.NDArray[np.float64] = field(repr=False)
    maturities: npt.NDArray[np.float64]
    mu: npt.NDArray[np.float64]
    ar: npt.NDArray[np.float64]
    state_innovation_cov: npt.NDArray[np.float64] = field(repr=False)
    measurement_var: npt.NDArray[np.float64] = field(repr=False)
    log_decay_mean: float
    decay_ar: float
    decay_sd: float
    factors: npt.NDArray[np.float64] = field(repr=False)
    factor_cov: npt.NDArray[np.float64] = field(repr=False)
    log_decay: npt.NDArray[np.float64] = field(repr=False)
    log_decay_std: npt.NDArray[np.float64] = field(repr=False)
    llf: float
    filter: str
    nobs: int
    n_params: float

    @classmethod
    def _from_fit(
        cls, fit: _DecayNelsonSiegelFit, model: TimeVaryingNelsonSiegel
    ) -> TimeVaryingNelsonSiegelResult:
        """Assemble the public result from a raw fit and its specification."""
        p = fit.params
        return cls(
            panel=model.panel,
            maturities=model.maturities,
            mu=p.mu,
            ar=p.ar,
            state_innovation_cov=p.state_chol @ p.state_chol.T,
            measurement_var=p.obs_var,
            log_decay_mean=p.log_decay_mean,
            decay_ar=p.decay_ar,
            decay_sd=p.decay_sd,
            factors=fit.factors,
            factor_cov=fit.factor_cov,
            log_decay=fit.log_decay,
            log_decay_std=fit.log_decay_std,
            llf=fit.llf,
            filter=fit.filter,
            nobs=fit.nobs,
            n_params=float(fit.n_params),
        )

    @property
    def n_maturities(self) -> int:
        """Points on the curve."""
        return int(self.maturities.shape[0])

    @property
    def decay(self) -> npt.NDArray[np.float64]:
        """The smoothed decay path ``exp(log lambda_t)``, ``(nobs,)``."""
        return np.asarray(np.exp(self.log_decay), dtype=np.float64)

    @property
    def decay_mean(self) -> float:
        """The unconditional decay ``exp(log_decay_mean)``."""
        return float(np.exp(self.log_decay_mean))

    @property
    def hump_maturity(self) -> npt.NDArray[np.float64]:
        """Where the curvature loading peaks each date, ``1.79 / lambda_t``."""
        return np.asarray(1.7916 / self.decay, dtype=np.float64)

    @property
    def level(self) -> npt.NDArray[np.float64]:
        """The smoothed level factor, ``(nobs,)``."""
        return np.asarray(self.factors[:, 0], dtype=np.float64)

    @property
    def slope(self) -> npt.NDArray[np.float64]:
        """The smoothed slope factor, ``(nobs,)``."""
        return np.asarray(self.factors[:, 1], dtype=np.float64)

    @property
    def curvature(self) -> npt.NDArray[np.float64]:
        """The smoothed curvature factor, ``(nobs,)``."""
        return np.asarray(self.factors[:, 2], dtype=np.float64)

    def loadings(self, t: int) -> npt.NDArray[np.float64]:
        """The ``(p, 3)`` Nelson-Siegel loadings at date ``t``'s smoothed decay."""
        return _nelson_siegel_loadings(self.maturities, float(self.decay[t]))

    @property
    def fitted(self) -> npt.NDArray[np.float64]:
        """Smoothed fitted curves, ``(nobs, p)``."""
        return np.asarray(
            np.stack([self.loadings(t) @ self.factors[t] for t in range(self.nobs)]),
            dtype=np.float64,
        )

    @property
    def _params(self) -> _DecayNelsonSiegelParameters:
        """The parameter record, rebuilt for the emitter."""
        return _DecayNelsonSiegelParameters(
            mu=self.mu,
            ar=self.ar,
            state_chol=np.linalg.cholesky(self.state_innovation_cov),
            obs_var=self.measurement_var,
            log_decay_mean=self.log_decay_mean,
            decay_ar=self.decay_ar,
            decay_sd=self.decay_sd,
        )

    @property
    def state_space(self) -> NonlinearSSM:
        """The fitted system on the nonlinear substrate.

        Additive-Gaussian, so the extended, unscented, and particle
        filters all read it; the unscented filter on the estimation panel
        reproduces ``llf`` when the fit used it.
        """
        return _decay_nelson_siegel_state_space(self._params, self.maturities)

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        return f"DNS-TV[{self.filter}]"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        ic = self.information_criteria
        rows: list[tuple[str, str, str, str]] = []
        for j, name in enumerate(("level", "slope", "curvature")):
            rows.append(
                (
                    name,
                    f"{self.mu[j]:.4f}",
                    f"{self.ar[j]:.4f}",
                    f"{np.sqrt(self.state_innovation_cov[j, j]):.4f}",
                )
            )
        rows.append(
            (
                "log decay",
                f"{self.log_decay_mean:.4f}",
                f"{self.decay_ar:.4f}",
                f"{self.decay_sd:.4f}",
            )
        )
        notes = (
            f"Likelihood is the {self.filter} filter's Gaussian approximation, not "
            "the exact likelihood; treat information criteria against the "
            "constant-decay model as heuristic.",
            f"Unconditional decay {self.decay_mean:.4f} (curvature hump at maturity "
            f"{1.7916 / self.decay_mean:.2f}); smoothed decay ranges "
            f"{self.decay.min():.4f} to {self.decay.max():.4f}.",
        )
        return SummaryTable(
            title="Dynamic Nelson-Siegel (time-varying decay) Results",
            metadata=(
                ("Filter", self.filter),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Dates", f"{self.nobs}"),
                ("AIC", f"{ic.aic:.3f}"),
                ("Maturities", f"{self.n_maturities}"),
                ("BIC", f"{ic.bic:.3f}"),
            ),
            columns=("factor", "mean", "persistence", "innovation sd"),
            rows=tuple(rows),
            notes=notes,
        )


class TimeVaryingNelsonSiegel(_DecayNelsonSiegelModel[TimeVaryingNelsonSiegelResult]):
    """The dynamic Nelson-Siegel model with a time-varying loading decay.

    The state is ``(level, slope, curvature, log lambda)``, each a diagonal
    AR(1); the measurement is the Nelson-Siegel curve at the current
    ``lambda_t``. Estimated by maximizing the unscented (default) or
    extended filter's likelihood from the constant-decay exact fit.

    Args:
        panel: The ``(nobs, p)`` yield panel. Rows must be wholly observed
            or wholly missing; a partially observed row is refused with
            a pointer to the constant-decay model.
        maturities: The ``(p,)`` strictly positive maturities.

    Example:
        >>> import numpy as np
        >>> from cultivars._core import _nelson_siegel_loadings
        >>> rng = np.random.default_rng(0)
        >>> taus = np.array([0.25, 1.0, 2.0, 5.0, 10.0])
        >>> f = np.zeros((120, 3))
        >>> mu = np.array([5.0, -1.5, 0.5])
        >>> f[0] = mu
        >>> for t in range(1, 120):
        ...     f[t] = mu + 0.9 * (f[t - 1] - mu) + 0.2 * rng.standard_normal(3)
        >>> curves = f @ _nelson_siegel_loadings(taus, 0.6).T
        >>> curves = curves + 0.05 * rng.standard_normal((120, 5))
        >>> res = TimeVaryingNelsonSiegel(curves, taus).fit()
        >>> bool(np.corrcoef(res.level, f[:, 0])[0, 1] > 0.95)
        True
    """

    def fit(self, *, filter: str = "unscented") -> TimeVaryingNelsonSiegelResult:
        """Maximize the approximate likelihood.

        Args:
            filter: ``"unscented"`` (default; exact through the linear
                transition, third-order accurate through the loadings) or
                ``"extended"`` (central-difference Jacobians).

        Raises:
            SpecificationError: If the filter is unknown.
        """
        return TimeVaryingNelsonSiegelResult._from_fit(self._fit_decay(filter_name=filter), self)
