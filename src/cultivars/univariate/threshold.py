# filepath: /src/cultivars/univariate/threshold.py
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

"""Threshold and smooth-transition autoregressions.

Observable-regime nonlinear AR models. The regime is a function of an observed
threshold variable (a lag of the series for SETAR/LSTAR/ESTAR, or an external
variable for TAR), so these are estimated by conditional least squares rather
than through the state-space substrate:

- SETAR / TAR: a hard threshold. The model is conditionally linear given the
  threshold value, so estimation is a grid search over the threshold (and delay)
  with regime-wise OLS at each candidate.
- STAR (LSTAR / ESTAR): a smooth transition governed by a logistic or
  exponential function G in [0, 1]. Given the transition parameters the model is
  linear, so the regime coefficients are concentrated out and the SSR is
  minimized over (gamma, c) by nonlinear optimization.

References:
    Tong (1990); Tsay (1989); Terasvirta (1994); van Dijk, Terasvirta, Franses (2002).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from cultivars.exceptions import DimensionError, NumericalError, SpecificationError
from cultivars.univariate._base import InformationCriteria, information_criteria

_SCHEMA_VERSION = 1
_LOG_2PI = float(np.log(2.0 * np.pi))
Transition = Literal["logistic", "exponential"]


def _ols(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], float]:
    beta, _res, _rank, _sv = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return np.asarray(beta, dtype=np.float64), float(resid @ resid)


def _ar_design(y: npt.NDArray[np.float64], order: int, start: int) -> npt.NDArray[np.float64]:
    n = y.shape[0]
    cols = [np.ones(n - start)] + [y[start - i : n - i] for i in range(1, order + 1)]
    return np.column_stack(cols)


# --------------------------------------------------------------------------
# SETAR / TAR
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SETARResult:
    """Fitted 2-regime threshold AR.

    Attributes:
        order: AR order (per regime).
        delay: Threshold delay ``d`` (regime set by threshold variable at ``t-d``).
        threshold: Estimated threshold ``r``.
        lower_params: Regime coefficients ``[const, phi_1, ...]`` for ``z <= r``.
        upper_params: Regime coefficients for ``z > r``.
        sigma2: Residual variance.
        ssr: Total sum of squared residuals.
        n_lower: Observations in the lower regime.
        n_upper: Observations in the upper regime.
        llf: Gaussian log-likelihood.
        nobs: Observations used.
        resid: Residuals.
        fittedvalues: Fitted values.
        schema_version: Serialization schema version.
    """

    order: int
    delay: int
    threshold: float
    lower_params: npt.NDArray[np.float64]
    upper_params: npt.NDArray[np.float64]
    sigma2: float
    ssr: float
    n_lower: int
    n_upper: int
    llf: float
    nobs: int
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    _n_params: int = field(repr=False)
    schema_version: int = _SCHEMA_VERSION

    @property
    def information_criteria(self) -> InformationCriteria:
        return information_criteria(self.llf, self.nobs, self._n_params)

    @property
    def aic(self) -> float:
        return self.information_criteria.aic

    @property
    def bic(self) -> float:
        return self.information_criteria.bic


def _fit_setar(
    y: npt.NDArray[np.float64],
    order: int,
    delays: list[int],
    trim: float,
    n_grid: int,
    threshold_var: npt.NDArray[np.float64] | None,
) -> SETARResult:
    n = y.shape[0]
    max_delay = max(delays)
    start = max(order, max_delay)
    target = y[start:]
    n_eff = target.shape[0]
    design = _ar_design(y, order, start)
    base = threshold_var if threshold_var is not None else y
    min_regime = order + 2

    best_ssr = np.inf
    best: tuple[int, float, npt.NDArray[np.float64], npt.NDArray[np.float64], int, int] | None = None
    for d in delays:
        z = base[start - d : n - d]
        grid = np.quantile(z, np.linspace(trim, 1.0 - trim, n_grid))
        for r in np.unique(grid):
            lower = z <= r
            n_lo = int(lower.sum())
            n_hi = n_eff - n_lo
            if n_lo < min_regime or n_hi < min_regime:
                continue
            b_lo, ssr_lo = _ols(design[lower], target[lower])
            b_hi, ssr_hi = _ols(design[~lower], target[~lower])
            ssr = ssr_lo + ssr_hi
            if ssr < best_ssr:
                best_ssr = ssr
                best = (d, float(r), b_lo, b_hi, n_lo, n_hi)

    if best is None:
        raise NumericalError("SETAR grid search found no admissible threshold; relax trim or shorten order.")

    d_star, r_star, b_lo, b_hi, n_lo, n_hi = best
    z = base[start - d_star : n - d_star]
    lower = z <= r_star
    fitted = np.where(lower, design @ b_lo, design @ b_hi)
    resid = target - fitted
    sigma2 = best_ssr / n_eff
    llf = -0.5 * n_eff * (_LOG_2PI + np.log(sigma2) + 1.0)
    n_params = 2 * (order + 1) + 1
    return SETARResult(
        order=order, delay=d_star, threshold=r_star,
        lower_params=b_lo, upper_params=b_hi,
        sigma2=sigma2, ssr=best_ssr, n_lower=n_lo, n_upper=n_hi,
        llf=llf, nobs=n_eff, resid=resid, fittedvalues=fitted, _n_params=n_params,
    )


def _validate_endog(endog: npt.ArrayLike) -> npt.NDArray[np.float64]:
    y = np.asarray(endog, dtype=np.float64)
    if y.ndim != 1:
        raise DimensionError(f"endog must be one-dimensional; got shape {y.shape}.")
    if not np.all(np.isfinite(y)):
        raise NumericalError("endog contains non-finite values.")
    return y


class SETAR:
    """Self-exciting threshold AR (2 regimes); threshold variable is a lag of the series.

    Args:
        endog: Univariate series.
        order: AR order per regime.
        delay: Threshold delay ``d``; if ``None``, searched over ``1..order``.
        trim: Fraction trimmed from each tail of the threshold grid.
        n_grid: Number of candidate thresholds per delay.
    """

    __slots__ = ("_y", "_order", "_delays", "_trim", "_n_grid")

    def __init__(self, endog: npt.ArrayLike, order: int, delay: int | None = None, trim: float = 0.15, n_grid: int = 300) -> None:
        self._y = _validate_endog(endog)
        if order < 1:
            raise SpecificationError(f"order must be >= 1; got {order}.")
        if not 0.0 < trim < 0.5:
            raise SpecificationError(f"trim must be in (0, 0.5); got {trim}.")
        self._order = int(order)
        self._delays = [int(delay)] if delay is not None else list(range(1, order + 1))
        self._trim = trim
        self._n_grid = int(n_grid)

    def fit(self) -> SETARResult:
        return _fit_setar(self._y, self._order, self._delays, self._trim, self._n_grid, None)


class TAR:
    """Threshold AR (2 regimes) with an external threshold variable.

    Args:
        endog: Univariate series.
        order: AR order per regime.
        threshold_variable: The observed threshold variable, aligned with ``endog``.
        delay: Threshold delay ``d``.
        trim: Fraction trimmed from each tail of the threshold grid.
        n_grid: Number of candidate thresholds.
    """

    __slots__ = ("_y", "_order", "_z", "_delays", "_trim", "_n_grid")

    def __init__(self, endog: npt.ArrayLike, order: int, threshold_variable: npt.ArrayLike, delay: int = 1, trim: float = 0.15, n_grid: int = 300) -> None:
        self._y = _validate_endog(endog)
        z = np.asarray(threshold_variable, dtype=np.float64)
        if z.shape != self._y.shape:
            raise DimensionError(f"threshold_variable must match endog shape {self._y.shape}; got {z.shape}.")
        if not np.all(np.isfinite(z)):
            raise NumericalError("threshold_variable contains non-finite values.")
        if order < 1:
            raise SpecificationError(f"order must be >= 1; got {order}.")
        self._order, self._z, self._delays = int(order), z, [int(delay)]
        self._trim, self._n_grid = trim, int(n_grid)

    def fit(self) -> SETARResult:
        return _fit_setar(self._y, self._order, self._delays, self._trim, self._n_grid, self._z)


# --------------------------------------------------------------------------
# STAR (LSTAR / ESTAR)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class STARResult:
    """Fitted smooth-transition AR (2 regimes).

    Attributes:
        order: AR order.
        delay: Transition-variable delay.
        transition: ``"logistic"`` (LSTAR) or ``"exponential"`` (ESTAR).
        lower_params: Coefficients weighted by ``1 - G`` (``[const, phi_1, ...]``).
        upper_params: Coefficients weighted by ``G``.
        gamma: Transition smoothness (standardized by the transition-variable SD).
        threshold: Transition center ``c``.
        sigma2: Residual variance.
        ssr: Sum of squared residuals.
        llf: Gaussian log-likelihood.
        nobs: Observations used.
        resid: Residuals.
        fittedvalues: Fitted values.
        schema_version: Serialization schema version.
    """

    order: int
    delay: int
    transition: Transition
    lower_params: npt.NDArray[np.float64]
    upper_params: npt.NDArray[np.float64]
    gamma: float
    threshold: float
    sigma2: float
    ssr: float
    llf: float
    nobs: int
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    _n_params: int = field(repr=False)
    schema_version: int = _SCHEMA_VERSION

    @property
    def information_criteria(self) -> InformationCriteria:
        return information_criteria(self.llf, self.nobs, self._n_params)

    @property
    def aic(self) -> float:
        return self.information_criteria.aic

    @property
    def bic(self) -> float:
        return self.information_criteria.bic


def _fit_star(y: npt.NDArray[np.float64], order: int, delay: int, transition: Transition) -> STARResult:
    n = y.shape[0]
    start = max(order, delay)
    target = y[start:]
    n_eff = target.shape[0]
    design = _ar_design(y, order, start)
    z = y[start - delay : n - delay]
    sd = float(np.std(z))
    if sd == 0.0:
        raise NumericalError("transition variable has zero variance.")

    def transition_weights(gamma: float, c: float) -> npt.NDArray[np.float64]:
        u = (z - c) / sd
        if transition == "logistic":
            return 1.0 / (1.0 + np.exp(-np.clip(gamma * u, -50.0, 50.0)))
        return 1.0 - np.exp(-np.clip(gamma * u ** 2, 0.0, 50.0))

    def concentrated(gamma: float, c: float) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        g = transition_weights(gamma, c)
        regressors = np.column_stack([design * (1.0 - g)[:, None], design * g[:, None]])
        beta, ssr = _ols(regressors, target)
        return ssr, beta, target - regressors @ beta

    # seed the transition center from a hard-threshold (SETAR-style) grid
    c_seed = float(np.median(z))
    best_hard = np.inf
    for cc in np.quantile(z, np.linspace(0.15, 0.85, 50)):
        lower = z <= cc
        n_lo = int(lower.sum())
        if n_lo < order + 2 or n_eff - n_lo < order + 2:
            continue
        ssr_hard = _ols(design[lower], target[lower])[1] + _ols(design[~lower], target[~lower])[1]
        if ssr_hard < best_hard:
            best_hard, c_seed = ssr_hard, float(cc)

    def objective(par: npt.NDArray[np.float64]) -> float:
        return concentrated(float(np.exp(par[0])), float(par[1]))[0]

    best_ssr, best_par = np.inf, np.array([np.log(5.0), c_seed])
    for c0 in (c_seed, float(np.median(z))):
        for g0 in (2.0, 5.0, 10.0, 25.0):
            res = minimize(
                objective, np.array([np.log(g0), c0]), method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000},
            )
            if float(res.fun) < best_ssr:
                best_ssr, best_par = float(res.fun), np.asarray(res.x, dtype=np.float64)

    gamma = float(np.exp(best_par[0]))
    c = float(best_par[1])
    ssr, beta, resid = concentrated(gamma, c)
    lower = beta[: order + 1]
    upper = beta[order + 1 :]
    fitted = target - resid
    sigma2 = ssr / n_eff
    llf = -0.5 * n_eff * (_LOG_2PI + np.log(sigma2) + 1.0)
    n_params = 2 * (order + 1) + 2
    return STARResult(
        order=order, delay=delay, transition=transition,
        lower_params=lower, upper_params=upper, gamma=gamma, threshold=c,
        sigma2=sigma2, ssr=ssr, llf=llf, nobs=n_eff,
        resid=resid, fittedvalues=fitted, _n_params=n_params,
    )


class LSTAR:
    """Logistic smooth-transition AR (asymmetric regimes)."""

    __slots__ = ("_y", "_order", "_delay")

    def __init__(self, endog: npt.ArrayLike, order: int, delay: int = 1) -> None:
        self._y = _validate_endog(endog)
        if order < 1:
            raise SpecificationError(f"order must be >= 1; got {order}.")
        if delay < 1:
            raise SpecificationError(f"delay must be >= 1; got {delay}.")
        self._order, self._delay = int(order), int(delay)

    def fit(self) -> STARResult:
        return _fit_star(self._y, self._order, self._delay, "logistic")


class ESTAR:
    """Exponential smooth-transition AR (symmetric regimes)."""

    __slots__ = ("_y", "_order", "_delay")

    def __init__(self, endog: npt.ArrayLike, order: int, delay: int = 1) -> None:
        self._y = _validate_endog(endog)
        if order < 1:
            raise SpecificationError(f"order must be >= 1; got {order}.")
        if delay < 1:
            raise SpecificationError(f"delay must be >= 1; got {delay}.")
        self._order, self._delay = int(order), int(delay)

    def fit(self) -> STARResult:
        return _fit_star(self._y, self._order, self._delay, "exponential")
