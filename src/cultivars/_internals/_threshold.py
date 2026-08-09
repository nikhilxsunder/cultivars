# filepath: /src/cultivars/_internals/_threshold.py
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

"""Threshold and smooth-transition autoregression engines.

Two estimators over the same two-regime idea. SETAR/TAR split the sample
hard at a threshold and search a trimmed quantile grid, because the SSR is a
step function of the threshold and no gradient method applies. STAR replaces
the indicator with a smooth weight and estimates the transition parameters by
concentrated nonlinear least squares: for any ``(gamma, c)`` the regime
coefficients are a linear solve, so only two parameters are searched.

The STAR search is seeded from a hard-threshold grid and restarted over four
smoothness values, because the SSR surface in ``gamma`` is close to flat once
the transition is sharp and a single start lands in whichever basin it began.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from .._core._defaults import _DEFAULT_GRID, _DEFAULT_TRIM, _LOG_2PI
from .._core._design import deterministic_columns, lag_matrix
from .._core._linalg import ols
from .._core._validators import (
    validate_aligned,
    validate_choice,
    validate_open_interval,
    validate_order,
)
from ..exceptions import NumericalError
from ._models import _UnivariateModel

_TRANSITIONS = ("logistic", "exponential")


@dataclass(frozen=True, slots=True)
class _ThresholdFit:
    """Raw outputs of a two-regime threshold grid search."""

    delay: int
    threshold: float
    lower_params: npt.NDArray[np.float64]
    upper_params: npt.NDArray[np.float64]
    sigma2: float
    ssr: float
    n_lower: int
    n_upper: int
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


@dataclass(frozen=True, slots=True)
class _STARFit:
    """Raw outputs of a smooth-transition autoregression fit."""

    delay: int
    threshold: float
    gamma: float
    lower_params: npt.NDArray[np.float64]
    upper_params: npt.NDArray[np.float64]
    sigma2: float
    ssr: float
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


def _ar_design(y: npt.NDArray[np.float64], order: int, start: int) -> npt.NDArray[np.float64]:
    """Intercept plus lagged levels, aligned to begin at ``start``.

    Args:
        y: The series.
        order: AR order per regime.
        start: First retained time index.

    Returns:
        An ``(n - start, order + 1)`` design matrix.
    """
    return np.column_stack(
        [
            deterministic_columns("c", y.shape[0] - start),
            lag_matrix(y, order, start=start),
        ]
    )


def _fit_threshold(
    y: npt.NDArray[np.float64],
    order: int,
    delays: list[int],
    trim: float,
    n_grid: int,
    threshold_var: npt.NDArray[np.float64] | None,
) -> _ThresholdFit:
    """Fit a two-regime threshold autoregression by grid search.

    Searches every ``(delay, threshold)`` pair on a trimmed quantile grid and
    keeps the pair minimizing total SSR. Splits that leave either regime with
    fewer than ``order + 2`` observations are skipped, since those coefficients
    would not be identified.

    Args:
        y: The series.
        order: AR order per regime.
        delays: Candidate threshold delays.
        trim: Fraction trimmed from each tail of the grid.
        n_grid: Number of candidate thresholds per delay.
        threshold_var: External threshold variable, or ``None`` for self-exciting.

    Returns:
        The packed :class:`_ThresholdFit`.

    Raises:
        NumericalError: If no admissible split exists.
    """
    n = y.shape[0]
    start = max(order, max(delays))
    target = y[start:]
    n_eff = target.shape[0]
    design = _ar_design(y, order, start)
    base = threshold_var if threshold_var is not None else y
    min_regime = order + 2

    best_ssr = np.inf
    best: tuple[int, float, npt.NDArray[np.float64], npt.NDArray[np.float64], int, int] | None = (
        None
    )
    for d in delays:
        z = base[start - d : n - d]
        grid = np.quantile(z, np.linspace(trim, 1.0 - trim, n_grid))
        for r in np.unique(grid):
            lower = z <= r
            n_lo = int(lower.sum())
            n_hi = n_eff - n_lo
            if n_lo < min_regime or n_hi < min_regime:
                continue
            b_lo, ssr_lo = ols(design[lower], target[lower])
            b_hi, ssr_hi = ols(design[~lower], target[~lower])
            ssr = ssr_lo + ssr_hi
            if ssr < best_ssr:
                best_ssr = ssr
                best = (d, float(r), b_lo, b_hi, n_lo, n_hi)

    if best is None:
        raise NumericalError(
            "threshold grid search found no admissible split; relax trim or shorten order."
        )
    d_star, r_star, b_lo, b_hi, n_lo, n_hi = best
    z = base[start - d_star : n - d_star]
    lower = z <= r_star
    fitted = np.where(lower, design @ b_lo, design @ b_hi)
    resid = target - fitted
    sigma2 = best_ssr / n_eff
    llf = -0.5 * n_eff * (_LOG_2PI + np.log(sigma2) + 1.0)
    return _ThresholdFit(
        delay=d_star,
        threshold=r_star,
        lower_params=b_lo,
        upper_params=b_hi,
        sigma2=sigma2,
        ssr=float(best_ssr),
        n_lower=n_lo,
        n_upper=n_hi,
        resid=resid,
        fittedvalues=fitted,
        llf=float(llf),
        nobs=n_eff,
        n_params=2 * (order + 1) + 1,
    )


def _fit_star(y: npt.NDArray[np.float64], order: int, delay: int, transition: str) -> _STARFit:
    """Fit a smooth-transition autoregression by concentrated least squares.

    The transition variable is standardized before entering the transition
    function, so ``gamma`` is scale-free and comparable across series.

    Args:
        y: The series.
        order: AR order per regime.
        delay: Transition-variable delay.
        transition: ``"logistic"`` (LSTAR) or ``"exponential"`` (ESTAR).

    Returns:
        The packed :class:`_STARFit`.

    Raises:
        NumericalError: If the transition variable has zero variance.
    """
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
        return 1.0 - np.exp(-np.clip(gamma * u**2, 0.0, 50.0))

    def concentrated(
        gamma: float, c: float
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        g = transition_weights(gamma, c)
        regressors = np.column_stack([design * (1.0 - g)[:, None], design * g[:, None]])
        beta, ssr = ols(regressors, target)
        return ssr, beta, target - regressors @ beta

    c_seed = float(np.median(z))
    best_hard = np.inf
    for cc in np.quantile(z, np.linspace(0.15, 0.85, 50)):
        lower = z <= cc
        n_lo = int(lower.sum())
        if n_lo < order + 2 or n_eff - n_lo < order + 2:
            continue
        ssr_hard = ols(design[lower], target[lower])[1] + ols(design[~lower], target[~lower])[1]
        if ssr_hard < best_hard:
            best_hard, c_seed = ssr_hard, float(cc)

    def objective(par: npt.NDArray[np.float64]) -> float:
        return concentrated(float(np.exp(par[0])), float(par[1]))[0]

    best_ssr, best_par = np.inf, np.array([np.log(5.0), c_seed])
    for c0 in (c_seed, float(np.median(z))):
        for g0 in (2.0, 5.0, 10.0, 25.0):
            res = minimize(
                objective,
                np.array([np.log(g0), c0]),
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000},
            )
            if float(res.fun) < best_ssr:
                best_ssr, best_par = float(res.fun), np.asarray(res.x, dtype=np.float64)

    gamma = float(np.exp(best_par[0]))
    c = float(best_par[1])
    ssr, beta, resid = concentrated(gamma, c)
    sigma2 = ssr / n_eff
    llf = -0.5 * n_eff * (_LOG_2PI + np.log(sigma2) + 1.0)
    return _STARFit(
        delay=delay,
        threshold=c,
        gamma=gamma,
        lower_params=beta[: order + 1],
        upper_params=beta[order + 1 :],
        sigma2=sigma2,
        ssr=ssr,
        resid=resid,
        fittedvalues=target - resid,
        llf=float(llf),
        nobs=n_eff,
        n_params=2 * (order + 1) + 2,
    )


class _ThresholdModel[R](_UnivariateModel[R]):
    """Shared specification surface for SETAR/TAR grid-search models.

    Args:
        endog: The series.
        order: AR order per regime.
        delay: Threshold delay; ``None`` searches ``1..order``.
        trim: Fraction trimmed from each tail of the grid.
        n_grid: Number of candidate thresholds per delay.
        threshold_variable: External threshold variable; ``None`` is self-exciting.
    """

    __slots__ = ("_delays", "_n_grid", "_order", "_threshold_variable", "_trim")

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        delay: int | None = None,
        trim: float = _DEFAULT_TRIM,
        n_grid: int = _DEFAULT_GRID,
        threshold_variable: npt.ArrayLike | None = None,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog)
        self._order = validate_order(order, "order", minimum=1)
        self._trim = validate_open_interval(trim, "trim", low=0.0, high=0.5)
        self._n_grid = validate_order(n_grid, "n_grid", minimum=1)
        self._delays = (
            [validate_order(delay, "delay", minimum=1)]
            if delay is not None
            else list(range(1, self._order + 1))
        )
        self._threshold_variable = (
            None
            if threshold_variable is None
            else validate_aligned(threshold_variable, self.endog.shape[0], "threshold_variable")
        )
        self._ensure_length(
            2 * (self._order + 2) + max(self._delays), f"threshold AR({self._order})"
        )

    @property
    def order(self) -> int:
        """AR order per regime."""
        return self._order

    @property
    def delay(self) -> int | None:
        """The fixed delay, or ``None`` when the delay is searched."""
        return self._delays[0] if len(self._delays) == 1 else None

    @property
    def self_exciting(self) -> bool:
        """Whether the threshold variable is a lag of the series itself."""
        return self._threshold_variable is None

    def _fit_family(self) -> _ThresholdFit:
        """Run the shared grid-search engine for this specification."""
        return _fit_threshold(
            self.endog,
            self._order,
            self._delays,
            self._trim,
            self._n_grid,
            self._threshold_variable,
        )


class _STARModel[R](_UnivariateModel[R]):
    """Shared specification surface for LSTAR/ESTAR smooth-transition models.

    Args:
        endog: The series.
        order: AR order per regime.
        transition: ``"logistic"`` or ``"exponential"``.
        delay: Transition-variable delay.
    """

    __slots__ = ("_delay", "_order", "_transition")

    def __init__(
        self, endog: npt.ArrayLike, *, order: int, transition: str, delay: int = 1
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog)
        self._order = validate_order(order, "order", minimum=1)
        self._delay = validate_order(delay, "delay", minimum=1)
        self._transition = validate_choice(transition, _TRANSITIONS, "transition")
        self._ensure_length(2 * (self._order + 2) + self._delay, f"STAR({self._order})")

    @property
    def order(self) -> int:
        """AR order per regime."""
        return self._order

    @property
    def delay(self) -> int:
        """The transition-variable delay."""
        return self._delay

    @property
    def transition(self) -> str:
        """The transition function family."""
        return self._transition

    def _fit_family(self) -> _STARFit:
        """Run the smooth-transition engine for this specification."""
        return _fit_star(self.endog, self._order, self._delay, self._transition)
