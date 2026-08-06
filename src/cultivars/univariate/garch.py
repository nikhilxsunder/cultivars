

"""Conditional-variance models: GARCH, GJR-GARCH, EGARCH.

These are NOT state-space models; the conditional variance follows a
deterministic recursion evaluated on the mean-model residuals, and the
(mean, variance) parameters are estimated jointly by exact Gaussian maximum
likelihood. A shared engine drives all three; the variance recursion differs.

Conventions (matching Sheppard's ``arch``): ``p`` = ARCH (squared-resid) lags,
``o`` = asymmetry lags, ``q`` = lagged-variance lags. Pre-sample variance is
initialized by a 0.94-decay EWMA backcast of the squared residuals.

References:
    Bollerslev (1986); Glosten, Jagannathan, Runkle (1993); Nelson (1991).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from ..exceptions import DimensionError, NumericalError, SpecificationError
from ..univariate._base import InformationCriteria, information_criteria


@dataclass(frozen=True, kw_only=True, slots=True)
class GARCHResult(_MeanResult, _StationarityMixin, _ConditionalVarianceMixin):
    vol: str
    order: tuple[int, int, int]
    const: float | None
    ar_params: npt.NDArray[np.float64]
    omega: float
    alpha: npt.NDArray[np.float64]
    gamma: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    fractional_d: float | None
    conditional_variance: npt.NDArray[np.float64]


def _fit_garch(
    endog: npt.NDArray[np.float64],
    p: int,
    o: int,
    q: int,
    ar_lags: int,
    include_const: bool,
    vol: Vol,
) -> GARCHResult:
    y = endog
    n_full = y.shape[0]
    # mean design (constant + AR lags), conditioning on the first ar_lags obs
    start = ar_lags
    target = y[start:]
    n = target.shape[0]
    mean_cols: list[npt.NDArray[np.float64]] = []
    if include_const:
        mean_cols.append(np.ones(n))
    for i in range(1, ar_lags + 1):
        mean_cols.append(y[start - i : n_full - i])
    mean_x = np.column_stack(mean_cols) if mean_cols else np.zeros((n, 0))
    k_mean = mean_x.shape[1]

    # starting values
    if k_mean:
        mean0 = np.linalg.lstsq(mean_x, target, rcond=None)[0]
        resid0 = target - mean_x @ mean0
    else:
        mean0 = np.zeros(0)
        resid0 = target.copy()
    var0 = max(float(np.var(resid0)), 1e-8)
    backcast = _backcast(resid0)

    a_init, b_init, g_init = 0.05, 0.90, 0.05
    if vol == "GARCH":
        var_raw0 = np.concatenate([
            [np.log(var0 * (1 - a_init - b_init))],
            [_inv_softplus(a_init)] * p,
            [_inv_softplus(b_init)] * q,
        ])
    elif vol == "GJR":
        var_raw0 = np.concatenate([
            [np.log(var0 * (1 - a_init - b_init - 0.5 * g_init))],
            [_inv_softplus(a_init)] * p,
            [g_init] * o,
            [_inv_softplus(b_init)] * q,
        ])
    else:  # EGARCH
        var_raw0 = np.concatenate([
            [np.log(var0) * (1 - 0.95)],
            [0.1] * p,
            [-0.05] * o,
            [0.95] * q,
        ])
    theta0 = np.concatenate([mean0, var_raw0])
    m_idx = k_mean

    def unpack_var(v: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if vol == "GARCH":
            omega = float(np.exp(v[0]))
            alpha = _softplus(v[1 : 1 + p])
            gamma = np.zeros(0)
            beta = _softplus(v[1 + p : 1 + p + q])
        elif vol == "GJR":
            omega = float(np.exp(v[0]))
            alpha = _softplus(v[1 : 1 + p])
            gamma = v[1 + p : 1 + p + o]
            beta = _softplus(v[1 + p + o : 1 + p + o + q])
        else:
            omega = float(v[0])
            alpha = v[1 : 1 + p]
            gamma = v[1 + p : 1 + p + o]
            beta = v[1 + p + o : 1 + p + o + q]
        return omega, alpha, gamma, beta

    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        mean = theta[:m_idx]
        resid = target - mean_x @ mean if k_mean else target
        omega, alpha, gamma, beta = unpack_var(theta[m_idx:])
        if vol == "EGARCH":
            if abs(beta.sum()) >= 0.999:
                return 1e10
            sigma2 = _egarch_variance(resid, omega, alpha, gamma, beta, backcast)
        else:
            if alpha.sum() + 0.5 * gamma.sum() + beta.sum() >= 0.999:
                return 1e10
            sigma2 = _garch_variance(resid, omega, alpha, gamma, beta, backcast)
        if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
            return 1e10
        ll = -0.5 * np.sum(_LOG_2PI + np.log(sigma2) + resid ** 2 / sigma2)
        return float(-ll) if np.isfinite(ll) else 1e10

    result = minimize(negloglik, theta0, method="L-BFGS-B")
    theta = np.asarray(result.x, dtype=np.float64)
    mean = theta[:m_idx]
    resid = target - mean_x @ mean if k_mean else target
    omega, alpha, gamma, beta = unpack_var(theta[m_idx:])
    sigma2 = (
        _egarch_variance(resid, omega, alpha, gamma, beta, backcast)
        if vol == "EGARCH"
        else _garch_variance(resid, omega, alpha, gamma, beta, backcast)
    )
    const = float(mean[0]) if include_const else None
    ar_params = mean[1:] if include_const else mean
    n_params = k_mean + 1 + p + o + q
    return GARCHResult(
        vol=vol,
        order=(p, o, q),
        const=const,
        ar_params=np.asarray(ar_params, dtype=np.float64),
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        sigma2=sigma2,
        resid=resid,
        llf=-float(result.fun),
        nobs=n,
        _n_params=n_params,
    )


def _fit_figarch(
    endog: npt.NDArray[np.float64],
    include_const: bool,
    truncation: int = 1000,
) -> GARCHResult:
    n = endog.shape[0]
    mean_x = np.ones((n, 1)) if include_const else np.zeros((n, 0))
    k_mean = mean_x.shape[1]
    if k_mean:
        mean0 = np.linalg.lstsq(mean_x, endog, rcond=None)[0]
        resid0 = endog - mean_x @ mean0
    else:
        mean0 = np.zeros(0)
        resid0 = endog.copy()
    var0 = max(float(np.var(resid0)), 1e-8)
    backcast = _backcast(resid0)
    # unconstrained params: mean, log_omega, u_phi, u_d, u_beta (phi,d,beta in (0,1) via sigmoid)
    theta0 = np.concatenate([mean0, [np.log(var0 * 0.4), -1.0, -0.2, 0.4]])


    def sig(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))


    def unpack(theta: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], float, float, float, float]:
        mean = theta[:k_mean]
        omega = float(np.exp(theta[k_mean]))
        phi = sig(float(theta[k_mean + 1]))
        d = sig(float(theta[k_mean + 2]))
        beta = sig(float(theta[k_mean + 3]))
        return mean, omega, phi, d, beta


    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        mean, omega, phi, d, beta = unpack(theta)
        resid = endog - mean_x @ mean if k_mean else endog
        lam = _figarch_weights(phi, d, beta, min(truncation, 200))
        if np.any(lam < -1e-6):
            return 1e10
        sigma2 = _figarch_variance(resid, omega, phi, d, beta, backcast, truncation)
        if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
            return 1e10
        ll = -0.5 * np.sum(_LOG_2PI + np.log(sigma2) + resid ** 2 / sigma2)
        return float(-ll) if np.isfinite(ll) else 1e10

    result = minimize(negloglik, theta0, method="L-BFGS-B")
    mean, omega, phi, d, beta = unpack(np.asarray(result.x, dtype=np.float64))
    resid = endog - mean_x @ mean if k_mean else endog
    sigma2 = _figarch_variance(resid, omega, phi, d, beta, backcast, truncation)
    const = float(mean[0]) if include_const else None
    n_params = k_mean + 3
    return GARCHResult(
        vol="FIGARCH",
        order=(1, 0, 1),
        const=const,
        ar_params=np.zeros(0),
        omega=omega,
        alpha=np.array([phi]),
        gamma=np.zeros(0),
        beta=np.array([beta]),
        sigma2=sigma2,
        resid=resid,
        llf=-float(result.fun),
        nobs=n,
        _n_params=n_params,
        fractional_d=d,
    )


class GARCH:
    """GARCH(p, q) with a constant (and optional AR) mean.

    Args:
        endog: Univariate series (typically returns/residuals).
        p: ARCH order (lagged squared residuals).
        q: GARCH order (lagged variances).
        mean: ``"constant"`` (default) or ``"zero"``.
        ar_lags: Optional AR order for the conditional mean (AR-GARCH).
    """

    __slots__ = ("_y", "_p", "_q", "_ar", "_const")

    def __init__(self, endog: npt.ArrayLike, p: int = 1, q: int = 1, mean: Literal["constant", "zero"] = "constant", ar_lags: int = 0) -> None:
        self._y = _validate_garch(endog, p, 0, q, ar_lags)
        self._p, self._q, self._ar, self._const = int(p), int(q), int(ar_lags), mean == "constant"

    def fit(self) -> GARCHResult:
        return _fit_garch(self._y, self._p, 0, self._q, self._ar, self._const, "GARCH")


class GJR:
    """GJR-GARCH(p, o, q) — asymmetric (leverage) GARCH."""

    __slots__ = ("_y", "_p", "_o", "_q", "_ar", "_const")

    def __init__(self, endog: npt.ArrayLike, p: int = 1, o: int = 1, q: int = 1, mean: Literal["constant", "zero"] = "constant", ar_lags: int = 0) -> None:
        self._y = _validate_garch(endog, p, o, q, ar_lags)
        self._p, self._o, self._q, self._ar, self._const = int(p), int(o), int(q), int(ar_lags), mean == "constant"

    def fit(self) -> GARCHResult:
        return _fit_garch(self._y, self._p, self._o, self._q, self._ar, self._const, "GJR")


class EGARCH:
    """EGARCH(p, o, q) — exponential GARCH (log-variance, asymmetric).

    NOTE: pre-sample convention not yet reconciled with a reference
    implementation; flagged for the validation pass.
    """

    __slots__ = ("_y", "_p", "_o", "_q", "_ar", "_const")

    def __init__(self, endog: npt.ArrayLike, p: int = 1, o: int = 1, q: int = 1, mean: Literal["constant", "zero"] = "constant", ar_lags: int = 0) -> None:
        self._y = _validate_garch(endog, p, o, q, ar_lags)
        self._p, self._o, self._q, self._ar, self._const = int(p), int(o), int(q), int(ar_lags), mean == "constant"

    def fit(self) -> GARCHResult:
        return _fit_garch(self._y, self._p, self._o, self._q, self._ar, self._const, "EGARCH")


class FIGARCH:
    """FIGARCH(1, d, 1) — fractionally integrated GARCH (long-memory volatility).

    Args:
        endog: Univariate series (returns / residuals).
        mean: ``"constant"`` (default) or ``"zero"``.
        truncation: ARCH(inf) truncation lag for the lambda weights.
    """

    __slots__ = ("_y", "_const", "_truncation")

    def __init__(self, endog: npt.ArrayLike, mean: Literal["constant", "zero"] = "constant", truncation: int = 1000) -> None:
        self._y = _validate_garch(endog, 1, 0, 1, 0)
        if truncation < 1:
            raise SpecificationError(f"truncation must be >= 1; got {truncation}.")
        self._const = mean == "constant"
        self._truncation = int(truncation)

    def fit(self) -> GARCHResult:
        return _fit_figarch(self._y, self._const, self._truncation)
