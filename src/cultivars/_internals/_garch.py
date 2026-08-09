# filepath: /src/cultivars/_internals/_garch.py
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

"""Conditional-variance recursions, estimation engines, and the shared base.

Four volatility families share one specification surface and one likelihood
loop; they differ only in the variance recursion and the parameter transform
that keeps that recursion admissible.

- GARCH and GJR need a strictly positive variance, so ``omega`` is estimated
  as ``log omega`` and the ARCH/GARCH weights through a softplus. The
  asymmetry term of GJR is left unconstrained -- it may legitimately be
  negative -- and positivity is instead enforced by the persistence check.
- EGARCH models ``log sigma**2`` directly, so no parameter needs constraining
  and every coefficient is estimated raw.
- FIGARCH maps ``phi``, ``d`` and ``beta`` through a sigmoid into ``(0, 1)``
  and rejects any draw whose ARCH(infinity) weights turn negative.

Pre-sample variance is initialized by an exponentially weighted backcast over
the first 75 residuals rather than the unconditional variance, which is not
defined for the integrated members of the family.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from .._core._defaults import _DEFAULT_TRUNCATION, _LOG_2PI, _SQRT_2_OVER_PI
from .._core._validators import validate_choice, validate_order
from ..exceptions import SpecificationError
from ._models import _UnivariateModel

_VOL_FAMILIES = ("GARCH", "GJR", "EGARCH", "FIGARCH")
_MEANS = ("constant", "zero")


@dataclass(frozen=True, slots=True)
class _GARCHFit:
    """Raw outputs of a conditional-variance fit.

    Attributes:
        const: Mean intercept, or ``None`` when ``mean == "zero"``.
        ar_params: Conditional-mean AR coefficients (empty when ``ar_lags == 0``).
        omega: Variance intercept.
        alpha: ARCH coefficients (the ``phi`` weight for FIGARCH).
        gamma: Asymmetry coefficients.
        beta: GARCH coefficients.
        fractional_d: Fractional integration order (FIGARCH only).
        conditional_variance: The fitted variance path.
        resid: Mean residuals.
        fittedvalues: Fitted conditional means.
        llf: Maximized Gaussian log-likelihood.
        nobs: Effective observations.
        n_params: Free parameter count.
    """

    const: float | None
    ar_params: npt.NDArray[np.float64]
    omega: float
    alpha: npt.NDArray[np.float64]
    gamma: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    fractional_d: float | None
    conditional_variance: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


def _backcast(resid: npt.NDArray[np.float64]) -> float:
    """Exponentially weighted pre-sample variance estimate.

    Args:
        resid: Mean residuals.

    Returns:
        The weighted mean squared residual over the first 75 observations.
    """
    tau = min(75, resid.shape[0])
    w = 0.94 ** np.arange(tau)
    w /= w.sum()
    return float(np.sum(w * resid[:tau] ** 2))


def _softplus(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Map the real line to the positive half-line, overflow-safe."""
    return np.logaddexp(0.0, x)


def _inv_softplus(x: float) -> float:
    """Inverse of :func:`_softplus`, for constructing starting values."""
    return float(np.log(np.expm1(x)))


def _garch_variance(
    resid: npt.NDArray[np.float64],
    omega: float,
    alpha: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    backcast: float,
) -> npt.NDArray[np.float64]:
    """Variance recursion for the GARCH and GJR families.

    Pre-sample squared residuals and variances are replaced by ``backcast``;
    the asymmetry term uses half the backcast pre-sample, matching the
    unconditional probability of a negative shock.

    Args:
        resid: Mean residuals.
        omega: Variance intercept.
        alpha: ARCH coefficients.
        gamma: Asymmetry coefficients (empty for symmetric GARCH).
        beta: GARCH coefficients.
        backcast: Pre-sample variance.

    Returns:
        The conditional-variance path.
    """
    n, p, o, q = resid.shape[0], alpha.size, gamma.size, beta.size
    sigma2 = np.empty(n)
    r2 = resid**2
    neg = (resid < 0.0).astype(np.float64)
    for t in range(n):
        s = omega
        for i in range(p):
            s += alpha[i] * (r2[t - 1 - i] if t - 1 - i >= 0 else backcast)
        for k in range(o):
            if t - 1 - k >= 0:
                s += gamma[k] * r2[t - 1 - k] * neg[t - 1 - k]
            else:
                s += gamma[k] * backcast * 0.5
        for j in range(q):
            s += beta[j] * (sigma2[t - 1 - j] if t - 1 - j >= 0 else backcast)
        sigma2[t] = s
    return sigma2


def _egarch_variance(
    resid: npt.NDArray[np.float64],
    omega: float,
    alpha: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    backcast: float,
) -> npt.NDArray[np.float64]:
    """Log-variance recursion for EGARCH.

    The magnitude term is centered by ``E|z| = sqrt(2 / pi)`` so that a
    standard-normal shock contributes zero drift to the log variance.

    Args:
        resid: Mean residuals.
        omega: Log-variance intercept.
        alpha: Magnitude coefficients.
        gamma: Sign (leverage) coefficients.
        beta: Log-variance persistence coefficients.
        backcast: Pre-sample variance (used in logs).

    Returns:
        The conditional-variance path, exponentiated back to levels.
    """
    n, p, o, q = resid.shape[0], alpha.size, gamma.size, beta.size
    ln_sigma2 = np.empty(n)
    ln_bc = float(np.log(backcast))
    e = np.zeros(n)
    for t in range(n):
        s = omega
        for i in range(p):
            s += alpha[i] * ((abs(e[t - 1 - i]) - _SQRT_2_OVER_PI) if t - 1 - i >= 0 else 0.0)
        for k in range(o):
            s += gamma[k] * (e[t - 1 - k] if t - 1 - k >= 0 else 0.0)
        for j in range(q):
            s += beta[j] * (ln_sigma2[t - 1 - j] if t - 1 - j >= 0 else ln_bc)
        ln_sigma2[t] = s
        sig = np.sqrt(np.exp(ln_sigma2[t]))
        e[t] = resid[t] / sig if sig > 0 else 0.0
    return np.exp(ln_sigma2)


def _figarch_weights(phi: float, d: float, beta: float, truncation: int) -> npt.NDArray[np.float64]:
    """ARCH(infinity) lambda weights for FIGARCH(1, d, 1).

    Uses the Chung (1999) recursion, which is numerically better behaved than
    expanding the fractional operator and dividing polynomials.

    Args:
        phi: The ARCH weight.
        d: Fractional integration order.
        beta: The GARCH weight.
        truncation: Number of weights to generate.

    Returns:
        The weights ``lambda_1, ..., lambda_truncation``.
    """
    lam = np.empty(truncation)
    delta = np.empty(truncation)
    lam[0] = phi - beta + d
    delta[0] = d
    for i in range(1, truncation):
        delta[i] = (i - d) / (i + 1) * delta[i - 1]
        lam[i] = beta * lam[i - 1] + (delta[i] - phi * delta[i - 1])
    return lam


def _figarch_variance(
    resid: npt.NDArray[np.float64],
    omega: float,
    phi: float,
    d: float,
    beta: float,
    backcast: float,
    truncation: int = _DEFAULT_TRUNCATION,
) -> npt.NDArray[np.float64]:
    """Variance recursion for FIGARCH via its ARCH(infinity) representation.

    Weights beyond the available history are applied to ``backcast``, so the
    truncation tail contributes a constant rather than being silently dropped.

    Args:
        resid: Mean residuals.
        omega: Variance intercept.
        phi: The ARCH weight.
        d: Fractional integration order.
        beta: The GARCH weight.
        backcast: Pre-sample variance.
        truncation: ARCH(infinity) truncation lag.

    Returns:
        The conditional-variance path.
    """
    n = resid.shape[0]
    lam = _figarch_weights(phi, d, beta, truncation)
    r2 = resid**2
    sigma2 = np.empty(n)
    intercept = omega / (1.0 - beta)
    for t in range(n):
        m = min(t, truncation)
        acc = intercept
        if m > 0:
            acc += float(np.dot(lam[:m], r2[t - 1 :: -1][:m]))
        if truncation > m:
            acc += backcast * float(lam[m:truncation].sum())
        sigma2[t] = acc
    return sigma2


def _fit_garch(
    endog: npt.NDArray[np.float64],
    p: int,
    o: int,
    q: int,
    ar_lags: int,
    include_const: bool,
    vol: str,
) -> _GARCHFit:
    """Fit a GARCH, GJR, or EGARCH model by Gaussian maximum likelihood.

    Mean and variance parameters are estimated jointly rather than in two
    steps, so the reported likelihood is the true joint one.

    Args:
        endog: The series, typically returns or residuals.
        p: ARCH order.
        o: Asymmetry order.
        q: GARCH order.
        ar_lags: Conditional-mean AR order.
        include_const: Whether to estimate a mean intercept.
        vol: Volatility family (not ``"FIGARCH"``; see :func:`_fit_figarch`).

    Returns:
        The packed :class:`_GARCHFit`.
    """
    y = endog
    n_full = y.shape[0]
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
        var_raw0 = np.concatenate(
            [
                [np.log(var0 * (1 - a_init - b_init))],
                [_inv_softplus(a_init)] * p,
                [_inv_softplus(b_init)] * q,
            ]
        )
    elif vol == "GJR":
        var_raw0 = np.concatenate(
            [
                [np.log(var0 * (1 - a_init - b_init - 0.5 * g_init))],
                [_inv_softplus(a_init)] * p,
                [g_init] * o,
                [_inv_softplus(b_init)] * q,
            ]
        )
    else:
        var_raw0 = np.concatenate(
            [
                [np.log(var0) * (1 - 0.95)],
                [0.1] * p,
                [-0.05] * o,
                [0.95] * q,
            ]
        )
    theta0 = np.concatenate([mean0, var_raw0])
    m_idx = k_mean

    def unpack_var(
        v: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if vol == "GARCH":
            return (
                float(np.exp(v[0])),
                _softplus(v[1 : 1 + p]),
                np.zeros(0),
                _softplus(v[1 + p : 1 + p + q]),
            )
        if vol == "GJR":
            return (
                float(np.exp(v[0])),
                _softplus(v[1 : 1 + p]),
                v[1 + p : 1 + p + o],
                _softplus(v[1 + p + o : 1 + p + o + q]),
            )
        return (
            float(v[0]),
            v[1 : 1 + p],
            v[1 + p : 1 + p + o],
            v[1 + p + o : 1 + p + o + q],
        )

    def variance_path(
        resid: npt.NDArray[np.float64],
        omega: float,
        alpha: npt.NDArray[np.float64],
        gamma: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        if vol == "EGARCH":
            return _egarch_variance(resid, omega, alpha, gamma, beta, backcast)
        return _garch_variance(resid, omega, alpha, gamma, beta, backcast)

    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        mean = theta[:m_idx]
        resid = target - mean_x @ mean if k_mean else target
        omega, alpha, gamma, beta = unpack_var(theta[m_idx:])
        if vol == "EGARCH":
            if abs(beta.sum()) >= 0.999:
                return 1e10
        elif alpha.sum() + 0.5 * gamma.sum() + beta.sum() >= 0.999:
            return 1e10
        sigma2 = variance_path(resid, omega, alpha, gamma, beta)
        if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
            return 1e10
        ll = -0.5 * np.sum(_LOG_2PI + np.log(sigma2) + resid**2 / sigma2)
        return float(-ll) if np.isfinite(ll) else 1e10

    result = minimize(negloglik, theta0, method="L-BFGS-B")
    theta = np.asarray(result.x, dtype=np.float64)
    mean = theta[:m_idx]
    fitted = mean_x @ mean if k_mean else np.zeros(n)
    resid = target - fitted
    omega, alpha, gamma, beta = unpack_var(theta[m_idx:])
    sigma2 = variance_path(resid, omega, alpha, gamma, beta)
    return _GARCHFit(
        const=float(mean[0]) if include_const else None,
        ar_params=np.asarray(mean[1:] if include_const else mean, dtype=np.float64),
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        fractional_d=None,
        conditional_variance=sigma2,
        resid=resid,
        fittedvalues=fitted,
        llf=-float(result.fun),
        nobs=n,
        n_params=k_mean + 1 + p + o + q,
    )


def _fit_figarch(
    endog: npt.NDArray[np.float64],
    include_const: bool,
    truncation: int = _DEFAULT_TRUNCATION,
) -> _GARCHFit:
    """Fit a FIGARCH(1, d, 1) model by Gaussian maximum likelihood.

    Admissibility is checked on a short prefix of the lambda weights each
    iteration: a negative weight implies a negative conditional variance
    somewhere in the sample, so the draw is rejected before the full
    recursion runs.

    Args:
        endog: The series, typically returns or residuals.
        include_const: Whether to estimate a mean intercept.
        truncation: ARCH(infinity) truncation lag.

    Returns:
        The packed :class:`_GARCHFit`, with ``fractional_d`` populated.
    """
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
    theta0 = np.concatenate([mean0, [np.log(var0 * 0.4), -1.0, -0.2, 0.4]])

    def sig(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def unpack(
        theta: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], float, float, float, float]:
        return (
            theta[:k_mean],
            float(np.exp(theta[k_mean])),
            sig(float(theta[k_mean + 1])),
            sig(float(theta[k_mean + 2])),
            sig(float(theta[k_mean + 3])),
        )

    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        mean, omega, phi, d, beta = unpack(theta)
        resid = endog - mean_x @ mean if k_mean else endog
        lam = _figarch_weights(phi, d, beta, min(truncation, 200))
        if np.any(lam < -1e-6):
            return 1e10
        sigma2 = _figarch_variance(resid, omega, phi, d, beta, backcast, truncation)
        if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
            return 1e10
        ll = -0.5 * np.sum(_LOG_2PI + np.log(sigma2) + resid**2 / sigma2)
        return float(-ll) if np.isfinite(ll) else 1e10

    result = minimize(negloglik, theta0, method="L-BFGS-B")
    mean, omega, phi, d, beta = unpack(np.asarray(result.x, dtype=np.float64))
    fitted = mean_x @ mean if k_mean else np.zeros(n)
    resid = endog - fitted
    sigma2 = _figarch_variance(resid, omega, phi, d, beta, backcast, truncation)
    return _GARCHFit(
        const=float(mean[0]) if include_const else None,
        ar_params=np.zeros(0),
        omega=omega,
        alpha=np.array([phi]),
        gamma=np.zeros(0),
        beta=np.array([beta]),
        fractional_d=d,
        conditional_variance=sigma2,
        resid=resid,
        fittedvalues=fitted,
        llf=-float(result.fun),
        nobs=n,
        n_params=k_mean + 3,
    )


class _GARCHModel[R](_UnivariateModel[R]):
    """Shared specification surface for the conditional-variance family.

    Args:
        endog: The series, typically returns or residuals.
        vol: Volatility family.
        p: ARCH order.
        o: Asymmetry order.
        q: GARCH order.
        ar_lags: Conditional-mean AR order.
        mean: ``"constant"`` or ``"zero"``.
        truncation: ARCH(infinity) truncation lag (FIGARCH only).

    Raises:
        SpecificationError: If an order is negative, ``GARCH`` is given a
            non-zero asymmetry order, or ``GJR``/``EGARCH`` is given ``o < 1``.
        DimensionError: If the series is too short for the specification.
    """

    __slots__ = (
        "_ar_lags",
        "_const",
        "_o",
        "_p",
        "_q",
        "_truncation",
        "_vol",
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        vol: str,
        p: int,
        o: int,
        q: int,
        ar_lags: int = 0,
        mean: str = "constant",
        truncation: int = _DEFAULT_TRUNCATION,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog)
        self._vol = validate_choice(vol, _VOL_FAMILIES, "vol")
        self._p = validate_order(p, "p")
        self._o = validate_order(o, "o")
        self._q = validate_order(q, "q")
        self._ar_lags = validate_order(ar_lags, "ar_lags")
        self._const = validate_choice(mean, _MEANS, "mean") == "constant"
        self._truncation = validate_order(truncation, "truncation", minimum=1)
        if self._vol == "GARCH" and self._o != 0:
            raise SpecificationError("GARCH has no asymmetry term; set the asymmetry order o = 0.")
        if self._vol in ("GJR", "EGARCH") and self._o < 1:
            raise SpecificationError(f"{self._vol} requires an asymmetry order o >= 1.")
        self._ensure_length(
            max(self._p, self._o, self._q) + self._ar_lags + 2,
            f"{self._vol}({self._p}, {self._o}, {self._q})",
        )

    @property
    def vol(self) -> str:
        """The volatility family."""
        return self._vol

    @property
    def order(self) -> tuple[int, int, int]:
        """The variance order ``(p, o, q)``."""
        return (self._p, self._o, self._q)

    def _fit_family(self) -> _GARCHFit:
        """Dispatch to the engine for this volatility family."""
        if self._vol == "FIGARCH":
            return _fit_figarch(self.endog, self._const, self._truncation)
        return _fit_garch(
            self.endog,
            self._p,
            self._o,
            self._q,
            self._ar_lags,
            self._const,
            self._vol,
        )
