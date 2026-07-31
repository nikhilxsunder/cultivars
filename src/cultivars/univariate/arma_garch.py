# filepath: /src/cultivars/univariate/arma_garch.py
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

"""ARMA-mean conditional-variance models: ARMA-{GARCH, GJR, EGARCH, FIGARCH}.

The GARCH-family estimators in :mod:`cultivars.univariate.garch` model the
conditional mean as a constant plus optional pure-AR terms. This module supplies
the remaining piece of the univariate surface: a full **ARMA(p, q) conditional
mean** estimated *jointly* with any of the four volatility processes, so
ARMA-GARCH, ARMA-GJR, ARMA-EGARCH, and ARMA-FIGARCH are all reachable — and
FIGARCH gains a mean beyond a constant for the first time.

One mean, four backends
--------------------------------------------------------------------------
Rather than a class per (mean x variance) pair — the combinatorial explosion the
architecture forbids — a single :class:`ARMAGARCH` estimator carries the ARMA
mean and takes the volatility family as a ``vol`` strategy. The four variance
recursions are reused verbatim from :mod:`cultivars.univariate.garch`; only the
mean layer is new. The mean is computed by the conditional-sum-of-squares (CSS)
ARMA recursion rather than the exact-ML state-space form used by
:mod:`cultivars.univariate.arma`, because under GARCH innovations are
heteroskedastic and the constant-variance Kalman likelihood no longer applies;
the joint Gaussian likelihood
``-0.5 sum_t [ log(2 pi) + log(sigma2_t) + eps_t^2 / sigma2_t ]``
is maximized over the mean and variance parameters together.

Stationarity (AR) and invertibility (MA) of the mean are imposed structurally by
the partial-autocorrelation reparameterization in
:mod:`cultivars.univariate._base`, so the CSS residual recursion can never
explode and the optimizer searches an unconstrained space.

References:
    Bollerslev (1986); Glosten, Jagannathan, Runkle (1993); Nelson (1991);
    Baillie, Bollerslev & Mikkelsen (1996); Hannan & Rissanen (1982).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._base import (
    InformationCriteria,
    coeffs_to_pacf,
    information_criteria,
    pacf_to_coeffs,
)
from .garch import (
    Vol,
    _backcast,
    _egarch_variance,
    _figarch_variance,
    _garch_variance,
    _inv_softplus,
    _softplus,
)

# --------------------------------------------------------------------------
# ARMA conditional mean (conditional sum of squares)
# --------------------------------------------------------------------------

def _arma_residuals(
    w: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    theta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """CSS residuals of ``phi(L) w_t = theta(L) eps_t`` (plus-sign MA convention).

    ``eps_t = w_t - sum_i phi_i w_{t-i} - sum_j theta_j eps_{t-j}`` with pre-sample
    ``w`` and ``eps`` set to zero. ``w`` is the demeaned series.
    """
    n, p, q = w.shape[0], phi.size, theta.size
    eps = np.empty(n, dtype=np.float64)
    for t in range(n):
        value = w[t]
        for i in range(p):
            if t - 1 - i >= 0:
                value -= phi[i] * w[t - 1 - i]
        for j in range(q):
            if t - 1 - j >= 0:
                value -= theta[j] * eps[t - 1 - j]
        eps[t] = value
    return eps


def _hannan_rissanen(
    w: npt.NDArray[np.float64], p: int, q: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Two-stage Hannan-Rissanen starting values for (phi, theta)."""
    n = w.shape[0]
    if p == 0 and q == 0:
        return np.zeros(0), np.zeros(0)
    long_order = int(min(max(10, p + q + 5), n // 2 - 1))
    resid = w.copy()
    if long_order >= 1 and n > long_order:
        design = np.column_stack([w[long_order - i : n - i] for i in range(1, long_order + 1)])
        coeffs, *_ = np.linalg.lstsq(design, w[long_order:], rcond=None)
        resid = np.zeros(n)
        resid[long_order:] = w[long_order:] - design @ coeffs
    start = max(p, q)
    cols: list[npt.NDArray[np.float64]] = []
    for i in range(1, p + 1):
        cols.append(w[start - i : n - i])
    for j in range(1, q + 1):
        cols.append(resid[start - j : n - j])
    if not cols:
        return np.zeros(0), np.zeros(0)
    beta, *_ = np.linalg.lstsq(np.column_stack(cols), w[start:], rcond=None)
    return np.asarray(beta[:p], dtype=np.float64), np.asarray(beta[p:], dtype=np.float64)


def _pack_pacf(coeffs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Map raw ARMA coefficients to unconstrained space via the pacf transform."""
    if coeffs.size == 0:
        return coeffs
    try:
        pacf = np.clip(coeffs_to_pacf(coeffs), -_PACF_CLIP, _PACF_CLIP)
    except (ValueError, FloatingPointError):
        pacf = np.zeros(coeffs.size)
    return np.arctanh(pacf)


# --------------------------------------------------------------------------
# Variance-block bookkeeping (reuses garch.py recursions)
# --------------------------------------------------------------------------

def _variance_block_size(vol: Vol, p: int, o: int, q: int) -> int:
    if vol == "FIGARCH":
        return 4
    if vol == "GARCH":
        return 1 + p + q
    return 1 + p + o + q            # GJR, EGARCH


def _sig(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _unpack_variance(
    raw: npt.NDArray[np.float64], vol: Vol, p: int, o: int, q: int
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Return (omega, alpha, gamma, beta, frac_d); frac_d is ``nan`` unless FIGARCH."""
    if vol == "FIGARCH":
        omega = float(np.exp(raw[0]))
        fig_phi = _sig(float(raw[1]))
        fig_d = _sig(float(raw[2]))
        fig_beta = _sig(float(raw[3]))
        return omega, np.array([fig_phi]), np.zeros(0), np.array([fig_beta]), fig_d
    if vol == "GARCH":
        omega = float(np.exp(raw[0]))
        alpha = _softplus(raw[1 : 1 + p])
        gamma = np.zeros(0)
        beta = _softplus(raw[1 + p : 1 + p + q])
    elif vol == "GJR":
        omega = float(np.exp(raw[0]))
        alpha = _softplus(raw[1 : 1 + p])
        gamma = raw[1 + p : 1 + p + o]
        beta = _softplus(raw[1 + p + o : 1 + p + o + q])
    else:  # EGARCH
        omega = float(raw[0])
        alpha = raw[1 : 1 + p]
        gamma = raw[1 + p : 1 + p + o]
        beta = raw[1 + p + o : 1 + p + o + q]
    return omega, alpha, gamma, beta, float("nan")


def _variance_init(vol: Vol, var0: float, p: int, o: int, q: int) -> npt.NDArray[np.float64]:
    a_init, b_init, g_init = 0.05, 0.90, 0.05
    if vol == "FIGARCH":
        return np.array([np.log(var0 * 0.4), -1.0, -0.2, 0.4])
    if vol == "GARCH":
        return np.concatenate([
            [np.log(var0 * (1 - a_init - b_init))],
            [_inv_softplus(a_init)] * p,
            [_inv_softplus(b_init)] * q,
        ])
    if vol == "GJR":
        return np.concatenate([
            [np.log(var0 * (1 - a_init - b_init - 0.5 * g_init))],
            [_inv_softplus(a_init)] * p,
            [g_init] * o,
            [_inv_softplus(b_init)] * q,
        ])
    return np.concatenate([                      # EGARCH
        [np.log(var0) * (1 - 0.95)],
        [0.1] * p,
        [-0.05] * o,
        [0.95] * q,
    ])


def _conditional_variance(
    resid: npt.NDArray[np.float64],
    vol: Vol,
    omega: float,
    alpha: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    frac_d: float,
    backcast: float,
    truncation: int,
) -> npt.NDArray[np.float64]:
    if vol == "FIGARCH":
        return _figarch_variance(
            resid, omega, float(alpha[0]), frac_d, float(beta[0]), backcast, truncation
        )
    if vol == "EGARCH":
        return _egarch_variance(resid, omega, alpha, gamma, beta, backcast)
    return _garch_variance(resid, omega, alpha, gamma, beta, backcast)


def _persistence_ok(
    vol: Vol,
    alpha: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    frac_d: float,
    truncation: int,
) -> bool:
    if vol == "FIGARCH":
        from .garch import _figarch_weights

        lam = _figarch_weights(float(alpha[0]), frac_d, float(beta[0]), min(truncation, 200))
        return bool(np.all(lam >= -1e-6))
    if vol == "EGARCH":
        return abs(beta.sum()) < 0.999
    return alpha.sum() + 0.5 * gamma.sum() + beta.sum() < 0.999


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ARMAGARCHResult:
    """Fitted ARMA-mean conditional-variance model.

    Attributes:
        vol: Volatility family (``"GARCH"``, ``"GJR"``, ``"EGARCH"``, ``"FIGARCH"``).
        order: Mean ARMA order ``(p, q)``.
        garch_order: Variance order ``(p, o, q)`` (``(1, d, 1)`` for FIGARCH).
        const: Mean intercept ``c`` (``None`` if ``mean == "zero"``).
        ar_params: Mean AR coefficients.
        ma_params: Mean MA coefficients.
        omega: Variance intercept.
        alpha: ARCH coefficients (the ``phi`` weight for FIGARCH).
        gamma: Asymmetry coefficients.
        beta: GARCH coefficients.
        fractional_d: Fractional integration order (FIGARCH only, else ``None``).
        sigma2: Fitted conditional variances.
        resid: Mean residuals.
        fittedvalues: Fitted conditional means.
        llf: Maximized Gaussian log-likelihood.
        nobs: Effective observations (after conditioning on ``max(p, q)`` lags).
        schema_version: Serialization schema version.
    """

    vol: Vol
    order: tuple[int, int]
    garch_order: tuple[int, int, int]
    const: float | None
    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    omega: float
    alpha: npt.NDArray[np.float64]
    gamma: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    fractional_d: float | None
    sigma2: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    endog: npt.NDArray[np.float64] = field(repr=False)
    _burn: int = field(repr=False, default=0)
    _n_params: int = field(repr=False, default=0)
    schema_version: int = _SCHEMA_VERSION

    @property
    def information_criteria(self) -> InformationCriteria:
        """AIC/BIC/HQIC for this fit."""
        return information_criteria(self.llf, self.nobs, self._n_params)

    @property
    def aic(self) -> float:
        return self.information_criteria.aic

    @property
    def bic(self) -> float:
        return self.information_criteria.bic

    @property
    def persistence(self) -> float:
        """Variance persistence (``sum beta`` for EGARCH/FIGARCH, else ARCH+GARCH)."""
        if self.vol in ("EGARCH", "FIGARCH"):
            return float(self.beta.sum())
        return float(self.alpha.sum() + 0.5 * self.gamma.sum() + self.beta.sum())

    def forecast(self, h: int) -> npt.NDArray[np.float64]:
        """Return ``h``-step-ahead conditional-mean forecasts ``E[y_{T+k}]``.

        Args:
            h: Forecast horizon; a positive integer.

        Returns:
            Point forecasts of shape ``(h,)``.

        Raises:
            DimensionError: If ``h`` is not positive.
        """
        if h < 1:
            raise DimensionError(f"forecast horizon h must be >= 1; got {h}.")
        p, q = self.order
        c = self.const if self.const is not None else 0.0
        w_hist = [float(v) - c for v in self.endog]
        e_hist = list(self.resid)
        out = np.empty(h, dtype=np.float64)
        for k in range(h):
            w_hat = 0.0
            for i in range(p):
                w_hat += self.ar_params[i] * w_hist[-1 - i]
            for j in range(q):
                w_hat += self.ma_params[j] * e_hist[-1 - j]
            out[k] = c + w_hat
            w_hist.append(w_hat)
            e_hist.append(0.0)                    # future innovation expected to be zero
        return out

    def forecast_variance(self, h: int) -> npt.NDArray[np.float64]:
        """Return ``h``-step-ahead conditional-variance forecasts.

        Implemented for ``GARCH`` and ``GJR`` (the standard multi-step recursion,
        with future ``E[eps^2] = sigma2`` and, for GJR, ``E[1_{eps<0} eps^2] =
        0.5 sigma2``). EGARCH and FIGARCH multi-step variance forecasts require
        simulation and are not provided here.

        Args:
            h: Forecast horizon; a positive integer.

        Returns:
            Variance forecasts of shape ``(h,)``.

        Raises:
            DimensionError: If ``h`` is not positive.
            SpecificationError: If ``vol`` is ``EGARCH`` or ``FIGARCH``.
        """
        if h < 1:
            raise DimensionError(f"forecast horizon h must be >= 1; got {h}.")
        if self.vol in ("EGARCH", "FIGARCH"):
            raise SpecificationError(
                f"multi-step variance forecasts for {self.vol} require simulation; "
                "not provided in this release."
            )
        p, o, q = self.garch_order
        neg = (self.resid < 0.0).astype(np.float64)
        r2 = list(self.resid ** 2)
        asy = [float(r2[i] * neg[i]) for i in range(len(r2))]   # observed 1_{eps<0} eps^2
        s2 = list(self.sigma2)
        out = np.empty(h, dtype=np.float64)
        for k in range(h):
            value = self.omega
            for i in range(p):
                value += self.alpha[i] * r2[-1 - i]
            for m in range(o):
                value += self.gamma[m] * asy[-1 - m]
            for j in range(q):
                value += self.beta[j] * s2[-1 - j]
            out[k] = value
            s2.append(value)
            r2.append(value)                     # E[eps^2] = sigma2 for future steps
            asy.append(0.5 * value)              # E[1_{eps<0} eps^2] = 0.5 sigma2
        return out


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

def _fit_arma_garch(
    y: npt.NDArray[np.float64],
    mean_p: int,
    mean_q: int,
    include_const: bool,
    vol: Vol,
    p: int,
    o: int,
    q: int,
    truncation: int,
) -> ARMAGARCHResult:
    burn = max(mean_p, mean_q)

    # Warm start: mean via Hannan-Rissanen on the demeaned series.
    c0 = float(y.mean()) if include_const else 0.0
    w0 = y - c0
    phi0, theta0_ma = _hannan_rissanen(w0, mean_p, mean_q)
    resid0 = _arma_residuals(w0, phi0, theta0_ma)[burn:]
    var0 = max(float(np.var(resid0)), 1e-8)
    backcast = _backcast(resid0)

    mean_start: list[npt.NDArray[np.float64]] = []
    if include_const:
        mean_start.append(np.array([c0]))
    mean_start.append(_pack_pacf(phi0))
    mean_start.append(_pack_pacf(theta0_ma))
    var_start = _variance_init(vol, var0, p, o, q)
    theta_init = np.concatenate([*mean_start, var_start])

    i_const = 1 if include_const else 0
    i_ar = i_const
    i_ma = i_ar + mean_p
    i_var = i_ma + mean_q

    def unpack_mean(
        theta: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        c = float(theta[0]) if include_const else 0.0
        phi = pacf_to_coeffs(np.tanh(theta[i_ar:i_ma])) if mean_p else np.zeros(0)
        theta_ma = pacf_to_coeffs(np.tanh(theta[i_ma:i_var])) if mean_q else np.zeros(0)
        return c, phi, theta_ma

    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        c, phi, theta_ma = unpack_mean(theta)
        resid = _arma_residuals(y - c, phi, theta_ma)[burn:]
        omega, alpha, gamma, beta, frac_d = _unpack_variance(theta[i_var:], vol, p, o, q)
        if not _persistence_ok(vol, alpha, gamma, beta, frac_d, truncation):
            return 1e10
        try:
            sigma2 = _conditional_variance(
                resid, vol, omega, alpha, gamma, beta, frac_d, backcast, truncation
            )
        except (ValueError, FloatingPointError):
            return 1e10
        if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
            return 1e10
        ll = -0.5 * np.sum(_LOG_2PI + np.log(sigma2) + resid ** 2 / sigma2)
        return float(-ll) if np.isfinite(ll) else 1e10

    result = minimize(negloglik, theta_init, method="L-BFGS-B")
    theta = np.asarray(result.x, dtype=np.float64)
    c, phi, theta_ma = unpack_mean(theta)
    eps_full = _arma_residuals(y - c, phi, theta_ma)
    resid = eps_full[burn:]
    omega, alpha, gamma, beta, frac_d = _unpack_variance(theta[i_var:], vol, p, o, q)
    sigma2 = _conditional_variance(
        resid, vol, omega, alpha, gamma, beta, frac_d, backcast, truncation
    )
    fitted = y[burn:] - resid
    n_params = i_const + mean_p + mean_q + _variance_block_size(vol, p, o, q)
    return ARMAGARCHResult(
        vol=vol,
        order=(mean_p, mean_q),
        garch_order=(p, o, q),
        const=c if include_const else None,
        ar_params=phi,
        ma_params=theta_ma,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        fractional_d=frac_d if vol == "FIGARCH" else None,
        sigma2=sigma2,
        resid=resid,
        fittedvalues=fitted,
        llf=-float(result.fun),
        nobs=resid.shape[0],
        endog=y,
        _burn=burn,
        _n_params=n_params,
    )


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------

class ARMAGARCH:
    """ARMA(p, q) mean with a GARCH-family conditional variance.

    A single estimator spanning ARMA-GARCH, ARMA-GJR, ARMA-EGARCH, and
    ARMA-FIGARCH: the volatility family is selected by ``vol`` and the four
    variance recursions are shared with :mod:`cultivars.univariate.garch`.

    Args:
        endog: Univariate series (1-D array-like), typically a return series.
        order: Mean ARMA order ``(p, q)``. Defaults to ``(1, 0)``.
        vol: Volatility family — ``"GARCH"`` (default), ``"GJR"``, ``"EGARCH"``,
            or ``"FIGARCH"``.
        garch_order: Variance order ``(p, o, q)`` (ARCH, asymmetry, GARCH lags).
            Defaults to a sensible value per family; ignored for FIGARCH, which is
            fixed at ``(1, d, 1)``.
        mean: ``"constant"`` (default) or ``"zero"``.
        truncation: ARCH(inf) truncation lag (FIGARCH only).

    Raises:
        SpecificationError: If the orders/flags are invalid or the series is too
            short.
        DimensionError: If ``endog`` is not one-dimensional.
        NumericalError: If ``endog`` contains non-finite values.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> # AR(1) mean + GARCH(1,1) volatility.
        >>> n = 3000
        >>> eps = np.zeros(n); s2 = np.zeros(n); s2[0] = 1.0
        >>> for t in range(1, n):
        ...     s2[t] = 0.02 + 0.08 * eps[t - 1] ** 2 + 0.90 * s2[t - 1]
        ...     eps[t] = np.sqrt(s2[t]) * rng.standard_normal()
        >>> y = np.zeros(n)
        >>> for t in range(1, n):
        ...     y[t] = 0.05 + 0.5 * y[t - 1] + eps[t]
        >>> res = ARMAGARCH(y, order=(1, 0), vol="GARCH").fit()
        >>> bool(0.3 < res.ar_params[0] < 0.7)
        True
        >>> res.forecast(3).shape
        (3,)
    """

    __slots__ = ("_y", "_p", "_q", "_vol", "_go", "_const", "_truncation")

    def __init__(
        self,
        endog: npt.ArrayLike,
        order: tuple[int, int] = (1, 0),
        vol: Vol = "GARCH",
        *,
        garch_order: tuple[int, int, int] | None = None,
        mean: str = "constant",
        truncation: int = 1000,
    ) -> None:
        """Initialize the ARMA-GARCH specification."""
        y = np.asarray(endog, dtype=np.float64)
        if y.ndim != 1:
            raise DimensionError(f"endog must be one-dimensional; got shape {y.shape}.")
        if not np.all(np.isfinite(y)):
            raise NumericalError("endog contains non-finite values.")
        p_mean, q_mean = order
        if min(p_mean, q_mean) < 0:
            raise SpecificationError(f"mean order (p, q) must be non-negative; got {order}.")
        if vol not in ("GARCH", "GJR", "EGARCH", "FIGARCH"):
            raise SpecificationError(
                f"vol must be one of 'GARCH', 'GJR', 'EGARCH', 'FIGARCH'; got {vol!r}."
            )
        if mean not in ("constant", "zero"):
            raise SpecificationError(f"mean must be 'constant' or 'zero'; got {mean!r}.")
        go = garch_order if garch_order is not None else _DEFAULT_GARCH_ORDER[vol]
        gp, go_, gq = go
        if min(gp, go_, gq) < 0:
            raise SpecificationError(f"garch_order must be non-negative; got {go}.")
        if vol == "GARCH" and go_ != 0:
            raise SpecificationError("GARCH has no asymmetry term; set garch_order o = 0.")
        if vol in ("GJR", "EGARCH") and go_ < 1:
            raise SpecificationError(f"{vol} requires an asymmetry order o >= 1.")
        if truncation < 1:
            raise SpecificationError(f"truncation must be >= 1; got {truncation}.")
        min_len = 4 * (max(p_mean, q_mean) + max(gp, gq) + 1)
        if y.shape[0] < min_len:
            raise SpecificationError(
                f"series of length {y.shape[0]} is too short for ARMA{order}-{vol} "
                f"(need at least {min_len})."
            )
        self._y = y
        self._p, self._q = int(p_mean), int(q_mean)
        self._vol: Vol = vol
        self._go = (int(gp), int(go_), int(gq))
        self._const = mean == "constant"
        self._truncation = int(truncation)

    @property
    def order(self) -> tuple[int, int]:
        """The mean ARMA order ``(p, q)``."""
        return (self._p, self._q)

    @property
    def vol(self) -> Vol:
        """The volatility family."""
        return self._vol

    def fit(self) -> ARMAGARCHResult:
        """Estimate the mean and variance jointly by Gaussian maximum likelihood."""
        gp, go_, gq = self._go
        return _fit_arma_garch(
            self._y, self._p, self._q, self._const, self._vol,
            gp, go_, gq, self._truncation,
        )