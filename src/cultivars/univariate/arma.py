# filepath: /src/cultivars/univariate/arma.py
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

"""ARMA / ARIMA / SARIMA / SARIMAX — the linear-Gaussian mean engine.

A single exact-maximum-likelihood engine estimates the whole group by composing
primitives you already have:

- The conditional mean is a (multiplicative seasonal) ARMA in the Harvey
  state-space form, so estimation runs through :class:`LinearGaussianStateSpace`.
- Integration (``d``, ``D``) is applied by simple differencing
  (:func:`cultivars.core.transforms.difference` / ``seasonal_difference``); the
  remaining structure is a stationary ARMA fit exactly on the differenced series.
- Multiplicative seasonal polynomials phi(L)Phi(L**s), theta(L)Theta(L**s) are
  formed with :class:`LagPolynomial` multiplication.
- Deterministic terms (constant, trend) and exogenous regressors enter the
  observation intercept as a regression with ARMA errors, so ``ARIMAX`` /
  ``SARIMAX`` are just ``ARIMA`` / ``SARIMA`` with ``exog`` supplied.

Public classes ``ARMA``, ``ARIMA``, ``SARIMA`` are thin fronts over one engine;
all return :class:`ARMAResult`.

Stationarity (AR) and invertibility (MA) are enforced structurally by the
partial-autocorrelation reparameterization in :mod:`cultivars.univariate._base`,
so the optimizer searches an unconstrained space.

References:
    Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space
    Methods*, ch. 3 (ARMA state-space form).
    Box, Jenkins, Reinsel, Ljung (2015). *Time Series Analysis*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from ..core.lag import LagPolynomial
from ..core.stability import StabilityResult, assess_stability
from ..core.transforms import difference, seasonal_difference
from ..exceptions import DimensionError, NumericalError, SpecificationError
from ..state_space.linear_gaussian import LinearGaussianStateSpace
from .. univariate._base import (
    InformationCriteria,
    coeffs_to_pacf,
    information_criteria,
    pacf_to_coeffs,
)

_SCHEMA_VERSION = 1
Trend = Literal["n", "c", "ct"]


# --------------------------------------------------------------------------
# Polynomial expansion (multiplicative seasonal <-> expanded coefficients)
# --------------------------------------------------------------------------

def _expand_ar(phi: npt.NDArray[np.float64], sphi: npt.NDArray[np.float64], s: int) -> npt.NDArray[np.float64]:
    """Expanded AR coefficients of phi(L) * Phi(L**s)."""
    poly = LagPolynomial.from_ar_coeffs(phi) if phi.size else LagPolynomial([1.0])
    if sphi.size:
        scoef = np.zeros(s * sphi.size + 1)
        scoef[0] = 1.0
        for k in range(sphi.size):
            scoef[(k + 1) * s] = -sphi[k]
        poly = poly * LagPolynomial(scoef)
    return poly.ar_coeffs()


def _expand_ma(theta: npt.NDArray[np.float64], stheta: npt.NDArray[np.float64], s: int) -> npt.NDArray[np.float64]:
    """Expanded MA coefficients of theta(L) * Theta(L**s) (plus-sign convention)."""
    base = np.concatenate([[1.0], theta]) if theta.size else np.array([1.0])
    poly = LagPolynomial(base)
    if stheta.size:
        scoef = np.zeros(s * stheta.size + 1)
        scoef[0] = 1.0
        for k in range(stheta.size):
            scoef[(k + 1) * s] = stheta[k]
        poly = poly * LagPolynomial(scoef)
    return poly.coefficients[1:]


def _arma_state_space(
    phi_star: npt.NDArray[np.float64],
    theta_star: npt.NDArray[np.float64],
    sigma2: float,
    obs_intercept: npt.NDArray[np.float64],
) -> LinearGaussianStateSpace:
    """Build the Harvey state-space form of an ARMA(len phi*, len theta*)."""
    r = max(phi_star.size, theta_star.size + 1)
    phi_full = np.zeros(r)
    phi_full[: phi_star.size] = phi_star
    transition = np.zeros((r, r))
    transition[:, 0] = phi_full
    for i in range(r - 1):
        transition[i, i + 1] = 1.0
    selection = np.zeros((r, 1))
    selection[0, 0] = 1.0
    selection[1 : 1 + theta_star.size, 0] = theta_star
    design = np.zeros((1, r))
    design[0, 0] = 1.0
    return LinearGaussianStateSpace(
        design,
        np.zeros((1, 1)),
        transition,
        selection,
        np.array([[sigma2]]),
        obs_intercept=obs_intercept.reshape(-1, 1),
    )


# --------------------------------------------------------------------------
# Deterministic design
# --------------------------------------------------------------------------

def _deterministic(trend: Trend, nobs: int) -> npt.NDArray[np.float64]:
    cols: list[npt.NDArray[np.float64]] = []
    if trend in ("c", "ct"):
        cols.append(np.ones(nobs))
    if trend == "ct":
        cols.append(np.arange(1, nobs + 1, dtype=np.float64))
    if not cols:
        return np.zeros((nobs, 0))
    return np.column_stack(cols)


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ARMAResult:
    """Fitted (S)ARIMA(X) model.

    Attributes:
        order: ``(p, d, q)``.
        seasonal_order: ``(P, D, Q, s)``.
        trend: Deterministic specification.
        ar_params: Non-seasonal AR coefficients ``(phi_1, ..., phi_p)``.
        ma_params: Non-seasonal MA coefficients ``(theta_1, ..., theta_q)``.
        seasonal_ar_params: Seasonal AR coefficients.
        seasonal_ma_params: Seasonal MA coefficients.
        beta: Deterministic + exogenous regression coefficients.
        sigma2: Innovation variance.
        llf: Exact log-likelihood on the (differenced) modeling series.
        nobs: Effective observations after differencing.
        resid: One-step residuals on the modeling series.
        fittedvalues: One-step fitted values on the modeling series.
        schema_version: Serialization schema version.
    """

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    trend: Trend
    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    seasonal_ar_params: npt.NDArray[np.float64]
    seasonal_ma_params: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    sigma2: float
    llf: float
    nobs: int
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    _n_params: int = field(repr=False)
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
    def stability(self) -> StabilityResult:
        """Stationarity of the expanded AR polynomial (None-safe for pure MA)."""
        phi_star = _expand_ar(self.ar_params, self.seasonal_ar_params, self.seasonal_order[3])
        if phi_star.size == 0:
            return assess_stability(np.zeros(0))
        return assess_stability(phi_star)

    @property
    def is_stationary(self) -> bool:
        return self.stability.is_stable


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

def _difference_series(
    y: npt.NDArray[np.float64], d: int, capital_d: int, s: int
) -> npt.NDArray[np.float64]:
    w = y
    if d > 0:
        w = difference(w, d)
    if capital_d > 0:
        w = seasonal_difference(w, s, capital_d)
    return w


def _fit_sarimax(
    endog: npt.NDArray[np.float64],
    exog: npt.NDArray[np.float64] | None,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    trend: Trend,
) -> ARMAResult:
    p, d, q = order
    cap_p, cap_d, cap_q, s = seasonal_order
    w = _difference_series(endog, d, cap_d, s)
    n_eff = w.shape[0]
    det = _deterministic(trend, n_eff)
    if exog is not None:
        exog_w = _difference_series(exog, d, cap_d, s) if (d or cap_d) else exog
        design_x = np.column_stack([det, exog_w]) if det.shape[1] else exog_w
    else:
        design_x = det
    k_beta = design_x.shape[1]

    # starting values
    if k_beta:
        beta0, _r, _rk, _sv = np.linalg.lstsq(design_x, w, rcond=None)
        resid0 = w - design_x @ beta0
    else:
        beta0 = np.zeros(0)
        resid0 = w - w.mean()
    sigma2_0 = max(float(resid0 @ resid0) / n_eff, 1e-8)
    # short AR init for the non-seasonal AR block
    ar0 = np.zeros(p)
    if p:
        lag_mat = np.column_stack([resid0[p - i - 1 : n_eff - i - 1] for i in range(p)])
        tgt = resid0[p:]
        try:
            ar0 = np.asarray(np.linalg.lstsq(lag_mat, tgt, rcond=None)[0], dtype=np.float64)
            if not assess_stability(ar0).is_stable:
                ar0 = np.zeros(p)
        except np.linalg.LinAlgError:
            ar0 = np.zeros(p)

    def pack_ar(coeffs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if coeffs.size == 0:
            return coeffs
        pac = np.clip(coeffs_to_pacf(coeffs), -0.999, 0.999)
        return np.arctanh(pac)

    psi = np.concatenate([
        beta0,
        pack_ar(ar0),
        np.zeros(cap_p),
        np.zeros(q),
        np.zeros(cap_q),
        [np.log(sigma2_0)],
    ])

    idx = np.cumsum([k_beta, p, cap_p, q, cap_q])

    def unpack(
        vec: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        beta = vec[: idx[0]]
        phi = pacf_to_coeffs(np.tanh(vec[idx[0] : idx[1]])) if p else np.zeros(0)
        sphi = pacf_to_coeffs(np.tanh(vec[idx[1] : idx[2]])) if cap_p else np.zeros(0)
        theta_c = -pacf_to_coeffs(np.tanh(vec[idx[2] : idx[3]])) if q else np.zeros(0)
        stheta = -pacf_to_coeffs(np.tanh(vec[idx[3] : idx[4]])) if cap_q else np.zeros(0)
        sigma2 = float(np.exp(vec[idx[4]]))
        return beta, phi, sphi, theta_c, stheta, sigma2

    def obs_intercept(beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return design_x @ beta if k_beta else np.zeros(n_eff)

    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        beta, phi, sphi, theta_c, stheta, sigma2 = unpack(theta)
        phi_star = _expand_ar(phi, sphi, s)
        theta_star = _expand_ma(theta_c, stheta, s)
        try:
            ss = _arma_state_space(phi_star, theta_star, sigma2, obs_intercept(beta))
            return -ss.loglikelihood(w)
        except (NumericalError, np.linalg.LinAlgError):
            return 1e10

    result = minimize(negloglik, psi, method="L-BFGS-B")
    beta, phi, sphi, theta_c, stheta, sigma2 = unpack(np.asarray(result.x, dtype=np.float64))
    phi_star = _expand_ar(phi, sphi, s)
    theta_star = _expand_ma(theta_c, stheta, s)
    ss = _arma_state_space(phi_star, theta_star, sigma2, obs_intercept(beta))
    filt = ss.filter(w)
    fitted = filt.predicted_state[:, 0] + obs_intercept(beta)
    resid = w - fitted
    n_params = k_beta + p + cap_p + q + cap_q + 1
    return ARMAResult(
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        ar_params=phi,
        ma_params=theta_c,
        seasonal_ar_params=sphi,
        seasonal_ma_params=stheta,
        beta=beta,
        sigma2=sigma2,
        llf=-float(result.fun),
        nobs=n_eff,
        resid=resid,
        fittedvalues=fitted,
        _n_params=n_params,
    )


def _validate(endog: npt.ArrayLike, exog: npt.ArrayLike | None) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    y = np.asarray(endog, dtype=np.float64)
    if y.ndim != 1:
        raise DimensionError(f"endog must be one-dimensional; got shape {y.shape}.")
    if not np.all(np.isfinite(y)):
        raise NumericalError("endog contains non-finite values.")
    x: npt.NDArray[np.float64] | None = None
    if exog is not None:
        x = np.asarray(exog, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise DimensionError(
                f"exog must be (nobs, k) aligned with endog ({y.shape[0]}); got {x.shape}."
            )
        if not np.all(np.isfinite(x)):
            raise NumericalError("exog contains non-finite values.")
    return y, x


class ARMA:
    """ARMA(p, q) specification.

    Args:
        endog: Univariate series (1-D array-like).
        order: ``(p, q)``.
        trend: ``"n"``, ``"c"`` (default), or ``"ct"``.
        exog: Optional exogenous regressors of shape ``(nobs, k)``.
    """

    __slots__ = ("_endog", "_exog", "_p", "_q", "_trend")

    def __init__(self, endog: npt.ArrayLike, order: tuple[int, int], trend: Trend = "c", exog: npt.ArrayLike | None = None) -> None:
        self._endog, self._exog = _validate(endog, exog)
        p, q = order
        if p < 0 or q < 0:
            raise SpecificationError(f"order (p, q) must be non-negative; got {order}.")
        if trend not in ("n", "c", "ct"):
            raise SpecificationError(f"trend must be 'n', 'c', 'ct'; got {trend!r}.")
        self._p, self._q, self._trend = int(p), int(q), trend

    def fit(self) -> ARMAResult:
        """Estimate by exact maximum likelihood."""
        return _fit_sarimax(self._endog, self._exog, (self._p, 0, self._q), (0, 0, 0, 0), self._trend)


class ARIMA:
    """ARIMA(p, d, q) specification (ARIMAX when ``exog`` is supplied).

    Estimation uses simple differencing: the series is differenced ``d`` times
    and a stationary ARMA(p, q) is fit exactly to the result.
    """

    __slots__ = ("_endog", "_exog", "_order", "_trend")

    def __init__(self, endog: npt.ArrayLike, order: tuple[int, int, int], trend: Trend = "c", exog: npt.ArrayLike | None = None) -> None:
        self._endog, self._exog = _validate(endog, exog)
        p, d, q = order
        if min(p, d, q) < 0:
            raise SpecificationError(f"order (p, d, q) must be non-negative; got {order}.")
        if trend not in ("n", "c", "ct"):
            raise SpecificationError(f"trend must be 'n', 'c', 'ct'; got {trend!r}.")
        self._order = (int(p), int(d), int(q))
        self._trend = trend

    def fit(self) -> ARMAResult:
        """Estimate by exact maximum likelihood (simple differencing)."""
        return _fit_sarimax(self._endog, self._exog, self._order, (0, 0, 0, 0), self._trend)


class SARIMA:
    """SARIMA(p, d, q)(P, D, Q)_s specification (SARIMAX when ``exog`` is supplied)."""

    __slots__ = ("_endog", "_exog", "_order", "_seasonal", "_trend")

    def __init__(
        self,
        endog: npt.ArrayLike,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
        trend: Trend = "c",
        exog: npt.ArrayLike | None = None,
    ) -> None:
        self._endog, self._exog = _validate(endog, exog)
        if min(order) < 0:
            raise SpecificationError(f"order must be non-negative; got {order}.")
        cap_p, cap_d, cap_q, s = seasonal_order
        if min(cap_p, cap_d, cap_q) < 0:
            raise SpecificationError(f"seasonal order must be non-negative; got {seasonal_order}.")
        if (cap_p or cap_d or cap_q) and s < 2:
            raise SpecificationError(f"seasonal period s must be >= 2 when seasonal terms are present; got s={s}.")
        if trend not in ("n", "c", "ct"):
            raise SpecificationError(f"trend must be 'n', 'c', 'ct'; got {trend!r}.")
        self._order = (int(order[0]), int(order[1]), int(order[2]))
        self._seasonal = (int(cap_p), int(cap_d), int(cap_q), int(s))
        self._trend = trend

    def fit(self) -> ARMAResult:
        """Estimate by exact maximum likelihood (simple differencing)."""
        return _fit_sarimax(self._endog, self._exog, self._order, self._seasonal, self._trend)