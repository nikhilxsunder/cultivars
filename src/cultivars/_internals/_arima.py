# filepath: /src/cultivars/_internals/_arima.py
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

"""Seasonal ARIMA estimation engine and the shared ARMA/ARIMA/SARIMA base.

ARMA, ARIMA and SARIMA are one estimator under three parameterizations, so
they share both the specification surface (:class:`_SARIMAXModel`) and the
engine (:func:`_fit_sarimax`). The public leaves supply only their defaults.

Estimation is exact maximum likelihood in the Harvey state-space form. AR
blocks are optimized in the Monahan reparameterization so stationarity holds
by construction; MA blocks reuse the same map with a sign flip, which enforces
invertibility the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from .._core._design import deterministic_columns, expand_ar, expand_ma
from .._core._reparam import pack_stationary, unpack_stationary
from .._core._stability import assess_stability
from .._core._transforms import difference, seasonal_difference
from .._core._validators import validate_choice, validate_exog, validate_order_tuple
from ..exceptions import NumericalError, SpecificationError
from ._linear_gaussian import LinearGaussianStateSpace
from ._models import _UnivariateModel

_TRENDS = ("n", "c", "ct")


@dataclass(frozen=True, slots=True)
class _SARIMAXFit:
    """Raw outputs of a seasonal ARIMA-with-regressors fit."""

    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    seasonal_ar_params: npt.NDArray[np.float64]
    seasonal_ma_params: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    sigma2: float
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


def _arma_state_space(
    phi_star: npt.NDArray[np.float64],
    theta_star: npt.NDArray[np.float64],
    sigma2: float,
    obs_intercept: npt.NDArray[np.float64],
) -> LinearGaussianStateSpace:
    """Build the Harvey state-space form of an ARMA process.

    The state dimension is ``max(p, q + 1)``, which is the minimal realization:
    a larger companion would be observationally equivalent but would make the
    Kalman recursion carry redundant states.

    Args:
        phi_star: Expanded AR coefficients.
        theta_star: Expanded MA coefficients.
        sigma2: Innovation variance.
        obs_intercept: Per-observation mean shift from the regression block.

    Returns:
        The configured :class:`LinearGaussianStateSpace`.
    """
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


def _difference_series(
    y: npt.NDArray[np.float64], d: int, capital_d: int, s: int
) -> npt.NDArray[np.float64]:
    """Apply non-seasonal then seasonal differencing.

    Args:
        y: The series.
        d: Non-seasonal differencing order.
        capital_d: Seasonal differencing order.
        s: Seasonal period.

    Returns:
        The differenced series, shorter by ``d + s * capital_d``.
    """
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
    trend: str,
) -> _SARIMAXFit:
    """Fit a seasonal ARIMA with optional regressors by exact ML.

    Starting values come from a regression of the differenced series on the
    deterministic and exogenous block, then a short AR fit to those residuals;
    an explosive AR start is replaced by zeros.

    Args:
        endog: The endogenous series.
        exog: Optional exogenous regressors, differenced alongside ``endog``.
        order: Non-seasonal ``(p, d, q)``.
        seasonal_order: Seasonal ``(P, D, Q, s)``.
        trend: Deterministic specification.

    Returns:
        The packed :class:`_SARIMAXFit` on the differenced modeling series.
    """
    p, d, q = order
    cap_p, cap_d, cap_q, s = seasonal_order
    w = _difference_series(endog, d, cap_d, s)
    n_eff = w.shape[0]
    det = deterministic_columns(trend, n_eff)
    if exog is not None:
        exog_w = _difference_series(exog, d, cap_d, s) if (d or cap_d) else exog
        design_x = np.column_stack([det, exog_w]) if det.shape[1] else exog_w
    else:
        design_x = det
    k_beta = design_x.shape[1]

    if k_beta:
        beta0, _r, _rk, _sv = np.linalg.lstsq(design_x, w, rcond=None)
        resid0 = w - design_x @ beta0
    else:
        beta0 = np.zeros(0)
        resid0 = w - w.mean()
    sigma2_0 = max(float(resid0 @ resid0) / n_eff, 1e-8)
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

    psi = np.concatenate(
        [
            beta0,
            pack_stationary(ar0),
            np.zeros(cap_p),
            np.zeros(q),
            np.zeros(cap_q),
            [np.log(sigma2_0)],
        ]
    )
    idx = np.cumsum([k_beta, p, cap_p, q, cap_q])

    def unpack(
        vec: npt.NDArray[np.float64],
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        beta = vec[: idx[0]]
        phi = unpack_stationary(vec[idx[0] : idx[1]]) if p else np.zeros(0)
        sphi = unpack_stationary(vec[idx[1] : idx[2]]) if cap_p else np.zeros(0)
        theta_c = -unpack_stationary(vec[idx[2] : idx[3]]) if q else np.zeros(0)
        stheta = -unpack_stationary(vec[idx[3] : idx[4]]) if cap_q else np.zeros(0)
        return beta, phi, sphi, theta_c, stheta, float(np.exp(vec[idx[4]]))

    def obs_intercept(beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return design_x @ beta if k_beta else np.zeros(n_eff)

    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        beta, phi, sphi, theta_c, stheta, sigma2 = unpack(theta)
        try:
            ss = _arma_state_space(
                expand_ar(phi, sphi, s),
                expand_ma(theta_c, stheta, s),
                sigma2,
                obs_intercept(beta),
            )
            return -ss.loglikelihood(w)
        except (NumericalError, np.linalg.LinAlgError):
            return 1e10

    result = minimize(negloglik, psi, method="L-BFGS-B")
    beta, phi, sphi, theta_c, stheta, sigma2 = unpack(np.asarray(result.x, dtype=np.float64))
    ss = _arma_state_space(
        expand_ar(phi, sphi, s), expand_ma(theta_c, stheta, s), sigma2, obs_intercept(beta)
    )
    fitted = ss.filter(w).predicted_state[:, 0] + obs_intercept(beta)
    resid = w - fitted
    return _SARIMAXFit(
        ar_params=phi,
        ma_params=theta_c,
        seasonal_ar_params=sphi,
        seasonal_ma_params=stheta,
        beta=beta,
        sigma2=sigma2,
        resid=resid,
        fittedvalues=fitted,
        llf=-float(result.fun),
        nobs=n_eff,
        n_params=k_beta + p + cap_p + q + cap_q + 1,
    )


class _SARIMAXModel[R](_UnivariateModel[R]):
    """Shared specification surface for the ARMA/ARIMA/SARIMA family.

    Args:
        endog: The endogenous series.
        order: Non-seasonal ``(p, d, q)``.
        seasonal_order: Seasonal ``(P, D, Q, s)``; defaults to no seasonal block.
        trend: Deterministic specification.
        exog: Optional exogenous regressors.

    Raises:
        SpecificationError: If any order is negative, or seasonal terms are
            requested with a period below 2.
        DimensionError: If the series is too short, or ``exog`` is misshapen.
    """

    __slots__ = (
        "_exog",
        "_order",
        "_seasonal",
        "_trend",
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str = "c",
        exog: npt.ArrayLike | None = None,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog)
        p, d, q = validate_order_tuple(order, ("p", "d", "q"))
        cap_p, cap_d, cap_q, s = validate_order_tuple(seasonal_order, ("P", "D", "Q", "s"))
        if (cap_p or cap_d or cap_q) and s < 2:
            raise SpecificationError(
                f"seasonal period s must be >= 2 when seasonal terms are present; got s={s}."
            )
        self._order = (p, d, q)
        self._seasonal = (cap_p, cap_d, cap_q, s)
        self._trend = validate_choice(trend, _TRENDS, "trend")
        self._exog = validate_exog(exog, self.endog.shape[0])
        self._ensure_length(
            p + d + q + s * (cap_p + cap_d + cap_q) + 2,
            f"SARIMA{self._order}{self._seasonal}",
        )

    @property
    def order(self) -> tuple[int, int, int]:
        """The non-seasonal order ``(p, d, q)``."""
        return self._order

    @property
    def seasonal_order(self) -> tuple[int, int, int, int]:
        """The seasonal order ``(P, D, Q, s)``."""
        return self._seasonal

    @property
    def trend(self) -> str:
        """The deterministic specification."""
        return self._trend

    @property
    def exog(self) -> npt.NDArray[np.float64] | None:
        """The validated exogenous regressors, or ``None``."""
        return self._exog

    def _fit_family(self) -> _SARIMAXFit:
        """Run the shared state-space engine for this specification."""
        return _fit_sarimax(self.endog, self._exog, self._order, self._seasonal, self._trend)
