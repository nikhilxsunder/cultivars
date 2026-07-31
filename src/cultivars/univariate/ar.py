# filepath: /src/cultivars/univariate/ar.py
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
"""Univariate autoregressive model AR(p).

Implements ``y_t = c + delta*t + phi_1 y_{t-1} + ... + phi_p y_{t-p} + eps_t``,
``eps_t ~ N(0, sigma2)``, following cultivars' three-object discipline:

- :class:`AR` is the immutable specification (data, order, trend). It validates
  at construction and exposes ``fit``.
- Estimation is dispatched to two closed-form / optimizer routines. ``"css"``
  (conditional sum of squares) is the fast ordinary-least-squares path;
  ``"exact"`` maximizes the exact Gaussian likelihood through the linear-Gaussian
  state-space substrate (the AR(p) -> companion embedding), keeping the parameter
  search in the stationary region via a partial-autocorrelation reparameterization.
- :class:`ARResult` is the frozen, serializable result consumed by downstream
  forecasting and diagnostics.

The two estimators differ in finite samples: CSS conditions on the first ``p``
observations, exact ML models their (stationary) density too; they converge as
the sample grows.

References:
    Hamilton, J. D. (1994). *Time Series Analysis*, ch. 5 & 13.
    Monahan, J. F. (1984). A note on enforcing stationarity in ARMA models.
    *Biometrika*, 71(2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from .._core.companion import companion_matrix
from .._core.stability import StabilityResult, assess_stability
from ..exceptions import DimensionError, NumericalError, SpecificationError
from ..state_space.linear_gaussian import LinearGaussianStateSpace


# --------------------------------------------------------------------------
# Internal fit container
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _ARFit:
    const: float | None
    trend_coeff: float | None
    ar_params: npt.NDArray[np.float64]
    sigma2: float
    llf: float
    nobs: int
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]


def _n_deterministic(trend: Trend) -> int:
    if trend == "n":
        return 0
    if trend == "c":
        return 1
    return 2


def _design(
    y: npt.NDArray[np.float64], order: int, trend: Trend
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Return (target, regressors, effective_nobs) for the CSS regression."""
    n = y.shape[0]
    eff = n - order
    target = y[order:]
    lag_cols = [y[order - i : n - i] for i in range(1, order + 1)]
    det_cols: list[npt.NDArray[np.float64]] = []
    if trend in ("c", "ct"):
        det_cols.append(np.ones(eff, dtype=np.float64))
    if trend == "ct":
        det_cols.append(np.arange(order + 1, n + 1, dtype=np.float64))
    columns = det_cols + lag_cols
    return target, np.column_stack(columns), eff


def _fit_css(y: npt.NDArray[np.float64], order: int, trend: Trend) -> _ARFit:
    target, regressors, eff = _design(y, order, trend)
    beta, _res, _rank, _sv = np.linalg.lstsq(regressors, target, rcond=None)
    fitted = regressors @ beta
    resid = target - fitted
    ssr = float(resid @ resid)
    sigma2 = ssr / eff
    n_det = _n_deterministic(trend)
    const = float(beta[0]) if trend in ("c", "ct") else None
    trend_coeff = float(beta[1]) if trend == "ct" else None
    ar_params = np.asarray(beta[n_det:], dtype=np.float64)
    llf = -0.5 * eff * (_LOG_2PI + np.log(sigma2) + 1.0)
    return _ARFit(const, trend_coeff, ar_params, sigma2, llf, eff, resid, fitted)


def _fit_exact(y: npt.NDArray[np.float64], order: int, trend: Trend) -> _ARFit:
    if trend == "ct":
        raise SpecificationError(
            "exact ML with trend='ct' is not supported in this release; "
            "use method='css' for a linear trend."
        )
    has_const = trend == "c"
    p = order
    e1 = np.zeros(p, dtype=np.float64)
    e1[0] = 1.0
    design = np.zeros((1, p), dtype=np.float64)
    design[0, 0] = 1.0
    obs_cov = np.zeros((1, 1), dtype=np.float64)
    selection = e1.reshape(p, 1)
    identity = np.eye(p, dtype=np.float64)

    warm = _fit_css(y, order, trend)
    phi0 = warm.ar_params
    if not assess_stability(phi0).is_stable:
        phi0 = np.zeros(p, dtype=np.float64)
    pacf0 = np.clip(_ar_to_pacf(phi0), -0.999, 0.999)
    psi0 = np.arctanh(pacf0)
    log_sigma0 = np.log(warm.sigma2)
    if has_const:
        theta0 = np.concatenate([[warm.const or 0.0], psi0, [log_sigma0]])
    else:
        theta0 = np.concatenate([psi0, [log_sigma0]])

    def unpack(theta: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64], float]:
        if has_const:
            const = float(theta[0])
            psi = theta[1 : 1 + p]
            log_sigma2 = float(theta[1 + p])
        else:
            const = 0.0
            psi = theta[:p]
            log_sigma2 = float(theta[p])
        phi = _pacf_to_ar(np.tanh(psi))
        return const, phi, log_sigma2

    def build(
        const: float, phi: npt.NDArray[np.float64], sigma2: float
    ) -> LinearGaussianStateSpace:
        transition = companion_matrix(phi)
        state_cov = np.array([[sigma2]], dtype=np.float64)
        state_intercept = const * e1
        initial_state = (
            np.linalg.solve(identity - transition, state_intercept)
            if has_const
            else np.zeros(p, dtype=np.float64)
        )
        return LinearGaussianStateSpace(
            design, obs_cov, transition, selection, state_cov,
            state_intercept=state_intercept, initial_state=initial_state,
        )

    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        const, phi, log_sigma2 = unpack(theta)
        try:
            model = build(const, phi, float(np.exp(log_sigma2)))
            return -model.loglikelihood(y)
        except (NumericalError, np.linalg.LinAlgError):
            return 1e10

    result = minimize(negloglik, theta0, method="L-BFGS-B")
    const, phi, log_sigma2 = unpack(np.asarray(result.x, dtype=np.float64))
    sigma2 = float(np.exp(log_sigma2))
    model = build(const, phi, sigma2)
    filtered = model.filter(y)
    fitted = filtered.predicted_state[:, 0].copy()
    resid = y - fitted
    const_out = const if has_const else None
    return _ARFit(
        const_out, None, phi, sigma2, -float(result.fun), y.shape[0], resid, fitted
    )


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ARResult:
    """Fitted AR(p) model.

    Attributes:
        order: The autoregressive order ``p``.
        trend: The deterministic specification (``"n"``, ``"c"``, or ``"ct"``).
        method: The estimator used (``"css"`` or ``"exact"``).
        const: The intercept ``c`` (``None`` when ``trend == "n"``).
        trend_coeff: The linear-trend coefficient (only for ``trend == "ct"``).
        ar_params: The AR coefficients ``(phi_1, ..., phi_p)``.
        sigma2: The innovation variance.
        llf: The maximized log-likelihood (conditional for CSS, exact otherwise).
        nobs: The number of observations used by the estimator.
        resid: One-step residuals.
        fittedvalues: One-step fitted values.
        schema_version: Serialization schema version.
    """

    order: int
    trend: Trend
    method: Method
    const: float | None
    trend_coeff: float | None
    ar_params: npt.NDArray[np.float64]
    sigma2: float
    llf: float
    nobs: int
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    endog: npt.NDArray[np.float64] = field(repr=False)
    schema_version: int = _SCHEMA_VERSION

    @property
    def stability(self) -> StabilityResult:
        """Stationarity assessment of the fitted AR polynomial."""
        return assess_stability(self.ar_params)

    @property
    def is_stationary(self) -> bool:
        """Whether the fitted model is stationary."""
        return self.stability.is_stable

    def forecast(self, h: int) -> npt.NDArray[np.float64]:
        """Return ``h``-step-ahead point forecasts.

        Args:
            h: Forecast horizon; a positive integer.

        Returns:
            Point forecasts of shape ``(h,)``.

        Raises:
            DimensionError: If ``h`` is not positive.
        """
        if h < 1:
            raise DimensionError(f"forecast horizon h must be >= 1; got {h}.")
        p = self.order
        history = list(self.endog[-p:])
        n = self.endog.shape[0]
        out = np.empty(h, dtype=np.float64)
        for k in range(h):
            deterministic = 0.0
            if self.const is not None:
                deterministic += self.const
            if self.trend_coeff is not None:
                deterministic += self.trend_coeff * (n + k + 1)
            value = deterministic + sum(
                self.ar_params[i] * history[-1 - i] for i in range(p)
            )
            out[k] = value
            history.append(value)
        return out


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------

class AR:
    """Autoregressive model specification AR(p).

    Args:
        endog: The observed univariate series (1-D array-like).
        order: The autoregressive order ``p`` (a positive integer).
        trend: Deterministic terms — ``"n"`` (none), ``"c"`` (constant),
            or ``"ct"`` (constant + linear trend). Defaults to ``"c"``.

    Raises:
        SpecificationError: If ``order`` is not a positive integer, ``trend`` is
            invalid, or the series is too short for the requested order.
        DimensionError: If ``endog`` is not one-dimensional.
        NumericalError: If ``endog`` contains non-finite values.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(500)
        >>> for t in range(1, 500):
        ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal()
        >>> res = AR(y, order=1, trend="n").fit()
        >>> bool(0.5 < res.ar_params[0] < 0.7)
        True
    """

    __slots__ = ("_endog", "_order", "_trend")

    def __init__(self, endog: npt.ArrayLike, order: int, trend: Trend = "c") -> None:
        """Initialize the AR model specification."""
        if not isinstance(order, (int, np.integer)) or order < 1:
            raise SpecificationError(f"order must be a positive integer; got {order!r}.")
        if trend not in ("n", "c", "ct"):
            raise SpecificationError(
                f"trend must be one of 'n', 'c', 'ct'; got {trend!r}."
            )
        arr = np.asarray(endog, dtype=np.float64)
        if arr.ndim != 1:
            raise DimensionError(f"endog must be one-dimensional; got shape {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise NumericalError("endog contains non-finite values.")
        min_len = order + _n_deterministic(trend) + 1
        if arr.shape[0] < min_len:
            raise SpecificationError(
                f"series of length {arr.shape[0]} is too short for AR({order}) "
                f"with trend={trend!r} (need at least {min_len})."
            )
        self._endog = arr
        self._order = int(order)
        self._trend = trend

    @property
    def order(self) -> int:
        """The autoregressive order ``p``."""
        return self._order

    @property
    def trend(self) -> Trend:
        """The deterministic specification."""
        return self._trend

    def fit(self, method: Method = "css") -> ARResult:
        """Estimate the model.

        Args:
            method: ``"css"`` for conditional-sum-of-squares (OLS, the fast
                default) or ``"exact"`` for exact maximum likelihood via the
                state-space substrate.

        Returns:
            The fitted :class:`ARResult`.

        Raises:
            SpecificationError: If ``method`` is unknown, or ``method="exact"``
                is combined with ``trend="ct"``.
        """
        if method == "css":
            fit = _fit_css(self._endog, self._order, self._trend)
        elif method == "exact":
            fit = _fit_exact(self._endog, self._order, self._trend)
        else:
            raise SpecificationError(
                f"method must be 'css' or 'exact'; got {method!r}."
            )
        return ARResult(
            order=self._order,
            trend=self._trend,
            method=method,
            const=fit.const,
            trend_coeff=fit.trend_coeff,
            ar_params=fit.ar_params,
            sigma2=fit.sigma2,
            llf=fit.llf,
            nobs=fit.nobs,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            endog=self._endog,
        )
