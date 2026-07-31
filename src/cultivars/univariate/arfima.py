# filepath: /src/cultivars/univariate/arfima.py
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

"""ARFIMA(p, d, q) — autoregressive fractionally integrated moving average.

Long memory sits between the I(0) short memory of ARMA and the I(1) unit root of
ARIMA: the fractional differencing parameter ``d in (-0.5, 0.5)`` gives an
autocorrelation function that decays hyperbolically (``rho_k ~ k^{2d-1}``) rather
than geometrically, and a spectral density that diverges like ``lambda^{-2d}`` at
the origin. Realized variance, electricity load, inflation, and volume all show
this signature.

The model is ``phi(L) (1 - L)**d (y_t - mu) = theta(L) eps_t`` with
``eps_t ~ N(0, sigma2)``. Estimation composes the pieces already in cultivars:

- Fractional differencing ``(1 - L)**d`` is a truncated binomial filter
  (:func:`fractional_difference`); applying it to the demeaned series leaves a
  stationary, invertible ARMA(p, q).
- That ARMA is scored by **exact** Gaussian maximum likelihood through the same
  Harvey state-space form :mod:`cultivars.univariate.arma` uses.

So the primary estimator (:meth:`ARFIMA.fit`) is a single joint optimization over
``(mu, d, phi, theta, sigma2)`` — exact ML in the ARMA block, conditional-sum-of
squares in ``d`` via the truncated filter — with the AR/MA blocks reparameterized
into the stationary/invertible region by the partial-autocorrelation map in
:mod:`cultivars.univariate._base`.

Two semiparametric estimators of ``d`` alone are provided as well and double as
warm starts: the Geweke-Porter-Hudak (1983) log-periodogram regression
(:func:`gph`) and the Robinson (1995) local Whittle estimator
(:func:`local_whittle`). These are the frequency-domain, model-free counterparts
that make long memory a spectral question — cultivars' intended differentiator.

For ``d >= 0.5`` (nonstationary long memory) difference the series first and model
the result here; this class targets the stationary region ``|d| < 0.5``.

References:
    Granger, C. W. J. & Joyeux, R. (1980). An introduction to long-memory time
    series models and fractional differencing. *J. Time Ser. Anal.*, 1(1).
    Hosking, J. R. M. (1981). Fractional differencing. *Biometrika*, 68(1).
    Geweke, J. & Porter-Hudak, S. (1983). The estimation and application of long
    memory time series models. *J. Time Ser. Anal.*, 4(4).
    Robinson, P. M. (1995). Gaussian semiparametric estimation of long range
    dependence. *Annals of Statistics*, 23(5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, minimize_scalar

from .._core.stability import StabilityResult, assess_stability
from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._base import (
    InformationCriteria,
    information_criteria,
    pacf_to_coeffs,
)
from .arma import _arma_state_space                   # keeps |d| strictly inside the stationary band



# --------------------------------------------------------------------------
# Semiparametric d estimators (spectral)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LongMemoryEstimate:
    """A semiparametric estimate of the memory parameter ``d``.

    Attributes:
        d: The estimated fractional differencing order.
        se: Asymptotic standard error of ``d``.
        n_freq: Number of Fourier frequencies (the bandwidth ``m``) used.
        method: ``"gph"`` or ``"local_whittle"``.
    """

    d: float
    se: float
    n_freq: int
    method: str


def _periodogram(
    y: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return positive Fourier frequencies and the periodogram (DC term dropped)."""
    n = y.shape[0]
    centered = y - y.mean()
    transform = np.fft.rfft(centered)
    periodogram = (np.abs(transform) ** 2) / n
    freqs = 2.0 * np.pi * np.arange(transform.shape[0]) / n
    return freqs[1:], periodogram[1:]


def _bandwidth(n: int, m: int | None, exponent: float) -> int:
    if m is not None:
        if m < 2:
            raise SpecificationError(f"bandwidth m must be >= 2; got {m}.")
        return int(m)
    return max(2, int(np.floor(n ** exponent)))


def _validate_series(endog: npt.ArrayLike) -> npt.NDArray[np.float64]:
    arr = np.asarray(endog, dtype=np.float64)
    if arr.ndim != 1:
        raise DimensionError(f"endog must be one-dimensional; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise NumericalError("endog contains non-finite values.")
    return arr


def gph(
    endog: npt.ArrayLike, *, m: int | None = None, bandwidth_exponent: float = 0.5
) -> LongMemoryEstimate:
    """Estimate ``d`` by the Geweke-Porter-Hudak log-periodogram regression.

    Regresses ``log I(lambda_j)`` on ``log(4 sin^2(lambda_j / 2))`` over the
    lowest ``m`` Fourier frequencies; ``d`` is minus the slope. The error
    variance is asymptotically ``pi^2 / 6``, giving the reported standard error.

    Args:
        endog: The series (1-D array-like).
        m: Number of frequencies (bandwidth). Defaults to ``floor(n**exponent)``.
        bandwidth_exponent: Exponent for the default bandwidth (typically 0.5).

    Returns:
        A :class:`LongMemoryEstimate`.

    Raises:
        SpecificationError: If ``m`` is too small.
        NumericalError: If the regression is degenerate.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(rng.standard_normal(4096))     # d = 1 process
        >>> est = gph(np.diff(y))                        # differenced -> d ~ 0
        >>> abs(est.d) < 0.2
        True
    """
    y = _validate_series(endog)
    freqs, periodogram = _periodogram(y)
    m_eff = min(_bandwidth(y.shape[0], m, bandwidth_exponent), freqs.shape[0])
    lam = freqs[:m_eff]
    power = periodogram[:m_eff]
    if np.any(power <= 0.0):
        raise NumericalError("periodogram has non-positive ordinates; cannot take logs.")
    regressor = np.log(4.0 * np.sin(lam / 2.0) ** 2)
    response = np.log(power)
    centered = regressor - regressor.mean()
    denom = float(centered @ centered)
    if denom <= 0.0:
        raise NumericalError("degenerate GPH regression (zero regressor variance).")
    slope = float(centered @ (response - response.mean()) / denom)
    d_hat = -slope
    se = float(np.sqrt((np.pi ** 2 / 6.0) / denom))
    return LongMemoryEstimate(d=d_hat, se=se, n_freq=m_eff, method="gph")


def local_whittle(
    endog: npt.ArrayLike, *, m: int | None = None, bandwidth_exponent: float = 0.65
) -> LongMemoryEstimate:
    """Estimate ``d`` by the Robinson (1995) Gaussian semiparametric estimator.

    Minimizes the local Whittle objective
    ``R(d) = log( m^{-1} sum_j lambda_j^{2d} I(lambda_j) )
    - (2d / m) sum_j log lambda_j`` over ``|d| < 0.5``. The estimator is
    ``sqrt(m)``-consistent with asymptotic variance ``1 / 4``.

    Args:
        endog: The series (1-D array-like).
        m: Number of frequencies (bandwidth). Defaults to ``floor(n**exponent)``.
        bandwidth_exponent: Exponent for the default bandwidth (typically ~0.65).

    Returns:
        A :class:`LongMemoryEstimate`.

    Example:
        >>> rng = np.random.default_rng(1)
        >>> y = np.cumsum(rng.standard_normal(4096))
        >>> est = local_whittle(np.diff(y))
        >>> abs(est.d) < 0.2
        True
    """
    y = _validate_series(endog)
    freqs, periodogram = _periodogram(y)
    m_eff = min(_bandwidth(y.shape[0], m, bandwidth_exponent), freqs.shape[0])
    lam = freqs[:m_eff]
    power = periodogram[:m_eff]
    log_lam = np.log(lam)

    def objective(d: float) -> float:
        g = np.mean(lam ** (2.0 * d) * power)
        if not np.isfinite(g) or g <= 0.0:
            return 1e10
        return float(np.log(g) - 2.0 * d * log_lam.mean())

    result = minimize_scalar(objective, bounds=(-_D_MAX, _D_MAX), method="bounded")
    d_hat = float(result.x)
    se = float(1.0 / (2.0 * np.sqrt(m_eff)))
    return LongMemoryEstimate(d=d_hat, se=se, n_freq=m_eff, method="local_whittle")


# --------------------------------------------------------------------------
# AR(infinity) weights (for forecasting)
# --------------------------------------------------------------------------

def _ar_infinity(
    d: float,
    ar_params: npt.NDArray[np.float64],
    ma_params: npt.NDArray[np.float64],
    truncation: int,
) -> npt.NDArray[np.float64]:
    """Coefficients ``c_1..c_M`` of ``Pi(L) = phi(L)(1-L)**d / theta(L)`` (monic).

    The model is ``Pi(L)(y_t - mu) = eps_t`` with ``Pi(L) = 1 - c_1 L - ...``, so
    the one-step recursion is ``(y_t - mu) = sum_j c_j (y_{t-j} - mu) + eps_t``.
    """
    frac = fractional_difference_weights(d, truncation + 1)
    phi_poly = np.concatenate([[1.0], -ar_params]) if ar_params.size else np.array([1.0])
    numerator = np.convolve(phi_poly, frac)[: truncation + 1]
    theta_poly = np.concatenate([[1.0], ma_params]) if ma_params.size else np.array([1.0])
    # Series division numerator / theta_poly (theta_poly monic).
    pi = np.zeros(truncation + 1, dtype=np.float64)
    q = ma_params.size
    for i in range(truncation + 1):
        acc = numerator[i]
        for j in range(1, min(i, q) + 1):
            acc -= theta_poly[j] * pi[i - j]
        pi[i] = acc
    return -pi[1:]                       # c_j = -pi_j for the AR(inf) recursion


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ARFIMAResult:
    """Fitted ARFIMA(p, d, q) model.

    Attributes:
        order: ``(p, q)`` — the short-memory ARMA orders.
        d: The fractional differencing parameter.
        mean: The estimated series mean ``mu`` (``None`` when ``trend == "n"``).
        ar_params: AR coefficients ``(phi_1, ..., phi_p)``.
        ma_params: MA coefficients ``(theta_1, ..., theta_q)``.
        sigma2: Innovation variance.
        llf: Maximized log-likelihood (exact in the ARMA block, conditional in d).
        nobs: Number of observations.
        resid: One-step residuals on the fractionally differenced series.
        fittedvalues: One-step fitted values on the fractionally differenced series.
        endog: The original series (kept for forecasting).
        schema_version: Serialization schema version.
    """

    order: tuple[int, int]
    d: float
    mean: float | None
    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    sigma2: float
    llf: float
    nobs: int
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    endog: npt.NDArray[np.float64] = field(repr=False)
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
    def stability(self) -> StabilityResult:
        """Stationarity of the short-memory AR polynomial (long memory aside)."""
        return assess_stability(self.ar_params)

    @property
    def is_stationary(self) -> bool:
        """Whether the model is stationary: ``|d| < 0.5`` and AR roots outside."""
        return abs(self.d) < 0.5 and self.stability.is_stable

    @property
    def is_long_memory(self) -> bool:
        """Whether the process has positive long memory (``d > 0``)."""
        return self.d > 0.0

    def forecast(self, h: int, *, truncation: int = 500) -> npt.NDArray[np.float64]:
        """Return ``h``-step-ahead point forecasts via the AR(infinity) form.

        Args:
            h: Forecast horizon; a positive integer.
            truncation: Number of AR(infinity) weights retained. Long-memory
                weights decay slowly (``~ j**(-d-1)``), so larger values improve
                accuracy at long horizons.

        Returns:
            Point forecasts of shape ``(h,)``.

        Raises:
            DimensionError: If ``h`` is not positive.
            SpecificationError: If ``truncation < 1``.
        """
        if h < 1:
            raise DimensionError(f"forecast horizon h must be >= 1; got {h}.")
        if truncation < 1:
            raise SpecificationError(f"truncation must be >= 1; got {truncation}.")
        mu = self.mean if self.mean is not None else 0.0
        weights = _ar_infinity(self.d, self.ar_params, self.ma_params, truncation)
        history = list(self.endog - mu)
        out = np.empty(h, dtype=np.float64)
        for k in range(h):
            reach = min(len(weights), len(history))
            value = float(sum(weights[j] * history[-1 - j] for j in range(reach)))
            out[k] = value + mu
            history.append(value)
        return out


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

def _fit_arfima(
    y: npt.NDArray[np.float64], p: int, q: int, estimate_mean: bool, truncation: int
) -> ARFIMAResult:
    n = y.shape[0]

    # Warm start: local Whittle for d, sample mean for mu.
    try:
        d0 = float(np.clip(local_whittle(y).d, -_D_MAX + 1e-3, _D_MAX - 1e-3))
    except (NumericalError, SpecificationError):
        d0 = 0.0
    mu0 = float(y.mean()) if estimate_mean else 0.0
    w0 = fractional_difference(y - mu0, d0, truncation=truncation)
    log_sigma0 = float(np.log(max(float(np.var(w0)), 1e-8)))

    raw_d0 = float(np.arctanh(d0 / _D_MAX))
    parts: list[npt.NDArray[np.float64]] = []
    if estimate_mean:
        parts.append(np.array([mu0]))
    parts.extend([np.array([raw_d0]), np.zeros(p), np.zeros(q), np.array([log_sigma0])])
    theta0 = np.concatenate(parts)

    offset = 1 if estimate_mean else 0
    i_d = offset
    i_ar = i_d + 1
    i_ma = i_ar + p
    i_sig = i_ma + q

    def unpack(
        theta: npt.NDArray[np.float64],
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        mu = float(theta[0]) if estimate_mean else 0.0
        d = _D_MAX * float(np.tanh(theta[i_d]))
        phi = pacf_to_coeffs(np.tanh(theta[i_ar:i_ma])) if p else np.zeros(0)
        theta_c = -pacf_to_coeffs(np.tanh(theta[i_ma:i_sig])) if q else np.zeros(0)
        sigma2 = float(np.exp(theta[i_sig]))
        return mu, d, phi, theta_c, sigma2

    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        mu, d, phi, theta_c, sigma2 = unpack(theta)
        try:
            w = fractional_difference(y - mu, d, truncation=truncation)
            ss = _arma_state_space(phi, theta_c, sigma2, np.zeros(n))
            return -ss.loglikelihood(w)
        except (NumericalError, np.linalg.LinAlgError, ValueError):
            return 1e10

    result = minimize(negloglik, theta0, method="L-BFGS-B")
    mu, d, phi, theta_c, sigma2 = unpack(np.asarray(result.x, dtype=np.float64))
    w = fractional_difference(y - mu, d, truncation=truncation)
    ss = _arma_state_space(phi, theta_c, sigma2, np.zeros(n))
    filtered = ss.filter(w)
    fitted = filtered.predicted_state[:, 0]
    resid = w - fitted
    n_params = offset + 1 + p + q + 1
    return ARFIMAResult(
        order=(p, q),
        d=d,
        mean=mu if estimate_mean else None,
        ar_params=phi,
        ma_params=theta_c,
        sigma2=sigma2,
        llf=-float(result.fun),
        nobs=n,
        resid=resid,
        fittedvalues=fitted,
        endog=y,
        _n_params=n_params,
    )


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------

class ARFIMA:
    """ARFIMA(p, d, q) specification with jointly estimated fractional ``d``.

    Args:
        endog: Univariate series (1-D array-like).
        order: ``(p, q)`` — the AR and MA orders of the short-memory component.
        trend: ``"c"`` to estimate a mean ``mu`` (default) or ``"n"`` for a
            zero-mean series.
        truncation: Length of the fractional-difference filter. Defaults to the
            series length; smaller values trade accuracy for speed on long series.

    Raises:
        SpecificationError: If ``order`` is invalid, ``trend`` is unknown, or the
            series is too short.
        DimensionError: If ``endog`` is not one-dimensional.
        NumericalError: If ``endog`` contains non-finite values.

    Example:
        >>> import numpy as np
        >>> from cultivars.univariate.arfima import ARFIMA, fractional_difference
        >>> rng = np.random.default_rng(0)
        >>> # ARFIMA(0, 0.3, 0): fractionally integrate white noise.
        >>> e = rng.standard_normal(4000)
        >>> y = fractional_difference(e, -0.3)        # (1-L)^{-0.3} e_t
        >>> res = ARFIMA(y, order=(0, 0), trend="n").fit()
        >>> bool(0.2 < res.d < 0.4)
        True
    """

    __slots__ = ("_y", "_p", "_q", "_trend", "_truncation")

    def __init__(
        self,
        endog: npt.ArrayLike,
        order: tuple[int, int],
        trend: str = "c",
        *,
        truncation: int | None = None,
    ) -> None:
        """Initialize the ARFIMA specification."""
        self._y = _validate_series(endog)
        p, q = order
        if not isinstance(p, (int, np.integer)) or not isinstance(q, (int, np.integer)):
            raise SpecificationError(f"order (p, q) must be integers; got {order!r}.")
        if p < 0 or q < 0:
            raise SpecificationError(f"order (p, q) must be non-negative; got {order}.")
        if trend not in ("n", "c"):
            raise SpecificationError(f"trend must be 'n' or 'c'; got {trend!r}.")
        min_len = 2 * (p + q) + 8
        if self._y.shape[0] < min_len:
            raise SpecificationError(
                f"series of length {self._y.shape[0]} is too short for "
                f"ARFIMA({p}, d, {q}) (need at least {min_len})."
            )
        if truncation is not None and truncation < 1:
            raise SpecificationError(f"truncation must be >= 1; got {truncation}.")
        self._p, self._q, self._trend = int(p), int(q), trend
        self._truncation = int(truncation) if truncation is not None else self._y.shape[0]

    @property
    def order(self) -> tuple[int, int]:
        """The short-memory ARMA orders ``(p, q)``."""
        return (self._p, self._q)

    def fit(self) -> ARFIMAResult:
        """Estimate ``(mu, d, phi, theta, sigma2)`` by joint maximum likelihood.

        Returns:
            The fitted :class:`ARFIMAResult`.
        """
        return _fit_arfima(
            self._y, self._p, self._q, self._trend == "c", self._truncation
        )
