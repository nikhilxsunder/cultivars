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

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from ..exceptions import NumericalError, SpecificationError
from ._base import (
    pacf_to_coeffs,
)
from .arma import _arma_state_space  # keeps |d| strictly inside the stationary band

# --------------------------------------------------------------------------
# AR(infinity) weights (for forecasting)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, slots=True)
class ARFIMAResult(_MeanResult, _StationarityMixin):
    order: tuple[int, int]
    d: float
    mean: float | None
    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    sigma2: float


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

    __slots__ = ("_p", "_q", "_trend", "_truncation", "_y")

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
        return _fit_arfima(self._y, self._p, self._q, self._trend == "c", self._truncation)
