# filepath: /src/cultivars/_internals/_arfima.py
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

"""Fractional differencing and the ARFIMA estimation engine.

ARFIMA separates long memory from short memory: the fractional operator
``(1 - L)**d`` captures hyperbolic autocorrelation decay, and a conventional
ARMA block captures the rest. Estimation is joint -- ``d`` and the ARMA
parameters are optimized together, warm-started from a local Whittle estimate
of ``d`` -- so the short-memory block does not absorb long-memory dependence
that a two-step procedure would leave to it.

``d`` is optimized through ``_D_MAX * tanh(.)``, keeping it strictly inside
the stationary and invertible region ``(-0.5, 0.5)`` at every iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from .._core._defaults import _D_MAX
from .._core._reparam import unpack_stationary
from .._core._spectral import local_whittle_d
from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._arima import _arma_state_space


@dataclass(frozen=True, slots=True)
class _ARFIMAFit:
    """Raw outputs of a fractionally integrated ARMA fit."""

    d: float
    mean: float | None
    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    sigma2: float
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


def fractional_difference_weights(d: float, length: int) -> npt.NDArray[np.float64]:
    """First ``length`` coefficients of the operator ``(1 - L)**d``.

    The coefficients satisfy ``b_0 = 1`` and ``b_k = b_{k-1} * (k - 1 - d) / k``.
    For ``d > 0`` they are negative for ``k >= 1`` and decay like ``k**(-d-1)``,
    which is the slow decay that encodes long memory.

    Args:
        d: The fractional differencing order.
        length: Number of coefficients to return.

    Returns:
        The coefficients ``[b_0, ..., b_{length-1}]``.

    Raises:
        SpecificationError: If ``length < 1``.

    Example:
        >>> np.round(fractional_difference_weights(0.5, 4), 4)
        array([ 1.    , -0.5   , -0.125 , -0.0625])
    """
    if length < 1:
        raise SpecificationError(f"length must be >= 1; got {length}.")
    weights = np.empty(length, dtype=np.float64)
    weights[0] = 1.0
    for k in range(1, length):
        weights[k] = weights[k - 1] * (k - 1 - d) / k
    return weights


def fractional_difference(
    y: npt.ArrayLike, d: float, *, truncation: int | None = None
) -> npt.NDArray[np.float64]:
    """Apply the truncated fractional difference ``(1 - L)**d``.

    Uses all available history at each ``t``, so the series is not shortened.
    At the sample start the filter is necessarily truncated, inducing a
    transient conditioned on the same way CSS conditions on its first
    observations.

    Args:
        y: Input series (1-D array-like).
        d: Fractional differencing order.
        truncation: Maximum filter length; defaults to the series length.

    Returns:
        The fractionally differenced series, same length as ``y``.

    Raises:
        DimensionError: If ``y`` is not one-dimensional.
        NumericalError: If ``y`` contains non-finite values.
        SpecificationError: If ``truncation < 1``.

    Example:
        >>> np.round(fractional_difference(np.array([1.0, 2.0, 3.0, 4.0]), 1.0), 4)
        array([1., 1., 1., 1.])
    """
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise DimensionError(f"y must be one-dimensional; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise NumericalError("y contains non-finite values.")
    n = arr.shape[0]
    m = n if truncation is None else int(truncation)
    if m < 1:
        raise SpecificationError(f"truncation must be >= 1; got {m}.")
    weights = fractional_difference_weights(d, min(m, n))
    return np.convolve(arr, weights)[:n]


def _ar_infinity(
    d: float,
    ar_params: npt.NDArray[np.float64],
    ma_params: npt.NDArray[np.float64],
    truncation: int,
) -> npt.NDArray[np.float64]:
    """Coefficients of ``Pi(L) = phi(L)(1 - L)**d / theta(L)``, in AR form.

    The model is ``Pi(L)(y_t - mu) = eps_t`` with ``Pi(L) = 1 - c_1 L - ...``,
    so the one-step recursion driving forecasts is
    ``(y_t - mu) = sum_j c_j (y_{t-j} - mu) + eps_t``.

    Args:
        d: Fractional differencing order.
        ar_params: Short-memory AR coefficients.
        ma_params: Short-memory MA coefficients.
        truncation: Number of coefficients to retain.

    Returns:
        The coefficients ``c_1, ..., c_truncation``.
    """
    frac = fractional_difference_weights(d, truncation + 1)
    phi_poly = np.concatenate([[1.0], -ar_params]) if ar_params.size else np.array([1.0])
    numerator = np.convolve(phi_poly, frac)[: truncation + 1]
    theta_poly = np.concatenate([[1.0], ma_params]) if ma_params.size else np.array([1.0])
    pi = np.zeros(truncation + 1, dtype=np.float64)
    q = ma_params.size
    for i in range(truncation + 1):
        acc = numerator[i]
        for j in range(1, min(i, q) + 1):
            acc -= theta_poly[j] * pi[i - j]
        pi[i] = acc
    return -pi[1:]


def _fit_arfima(
    y: npt.NDArray[np.float64], p: int, q: int, estimate_mean: bool, truncation: int
) -> _ARFIMAFit:
    """Fit an ARFIMA(p, d, q) by joint maximum likelihood.

    Warm-starts ``d`` from a local Whittle estimate, falling back to zero if
    that estimator fails; the ARMA block starts at zero, since a short-memory
    warm start fitted before ``d`` is known tends to absorb long memory.

    Args:
        y: The series.
        p: Short-memory AR order.
        q: Short-memory MA order.
        estimate_mean: Whether to estimate a mean ``mu``.
        truncation: Fractional-filter length.

    Returns:
        The packed :class:`_ARFIMAFit` on the fractionally differenced series.
    """
    n = y.shape[0]
    try:
        d0 = float(np.clip(local_whittle_d(y)[0], -_D_MAX + 1e-3, _D_MAX - 1e-3))
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
        phi = unpack_stationary(theta[i_ar:i_ma]) if p else np.zeros(0)
        theta_c = -unpack_stationary(theta[i_ma:i_sig]) if q else np.zeros(0)
        return mu, d, phi, theta_c, float(np.exp(theta[i_sig]))

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
    fitted = ss.filter(w).predicted_state[:, 0]
    return _ARFIMAFit(
        d=d,
        mean=mu if estimate_mean else None,
        ar_params=phi,
        ma_params=theta_c,
        sigma2=sigma2,
        resid=w - fitted,
        fittedvalues=fitted,
        llf=-float(result.fun),
        nobs=n,
        n_params=offset + 1 + p + q + 1,
    )
