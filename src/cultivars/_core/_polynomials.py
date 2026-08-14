# filepath: /src/cultivars/_core/_polynomials.py
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

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..exceptions import SpecificationError


def expand_ar(
    ar: npt.NDArray[np.float64], seasonal_ar: npt.NDArray[np.float64], period: int
) -> npt.NDArray[np.float64]:
    """Expand the product ``phi(L) * Phi(L**s)`` into a single AR coefficient vector.

    A seasonal model's stationarity is governed by the composed polynomial,
    not by either factor alone, so this is what the stationarity check must
    be applied to.

    Args:
        ar: Non-seasonal coefficients ``phi_1, ..., phi_p``.
        seasonal_ar: Seasonal coefficients ``Phi_1, ..., Phi_P``.
        period: The seasonal period ``s``.

    Returns:
        The expanded coefficients of length ``p + s * P``.

    Raises:
        SpecificationError: If seasonal terms are present but ``period < 1``.

    Example:
        >>> np.round(expand_ar(np.array([0.5]), np.array([0.4]), 4), 3)
        array([ 0.5, -0. , -0. ,  0.4, -0.2])
    """
    poly = np.concatenate(([1.0], -np.asarray(ar, dtype=np.float64)))
    if seasonal_ar.size:
        if period < 1:
            raise SpecificationError(
                f"seasonal period must be >= 1 to expand a seasonal polynomial; got {period}."
            )
        seasonal = np.zeros(period * seasonal_ar.size + 1, dtype=np.float64)
        seasonal[0] = 1.0
        for i, value in enumerate(seasonal_ar, start=1):
            seasonal[period * i] = -value
        poly = np.convolve(poly, seasonal)
    return -poly[1:]


def expand_ma(
    ma: npt.NDArray[np.float64], seasonal_ma: npt.NDArray[np.float64], period: int
) -> npt.NDArray[np.float64]:
    """Expand the product ``theta(L) * Theta(L**s)`` into a single MA coefficient vector.

    Args:
        ma: Non-seasonal coefficients ``theta_1, ..., theta_q``.
        seasonal_ma: Seasonal coefficients ``Theta_1, ..., Theta_Q``.
        period: The seasonal period ``s``.

    Returns:
        The expanded coefficients of length ``q + s * Q``.

    Raises:
        SpecificationError: If seasonal terms are present but ``period < 1``.
    """
    poly = np.concatenate(([1.0], np.asarray(ma, dtype=np.float64)))
    if seasonal_ma.size:
        if period < 1:
            raise SpecificationError(
                f"seasonal period must be >= 1 to expand a seasonal polynomial; got {period}."
            )
        seasonal = np.zeros(period * seasonal_ma.size + 1, dtype=np.float64)
        seasonal[0] = 1.0
        for i, value in enumerate(seasonal_ma, start=1):
            seasonal[period * i] = value
        poly = np.convolve(poly, seasonal)
    return poly[1:]


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


def ar_infinity_weights(
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


def ar_recursion(
    history: npt.NDArray[np.float64],
    ar_params: npt.NDArray[np.float64],
    const: float,
    trend_coeff: float,
    origin: int,
    h: int,
) -> npt.NDArray[np.float64]:
    """Recursive point forecast for an AR(p) mean model.

    Args:
        history: The observed series, most recent value last.
        ar_params: Autoregressive coefficients.
        const: Intercept, ``0.0`` when absent.
        trend_coeff: Linear-trend slope, ``0.0`` when absent.
        origin: Time index of the last observation, so the deterministic trend
            continues from the right point rather than restarting at 1.
        h: Forecast horizon.

    Returns:
        Point forecasts of shape ``(h,)``.
    """
    p = ar_params.shape[0]
    buf = list(history[-p:]) if p else []
    out = np.empty(h, dtype=np.float64)
    for step in range(h):
        value = const + trend_coeff * (origin + step + 1)
        for i in range(p):
            value += ar_params[i] * buf[-1 - i]
        out[step] = value
        buf.append(value)
    return out
