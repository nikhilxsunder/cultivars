# filepath: /src/cultivars/_core/_design.py
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

"""Regressor-matrix construction and lag-polynomial expansion.

Every conditional estimator in the package builds the same two things: a
block of deterministic columns and a block of lagged levels. Centralizing
them here means the ``t`` index of a linear trend is defined in exactly one
place, which is the detail that most often drifts between models.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError, SpecificationError

_TREND_WIDTH: dict[str, int] = {"n": 0, "c": 1, "ct": 2}
"""Number of deterministic columns implied by each trend specification."""


def n_deterministic(trend: str) -> int:
    """Number of deterministic regressors implied by a trend specification.

    Args:
        trend: One of ``"n"``, ``"c"``, ``"ct"``.

    Returns:
        ``0``, ``1``, or ``2``.

    Raises:
        SpecificationError: If ``trend`` is not a recognized specification.

    Example:
        >>> n_deterministic("ct")
        2
    """
    try:
        return _TREND_WIDTH[trend]
    except KeyError:
        raise SpecificationError(
            f"trend must be one of {tuple(_TREND_WIDTH)}; got {trend!r}."
        ) from None


def deterministic_columns(
    trend: str, nobs: int, *, start: int = 1
) -> npt.NDArray[np.float64]:
    """Build the deterministic regressor block.

    Args:
        trend: One of ``"n"``, ``"c"``, ``"ct"``.
        nobs: Number of rows.
        start: Time index of the first row, so that a conditional sample
            beginning at observation ``p + 1`` carries the trend values it
            would have had in the full sample.

    Returns:
        An ``(nobs, n_deterministic(trend))`` array.

    Example:
        >>> deterministic_columns("ct", 3, start=2)
        array([[1., 2.],
               [1., 3.],
               [1., 4.]])
    """
    width = n_deterministic(trend)
    out = np.empty((nobs, width), dtype=np.float64)
    if width >= 1:
        out[:, 0] = 1.0
    if width == 2:
        out[:, 1] = np.arange(start, start + nobs, dtype=np.float64)
    return out


def lag_matrix(
    y: npt.NDArray[np.float64], order: int, *, start: int | None = None
) -> npt.NDArray[np.float64]:
    """Build the matrix of lagged levels ``[y_{t-1}, ..., y_{t-order}]``.

    Args:
        y: The series.
        order: Number of lags; ``0`` yields a zero-width matrix.
        start: First time index retained. Defaults to ``order``, which drops
            exactly the observations that have no complete lag history.

    Returns:
        An ``(n - start, order)`` array.

    Raises:
        DimensionError: If ``start`` is smaller than ``order``, which would
            reference lags before the beginning of the series.

    Example:
        >>> lag_matrix(np.arange(5.0), 2)
        array([[1., 0.],
               [2., 1.],
               [3., 2.]])
    """
    n = y.shape[0]
    first = order if start is None else start
    if first < order:
        raise DimensionError(f"start ({first}) must be at least order ({order}).")
    if order == 0:
        return np.zeros((n - first, 0), dtype=np.float64)
    return np.column_stack([y[first - i : n - i] for i in range(1, order + 1)])


def css_design(
    y: npt.NDArray[np.float64], order: int, trend: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Build the target and regressor matrix for a conditional least-squares fit.

    Args:
        y: The series.
        order: Autoregressive order ``p``.
        trend: Deterministic specification.

    Returns:
        A tuple ``(target, design, nobs)`` where ``target`` is ``y[p:]``,
        ``design`` stacks the deterministic block ahead of the lag block, and
        ``nobs`` is the effective sample size ``n - p``.

    Raises:
        DimensionError: If the series is not longer than ``order``.
    """
    n = y.shape[0]
    if n <= order:
        raise DimensionError(
            f"series of length {n} is too short for a conditional design of order {order}."
        )
    eff = n - order
    det = deterministic_columns(trend, eff, start=order + 1)
    lags = lag_matrix(y, order)
    return y[order:], np.column_stack([det, lags]), eff


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
