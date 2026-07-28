# filepath: /src/cultivars/core/transforms.py
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

"""Stationarity-inducing data transforms and their inverses.

Operates on arrays with time along ``axis=0`` (the cultivars ``T x k``
convention). These transforms require finite inputs; missing-observation
handling belongs to the state-space layer, not here.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from cultivars.exceptions import DimensionError, NumericalError, SpecificationError


class Standardized(NamedTuple):
    """A standardized series together with the parameters needed to invert it.

    Attributes:
        values: The standardized data, same shape as the input.
        mean: The per-column mean that was subtracted.
        scale: The per-column standard deviation that was divided out.
    """

    values: npt.NDArray[np.float64]
    mean: npt.NDArray[np.float64]
    scale: npt.NDArray[np.float64]

    def inverse(self, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map standardized values back to the original scale."""
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean


def _as_finite(y: npt.ArrayLike) -> npt.NDArray[np.float64]:
    arr = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise NumericalError(
            "Input contains non-finite values; handle missing data at the "
            "state-space layer before transforming."
        )
    return arr


def difference(y: npt.ArrayLike, d: int = 1, *, axis: int = 0) -> npt.NDArray[np.float64]:
    """Apply the ``d``-th difference ``(1 - L)**d``.

    Args:
        y: Input array.
        d: Non-negative order of differencing. ``d == 0`` returns a copy.
        axis: Axis along which to difference.

    Returns:
        The differenced array, shorter by ``d`` along ``axis``.

    Raises:
        SpecificationError: If ``d`` is negative.
        NumericalError: If ``y`` contains non-finite values.

    Example:
        >>> difference(np.array([1.0, 3.0, 6.0, 10.0]))
        array([2., 3., 4.])
    """
    if d < 0:
        raise SpecificationError(f"Differencing order d must be >= 0; got {d}.")
    arr = _as_finite(y)
    if d == 0:
        return arr.copy()
    return np.diff(arr, n=d, axis=axis)


def seasonal_difference(
    y: npt.ArrayLike, s: int, capital_d: int = 1, *, axis: int = 0
) -> npt.NDArray[np.float64]:
    """Apply the seasonal difference ``(1 - L**s)**capital_d``.

    Args:
        y: Input array.
        s: Seasonal period (e.g. 4 for quarterly, 12 for monthly); must be >= 1.
        capital_d: Non-negative number of seasonal differences.
        axis: Axis along which to difference.

    Returns:
        The seasonally differenced array, shorter by ``s * capital_d`` along ``axis``.

    Raises:
        SpecificationError: If ``s < 1`` or ``capital_d < 0``.
        DimensionError: If the series is too short for the requested differencing.
        NumericalError: If ``y`` contains non-finite values.
    """
    if s < 1:
        raise SpecificationError(f"Seasonal period s must be >= 1; got {s}.")
    if capital_d < 0:
        raise SpecificationError(f"Seasonal order must be >= 0; got {capital_d}.")
    out = _as_finite(y)
    for _ in range(capital_d):
        if out.shape[axis] <= s:
            raise DimensionError(
                f"Series too short for seasonal differencing at period {s}."
            )
        lead = np.take(out, indices=range(s, out.shape[axis]), axis=axis)
        lag = np.take(out, indices=range(0, out.shape[axis] - s), axis=axis)
        out = lead - lag
    return out


def log_transform(y: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Natural logarithm, with an explicit positivity guard.

    Raises:
        SpecificationError: If any value is non-positive.
        NumericalError: If ``y`` contains non-finite values.
    """
    arr = _as_finite(y)
    if np.any(arr <= 0.0):
        raise SpecificationError("log_transform requires strictly positive values.")
    return np.log(arr)


def log_difference(y: npt.ArrayLike, *, axis: int = 0) -> npt.NDArray[np.float64]:
    """First difference of the log (approximate growth rate)."""
    return difference(log_transform(y), 1, axis=axis)


def standardize(y: npt.ArrayLike, *, axis: int = 0, ddof: int = 0) -> Standardized:
    """Center and scale to zero mean and unit standard deviation.

    Args:
        y: Input array.
        axis: Axis over which to compute the mean and standard deviation.
        ddof: Delta degrees of freedom for the standard deviation.

    Returns:
        A :class:`Standardized` carrying the values, mean, and scale.

    Raises:
        NumericalError: If any column has zero standard deviation, or ``y``
            contains non-finite values.

    Example:
        >>> z = standardize(np.array([1.0, 2.0, 3.0]))
        >>> np.round(z.values, 4)
        array([-1.2247,  0.    ,  1.2247])
        >>> round(float(z.mean), 4)
        2.0
    """
    arr = _as_finite(y)
    mean = np.mean(arr, axis=axis, keepdims=True)
    scale = np.std(arr, axis=axis, ddof=ddof, keepdims=True)
    if np.any(scale == 0.0):
        raise NumericalError("Cannot standardize a series with zero standard deviation.")
    values = (arr - mean) / scale
    return Standardized(
        values=values,
        mean=np.squeeze(mean, axis=axis),
        scale=np.squeeze(scale, axis=axis),
    )


def undifference(
    dx: npt.ArrayLike, initials: npt.ArrayLike, *, d: int = 1
) -> npt.NDArray[np.float64]:
    """Invert ``d``-th differencing along ``axis=0`` (integration).

    Reconstructs the original ``T x k`` (or length-``T``) series from its
    ``d``-th difference and the ``d`` anchor values.

    Args:
        dx: The ``d``-th differenced series along axis 0.
        initials: The ``d`` anchor values, highest order first:
            ``[Delta^{d-1} x[0], ..., Delta^1 x[0], x[0]]``. Shape ``(d,)`` for a
            scalar series or ``(d, k)`` for a ``k``-column series.
        d: Order of differencing to invert; must be >= 1.

    Returns:
        The reconstructed series, longer by ``d`` along axis 0.

    Raises:
        SpecificationError: If ``d < 1`` or ``initials`` does not supply exactly
            ``d`` anchor values.
        NumericalError: If ``dx`` or ``initials`` contain non-finite values.

    Example:
        >>> x = np.array([1.0, 3.0, 6.0, 10.0])
        >>> undifference(difference(x), [1.0])
        array([ 1.,  3.,  6., 10.])
    """
    if d < 1:
        raise SpecificationError(f"undifference requires d >= 1; got {d}.")
    cur = _as_finite(dx)
    anchors = _as_finite(initials)
    if anchors.shape[0] != d:
        raise SpecificationError(
            f"Expected {d} anchor value(s) for d={d}; got {anchors.shape[0]}."
        )
    for order in range(d):
        cumulative = np.cumsum(cur, axis=0)
        anchor = anchors[order]
        prefix = np.asarray(anchor, dtype=np.float64)[np.newaxis, ...]
        cur = np.concatenate([prefix, anchor + cumulative], axis=0)
    return cur
