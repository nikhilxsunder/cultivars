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

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._types import Frequency


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


def star_variables(
    panel: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    unit_of_column: Sequence[int],
    variable_of_column: Sequence[int],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Build each unit's foreign aggregates from the global panel.

    The foreign counterpart of a variable is the weighted average of *that same
    variable* across the other units, so the construction needs to know which
    global column is which variable as well as which unit owns it. A unit that
    lacks a variable some other unit has simply does not get a star series for
    it, which is the ordinary situation once the sample spans economies with
    different data.

    Args:
        panel: The ``(nobs, k)`` global panel, all units side by side.
        weights: A validated ``(n_units, n_units)`` matrix.
        unit_of_column: Owning unit index for each global column.
        variable_of_column: Variable identity for each global column, shared
            across units so that like is averaged with like.

    Returns:
        One ``(nobs, k_i)`` array per unit, columns in the same order as that
        unit's own columns appear in the panel.

    Raises:
        DimensionError: If the label sequences do not match the panel width.
    """
    data = np.asarray(panel, dtype=np.float64)
    matrix = np.asarray(weights, dtype=np.float64)
    width = data.shape[1]
    owners = tuple(int(u) for u in unit_of_column)
    kinds = tuple(int(v) for v in variable_of_column)
    if len(owners) != width or len(kinds) != width:
        raise DimensionError(
            f"unit_of_column and variable_of_column must each have one entry per global "
            f"column ({width}); got {len(owners)} and {len(kinds)}."
        )
    out: list[npt.NDArray[np.float64]] = []
    for unit in range(matrix.shape[0]):
        own = [c for c in range(width) if owners[c] == unit]
        block = np.zeros((data.shape[0], len(own)), dtype=np.float64)
        for position, column in enumerate(own):
            for source in range(width):
                if kinds[source] == kinds[column]:
                    block[:, position] += matrix[unit, owners[source]] * data[:, source]
        out.append(block)
    return tuple(out)


def aggregation_weights(
    kind: Frequency, period: int, *, weights: npt.ArrayLike | None = None
) -> npt.NDArray[np.float64]:
    """Sub-period weights a single low-frequency reading places on the latent path.

    Args:
        kind: One of :data:`Frequency`.
        period: Sub-periods per low-frequency period.
        weights: Explicit flow weights, newest sub-period first. Defaults to a
            simple average, which is the right convention for a series in
            levels. A series in log-differences needs the triangular weights of
            Mariano and Murasawa instead, because summing growth rates is not
            the same as averaging levels -- pass them here rather than hoping
            the default is close.

    Returns:
        A vector of length ``period``, newest sub-period first.

    Raises:
        SpecificationError: If the kind or period is unrecognized, or explicit
            weights are the wrong length.
    """
    if kind not in Frequency.__value__:
        raise SpecificationError(f"kind must be one of {Frequency.__value__}; got {kind!r}.")
    if period < 1:
        raise SpecificationError(f"period must be at least 1; got {period}.")
    if kind in ("high", "stock"):
        out = np.zeros(period, dtype=np.float64)
        out[0] = 1.0
        return out
    if weights is None:
        return np.full(period, 1.0 / period, dtype=np.float64)
    supplied = np.asarray(weights, dtype=np.float64).ravel()
    if supplied.shape != (period,):
        raise SpecificationError(
            f"weights must have one entry per sub-period ({period}); got {supplied.shape}."
        )
    return supplied


def _aggregation_weights(
    kind: Frequency, period: int, *, weights: npt.ArrayLike | None = None
) -> npt.NDArray[np.float64]:
    """Sub-period weights a single low-frequency reading places on the latent path.

    Args:
        kind: How the series relates to the latent path.
        period: Number of high-frequency sub-periods per low-frequency period.
        weights: Explicit sub-period weights, most recent first. Overrides the
            default for ``kind``. Length must equal ``period``.

    Returns:
        Weights of length ``period``, most recent sub-period first.

    Raises:
        SpecificationError: If ``weights`` has the wrong length or is not finite.

    Example:
        >>> _aggregation_weights("flow", 3)
        array([0.33333333, 0.33333333, 0.33333333])
        >>> _aggregation_weights("stock", 3)
        array([1., 0., 0.])
    """
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape[0] != period:
            raise SpecificationError(f"weights must have length period={period}; got {w.shape[0]}.")
        if not np.all(np.isfinite(w)):
            raise SpecificationError("weights must be finite.")
        return w
    out = np.zeros(period, dtype=np.float64)
    if kind == "flow":
        out[:] = 1.0 / period
    else:
        out[0] = 1.0
    return out


def _midas_weights(theta: npt.ArrayLike, lags: int) -> npt.NDArray[np.float64]:
    """Normalized exponential Almon weights, most recent sub-period first.

    The two-parameter family of Ghysels, Santa-Clara, and Valkanov:
    ``w_j proportional to exp(theta_1 j + theta_2 j^2)`` for ``j = 1, ..., lags``,
    normalized to sum to one so that the scale of the regressor lives in its
    slope coefficient rather than in the polynomial. The exponent is shifted by
    its maximum before exponentiating, so the normalization is exact for any
    finite parameters and no clipping is needed.

    Args:
        theta: The two polynomial parameters ``(theta_1, theta_2)``.
        lags: Window length in sub-periods, at least one.

    Returns:
        Weights of length ``lags`` summing to one, weight on the most recent
        sub-period first.

    Raises:
        SpecificationError: If ``theta`` does not hold exactly two finite
            values, or ``lags`` is less than one.

    Example:
        >>> _midas_weights([0.0, 0.0], 3)
        array([0.33333333, 0.33333333, 0.33333333])
        >>> w = _midas_weights([0.0, -0.5], 4)
        >>> bool(w[0] > w[1] > w[2] > w[3])
        True
    """
    if lags < 1:
        raise SpecificationError(f"lags must be at least 1; got {lags}.")
    values = np.asarray(theta, dtype=np.float64).ravel()
    if values.shape[0] != 2:
        raise SpecificationError(
            f"theta must hold exactly two parameters (theta_1, theta_2); got {values.shape[0]}."
        )
    if not np.all(np.isfinite(values)):
        raise SpecificationError("theta must be finite.")
    j = np.arange(1, lags + 1, dtype=np.float64)
    exponent = values[0] * j + values[1] * j**2
    weights = np.exp(exponent - exponent.max())
    return weights / weights.sum()


def _midas_windows(
    exog_high: npt.NDArray[np.float64], *, nobs: int, period: int, lags: int, start: int
) -> npt.NDArray[np.float64]:
    """Stack the high-frequency history behind each low-frequency row.

    Args:
        exog_high: ``(nobs * period, m)`` high-frequency panel, chronological.
        nobs: Number of low-frequency periods the panel spans.
        period: Sub-periods per low-frequency period.
        lags: Window length in sub-periods.
        start: First low-frequency row to keep. The caller guarantees the
            window behind it does not reach before the sample.

    Returns:
        A ``(nobs - start, lags, m)`` array whose ``[t, j]`` entry is the
        ``j``-th most recent sub-period reading at the end of low-frequency
        period ``start + t``.
    """
    ends = (np.arange(start, nobs) + 1) * period - 1
    idx = ends[:, np.newaxis] - np.arange(lags)[np.newaxis, :]
    return exog_high[idx]


def _nelson_siegel_loadings(grid: npt.NDArray[np.float64], decay: float) -> npt.NDArray[np.float64]:
    """The level, slope, and curvature loadings at one decay.

    Args:
        grid: Strictly positive maturities.
        decay: The exponential decay rate ``lambda``.

    Returns:
        An ``(n_points, 3)`` matrix; columns are level ``1``, slope
        ``(1 - e^{-lt}) / (lt)``, and curvature ``(1 - e^{-lt}) / (lt) -
        e^{-lt}``.
    """
    x = decay * grid
    slope = np.where(x > 1e-10, (1.0 - np.exp(-x)) / np.where(x > 1e-10, x, 1.0), 1.0)
    curvature = slope - np.exp(-x)
    return np.column_stack([np.ones_like(grid), slope, curvature])


def _projection_scores(
    curves: npt.NDArray[np.float64], basis: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Least-squares coordinates of each curve in a basis.

    Args:
        curves: The ``(nobs, n_points)`` panel.
        basis: The ``(n_points, r)`` basis matrix.

    Returns:
        The ``(nobs, r)`` score panel.

    Raises:
        NumericalError: If the basis is rank-deficient on this grid.
    """
    gram = basis.T @ basis
    try:
        return np.linalg.solve(gram, basis.T @ curves.T).T
    except np.linalg.LinAlgError as error:
        raise NumericalError(
            "the basis is rank-deficient on this grid: two of its functions are "
            "numerically indistinguishable at the observed points, so the "
            "projection has no unique answer. Fewer components, fewer spline "
            "degrees of freedom, or a larger decay usually resolves it."
        ) from error
