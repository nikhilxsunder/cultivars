# filepath: /src/cultivars/_core/_matrices.py
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
from ._containers import LagPolynomial
from ._defaults import _TREND_WIDTH


def companion_matrix(ar_coeffs: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Build the companion matrix from autoregressive coefficients.

    For ``y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + u_t`` the companion is the
    ``(kp, kp)`` matrix whose top block row is ``[A_1, ..., A_p]`` and whose
    sub-diagonal is the identity.

    Args:
        ar_coeffs: Coefficients ``A_1, ..., A_p``. Shape ``(p,)`` (scalar,
            ``k = 1``) or ``(p, k, k)`` (matrix).

    Returns:
        The companion matrix of shape ``(k * p, k * p)``.

    Raises:
        DimensionError: If ``ar_coeffs`` has an unsupported rank or is not
            square (matrix case), or if ``p < 1``.
        NumericalError: If ``ar_coeffs`` contains non-finite values.

    Example:
        >>> C = companion_matrix([0.5, -0.3])
        >>> C.shape
        (2, 2)
        >>> bool(C[1, 0] == 1.0)
        True
    """
    ar = _as_coefficient_stack(ar_coeffs)
    p, k = ar.shape[0], ar.shape[1]
    if p < 1:
        raise DimensionError("companion_matrix requires at least one lag (p >= 1).")
    size = k * p
    companion = np.zeros((size, size), dtype=np.float64)
    companion[:k, :] = np.concatenate([ar[i] for i in range(p)], axis=1)
    if p > 1:
        companion[k:, : k * (p - 1)] = np.eye(k * (p - 1))
    return companion


def companion_from_polynomial(poly: LagPolynomial) -> npt.NDArray[np.float64]:
    """Build the companion matrix from a monic AR :class:`LagPolynomial`.

    Args:
        poly: A polynomial in monic AR form (constant term == identity).

    Returns:
        The companion matrix of shape ``(k * p, k * p)``.

    Raises:
        SpecificationError: If ``poly`` is not in monic AR form.
    """
    return companion_matrix(poly.ar_coeffs())


def selector_matrix(k: int, p: int) -> npt.NDArray[np.float64]:
    """The selector ``J = [I_k, 0, ..., 0]`` extracting the leading block.

    Used to project companion powers back to the original ``k`` variables, e.g.
    the reduced-form impulse responses ``Psi_h = J @ C**h @ J.T``.

    Args:
        k: System dimension.
        p: Lag order.

    Returns:
        A ``(k, k * p)`` array.

    Raises:
        DimensionError: If ``k < 1`` or ``p < 1``.
    """
    if k < 1 or p < 1:
        raise DimensionError(f"selector_matrix requires k >= 1 and p >= 1; got k={k}, p={p}.")
    selector = np.zeros((k, k * p), dtype=np.float64)
    selector[:, :k] = np.eye(k)
    return selector


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


def deterministic_columns(trend: str, nobs: int, *, start: int = 1) -> npt.NDArray[np.float64]:
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


def trailing_lag(
    series: npt.NDArray[np.float64], *, delay: int, length: int
) -> npt.NDArray[np.float64]:
    """Take the last ``length`` values of ``series`` shifted back by ``delay``.

    The slice a regime-indexed model needs to line its transition variable up
    with an already-trimmed effective sample. Deriving the trim from ``length``
    rather than recomputing it from an order and a delay keeps the result
    correct when the delay was *searched*: the selected delay can be shorter
    than the one that set the trim.

    Args:
        series: The full-length variable to slice.
        delay: Periods to shift back.
        length: Length of the effective sample to align with.

    Returns:
        An array of length ``length``.

    Raises:
        DimensionError: If the requested window runs off the front of the
            series, which means the caller trimmed by less than ``delay``.

    Example:
        >>> import numpy as np
        >>> trailing_lag(np.arange(6.0), delay=2, length=3)
        array([1., 2., 3.])
    """
    n = series.shape[0]
    start = n - length
    if start - delay < 0:
        raise DimensionError(
            f"a lag of {delay} needs at least {delay} observations ahead of an "
            f"effective sample of {length}; the series has {n}."
        )
    return series[start - delay : n - delay]


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


def conditional_design(
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


def psd_sqrt(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A matrix square root valid for symmetric positive-semidefinite input.

    Negative eigenvalues arising from round-off are clipped to zero, so a
    covariance that is singular or marginally indefinite still yields a usable
    factor rather than a NaN.

    Args:
        matrix: A symmetric positive-semidefinite matrix.

    Returns:
        A factor ``S`` with ``S @ S.T`` equal to ``matrix``.

    Raises:
        DimensionError: If ``matrix`` is not square.
        NumericalError: If the eigendecomposition is not finite.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise DimensionError(f"psd_sqrt requires a square matrix; got {matrix.shape}.")
    eigvals, eigvecs = np.linalg.eigh(matrix)
    if not np.all(np.isfinite(eigvals)):
        raise NumericalError("psd_sqrt eigendecomposition produced non-finite values.")
    root = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0.0, None)))
    return np.asarray(root, dtype=np.float64)


def _as_coefficient_stack(ar_coeffs: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return AR coefficients as a ``(p, k, k)`` float array (scalar -> k=1)."""
    ar = np.asarray(ar_coeffs, dtype=np.float64)
    if ar.ndim == 1:
        ar = ar.reshape(ar.shape[0], 1, 1)
    elif ar.ndim == 3:
        if ar.shape[1] != ar.shape[2]:
            raise DimensionError(
                f"Matrix AR coefficients must be square; got trailing dimensions "
                f"{ar.shape[1]}x{ar.shape[2]}."
            )
    else:
        raise DimensionError(
            f"AR coefficients must be rank 1 (scalar) or rank 3 (matrix); got rank {ar.ndim}."
        )
    if not np.all(np.isfinite(ar)):
        raise NumericalError("AR coefficients contain non-finite values.")
    return ar


def link_matrix(
    unit: int,
    *,
    weights: npt.ArrayLike,
    unit_of_column: Sequence[int],
    variable_of_column: Sequence[int],
) -> npt.NDArray[np.float64]:
    """The selector ``W_i`` with ``z_it = W_i x_t`` for one unit.

    Stacks the unit's own rows -- plain selections out of the global vector --
    above its star rows, which are the weighted combinations
    :func:`star_variables` computes. Expressing the foreign variables as a
    matrix acting on the global vector is the whole trick behind a global
    system: once every unit's equations are written against ``x_t`` rather than
    against its own private mixture, they stack into one square system that can
    be solved.

    Args:
        unit: Index of the unit.
        weights: A validated ``(n_units, n_units)`` matrix.
        unit_of_column: Owning unit index for each global column.
        variable_of_column: Variable identity for each global column.

    Returns:
        A ``(2 * k_i, k)`` array: own rows first, then star rows.
    """
    matrix = np.asarray(weights, dtype=np.float64)
    owners = tuple(int(u) for u in unit_of_column)
    kinds = tuple(int(v) for v in variable_of_column)
    width = len(owners)
    own = [c for c in range(width) if owners[c] == unit]
    rows = [np.eye(width, dtype=np.float64)[c] for c in own]
    for column in own:
        row = np.zeros(width, dtype=np.float64)
        for source in range(width):
            if kinds[source] == kinds[column]:
                row[source] += matrix[unit, owners[source]]
        rows.append(row)
    return np.vstack(rows)


def _lower_cholesky(matrix: npt.NDArray[np.float64], label: str) -> npt.NDArray[np.float64]:
    """The lower Cholesky factor, with a specification error rather than LinAlgError.

    Args:
        matrix: A symmetric matrix expected to be positive definite.
        label: What the matrix is, for the error message.

    Returns:
        The lower-triangular factor.

    Raises:
        NumericalError: If the matrix is not positive definite, which means one
            innovation is an exact linear combination of the others and no
            factorization into ``k`` independent shocks exists.
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise NumericalError(
            f"{label} is not positive definite, so no factorization into "
            "independent shocks exists; one innovation is an exact linear "
            "combination of the others."
        ) from error


def _long_run_matrix(
    coefficients: npt.NDArray[np.float64],
    ma_coefficients: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """The cumulated response of a closed system to its innovations, in closed form.

    ``F = (I - A_1 - ... - A_p)^{-1} (I + M_1 + ... + M_q)``, the sum of the
    moving-average matrices without truncating the infinite series. A pure
    autoregression passes no moving-average stack and gets ``A(1)^{-1}``.

    Args:
        coefficients: ``(p, k, k)`` autoregressive stack.
        ma_coefficients: Optional ``(q, k, k)`` moving-average stack.

    Returns:
        The ``(k, k)`` long-run impact matrix of the innovations.

    Raises:
        SpecificationError: If the autoregressive polynomial has a unit root,
            in which case cumulated responses diverge and no long-run object
            exists; difference the integrated variables first, which is what
            Blanchard-Quah do with output.
    """
    k = int(coefficients.shape[1])
    identity = np.eye(k)
    ar_sum = coefficients.sum(axis=0) if coefficients.shape[0] else np.zeros((k, k))
    ma_sum = identity + (
        ma_coefficients.sum(axis=0)
        if ma_coefficients is not None and ma_coefficients.shape[0]
        else 0.0
    )
    lhs = identity - ar_sum
    if abs(float(np.linalg.det(lhs))) < 1e-12:
        raise SpecificationError(
            "the autoregressive polynomial has a (near-)unit root, so cumulated "
            "responses diverge and the long-run impact matrix does not exist. "
            "Long-run identification lives in a stationary representation: "
            "difference the integrated variables first."
        )
    return np.linalg.solve(lhs, ma_sum)


def _orthogonal_from_angles(angles: npt.NDArray[np.float64], size: int) -> npt.NDArray[np.float64]:
    """Compose Givens rotations into a special orthogonal matrix.

    One rotation per coordinate plane, ``size (size - 1) / 2`` in all, which
    is exactly the dimension of the rotation group: the parameterization is
    neither redundant nor short, so a search over the angles is a search over
    rotations with no flat directions built in. Reflections -- the other
    component of the orthogonal group -- are unreachable by construction;
    callers that need them compose a fixed sign flip, which is how a shock and
    its negative differ anyway.

    Args:
        angles: The plane-rotation angles, in the row-major plane order
            ``(0,1), (0,2), ..., (size-2, size-1)``.
        size: Matrix dimension.

    Returns:
        A ``(size, size)`` orthogonal matrix with determinant one.

    Raises:
        DimensionError: If the angle count does not match the plane count.
    """
    expected = size * (size - 1) // 2
    values = np.asarray(angles, dtype=np.float64).ravel()
    if values.shape[0] != expected:
        raise DimensionError(
            f"a {size}-dimensional rotation needs {expected} angles; got {values.shape[0]}."
        )
    out = np.eye(size, dtype=np.float64)
    position = 0
    for i in range(size - 1):
        for j in range(i + 1, size):
            cos = float(np.cos(values[position]))
            sin = float(np.sin(values[position]))
            rotation = np.eye(size, dtype=np.float64)
            rotation[i, i] = cos
            rotation[j, j] = cos
            rotation[i, j] = -sin
            rotation[j, i] = sin
            out = out @ rotation
            position += 1
    return out
