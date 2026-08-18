# filepath: /src/cultivars/_core/_validators.py
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

"""Input validation primitives.

Every public constructor routes its argument checking through this module,
so a malformed input produces the same exception type and the same message
shape regardless of which model raised it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeAliasType, cast, get_args

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError, NumericalError, SpecificationError


def validate_endog(endog: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce and check an endogenous series.

    Args:
        endog: The observed univariate series (1-D array-like).

    Returns:
        A 1-D float array.

    Raises:
        DimensionError: If ``endog`` is not one-dimensional.
        NumericalError: If ``endog`` contains non-finite values.
    """
    arr = np.asarray(endog, dtype=np.float64)
    if arr.ndim != 1:
        raise DimensionError(f"endog must be one-dimensional; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise NumericalError("endog contains non-finite values.")
    return arr


def validate_exog(exog: npt.ArrayLike | None, nobs: int) -> npt.NDArray[np.float64] | None:
    """Coerce optional exogenous regressors to a ``(nobs, k)`` matrix.

    A 1-D input is promoted to a single column.

    Args:
        exog: Regressors, or ``None`` when the model has none.
        nobs: Required number of rows.

    Returns:
        A ``(nobs, k)`` float array, or ``None``.

    Raises:
        DimensionError: If ``exog`` cannot be shaped to ``(nobs, k)``.
        NumericalError: If ``exog`` contains non-finite values.
    """
    if exog is None:
        return None
    x = np.asarray(exog, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != nobs:
        raise DimensionError(f"exog must have shape ({nobs}, k); got {x.shape}.")
    if not np.all(np.isfinite(x)):
        raise NumericalError("exog contains non-finite values.")
    return x


def validate_aligned(values: npt.ArrayLike, nobs: int, label: str) -> npt.NDArray[np.float64]:
    """Coerce a covariate that must align one-to-one with the endogenous series.

    Args:
        values: The covariate (e.g. an external threshold variable).
        nobs: Required length.
        label: Argument name, used in error messages.

    Returns:
        A 1-D float array of length ``nobs``.

    Raises:
        DimensionError: If the length or rank does not match.
        NumericalError: If the covariate contains non-finite values.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != nobs:
        raise DimensionError(
            f"{label} must be one-dimensional with length {nobs}; got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise NumericalError(f"{label} contains non-finite values.")
    return arr


def validate_order(value: object, label: str, *, minimum: int = 0) -> int:
    """Check a single integral order term.

    ``bool`` is rejected explicitly: ``isinstance(True, int)`` is ``True`` in
    Python, so without this guard ``order=True`` would silently mean ``1``.

    Args:
        value: The candidate order.
        label: Argument name, used in error messages.
        minimum: Smallest permitted value.

    Returns:
        The order as an ``int``.

    Raises:
        SpecificationError: If ``value`` is not integral or is below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SpecificationError(f"{label} must be an integer; got {value!r}.")
    out = int(value)
    if out < minimum:
        raise SpecificationError(f"{label} must be >= {minimum}; got {out}.")
    return out


def validate_order_tuple(
    order: Sequence[int], labels: Sequence[str], *, minimum: int = 0
) -> tuple[int, ...]:
    """Check a tuple of integral order terms such as ``(p, d, q)``.

    Args:
        order: The candidate order tuple.
        labels: One name per element, used in error messages.
        minimum: Smallest permitted value for every element.

    Returns:
        The validated orders as a tuple of ``int``.

    Raises:
        SpecificationError: If the arity is wrong or any element is invalid.

    Example:
        >>> validate_order_tuple((1, 0, 2), ("p", "d", "q"))
        (1, 0, 2)
    """
    if len(order) != len(labels):
        raise SpecificationError(
            f"order must have {len(labels)} elements {tuple(labels)}; got {tuple(order)}."
        )
    return tuple(
        validate_order(v, lab, minimum=minimum) for v, lab in zip(order, labels, strict=True)
    )


def validate_choice[T](value: T, allowed: object, label: str) -> T:
    """Check a categorical specification against its permitted values.

    Accepts either a plain sequence of options or a PEP 695 ``type`` alias over
    a :data:`~typing.Literal`, which is how every categorical option in this
    package is declared. A ``TypeAliasType`` is not iterable, so passing one to
    a naive ``value not in allowed`` raises ``TypeError`` rather than
    validating anything; the alias is unwrapped through ``__value__`` first.

    Args:
        value: The candidate option.
        allowed: A ``type`` alias over a ``Literal``, or any iterable of the
            permitted options.
        label: Argument name, used in error messages.

    Returns:
        The value unchanged, so the call can sit inside an assignment.

    Raises:
        SpecificationError: If ``value`` is not among the permitted options.

    Example:
        >>> from typing import Literal
        >>> type Trend = Literal["n", "c", "ct"]
        >>> validate_choice("c", Trend, "trend")
        'c'
        >>> validate_choice("c", ("n", "c", "ct"), "trend")
        'c'
    """
    if isinstance(allowed, TypeAliasType):
        options: tuple[object, ...] = get_args(allowed.__value__)
    else:
        options = tuple(cast("Iterable[object]", allowed))
    if value not in options:
        raise SpecificationError(f"{label} must be one of {options}; got {value!r}.")
    return value


def validate_open_interval(value: float, label: str, *, low: float, high: float) -> float:
    """Check that a float lies strictly inside ``(low, high)``.

    Args:
        value: The candidate value.
        label: Argument name, used in error messages.
        low: Exclusive lower bound.
        high: Exclusive upper bound.

    Returns:
        The value as a ``float``.

    Raises:
        SpecificationError: If ``value`` is non-finite or outside the interval.
    """
    out = float(value)
    if not np.isfinite(out) or not (low < out < high):
        raise SpecificationError(f"{label} must lie in ({low}, {high}); got {out}.")
    return out


def validate_transition(transition: npt.ArrayLike, n_regimes: int) -> npt.NDArray[np.float64]:
    """Check a row-stochastic Markov transition matrix.

    Args:
        transition: Candidate ``(K, K)`` transition matrix.
        n_regimes: The expected number of regimes ``K``.

    Returns:
        The validated ``(K, K)`` float array.

    Raises:
        DimensionError: If the shape is not ``(K, K)``.
        NumericalError: If the matrix contains non-finite values.
        SpecificationError: If any entry is negative or any row does not sum to 1.
    """
    from ._defaults import _ROW_SUM_ATOL

    mat = np.asarray(transition, dtype=np.float64)
    if mat.shape != (n_regimes, n_regimes):
        raise DimensionError(
            f"transition must have shape ({n_regimes}, {n_regimes}); got {mat.shape}."
        )
    if not np.all(np.isfinite(mat)):
        raise NumericalError("transition contains non-finite values.")
    if np.any(mat < 0.0):
        raise SpecificationError("transition contains negative probabilities.")
    if not np.allclose(mat.sum(axis=1), 1.0, atol=_ROW_SUM_ATOL):
        raise SpecificationError("transition rows must each sum to 1.")
    return mat


def validate_endog_matrix(endog: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce and check an endogenous panel for a vector model.

    The counterpart to :func:`validate_endog` for models whose observation is a
    vector rather than a scalar. A one-dimensional input is promoted to a
    single column rather than rejected, so a one-variable VAR is reachable
    without a reshape and behaves like the autoregression it is.

    Args:
        endog: The observed panel, shape ``(nobs, k)``, or a 1-D series.

    Returns:
        A 2-D float array with time down the rows.

    Raises:
        DimensionError: If the input is not two-dimensional after promotion,
            has no columns, or has no more rows than columns.
        NumericalError: If the panel contains non-finite values.

    Example:
        >>> validate_endog_matrix(np.arange(6.0)).shape
        (6, 1)
        >>> validate_endog_matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).shape
        (3, 2)
    """
    arr = np.asarray(endog, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise DimensionError(
            f"endog must be two-dimensional (nobs, k); got a {arr.ndim}-dimensional "
            f"array of shape {arr.shape}."
        )
    nobs, k = arr.shape
    if k < 1:
        raise DimensionError("endog must have at least one column.")
    if nobs <= k:
        raise DimensionError(
            f"endog has {nobs} observations for {k} variables; a vector model reads "
            f"time down the rows, so this is almost certainly transposed."
        )
    if not np.all(np.isfinite(arr)):
        raise NumericalError("endog contains non-finite values.")
    return arr
