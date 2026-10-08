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

from collections.abc import Iterable, Mapping, Sequence
from typing import TypeAliasType, cast, get_args

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._converters import _variable_names
from ._defaults import _MIN_SEASONAL_CYCLES
from ._matrices import deterministic_columns
from ._protocols import PredictiveResult


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


def validate_exog(exog: npt.ArrayLike | None, nobs: int) -> npt.NDArray[np.float64]:
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
        return np.empty((nobs, 0), dtype=np.float64)
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


def validate_exog_matrix(
    exog: npt.ArrayLike, *, nobs: int, label: str = "exog"
) -> npt.NDArray[np.float64]:
    """Coerce a required exogenous regressor block and align it to a time index.

    The multivariate counterpart of :func:`validate_exog`, standing to it as
    :func:`validate_endog_matrix` stands to :func:`validate_endog`. The split is
    the same one and it is about the contract rather than the shape: a
    univariate family may or may not carry exogenous regressors, so its
    validator accepts ``None`` and hands back an ``Optional``; a VARX is defined
    by having them, so requiring one here puts that in the signature instead of
    in a runtime guard, and no caller downstream has to narrow a value that was
    never going to be absent.

    Carries no minimum-length rule and no transpose heuristic. An exogenous
    block has no sample of its own -- it is only ever meaningful alongside an
    endogenous panel -- so the one structural question worth asking is whether
    its rows line up with that panel, and a mismatch there is unambiguous in a
    way that a shape guess never is.

    Args:
        exog: The regressor block. A one-dimensional input is promoted to a
            single column, since one exogenous variable is the common case and
            requiring a trailing axis for it is friction with no payoff.
        nobs: Number of rows the endogenous panel carries.
        label: Name used in error messages, so a second block in the same call
            -- a future path, an instrument set -- reports under its own name.

    Returns:
        A ``(nobs, m)`` float array.

    Raises:
        DimensionError: If the input is not one- or two-dimensional, has no
            columns, or has a row count other than ``nobs``.
        NumericalError: If the block contains non-finite values.

    Example:
        >>> validate_exog_matrix([1.0, 2.0, 3.0], nobs=3).shape
        (3, 1)
    """
    arr = np.asarray(exog, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise DimensionError(
            f"{label} must be two-dimensional (nobs, m); got a {arr.ndim}-dimensional "
            f"array of shape {arr.shape}."
        )
    if arr.shape[1] < 1:
        raise DimensionError(f"{label} must have at least one column.")
    if arr.shape[0] != nobs:
        raise DimensionError(
            f"{label} has {arr.shape[0]} rows but the endogenous panel has {nobs}; "
            "the two are read against the same time index and must be aligned."
        )
    if not np.all(np.isfinite(arr)):
        raise NumericalError(f"{label} contains non-finite values.")
    return arr


def validate_panel(
    panel: npt.ArrayLike | Sequence[npt.ArrayLike], *, label: str = "panel"
) -> tuple[npt.NDArray[np.float64], ...]:
    """Coerce a collection of per-unit series into a tuple of aligned matrices.

    Two input shapes are accepted and they mean different things. A three-
    dimensional array is a balanced panel indexed ``(unit, time, variable)``. A
    sequence of two-dimensional arrays is the general case and may be ragged,
    which is what an unbalanced panel is.

    The per-unit checks deliberately omit the transpose heuristic that
    :func:`validate_endog_matrix` applies. That heuristic reads "more columns
    than rows" as a transposed series, which is sound for one long series and
    wrong for a panel unit, where a short ``T`` next to a moderate number of
    variables is ordinary rather than suspicious. The stacked panel is validated
    as a whole downstream, which is where the heuristic still has purchase.

    Args:
        panel: A ``(n_units, nobs, k)`` array, or a sequence of ``(nobs_i, k)``
            arrays. One-dimensional units are promoted to a single column.
        label: Name used in error messages.

    Returns:
        One float matrix per unit, in the order given.

    Raises:
        DimensionError: If the input is an array of rank other than three, is
            not a sequence, is empty, contains a unit of rank other than one or
            two, contains a unit with no columns, or mixes column counts.
        NumericalError: If any unit contains non-finite values.

    Example:
        >>> units = validate_panel(np.zeros((3, 20, 2)))
        >>> len(units), units[0].shape
        (3, (20, 2))
    """
    if isinstance(panel, np.ndarray):
        if panel.ndim != 3:
            raise DimensionError(
                f"{label} given as an array must be three-dimensional "
                f"(units, time, variables); got shape {panel.shape}. Pass a sequence of "
                "two-dimensional arrays for an unbalanced panel."
            )
        raw: list[npt.ArrayLike] = [panel[i] for i in range(panel.shape[0])]
    elif isinstance(panel, Sequence):
        raw = list(panel)
    else:
        raise DimensionError(
            f"{label} must be a three-dimensional array or a sequence of two-dimensional "
            f"arrays; got {type(panel).__name__}."
        )
    if not raw:
        raise DimensionError(f"{label} must contain at least one unit.")
    blocks: list[npt.NDArray[np.float64]] = []
    for index, unit in enumerate(raw):
        arr = np.asarray(unit, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise DimensionError(
                f"unit {index} of {label} must be two-dimensional (nobs, k); got a "
                f"{arr.ndim}-dimensional array of shape {arr.shape}."
            )
        if arr.shape[1] < 1:
            raise DimensionError(f"unit {index} of {label} must have at least one column.")
        if not np.all(np.isfinite(arr)):
            raise NumericalError(f"unit {index} of {label} contains non-finite values.")
        blocks.append(arr)
    widths = {block.shape[1] for block in blocks}
    if len(widths) != 1:
        raise DimensionError(
            f"every unit of {label} must carry the same variables in the same order; got "
            f"column counts {sorted(widths)}."
        )
    return tuple(blocks)


def validate_weights(
    weights: npt.ArrayLike, *, n_units: int, label: str = "weights"
) -> npt.NDArray[np.float64]:
    """Coerce and check a cross-unit weight matrix.

    Row ``i`` says how unit ``i`` sees the rest of the world: ``w_ij`` is the
    share unit ``j`` contributes to unit ``i``'s foreign aggregate. Three
    properties are checked rather than assumed, because each failure produces a
    model that estimates cleanly and means something different from what was
    intended.

    A non-zero diagonal makes a unit part of its own foreign aggregate, which
    puts the dependent variable on both sides of the equation and destroys the
    weak exogeneity the whole construction rests on. Rows that do not sum to one
    silently rescale the foreign variables, so a coefficient on them is no
    longer the elasticity anyone thinks it is. Negative weights are not
    obviously wrong -- a net-position matrix can carry them -- but they are
    unusual enough that passing one by accident is far more likely than passing
    one on purpose, so they are refused unless asked for.

    Args:
        weights: An ``(n_units, n_units)`` array.
        n_units: Number of units the global system links.
        label: Name used in error messages.

    Returns:
        An ``(n_units, n_units)`` float array.

    Raises:
        DimensionError: If the matrix is not square of the expected size.
        NumericalError: If it contains non-finite values.
        SpecificationError: If the diagonal is non-zero, a row does not sum to
            one, or any weight is negative.

    Example:
        >>> validate_weights([[0.0, 1.0], [1.0, 0.0]], n_units=2).shape
        (2, 2)
    """
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape != (n_units, n_units):
        raise DimensionError(
            f"{label} must be ({n_units}, {n_units}), one row and column per unit; "
            f"got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise NumericalError(f"{label} contains non-finite values.")
    diagonal = np.diag(arr)
    if np.any(np.abs(diagonal) > 1e-12):
        offenders = tuple(int(i) for i in np.flatnonzero(np.abs(diagonal) > 1e-12))
        raise SpecificationError(
            f"{label} must have a zero diagonal; units {offenders} carry weight on "
            "themselves, which would put a unit inside its own foreign aggregate and "
            "destroy the weak exogeneity the linkage depends on."
        )
    if np.any(arr < -1e-12):
        raise SpecificationError(
            f"{label} contains negative entries. A net-position matrix can legitimately "
            "have them, but passing one unintentionally is the far more common case, so "
            "they are refused here; take the absolute value and renormalize if the sign "
            "is meant."
        )
    sums = arr.sum(axis=1)
    bad = np.flatnonzero(np.abs(sums - 1.0) > 1e-8)
    if bad.size:
        worst = int(bad[np.argmax(np.abs(sums[bad] - 1.0))])
        raise SpecificationError(
            f"every row of {label} must sum to one; row {worst} sums to {sums[worst]:.6g}. "
            "Unnormalized rows rescale the foreign variables, so the coefficients on them "
            "stop being the elasticities they are read as."
        )
    return arr


def _validate_observed(endog: npt.ArrayLike, *, label: str = "endog") -> npt.NDArray[np.float64]:
    """Coerce a panel in which missing entries are meaningful rather than errors.

    Every other validator in the package rejects non-finite input, because
    everywhere else a NaN is a data problem. Here it is the specification: an
    unobserved month of a quarterly series is how the calendar is expressed.
    Infinities are still rejected -- those are never meaningful -- and so is a
    column with no observations at all, which cannot inform anything and whose
    presence is almost always an alignment mistake.

    Args:
        endog: The ``(nobs, k)`` panel at the high frequency, with ``nan`` in
            every period a series is not observed.
        label: Name used in error messages.

    Returns:
        A float64 array of shape ``(nobs, k)``, missing entries preserved.

    Raises:
        DimensionError: If the input is not two-dimensional after promotion,
            or has no rows or columns.
        NumericalError: If any entry is infinite.
        SpecificationError: If any column is entirely missing.
    """
    arr = np.asarray(endog, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise DimensionError(f"{label} must be at most two-dimensional; got {arr.ndim}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise DimensionError(f"{label} must have at least one row and column; got {arr.shape}.")
    if np.isinf(arr).any():
        raise NumericalError(
            f"{label} contains infinite values. Missing observations must be "
            "nan; an infinity is never a valid observation."
        )
    empty = [int(j) for j in range(arr.shape[1]) if not np.isfinite(arr[:, j]).any()]
    if empty:
        raise SpecificationError(
            f"{label} columns {empty} are entirely missing. A series with no "
            "observations cannot be identified from the others; drop it or "
            "check the frequency alignment."
        )
    return arr


def _validate_curves(curves: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce and check a panel of discretized curves.

    Deliberately not :func:`validate_endog_matrix`: that validator rejects a
    panel with more columns than rows, which for a VAR is the right guard and
    for curves is wrong on its face -- a daily yield curve observed at thirty
    maturities over twenty days is a perfectly good functional sample.

    Args:
        curves: The ``(nobs, n_points)`` panel, one curve per row.

    Returns:
        A 2-D float array.

    Raises:
        DimensionError: If the input is not two-dimensional or has fewer than
            two rows or two columns.
        NumericalError: If any entry is non-finite.
    """
    arr = np.asarray(curves, dtype=np.float64)
    if arr.ndim != 2:
        raise DimensionError(
            f"curves must be 2-D with one curve per row; got {arr.ndim} dimension(s)."
        )
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise DimensionError(
            f"curves must have at least two rows and two columns; got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise NumericalError("curves must be finite.")
    return arr


def _validate_ordering(names: tuple[str, ...], order: Sequence[str] | None) -> tuple[int, ...]:
    """Map a declared variable ordering onto column indices.

    Args:
        names: The result's variable labels.
        order: The declared ordering, or ``None`` for the labels as given.

    Returns:
        A permutation of ``range(k)``.

    Raises:
        SpecificationError: If ``order`` is not a permutation of ``names``.
    """
    if order is None:
        return tuple(range(len(names)))
    declared = tuple(str(name) for name in order)
    if sorted(declared) != sorted(names):
        raise SpecificationError(
            f"order must be a permutation of the variable names {names}; got {declared}."
        )
    return tuple(names.index(name) for name in declared)


def _validate_sign_patterns(
    restrictions: Mapping[str, Mapping[str, str]], names: tuple[str, ...]
) -> tuple[tuple[tuple[int, float], ...], ...]:
    """Compile declared sign patterns into per-column (variable, sign) pairs.

    Args:
        restrictions: Mapping from shock label to its pattern, itself a mapping
            from variable name to ``"+"`` or ``"-"``.
        names: The result's variable labels.

    Returns:
        One tuple of ``(variable index, +-1.0)`` pairs per restricted column,
        in declaration order.

    Raises:
        SpecificationError: If the declaration is empty, names more shocks than
            the system has, restricts a shock with no signs, references an
            unknown variable, or uses a symbol other than ``"+"`` or ``"-"``.
    """
    if not restrictions:
        raise SpecificationError(
            "restrictions must declare at least one shock; an empty declaration identifies nothing."
        )
    if len(restrictions) > len(names):
        raise SpecificationError(
            f"{len(restrictions)} restricted shocks exceed the {len(names)} shocks the system has."
        )
    compiled: list[tuple[tuple[int, float], ...]] = []
    for label, pattern in restrictions.items():
        if not pattern:
            raise SpecificationError(f"shock {label!r} declares no signs; drop it or restrict it.")
        cells: list[tuple[int, float]] = []
        for variable, sign in pattern.items():
            if variable not in names:
                raise SpecificationError(
                    f"unknown variable {variable!r} in shock {label!r}; expected one of {names}."
                )
            if sign not in ("+", "-"):
                raise SpecificationError(
                    f"sign for {variable!r} in shock {label!r} must be '+' or '-'; got {sign!r}."
                )
            cells.append((names.index(variable), 1.0 if sign == "+" else -1.0))
        compiled.append(tuple(cells))
    return tuple(compiled)


def _validate_impact_pattern(
    pattern: npt.ArrayLike | None,
    *,
    size: int,
    label: str,
    default_diagonal: float | None = None,
) -> tuple[npt.NDArray[np.float64], tuple[tuple[int, int], ...]]:
    """Split a contemporaneous-restriction pattern into fixed values and free cells.

    The convention is the one the SVAR literature writes on paper: a finite
    entry is a restriction -- almost always zero, sometimes a normalization of
    one -- and ``nan`` marks a coefficient the likelihood must estimate.

    Args:
        pattern: The ``(size, size)`` pattern, or ``None`` for a default: the
            identity when ``default_diagonal`` is given, with that value fixed
            on the diagonal and ``nan`` off it reserved to the caller's
            convention.
        size: System dimension.
        label: Which matrix this is, for error messages.
        default_diagonal: When ``pattern`` is ``None``, the fixed diagonal
            value of the default pattern; off-diagonal entries default to
            zero. ``None`` forbids omission.

    Returns:
        The matrix of fixed values with zeros in the free cells, and the free
        cell coordinates in row-major order.

    Raises:
        SpecificationError: If the pattern is omitted without a default, has
            the wrong shape, or contains an infinity.
    """
    if pattern is None:
        if default_diagonal is None:
            raise SpecificationError(f"{label} must be supplied for this model.")
        base = np.eye(size, dtype=np.float64) * default_diagonal
        return base, ()
    arr = np.asarray(pattern, dtype=np.float64)
    if arr.shape != (size, size):
        raise SpecificationError(f"{label} must have shape ({size}, {size}); got {arr.shape}.")
    if np.isinf(arr).any():
        raise SpecificationError(
            f"{label} entries must be finite restrictions or nan for a free "
            "coefficient; infinities restrict nothing."
        )
    free = tuple((int(i), int(j)) for i in range(size) for j in range(size) if np.isnan(arr[i, j]))
    base = np.where(np.isnan(arr), 0.0, arr)
    return np.asarray(base, dtype=np.float64), free


def _validate_narrative_events(
    shock_signs: Sequence[tuple[str, int, str]],
    contributions: Sequence[tuple[str, str, int, str]],
    *,
    labels: tuple[str, ...],
    names: tuple[str, ...],
    nobs: int,
) -> tuple[
    tuple[tuple[tuple[int, float], ...], ...],
    tuple[tuple[int, int, int, bool], ...],
    tuple[int, ...],
]:
    """Compile narrative restrictions into index form against one sample.

    Args:
        shock_signs: Declared shock-sign events, each ``(shock label, period,
            sign)`` with the period indexing the effective sample -- one row
            per residual row of the result being identified.
        contributions: Declared contribution events, each ``(shock label,
            variable, period, kind)`` with ``kind`` one of ``"most"`` or
            ``"overwhelming"``.
        labels: The shock labels, one per column, in column order.
        names: The result's variable labels.
        nobs: Rows of the effective sample the periods index.

    Returns:
        Per-column shock-sign requirements as ``(event index, +-1.0)`` pairs;
        contribution requirements as ``(column, variable index, event index,
        overwhelming?)``; and the unique periods the events reference, which
        is what the event indices index into.

    Raises:
        SpecificationError: If an event references an unknown shock label or
            variable, a period outside the effective sample, or an
            unrecognized sign or kind.
    """
    periods: list[int] = []

    def _event_index(period: int, what: str) -> int:
        moment = int(period)
        if not 0 <= moment < nobs:
            raise SpecificationError(
                f"{what} references period {period}, outside the effective "
                f"sample of {nobs} residual rows. Narrative periods index the "
                "effective sample -- the estimation sample minus the burned "
                "lags -- exactly as a proxy's instrument rows do."
            )
        if moment not in periods:
            periods.append(moment)
        return periods.index(moment)

    per_column: list[list[tuple[int, float]]] = [[] for _ in labels]
    for label, period, sign in shock_signs:
        if label not in labels:
            raise SpecificationError(
                f"unknown shock {label!r} in a shock-sign event; expected one of {labels}."
            )
        if sign not in ("+", "-"):
            raise SpecificationError(f"sign for shock {label!r} must be '+' or '-'; got {sign!r}.")
        index = _event_index(period, f"shock-sign event on {label!r}")
        per_column[labels.index(label)].append((index, 1.0 if sign == "+" else -1.0))

    compiled_contributions: list[tuple[int, int, int, bool]] = []
    for label, variable, period, kind in contributions:
        if label not in labels:
            raise SpecificationError(
                f"unknown shock {label!r} in a contribution event; expected one of {labels}."
            )
        if variable not in names:
            raise SpecificationError(
                f"unknown variable {variable!r} in a contribution event; expected one of {names}."
            )
        if kind not in ("most", "overwhelming"):
            raise SpecificationError(
                f"contribution kind must be 'most' or 'overwhelming'; got {kind!r}."
            )
        index = _event_index(period, f"contribution event on {label!r}")
        compiled_contributions.append(
            (
                labels.index(label),
                names.index(variable),
                index,
                kind == "overwhelming",
            )
        )
    return (
        tuple(tuple(cells) for cells in per_column),
        tuple(compiled_contributions),
        tuple(periods),
    )


def _validate_regimes(
    regimes: npt.ArrayLike, *, nobs: int
) -> tuple[tuple[str, ...], npt.NDArray[np.int64]]:
    """Coerce a per-period regime assignment over one effective sample.

    Args:
        regimes: One regime label per residual row of the result being
            identified -- the effective sample, exactly as a proxy's
            instrument rows index it. Labels may be anything hashable a
            string can name.
        nobs: Rows of the effective sample.

    Returns:
        The regime labels in order of first appearance, and the integer
        assignment mapping each period to its label's position.

    Raises:
        SpecificationError: If the assignment does not align with the
            effective sample or names fewer than two regimes.
    """
    values = np.asarray(regimes)
    if values.ndim != 1 or values.shape[0] != nobs:
        raise SpecificationError(
            f"regimes must assign one label per residual row ({nobs}), the "
            f"effective sample after the burned lags; got shape {values.shape}."
        )
    labels: list[str] = []
    assignment = np.empty(nobs, dtype=np.int64)
    for position, raw in enumerate(values):
        label = str(raw)
        if label not in labels:
            labels.append(label)
        assignment[position] = labels.index(label)
    if len(labels) < 2:
        raise SpecificationError(
            "heteroskedasticity identifies through variance *change*: at "
            f"least two regimes are needed, got {labels}."
        )
    return tuple(labels), assignment


def _validate_wide_panel(panel: npt.ArrayLike, *, label: str = "panel") -> npt.NDArray[np.float64]:
    """Coerce an informational panel that may be wider than it is long.

    The counterpart of :func:`validate_endog_matrix` for data that summarize
    rather than get modelled equation by equation: a factor model's
    informational panel routinely carries more series than observations, so
    the rows-versus-columns guard that protects a VAR would reject perfectly
    good input here.

    Args:
        panel: The ``(nobs, n_series)`` panel, one period per row.
        label: Name used in error messages.

    Returns:
        A 2-D float array.

    Raises:
        DimensionError: If the input is not two-dimensional or has fewer than
            two rows or two columns.
        NumericalError: If any entry is non-finite.
    """
    arr = np.asarray(panel, dtype=np.float64)
    if arr.ndim != 2:
        raise DimensionError(
            f"{label} must be 2-D with one period per row; got {arr.ndim} dimension(s)."
        )
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise DimensionError(
            f"{label} must have at least two rows and two columns; got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise NumericalError(f"{label} must be finite.")
    return arr


def _validate_band(low: float, high: float) -> None:
    """Reject a band that does not describe periods.

    Raises:
        SpecificationError: If the band is malformed.
    """
    if not 2.0 <= low < high:
        raise SpecificationError(
            f"the band needs 2 <= low < high in periods (observations per "
            f"cycle); got {low}, {high}. Two observations per cycle is the "
            "shortest period the sampling can represent."
        )


def _validate_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    """Coerce probability levels to floats and require each to lie in ``(0, 1)``.

    Args:
        quantiles: Probability levels for pointwise quantile summaries.

    Returns:
        The levels as a tuple of floats, in the order given.

    Raises:
        SpecificationError: If any level is outside the open unit interval.
    """
    levels = tuple(float(q) for q in quantiles)
    if any(not 0.0 < q < 1.0 for q in levels):
        raise SpecificationError(f"quantiles must lie in (0, 1); got {levels}.")
    return levels


def _validate_posterior_draws(
    draws: npt.ArrayLike, log_kernel: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Coerce draws to ``(S, d)`` and kernel values to ``(S,)``."""
    theta = np.asarray(draws, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta[:, None]
    if theta.ndim != 2:
        raise DimensionError(f"draws must be (S,) or (S, d); got shape {theta.shape}.")
    values = np.asarray(log_kernel, dtype=np.float64)
    if values.shape != (theta.shape[0],):
        raise DimensionError(
            f"log_kernel must have one value per draw, shape ({theta.shape[0]},); "
            f"got {values.shape}."
        )
    if theta.shape[0] <= 2 * theta.shape[1] + 2:
        raise SpecificationError(
            f"Need more draws than twice the dimension plus two to fit a Gaussian "
            f"envelope; got {theta.shape[0]} draws in {theta.shape[1]} dimensions."
        )
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(values))):
        raise NumericalError("draws and log_kernel must be finite.")
    return theta, values


def _validate_log_density_matrix(log_density: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce held-out predictive log densities to a validated ``(T, M)`` matrix.

    Args:
        log_density: One row per evaluation origin, one column per model.

    Returns:
        The ``(T, M)`` float array.

    Raises:
        DimensionError: If the input is not two-dimensional.
        SpecificationError: If there are fewer than two models or fewer
            origins than models.
        NumericalError: If any entry is not finite.

    Example:
        >>> _validate_log_density_matrix(np.zeros((10, 2))).shape
        (10, 2)
    """
    matrix = np.asarray(log_density, dtype=np.float64)
    if matrix.ndim != 2:
        raise DimensionError(f"log densities must be (T, M); got shape {matrix.shape}.")
    n_origins, n_models = matrix.shape
    if n_models < 2:
        raise SpecificationError(f"Combination needs at least two models; got {n_models}.")
    if n_origins < n_models:
        raise SpecificationError(
            f"Stacking weights over {n_models} models need at least {n_models} evaluation "
            f"origins; got {n_origins}."
        )
    if not np.all(np.isfinite(matrix)):
        raise NumericalError("log densities must be finite.")
    return matrix


def _validate_statistics(
    statistics: Sequence[str] | None, *, known: Sequence[str], default: Sequence[str]
) -> tuple[str, ...]:
    """Resolve the discrepancy statistics a predictive check computes.

    Args:
        statistics: Requested names, or ``None`` for the default set.
        known: Every name the registry offers.
        default: The set used when none is requested.

    Returns:
        The names in the order given, duplicates removed.

    Raises:
        SpecificationError: If a name is unknown or the request is empty.

    Example:
        >>> _validate_statistics(None, known=("mean", "sd"), default=("sd",))
        ('sd',)
        >>> _validate_statistics(["sd", "mean", "sd"], known=("mean", "sd"), default=("sd",))
        ('sd', 'mean')
    """
    chosen = tuple(default) if statistics is None else tuple(dict.fromkeys(statistics))
    if not chosen:
        raise SpecificationError("at least one discrepancy statistic is required.")
    unknown = [name for name in chosen if name not in known]
    if unknown:
        raise SpecificationError(
            f"unknown discrepancy statistic(s) {unknown}; known: {', '.join(known)}."
        )
    return chosen


def bandwidth(nobs: int, m: int | None, exponent: float) -> int:
    """Resolve the number of Fourier frequencies for a semiparametric estimator.

    Args:
        nobs: Series length.
        m: Explicit bandwidth, or ``None`` to derive it from ``exponent``.
        exponent: Exponent in the default rule ``m = floor(n ** exponent)``.

    Returns:
        The bandwidth, never below 2.

    Raises:
        SpecificationError: If an explicit ``m`` is below 2, or ``exponent``
            does not lie in ``(0, 1)``.

    Example:
        >>> bandwidth(400, None, 0.5)
        20
    """
    if m is not None:
        if m < 2:
            raise SpecificationError(f"bandwidth m must be >= 2; got {m}.")
        return int(m)
    if not (0.0 < exponent < 1.0):
        raise SpecificationError(f"bandwidth_exponent must lie in (0, 1); got {exponent}.")
    return max(2, int(np.floor(nobs**exponent)))


def _validate_semiparametric(
    endog: npt.ArrayLike, m: int | None, exponent: float, *, minimum: int
) -> tuple[npt.NDArray[np.float64], int]:
    """Coerce a series for a frequency-domain estimator and resolve its bandwidth.

    Args:
        endog: The series.
        m: Explicit bandwidth, or ``None`` to derive it from ``exponent``.
        exponent: Exponent of the rule ``m = floor(T ** exponent)``.
        minimum: Fewest observations accepted.

    Returns:
        ``(y, m)``: the series as a flat float array and the bandwidth.

    Raises:
        DimensionError: If the series is not one-dimensional.
        NumericalError: If it is not finite.
        SpecificationError: If it is shorter than ``minimum``, or the
            bandwidth is unusable or exceeds the Fourier frequencies
            available.

    Example:
        >>> y, m = _validate_semiparametric(np.arange(400.0), None, 0.5, minimum=64)
        >>> y.shape, m
        ((400,), 20)
    """
    y = validate_endog(endog)
    n = y.shape[0]
    if n < minimum:
        raise SpecificationError(
            f"a semiparametric estimate needs at least {minimum} observations; got {n}."
        )
    resolved = bandwidth(n, m, exponent)
    if resolved > n // 2:
        raise SpecificationError(
            f"bandwidth {resolved} exceeds the {n // 2} Fourier frequencies available."
        )
    return y, resolved


def _validate_spectrum_panel(
    data: npt.ArrayLike, detrend: str, names: tuple[str, ...] | None, *, minimum: int
) -> tuple[npt.NDArray[np.float64], str, tuple[str, ...]]:
    """Coerce a panel for a frequency-domain estimate and resolve its detrending and labels.

    Args:
        data: ``(T, k)`` panel or one-dimensional series.
        detrend: ``"n"``, ``"c"`` or ``"ct"``.
        names: Series labels, or ``None`` to read them from the source or
            fall back to ``y1 .. yk``.
        minimum: Fewest observations accepted.

    Returns:
        ``(panel, detrend, labels)`` with the panel untouched -- the
        detrending itself is a transform, applied by the caller.

    Raises:
        DimensionError: If the panel is malformed.
        NumericalError: If it is not finite.
        SpecificationError: If the trend is unknown, the panel is shorter
            than ``minimum``, or the label count is wrong.

    Example:
        >>> panel, trend, labels = _validate_spectrum_panel(
        ...     np.ones((40, 2)), "c", None, minimum=32
        ... )
        >>> panel.shape, trend, labels
        ((40, 2), 'c', ('y1', 'y2'))
    """
    panel = validate_endog_matrix(data)
    detrend = validate_choice(detrend, ("n", "c", "ct"), "detrend")
    n, k = panel.shape
    if n < minimum:
        raise SpecificationError(
            f"a nonparametric spectrum needs at least {minimum} observations; got {n}."
        )
    labels = _variable_names(data, k) if names is None else tuple(names)
    if len(labels) != k:
        raise SpecificationError(f"names must have {k} entries; got {len(labels)}.")
    return panel, detrend, labels


def _validate_replications(
    observed: npt.ArrayLike, replicated: npt.ArrayLike, *, minimum: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Check an observed panel against its replications.

    Args:
        observed: ``(n,)`` or ``(n, k)`` data.
        replicated: ``(R, n)`` or ``(R, n, k)`` replicated data sets.
        minimum: Fewest replications accepted.

    Returns:
        ``(observed, replicated)`` as ``(n, k)`` and ``(R, n, k)`` float arrays.

    Raises:
        DimensionError: If the shapes disagree or there are too few
            replications.
        NumericalError: If a replication is not finite.

    Example:
        >>> y, rep = _validate_replications(np.zeros(5), np.zeros((3, 5)), minimum=2)
        >>> y.shape, rep.shape
        ((5, 1), (3, 5, 1))
    """
    data = np.asarray(observed, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise DimensionError(f"observed must be 1-D or 2-D; got {data.ndim}-D.")
    reps = np.asarray(replicated, dtype=np.float64)
    if reps.ndim == 2:
        reps = reps[:, :, None]
    if reps.ndim != 3:
        raise DimensionError(f"replicated must be 2-D or 3-D; got {reps.ndim}-D.")
    if reps.shape[1:] != data.shape:
        raise DimensionError(
            f"each replication has shape {reps.shape[1:]}; the observed panel has {data.shape}."
        )
    if reps.shape[0] < minimum:
        raise DimensionError(
            f"{reps.shape[0]} replications; at least {minimum} are needed for a tail "
            "probability with any resolution."
        )
    if not np.all(np.isfinite(reps)):
        raise NumericalError("a replicated data set is not finite; the simulator diverged.")
    if not np.all(np.isfinite(data)):
        raise NumericalError("the observed panel is not finite.")
    return data, reps


def _validate_hyperparameter_pair(
    value: Iterable[float], *, name: str, positive_first: bool
) -> tuple[float, float]:
    """Check a two-number prior hyperparameter such as ``(mean, variance)`` or ``(shape, rate)``.

    Args:
        value: The pair as given.
        name: Argument name, for error messages.
        positive_first: Whether the first entry must be positive as well as
            the second (a shape or a Beta parameter, as opposed to a mean).

    Returns:
        The pair as floats.

    Raises:
        SpecificationError: If the pair is not two finite numbers or a
            required entry is not positive.

    Example:
        >>> _validate_hyperparameter_pair((20, 1.5), name="phi", positive_first=True)
        (20.0, 1.5)
    """
    try:
        first, second = (float(entry) for entry in value)
    except (TypeError, ValueError) as error:
        raise SpecificationError(f"{name} must be a pair of numbers; got {value!r}.") from error
    if not (np.isfinite(first) and np.isfinite(second)):
        raise SpecificationError(f"{name} must be finite; got {value!r}.")
    if second <= 0.0:
        raise SpecificationError(f"the second entry of {name} must be positive; got {value!r}.")
    if positive_first and first <= 0.0:
        raise SpecificationError(f"both entries of {name} must be positive; got {value!r}.")
    return first, second


def _validate_conditions(
    conditions: Mapping[str, Sequence[float | None]], names: tuple[str, ...], steps: int
) -> npt.NDArray[np.float64]:
    """Lay stated future values on a ``(steps, k)`` grid, ``nan`` where a cell is free.

    Args:
        conditions: Variable name to its conditioned path from the first
            horizon; ``None`` or ``nan`` leaves a horizon free, and a
            path shorter than ``steps`` leaves the rest free.
        names: The system's variable names, fixing the column order.
        steps: Forecast horizons.

    Returns:
        The ``(steps, k)`` grid.

    Raises:
        SpecificationError: If nothing is conditioned, a name is unknown,
            or every stated value is free.
        DimensionError: If a path is longer than ``steps``.

    Example:
        >>> _validate_conditions({"b": [1.0, None]}, ("a", "b"), 3)
        array([[nan,  1.],
               [nan, nan],
               [nan, nan]])
    """
    if not conditions:
        raise SpecificationError("at least one variable must be conditioned.")
    grid = np.full((steps, len(names)), np.nan)
    for name, path in conditions.items():
        if name not in names:
            raise SpecificationError(f"unknown variable {name!r}; expected one of {names}.")
        values = list(path)
        if len(values) > steps:
            raise DimensionError(
                f"the condition on {name!r} states {len(values)} horizons; the forecast has "
                f"{steps}."
            )
        for h, value in enumerate(values):
            if value is None:
                continue
            number = float(value)
            if np.isfinite(number):
                grid[h, names.index(name)] = number
    if not np.any(np.isfinite(grid)):
        raise SpecificationError("every stated condition is None or nan; nothing to condition on.")
    return grid


def _validate_regression(
    endog: npt.ArrayLike, exog: npt.ArrayLike | None, trend: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Coerce a target and its regression design: deterministic terms, then regressors.

    Args:
        endog: The ``(T,)`` target.
        exog: ``(T, k)`` or ``(T,)`` regressors, or ``None``.
        trend: ``"n"``, ``"c"`` or ``"ct"``.

    Returns:
        ``(y, design)`` with ``design`` of shape ``(T, q)``, ``q >= 1``.

    Raises:
        SpecificationError: If the trend is unknown or the design has no
            columns.
        DimensionError: If the target is not one-dimensional or the
            regressors do not align.
        NumericalError: If a value is not finite or the design is rank
            deficient.

    Example:
        >>> y, x = _validate_regression([1.0, 2.0, 3.0, 5.0], [0.0, 1.0, 0.0, 1.0], "c")
        >>> x.shape
        (4, 2)
    """
    validate_choice(trend, ("n", "c", "ct"), "trend")
    y = validate_endog(endog)
    nobs = y.shape[0]
    design = np.hstack([deterministic_columns(trend, nobs), validate_exog(exog, nobs)])
    if design.shape[1] == 0:
        raise SpecificationError("the regression has no coefficients: pass exog or a trend.")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise NumericalError("the regression design is rank deficient.")
    return y, design


def _validate_aligned_series(
    *series: npt.ArrayLike, horizon: int, minimum: int, labels: str
) -> tuple[list[npt.NDArray[np.float64]], int]:
    """Check origin-aligned evaluation series for a comparison at one horizon.

    Args:
        *series: Two or more ``(T,)`` series -- outcomes, forecasts, or
            losses -- aligned origin by origin.
        horizon: The forecast horizon behind them.
        minimum: Fewest origins accepted.
        labels: What the series are, for error messages.

    Returns:
        ``(blocks, count)``: the series as flat float arrays, and ``T``.

    Raises:
        DimensionError: If the shapes disagree.
        SpecificationError: If there are fewer than ``minimum`` origins,
            the horizon is not positive, or it is too long for the window.
        NumericalError: If a series is not finite.

    Example:
        >>> blocks, count = _validate_aligned_series(
        ...     np.zeros(20), np.ones(20), horizon=2, minimum=8, labels="losses"
        ... )
        >>> len(blocks), count
        (2, 20)
    """
    blocks = [np.asarray(block, dtype=np.float64).ravel() for block in series]
    if any(block.shape != blocks[0].shape for block in blocks):
        raise DimensionError(
            f"{labels} must align origin by origin; got shapes "
            f"{', '.join(str(block.shape) for block in blocks)}."
        )
    count = blocks[0].shape[0]
    if count < minimum:
        raise SpecificationError(
            f"a comparison over {count} origins has no power and unreliable size; provide at "
            f"least {minimum}."
        )
    if not all(np.all(np.isfinite(block)) for block in blocks):
        raise NumericalError(f"{labels} must be finite.")
    if horizon < 1:
        raise SpecificationError(f"horizon must be at least 1; got {horizon}.")
    if horizon >= count // 2:
        raise SpecificationError(
            f"a horizon of {horizon} needs more than {2 * horizon} evaluation origins; got {count}."
        )
    return blocks, count


def _validate_predictive(
    results: Sequence[object], *, minimum: int = 2, purpose: str = "a combination"
) -> tuple[PredictiveResult, ...]:
    """Check that every result simulates its predictive and that there are enough of them.

    Args:
        results: Candidate fitted results.
        minimum: Fewest results the caller can work with.
        purpose: What the results are for, used in messages.

    Returns:
        The results as a tuple, typed as :class:`PredictiveResult`.

    Raises:
        SpecificationError: If there are too few results or one exposes
            no ``forecast_paths``.

    Example:
        >>> _validate_predictive([object()], minimum=1)
        Traceback (most recent call last):
        ...
        cultivars.exceptions.SpecificationError: object exposes no forecast_paths(); ...
    """
    if len(results) < minimum:
        raise SpecificationError(f"{purpose} needs at least {minimum} models; got {len(results)}.")
    checked: list[PredictiveResult] = []
    for result in results:
        if not isinstance(result, PredictiveResult):
            raise SpecificationError(
                f"{type(result).__name__} exposes no forecast_paths(); {purpose} can only use "
                "results that simulate their posterior predictive."
            )
        checked.append(result)
    return tuple(checked)


def _validate_names(
    names: Sequence[str] | None, fallback: Sequence[str], *, label: str = "models"
) -> tuple[str, ...]:
    """Resolve optional labels against a default, requiring the count to match and no repeats.

    Args:
        names: Labels supplied by the caller, or ``None`` for the default.
        fallback: Default labels, one per item; also fixes the count.
        label: What is being named, used in messages.

    Returns:
        The labels as a tuple of strings.

    Raises:
        DimensionError: If the count does not match.
        SpecificationError: If a label repeats.

    Example:
        >>> _validate_names(None, ["a", "b"])
        ('a', 'b')
        >>> _validate_names(["x", "x"], ["a", "b"])
        Traceback (most recent call last):
        ...
        cultivars.exceptions.SpecificationError: models names must be distinct; got ('x', 'x').
    """
    labels = tuple(fallback) if names is None else tuple(str(name) for name in names)
    if len(labels) != len(fallback):
        raise DimensionError(f"{len(labels)} names for {len(fallback)} {label}.")
    if len(set(labels)) != len(labels):
        raise SpecificationError(f"{label} names must be distinct; got {labels}.")
    return labels


def _validate_seasonal(endog: npt.ArrayLike, period: int) -> npt.NDArray[np.float64]:
    """Coerce the series and check the period against it.

    Raises:
        SpecificationError: If the period is odd or below 2, or the
            series holds fewer than ``_MIN_SEASONAL_CYCLES`` full cycles.
    """
    y = validate_endog(endog)
    period = validate_order(period, "period", minimum=2)
    if period % 2:
        raise SpecificationError(f"period must be even; got {period}.")
    if y.shape[0] < _MIN_SEASONAL_CYCLES * period:
        raise SpecificationError(
            f"a seasonal unit-root test at period {period} needs at least "
            f"{_MIN_SEASONAL_CYCLES * period} observations; got {y.shape[0]}."
        )
    return y


def _validate_chronology(value: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Coerce a binary chronology: booleans as given, probabilities thresholded at one half.

    Args:
        value: ``(T,)`` boolean flags or a probability series such as a
            Markov-switching model's smoothed regime probability.

    Returns:
        ``(T,)`` boolean flags.

    Raises:
        DimensionError: If the input is not one-dimensional.
        NumericalError: If a probability series is non-finite.

    Example:
        >>> _validate_chronology([0.9, 0.4, 0.5]).tolist()
        [True, False, False]
    """
    array = np.asarray(value)
    if array.ndim != 1:
        raise DimensionError(f"a chronology must be one-dimensional; got shape {array.shape}.")
    if array.dtype == np.bool_:
        return np.asarray(array, dtype=np.bool_)
    probabilities = array.astype(np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise NumericalError("a probability chronology must be finite.")
    return np.asarray(probabilities > 0.5, dtype=np.bool_)


def _validate_specification(y: npt.NDArray[np.float64], order: int, delay: int) -> None:
    """Refuse an order or delay the series cannot carry.

    Raises:
        SpecificationError: If the sample is shorter than the auxiliary
            regression needs.
    """
    needed = max(order, delay) + 4 * order + 10
    if y.shape[0] < needed:
        raise SpecificationError(
            f"a nonlinearity test of order {order} and delay {delay} needs at least {needed} "
            f"observations; got {y.shape[0]}."
        )
