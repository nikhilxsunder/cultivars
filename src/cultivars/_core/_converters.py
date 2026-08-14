# filepath: /src/cultivars/_core/_converters.py
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

"""Converters between the numpy core and the optional dataframe ecosystems.

Every computational path in this package is numpy and scipy only. Dataframes
are an *output* convenience, so the libraries that provide them are optional
extras rather than runtime dependencies, and nothing here is imported until a
user actually asks for a frame.

That import is deferred rather than guarded at module scope for two reasons: it
keeps ``import cultivars`` free of a pandas import even when pandas happens to
be installed, and it lets the error message name the extra to install rather
than surfacing a bare :exc:`ModuleNotFoundError` from three frames down.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .unsorted import require_optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

    import pandas as pd
    import polars as pl


def to_pandas_frame(
    columns: Mapping[str, npt.NDArray[Any]],
    *,
    index: npt.ArrayLike | None = None,
    index_name: str | None = None,
) -> pd.DataFrame:
    """Build a :class:`pandas.DataFrame` from equal-length columns.

    Args:
        columns: Column name to one-dimensional array. All arrays must share a
            length.
        index: Optional row index; defaults to a zero-based integer range.
        index_name: Optional name for the index.

    Returns:
        The assembled ``DataFrame``.

    Raises:
        ImportError: If pandas is not installed.
        ValueError: If the columns are not all the same length.
    """
    pd = require_optional("pandas")
    _check_equal_length(columns)
    frame = pd.DataFrame(dict(columns), index=None if index is None else np.asarray(index))
    if index_name is not None:
        frame.index.name = index_name
    return frame


def to_polars_frame(columns: Mapping[str, npt.NDArray[Any]]) -> pl.DataFrame:
    """Build a :class:`polars.DataFrame` from equal-length columns.

    Polars has no row index, so any positional information must be passed as an
    explicit column by the caller rather than smuggled in as an index.

    Args:
        columns: Column name to one-dimensional array.

    Returns:
        The assembled ``DataFrame``.

    Raises:
        ImportError: If polars is not installed.
        ValueError: If the columns are not all the same length.
    """
    pl = require_optional("polars")
    _check_equal_length(columns)
    return pl.DataFrame({name: np.asarray(values) for name, values in columns.items()})


def _check_equal_length(columns: Mapping[str, npt.NDArray[Any]]) -> None:
    """Reject ragged column sets before handing them to a frame constructor.

    Args:
        columns: Column name to array.

    Raises:
        ValueError: If two columns differ in length.
    """
    lengths = {name: int(np.asarray(values).shape[0]) for name, values in columns.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"columns must share a length; got {lengths}.")


def as_sequence(values: npt.ArrayLike) -> Sequence[Any]:
    """Coerce array-like input to a plain Python sequence for rendering.

    Args:
        values: Any array-like.

    Returns:
        A list of Python scalars, so that formatting never depends on numpy's
        own repr rules.
    """
    return list(np.asarray(values).tolist())
