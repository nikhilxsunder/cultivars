# filepath: /examples/_fred.py
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
"""Shared FRED access for the example gallery.

Every example pulls its data live from FRED through ``fedfred``. This
helper does the three things each script would otherwise repeat: resolve
the API key with an actionable error when it is missing, fetch a named
set of series with one call, and align them on a single date index with
the undefined leading observations of transformed series dropped.

The leading underscore keeps the module out of the gallery
(``ignore_pattern`` in ``conf.py``); sphinx-gallery puts the examples
directory on ``sys.path`` while a script runs, so ``from _fred import
fetch`` resolves there exactly as it does from a notebook started in this
directory.

Example:
    >>> from _fred import fetch  # doctest: +SKIP
    >>> data = fetch(  # doctest: +SKIP
    ...     {"ip": "INDPRO", "ffr": "FEDFUNDS"},
    ...     start="1965-01-01",
    ...     end="2007-12-01",
    ...     units={"ip": "log"},
    ... )
"""

from __future__ import annotations

from collections.abc import Mapping

import fedfred
import pandas as pd

_KEY_HELP = (
    "No FRED API key is configured. The examples download their data live, "
    "so set the FRED_API_KEY environment variable (a free key is issued at "
    "https://fred.stlouisfed.org/docs/api/api_key.html) or call "
    "fedfred.set_api_key(...) before importing this module's helpers."
)


def client(*, cache: bool = True) -> fedfred.FredAPI:
    """Build a ``fedfred`` client from the configured key.

    Args:
        cache: Whether to memoize responses for the life of the client, so a
            notebook that re-runs a cell does not re-download the series.

    Returns:
        A ready ``FredAPI`` instance.

    Raises:
        RuntimeError: If neither ``fedfred.set_api_key`` nor the
            ``FRED_API_KEY`` environment variable supplies a key.
    """
    key = fedfred.get_api_key()
    if not key:
        raise RuntimeError(_KEY_HELP)
    return fedfred.FredAPI(api_key=key, cache_mode=cache)


def fetch(
    series: Mapping[str, str],
    *,
    start: str,
    end: str | None = None,
    frequency: str | None = None,
    aggregation_method: str | None = None,
    units: str | Mapping[str, str] | None = None,
    require_all: bool = True,
) -> pd.DataFrame:
    """Download several FRED series and align them on one date index.

    Args:
        series: Column name to FRED series id, in the column order wanted.
        start: First observation date, ``"YYYY-MM-DD"``.
        end: Last observation date; ``None`` means the latest available.
        frequency: FRED frequency code applied to every series (``"d"``,
            ``"w"``, ``"m"``, ``"q"``, ``"a"``); ``None`` keeps each series'
            native frequency, which must then agree across the request.
        aggregation_method: How FRED aggregates to a lower frequency,
            ``"avg"``, ``"sum"`` or ``"eop"``; ignored without ``frequency``.
        units: A FRED units code applied to every series, or a mapping from
            column name to code for a per-series choice (``"lin"``, ``"log"``,
            ``"chg"``, ``"pch"``, ``"pc1"``, ``"pca"``, ``"cch"``, ``"cca"``).
        require_all: Keep only dates at which every series is observed
            (default). ``False`` keeps every date at which at least one is,
            with ``NaN`` elsewhere, for models that accept ragged panels.

    Returns:
        A frame indexed by date with one float column per requested series,
        restricted to dates at which every series is observed unless
        ``require_all`` is ``False``.

    Raises:
        RuntimeError: If no API key is configured.
        ValueError: If the aligned frame is empty.
    """
    api = client()
    per_series: dict[str, str | None]
    if units is None or isinstance(units, str):
        per_series = dict.fromkeys(series, units)
    else:
        per_series = {name: units.get(name) for name in series}
    columns: dict[str, pd.Series] = {}
    for name, series_id in series.items():
        frame = api.get_series_observations(
            series_id,
            observation_start=start,
            observation_end=end,
            frequency=frequency,
            aggregation_method=aggregation_method,
            units=per_series[name],
        )
        columns[name] = pd.to_numeric(frame["value"], errors="coerce").rename(name)
    if require_all:
        data = pd.concat(columns.values(), axis=1, join="inner").dropna()
    else:
        data = pd.concat(columns.values(), axis=1, join="outer").dropna(how="all")
    if data.empty:
        raise ValueError(
            f"no dates at which all of {tuple(series.values())} are observed between "
            f"{start} and {end or 'the latest release'}"
        )
    data.index.name = "date"
    return data.astype(float)
