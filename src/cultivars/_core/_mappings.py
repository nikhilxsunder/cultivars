from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ._estimators import _excess_kurtosis, _lag_one_autocorrelation, _skewness, _standardized
from ._types import CointegrationTrend

#: Unrestricted deterministic terms each Johansen case leaves in the short-run equation.
_UNRESTRICTED_TREND: dict[CointegrationTrend, str] = {
    "none": "n",
    "restricted_constant": "n",
    "constant": "c",
    "restricted_trend": "c",
    "trend": "ct",
}

#: Deterministic terms the implied levels representation carries once the
#: restricted terms are folded back out of the cointegrating space.
_LEVELS_TREND: dict[CointegrationTrend, str] = {
    "none": "n",
    "restricted_constant": "c",
    "constant": "c",
    "restricted_trend": "ct",
    "trend": "ct",
}

_KASS_RAFTERY_SCALE: tuple[tuple[float, str], ...] = (
    (2.0, "not worth more than a bare mention"),
    (6.0, "positive"),
    (10.0, "strong"),
    (float("inf"), "very strong"),
)
"""Kass and Raftery's (1995) verbal scale, keyed by the upper bound on ``|2 log BF|``."""

_DISCREPANCIES: dict[str, Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = {
    "mean": lambda panel: np.asarray(panel.mean(axis=0)),
    "sd": lambda panel: np.asarray(panel.std(axis=0, ddof=1)),
    "min": lambda panel: np.asarray(panel.min(axis=0)),
    "max": lambda panel: np.asarray(panel.max(axis=0)),
    "skewness": _skewness,
    "kurtosis": _excess_kurtosis,
    "acf1": _lag_one_autocorrelation,
    "arch1": lambda panel: _lag_one_autocorrelation(_standardized(panel) ** 2),
}
"""Named per-variable discrepancy statistics, each ``(n, k) -> (k,)``.

``kurtosis`` is excess kurtosis; ``arch1`` is the lag-one autocorrelation
of the squared demeaned series, the volatility-clustering statistic.
"""

_DISCREPANCY_NAMES: tuple[str, ...] = tuple(_DISCREPANCIES)
"""Every discrepancy statistic a check can compute, by name.

``mean``, ``sd``, ``min``, ``max``, ``skewness``, ``kurtosis`` (excess),
``acf1`` (lag-one autocorrelation), and ``arch1`` (lag-one autocorrelation
of the squared demeaned series).
"""
