from __future__ import annotations

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
