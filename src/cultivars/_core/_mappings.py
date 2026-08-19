
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
