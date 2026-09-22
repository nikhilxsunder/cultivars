from __future__ import annotations

from typing import Final

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

_MACKINNON_TAU_SMALL: Final[dict[str, tuple[float, ...]]] = {
    "n": (0.6344, 1.2378, 3.2496e-2),
    "c": (2.1659, 1.4412, 3.8269e-2),
    "ct": (3.2512, 1.6047, 4.9588e-2),
}
"""MacKinnon (1994) response-surface coefficients for the lower tail of the DF ``tau`` law.

Keyed by deterministic specification; a polynomial in the statistic whose
value the standard normal CDF turns into a p-value, used below ``_MACKINNON_TAU_STAR``.
"""

_MACKINNON_TAU_LARGE: Final[dict[str, tuple[float, ...]]] = {
    "n": (0.4797, 9.3557e-1, -0.6999e-1, 3.3066e-2),
    "c": (1.7339, 9.3202e-1, -1.2745e-1, -1.0368e-2),
    "ct": (2.5261, 6.1654e-1, -3.7956e-1, -6.0285e-2),
}
"""MacKinnon (1994) coefficients for the body and upper tail of the DF ``tau`` law."""

_MACKINNON_TAU_STAR: Final[dict[str, float]] = {"n": -1.04, "c": -1.61, "ct": -2.89}
"""Statistic below which the small-p surface applies."""

_MACKINNON_TAU_MIN: Final[dict[str, float]] = {"n": -19.0, "c": -18.83, "ct": -16.18}
"""Statistic below which the p-value is reported as zero."""

_MACKINNON_TAU_MAX: Final[dict[str, float]] = {"n": 1.51, "c": 2.74, "ct": 0.7}
"""Statistic above which the p-value is reported as one."""

_MACKINNON_CRITICAL_2010: Final[dict[str, tuple[tuple[float, float, float, float], ...]]] = {
    "n": (
        (-2.56574, -2.2358, -3.627, 0.0),
        (-1.94100, -0.2686, -3.365, 31.223),
        (-1.61682, 0.2656, -2.714, 25.364),
    ),
    "c": (
        (-3.43035, -6.5393, -16.786, -79.433),
        (-2.86154, -2.8903, -4.234, -40.040),
        (-2.56677, -1.5384, -2.809, 0.0),
    ),
    "ct": (
        (-3.95877, -9.0531, -28.428, -134.155),
        (-3.41049, -4.3904, -9.036, -45.374),
        (-3.12705, -2.5856, -3.925, -22.380),
    ),
}
"""MacKinnon (2010) finite-sample critical-value surfaces for the DF ``tau`` law.

Rows are the 1%, 5%, and 10% levels; each is ``(b0, b1, b2, b3)`` in
``cv = b0 + b1 / T + b2 / T**2 + b3 / T**3``.
"""

_KPSS_CRITICAL: Final[dict[str, tuple[tuple[float, float], ...]]] = {
    "c": ((0.10, 0.347), (0.05, 0.463), (0.025, 0.574), (0.01, 0.739)),
    "ct": ((0.10, 0.119), (0.05, 0.146), (0.025, 0.176), (0.01, 0.216)),
}
"""Kwiatkowski et al. (1992) Table 1: ``(level, critical value)`` pairs, level then trend."""

_DFGLS_CRITICAL: Final[dict[str, tuple[float, float, float]]] = {
    "c": (-2.58, -1.95, -1.62),
    "ct": (-3.48, -2.89, -2.57),
}
"""Elliott, Rothenberg and Stock (1996) asymptotic 1%, 5%, 10% critical values of DF-GLS."""

_NG_PERRON_CRITICAL: Final[dict[str, dict[str, tuple[float, float, float]]]] = {
    "c": {
        "MZa": (-13.8, -8.1, -5.7),
        "MZt": (-2.58, -1.98, -1.62),
        "MSB": (0.174, 0.233, 0.275),
        "MPT": (1.78, 3.17, 4.45),
    },
    "ct": {
        "MZa": (-23.8, -17.3, -14.2),
        "MZt": (-3.42, -2.91, -2.62),
        "MSB": (0.143, 0.168, 0.185),
        "MPT": (4.03, 5.48, 6.67),
    },
}
"""Ng and Perron (2001) Table 1: asymptotic 1%, 5%, 10% critical values, GLS-detrended."""

_GLS_DETREND_C: Final[dict[str, float]] = {"c": -7.0, "ct": -13.5}
"""Elliott-Rothenberg-Stock local-to-unity constants for GLS detrending."""

_ZIVOT_ANDREWS_CRITICAL: Final[dict[str, tuple[float, float, float]]] = {
    "c": (-5.34, -4.80, -4.58),
    "t": (-4.93, -4.42, -4.11),
    "ct": (-5.57, -5.08, -4.82),
}
"""Zivot and Andrews (1992) 1%, 5%, 10% critical values by break model.

``"c"`` breaks the intercept (their model A), ``"t"`` the trend slope
(model B), ``"ct"`` both (model C).
"""

_CUSUM_BOUNDARY: Final[dict[str, float]] = {"1%": 1.143, "5%": 0.948, "10%": 0.850}
"""Brown, Durbin and Evans (1975) multipliers of the CUSUM boundary."""

_TURNING_POINT_RULES: Final[dict[str, tuple[int, int, int]]] = {
    "quarterly": (2, 2, 5),
    "monthly": (5, 6, 15),
}
"""``(window, min_phase, min_cycle)`` of Harding-Pagan (quarterly) and Bry-Boschan (monthly)."""
