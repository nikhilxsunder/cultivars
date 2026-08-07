# filepath: /src/cultivars/core/__init__.py
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

"""Primitive layer: constants, value objects, contracts, and pure functions.

Nothing in this subpackage knows what a model is. Every symbol here has more
than one consumer -- that is the test for belonging in ``_core`` rather than
in ``_internals``, which houses mono-consumer mechanism plumbing.
"""

from __future__ import annotations

from ._companion import companion_from_polynomial, companion_matrix, selector_matrix
from ._containers import InformationCriteria, information_criteria
from ._defaults import (
    _D_MAX,
    _DEFAULT_GRID,
    _DEFAULT_MAX_ITER,
    _DEFAULT_STARTS,
    _DEFAULT_TOL,
    _DEFAULT_TRIM,
    _DEFAULT_TRUNCATION,
    _GPH_EXPONENT,
    _LOG_2PI,
    _PACF_CLIP,
    _PERSISTENCE_MAX,
    _ROW_SUM_ATOL,
    _SCHEMA_VERSION,
    _SQRT_2_OVER_PI,
    _STABILITY_TOL,
    _TINY,
    _WHITTLE_EXPONENT,
)
from ._design import (
    css_design,
    deterministic_columns,
    expand_ar,
    expand_ma,
    lag_matrix,
    n_deterministic,
)
from ._lag import LagPolynomial
from ._linalg import ergodic_distribution, ols, psd_sqrt
from ._protocols import (
    FittedResult,
    Forecaster,
    MeanModelResult,
    TimeSeriesModel,
    VolatilityResult,
)
from ._reparam import (
    coeffs_to_pacf,
    pacf_to_coeffs,
    pack_stationary,
    unpack_stationary,
)
from ._spectral import bandwidth, periodogram
from ._stability import (
    STABLE_TRIVIAL,
    StabilityResult,
    assess_stability,
    assess_stability_from_companion,
    is_invertible,
    is_stationary,
)
from ._transforms import (
    Standardized,
    difference,
    log_difference,
    log_transform,
    seasonal_difference,
    standardize,
    undifference,
)
from ._types import (
    Activation,
    LongMemoryMethod,
    Mean,
    Method,
    Transition,
    Trend,
    VolatilityResult,
)

__all__ = [
    "STABLE_TRIVIAL",
    "Activation",
    "FittedResult",
    "Forecaster",
    "InformationCriteria",
    "LagPolynomial",
    "LongMemoryMethod",
    "Mean",
    "MeanModelResult",
    "Method",
    "StabilityResult",
    "Standardized",
    "TimeSeriesModel",
    "Transition",
    "Trend",
    "Vol",
    "VolatilityResult",
    "assess_stability",
    "assess_stability_from_companion",
    "bandwidth",
    "coeffs_to_pacf",
    "companion_from_polynomial",
    "companion_matrix",
    "css_design",
    "deterministic_columns",
    "difference",
    "ergodic_distribution",
    "expand_ar",
    "expand_ma",
    "information_criteria",
    "is_invertible",
    "is_stationary",
    "lag_matrix",
    "log_difference",
    "log_transform",
    "n_deterministic",
    "ols",
    "pacf_to_coeffs",
    "pack_stationary",
    "periodogram",
    "psd_sqrt",
    "seasonal_difference",
    "selector_matrix",
    "standardize",
    "undifference",
    "unpack_stationary",
]
