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

from ._containers import InformationCriteria, SummaryTable, _ForwardPass
from ._converters import (
    _mean_label,
    to_pandas_frame,
    to_polars_frame,
)
from ._defaults import (
    _CAPACITY_WARNING,
    _CHOLESKY_NOTE,
    _D_MAX,
    _DEFAULT_ALPHA,
    _DEFAULT_GRID,
    _DEFAULT_MAX_ITER,
    _DEFAULT_STARTS,
    _DEFAULT_TOL,
    _DEFAULT_TRIM,
    _DEFAULT_TRUNCATION,
    _LOG_2PI,
    _PENALTY,
    _ROW_SUM_ATOL,
    _SCHEMA_VERSION,
    _SQRT_2_OVER_PI,
    _TINY,
    _UNSTABLE_NOTE,
)
from ._estimators import (
    _gaussian_negloglik,
    concentrated_gaussian,
    ergodic_distribution,
    ewma_mean_square,
    local_whittle_d,
    ols,
    simulate_cointegration_null,
)
from ._mappings import (
    _LEVELS_TREND,
    _UNRESTRICTED_TREND,
)
from ._matrices import (
    companion_matrix,
    conditional_design,
    deterministic_columns,
    lag_matrix,
    n_deterministic,
    psd_sqrt,
    trailing_lag,
)
from ._polynomials import expand_ar, expand_ma
from ._recursions import (
    _arch_infinity_variance,
    _arch_infinity_weights,
    _linear_variance_recursion,
    _log_variance_recursion,
)
from ._reparam import inv_softplus, pack_stationary, sigmoid, softplus, unpack_stationary
from ._transforms import combined_difference, fractional_difference, fractional_difference_weights
from ._types import (
    CointegrationTrend,
    Mean,
    Method,
    OptimizerMethod,
    OptimizerOptions,
    PanelEffects,
    ProbabilityType,
    Transition,
    Trend,
    Vol,
)
from ._validators import (
    validate_aligned,
    validate_choice,
    validate_endog,
    validate_endog_matrix,
    validate_exog,
    validate_exog_matrix,
    validate_open_interval,
    validate_order,
    validate_order_tuple,
    validate_panel,
    validate_transition,
)

__all__ = [
    "_CAPACITY_WARNING",
    "_CHOLESKY_NOTE",
    "_DEFAULT_ALPHA",
    "_DEFAULT_GRID",
    "_DEFAULT_MAX_ITER",
    "_DEFAULT_STARTS",
    "_DEFAULT_TOL",
    "_DEFAULT_TRIM",
    "_DEFAULT_TRUNCATION",
    "_D_MAX",
    "_LEVELS_TREND",
    "_LOG_2PI",
    "_NO_STDERR_NOTE",
    "_PENALTY",
    "_ROW_SUM_ATOL",
    "_SCHEMA_VERSION",
    "_SQRT_2_OVER_PI",
    "_TINY",
    "_UNRESTRICTED_TREND",
    "_UNSTABLE_NOTE",
    "CointegrationTrend",
    "InformationCriteria",
    "Mean",
    "Method",
    "OptimizerMethod",
    "OptimizerOptions",
    "PanelEffects",
    "ProbabilityType",
    "SummaryTable",
    "Transition",
    "Trend",
    "Vol",
    "_ForwardPass",
    "_arch_infinity_variance",
    "_arch_infinity_weights",
    "_gaussian_negloglik",
    "_linear_variance_recursion",
    "_log_variance_recursion",
    "_mean_label",
    "combined_difference",
    "companion_matrix",
    "concentrated_gaussian",
    "conditional_design",
    "deterministic_columns",
    "ergodic_distribution",
    "ewma_mean_square",
    "expand_ar",
    "expand_ma",
    "fractional_difference",
    "fractional_difference_weights",
    "inv_softplus",
    "lag_matrix",
    "local_whittle_d",
    "n_deterministic",
    "ols",
    "pack_stationary",
    "psd_sqrt",
    "sigmoid",
    "simulate_cointegration_null",
    "softplus",
    "to_pandas_frame",
    "to_polars_frame",
    "trailing_lag",
    "unpack_stationary",
    "validate_aligned",
    "validate_choice",
    "validate_endog",
    "validate_endog_matrix",
    "validate_exog",
    "validate_exog_matrix",
    "validate_open_interval",
    "validate_order",
    "validate_order_tuple",
    "validate_panel",
    "validate_transition",
]
