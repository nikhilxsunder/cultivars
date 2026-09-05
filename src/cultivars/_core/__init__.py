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
    _mixed_frequency_system,
    to_pandas_frame,
    to_polars_frame,
)
from ._defaults import (
    _CAPACITY_WARNING,
    _D_MAX,
    _DEFAULT_ALPHA,
    _DEFAULT_GRID,
    _DEFAULT_MAX_ITER,
    _DEFAULT_STARTS,
    _DEFAULT_TOL,
    _DEFAULT_TRIM,
    _DEFAULT_TRUNCATION,
    _KSC_MEAN,
    _KSC_VAR,
    _LOG_2PI,
    _OFFSET,
    _PENALTY,
    _ROW_SUM_ATOL,
    _SCHEMA_VERSION,
    _SQRT_2_OVER_PI,
    _TINY,
)
from ._estimators import (
    _cumulant_slices,
    _gaussian_negloglik,
    concentrated_gaussian,
    ergodic_distribution,
    ewma_mean_square,
    local_whittle_d,
    minnesota_scales,
    ols,
    simulate_cointegration_null,
)
from ._mappings import (
    _LEVELS_TREND,
    _UNRESTRICTED_TREND,
)
from ._matrices import (
    _face_projectors,
    _long_run_matrix,
    _lower_cholesky,
    _null_basis,
    _orthogonal_from_angles,
    _sphere_extrema,
    companion_matrix,
    conditional_design,
    deterministic_columns,
    lag_matrix,
    link_matrix,
    n_deterministic,
    psd_sqrt,
    trailing_lag,
)
from ._notes import (
    _AGGREGATION_NOTE,
    _CHOLESKY_NOTE,
    _CONDITIONAL_REFUSAL,
    _HR_CONDITIONAL_NOTE,
    _MIDAS_CONDITIONAL_NOTE,
    _NARRATIVE_NOTE,
    _NO_CLOSED_SYSTEM,
    _PARTIAL_IDENTIFICATION_NOTE,
    _SET_BOUNDS_NOTE,
    _SIGN_QUANTILE_NOTE,
    _UNIT_SHOCK_NOTE,
    _UNSTABLE_NOTE,
    _VARMA_IDENTIFICATION_NOTE,
)
from ._polynomials import (
    _aggregation_weights,
    _midas_weights,
    _midas_windows,
    _nelson_siegel_loadings,
    _projection_scores,
    aggregation_weights,
    expand_ar,
    expand_ma,
)
from ._protocols import ClosedSystemResult, Identification, StructuralResult
from ._recursions import (
    _arch_infinity_variance,
    _arch_infinity_weights,
    _linear_variance_recursion,
    _log_variance_recursion,
)
from ._reparam import inv_softplus, pack_stationary, sigmoid, softplus, unpack_stationary
from ._resolvers import _resolve_ordering
from ._rotations import _accepted_rotations, _haar_rotation, _narrative_rotations
from ._samplers import _draw_inverse_gamma, _draw_inverse_wishart, _draw_mixture_indicators
from ._transforms import combined_difference, fractional_difference, fractional_difference_weights
from ._types import (
    CointegrationTrend,
    Frequency,
    FunctionalBasis,
    Mean,
    Method,
    OptimizerMethod,
    OptimizerOptions,
    PanelEffects,
    ProbabilityType,
    Regime,
    Transition,
    Trend,
    Vol,
)
from ._validators import (
    _validate_curves,
    _validate_impact_pattern,
    _validate_narrative_events,
    _validate_observed,
    _validate_ordering,
    _validate_regimes,
    _validate_sign_patterns,
    _validate_wide_panel,
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
    validate_weights,
)

__all__ = [
    "_AGGREGATION_NOTE",
    "_CAPACITY_WARNING",
    "_CHOLESKY_NOTE",
    "_CONDITIONAL_REFUSAL",
    "_DEFAULT_ALPHA",
    "_DEFAULT_GRID",
    "_DEFAULT_MAX_ITER",
    "_DEFAULT_STARTS",
    "_DEFAULT_TOL",
    "_DEFAULT_TRIM",
    "_DEFAULT_TRUNCATION",
    "_D_MAX",
    "_HR_CONDITIONAL_NOTE",
    "_KSC_MEAN",
    "_KSC_VAR",
    "_LEVELS_TREND",
    "_LOG_2PI",
    "_MIDAS_CONDITIONAL_NOTE",
    "_NARRATIVE_NOTE",
    "_NO_CLOSED_SYSTEM",
    "_OFFSET",
    "_PARTIAL_IDENTIFICATION_NOTE",
    "_PENALTY",
    "_ROW_SUM_ATOL",
    "_SCHEMA_VERSION",
    "_SET_BOUNDS_NOTE",
    "_SIGN_QUANTILE_NOTE",
    "_SQRT_2_OVER_PI",
    "_TINY",
    "_UNIT_SHOCK_NOTE",
    "_UNRESTRICTED_TREND",
    "_UNSTABLE_NOTE",
    "_VARMA_IDENTIFICATION_NOTE",
    "ClosedSystemResult",
    "CointegrationTrend",
    "Frequency",
    "FunctionalBasis",
    "Identification",
    "InformationCriteria",
    "Mean",
    "Method",
    "OptimizerMethod",
    "OptimizerOptions",
    "PanelEffects",
    "ProbabilityType",
    "Regime",
    "StructuralResult",
    "SummaryTable",
    "Transition",
    "Trend",
    "Vol",
    "_ForwardPass",
    "_accepted_rotations",
    "_aggregation_weights",
    "_arch_infinity_variance",
    "_arch_infinity_weights",
    "_cumulant_slices",
    "_draw_inverse_gamma",
    "_draw_inverse_wishart",
    "_draw_mixture_indicators",
    "_face_projectors",
    "_gaussian_negloglik",
    "_haar_rotation",
    "_linear_variance_recursion",
    "_log_variance_recursion",
    "_long_run_matrix",
    "_lower_cholesky",
    "_mean_label",
    "_midas_weights",
    "_midas_windows",
    "_mixed_frequency_system",
    "_narrative_rotations",
    "_nelson_siegel_loadings",
    "_null_basis",
    "_orthogonal_from_angles",
    "_projection_scores",
    "_resolve_ordering",
    "_sphere_extrema",
    "_validate_curves",
    "_validate_impact_pattern",
    "_validate_narrative_events",
    "_validate_observed",
    "_validate_ordering",
    "_validate_regimes",
    "_validate_sign_patterns",
    "_validate_wide_panel",
    "aggregation_weights",
    "combined_difference",
    "companion_matrix",
    "concentrated_gaussian",
    "conditional_design",
    "deterministic_columns",
    "ergodic_distribution",
    "ewma_mean_square",
    "expand_ar",
    "expand_ma",
    "face_projectors",
    "fractional_difference",
    "fractional_difference_weights",
    "inv_softplus",
    "lag_matrix",
    "link_matrix",
    "local_whittle_d",
    "minnesota_scales",
    "n_deterministic",
    "null_basis",
    "ols",
    "pack_stationary",
    "psd_sqrt",
    "sigmoid",
    "simulate_cointegration_null",
    "softplus",
    "sphere_extrema",
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
    "validate_weights",
]
