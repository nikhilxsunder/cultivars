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

from ._defaults import (
    _D_MAX,
    _DEFAULT_GRID,
    _DEFAULT_MAX_ITER,
    _DEFAULT_STARTS,
    _DEFAULT_TOL,
    _DEFAULT_TRIM,
    _DEFAULT_TRUNCATION,
    _LOG_2PI,
    _SQRT_2_OVER_PI,
)
from ._estimators import (
    concentrated_gaussian,
    ergodic_distribution,
    ewma_mean_square,
    local_whittle_d,
    ols,
)
from ._matrices import (
    companion_matrix,
    conditional_design,
    deterministic_columns,
    lag_matrix,
    n_deterministic,
    psd_sqrt,
)
from ._polynomials import expand_ar, expand_ma
from ._reparam import inv_softplus, pack_stationary, softplus, unpack_stationary
from ._transforms import combined_difference, fractional_difference, fractional_difference_weights
from ._types import Mean, Transition, Trend, Vol
from ._validators import (
    validate_aligned,
    validate_choice,
    validate_endog,
    validate_exog,
    validate_open_interval,
    validate_order,
    validate_order_tuple,
)

__all__ = [
    "_DEFAULT_GRID",
    "_DEFAULT_MAX_ITER",
    "_DEFAULT_STARTS",
    "_DEFAULT_TOL",
    "_DEFAULT_TRIM",
    "_DEFAULT_TRUNCATION",
    "_D_MAX",
    "_LOG_2PI",
    "_SQRT_2_OVER_PI",
    "InformationCriteria",
    "Mean",
    "Transition",
    "Trend",
    "Vol",
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
    "softplus",
    "unpack_stationary",
    "validate_aligned",
    "validate_choice",
    "validate_endog",
    "validate_exog",
    "validate_open_interval",
    "validate_order",
    "validate_order_tuple",
]
