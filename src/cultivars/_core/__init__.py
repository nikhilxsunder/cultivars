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

from ._defaults import _D_MAX, _DEFAULT_MAX_ITER, _LOG_2PI, _SQRT_2_OVER_PI, _DEFAULT_TRUNCATION, _DEFAULT_TOL, _DEFAULT_STARTS, _DEFAULT_TRIM, _DEFAULT_GRID
from ._estimators import local_whittle_d, ols, ewma_mean_square, concentrated_gaussian, ergodic_distribution
from ._matrices import companion_matrix, conditional_design, deterministic_columns, n_deterministic, lag_matrix, psd_sqrt
from ._polynomials import expand_ar, expand_ma
from ._reparam import pack_stationary, unpack_stationary, softplus, inv_softplus
from ._transforms import combined_difference, fractional_difference, fractional_difference_weights
from ._validators import (
    validate_aligned,
    validate_choice,
    validate_endog,
    validate_exog,
    validate_open_interval,
    validate_order,
    validate_order_tuple,
)
from ._types import Trend, Vol, Mean, Transition

__all__ = [
    "_DEFAULT_MAX_ITER",
    "_D_MAX",
    "_LOG_2PI",
    "companion_matrix",
    "conditional_design",
    "n_deterministic",
    "pack_stationary",
    "unpack_stationary",
]
