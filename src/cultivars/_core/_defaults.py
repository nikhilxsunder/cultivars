# filepath: /src/cultivars/_core/_defaults.py
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

"""Package-wide numeric constants and estimator defaults.

Every magic number in the package resolves here. Constants are private by
naming convention (leading underscore) because they are implementation
detail: changing one changes fitted output, so they are not public API.
"""

from __future__ import annotations

import numpy as np

_SCHEMA_VERSION: int = 1
"""Serialization schema version stamped onto every result object."""

_LOG_2PI: float = float(np.log(2.0 * np.pi))
"""``log(2 * pi)``; the Gaussian log-likelihood constant."""

_SQRT_2_OVER_PI: float = float(np.sqrt(2.0 / np.pi))
"""``E|z|`` for standard normal ``z``; the EGARCH asymmetry centering term."""

_PACF_CLIP: float = 0.999
"""Clip applied to partial autocorrelations before the arctanh transform."""

_D_MAX: float = 0.499
"""Upper bound on the fractional differencing parameter (stationary region)."""

_PERSISTENCE_MAX: float = 0.999
"""Upper bound on total variance persistence for a covariance-stationary fit."""

_ROW_SUM_ATOL: float = 1e-6
"""Absolute tolerance when checking that transition-matrix rows sum to one."""

_STABILITY_TOL: float = 1e-8
"""Default modulus tolerance for unit-root and explosive-root classification."""

_TINY: float = 1e-300
"""Floor guarding logarithms of mixture densities against underflow."""

_DEFAULT_TRUNCATION: int = 1000
"""Default ARCH(infinity) / AR(infinity) truncation lag."""

_DEFAULT_GRID: int = 300
"""Default number of candidate thresholds in a threshold grid search."""

_DEFAULT_TRIM: float = 0.15
"""Default fraction trimmed from each tail of a threshold grid."""

_DEFAULT_MAX_ITER: int = 500
"""Default maximum EM iterations."""

_DEFAULT_TOL: float = 1e-6
"""Default relative convergence tolerance for iterative estimators."""

_DEFAULT_STARTS: int = 10
"""Default number of random restarts for multimodal likelihoods."""

_GPH_EXPONENT: float = 0.5
"""Default bandwidth exponent ``m = floor(n ** 0.5)`` for the GPH estimator."""

_WHITTLE_EXPONENT: float = 0.65
"""Default bandwidth exponent for the local Whittle estimator."""

_TREND_WIDTH: dict[str, int] = {"n": 0, "c": 1, "ct": 2}
"""Number of deterministic columns implied by each trend specification."""
