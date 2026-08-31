# filepath: /src/cultivars/univariate/__init__.py
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
"""Cultivars univariate time series module."""

from .autoregression import AR, ARResult
from .box_jenkins import ARIMA, ARIMAX, ARMA, SARIMA, SARIMAX, ARMAResult
from .conditional_variance import (
    ARGARCH,
    ARMAEGARCH,
    ARMAFIGARCH,
    ARMAGARCH,
    ARMAGJR,
    EGARCH,
    FIGARCH,
    GARCH,
    GJR,
    ARMAFIGARCHResult,
    ARMAGARCHResult,
    FIGARCHResult,
    GARCHResult,
)
from .fractional_integration import ARFIMA, ARFIMAResult
from .markov_switching import MSAR, MSARResult
from .mean_function import ARNN, TARNN, ARNNResult, TARNNResult
from .smooth_transition import ESTAR, LSTAR, STARResult
from .threshold import SETAR, TAR, SETARResult

__all__ = [
    "AR",
    "ARFIMA",
    "ARGARCH",
    "ARIMA",
    "ARIMAX",
    "ARMA",
    "ARMAEGARCH",
    "ARMAFIGARCH",
    "ARMAGARCH",
    "ARMAGJR",
    "ARNN",
    "EGARCH",
    "ESTAR",
    "FIGARCH",
    "GARCH",
    "GJR",
    "LSTAR",
    "MSAR",
    "SARIMA",
    "SARIMAX",
    "SETAR",
    "TAR",
    "TARNN",
    "ARFIMAResult",
    "ARMAFIGARCHResult",
    "ARMAGARCHResult",
    "ARMAResult",
    "ARNNResult",
    "ARResult",
    "FIGARCHResult",
    "GARCHResult",
    "MSARResult",
    "SETARResult",
    "STARResult",
    "TARNNResult",
]
