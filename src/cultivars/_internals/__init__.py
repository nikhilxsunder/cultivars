# filepath: /src/cultivars/_internals/__init__.py
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

"""Protected mechanism layer: base classes and estimation engines.

Where ``_core`` holds multi-consumer primitives that know nothing about
models, this subpackage holds the machinery a specific model family needs:
its specification base, its raw ``_*Fit`` container, and its engine.

Nothing here imports from a public subpackage. Engines return ``_*Fit``
dataclasses rather than public result types, and the public result assembles
itself from one via a ``_from_fit`` classmethod -- which is what keeps the
dependency edges pointing one way.
"""

from __future__ import annotations

from ._arfima import (
    _ar_infinity,
    _ARFIMAFit,
    _fit_arfima,
    fractional_difference,
    fractional_difference_weights,
)
from ._arima import _arma_state_space, _fit_sarimax, _SARIMAXFit, _SARIMAXModel
from ._garch import _fit_figarch, _fit_garch, _GARCHFit, _GARCHModel
from ._linear_gaussian import LinearGaussianStateSpace
from ._mean_function import (
    MeanFunctionEngine,
    MeanPredictor,
    NumpyMLPEngine,
    _ARNNFit,
    _fit_arnn,
    _fit_tarnn,
    _MeanFunctionModel,
    _TARNNFit,
    _ThresholdMeanFunctionModel,
)
from ._models import (
    _ConditionalVarianceMixin,
    _MeanResult,
    _ModelBase,
    _Result,
    _StationarityMixin,
    _UnivariateModel,
)
from ._ms_ar import _MSARFit, _MSARModel
from ._regime_switching import (
    HamiltonFilterResult,
    KimSmootherResult,
    hamilton_filter,
    kim_smoother,
)
from ._state_space import FilterResult, SmootherResult, StateSpaceModel
from ._threshold import (
    _fit_star,
    _fit_threshold,
    _STARFit,
    _STARModel,
    _ThresholdFit,
    _ThresholdModel,
)
from ._univariate import _ARFit, _fit_ar_css, _fit_ar_exact, _forecast_ar

__all__ = [
    "FilterResult",
    "HamiltonFilterResult",
    "KimSmootherResult",
    "LinearGaussianStateSpace",
    "MeanFunctionEngine",
    "MeanPredictor",
    "NumpyMLPEngine",
    "SmootherResult",
    "StateSpaceModel",
    "_ARFIMAFit",
    "_ARFit",
    "_ARNNFit",
    "_ConditionalVarianceMixin",
    "_GARCHFit",
    "_GARCHModel",
    "_MSARFit",
    "_MSARModel",
    "_MeanFunctionModel",
    "_MeanResult",
    "_ModelBase",
    "_Result",
    "_SARIMAXFit",
    "_SARIMAXModel",
    "_STARFit",
    "_STARModel",
    "_StationarityMixin",
    "_TARNNFit",
    "_ThresholdFit",
    "_ThresholdMeanFunctionModel",
    "_ThresholdModel",
    "_UnivariateModel",
    "_ar_infinity",
    "_arma_state_space",
    "_fit_ar_css",
    "_fit_ar_exact",
    "_fit_arfima",
    "_fit_arnn",
    "_fit_figarch",
    "_fit_garch",
    "_fit_sarimax",
    "_fit_star",
    "_fit_tarnn",
    "_fit_threshold",
    "_forecast_ar",
    "fractional_difference",
    "fractional_difference_weights",
    "hamilton_filter",
    "kim_smoother",
]
