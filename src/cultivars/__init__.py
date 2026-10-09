# filepath: /src/cultivars/__init__.py
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
"""Cultivars: research-grade time series econometrics, from ARIMA to state space.

The package covers the working range of applied macroeconometrics and
financial time series: univariate conditional-mean and
conditional-variance models, vector autoregressions in reduced,
structural, Bayesian, large-dimensional and nonlinear forms, the
state-space engines beneath them, the diagnostics that precede a fit
and the forecast evaluation that follows one, frequency-domain and
wavelet views of both data and fitted systems, and the Bayesian
workflow around any sampler. The workflow the layout follows runs in
the order an analysis does. Diagnose the series -- unit roots,
cointegration, breaks, seasonality, nonlinearity -- before choosing a
model; fit the model and read its result, whose summary states its own
caveats; evaluate the forecast against outcomes and against rivals;
and, where the question is one of frequency or of a hidden state, read
the same fitted object through the spectral and state-space views
rather than fitting something else.

The package has one rule that every subpackage keeps: a fitted result
is a record, not a handle. Every ``fit`` returns a frozen dataclass
carrying the estimates, the data needed to forecast or simulate from
them, and a ``summary`` whose notes say what the numbers assume; the
model object that produced it holds only the specification. That is
what lets a result be handed to any consumer that reads its surface --
a backtest, a spectral density, a structural identification, a model
comparison -- without the consumer knowing which model fitted it, and
what keeps a number from a model and a number from the data as
distinct types wherever the two could be confused.

Each subpackage is a namespace of leaf modules, and names are imported
from the leaf: ``from cultivars.univariate.box_jenkins import ARIMA``,
``from cultivars.multivariate.reduced_form.vector_autoregression import
VAR``. Nothing below the subpackage level is re-exported here or in the
subpackage inits, so an import path names the model family, the
specification, and the class, and reads as the specification it is.

Layout. :mod:`~cultivars.univariate` holds the single-series models,
:mod:`~cultivars.multivariate` the vector models in five families
(``reduced_form``, ``structural``, ``large_dim``, ``nonlinear``,
``regime_switching``), and :mod:`~cultivars.state_space` the
linear-Gaussian, switching and nonlinear engines both build on.
:mod:`~cultivars.diagnostics` is what to check before and after a fit,
:mod:`~cultivars.forecast` how to grade a forecast,
:mod:`~cultivars.spectral` the frequency-domain views and filters, and
:mod:`~cultivars.bayes` the priors, chain diagnostics, posterior checks,
evidence and combination that surround a sampler.
:mod:`~cultivars.typing` holds the categorical option aliases,
:mod:`~cultivars.exceptions` the error hierarchy every module raises
from, and :mod:`~cultivars.engine` exposes the private ``_core`` and
``_internals`` layers for anyone extending the package.

Example:
    Diagnose, difference, fit, and evaluate, each step from its leaf
    module:

    >>> import numpy as np
    >>> from cultivars.diagnostics.unit_roots import adf
    >>> from cultivars.univariate.box_jenkins import ARIMA
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> from cultivars.forecast.backtest import Backtest
    >>> rng = np.random.default_rng(0)
    >>> level = np.cumsum(0.1 + rng.standard_normal(200))
    >>> bool(adf(level).reject(alpha=0.05)), bool(adf(np.diff(level)).reject(alpha=0.05))
    (False, True)
    >>> arima = ARIMA(level, order=(1, 1, 0)).fit()
    >>> arima.order, bool(0.0 < arima.beta[0] < 0.2)
    ((1, 1, 0), True)
    >>> panel = np.column_stack([np.diff(level), rng.standard_normal(199)])
    >>> VAR(panel, order=1).fit().forecast(4).shape
    (4, 2)
    >>> Backtest(panel, lambda w: VAR(w, order=1).fit(), start=150).run().n_origins
    49
"""

from __future__ import annotations

from . import (
    bayes,
    data,
    diagnostics,
    engine,
    exceptions,
    forecast,
    multivariate,
    spectral,
    state_space,
    summary,
    typing,
    univariate,
)

__all__ = [
    "bayes",
    "data",
    "diagnostics",
    "engine",
    "exceptions",
    "forecast",
    "multivariate",
    "spectral",
    "state_space",
    "summary",
    "typing",
    "univariate",
]
