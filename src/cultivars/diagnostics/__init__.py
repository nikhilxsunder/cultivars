# filepath: /src/cultivars/diagnostics/__init__.py
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
"""Diagnostics: what a series is before a model is fitted, and what a fit left behind.

A time-series model is a set of claims about the data, and the
diagnostics are how those claims are checked. They are read in the order
a specification is built. Whether a series carries a stochastic trend
decides between a level model, a difference, and a cointegrating system
(:mod:`~cultivars.diagnostics.unit_roots`); whether its seasonal pattern
is fixed or wandering decides between dummies and a seasonal difference
(:mod:`~cultivars.diagnostics.seasonality`); when several integrated
series are in hand, how many stationary combinations they share decides
the rank of the error-correction model
(:mod:`~cultivars.diagnostics.cointegration`); whether a series has
long memory refines the integer question of a unit root into a
fractional one (:mod:`~cultivars.diagnostics.long_memory`); whether one
regression held over the whole sample, or broke, and where
(:mod:`~cultivars.diagnostics.breaks`); and whether a linear
autoregression is wrong in the specific way a threshold or
smooth-transition model would fix
(:mod:`~cultivars.diagnostics.nonlinearity`). Underneath them all is one
record type (:mod:`~cultivars.diagnostics.hypothesis`), so that a study
which collects a dozen verdicts holds one kind of object with one
``reject`` and one ``summary``, and the tests a fitted model runs on its
own residuals, Granger causality, portmanteau, normality, ARCH, return
the same records from the result rather than from here.

The package has one rule that every module keeps: the null distribution
is the one the test needs. Where the literature supplies a response
surface or a dense table the p-value is read from it; where a limit is
a functional of Brownian motion with no table at the trimming, period,
or dimension in hand, the limit is simulated at the values in hand;
where only three asymptotic critical values exist the record carries
them and no p-value, and refuses a level it does not have. A verdict
never comes from a distribution that is merely nearby.

Each module is a leaf: names are imported from the module, not from
this package, and every producer takes the series and the specification
it needs, fitting any auxiliary regression inside, so the null is
always exactly what the docstring says it is.

Layout. :mod:`~cultivars.diagnostics.hypothesis` names the records,
:class:`~cultivars.diagnostics.hypothesis.HypothesisTest` and its
:class:`~cultivars.diagnostics.hypothesis.TabulatedTest`,
:class:`~cultivars.diagnostics.hypothesis.ChiSquaredTest`,
:class:`~cultivars.diagnostics.hypothesis.WaldTest`, and
:class:`~cultivars.diagnostics.hypothesis.LikelihoodRatioTest`.
:mod:`~cultivars.diagnostics.unit_roots` is the Dickey-Fuller line,
KPSS, and Zivot-Andrews with the
:class:`~cultivars.diagnostics.unit_roots.UnitRootTest` record;
:mod:`~cultivars.diagnostics.seasonality` is HEGY and Canova-Hansen;
:mod:`~cultivars.diagnostics.cointegration` is the public home of the
Johansen rank record the error-correction models produce;
:mod:`~cultivars.diagnostics.long_memory` is GPH, local Whittle, and
exact local Whittle; :mod:`~cultivars.diagnostics.breaks` is the
Andrews, Bai-Perron, and CUSUM tests; and
:mod:`~cultivars.diagnostics.nonlinearity` is Teräsvirta, Tsay, RESET,
Hansen, and BDS. Convergence diagnostics for a sampler's draws live
with the Bayesian workflow in :mod:`~cultivars.bayes.chains`, and
forecast-comparison tests with the forecast evaluation in
:mod:`~cultivars.forecast.comparison`.

Example:
    Classify a series before modelling it: integrated at the zero
    frequency, with a stable seasonal pattern, and no break:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = np.cumsum(rng.standard_normal(200)) + np.tile([1.0, -0.5, 2.0, -2.5], 50)
    >>> unit_roots.adf(y).reject(), unit_roots.kpss(y).reject()
    (False, True)
    >>> root = seasonality.hegy(y, period=4, replications=500, seed=0)
    >>> root.reject(), all(c.reject() for c in root.companions)
    (False, True)
    >>> breaks.sup_wald(np.diff(y), seed=0).reject()
    False
"""

from . import breaks, cointegration, hypothesis, long_memory, nonlinearity, seasonality, unit_roots

__all__ = [
    "breaks",
    "cointegration",
    "hypothesis",
    "long_memory",
    "nonlinearity",
    "seasonality",
    "unit_roots",
]
