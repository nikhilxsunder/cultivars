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
"""Univariate models: one series, its mean, its variance, and its components.

Every model here takes one observed series and answers one of three
questions about it. What drives its conditional mean: a linear
autoregression (:mod:`~cultivars.univariate.autoregression`), the
Box-Jenkins family with differencing, seasonality and regressors
(:mod:`~cultivars.univariate.box_jenkins`), long memory through a
fractional difference (:mod:`~cultivars.univariate.fractional_integration`),
a regime that switches on an observed variable, abruptly
(:mod:`~cultivars.univariate.threshold`) or smoothly
(:mod:`~cultivars.univariate.smooth_transition`), a regime that switches
on a latent Markov chain (:mod:`~cultivars.univariate.regime_switching`),
or a learned nonlinear function of the lags
(:mod:`~cultivars.univariate.mean_function`). What drives its
conditional variance: a deterministic recursion on past shocks
(:mod:`~cultivars.univariate.conditional_variance`) or a latent process
with its own innovation (:mod:`~cultivars.univariate.stochastic_volatility`).
Or what it is made of: a trend, a cycle, a seasonal and an irregular,
each with a likelihood and a band
(:mod:`~cultivars.univariate.unobserved_components`).

The models share one shape. A model object holds the specification and
nothing else; ``fit`` (or ``sample``, for the Bayesian estimators)
returns an immutable record that carries the estimates, the fit
statistics and the aligned per-observation series, and that record is
the only thing a consumer ever reads. Every record renders the same
way (``print``, a notebook cell, ``summary().to_pandas()``), exposes its
series the same way (``to_pandas``, ``to_polars``), ranks against other
records fitted on the same sample with ``compare`` and
``information_criteria``, and -- where the model has a state-space
reading -- emits it as a :mod:`~cultivars.state_space` system that can
filter data it was not estimated on. Refusals are typed: a test whose
null distribution does not exist for the model raises rather than
returning a number, and a comparison across different effective samples
raises rather than ranking incomparable likelihoods.

Names are imported from the leaf module, never from here --
``from cultivars.univariate.box_jenkins import ARIMA`` -- so an import
path names the family, the specification and the class. This package
exposes only the modules.

Layout. The public classes are thin: each validates its specification
on a model base in ``cultivars._internals`` and assembles its public
record from a packed fit, while the objectives, solvers, simulators,
system builders and the shared result mixins live in ``_internals`` and
the numerical primitives in ``_core``. Diagnostics to run before a fit
are in :mod:`~cultivars.diagnostics`; the forecast evaluation that reads
these records is in :mod:`~cultivars.forecast`; the vector counterparts
of the mean models are in :mod:`~cultivars.multivariate`.

Example:
    A linear and a threshold autoregression on one sample, ranked by
    BIC through the shared comparison surface:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros(400)
    >>> for t in range(1, 400):
    ...     phi = 0.6 if y[t - 1] <= 0.0 else -0.4
    ...     y[t] = phi * y[t - 1] + rng.standard_normal()
    >>> linear = autoregression.AR(y, order=1).fit()
    >>> hard = threshold.SETAR(y, order=1, delay=1).fit()
    >>> table = linear.compare(hard, criterion="bic")
    >>> [row[0] for row in table.rows]
    ['SETAR(1, d=1)', 'AR(1)']
    >>> list(hard.to_pandas().columns)
    ['observed', 'fitted', 'resid', 'threshold_variable', 'regime_weight']
"""

from . import (
    autoregression,
    box_jenkins,
    conditional_variance,
    fractional_integration,
    mean_function,
    regime_switching,
    smooth_transition,
    stochastic_volatility,
    threshold,
    unobserved_components,
)

__all__ = [
    "autoregression",
    "box_jenkins",
    "conditional_variance",
    "fractional_integration",
    "mean_function",
    "regime_switching",
    "smooth_transition",
    "stochastic_volatility",
    "threshold",
    "unobserved_components",
]
