# filepath: /src/cultivars/multivariate/nonlinear/smooth_transition.py
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
"""Nonlinear multivariate models: coefficients that depend on a state, a quantile, or the date.

A linear VAR has one coefficient stack for every date and every part of
the distribution. Every model here lets that stack move, and they differ
in what it moves with. With an observable state: abruptly, two complete
systems split at a threshold on a named lag or an external series
(:mod:`~cultivars.multivariate.nonlinear.threshold`); smoothly, a convex
blend of two systems weighted by a logistic or exponential function of
the state (:mod:`~cultivars.multivariate.nonlinear.smooth_transition`); or
nonparametrically, every coefficient an unknown smooth function of the
state estimated by local-linear kernel regression, which is also the
specification check on the two parametric forms
(:mod:`~cultivars.multivariate.nonlinear.functional_coefficient`). With
the quantile level: one coefficient stack per conditional quantile, so
the tails get their own dynamics
(:mod:`~cultivars.multivariate.nonlinear.quantile`). With the date:
coefficients that drift as random walks, with or without stochastic
volatility, estimated as Bayesian posteriors
(:mod:`~cultivars.multivariate.nonlinear.time_varying`). Parameters that
switch with a latent Markov chain are the other kind of regime model and
live in :mod:`~cultivars.multivariate.regime_switching`.

The models share one shape and three refusals. A model object holds the
specification; ``fit`` returns an immutable record that renders the same
way everywhere (``print``, a notebook cell, ``summary()``), and every
propagation question must name where it is asked -- a regime, a state
value, a quantile level, a date -- because a nonlinear system has no
single companion matrix; the answer it gets holds that regime, state,
level or date frozen, and the records say so. First refusal: no result
here satisfies the closed-system contract or offers a chi-squared test
for nonlinearity, because the nuisance parameters are not identified
under linearity (Davies); the likelihood-based results rank one another
by information criteria instead. Second: the kernel, quantile and
Bayesian results report no likelihood, parameter count or information
criteria, because a smoother, a check loss and a posterior have none.
Third: forecasts are one step, or a skeleton iteration labelled as such,
because the multi-step conditional mean of a nonlinear model is a
simulation problem these records do not fake. Standard errors are not
yet offered by any estimator in this package.

Names are imported from the leaf module, never from here --
``from cultivars.multivariate.nonlinear.threshold import TVAR`` -- so an
import path names the family, the specification and the class. This
package exposes only the modules.

Layout. The public classes are thin: the three observed-state models
validate on ``_ObservedRegimeVectorModel`` in ``cultivars._internals``
and its threshold, smooth-transition and functional-coefficient
extensions, the quantile model on ``_QuantileVectorAutoRegressionModel``,
the time-varying models on ``_TimeVaryingVectorAutoRegressionModel``;
each assembles its public record from a packed fit. The grid search, the
concentrated objective and its solver, the kernel engine, the HiGHS
programs and the Gibbs sampler live in ``_internals`` and ``_core``; the
state-space substrate the time-varying sampler runs on is
:mod:`~cultivars.state_space`. The linear system every model here
generalizes is :mod:`~cultivars.multivariate.reduced_form`.

Example:
    One sample, three readings of its state dependence: the hard split,
    the smooth blend, and the kernel curve that adjudicates between them.
    The parametric pair rank by BIC through the shared comparison
    surface; the kernel curve falls from the lower regime's value toward
    the upper regime's as the state rises, which is the smooth form's
    account:

    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> y = np.zeros((400, 2))
    >>> for t in range(1, 400):
    ...     g = 1.0 / (1.0 + np.exp(-2.0 * y[t - 1, 0]))
    ...     y[t] = ((1.0 - g) * 0.6 - g * 0.3) * y[t - 1] + rng.standard_normal(2)
    >>> hard = threshold.TVAR(y, order=1, transition_variable="y1", delay=1).fit()
    >>> smooth = smooth_transition.STVAR(y, order=1, transition_variable="y1").fit()
    >>> [row[0] for row in hard.compare(smooth, criterion="bic").rows]
    ['STVAR(1, logistic, d=1)', 'TVAR(1, d=1)']
    >>> model = functional_coefficient.FunctionalCoefficientVAR(
    ...     y, order=1, transition_variable="y1"
    ... )
    >>> curve = model.fit(n_grid=21).coefficient_curve("y1", "y1.L1")[:, 1]
    >>> bool(curve[0] > 0.4), bool(curve[-1] < 0.0), bool(np.all(np.diff(curve) < 0.0))
    (True, True, True)
"""

from . import functional_coefficient, quantile, smooth_transition, threshold, time_varying

__all__ = [
    "functional_coefficient",
    "quantile",
    "smooth_transition",
    "threshold",
    "time_varying",
]
