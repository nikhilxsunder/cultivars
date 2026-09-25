# filepath: /src/cultivars/bayes/evidence.py
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
r"""Bayesian workflow for the package's samplers: priors, chains, checks, evidence, combination.

The Bayesian models live with their frequentist counterparts, a
:class:`~cultivars.multivariate.large_dim.bayesian.BVAR` beside the
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`,
a :class:`~cultivars.univariate.stochastic_volatility.SV` among the
univariate models. This package holds what surrounds a fit rather than
the fit itself, in the order a Bayesian analysis runs. A prior is chosen
and stated as a record (:mod:`~cultivars.bayes.priors`); the sampler's
draws are checked for convergence before they are read as a posterior
(:mod:`~cultivars.bayes.chains`); the fitted model is run as a
data-generating machine and compared with the sample it was fitted to,
and the prior is run the same way before any sample is touched
(:mod:`~cultivars.bayes.checks`); competing models are weighed by
marginal likelihood, Bayes factor, and posterior model probability
(:mod:`~cultivars.bayes.evidence`); and, when no model deserves the
whole verdict, their predictive densities are combined by model
averaging or stacking (:mod:`~cultivars.bayes.combination`). Each module
is a leaf: names are imported from the module, not from this package,
and nothing here depends on which model produced the draws beyond the
result protocols in ``_core``.

Layout. :mod:`~cultivars.bayes.priors` is the one module with state: the
Minnesota family, the conjugate and independent Normal-Wishart pairings,
the dummy-observation priors, and the adaptive shrinkage hierarchies,
composable with ``+``. :mod:`~cultivars.bayes.chains` is
:func:`~cultivars.bayes.chains.rhat`, the split-chain effective sample
sizes, the Monte Carlo standard error, Geweke's test, and the
:func:`~cultivars.bayes.chains.convergence` record that bundles them.
:mod:`~cultivars.bayes.checks` is
:func:`~cultivars.bayes.checks.posterior_predictive_check`,
:func:`~cultivars.bayes.checks.prior_predictive_check`, and the
:class:`~cultivars.bayes.checks.PredictiveCheckTest` record they produce.
:mod:`~cultivars.bayes.evidence` is
:func:`~cultivars.bayes.evidence.marginal_likelihood`, which selects the
exact, Chib, harmonic-mean, or bridge estimate a result supports, and
:func:`~cultivars.bayes.evidence.compare`, which turns a set of them into
posterior model probabilities. :mod:`~cultivars.bayes.combination` is
:func:`~cultivars.bayes.combination.bayesian_model_average` and
:func:`~cultivars.bayes.combination.stacking`.

Example:
    State a prior, fit, check the chain, and read the evidence:

    >>> import numpy as np
    >>> from cultivars.multivariate.large_dim.bayesian import BVAR
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((120, 2))
    >>> for t in range(1, 120):
    ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
    >>> prior = priors.NormalInverseWishartPrior(tightness=0.1)
    >>> res = BVAR(y, order=1, prior=prior).fit(n_draws=400, seed=0)
    >>> evidence.marginal_likelihood(res).method
    'analytic'
    >>> checks.posterior_predictive_check(res, seed=0).adequate()
    True
"""

from . import chains, checks, combination, evidence, priors

__all__ = [
    "chains",
    "checks",
    "combination",
    "evidence",
    "priors",
]
