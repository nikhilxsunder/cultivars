# filepath: /src/cultivars/multivariate/large_dim/__init__.py
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
"""Large-dimensional multivariate models: shrink, select, or factor the panel.

An unrestricted VAR on :math:`k` series with :math:`p` lags has
:math:`p k^2` slope coefficients, and past a handful of variables a
macroeconomic sample cannot pin them down. Every model here is one of
three answers to that count. Shrink the coefficients toward a prior and
report a posterior: the conjugate Normal-inverse-Wishart VAR with its
exact posterior and closed-form marginal likelihood
(:mod:`~cultivars.multivariate.large_dim.bayesian`), the same model with
the tightness itself estimated
(:mod:`~cultivars.multivariate.large_dim.hierarchical`), Gibbs sampling
under independent priors that select or shrink coefficient by coefficient
(:mod:`~cultivars.multivariate.large_dim.gibbs`), and the two relaxations
of the Gaussian likelihood -- Student-t innovations
(:mod:`~cultivars.multivariate.large_dim.student`) and stochastic
volatility (:mod:`~cultivars.multivariate.large_dim.volatility`). Set
most of the coefficients to exactly zero and report the survivors: the
penalized VAR under five penalties, whose support is a Granger network
(:mod:`~cultivars.multivariate.large_dim.sparse`), and its extension to
the contemporaneous precision network
(:mod:`~cultivars.multivariate.large_dim.graphical`). Or compress the
panel to a few factors and model those: the dynamic factor model
(:mod:`~cultivars.multivariate.large_dim.dynamic_factor`), the
factor-augmented VAR that puts observed policy variables beside the
factors (:mod:`~cultivars.multivariate.large_dim.factor_augmented`), and
the factor stochastic-volatility model for a wide panel's time-varying
covariance (:mod:`~cultivars.multivariate.large_dim.factor_volatility`).
One module is a view rather than an estimator:
:mod:`~cultivars.multivariate.large_dim.spillover` reads any fitted
closed reduced form as a Diebold-Yilmaz connectedness network.

The models share one shape, with one honest split. A model object holds
the specification; ``fit`` returns an immutable record that renders the
same way everywhere (``print``, a notebook cell, ``summary()``). The
Bayesian records carry draws and propagate them -- credible intervals,
impulse responses with posterior bands, a predictive whose bands are
bands of the predictive distribution, a stability share, and chain
diagnostics where the draws are a chain -- and deliberately report no
likelihood, parameter count or information criteria, because a posterior
has none; the conjugate and hierarchical models offer a marginal
likelihood instead, and the others say why they do not. The penalized
records carry the selected coefficients, the penalty path and its
cross-validation errors, and deliberately report no standard errors,
because inference after selection is not a solved problem at this
generality. The penalized results satisfy the closed-system contract and
so feed the structural layer and the spillover view directly; the
Bayesian results propagate through their own draws instead.

Names are imported from the leaf module, never from here --
``from cultivars.multivariate.large_dim.sparse import SparseVAR`` -- so
an import path names the family, the specification and the class. This
package exposes only the modules.

Layout. The public classes are thin: each validates its specification on
a model base in ``cultivars._internals`` -- the Bayesian family on
``_BayesianVectorAutoRegressionModel`` and its Student-t, Gibbs and
stochastic-volatility extensions, the penalized family on
``_SparseVectorAutoRegressionModel``, the factor stochastic-volatility
model on ``_FactorVolatilityModel``, while the two least-squares factor
models validate in place -- and assembles its public record from a packed
fit. The samplers (``_draw_conjugate``, the volatility and factor
blocks), the FISTA solver and the information criteria live in ``_core``
and ``_internals``; the priors the Bayesian family accepts are in
:mod:`~cultivars.bayes.priors`.
The small-system reduced forms these models scale up are in
:mod:`~cultivars.multivariate.reduced_form`; identification of any
closed result is in :mod:`~cultivars.multivariate.structural`.

Example:
    One six-variable VAR(1) through the three answers: the conjugate
    posterior with its evidence, the lasso with its support, and the
    Student-t fit with its tail index -- then the sparse result read as
    a network:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> k, n = 6, 300
    >>> first = 0.5 * np.eye(k)
    >>> first[1, 0], first[3, 2] = 0.3, -0.3
    >>> y = np.zeros((n, k))
    >>> for t in range(1, n):
    ...     y[t] = first @ y[t - 1] + rng.standard_normal(k)
    >>> conjugate = bayesian.BVAR(y, order=1).fit(n_draws=200, seed=0)
    >>> bool(np.isfinite(conjugate.log_marginal_likelihood)), conjugate.forecast(4).shape
    (True, (4, 6, 3))
    >>> lasso = sparse.SparseVAR(y, order=1).fit(penalty="lasso")
    >>> edges = lasso.granger_adjacency(threshold=0.15)
    >>> [(i, j) for i in range(k) for j in range(k) if edges[i, j]]
    [(1, 0), (3, 2)]
    >>> heavy = student.StudentBVAR(y, order=1).fit(n_draws=300, n_burn=100, seed=0)
    >>> bool(heavy.df > 10.0), heavy.outlier_dates().size
    (True, 0)
    >>> network = spillover.Spillover(lasso, horizon=10).compute()
    >>> network.transmitters()[:2]
    ('y1', 'y3')
"""

from . import (
    bayesian,
    dynamic_factor,
    factor_augmented,
    factor_volatility,
    gibbs,
    graphical,
    hierarchical,
    sparse,
    spillover,
    student,
    volatility,
)

__all__ = [
    "bayesian",
    "dynamic_factor",
    "factor_augmented",
    "factor_volatility",
    "gibbs",
    "graphical",
    "hierarchical",
    "sparse",
    "spillover",
    "student",
    "volatility",
]
