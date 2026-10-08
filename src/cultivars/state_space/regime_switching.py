# filepath: /src/cultivars/state_space/regime_switching.py
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
r"""Discrete Markov-switching inference: the Hamilton filter, the Kim smoother, and Kim's collapse.

A first-order :math:`K`-state Markov chain :math:`S_t` with
row-stochastic transition matrix :math:`P`,
:math:`P_{ij} = \Pr(S_t = j \mid S_{t-1} = i)`, drives the parameters
of an observation model. Given the per-regime conditional densities
:math:`\eta_{t,j} = p(y_t \mid S_t = j, y_{1:t-1})`, the Hamilton
filter carries the regime distribution forward,

.. math::

   \xi_{t \mid t-1} = P'\,\xi_{t-1 \mid t-1},\qquad
   \xi_{t \mid t} = \frac{\xi_{t \mid t-1} \odot \eta_t}
                         {\mathbf 1'(\xi_{t \mid t-1} \odot \eta_t)},\qquad
   \log L = \sum_t \log \mathbf 1'(\xi_{t \mid t-1} \odot \eta_t),

and the Kim smoother runs it backward,

.. math::

   \xi_{t \mid n} = \xi_{t \mid t} \odot
   \Bigl[P\,\bigl(\xi_{t+1 \mid n} \oslash \xi_{t+1 \mid t}\bigr)\Bigr],

so that the sum over all :math:`K^n` regime paths is carried out
implicitly by a :math:`K`-vector. When the observation model is an
autoregression whose intercept and variance switch with the regime,
:math:`\eta_{t,j}` depends on :math:`S_t` alone and the likelihood is
exact; that is :class:`MarkovSwitchingSSM`. When each regime carries a
whole linear-Gaussian state space, exact filtering would need one
Kalman filter per regime *path*, and Kim (1994) collapses instead: run
the :math:`K \times K` per-pair Kalman updates, weight them by the
Hamilton probabilities, and moment-match each regime's mixture back to
a single mean and covariance. That likelihood is an approximation and
:class:`RegimeSwitchingLinearSSM` labels it one; it is exact in the two
limits that verify it -- identical regimes, where it is the Kalman
filter, and a chain with no persistence, where it is the Hamilton
filter over a static mixture.

Two commitments shape the surface. First, the chain and the observation
model are separated: the recursions above know nothing about what
produced the densities, so :class:`MarkovSwitchingSSM` exposes
``density_matrix`` and ``filter_densities`` as well as ``filter``, and
a switching model of any other form -- a switching VAR, a switching
variance -- reuses the same two recursions by supplying its own
``(n, K)`` matrix of log densities. Second, the conventions are fixed
and stated: transition matrices are row-stochastic and validated as
such, regime vectors are length :math:`K` with the prediction step
``xi_filt @ transition``, and densities are passed in logarithms and
combined by log-sum-exp so that a small-variance regime meeting an
outlier does not underflow to a zero-probability path.

Layout. :class:`MarkovSwitchingSSM` holds a fully specified
intercept-switching AR model -- ``transition``, ``intercepts``,
``ar_params`` of shape ``(K, p)``, ``variances`` and an optional
``initial_prob`` defaulting to the ergodic distribution -- and its
``filter`` returns :class:`HamiltonFilterResult` with filtered and
predicted regime probabilities on the ``n - p`` rows the lags leave,
``smooth`` returns :class:`KimSmootherResult`, and ``loglikelihood``
the exact likelihood. :class:`RegimeSwitchingLinearSSM` holds per-regime
system matrices stacked on a leading regime axis with the chain's
``regime_transition``; its ``filter`` returns :class:`KimFilterResult`
carrying regime probabilities, the regime-marginal state and its
mixture covariance, and the per-regime collapsed states, and ``smooth``
returns the same :class:`KimSmootherResult`. All five names are public
aliases of ``_MarkovSwitchingStateSpace``,
``_RegimeSwitchingLinearStateSpace`` and the three result records in
``_internals``, where the generic ``hamilton_filter`` and
``kim_smoother`` recursions the two models share also live, in
``_filters`` and ``_smoothers``. The estimator that fits the switching
autoregression by maximum likelihood, and whose result exposes the
fitted :class:`MarkovSwitchingSSM` as ``state_space``, is
:class:`~cultivars.univariate.regime_switching.MSAR`; the multivariate
forms are in :mod:`~cultivars.multivariate.regime_switching.markov_switching`.

References:
    Hamilton, J. D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. *Econometrica*,
    57(2), 357-384.

    Hamilton, J. D. (1994). *Time Series Analysis*, chapter 22.
    Princeton University Press.

    Kim, C.-J. (1994). Dynamic linear models with Markov-switching.
    *Journal of Econometrics*, 60(1-2), 1-22.

    Kim, C.-J., & Nelson, C. R. (1999). *State-Space Models with Regime
    Switching*. MIT Press.

Example:
    A two-regime intercept-switching AR(1) with persistent regimes:
    the filter classifies 97% of periods correctly and the smoother 98%.
    A regime-switching linear model whose two regimes are identical
    reproduces the Kalman likelihood exactly, which is the first of the
    two limits that verify Kim's collapse:

    >>> import numpy as np
    >>> from cultivars.state_space.linear_gaussian import LinearGaussianSSM
    >>> rng = np.random.default_rng(0)
    >>> chain = np.array([[0.95, 0.05], [0.1, 0.9]])
    >>> regime, y = np.zeros(400, dtype=int), np.zeros(400)
    >>> for t in range(1, 400):
    ...     regime[t] = rng.choice(2, p=chain[regime[t - 1]])
    ...     y[t] = [1.0, -1.0][regime[t]] + 0.5 * y[t - 1] + np.sqrt(0.5) * rng.standard_normal()
    >>> model = MarkovSwitchingSSM(chain, [1.0, -1.0], [[0.5], [0.5]], [0.5, 0.5])
    >>> filtered, smoothed = model.filter(y), model.smooth(y)
    >>> filtered.filtered_prob.shape, round(filtered.loglikelihood, 2)
    ((399, 2), -501.16)
    >>> hit = (filtered.filtered_prob[:, 1] > 0.5) == (regime[1:] == 1)
    >>> hit_smoothed = (smoothed.smoothed_prob[:, 1] > 0.5) == (regime[1:] == 1)
    >>> round(float(hit.mean()), 2), round(float(hit_smoothed.mean()), 2)
    (0.97, 0.98)
    >>> switching = RegimeSwitchingLinearSSM(
    ...     design=np.ones((2, 1, 1)), obs_cov=np.ones((2, 1, 1)),
    ...     transition=np.full((2, 1, 1), 0.8), state_cov=np.full((2, 1, 1), 0.5),
    ...     regime_transition=chain, initial_state_cov=[[0.5 / (1 - 0.64)]],
    ... )
    >>> linear = LinearGaussianSSM(
    ...     design=[[1.0]], obs_cov=[[1.0]], transition=[[0.8]],
    ...     selection=[[1.0]], state_cov=[[0.5]],
    ... )
    >>> bool(abs(switching.loglikelihood(y) - linear.loglikelihood(y)) < 1e-8)
    True
"""

from __future__ import annotations

from .._internals import (
    _HamiltonFilterResult as HamiltonFilterResult,
)
from .._internals import (
    _KimFilterResult as KimFilterResult,
)
from .._internals import (
    _KimSmootherResult as KimSmootherResult,
)
from .._internals import (
    _MarkovSwitchingStateSpace as MarkovSwitchingSSM,
)
from .._internals import (
    _RegimeSwitchingLinearStateSpace as RegimeSwitchingLinearSSM,
)

__all__ = [
    "HamiltonFilterResult",
    "KimFilterResult",
    "KimSmootherResult",
    "MarkovSwitchingSSM",
    "RegimeSwitchingLinearSSM",
]
