# filepath: /src/cultivars/state_space/nonlinear.py
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
r"""Nonlinear and non-Gaussian state-space models: extended, unscented, and particle methods.

The additive-Gaussian core is

.. math::

   y_t = g(\alpha_t) + \varepsilon_t,\qquad \varepsilon_t \sim N(0, R),

   \alpha_{t+1} = f(\alpha_t) + \eta_t,\qquad \eta_t \sim N(0, Q),
   \qquad \alpha_1 \sim N(a_1, P_1),

with :math:`f` and :math:`g` arbitrary *batched* maps -- each takes an
``(n, m)`` block of states and returns ``(n, m)`` or ``(n, p)`` rows --
which is what lets sigma points and particle clouds move through them
without Python loops. Three filters read it, in increasing generality
and decreasing exactness of assumptions. The extended filter
linearizes :math:`f` and :math:`g` by central-difference Jacobians and
runs the Kalman recursions on the linearization. The unscented filter
propagates :math:`2m + 1` deterministic sigma points at spread
:math:`\lambda = \alpha^2 (m + \kappa) - m` and recombines them with
Julier-Uhlmann weights, exact through linear maps for any setting and
third-order accurate through smooth ones. The particle filter --
bootstrap, or auxiliary in the Pitt-Shephard form -- propagates a
weighted cloud, reweights it by the measurement density, resamples
systematically when the effective sample size
:math:`(\sum_i w_i^2)^{-1}` falls below a stated fraction, and returns
an unbiased *estimate* of the likelihood,

.. math::

   \hat L = \prod_{t=1}^{n} \frac{1}{N}\sum_{i=1}^{N} p\bigl(y_t \mid \alpha_t^{(i)}\bigr),

whose Monte Carlo noise is real and whose seed-to-seed spread is the
honest error bar. Each filter has a matching smoother: Rauch-Tung-
Striebel on the extended and unscented moments, and forward-filtering
backward-smoothing (Doucet-Godsill-Andrieu) on the particle clouds,
reweighting through the transition density :math:`N(f(x), Q)` at
quadratic cost per period.

Two commitments shape the surface. First, the filters are chosen by
name, not by dispatch, because their answers differ in kind: an
extended likelihood is the exact likelihood of an approximate model,
an unscented one is a better approximation of the same thing, and a
particle likelihood is a noisy unbiased estimate of the true one --
three numbers a user must not average or compare without knowing which
they hold. Second, non-Gaussian models enter through two hooks and the
linearizing filters refuse them: ``transition_sampler`` replaces the
additive state draw and ``observation_loglik`` replaces the Gaussian
measurement density (the stochastic-volatility model, whose measurement
noise is multiplicative, is the canonical customer), and a model
carrying either is outside the additive form, so ``extended_filter``
and ``unscented_filter`` raise rather than linearize an assumption that
no longer holds, and the particle smoother refuses a custom sampler
because it exposes draws without a density. Missing observations are
``numpy.nan`` rows: every filter skips the update and carries the
prediction, matching the linear substrate.

Layout. :class:`NonlinearSSM` is the model, constructed from the two
batched maps, the covariances ``state_cov`` and ``obs_cov``, the
initial state and covariance, and the optional hooks; its
``extended_filter`` and ``unscented_filter`` return the linear
substrate's :class:`~cultivars.state_space.linear_gaussian.KalmanFilterResult`
because they carry the same moments, ``extended_smoother`` and
``unscented_smoother`` return :class:`RtsSmootherResult` with the
``method`` recorded, and ``particle_filter`` and ``particle_smoother``
return :class:`ParticleFilterResult` and :class:`ParticleSmootherResult`
carrying cloud means and standard deviations rather than the clouds,
with the effective sample size per period as the degeneracy diagnostic.
All four names are public aliases of ``_NonlinearStateSpace``,
``_ParticleFilterResult``, ``_ParticleSmootherResult`` and
``_RtsSmootherResult`` in ``_internals``. The nonlinear engine
deliberately does not sign the shared ``_StateSpace`` contract of
:mod:`~cultivars.state_space.linear_gaussian`, since it has no single
``filter``; the univariate stochastic-volatility model built on its
particle filter is :mod:`~cultivars.univariate.stochastic_volatility`.

References:
    Julier, S. J., & Uhlmann, J. K. (1997). New extension of the Kalman
    filter to nonlinear systems. *Proceedings of SPIE*, 3068, 182-193.

    Gordon, N. J., Salmond, D. J., & Smith, A. F. M. (1993). Novel
    approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE
    Proceedings F*, 140(2), 107-113.

    Pitt, M. K., & Shephard, N. (1999). Filtering via simulation:
    Auxiliary particle filters. *Journal of the American Statistical
    Association*, 94(446), 590-599.

    Doucet, A., Godsill, S., & Andrieu, C. (2000). On sequential Monte
    Carlo sampling methods for Bayesian filtering. *Statistics and
    Computing*, 10(3), 197-208.

    Särkkä, S. (2013). *Bayesian Filtering and Smoothing*. Cambridge
    University Press.

Example:
    A linear AR(1) state through the nonlinear engine: the extended and
    unscented filters reproduce the Kalman likelihood to numerical
    precision (the Jacobians are central differences), and the
    particle filter's estimate is within its Monte Carlo error of it.
    A stochastic-volatility model, declared through the hooks, is
    refused by the extended filter and tracked by the particle filter:

    >>> import numpy as np
    >>> from cultivars.state_space.linear_gaussian import LinearGaussianSSM
    >>> rng = np.random.default_rng(0)
    >>> state = np.zeros(200)
    >>> for t in range(1, 200):
    ...     state[t] = 0.8 * state[t - 1] + np.sqrt(0.5) * rng.standard_normal()
    >>> y = state + rng.standard_normal(200)
    >>> linear = LinearGaussianSSM(
    ...     design=[[1.0]], obs_cov=[[1.0]], transition=[[0.8]],
    ...     selection=[[1.0]], state_cov=[[0.5]],
    ... )
    >>> model = NonlinearSSM(
    ...     lambda s: 0.8 * s, lambda s: s, state_cov=[[0.5]], obs_cov=[[1.0]],
    ...     initial_state=[0.0], initial_state_cov=[[0.5 / (1 - 0.64)]],
    ... )
    >>> exact = linear.loglikelihood(y)
    >>> bool(abs(model.extended_filter(y).loglikelihood - exact) < 1e-6)
    True
    >>> bool(abs(model.unscented_filter(y).loglikelihood - exact) < 1e-6)
    True
    >>> estimate = model.particle_filter(y, n_particles=2000, seed=0).loglikelihood
    >>> bool(abs(estimate - exact) < 2.0)
    True
    >>> def draw(particles, rng):
    ...     return 0.95 * particles + 0.3 * rng.standard_normal(particles.shape)
    >>> def measure(obs, particles):
    ...     h = particles[:, 0]
    ...     return -0.5 * (np.log(2 * np.pi) + h + obs[0] ** 2 * np.exp(-h))
    >>> volatility = NonlinearSSM(
    ...     lambda s: 0.95 * s, lambda s: s, state_cov=[[0.09]], obs_cov=[[1.0]],
    ...     initial_state=[0.0], initial_state_cov=[[1.0]],
    ...     transition_sampler=draw, observation_loglik=measure,
    ... )
    >>> volatility.is_additive_gaussian
    False
    >>> volatility.extended_filter(y)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    cultivars.exceptions.SpecificationError: the extended filter is defined only for ...
    >>> volatility.particle_filter(y, n_particles=1000, seed=0).filtered_state.shape
    (200, 1)
"""

from __future__ import annotations

from ..engine._internals import (
    _NonlinearStateSpace as NonlinearSSM,
)
from ..engine._internals import (
    _ParticleFilterResult as ParticleFilterResult,
)
from ..engine._internals import (
    _ParticleSmootherResult as ParticleSmootherResult,
)
from ..engine._internals import (
    _RtsSmootherResult as RtsSmootherResult,
)

__all__ = [
    "NonlinearSSM",
    "ParticleFilterResult",
    "ParticleSmootherResult",
    "RtsSmootherResult",
]
