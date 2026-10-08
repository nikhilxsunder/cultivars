# filepath: /src/cultivars/state_space/__init__.py
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
r"""State-space substrates: linear-Gaussian, nonlinear, and regime-switching.

Every model in the package that filters a latent state -- an unobserved
components model, a mixed-frequency VAR, a term-structure model, a
switching autoregression, a stochastic volatility model -- is built on
one of the four engines here, and the
engines share one grammar: construct a fully specified system, hand it
data to ``filter``, ``smooth``, or ``loglikelihood``, and read the
result. None of them estimates anything. A linear-Gaussian system

.. math::

   y_t = Z_t \alpha_t + d_t + \varepsilon_t,\qquad
   \alpha_{t+1} = T_t \alpha_t + c_t + R_t \eta_t,

gets the Kalman filter, the Durbin-Koopman smoother and simulation
smoother, and an exact likelihood by the prediction-error
decomposition. A discrete :math:`K`-state chain over an autoregression
gets the Hamilton filter and Kim smoother, also exact. A linear-Gaussian
system whose matrices switch with the chain gets Kim's (1994) collapse
and a likelihood labelled as the approximation it is. A system with
arbitrary transition and measurement maps gets the extended, unscented
and particle filters, each with its matching smoother, chosen by name
because an extended likelihood, an unscented one and a particle
estimate are three different kinds of number.

The package has one rule that every module keeps: the engine owns the
recursions and nothing else. Parameterization, estimation, the mapping
from a model's parameters to system matrices, and the interpretation of
a state all belong to the model that uses the engine, which is why a
fitted result elsewhere in the package can expose its engine (an
``MSAR`` result's ``state_space``) and why the same instance can be
handed a series it was not fitted on. The other half of the rule is
that every likelihood says what it is: the linear and discrete engines
return exact Gaussian likelihoods, the switching-linear engine an
approximation named in its docstring, the particle filter an unbiased
Monte Carlo estimate whose seed-to-seed spread is its error bar.

Each module is a leaf: names are imported from the module, not from
this package, and each public name is an alias of the ``_internals``
class that implements it. The three engines with a single ``filter``
sign the shared ``_StateSpace[F, S]`` contract -- ``k_endog``,
``k_states``, ``filter``, ``smooth``, ``loglikelihood`` -- generic in
their result types so that a regime-probability filter and a
state-covariance filter satisfy it without inheriting fields they
cannot fill; the nonlinear engine deliberately does not, because it has
three filters, not one.

Layout. :mod:`~cultivars.state_space.linear_gaussian` is
:class:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM` with
:class:`~cultivars.state_space.linear_gaussian.KalmanFilterResult` and
:class:`~cultivars.state_space.linear_gaussian.DurbinKoopmanSmootherResult`.
:mod:`~cultivars.state_space.regime_switching` is
:class:`~cultivars.state_space.regime_switching.MarkovSwitchingSSM`
(Hamilton filter, exact) and
:class:`~cultivars.state_space.regime_switching.RegimeSwitchingLinearSSM`
(Kim's collapse), with
:class:`~cultivars.state_space.regime_switching.HamiltonFilterResult`,
:class:`~cultivars.state_space.regime_switching.KimFilterResult` and
:class:`~cultivars.state_space.regime_switching.KimSmootherResult`.
:mod:`~cultivars.state_space.nonlinear` is
:class:`~cultivars.state_space.nonlinear.NonlinearSSM` with
:class:`~cultivars.state_space.nonlinear.RtsSmootherResult`,
:class:`~cultivars.state_space.nonlinear.ParticleFilterResult` and
:class:`~cultivars.state_space.nonlinear.ParticleSmootherResult`; the
extended and unscented filters return the linear engine's
``KalmanFilterResult`` because they carry the same moments. The models
that consume these engines live where their subject matter does:
:mod:`~cultivars.univariate.unobserved_components`,
:mod:`~cultivars.univariate.regime_switching`,
:mod:`~cultivars.univariate.stochastic_volatility`,
:mod:`~cultivars.multivariate.reduced_form.mixed_frequency`,
:mod:`~cultivars.multivariate.reduced_form.term_structure` and
:mod:`~cultivars.multivariate.regime_switching.markov_switching`.

Example:
    One AR(1)-plus-noise system, four ways. The nonlinear engine's
    unscented filter and the switching engine with two identical
    regimes reproduce the Kalman likelihood exactly, and the particle
    filter lands within its Monte Carlo error of it:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> state = np.zeros(200)
    >>> for t in range(1, 200):
    ...     state[t] = 0.8 * state[t - 1] + np.sqrt(0.5) * rng.standard_normal()
    >>> y = state + rng.standard_normal(200)
    >>> linear = linear_gaussian.LinearGaussianSSM(
    ...     design=[[1.0]], obs_cov=[[1.0]], transition=[[0.8]],
    ...     selection=[[1.0]], state_cov=[[0.5]],
    ... )
    >>> exact = linear.loglikelihood(y)
    >>> round(exact, 2)
    -344.14
    >>> smooth = nonlinear.NonlinearSSM(
    ...     lambda s: 0.8 * s, lambda s: s, state_cov=[[0.5]], obs_cov=[[1.0]],
    ...     initial_state=[0.0], initial_state_cov=[[0.5 / 0.36]],
    ... )
    >>> bool(abs(smooth.unscented_filter(y).loglikelihood - exact) < 1e-8)
    True
    >>> chain = np.array([[0.9, 0.1], [0.1, 0.9]])
    >>> switching = regime_switching.RegimeSwitchingLinearSSM(
    ...     design=np.ones((2, 1, 1)), obs_cov=np.ones((2, 1, 1)),
    ...     transition=np.full((2, 1, 1), 0.8), state_cov=np.full((2, 1, 1), 0.5),
    ...     regime_transition=chain, initial_state_cov=[[0.5 / 0.36]],
    ... )
    >>> bool(abs(switching.loglikelihood(y) - exact) < 1e-8)
    True
    >>> estimate = smooth.particle_filter(y, n_particles=2000, seed=0).loglikelihood
    >>> bool(abs(estimate - exact) < 2.0)
    True
"""

from __future__ import annotations

from . import linear_gaussian, nonlinear, regime_switching

__all__ = [
    "linear_gaussian",
    "nonlinear",
    "regime_switching",
]
