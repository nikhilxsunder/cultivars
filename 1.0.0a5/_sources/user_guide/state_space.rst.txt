.. meta::
   :description: Kalman, Durbin-Koopman, extended, unscented, particle, Hamilton and Kim filters in cultivars: system conventions, likelihoods, and use.

State-space models and filtering
================================

The :mod:`cultivars.state_space` package holds the four filtering engines
that every latent-state model in the library runs on: a linear-Gaussian
system, a nonlinear system, a discrete Markov chain over an
autoregression, and a linear system whose matrices switch with such a
chain. An engine is a fully specified system that filters, smooths and
evaluates a likelihood; it has no parameters, no ``fit`` and no
estimator. Use this page to write a system down directly, or to
understand and re-use the engine behind a fitted unobserved-components,
term-structure, volatility or regime-switching model. Estimating those
models is covered on their own pages (:doc:`univariate_linear`,
:doc:`volatility`, :doc:`univariate_nonlinear`, :doc:`var_extensions`,
:doc:`nonlinear_multivariate`).

.. contents:: On this page
   :local:
   :depth: 1

The family at a glance
----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 22 32 24

   * - Class
     - Module
     - System in one line
     - Use when
   * - :class:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM`
     - :mod:`~cultivars.state_space.linear_gaussian`
     - :math:`y_t = Z\alpha_t + d + \varepsilon_t`,
       :math:`\alpha_{t+1} = T\alpha_t + c + R\eta_t`, Gaussian noise
     - The system is linear and Gaussian; you want the exact likelihood,
       smoothed states, or conditional state draws.
   * - :class:`~cultivars.state_space.nonlinear.NonlinearSSM`
     - :mod:`~cultivars.state_space.nonlinear`
     - :math:`y_t = g(\alpha_t) + \varepsilon_t`,
       :math:`\alpha_{t+1} = f(\alpha_t) + \eta_t`, or non-Gaussian
       through hooks
     - The maps are nonlinear (extended, unscented filters) or the noise
       is non-Gaussian (particle filter).
   * - :class:`~cultivars.state_space.regime_switching.MarkovSwitchingSSM`
     - :mod:`~cultivars.state_space.regime_switching`
     - Intercept- and variance-switching AR(:math:`p`) driven by a
       :math:`K`-state chain
     - You need exact regime probabilities and likelihood (Hamilton
       filter, Kim smoother).
   * - :class:`~cultivars.state_space.regime_switching.RegimeSwitchingLinearSSM`
     - :mod:`~cultivars.state_space.regime_switching`
     - One linear-Gaussian system per regime, chain on top
     - A continuous state and a regime chain together (Kim's 1994
       collapse; approximate likelihood).

Every engine reports its output in a frozen result record:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Record
     - Produced by
     - Fields
   * - :class:`~cultivars.state_space.linear_gaussian.KalmanFilterResult`
     - ``LinearGaussianSSM.filter``, ``NonlinearSSM.extended_filter``,
       ``NonlinearSSM.unscented_filter``
     - ``loglikelihood``, ``loglikelihood_contributions``,
       ``predicted_state``, ``predicted_state_cov``, ``filtered_state``,
       ``filtered_state_cov``
   * - :class:`~cultivars.state_space.linear_gaussian.DurbinKoopmanSmootherResult`
     - ``LinearGaussianSSM.smooth``
     - ``smoothed_state``, ``smoothed_state_cov``
   * - :class:`~cultivars.state_space.nonlinear.RtsSmootherResult`
     - ``extended_smoother``, ``unscented_smoother``
     - ``smoothed_state``, ``smoothed_state_cov``, ``method``
   * - :class:`~cultivars.state_space.nonlinear.ParticleFilterResult`
     - ``particle_filter``
     - ``loglikelihood``, ``loglikelihood_contributions``,
       ``filtered_state``, ``filtered_state_std``,
       ``effective_sample_size``, ``n_particles``, ``method``
   * - :class:`~cultivars.state_space.nonlinear.ParticleSmootherResult`
     - ``particle_smoother``
     - ``smoothed_state``, ``smoothed_state_std``, ``n_particles``
   * - :class:`~cultivars.state_space.regime_switching.HamiltonFilterResult`
     - ``MarkovSwitchingSSM.filter``
     - ``loglikelihood``, ``loglikelihood_contributions``,
       ``filtered_prob``, ``predicted_prob``
   * - :class:`~cultivars.state_space.regime_switching.KimFilterResult`
     - ``RegimeSwitchingLinearSSM.filter``
     - the Hamilton fields plus ``filtered_state``,
       ``filtered_state_cov``, ``regime_state``
   * - :class:`~cultivars.state_space.regime_switching.KimSmootherResult`
     - ``smooth`` of either switching engine
     - ``smoothed_prob``, ``smoothed_joint_prob``

One grammar for four engines
----------------------------

The engines share one workflow: construct a fully specified system, hand
it data, and read a record. ``LinearGaussianSSM``,
``MarkovSwitchingSSM`` and ``RegimeSwitchingLinearSSM`` expose the same
five members: ``k_endog`` (observed dimension), ``k_states``,
``filter(y)``, ``smooth(y)`` and ``loglikelihood(y)``.
``NonlinearSSM`` has ``k_endog`` and ``k_states`` but no single
``filter``: its three filters return different kinds of likelihood, so
you choose one by name (``extended_filter``, ``unscented_filter``,
``particle_filter``).

Three rules hold across the package:

* **Data layout.** Observations are ``(n, p)`` with time down the rows;
  a 1-D array is accepted when ``p`` is one. Every state array in a
  result has time on the first axis: states are ``(n, m)``, covariances
  ``(n, m, m)``, regime probabilities ``(n, K)``.
* **Missing data are** ``numpy.nan``. A missing period contributes
  exactly zero to the log-likelihood and the filter carries its
  prediction forward. The linear engine handles element-wise gaps; the
  nonlinear and Kim engines accept only whole missing rows; the Hamilton
  engine accepts none (see `Nuances and gotchas`_).
* **Every likelihood says what it is.** The Kalman and Hamilton
  likelihoods are exact; the extended and unscented ones are exact
  likelihoods of an approximating model; the particle likelihood is an
  unbiased Monte Carlo estimate; the Kim likelihood is an approximation.
  ``loglikelihood`` is the sum of ``loglikelihood_contributions``, the
  per-period prediction-error terms :math:`\log p(y_t \mid y_{1:t-1})`.

Linear-Gaussian systems
-----------------------

The model
~~~~~~~~~

:class:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM` uses
the notation of Durbin and Koopman (2012):

.. math::

   y_t = Z_t \alpha_t + d_t + \varepsilon_t,
   \qquad \varepsilon_t \sim N(0, H_t),

   \alpha_{t+1} = T_t \alpha_t + c_t + R_t \eta_t,
   \qquad \eta_t \sim N(0, Q_t),
   \qquad \alpha_1 \sim N(a_1, P_1).

The Kalman filter returns the prediction-error decomposition

.. math::

   \log L = -\tfrac12 \sum_{t=1}^{n}
   \bigl(p_t \log 2\pi + \log|F_t| + v_t' F_t^{-1} v_t\bigr),
   \qquad v_t = y_t - Z_t a_t - d_t,\quad F_t = Z_t P_t Z_t' + H_t,

with :math:`p_t` the number of elements observed at :math:`t`. The
smoother is the Durbin-Koopman backward recursion, and the simulation
smoother draws :math:`\alpha \mid y` by the mean-correction device of
Durbin and Koopman (2002).

System matrices and shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~

With :math:`p` observed series (``k_endog``), :math:`m` states
(``k_states``) and :math:`r` state disturbances (``k_posdef``):

.. list-table::
   :header-rows: 1
   :widths: 10 24 22 22 22

   * - Symbol
     - Argument
     - Time-invariant
     - Time-varying
     - Default
   * - :math:`Z`
     - ``design``
     - ``(p, m)``
     - ``(n, p, m)``
     - required
   * - :math:`H`
     - ``obs_cov``
     - ``(p, p)``
     - ``(n, p, p)``
     - required
   * - :math:`T`
     - ``transition``
     - ``(m, m)``
     - ``(n, m, m)``
     - required
   * - :math:`R`
     - ``selection``
     - ``(m, r)``
     - ``(n, m, r)``
     - required
   * - :math:`Q`
     - ``state_cov``
     - ``(r, r)``
     - ``(n, r, r)``
     - required
   * - :math:`d`
     - ``obs_intercept``
     - ``(p,)``
     - ``(n, p)``
     - zero
   * - :math:`c`
     - ``state_intercept``
     - ``(m,)``
     - ``(n, m)``
     - zero
   * - :math:`a_1`
     - ``initial_state``
     - ``(m,)``
     - n/a
     - zero
   * - :math:`P_1`
     - ``initial_state_cov``
     - ``(m, m)``
     - n/a
     - stationary or :math:`10^6 I` (below)

The five system matrices are positional, in the order ``design,
obs_cov, transition, selection, state_cov``; the rest are keyword-only.
A 3-D matrix is time-varying with time on the leading axis, and each
matrix may be static or time-varying independently of the others.
Inconsistent trailing shapes raise
:class:`~cultivars.exceptions.DimensionError`; non-finite entries raise
:class:`~cultivars.exceptions.NumericalError`.

A local level, end to end
~~~~~~~~~~~~~~~~~~~~~~~~~

A random-walk level observed with noise, signal-to-noise ratio
:math:`q = 0.1`:

.. testcode::

   import numpy as np
   from cultivars.state_space.linear_gaussian import LinearGaussianSSM

   rng = np.random.default_rng(7)
   n = 200
   level = np.cumsum(np.sqrt(0.1) * rng.standard_normal(n))
   y = level + rng.standard_normal(n)

   model = LinearGaussianSSM(
       design=[[1.0]], obs_cov=[[1.0]], transition=[[1.0]],
       selection=[[1.0]], state_cov=[[0.1]],
   )
   print(model.k_endog, model.k_states, model.k_posdef)
   filtered = model.filter(y)
   print(round(filtered.loglikelihood, 3))
   print(filtered.filtered_state.shape, filtered.filtered_state_cov.shape)
   print(filtered.loglikelihood_contributions.shape)

.. testoutput::

   1 1 1
   -314.652
   (200, 1) (200, 1, 1)
   (200,)

The predicted variance converges to the steady state
:math:`\bar P = (q + \sqrt{q^2 + 4q})/2 = 0.3702`, and the smoother,
which conditions on the whole sample, cuts the mid-sample variance
from 0.27 to 0.156:

.. testcode::

   print(round(float(filtered.predicted_state_cov[-1, 0, 0]), 4))
   smoothed = model.smooth(y)
   print(smoothed.smoothed_state.shape, smoothed.smoothed_state_cov.shape)
   mid = n // 2
   print(round(float(smoothed.smoothed_state_cov[mid, 0, 0]), 3),
         round(float(filtered.filtered_state_cov[mid, 0, 0]), 3))

.. testoutput::

   0.3702
   (200, 1) (200, 1, 1)
   0.156 0.27

A gap is ``nan``. The gap's periods contribute exactly zero to the
likelihood and the filtered variance grows by :math:`q` per missing
period:

.. testcode::

   gappy = y.copy()
   gappy[80:90] = np.nan
   holes = model.filter(gappy)
   print(float(holes.loglikelihood_contributions[80:90].sum()))
   print(holes.filtered_state_cov[80:83, 0, 0].round(2).tolist())

.. testoutput::

   0.0
   [0.37, 0.47, 0.57]

``simulation_smoother`` returns ``n_sims`` draws of the whole state path
from :math:`p(\alpha \mid y)`, shape ``(n_sims, n, m)``; ``simulate``
draws a fresh observation path, shape ``(n, p)``:

.. testcode::

   draws = model.simulation_smoother(y, n_sims=20, seed=0)
   print(draws.shape, model.simulate(50, seed=0).shape)

.. testoutput::

   (20, 200, 1) (50, 1)

Initialization
~~~~~~~~~~~~~~

When ``initial_state_cov`` is omitted the engine picks :math:`P_1` by a
stated rule:

* If ``transition``, ``selection`` and ``state_cov`` are all
  time-invariant and every eigenvalue of :math:`T` is inside the unit
  circle, :math:`P_1` is the stationary covariance solving
  :math:`P = TPT' + RQR'`. The classmethod
  :meth:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM.stationary_covariance`
  solves the same Lyapunov equation on its own and raises
  :class:`~cultivars.exceptions.NumericalError` for an unstable
  :math:`T`.
* Otherwise :math:`P_1 = 10^6 I`: the *approximate* diffuse prior, not
  the exact diffuse initialization of Durbin and Koopman (2012, ch. 5).

Above, the random-walk level got the diffuse prior; its first
contribution is about :math:`-\tfrac12(\log 2\pi + \log 10^6)`. An
AR(1) state gets the stationary variance
:math:`0.5 / (1 - 0.8^2) = 1.3889`:

.. testcode::

   print(float(filtered.predicted_state_cov[0, 0, 0]),
         round(float(filtered.loglikelihood_contributions[0]), 3))
   ar1 = LinearGaussianSSM([[1.0]], [[1.0]], [[0.8]], [[1.0]], [[0.5]])
   print(ar1.filter(y).predicted_state_cov[0].round(4).tolist())
   print(LinearGaussianSSM.stationary_covariance(
       [[0.8]], [[1.0]], [[0.5]]).round(4).tolist())

.. testoutput::

   1000000.0 -7.827
   [[1.3889]]
   [[1.3889]]

The initial *mean* has no such rule: ``initial_state`` defaults to zero
even when a ``state_intercept`` moves the stationary mean to
:math:`(I - T)^{-1} c`. Pass it yourself:

.. testcode::

   shifted = LinearGaussianSSM(
       [[1.0]], [[1.0]], [[0.8]], [[1.0]], [[0.5]], state_intercept=[1.0],
   )
   print(shifted.filter(y).predicted_state[0].tolist())
   centred = LinearGaussianSSM(
       [[1.0]], [[1.0]], [[0.8]], [[1.0]], [[0.5]], state_intercept=[1.0],
       initial_state=[5.0],
   )
   print(centred.filter(y).predicted_state[0].tolist())

.. testoutput::

   [0.0]
   [5.0]

What the records give you
~~~~~~~~~~~~~~~~~~~~~~~~~

``filter`` returns
:class:`~cultivars.state_space.linear_gaussian.KalmanFilterResult`.
Row ``t`` of ``predicted_state`` is :math:`a_{t \mid t-1}` (row 0 is
:math:`a_1`), row ``t`` of ``filtered_state`` is
:math:`a_{t \mid t}`, and the covariances follow the same indexing.
``smooth`` returns
:class:`~cultivars.state_space.linear_gaussian.DurbinKoopmanSmootherResult`
with :math:`\hat\alpha_{t \mid n}` and its variance. ``loglikelihood(y)``
is the likelihood-only path, the call an optimizer should make.

Nonlinear and non-Gaussian systems
----------------------------------

The model
~~~~~~~~~

:class:`~cultivars.state_space.nonlinear.NonlinearSSM` holds the
additive-Gaussian system

.. math::

   y_t = g(\alpha_t) + \varepsilon_t,\qquad \varepsilon_t \sim N(0, H),
   \qquad
   \alpha_{t+1} = f(\alpha_t) + \eta_t,\qquad \eta_t \sim N(0, Q),
   \qquad \alpha_1 \sim N(a_1, P_1).

There is no selection matrix: ``state_cov`` is the ``(m, m)`` covariance
of the full state disturbance. The two maps are positional and
*batched*: ``transition`` maps an ``(N, m)`` block of states to
``(N, m)`` and ``observation`` maps it to ``(N, p)``, so sigma points
and particle clouds pass through them in one call. The constructor
probes both maps with a two-row batch and raises
:class:`~cultivars.exceptions.DimensionError` if the shapes are wrong;
``p`` is read from ``obs_cov``. ``initial_state`` and
``initial_state_cov`` are required: there is no default initialization.

Three filters and their smoothers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 30 22 24

   * - Filter
     - Method
     - Likelihood is
     - Smoother
   * - ``extended_filter(y)``
     - Central-difference Jacobians, then the Kalman recursions
     - exact for the linearized model
     - ``extended_smoother(y)``, RTS
   * - ``unscented_filter(y, alpha=0.1, beta=2.0, kappa=0.0)``
     - :math:`2m+1` sigma points, Julier and Uhlmann (1997) weights
     - exact for the sigma-point approximation
     - ``unscented_smoother(y, ...)``, RTS
   * - ``particle_filter(y, n_particles=2000, method="bootstrap",
       ess_threshold=0.5, seed=None)``
     - Bootstrap (Gordon, Salmond and Smith 1993) or ``"auxiliary"``
       (Pitt and Shephard 1999), systematic resampling
     - an unbiased Monte Carlo estimate
     - ``particle_smoother(y, n_particles=500, ...)``, forward
       filtering backward smoothing (Doucet, Godsill and Andrieu 2000)

An AR(1) state seen through an exponential measurement:

.. testcode::

   from cultivars.state_space.nonlinear import NonlinearSSM

   rng = np.random.default_rng(11)
   n = 150
   state = np.zeros(n)
   for t in range(1, n):
       state[t] = 0.9 * state[t - 1] + 0.5 * rng.standard_normal()
   y_nl = np.exp(0.5 * state) + 0.5 * rng.standard_normal(n)

   curved = NonlinearSSM(
       lambda s: 0.9 * s,
       lambda s: np.exp(0.5 * s),
       state_cov=[[0.25]], obs_cov=[[0.25]],
       initial_state=[0.0], initial_state_cov=[[0.25 / 0.19]],
   )
   print(curved.k_states, curved.k_endog, curved.is_additive_gaussian)
   ekf = curved.extended_filter(y_nl)
   ukf = curved.unscented_filter(y_nl)
   pf = curved.particle_filter(y_nl, n_particles=500, seed=0)
   print(type(ekf).__name__ == type(ukf).__name__)
   print(round(ekf.loglikelihood, 2), round(ukf.loglikelihood, 2),
         round(pf.loglikelihood, 2))

.. testoutput::

   1 1 True
   True
   -137.52 -137.41 -135.97

The particle estimate is noisy, and its seed-to-seed spread is the error
bar. Here the spread is 0.25 log points, so the 1.5-point gap to the
extended and unscented likelihoods is linearization error, not Monte
Carlo noise:

.. testcode::

   spread = [curved.particle_filter(y_nl, n_particles=500, seed=s).loglikelihood
             for s in range(3)]
   print(round(float(np.std(spread)), 2))
   print(pf.n_particles, pf.method, pf.filtered_state_std.shape)
   print(round(float(pf.effective_sample_size.min()), 1))

.. testoutput::

   0.25
   500 bootstrap (150, 1)
   20.6

``effective_sample_size`` is :math:`(\sum_i w_i^2)^{-1}` per period, in
``(0, n_particles]``. A minimum near one means the cloud collapsed at
that period and its contribution is unreliable.

The smoothers record which approximation they used:

.. testcode::

   eks = curved.extended_smoother(y_nl)
   uks = curved.unscented_smoother(y_nl)
   def rmse(path):
       return round(float(np.sqrt(np.mean((path[:, 0] - state) ** 2))), 3)

   print(eks.method, uks.method)
   print(rmse(ukf.filtered_state), rmse(uks.smoothed_state))

.. testoutput::

   extended unscented
   0.524 0.457

Non-Gaussian models through hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three keyword hooks take a model outside the additive-Gaussian form:

``observation_loglik(obs, particles)``
   Replaces the Gaussian measurement density: ``obs`` is one ``(p,)``
   row, ``particles`` is ``(N, m)``, and it returns ``(N,)``
   log-densities. ``obs_cov`` is then ignored by the particle filter.
``transition_sampler(particles, rng)``
   Replaces the additive state draw; returns ``(N, m)``.
``observed_transition(particles, y_prev)``
   A state mean that conditions on the previous observation; noise stays
   :math:`N(0, Q)`.

With any hook set, ``is_additive_gaussian`` is ``False`` and the
extended and unscented filters and smoothers raise
:class:`~cultivars.exceptions.SpecificationError` instead of
linearizing. The particle filter accepts all three hooks. The particle
smoother accepts ``observation_loglik`` but refuses
``transition_sampler``, because the backward pass needs the transition
*density*, and refuses a singular ``state_cov`` for the same reason. A
stochastic-volatility measurement, :math:`y_t \mid h_t \sim N(0,
e^{h_t})`:

.. testcode::

   def measure(obs, particles):
       h = particles[:, 0]
       return -0.5 * (np.log(2 * np.pi) + h + obs[0] ** 2 * np.exp(-h))

   returns = np.exp(0.5 * state) * rng.standard_normal(n)
   sv = NonlinearSSM(
       lambda s: 0.9 * s, lambda s: s,
       state_cov=[[0.25]], obs_cov=[[1.0]],
       initial_state=[0.0], initial_state_cov=[[0.25 / 0.19]],
       observation_loglik=measure,
   )
   print(sv.is_additive_gaussian)
   try:
       sv.unscented_filter(returns)
   except Exception as err:
       print(type(err).__name__)
   track = sv.particle_filter(returns, n_particles=500, seed=0)
   print(round(float(np.corrcoef(track.filtered_state[:, 0], state)[0, 1]), 2))

.. testoutput::

   False
   SpecificationError
   0.52

Discrete regimes: the Hamilton filter
-------------------------------------

The model
~~~~~~~~~

:class:`~cultivars.state_space.regime_switching.MarkovSwitchingSSM` is an
autoregression whose intercept and variance switch with a :math:`K`-state
chain :math:`S_t`,

.. math::

   y_t = \mu_{S_t} + \sum_{i=1}^{p} \phi_{i,S_t}\, y_{t-i}
   + \sigma_{S_t} e_t,
   \qquad
   P_{ij} = \Pr(S_t = j \mid S_{t-1} = i).

Because the regime enters only through the current intercept and
variance, the per-regime density :math:`\eta_{t,j}` depends on
:math:`S_t` alone and the Hamilton (1989) filter

.. math::

   \xi_{t \mid t-1} = P'\,\xi_{t-1 \mid t-1},\qquad
   \xi_{t \mid t} = \frac{\xi_{t \mid t-1} \odot \eta_t}
                         {\mathbf 1'(\xi_{t \mid t-1} \odot \eta_t)},\qquad
   \log L = \sum_t \log \mathbf 1'(\xi_{t \mid t-1} \odot \eta_t)

is exact. The Kim smoother returns full-sample probabilities and the
consecutive-pair probabilities an EM step needs.

.. list-table::
   :header-rows: 1
   :widths: 24 26 50

   * - Argument
     - Shape
     - Meaning
   * - ``transition``
     - ``(K, K)``
     - Row-stochastic: row ``i`` is the distribution of :math:`S_t`
       given :math:`S_{t-1} = i`. Validated.
   * - ``intercepts``
     - ``(K,)``
     - :math:`\mu_j`
   * - ``ar_params``
     - ``(K, p)``
     - :math:`\phi_{\cdot, j}`; width zero is a switching mean with no
       dynamics.
   * - ``variances``
     - ``(K,)``
     - :math:`\sigma^2_j`, strictly positive
   * - ``initial_prob``
     - ``(K,)``
     - :math:`\Pr(S_1)`; defaults to the ergodic distribution

.. testcode::

   from cultivars.state_space.regime_switching import (
       MarkovSwitchingSSM, RegimeSwitchingLinearSSM,
   )

   rng = np.random.default_rng(3)
   chain = np.array([[0.95, 0.05], [0.10, 0.90]])
   n = 300
   regime = np.zeros(n, dtype=int)
   z = np.zeros(n)
   for t in range(1, n):
       regime[t] = rng.choice(2, p=chain[regime[t - 1]])
       z[t] = [0.8, -1.0][regime[t]] + 0.4 * z[t - 1] + 0.6 * rng.standard_normal()

   ms = MarkovSwitchingSSM(chain, [0.8, -1.0], [[0.4], [0.4]], [0.36, 0.36])
   print(ms.n_regimes, ms.order, ms.initial_prob.round(3).tolist())
   hamilton = ms.filter(z)
   kim = ms.smooth(z)
   print(hamilton.filtered_prob.shape, round(hamilton.loglikelihood, 2))
   print(kim.smoothed_prob.shape, kim.smoothed_joint_prob.shape)
   truth = regime[ms.order:] == 1
   print(round(float(np.mean((hamilton.filtered_prob[:, 1] > 0.5) == truth)), 2),
         round(float(np.mean((kim.smoothed_prob[:, 1] > 0.5) == truth)), 2))

.. testoutput::

   2 1 [0.667, 0.333]
   (299, 2) -341.35
   (299, 2) (298, 2, 2)
   0.97 0.99

The probability arrays have ``n - order`` rows: the first ``order``
observations are conditioned on, not modelled. Row ``t`` of
``filtered_prob`` belongs to observation ``t + order``.

The recursions never see the data, only an ``(n - order, K)`` matrix of
log densities. ``effective_sample``, ``density_matrix``,
``filter_densities`` and ``smooth_densities`` expose that split, so an
EM loop can build the lag block once, and a switching model of another
form can supply its own log densities:

.. testcode::

   target, lags = ms.effective_sample(z)
   logd = ms.density_matrix(target, lags)
   print(logd.shape,
         bool(np.isclose(ms.filter_densities(logd).loglikelihood,
                         hamilton.loglikelihood)))

.. testoutput::

   (299, 2) True

Switching linear systems: Kim's collapse
----------------------------------------

:class:`~cultivars.state_space.regime_switching.RegimeSwitchingLinearSSM`
gives each regime its own linear-Gaussian system,

.. math::

   y_t = Z_j \alpha_t + d_j + \varepsilon_t,\quad
   \varepsilon_t \sim N(0, H_j),
   \qquad
   \alpha_{t+1} = T_j \alpha_t + c_j + \eta_t,\quad
   \eta_t \sim N(0, Q_j),
   \qquad j = S_t.

Exact filtering would need one Kalman filter per regime path, so the
engine applies Kim's (1994) collapse: :math:`K \times K` pairwise
updates, weighted by the Hamilton probabilities and moment-matched back
to one mean and covariance per regime. The likelihood is therefore an
approximation (Kim and Nelson 1999). All arguments are keyword-only and
stacked on a leading regime axis: ``design`` ``(K, p, m)``, ``obs_cov``
``(K, p, p)``, ``transition`` ``(K, m, m)``, ``state_cov`` ``(K, m, m)``
(no selection matrix), ``regime_transition`` ``(K, K)``, and optional
``obs_intercept`` ``(K, p)``, ``state_intercept`` ``(K, m)``,
``initial_state`` ``(m,)``, ``initial_state_cov`` ``(m, m)`` and
``initial_regime_prob`` ``(K,)``.

A state whose innovation variance switches between a calm and a
volatile regime:

.. testcode::

   x = np.zeros(n)
   calm_or_not = np.zeros(n, dtype=int)
   for t in range(1, n):
       calm_or_not[t] = rng.choice(2, p=chain[calm_or_not[t - 1]])
       scale = np.sqrt([0.05, 1.0][calm_or_not[t]])
       x[t] = 0.9 * x[t - 1] + scale * rng.standard_normal()
   obs = x + np.sqrt(0.5) * rng.standard_normal(n)

   switching = RegimeSwitchingLinearSSM(
       design=np.ones((2, 1, 1)), obs_cov=np.full((2, 1, 1), 0.5),
       transition=np.full((2, 1, 1), 0.9),
       state_cov=np.array([[[0.05]], [[1.0]]]),
       regime_transition=chain,
   )
   print(switching.k_regimes, switching.k_states, switching.k_endog)
   collapsed = switching.filter(obs)
   print(collapsed.filtered_prob.shape, collapsed.filtered_state.shape,
         collapsed.regime_state.shape)
   print(round(collapsed.loglikelihood, 2))
   print(switching.smooth(obs).smoothed_prob.shape)

.. testoutput::

   2 1 1
   (300, 2) (300, 1) (300, 2, 1)
   -374.8
   (300, 2)

``filtered_state`` and ``filtered_state_cov`` are regime-marginal (the
covariance includes the between-regime spread); ``regime_state`` holds
each regime's collapsed mean. Unlike the Hamilton engine, the Kim
filter keeps all ``n`` rows. ``smooth`` smooths the regime
probabilities only: a smoothed continuous state would need a second
collapse and is not offered.

How the pieces connect
----------------------

Most users meet these engines through a fitted model. The model owns
the parameterization and the estimator; the engine owns the recursions.
Results that expose their engine let you re-filter new data with the
fitted parameters, which is the out-of-sample check the fitted arrays
alone cannot give.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Fitted object
     - Accessor
     - Engine returned
   * - :class:`~cultivars.univariate.unobserved_components.UnobservedComponentsResult`
     - ``state_space`` (property)
     - ``LinearGaussianSSM``
   * - :class:`~cultivars.multivariate.reduced_form.term_structure.DynamicNelsonSiegelResult`
     - ``state_space`` (property)
     - ``LinearGaussianSSM``
   * - :class:`~cultivars.multivariate.reduced_form.term_structure.TimeVaryingNelsonSiegelResult`
     - ``state_space`` (property)
     - ``NonlinearSSM``
   * - :class:`~cultivars.univariate.stochastic_volatility.SVResult`
     - ``state_space``; ``quasi_state_space``
     - ``NonlinearSSM`` (particle filter only); ``LinearGaussianSSM``
       on :math:`\log y_t^2`
   * - :class:`~cultivars.univariate.stochastic_volatility.SVPosterior`,
       :class:`~cultivars.univariate.stochastic_volatility.UCSVPosterior`
     - ``state_space`` (property)
     - ``NonlinearSSM``
   * - :class:`~cultivars.multivariate.structural.stochastic_volatility.LongMemorySVResult`
     - ``quasi_state_space(truncation=100)``
     - ``LinearGaussianSSM``
   * - :class:`~cultivars.univariate.regime_switching.MarkovSwitchingARResult`
     - ``state_space`` (property)
     - ``MarkovSwitchingSSM``
   * - :class:`~cultivars.univariate.threshold.SETARResult`,
       :class:`~cultivars.univariate.smooth_transition.STARResult`,
       :class:`~cultivars.univariate.mean_function.ARNNResult`,
       :class:`~cultivars.univariate.mean_function.TARNNResult`
     - ``nonlinear_state_space(measurement_variance)``
     - ``NonlinearSSM`` (errors-in-variables reading of the fit)
   * - :class:`~cultivars.multivariate.structural.perturbation.PerturbationDSGEResult`,
       :class:`~cultivars.multivariate.structural.perturbation.PerturbationDSGEPosterior`
     - ``state_space`` (property)
     - ``LinearGaussianSSM`` at first order, ``NonlinearSSM`` at second
   * - :class:`~cultivars.multivariate.reduced_form.mixed_frequency.MFVAR`
     - ``filter(endog)``, ``smooth(endog)``
     - runs ``LinearGaussianSSM`` internally; ``filter`` returns its
       ``KalmanFilterResult``

Other models use an engine without exposing it:
:class:`~cultivars.univariate.box_jenkins.ARIMA` evaluates its exact
likelihood through the linear engine's ARMA form,
:class:`~cultivars.multivariate.regime_switching.markov_switching.MarkovSwitchingVAR`
runs the Hamilton filter and Kim smoother inside its EM loop, and
:class:`~cultivars.multivariate.regime_switching.markov_switching.MarkovSwitchingDFM`
is maximized through ``RegimeSwitchingLinearSSM``.
:meth:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM.mixed_frequency_state_space`
builds the mixed-frequency system from a coefficient stack, for use
outside ``MFVAR``.

The emitted system reproduces the fitted likelihood. For an
unobserved-components local level:

.. testcode::

   from cultivars.univariate.unobserved_components import UnobservedComponents

   rng = np.random.default_rng(0)
   series = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
   uc = UnobservedComponents(series).fit()
   system = uc.state_space
   print(system.k_states,
         bool(abs(system.loglikelihood(series) - uc.llf) < 1e-8))
   print(bool(np.allclose(system.smooth(series).smoothed_state[:, 0], uc.level)))
   print(system.filter(series[:50]).filtered_state.shape)

.. testoutput::

   1 True
   True
   (50, 1)

Call an engine directly when the system is yours: a model the library
does not estimate, a likelihood you will maximize with your own
optimizer or sampler (build the engine inside the objective and call
``loglikelihood``), a fitted system evaluated on a new sample, or a
stress test of a fitted system under modified matrices.

Nuances and gotchas
-------------------

* **Result records are plain dataclasses.** None of the eight records
  has ``summary()``, ``to_pandas()`` or the comparison methods of
  fitted results; their repr is the default dataclass repr, which prints
  every array in full (over 20,000 characters for the 200-period local
  level) under the private class name:

  .. testcode::

     print(hasattr(filtered, "summary"), len(repr(filtered)) > 20000)
     print(type(filtered).__name__)

  .. testoutput::

     False True
     _KalmanFilterResult

  Read the fields by name instead of printing the record.
* **The diffuse prior is approximate.** :math:`P_1 = 10^6 I` puts about
  :math:`-\tfrac12\log 10^6 \approx -6.9` into the first contributions
  for each nonstationary state. Likelihoods are comparable across
  specifications with the same number of diffuse states on the same
  sample, not across different numbers of them. A time-varying
  ``transition``, ``selection`` or ``state_cov`` also triggers the
  diffuse default even when every :math:`T_t` is stable; pass
  ``initial_state_cov`` explicitly in that case.
* **The default initial mean is zero.** See `Initialization`_: with a
  ``state_intercept``, set ``initial_state`` to the stationary mean.
* **Missing-data rules differ by engine.** ``LinearGaussianSSM``
  accepts ``nan`` in any element. ``NonlinearSSM`` and
  ``RegimeSwitchingLinearSSM`` accept only whole missing rows and raise
  :class:`~cultivars.exceptions.SpecificationError` on a partially
  observed row. ``MarkovSwitchingSSM.filter`` raises on any non-finite
  value, because the lags of a missing value poison later densities;
  interpolate first or use ``RegimeSwitchingLinearSSM``.
* **Row counts differ.** Hamilton-filter arrays have ``n - order``
  rows; Kalman, Kim and particle arrays have ``n``. Align by index
  before comparing a regime path with anything dated on the full
  sample, such as a :class:`~cultivars.spectral.cycles.TurningPointsResult`
  contraction indicator.
* **Default initializations differ between the linear engines.**
  ``RegimeSwitchingLinearSSM`` defaults to :math:`P_1 = 100\, I`, not
  the stationary covariance, so two identical regimes do not reproduce
  ``LinearGaussianSSM`` until you pass the same ``initial_state_cov``:

  .. testcode::

     same = dict(
         design=np.ones((2, 1, 1)), obs_cov=np.ones((2, 1, 1)),
         transition=np.full((2, 1, 1), 0.8), state_cov=np.full((2, 1, 1), 0.5),
         regime_transition=chain,
     )
     kalman = LinearGaussianSSM([[1.0]], [[1.0]], [[0.8]], [[1.0]], [[0.5]])
     exact = kalman.loglikelihood(obs)
     print(round(RegimeSwitchingLinearSSM(**same).loglikelihood(obs) - exact, 3))
     matched = RegimeSwitchingLinearSSM(**same, initial_state_cov=[[0.5 / 0.36]])
     print(bool(abs(matched.loglikelihood(obs) - exact) < 1e-8))

  .. testoutput::

     -1.919
     True

* **A bad chain raises different exceptions.** A transition matrix that
  is not row-stochastic raises
  :class:`~cultivars.exceptions.SpecificationError` in
  ``MarkovSwitchingSSM`` and :class:`~cultivars.exceptions.NumericalError`
  in ``RegimeSwitchingLinearSSM``. Catch both if you validate user input.
* **The two nonlinear notations.** The ``NonlinearSSM`` docstrings call
  the measurement covariance :math:`R`; on this page it is :math:`H`, to
  match the linear engine, where :math:`R` is the selection matrix. The
  argument names (``obs_cov``, ``state_cov``) are the same in both
  engines.
* **Unscented tuning must match.** ``unscented_smoother`` reruns the
  filter with its own ``alpha``, ``beta`` and ``kappa``; pass the same
  values you gave ``unscented_filter`` or the two will not correspond.
* **Particle results do not keep the cloud.** ``ParticleFilterResult``
  carries cloud means and standard deviations, not particles or
  weights. ``particle_smoother`` costs :math:`O(N^2)` per period and
  stores every forward cloud, which is why its default ``n_particles`` is
  500 rather than 2000.
* **Seeds.** ``particle_filter``, ``particle_smoother``, ``simulate``
  and ``simulation_smoother`` take ``seed`` (an int or a
  ``numpy.random.Generator``); with ``seed=None`` every call differs.
* **No forecasting method.** The engines have no ``forecast``. Append
  ``nan`` rows to the data and read ``predicted_state`` from the filter,
  or use the ``forecast`` of the fitted model that emitted the engine.
* **The engines do not estimate.** There is no ``fit`` and no parameter
  vector. To estimate a custom system, build the engine inside your
  objective function and maximize ``loglikelihood``.

See also
--------

* :doc:`univariate_linear`: unobserved components and ARIMA, whose
  likelihoods run on the linear engine.
* :doc:`volatility`: stochastic-volatility models on the nonlinear
  engine.
* :doc:`univariate_nonlinear` and :doc:`nonlinear_multivariate`:
  Markov-switching, threshold and smooth-transition models.
* :doc:`var_extensions`: mixed-frequency VARs and the dynamic
  Nelson-Siegel model.
* :doc:`spectral`: the frequency-domain views of fitted systems.
* :doc:`conventions` and :doc:`nuances`: package-wide conventions.
* API: :mod:`cultivars.state_space.linear_gaussian`,
  :mod:`cultivars.state_space.nonlinear`,
  :mod:`cultivars.state_space.regime_switching`.

References: Durbin and Koopman (2002, 2012); Harvey (1989); Hamilton
(1989, 1994); Kim (1994); Kim and Nelson (1999); Julier and Uhlmann
(1997); Gordon, Salmond and Smith (1993); Pitt and Shephard (1999);
Doucet, Godsill and Andrieu (2000); Särkkä (2013).
