.. meta::
   :description: Threshold, smooth-transition, functional-coefficient,
      quantile, time-varying-parameter and Markov-switching VARs in
      cultivars, with regime-, state- and date-indexed responses.

Nonlinear, time-varying and regime-switching VARs
=================================================

Every model on this page lets the VAR coefficient stack move. They differ
in what it moves with: an observed state (threshold, smooth-transition and
functional-coefficient VARs), the quantile level (quantile VAR), the date
(time-varying-parameter VARs, Bayesian), or a latent Markov chain
(Markov-switching VAR). Use them to ask whether transmission differs across
regimes, states, tails or periods. The constant-coefficient VAR is in
:doc:`var`, Bayesian constant-coefficient VARs (including stochastic
volatility with fixed coefficients) in :doc:`bayesian_vars`, and the
univariate counterparts in :doc:`univariate_nonlinear`.

.. contents:: On this page
   :local:
   :depth: 1

The family at a glance
----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 26 30 22

   * - Class
     - Module
     - Model
     - Use when
   * - :class:`~cultivars.multivariate.nonlinear.threshold.TVAR`
     - :mod:`cultivars.multivariate.nonlinear.threshold`
     - Two complete VARs (coefficients and covariance) split at
       :math:`z_{t-d} \le r`
     - An observable crosses a level and the whole mechanism changes
       (credit regimes, Balke 2000)
   * - :class:`~cultivars.multivariate.nonlinear.smooth_transition.STVAR`
     - :mod:`cultivars.multivariate.nonlinear.smooth_transition`
     - Convex blend of two VARs, logistic or exponential weight in
       :math:`z_{t-d}`
     - Dates can be partly in each regime (state-dependent multipliers,
       Auerbach and Gorodnichenko 2012)
   * - :class:`~cultivars.multivariate.nonlinear.functional_coefficient.FunctionalCoefficientVAR`
     - :mod:`cultivars.multivariate.nonlinear.functional_coefficient`
     - Every coefficient an unknown smooth function of :math:`z_{t-d}`,
       local-linear kernel
     - You need the shape of the state dependence, or a check on TVAR and
       STVAR
   * - :class:`~cultivars.multivariate.nonlinear.quantile.QVAR`
     - :mod:`cultivars.multivariate.nonlinear.quantile`
     - One coefficient stack per conditional quantile
     - Tail dynamics differ from the mean's (growth at risk)
   * - :class:`~cultivars.multivariate.nonlinear.time_varying.TVPVAR`
     - :mod:`cultivars.multivariate.nonlinear.time_varying`
     - Random-walk coefficients, constant covariance, Gibbs sampler
     - Transmission drifts slowly over the sample
   * - :class:`~cultivars.multivariate.nonlinear.time_varying.TVPVARSV`
     - :mod:`cultivars.multivariate.nonlinear.time_varying`
     - Random-walk coefficients and log volatilities (Primiceri 2005)
     - "Has policy changed, or have the shocks?"
   * - :class:`~cultivars.multivariate.nonlinear.time_varying.TVPSVAR`
     - :mod:`cultivars.multivariate.nonlinear.time_varying`
     - Recursive identification of a fitted TVP result, date by date
     - Structural responses that drift
   * - :class:`~cultivars.multivariate.regime_switching.markov_switching.MarkovSwitchingVAR`
     - :mod:`cultivars.multivariate.regime_switching.markov_switching`
     - Intercepts, lags and covariance switch with a latent chain, EM
     - Regimes are not observable (Krolzig 1997; Sims and Zha 2006)
   * - :class:`~cultivars.multivariate.regime_switching.markov_switching.MarkovSwitchingSVAR`
     - :mod:`cultivars.multivariate.regime_switching.markov_switching`
     - One identification scheme applied in every regime
     - Structural responses per latent regime

The same module also holds
:class:`~cultivars.multivariate.regime_switching.markov_switching.MarkovSwitchingDFM`,
the Kim and Nelson (1998) one-factor model with a switching intercept,
which returns a recession-probability chronology.

Choosing among them
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Question
     - First choice
     - Estimation and output
   * - Does transmission change when an observable crosses a level?
     - TVAR, then STVAR; FC-VAR to check the shape
     - Least squares; likelihood and information criteria
   * - How does a coefficient vary over the whole range of a state?
     - FC-VAR
     - Kernel; curves with pointwise bands, no likelihood
   * - Do the tails respond differently from the median?
     - QVAR
     - Linear programs; check loss and pseudo :math:`R^1`, no likelihood
   * - Has the system drifted over time?
     - TVPVAR or TVPVARSV
     - Gibbs posterior; paths with 68% bands, no likelihood
   * - Are there unobserved regimes (recessions, policy regimes)?
     - MarkovSwitchingVAR
     - EM; posterior regime probabilities, likelihood and criteria

The likelihood-based results (TVAR, STVAR, MS-VAR) rank against one
another and against a linear
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
with ``compare``. The kernel, quantile and Bayesian results report no
``llf``, no ``n_params`` and no information criteria, and have no
``compare``.

The common contract: every response names where it is asked
-----------------------------------------------------------

A nonlinear VAR has no single companion matrix. Every propagation method
here takes a keyword that says which linear system to propagate, and the
answer holds that system frozen over the horizon:

.. list-table::
   :header-rows: 1
   :widths: 34 26 40

   * - Result
     - Keyword
     - Methods
   * - ``TVARResult``, ``STVARResult``
     - ``regime="lower"`` or ``"upper"``
     - ``irf``, ``fevd``, ``ma_representation``, ``equation``
   * - ``FunctionalCoefficientVARResult``
     - ``u=<state value>``
     - ``irf``, ``coefficients_at``
   * - ``QVARResult``
     - ``quantile=<level>``
     - ``irf``, ``ma_representation``, ``coefficients``, ``forecast``
   * - ``TVPVARResult``, ``TVPSVARResult``
     - ``t=<row>`` (position)
     - ``coefficients_at``, ``impact``, ``irf``
   * - ``MarkovSwitchingSVARResult``
     - ``regime=<int>``
     - ``irf``, ``impact``, ``structural``

These are linearizations: exact only while the shock does not move the
transition variable across the split, the state does not move, the
parameters do not drift, or the chain stays put. Generalized impulse
responses that average over histories and future shocks (Koop, Pesaran and
Potter, 1996) are **not implemented** for any model on this page. The
``generalized_irf`` on linear VAR results is the Pesaran-Shin
order-invariant response of a linear system, a different object.

Threshold VAR
-------------

The model
~~~~~~~~~

Two complete VARs glued along a threshold on an observed variable (Tsay,
1998):

.. math::

   y_t = \begin{cases}
   c_L + \sum_{i=1}^{p} A_{L,i}\, y_{t-i} + u_t,
   & u_t \sim N(0, \Sigma_L), \quad z_t \le r, \\
   c_U + \sum_{i=1}^{p} A_{U,i}\, y_{t-i} + u_t,
   & u_t \sim N(0, \Sigma_U), \quad z_t > r,
   \end{cases}
   \qquad z_t = s_{t-d},

where :math:`s` is a column of the system (self-exciting) or an external
series. The likelihood is a step function of :math:`r`, so the estimator
is an exhaustive grid over trimmed quantiles of :math:`z` (and over
delays ``1..order`` when the delay is left open), with regime-wise least
squares at each candidate. The criterion is the total Gaussian
log-likelihood with a separate covariance per regime, so the noisiest
equation does not choose the split.

The running example is persistent and quiet when ``y1``'s lag is at or
below zero, and mean-reverting and louder above it:

.. testcode::

   import numpy as np
   from cultivars.multivariate.nonlinear.threshold import TVAR
   from cultivars.multivariate.reduced_form.vector_autoregression import VAR

   rng = np.random.default_rng(7)
   y = np.zeros((300, 2))
   calm = np.array([[0.6, 0.0], [0.2, 0.5]])
   tight = np.array([[-0.2, 0.3], [0.1, -0.3]])
   for t in range(1, 300):
       above = y[t - 1, 0] > 0.0
       y[t] = (tight if above else calm) @ y[t - 1]
       y[t] += (1.5 if above else 1.0) * rng.standard_normal(2)

   tvar = TVAR(y, order=1, transition_variable="y1", delay=1).fit()
   print(round(tvar.threshold, 3), tvar.n_lower, tvar.n_upper, tvar.nobs, tvar.n_params)
   print(tvar.lower_coefficients[0].round(2))
   print(tvar.upper_coefficients[0].round(2))
   print(np.diag(tvar.lower_sigma_u).round(2), np.diag(tvar.upper_sigma_u).round(2))

.. testoutput::

   -0.012 216 83 299 19.0
   [[ 0.61 -0.  ]
    [ 0.36  0.44]]
   [[-0.31  0.32]
    [-0.09 -0.31]]
   [0.94 0.92] [1.34 1.52]

``lower_coefficients`` and ``upper_coefficients`` are ``(p, k, k)`` lag
stacks; entry ``[i - 1, r, c]`` is the coefficient on lag ``i`` of
variable ``c`` in equation ``r``. ``equation(name, regime=...)`` returns
one equation as a dict keyed ``const``, ``y1.L1``, ...

Regime-conditional propagation and forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. testcode::

   print(tvar.irf(4, regime="lower")[:, 0, 0].round(3))
   print(tvar.irf(4, regime="upper")[:, 0, 0].round(3))
   print(tvar.fevd(8, regime="upper")[-1].round(2))
   print(tvar.forecast(3).round(3))

.. testoutput::

   [0.967 0.593 0.363 0.223 0.136]
   [ 1.157 -0.361  0.08  -0.005 -0.007]
   [[0.87 0.13]
    [0.01 0.99]]
   [[-0.76   0.033]
    [-0.496 -0.319]
    [-0.334 -0.377]]

``irf`` is orthogonalized by default with the regime's own Cholesky
factor, so the ordering of ``names`` is an identifying assumption and the
impact differs across regimes because the covariances do. Pass
``orthogonalized=False`` for the moving-average matrices, ``cumulative=True``
for running sums. ``forecast(steps)`` is the skeleton iteration: at each
step the weight is evaluated on the forecast path itself. It is exact at
one step and an approximation beyond, because the multi-step conditional
mean averages the switch over future shocks.

Specification choices
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - Argument
     - Default
     - Meaning
   * - ``order``
     - required
     - Lag order within each regime
   * - ``transition_variable``
     - required
     - A name from ``names`` (self-exciting) or an aligned 1-D array
       (external)
   * - ``delay``
     - ``None``
     - ``None`` searches ``1..order`` (self-exciting only); an external
       driver requires an explicit delay, otherwise ``SpecificationError``
   * - ``trim``
     - ``0.15``
     - Fraction trimmed from each tail of the threshold grid
   * - ``n_grid``
     - ``300``
     - Candidate thresholds per delay
   * - ``trend``, ``names``
     - ``"c"``, ``y1..yk``
     - Deterministic terms per regime; variable labels

With an external driver, ``threshold_name`` is ``"external"`` and a
forecast beyond ``delay`` steps needs the driver's future path:

.. testcode::

   driver = np.random.default_rng(3).standard_normal(300)
   ext = TVAR(y, order=1, transition_variable=driver, delay=1).fit()
   print(ext.threshold_name, ext.self_exciting)
   print(ext.forecast(3, transition_future=[0.1, 0.2]).shape)

.. testoutput::

   external False
   (3, 2)

Without ``transition_future`` that call raises ``SpecificationError``:
the model holds no process for the driver and does not extrapolate one.

What the result gives you
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~cultivars.multivariate.nonlinear.threshold.TVARResult`:
``threshold``, ``delay``, ``searched_delay``, ``threshold_name``,
``threshold_values`` (the delayed transition variable aligned with
``resid``), ``transition_series`` (the raw series aligned with
``endog``), ``lower_coefficients``, ``upper_coefficients``,
``lower_deterministic``, ``upper_deterministic``, ``lower_sigma_u``,
``upper_sigma_u``, ``n_lower``, ``n_upper``, ``regime_weight``,
``upper_fraction``, ``lower_stability``, ``upper_stability``,
``is_regimewise_stationary``, ``fittedvalues``, ``resid``, ``llf``,
``information_criteria``, ``compare``, plus the methods above.
``likelihood_ratio_test`` always raises ``SpecificationError``: the
threshold is not identified under linearity (Hansen, 1996).

Smooth-transition VAR
---------------------

The model
~~~~~~~~~

.. math::

   y_t = \bigl(1 - G(z_t)\bigr)\bigl(c_L + \textstyle\sum_i A_{L,i} y_{t-i}\bigr)
   + G(z_t)\bigl(c_U + \textstyle\sum_i A_{U,i} y_{t-i}\bigr) + u_t,
   \qquad u_t \sim N(0, \Sigma),

with :math:`G(z) = (1 + e^{-\gamma (z - c)/\sigma_z})^{-1}`
(``transition="logistic"``, directional regimes) or
:math:`G(z) = 1 - e^{-\gamma (z - c)^2/\sigma_z^2}`
(``transition="exponential"``, the "upper" regime is both tails). The
speed ``gamma`` is per standard deviation of the transition variable.
Conditional on :math:`(\gamma, c)` the model is linear, so the regime
coefficients come from one weighted least-squares solve and only the pair
is searched, multi-start and derivative-free, on the log-determinant of
the residual covariance (Teräsvirta and Yang, 2014). The innovation
covariance is common across regimes: smooth weights never partition the
sample, so there is no subsample to estimate a per-regime covariance from.

.. testcode::

   from cultivars.multivariate.nonlinear.smooth_transition import STVAR

   rng = np.random.default_rng(1)
   x = np.zeros((400, 2))
   for t in range(1, 400):
       g = 1.0 / (1.0 + np.exp(-2.0 * x[t - 1, 0]))
       x[t] = ((1.0 - g) * 0.6 - g * 0.3) * x[t - 1] + rng.standard_normal(2)

   stvar = STVAR(x, order=1, transition_variable="y1").fit()
   print(stvar.transition, round(stvar.gamma, 2), round(stvar.transition_scale, 2))
   print(round(stvar.threshold, 2), stvar.n_params, stvar.is_effectively_abrupt)
   print(np.diag(stvar.lower_coefficients[0]).round(2))
   print(np.diag(stvar.upper_coefficients[0]).round(2))
   w = stvar.regime_weight
   print(round(float(np.mean((w > 0.1) & (w < 0.9))), 2))

.. testoutput::

   logistic 1.94 1.09
   0.32 17.0 False
   [0.64 0.54]
   [-0.41 -0.5 ]
   0.64

``gamma / transition_scale`` (about 1.8 here, against a true 2) is the
speed in the units of ``y1``. Nearly two thirds of the dates carry a
weight strictly between 0.1 and 0.9, which no hard split can represent.

On the TVAR sample the smooth fit collapses onto the threshold model:
``gamma`` runs off to thousands, ``is_effectively_abrupt`` turns
``True``, and the reported ``gamma`` is only a lower bound because the
likelihood is flat in it. The two then rank by information criteria:

.. testcode::

   collapsed = STVAR(y, order=1, transition_variable="y1").fit()
   print(collapsed.is_effectively_abrupt, collapsed.gamma > 1000)
   table = tvar.compare(collapsed, VAR(y, order=1).fit(), criterion="bic")
   for row in table.rows:
       print(row[0], row[2], row[4])

.. testoutput::

   True True
   STVAR(1, logistic, d=1) 17 0.000
   TVAR(1, d=1) 19 0.836
   VAR(1) 9 36.879

Arguments are those of TVAR without ``trim`` and ``n_grid``, plus
``transition`` (``"logistic"`` or ``"exponential"``); ``delay`` defaults
to 1 and is never searched. The result,
:class:`~cultivars.multivariate.nonlinear.smooth_transition.STVARResult`,
has the TVAR surface with ``sigma_u`` in place of the two covariances and
adds ``transition``, ``gamma``, ``transition_scale`` and
``is_effectively_abrupt``. ``irf(regime=...)`` uses the common covariance,
so the impact is identical in both regimes and only the propagation
differs.

Functional-coefficient VAR
--------------------------

The model
~~~~~~~~~

Every coefficient is an unknown smooth function of the state (Chen and
Tsay, 1993; Cai, Fan and Yao, 2000):

.. math::

   y_t = c(z_t) + \sum_{i=1}^{p} A_i(z_t)\, y_{t-i} + u_t,
   \qquad z_t = s_{t-d}.

At each grid point :math:`u` one Epanechnikov-weighted local-linear least
squares on :math:`[x_t,\ x_t (z_t - u)]` gives the level and slope of
every coefficient; the level is the reported curve. The bandwidth is
chosen by exact leave-one-out cross-validation around the Silverman rule
unless you state it. There is no likelihood; the complexity measure is
``effective_params``, the trace of the smoother matrix.

On the smooth-transition sample, the own-lag curve of ``y1`` falls from
the lower regime's value toward the upper regime's as the state rises:

.. testcode::

   from cultivars.multivariate.nonlinear.functional_coefficient import (
       FunctionalCoefficientVAR,
   )

   model = FunctionalCoefficientVAR(x, order=1, transition_variable="y1")
   fcvar = model.fit(n_grid=21)
   print(round(fcvar.bandwidth, 2), fcvar.bandwidth_searched)
   print(round(fcvar.effective_params, 1))
   print([round(v, 2) for v in fcvar.state_range], fcvar.curves.shape)
   curve = fcvar.coefficient_curve("y1", "y1.L1")
   print(curve[::5].round(2))

.. testoutput::

   2.3 True
   4.2
   [-2.43, 1.14] (21, 3, 2)
   [[ 0.55  0.67  0.79]
    [ 0.48  0.56  0.63]
    [ 0.25  0.31  0.37]
    [-0.    0.07  0.13]
    [-0.28 -0.18 -0.07]]

``coefficient_curve`` returns ``(low, point, high)`` with one pointwise
standard error each side, conditional on the bandwidth. ``curves`` has
shape ``(G, w, k)``: ``curves[g, :, i]`` is equation ``i``'s coefficient
vector at ``grid[g]``, deterministic terms first.

State-indexed outputs
~~~~~~~~~~~~~~~~~~~~~

.. testcode::

   print(fcvar.coefficients_at(0.0).round(2))
   print(fcvar.irf(3, u=-1.0)[:, 0, 0].round(3))
   print(fcvar.irf(3, u=1.0)[:, 0, 0].round(3))
   print(fcvar.stability_curve()[[0, -1]].round(2), fcvar.forecast().round(3))

.. testoutput::

   [[[ 0.13 -0.02]
     [ 0.06  0.14]]]
   [0.967 0.403 0.168 0.07 ]
   [ 0.967 -0.134  0.015 -0.001]
   [0.6  0.23] [-1.076 -0.825]

``coefficients_at(u)`` interpolates linearly between grid points and
raises ``SpecificationError`` for a state outside ``state_range``: kernel
estimates do not extrapolate. ``forecast()`` takes no arguments and is one
step only, because the state for :math:`T + 1` is already observed at
:math:`T`.

``fit`` takes ``bandwidth`` (``None`` to cross-validate), ``n_grid``
(default 51) and ``trim`` (quantile trimmed from each end of the state
range, default 0.05). The constructor takes ``order``,
``transition_variable``, ``delay`` (default 1, never searched), ``trend``
and ``names``.

Quantile VAR
------------

The model
~~~~~~~~~

Each conditional quantile is linear in the system's lags, with its own
coefficient stack at every level (White, Kim and Manganelli, 2015):

.. math::

   Q_\tau(y_{i,t} \mid y_{t-1}, \ldots, y_{t-p})
   = c_i(\tau)' d_t + \sum_{l=1}^{p} a_{i,l}(\tau)' y_{t-l},

estimated by the Koenker and Bassett (1978) check-loss program, solved
exactly as a linear program by HiGHS, equation by equation and level by
level. There is no error term and no innovation covariance.

The example is a growth-at-risk system in which financial conditions
``fci`` scale the downside shocks to ``growth`` (Adrian, Boyarchenko and
Giannone, 2019):

.. testcode::

   from cultivars.multivariate.nonlinear.quantile import QVAR

   rng = np.random.default_rng(8)
   g = np.zeros((400, 2))
   for t in range(1, 400):
       fci = 0.8 * g[t - 1, 1] + 0.5 * rng.standard_normal()
       growth = 0.3 * g[t - 1, 0] - 0.2 * g[t - 1, 1]
       growth += np.exp(0.5 * g[t - 1, 1]) * 0.5 * rng.standard_normal()
       g[t] = [growth, fci]

   qvar = QVAR(g, order=1, names=["growth", "fci"]).fit(quantiles=(0.1, 0.5, 0.9))
   print(qvar.quantiles, qvar.coefficient_stacks.shape, qvar.nobs)
   for tau in qvar.quantiles:
       print(tau, round(qvar.coefficients(tau)[0, 0, 1], 3))
   print({k: round(v, 3) for k, v in qvar.pseudo_r_squared(0.1).items()})
   print(round(qvar.crossing_share, 4), qvar.stability_profile().round(2))
   print(qvar.forecast(0.1).round(3), qvar.irf(3, quantile=0.1)[:, 0, 1].round(3))

.. testoutput::

   (0.1, 0.5, 0.9) (3, 1, 2, 2) 399
   0.1 -0.526
   0.5 -0.257
   0.9 0.048
   {'growth': 0.356, 'fci': 0.434}
   0.0 [0.67 0.85 0.87]
   [-1.083 -1.625] [ 0.    -0.526 -0.698 -0.688]

The coefficient of ``growth`` on lagged ``fci`` is -0.53 at the 10th
percentile and -0.26 at the median: the downside reacts more. Each
accessor (``coefficients``, ``deterministic``, ``quantile_path``,
``pseudo_r_squared``, ``irf``, ``forecast``) takes a level that was
estimated (to within 1e-9) and raises for any other; a new level is a new
fit.

What is deliberately missing: ``llf`` and information criteria (check loss
is not a likelihood; ``pseudo_r_squared`` is the Koenker and Machado,
1999, :math:`R^1`, local to its level), an orthogonalized ``irf`` (no
covariance to factor; ``irf`` propagates one level's coefficients as if
they were the law of motion), and multi-step forecasts (a quantile of
quantiles is not the multi-step quantile; fit direct regressions per
horizon instead). Separately fitted levels can cross; ``crossing_share``
measures it and nothing is rearranged.

Time-varying-parameter VARs
---------------------------

The model
~~~~~~~~~

The stacked coefficients follow a random walk and, under stochastic
volatility, so do the log variances (Cogley and Sargent, 2005; Primiceri,
2005):

.. math::

   y_t = X_t' \beta_t + u_t, \qquad
   \beta_t = \beta_{t-1} + \eta_t, \quad \eta_t \sim N(0, Q),

.. math::

   u_t \sim N(0, \Sigma_t), \qquad
   \Sigma_t = A^{-1} H_t A^{-\top}, \qquad
   H_t = \mathrm{diag}(e^{h_t}), \qquad
   h_t = h_{t-1} + \zeta_t, \quad \zeta_t \sim N(0, W).

:class:`~cultivars.multivariate.nonlinear.time_varying.TVPVAR` holds
:math:`\Sigma_t = \Sigma`. Estimation is Gibbs sampling on the package's
state-space layer: each coefficient-path draw is one Durbin and Koopman
(2002) simulation-smoother pass, the covariances are conjugate
inverse-Wisharts, and the volatility block is Kim, Shephard and Chib
(1998) with a diagonal :math:`W` as in Cogley and Sargent.

Priors
~~~~~~

Following Primiceri, a training sample is split off the front, its OLS
estimates calibrate the priors, and it is then discarded:

- initial state :math:`\beta_0 \sim N(\hat b_0, 4 \hat V_0)` from the
  training OLS;
- drift covariance :math:`Q` inverse-Wishart with scale
  :math:`k_Q^2 \nu_Q \hat V_0`, :math:`\nu_Q = \max(\tau, D + 2)`;
- :math:`\Sigma` inverse-Wishart centred on the training covariance with
  :math:`k + 2` degrees of freedom;
- under SV, each diagonal entry of :math:`W` inverse-Gamma with scale set
  by ``k_vol``.

``k_drift`` (Primiceri's :math:`k_Q`) and ``k_vol`` (:math:`k_W`) both
default to ``0.01``. At that default a coefficient that drifts by several
tenths over a few hundred observations is shrunk to a few hundredths.
Vary them; values of 0.05 to 0.3 let slow drift through.

A minimal fit
~~~~~~~~~~~~~

The own-lag coefficient of ``y1`` drifts from 0.1 to 0.8 and its shock
standard deviation from 0.5 to 2. The draw counts below keep the example
fast; real work needs the defaults (2000 draws, 1000 burn-in) or more.

.. testcode::

   from cultivars.multivariate.nonlinear.time_varying import TVPSVAR, TVPVAR, TVPVARSV

   rng = np.random.default_rng(5)
   d = np.zeros((120, 2))
   for t in range(1, 120):
       a = 0.1 + 0.7 * t / 120
       d[t] = np.array([[a, 0.0], [0.3, 0.3]]) @ d[t - 1]
       d[t] += np.array([0.5 + 1.5 * t / 120, 0.7]) * rng.standard_normal(2)

   draws = dict(n_draws=120, n_burn=60, thin=1, seed=0)
   tvp = TVPVAR(d, order=1, training=30).fit(k_drift=0.1, **draws)
   print(tvp.training, tvp.nobs, tvp.n_kept, tvp.beta_mean.shape, tvp.beta_draws.shape)
   path = tvp.coefficient_path("y1", "y1.L1")
   print(path[[0, -1]].round(2))
   print(tvp.coefficients_at(tvp.nobs - 1).round(2))
   print(tvp.stability_path()[[0, -1]].round(2))

.. testoutput::

   30 89 60 (89, 6) (60, 89, 6)
   [[0.34 0.5  0.65]
    [0.56 0.64 0.73]]
   [[[ 0.64  0.11]
     [ 0.46 -0.01]]]
   [0.55 0.71]

``coefficient_path(equation, regressor)`` returns ``(nobs, 3)`` columns
``(16th percentile, mean, 84th percentile)``. Rows index the estimation
sample, which starts ``training + order`` rows into ``endog``:
``coefficients_at(t)`` takes that row position, not a date label.
``beta_mean`` stacks all coefficients equation-major, ``D = k * w`` with
``w = n_det + k * p``.

At the default ``k_drift`` the same data give a flat path:

.. testcode::

   flat = TVPVAR(d, order=1, training=30).fit(**draws)
   print(flat.coefficient_path("y1", "y1.L1")[[0, -1], 1].round(2))

.. testoutput::

   [0.61 0.61]

The summary compares the largest posterior-mean move with the median 68%
band width for this reason; drift smaller than the band is not evidence of
time variation.

Stochastic volatility
~~~~~~~~~~~~~~~~~~~~~

.. testcode::

   sv = TVPVARSV(d, order=1, training=30).fit(k_drift=0.1, k_vol=0.1, **draws)
   print(sv.volatility_path("y1")[[0, -1]].round(2))
   print(sv.volatility_path("y2")[[0, -1]].round(2))
   print(sv.h_draws.shape, sv.impact_draws.shape, sv.vol_of_vol.shape)

.. testoutput::

   [[0.66 0.84 1.04]
    [1.63 1.97 2.33]]
   [[0.47 0.61 0.78]
    [0.43 0.56 0.66]]
   (60, 89, 2) (60, 2, 2) (2,)

``volatility_path(name)`` is the standard deviation
:math:`e^{h_t/2}` of the *orthogonalized* shock, with the same
``(low, mean, high)`` columns. ``h_draws`` holds the log-variance paths,
``impact_draws`` the draws of :math:`A^{-1}`, ``vol_of_vol`` the
posterior mean of :math:`\operatorname{diag} W`.

Identification: TVPSVAR
~~~~~~~~~~~~~~~~~~~~~~~

:class:`~cultivars.multivariate.nonlinear.time_varying.TVPSVAR` applies a
recursive (Cholesky) identification at every date. How the ordering is
declared depends on the source:

- **Homoskedastic source.** The ordering is applied to the posterior-mean
  covariance at identification time, and any permutation is accepted
  through ``order=``. The impact is the same at every date.
- **SV source.** The triangular :math:`A` *was* the estimation: :math:`A`
  and the volatilities were drawn in the order of ``names``. The impact at
  date :math:`t` is the posterior mean of :math:`A^{-1} H_t^{1/2}`, so it
  drifts with the volatilities. Passing ``order=`` raises
  ``SpecificationError``; to change the ordering, reorder the columns and
  refit.

.. testcode::

   ssvar = TVPSVAR(sv).identify()
   last = ssvar.nobs - 1
   print(np.diag(ssvar.impact(0)).round(2), np.diag(ssvar.impact(last)).round(2))
   print(ssvar.irf(3, t=last)[:, 0, 0].round(3))

   try:
       TVPSVAR(sv, order=["y2", "y1"])
   except Exception as err:
       print(type(err).__name__)

   reordered = TVPSVAR(tvp, order=["y2", "y1"]).identify()
   b = reordered.impact(0)
   print(bool(np.allclose(b @ b.T, tvp.sigma_u)), round(b[1, 0], 3))

.. testoutput::

   [0.84 0.61] [1.97 0.56]
   [1.974 1.08  0.709 0.449]
   SpecificationError
   True 0.0

The reordered homoskedastic impact is the Cholesky factor in the new
order written back into the original positions: it still reproduces
``sigma_u``, but it is upper triangular in the order of ``names``.

``irf(horizon, t=...)`` freezes the coefficients and the impact at date
``t``'s posterior means. Both are plug-ins: the impact is averaged over
draws before it is combined with the averaged coefficient path, so the
response is not the posterior mean of the response. For bands, propagate
``tvp.beta_draws`` (and ``h_draws``, ``impact_draws``) draw by draw.

Convergence and seeds
~~~~~~~~~~~~~~~~~~~~~

``fit`` takes ``n_draws`` (default 2000), ``n_burn`` (1000), ``thin``
(2), ``k_drift``, ``k_vol`` (SV only) and ``seed`` (integer or
``Generator``); ``ceil((n_draws - n_burn) / thin)`` draws are kept
(``n_kept``). The draws form one Markov chain; read it with
``convergence()`` (see :doc:`bayesian_workflow`):

.. testcode::

   check = tvp.convergence()
   print(check.n_chains, check.n_draws, check.converged)

.. testoutput::

   1 60 False

Sixty draws do not converge, as expected. ``TVPSVARResult`` keeps no draws
of its own; its ``convergence()`` raises, so read the source result's.

Markov-switching VAR
--------------------

The model
~~~~~~~~~

.. math::

   y_t = D_{S_t} d_t + \sum_{i=1}^{p} A_{i, S_t}\, y_{t-i} + u_t,
   \qquad u_t \mid S_t = m \sim N(0, \Sigma_m),
   \qquad \Pr(S_t = j \mid S_{t-1} = i) = P_{ij},

with :math:`S_t` a latent first-order Markov chain and each of the
deterministic block, the lag block and the covariance either switching or
common (Krolzig, 1997). Estimation is EM on the Hamilton filter and Kim
smoother; the M-step updates :math:`P` from expected transition counts and
all regimes' coefficients in one probability-weighted GLS system, so a
common block is estimated jointly. The likelihood is multimodal: ``fit``
screens several starts and refines the best.

.. testcode::

   from cultivars.multivariate.regime_switching.markov_switching import (
       MarkovSwitchingSVAR,
       MarkovSwitchingVAR,
   )

   rng = np.random.default_rng(12)
   P = np.array([[0.95, 0.05], [0.10, 0.90]])
   chain = np.zeros(300, dtype=int)
   for t in range(1, 300):
       chain[t] = rng.choice(2, p=P[chain[t - 1]])
   means = np.array([[0.5, 0.3], [-1.0, -0.5]])
   scales = np.array([0.5, 1.0])
   A = np.array([[0.4, 0.1], [0.0, 0.3]])
   z = np.zeros((300, 2))
   for t in range(1, 300):
       z[t] = means[chain[t]] + A @ z[t - 1] + scales[chain[t]] * rng.standard_normal(2)

   model = MarkovSwitchingVAR(z, order=1, n_regimes=2, names=["g", "r"])
   msvar = model.fit(seed=0, n_init=4)
   print(msvar.specification, msvar.label_ordering, msvar.converged)
   print(msvar.nobs, msvar.n_params)
   print(msvar.transition.round(3))
   print(msvar.expected_durations.round(1), round(msvar.regime_uncertainty, 3))

.. testoutput::

   MSIH(2)-VAR(1) first-variable intercept True
   299 16.0
   [[0.887 0.113]
    [0.033 0.967]]
   [ 8.9 30.2] 0.051

Regimes are sorted by the first variable's intercept (ascending), so
regime 0 is the low-mean regime, which in this simulation is the volatile
one (chain state 1). ``label_ordering`` names the rule: first-variable
intercept when the mean switches, else covariance log-determinant, else
the first own-lag coefficient.

Regimes and probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~

.. testcode::

   print(msvar.probabilities("filtered").shape)
   print(round(float(np.mean(msvar.most_likely_regime == 1 - chain[1:])), 3))
   low = msvar.regime(0)
   print(low.deterministic.round(2), round(low.nobs, 1), low.is_stable)
   print(low.coefficients[0].round(2))
   print(np.diag(low.sigma_u).round(2), np.diag(msvar.regime(1).sigma_u).round(2))

.. testoutput::

   (299, 2)
   0.993
   [[-1.05 -0.3 ]] 61.9 True
   [[0.38 0.22]
    [0.06 0.22]]
   [0.62 0.94] [0.22 0.27]

``probabilities(kind)`` returns ``"smoothed"`` (default), ``"filtered"``
or ``"predicted"`` posteriors of shape ``(nobs, M)``.
``most_likely_regime`` is the pointwise smoothed argmax, not a Viterbi
path. ``regime(m)`` returns the regime's closed-system view: its
``coefficients``, ``deterministic``, ``sigma_u``, ``weight`` (the smoothed
probability), ``nobs`` (the *expected* number of periods in the regime, a
float), ``stability_check()`` and ``ma_representation()``. The same object
is returned on every call. ``regime_table()`` prints log-determinants,
largest roots, ergodic probabilities, durations and realized shares per
regime.

Fit arguments are those of
:class:`~cultivars.univariate.regime_switching.MarkovSwitchingAR`
(``max_iter=500``, ``tol=1e-6``, ``n_init=10``, ``screen_iter=15``,
``seed=None``); constructor flags ``switching_mean=True``,
``switching_variance=True``, ``switching_ar=False``. A switching mean
needs a deterministic term, so ``trend="n"`` with ``switching_mean=True``
raises.

Tests and comparisons
~~~~~~~~~~~~~~~~~~~~~

.. testcode::

   linear = VAR(z, order=1, names=["g", "r"]).fit()
   try:
       msvar.likelihood_ratio_test(linear)
   except Exception as err:
       print(type(err).__name__)
   msi_model = MarkovSwitchingVAR(z, order=1, names=["g", "r"], switching_variance=False)
   msi = msi_model.fit(seed=0, n_init=4)
   lr = msi.likelihood_ratio_test(msvar)
   print(msi.specification, lr.df, lr.reject())
   print([row[0] for row in msvar.compare(msi, linear, criterion="bic").rows])

.. testoutput::

   SpecificationError
   MSI(2)-VAR(1) 3 True
   ['MSIH(2)-VAR(1)', 'MSI(2)-VAR(1)', 'VAR(1)']

Comparisons that change the number of regimes (including against a linear
VAR) are refused; nested comparisons at fixed :math:`M` are allowed.

Structural identification per regime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conditional on the chain sitting in regime :math:`m` the model is a
linear VAR, and ``regime(m)`` is accepted by every identification scheme
in :mod:`cultivars.multivariate.structural` (see
:doc:`structural_identification`).
:class:`~cultivars.multivariate.regime_switching.markov_switching.MarkovSwitchingSVAR`
packages one declaration applied in every regime (Sims, Waggoner and Zha,
2008). With no arguments it applies
:class:`~cultivars.multivariate.structural.zero_restrictions.RecursiveSVAR`
in the order of ``names`` (or ``order=``); alternatively pass one
point-identified result per regime, each built from that fit's own
``msvar.regime(m)``:

.. testcode::

   from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR

   mssvar = MarkovSwitchingSVAR(msvar).identify()
   print(mssvar.shock_names, mssvar.impact(0).round(2).tolist())
   print(mssvar.impact(1).round(2).tolist(), mssvar.irf(4, regime=1).shape)

   per_regime = [
       RecursiveSVAR(msvar.regime(m), order=["r", "g"]).identify() for m in range(2)
   ]
   packaged = MarkovSwitchingSVAR(msvar, per_regime).identify()
   print(packaged.shock_names, type(packaged.structural(0)).__name__)

.. testoutput::

   ('g', 'r') [[0.79, 0.0], [-0.09, 0.96]]
   [[0.47, 0.0], [-0.03, 0.52]] (5, 2, 2)
   ('r', 'g') SVARResult

``irf(horizon, regime=m)`` assumes the chain stays in regime ``m`` over
the whole horizon; read it against ``expected_durations[m]`` (8.9 periods
for regime 0 here). Responses that integrate over the chain's future path
are not offered.

How the pieces connect
----------------------

- **Linear baseline.** TVAR, STVAR and MS-VAR results rank against a
  linear :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
  through ``compare`` (common ``nobs`` required); see :doc:`var`.
- **Structural layer.** Only the MS-VAR exposes closed linear systems
  (``regime(m)``) that the schemes in :doc:`structural_identification`
  accept. TVAR, STVAR, FC-VAR and QVAR results cannot be passed to
  ``RecursiveSVAR`` or any other scheme; their orthogonalized ``irf``
  (where offered) is a recursive Cholesky in the order of ``names``. The
  TVP family has its own recursive wrapper, ``TVPSVAR``.
- **Bayesian tools.** TVP results carry ``convergence()`` from
  :mod:`cultivars.bayes.chains`; the constant-coefficient counterparts are
  :class:`~cultivars.multivariate.large_dim.bayesian.BVAR` and
  :class:`~cultivars.multivariate.large_dim.volatility.BVARSV` in
  :doc:`bayesian_vars`. See :doc:`bayesian_workflow` for diagnostics.
- **State space.** The TVP sampler runs on the linear-Gaussian simulation
  smoother of :doc:`state_space`; the MS-VAR filter is the Hamilton and
  Kim machinery shared with
  :class:`~cultivars.univariate.regime_switching.MarkovSwitchingAR`.
- **Univariate counterparts.** SETAR, LSTAR/ESTAR and MS-AR in
  :doc:`univariate_nonlinear`, where the linearity tests also live.
- **Forecast evaluation.** TVAR and STVAR give a skeleton multi-step
  path; FC-VAR and QVAR give one step; TVP and MS-VAR results have no
  ``forecast`` method. Plan forecast comparisons (:doc:`forecasting`)
  accordingly.

Nuances and gotchas
-------------------

- **No standard errors** on any estimator here except the kernel
  pointwise bands of FC-VAR (conditional on the bandwidth) and the
  posterior bands of the TVP models.
- **No generalized (history-averaged) impulse responses.** Every ``irf``
  is a frozen-regime, frozen-state, frozen-quantile or frozen-date
  linearization.
- **Regime names differ by family.** Observed-regime models take
  ``regime="lower"`` / ``"upper"``; the MS-SVAR takes an integer
  ``regime``. For an exponential STVAR, "upper" is both tails.
- **Missing methods.** ``MarkovSwitchingVARResult`` has no ``forecast``,
  ``simulate`` or ``irf`` (identify first, then use
  ``MarkovSwitchingSVARResult.irf``). TVP results have no ``forecast``,
  ``irf`` (use ``TVPSVAR``) or likelihood. FC-VAR and QVAR results have
  no ``compare``.
- **Delay searching.** Only a self-exciting TVAR searches the delay
  (``delay=None``). STVAR and FC-VAR default to ``delay=1`` and never
  search; an external-driver TVAR must state its delay.
- **Parameter counts.** ``TVARResult.n_params`` counts both regimes'
  coefficients, both covariances and the threshold (19 above);
  ``STVARResult.n_params`` counts one covariance, the threshold and
  ``gamma`` (17). Neither counts a searched delay.
- **``gamma`` is a lower bound when abrupt.** If
  ``is_effectively_abrupt`` is ``True``, use TVAR. For the exponential
  form the flag is always ``False``.
- **Effective samples.** Observed-regime models drop ``max(order,
  delay)`` rows; TVP models also drop ``training`` rows, and every path
  is indexed on what remains; ``compare`` raises ``DimensionError`` on
  unequal ``nobs``.
- **TVP prior scales are findings in disguise.** The default
  ``k_drift = k_vol = 0.01`` produce nearly flat paths on moderate drift;
  report results under several values.
- **TVP run time.** The sampler loops over dates in Python: about 0.02
  seconds per iteration for a bivariate VAR(1) with 90 estimation rows and
  about 0.08 with 160, so the default 2000 draws take one to three
  minutes on a small system. Scale ``n_draws`` accordingly when
  exploring.
- **TVPVARSV ``sigma_u``** is the posterior mean of the time-averaged
  covariance :math:`T^{-1}\sum_t \Sigma_t`, not the end-of-sample
  covariance; use ``h_draws`` and ``impact_draws`` for date-specific
  covariances.
- **TVPSVAR with SV is fixed at estimation.** Reorder the columns of
  ``endog`` and refit to change the recursive ordering.
- **MS-VAR convergence flag.** When the refining EM run ends below the
  best screened log-likelihood, the fit falls back to the screened start
  and reports ``converged=False`` with ``n_iter == screen_iter``, and
  the summary opens with a non-convergence warning. This happens often on
  well-behaved data. Refit with another ``seed`` or larger ``n_init``; if
  the log-likelihood matches across seeds, the optimum is likely found
  despite the flag.
- **MS-VAR regime views.** ``regime(m).nobs`` is a float (expected
  periods) and ``regime(m).resid`` is probability-weighted; schemes that
  read only the covariance are unaffected, schemes that read residual
  moments consume a weighted object.
- **lag_order_selection scores the linear VAR.** The method inherited by
  every model on this page fits *linear* VAR(:math:`p`) orders, not the
  nonlinear model. With ``max_lags=None`` it scores every order up to the
  largest the sample supports, where the residual covariance becomes
  nearly singular and all criteria can pick the maximum; always pass
  ``max_lags``.
- **QVAR levels.** Accessors refuse any level not in ``quantiles``; the
  ``quantiles`` passed to ``fit`` are sorted ascending.
- **FC-VAR extrapolation.** ``coefficients_at``, ``irf`` and ``forecast``
  raise when the state lies outside the trimmed ``state_range``.

See also
--------

- :doc:`var` for the linear reduced form and ``generalized_irf``.
- :doc:`structural_identification` for the schemes that accept
  ``MarkovSwitchingVARResult.regime(m)``.
- :doc:`bayesian_vars` and :doc:`bayesian_workflow` for BVAR, BVARSV and
  convergence diagnostics.
- :doc:`univariate_nonlinear` for SETAR, STAR, MS-AR and the linearity
  tests.
- :doc:`state_space` for the filters and smoothers underneath.
- :doc:`nuances` for package-wide refusals.
- API: :mod:`cultivars.multivariate.nonlinear.threshold`,
  :mod:`cultivars.multivariate.nonlinear.smooth_transition`,
  :mod:`cultivars.multivariate.nonlinear.functional_coefficient`,
  :mod:`cultivars.multivariate.nonlinear.quantile`,
  :mod:`cultivars.multivariate.nonlinear.time_varying`,
  :mod:`cultivars.multivariate.regime_switching.markov_switching`.
