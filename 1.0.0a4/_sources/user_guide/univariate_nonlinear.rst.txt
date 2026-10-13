.. meta::
   :description: Threshold, smooth-transition, Markov-switching and neural
      autoregressions in cultivars: specification, estimation, regimes and
      linearity tests.

Nonlinear and regime-switching univariate models
================================================

This page covers autoregressions whose conditional mean changes with a
regime: a hard switch on an observed variable (SETAR, TAR), a smooth blend
(LSTAR, ESTAR), a latent Markov chain (Markov-switching AR), or a mean
function learned by a neural network (AR-NN, TAR-NN). Use it when a linear
AR leaves structure in the residuals that depends on the level or sign of
the series. Linear ARMA models are in :doc:`univariate_linear`, volatility
models in :doc:`volatility`, and the vector versions of everything here in
:doc:`nonlinear_multivariate`.

.. contents:: On this page
   :local:
   :depth: 1

The family at a glance
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 30 30 22

   * - Class
     - Module
     - Model
     - Use when
   * - :class:`~cultivars.univariate.threshold.SETAR`
     - :mod:`cultivars.univariate.threshold`
     - Two AR regimes split by :math:`y_{t-d} \le r`
     - The regime is a lag of the series and switches abruptly
   * - :class:`~cultivars.univariate.threshold.TAR`
     - :mod:`cultivars.univariate.threshold`
     - Two AR regimes split by an external :math:`z_{t-d} \le r`
     - An observed driver (a spread, an index) decides the regime
   * - :class:`~cultivars.univariate.smooth_transition.LSTAR`
     - :mod:`cultivars.univariate.smooth_transition`
     - Two AR regimes blended by a logistic weight in :math:`y_{t-d}`
     - Asymmetry with a direction (contractions versus expansions)
   * - :class:`~cultivars.univariate.smooth_transition.ESTAR`
     - :mod:`cultivars.univariate.smooth_transition`
     - Two AR regimes blended by an exponential weight in :math:`y_{t-d}`
     - Mean reversion that strengthens with distance from equilibrium
   * - :class:`~cultivars.univariate.regime_switching.MarkovSwitchingAR`
     - :mod:`cultivars.univariate.regime_switching`
     - Intercept, AR block and variance switch with a latent chain
     - Regimes are not observable (recession dating, volatility states)
   * - :class:`~cultivars.univariate.mean_function.ARNN`
     - :mod:`cultivars.univariate.mean_function`
     - :math:`y_t = f(y_{t-1}, \ldots, y_{t-p}) + \varepsilon_t`, :math:`f`
       learned
     - The shape of the nonlinearity is unknown and prediction is the goal
   * - :class:`~cultivars.univariate.mean_function.TARNN`
     - :mod:`cultivars.univariate.mean_function`
     - One learned function per side of a fixed threshold
     - A known split point with unknown dynamics on each side

All seven take a one-dimensional array and are fitted with ``fit()``. Only
:class:`~cultivars.univariate.regime_switching.MarkovSwitchingAR` takes
arguments in ``fit``, because its likelihood is multimodal and the answer
depends on the starting values and the seed.

Test for linearity before fitting
---------------------------------

A threshold or smooth-transition model claims that the linear AR is wrong
in a specific way. Test that claim first, on the linear model's own terms,
with :mod:`cultivars.diagnostics.nonlinearity` (details in
:doc:`diagnostics`). Every test takes the raw series and fits the
AR(:math:`p`) null internally, so the degrees of freedom are always right.
The threshold and transition parameters are not identified under linearity
(Davies, 1987), so a plain likelihood-ratio test against the AR is invalid;
these tests are the valid route.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - What a rejection points to
   * - :func:`~cultivars.diagnostics.nonlinearity.tsay`
     - A threshold (Tsay, 1989, arranged autoregression)
   * - :func:`~cultivars.diagnostics.nonlinearity.hansen_threshold`
     - A SETAR, by a bootstrap sup-F over the threshold grid (Hansen, 1996);
       the record carries the threshold it found
   * - :func:`~cultivars.diagnostics.nonlinearity.terasvirta`
     - A smooth transition (Teräsvirta, 1994); ``suggestion`` reads LSTAR or
       ESTAR from the escalation sequence in ``companions``
   * - :func:`~cultivars.diagnostics.nonlinearity.ramsey_reset`
     - Some smooth function of the fit left in the residual (no regime
       implied)

The running example is a two-regime SETAR: the AR coefficient is 0.7 when
the previous value is at or below zero and -0.3 above it.

.. testcode::

   import numpy as np
   from cultivars.diagnostics.nonlinearity import (
       hansen_threshold,
       ramsey_reset,
       terasvirta,
       tsay,
   )

   rng = np.random.default_rng(42)
   y = np.zeros(400)
   for t in range(1, 400):
       phi = 0.7 if y[t - 1] <= 0.0 else -0.3
       y[t] = phi * y[t - 1] + rng.standard_normal()

   for test in (tsay(y, order=1), terasvirta(y, order=1), ramsey_reset(y, order=1)):
       print(f"{test.name:11s} p={test.pvalue:.4f} -> {test.alternative}")

   sup = hansen_threshold(y, order=1, replications=200, seed=0)
   print(round(sup.statistic, 2), round(sup.pvalue, 3), round(sup.threshold, 3))
   print({k: round(v, 2) for k, v in sup.critical_values.items()})

.. testoutput::

   Tsay        p=0.0000 -> threshold
   Teräsvirta  p=0.0000 -> smooth transition
   RESET       p=0.0000 -> smooth misspecification
   62.4 0.0 0.346
   {'1%': 19.42, '5%': 13.44, '10%': 10.96}

All four reject. Two further points before choosing a model. The tests
target the conditional mean, and neglected conditional heteroskedasticity
inflates every one of them, so run an ARCH test on the linear residuals
first (see :doc:`volatility`). And the Teräsvirta record always carries a
``suggestion``, even when it does not reject; read it only after
``reject()`` is ``True``.

Threshold autoregressions: SETAR and TAR
----------------------------------------

The model
~~~~~~~~~

Two linear autoregressions, selected by an indicator on an observed
transition variable (Tong, 1990):

.. math::

   y_t = \begin{cases}
   \phi_L' x_t + \varepsilon_t & z_{t-d} \le r \\
   \phi_U' x_t + \varepsilon_t & z_{t-d} > r
   \end{cases},
   \qquad x_t = (1, y_{t-1}, \ldots, y_{t-p})',

with :math:`z_t = y_t` for :class:`~cultivars.univariate.threshold.SETAR`
and an external series for :class:`~cultivars.univariate.threshold.TAR`.
The sum of squares is a step function of :math:`r`, so no gradient method
applies. The estimator is an exhaustive grid search: ``n_grid`` quantiles
of the transition variable with ``trim`` removed from each tail, two
least-squares regressions at every candidate, and (for SETAR with
``delay=None``) every delay in ``1..order`` on one common sample. The
threshold estimate is super-consistent (Chan, 1993), so the coefficients
are read as OLS conditional on it.

A minimal fit
~~~~~~~~~~~~~

.. testcode::

   from cultivars.univariate.autoregression import AR
   from cultivars.univariate.threshold import SETAR, TAR

   hard = SETAR(y, order=2).fit()
   print(hard.delay, hard.searched_delay, round(hard.threshold, 3))
   print(hard.n_lower, hard.n_upper, hard.nobs, hard.n_params)
   print(hard.lower_params.round(3), hard.upper_params.round(3))
   print(hard.is_regimewise_stationary)

.. testoutput::

   1 True 0.347
   325 73 398 7
   [ 0.056  0.858 -0.132] [-0.324 -0.082  0.188]
   True

``lower_params`` and ``upper_params`` are ``[const, ar.L1, ..., ar.Lp]``.
``params`` is the same information as a dict keyed ``lower.const``,
``lower.ar.L1``, ..., ``threshold``, ``sigma2``. The summary prints the
coefficient table and the regime split:

.. testcode::

   print(hard.summary())

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

                                SETAR(2, d=1) Results
   ================================================================================
   Model:                   SETAR(2, d=1)  Log-likelihood:               -535.834
   Delay:                               1  AIC:                          1085.669
   Threshold:                      0.3474  BIC:                          1113.574
   Observations:                      398  HQIC:                         1096.722
   --------------------------------------------------------------------------------
                      coef
   lower.const      0.0563
   lower.ar.L1      0.8583
   lower.ar.L2     -0.1322
   upper.const     -0.3243
   upper.ar.L1     -0.0822
   upper.ar.L2      0.1877
   threshold        0.3474
   sigma2           0.8648
   ================================================================================
   Regime-wise stationary: True   max |root| lower = 0.6572, upper = 0.4763
   Regime split: 325 below, 73 above (18.3% upper).
   The delay was selected by the same grid search as the threshold; d = 1 minimized
   the total sum of squares.
   The threshold is not identified under linearity, so information criteria rank
   specifications but do not test for a threshold.
   Standard errors are not yet available for this estimator.

Against the linear AR, rank by information criteria. ``compare`` requires
both fits to share the effective sample, which holds here because both
drop ``order = 2`` observations:

.. testcode::

   linear = AR(y, order=2).fit()
   table = hard.compare(linear, criterion="bic")
   print([row[0] for row in table.rows])

.. testoutput::

   ['SETAR(2, d=1)', 'AR(2)']

The likelihood-ratio test is refused rather than computed, because its
chi-squared reference is wrong under the Davies problem:

.. testcode::

   try:
       hard.likelihood_ratio_test(linear)
   except Exception as err:
       print(type(err).__name__)

.. testoutput::

   SpecificationError

Specification choices
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Argument
     - Default
     - Meaning
   * - ``order``
     - required
     - AR order :math:`p` within each regime, at least 1
   * - ``delay``
     - ``None`` (SETAR), ``1`` (TAR)
     - Lag :math:`d` of the transition variable. ``None`` searches
       ``1..order`` jointly with the threshold; only SETAR can search, TAR
       requires a fixed delay
   * - ``trim``
     - ``0.15``
     - Fraction removed from each tail of the threshold grid, in
       ``(0, 0.5)``; keeps either regime from being fitted on a handful of
       extreme points
   * - ``n_grid``
     - ``300``
     - Candidate thresholds per delay; splits leaving a regime fewer than
       ``order + 2`` observations are skipped
   * - ``threshold_variable``
     - TAR only, required
     - The external driver, same length as ``endog``

For :class:`~cultivars.univariate.threshold.TAR` the threshold is in the
units of the driver and ``threshold_values`` holds the lagged driver
aligned with the residuals:

.. testcode::

   rng = np.random.default_rng(1)
   driver = rng.standard_normal(400)
   x = np.zeros(400)
   for t in range(1, 400):
       phi = 0.5 if driver[t - 1] <= 0.0 else -0.5
       x[t] = phi * x[t - 1] + rng.standard_normal()

   ext = TAR(x, order=1, threshold_variable=driver).fit()
   print(ext.self_exciting, ext.delay, round(ext.threshold, 3))
   print(bool(np.allclose(ext.threshold_values, driver[:-1])))
   print(ext.lower_params.round(2), ext.upper_params.round(2))

.. testoutput::

   False 1 0.089
   True
   [-0.1   0.53] [ 0.04 -0.47]

What the result gives you
~~~~~~~~~~~~~~~~~~~~~~~~~

Both classes return a :class:`~cultivars.univariate.threshold.SETARResult`.

- Estimates: ``threshold``, ``delay``, ``lower_params``, ``upper_params``,
  ``sigma2`` (one pooled variance), ``ssr``, ``params``.
- Regimes: ``threshold_values`` (the lagged transition variable over the
  effective sample), ``regime_weight`` (1.0 above the threshold, 0.0 at or
  below), ``n_lower``, ``n_upper``, ``upper_fraction``.
- Stability per regime: ``lower_stability``, ``upper_stability``,
  ``is_regimewise_stationary``. Regime-wise stationarity is sufficient,
  not necessary, for the process to be stationary; an explosive regime
  that the other regime pulls back from is admissible.
- Comparison: ``information_criteria``, ``compare``.
- ``simulate(n, seed=...)`` and
  ``nonlinear_state_space(measurement_variance)`` (SETAR only; see
  `Simulation and state-space emission`_).
- ``fittedvalues``, ``resid`` and ``to_pandas``/``to_polars``/``to_dict``
  for the aligned series.

Smooth-transition autoregressions: LSTAR and ESTAR
--------------------------------------------------

The model
~~~~~~~~~

The indicator is replaced by a weight :math:`G \in [0, 1]` that moves
continuously with :math:`z_t = y_{t-d}` (Teräsvirta, 1994):

.. math::

   y_t = \bigl(1 - G(z_t)\bigr)\,\phi_L' x_t + G(z_t)\,\phi_U' x_t
   + \varepsilon_t,

.. math::

   G_{\text{logistic}}(z) = \Bigl(1 + e^{-\gamma (z - c)/s}\Bigr)^{-1},
   \qquad
   G_{\text{exponential}}(z) = 1 - e^{-\gamma (z - c)^2 / s^2},

where :math:`s` is the sample standard deviation of the transition
variable, so ``gamma`` is a speed per standard deviation and comparable
across series. The logistic weight is monotone (regimes are low and high).
The exponential weight is symmetric about :math:`c`: the "lower" regime is
the middle (weight 0 at :math:`z = c`) and the "upper" regime is both
tails. Conditional on :math:`(\gamma, c)` the model is linear, so the two
coefficient blocks are concentrated out by least squares and only
:math:`(\gamma, c)` are searched, by multi-start Nelder-Mead seeded from
the best hard split. The fit is deterministic; there is no seed.

LSTAR
~~~~~

.. testcode::

   from cultivars.univariate.smooth_transition import ESTAR, LSTAR

   rng = np.random.default_rng(1)
   s = np.zeros(500)
   for t in range(1, 500):
       w = 1.0 / (1.0 + np.exp(-6.0 * s[t - 1]))
       s[t] = (1 - w) * (0.4 + 0.6 * s[t - 1]) + w * (-0.4 - 0.5 * s[t - 1])
       s[t] += 0.4 * rng.standard_normal()

   print(terasvirta(s, order=1).suggestion)
   smooth = LSTAR(s, order=1).fit()
   print(smooth.transition, round(smooth.gamma, 3), round(smooth.transition_scale, 3))
   print(round(smooth.gamma / smooth.transition_scale, 2), round(smooth.threshold, 3))
   print(smooth.lower_params.round(2), smooth.upper_params.round(2))
   print(list(smooth.params))
   print(smooth.is_effectively_abrupt, round(smooth.upper_fraction, 3))

.. testoutput::

   LSTAR
   logistic 2.661 0.462
   5.75 -0.114
   [0.48 0.69] [-0.32 -0.67]
   ['lower.const', 'lower.ar.L1', 'upper.const', 'upper.ar.L1', 'threshold', 'gamma', 'sigma2']
   False 0.502

``gamma / transition_scale`` is the speed in the units of the series (the
truth here is 6). For the logistic form ``upper_fraction`` is the mean
weight, not a count.

When the fitted speed is so large that the weight is an indicator,
``is_effectively_abrupt`` turns ``True`` (logistic, ``gamma > 100``). The
likelihood is then flat in ``gamma``, the reported value is a lower bound,
and :class:`~cultivars.univariate.threshold.SETAR` says the same thing
with two fewer parameters. On this sample the hard switch wins on BIC even
though the truth is smooth:

.. testcode::

   rivals = [ESTAR(s, order=1).fit(), SETAR(s, order=1).fit(), AR(s, order=1).fit()]
   table = smooth.compare(*rivals, criterion="bic")
   for row in table.rows:
       print(row[0], row[2], row[4])

.. testoutput::

   SETAR(1, d=1) 5 0.000
   LSTAR(1, d=1) 6 5.870
   ESTAR(1, d=1) 6 5.958
   AR(1) 3 149.984

ESTAR
~~~~~

The canonical ESTAR is a real exchange rate under transaction costs:
persistent near equilibrium, mean-reverting far from it (Kapetanios, Shin
and Snell, 2003). Exponential fits need more data than logistic ones to
pin down :math:`c`; 800 observations are used here.

.. testcode::

   rng = np.random.default_rng(0)
   q = np.zeros(800)
   for t in range(1, 800):
       w = 1.0 - np.exp(-3.0 * q[t - 1] ** 2)
       q[t] = (1 - w) * 0.95 * q[t - 1] + w * 0.2 * q[t - 1]
       q[t] += 0.3 * rng.standard_normal()

   print(terasvirta(q, order=1).suggestion)
   band = ESTAR(q, order=1).fit()
   print(round(band.threshold, 3), round(band.gamma, 3))
   print(band.lower_params.round(2), band.upper_params.round(2))

.. testoutput::

   ESTAR
   -0.098 0.41
   [0.01 0.9 ] [0.04 0.23]

The "lower" block (near the threshold) carries the near-unit-root
persistence 0.9, the "upper" block (the tails) the fast reversion 0.23.

Specification choices and outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both constructors take ``endog``, ``order`` (at least 1) and ``delay``
(default 1). The delay is not searched; fit several and rank them on a
common sample. Both return a
:class:`~cultivars.univariate.smooth_transition.STARResult`, which carries
everything listed for ``SETARResult`` above (``threshold``,
``lower_params``, ``upper_params``, ``regime_weight``, the per-regime
stability verdicts, ``simulate``, ``nonlinear_state_space``) plus
``transition``, ``gamma``, ``transition_scale`` and
``is_effectively_abrupt``. ``regime_weight`` reproduces the estimator's
weight exactly, so ``(1 - w) * X @ lower_params + w * X @ upper_params``
equals ``fittedvalues``.

Markov-switching autoregression
-------------------------------

The model
~~~~~~~~~

A :math:`K`-regime autoregression whose intercept, AR coefficients and
variance may each switch with a latent first-order Markov chain
:math:`S_t` (Hamilton, 1989):

.. math::

   y_t = c_{S_t} + \sum_{i=1}^{p} \phi_{i,S_t}\, y_{t-i} + \varepsilon_t,
   \qquad \varepsilon_t \sim N(0, \sigma^2_{S_t}),
   \qquad P_{ij} = \Pr(S_t = j \mid S_{t-1} = i).

This is the *intercept*-switching form. Conditional on the observed lags
each regime is linear, so the Hamilton filter runs on :math:`K` states
rather than the :math:`K^{p+1}` that Hamilton's mean-switching form
needs; the two coincide only at :math:`p = 0`. Estimation is EM (Hamilton,
1990): the E-step is the Hamilton filter and Kim (1994) smoother, the
M-step updates :math:`P` from expected transition counts and the
coefficients by probability-weighted least squares. Which blocks switch is
reported in Krolzig's (1997) notation by ``specification``: ``I``
intercept, ``A`` AR block, ``H`` variance. Hamilton's recession model,
MSIH(2)-AR(4), is ``MarkovSwitchingAR(y, order=4, n_regimes=2)``.

A minimal fit
~~~~~~~~~~~~~

The example has a low-mean quiet regime and a high-mean volatile one,
with a common AR coefficient of 0.5:

.. testcode::

   from cultivars.univariate.regime_switching import MarkovSwitchingAR

   rng = np.random.default_rng(3)
   P = np.array([[0.95, 0.05], [0.10, 0.90]])
   state = np.zeros(400, dtype=int)
   for t in range(1, 400):
       state[t] = rng.choice(2, p=P[state[t - 1]])
   m = np.zeros(400)
   for t in range(1, 400):
       level, scale = [(-0.5, 0.5), (1.0, 1.0)][state[t]]
       m[t] = level + 0.5 * m[t - 1] + scale * rng.standard_normal()

   ms = MarkovSwitchingAR(m, order=1, n_regimes=2).fit(seed=0, n_init=4)
   print(ms.specification, ms.label_ordering, ms.converged, ms.nobs, ms.n_params)
   print(ms.intercepts.round(2), ms.ar_params.ravel().round(2), ms.variances.round(2))
   print(ms.transition.round(3))
   print(ms.expected_durations.round(1), ms.ergodic_prob.round(3))

.. testoutput::

   MSIH(2)-AR(1) intercept True 399 7
   [-0.5   1.17] [0.48 0.48] [0.24 1.02]
   [[0.959 0.041]
    [0.142 0.858]]
   [24.4  7. ] [0.776 0.224]

``ar_params`` has shape ``(K, p)``; a non-switching block is stored with
identical rows. ``n_params`` counts the :math:`K(K-1)` free transition
probabilities plus each block once per regime that switches it: here
2 + 2 intercepts + 1 AR + 2 variances = 7.

Reading regime probabilities and regimes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The regime is never observed. Three posteriors are returned, each of shape
``(nobs, K)`` and aligned with the last ``nobs`` observations:

- ``probabilities("smoothed")`` (the default, also ``smoothed_prob``):
  :math:`\Pr(S_t \mid y_{1:T})`, for dating regimes after the fact;
- ``probabilities("filtered")`` (``filtered_prob``):
  :math:`\Pr(S_t \mid y_{1:t})`, the real-time view;
- ``probabilities("predicted")`` (``predicted_prob``):
  :math:`\Pr(S_t \mid y_{1:t-1})`, the one-step-ahead state forecast.

.. testcode::

   smoothed = ms.probabilities()
   filtered = ms.probabilities("filtered")
   print(smoothed.shape, smoothed[:3].round(3).tolist())
   print(round(float(np.mean(ms.most_likely_regime == state[1:])), 3))
   print(round(float(np.mean(filtered.argmax(axis=1) == state[1:])), 3))
   print(ms.regime_shares.round(3), round(ms.regime_uncertainty, 3))
   print(round(ms.unconditional_mean, 3))

.. testoutput::

   (399, 2) [[0.991, 0.009], [0.996, 0.004], [0.998, 0.002]]
   0.98
   0.94
   [0.776 0.224] 0.115
   -0.245

``most_likely_regime`` is the pointwise argmax of the smoothed posterior,
not the Viterbi path; it can contain a one-period switch that, as a
sequence, has almost no posterior mass. ``regime_shares`` (average
smoothed probability) is the honest frequency count; compare it with
``ergodic_prob``. ``regime_uncertainty`` is the mean posterior entropy
normalized to ``[0, 1]``: near 0 the regimes are cleanly separated, near
1 the data barely distinguish them.

Two tables read across regimes rather than down parameters:

.. testcode::

   print(ms.regime_table())

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

                                MSIH(2)-AR(1) regimes
   ================================================================================
   Ordering:          ascending intercept  Observations:                      399
   --------------------------------------------------------------------------------
   regime       const   sigma2   ergodic   duration    share
   0          -0.5038   0.2381    0.7760      24.36   0.7764
   1           1.1700   1.0212    0.2240       7.03   0.2236
   ================================================================================
   'ergodic' is the stationary probability implied by the transition matrix;
   'share' is the average smoothed probability actually observed. 'duration' is 1 /
   (1 - P_jj), in periods.

``transition_table()`` prints :math:`P` with rows indexed by the origin
regime.

Specification and estimation choices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 14 62

   * - Argument
     - Default
     - Meaning
   * - ``order``
     - required
     - AR order :math:`p`; ``0`` is allowed and gives a Gaussian hidden
       Markov model
   * - ``n_regimes``
     - ``2``
     - :math:`K`, at least 2
   * - ``switching_mean``
     - ``True``
     - Intercept switches
   * - ``switching_variance``
     - ``True``
     - Variance switches
   * - ``switching_ar``
     - ``False``
     - AR block switches; multiplies the AR parameter count by :math:`K`
   * - ``fit(max_iter=)``
     - ``500``
     - Iteration cap for the refining EM run
   * - ``fit(tol=)``
     - ``1e-6``
     - Tolerance on the log-likelihood increment
   * - ``fit(n_init=)``
     - ``10``
     - Starting values screened; the first uses quantile intercepts and a
       0.9 diagonal, the rest draw both at random
   * - ``fit(screen_iter=)``
     - ``15``
     - EM iterations used to score each start; only the best is refined
   * - ``fit(seed=)``
     - ``None``
     - Seed or ``Generator`` for the starts; pass one for reproducibility

At least one block must switch. Regime labels are sorted to make two runs
comparable: by ascending intercept when it switches, else by variance,
else by the first AR coefficient; ``label_ordering`` names the rule. Under
a variance-only model, regime 0 is the quiet regime, not a low-mean one.

Testing against nested specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A likelihood-ratio test that holds :math:`K` fixed is valid and allowed;
here, whether the variance switches:

.. testcode::

   msi = MarkovSwitchingAR(m, order=1, switching_variance=False).fit(seed=0, n_init=4)
   lr = msi.likelihood_ratio_test(ms)
   print(msi.specification, round(lr.statistic, 2), lr.df, lr.reject())

.. testoutput::

   MSI(2)-AR(1) 40.19 1 True

A test that changes :math:`K` (including two regimes against a linear AR)
is refused with a ``SpecificationError``: under the smaller model the
extra regime's parameters are unidentified and its transition
probabilities sit on the boundary (Hansen, 1992). Rank such models by
``compare`` instead, or bootstrap.

What the result gives you
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~cultivars.univariate.regime_switching.MarkovSwitchingARResult`
carries ``transition``, ``intercepts``, ``ar_params``, ``variances``,
``params`` (keys ``const[j]``, ``ar.L1[j]``, ``sigma2[j]``,
``p[i->j]``), the three posteriors, ``ergodic_prob``,
``expected_durations``, ``n_iter``, ``converged``,
``regime_stability(j)``, ``is_regimewise_stationary``,
``unconditional_mean`` (``None`` if any regime is non-stationary),
``simulate``, ``simulate_regimes`` and ``state_space``. ``fittedvalues``
are posterior-weighted means :math:`\sum_j \Pr(S_t = j)\, m_{j,t}`, not
the fit of any one regime; ``to_pandas`` adds one smoothed-probability
column per regime and the most likely regime.

``state_space`` re-applies the fitted chain to data the model was not
estimated on, the natural out-of-sample check for a regime model:

.. testcode::

   system = ms.state_space
   print(abs(system.loglikelihood(m) - ms.llf) < 1e-8)
   print(system.filter(m[:100]).filtered_prob.shape)

.. testoutput::

   True
   (99, 2)

Learned conditional means: ARNN and TARNN
-----------------------------------------

The model
~~~~~~~~~

The mean is a function learned from lagged levels by a pluggable training
engine (Kuan and White, 1994):

.. math::

   y_t = f(y_{t-1}, \ldots, y_{t-p}) + \varepsilon_t,
   \qquad f(x) = b + \sum_{h=1}^{H} w_h \tanh(a_h' x + c_h),

for the default engine, a one-hidden-layer perceptron with 8 tanh units
trained by L-BFGS with an L2 penalty, 3 random restarts and a fixed seed
of 0. :class:`~cultivars.univariate.mean_function.TARNN` splits the
sample at a *fixed* threshold on :math:`z_{t-d}` (the median by default)
and trains one network per side with a pooled variance. The variance is
concentrated out as :math:`\mathrm{SSR}/n`.

There is no coefficient table: the weights are unidentified up to
permuting and sign-flipping hidden units. The results expose the fitted
function through ``predict`` and report ``r_squared``, the capacity, and
the engine.

.. testcode::

   from cultivars.univariate.mean_function import ARNN, TARNN

   rng = np.random.default_rng(0)
   v = np.zeros(200)
   for t in range(1, 200):
       v[t] = 0.8 * np.tanh(2.0 * v[t - 1]) + 0.3 * rng.standard_normal()

   net = ARNN(v, order=1).fit()
   print(net.nobs, net.n_learner_parameters, net.n_params, net.engine)
   print(round(net.r_squared, 2), round(net.parameters_per_observation, 3))
   grid = np.array([[-1.0], [0.0], [1.0]])
   print(net.predict(grid).round(1), (0.8 * np.tanh(2.0 * grid[:, 0])).round(1))

.. testoutput::

   199 25 26 NumpyMLPEngine
   0.78 0.126
   [-0.8  0.1  0.7] [-0.8  0.   0.8]

``predict`` takes a matrix of shape ``(n, order)`` with columns
``[y_{t-1}, ..., y_{t-p}]``. Columns in the wrong order do not raise; they
return wrong numbers. :class:`~cultivars.univariate.mean_function.TARNNResult`
``predict`` also needs the already-lagged threshold variable, because the
regime cannot always be read off the feature columns:

.. testcode::

   split = TARNN(v, order=1, threshold=0.0).fit()
   print(split.threshold, split.n_lower, split.n_upper, split.n_learner_parameters)
   print(split.predict(np.array([[-1.0], [1.0]]), np.array([-1.0, 1.0])).round(1))

.. testoutput::

   0.0 74 125 50
   [-0.8  0.7]

Plugging in an engine
~~~~~~~~~~~~~~~~~~~~~

Any object with ``fit(features, target)`` returning something with
``predict(features)`` and an ``n_parameters`` attribute satisfies
:class:`~cultivars.univariate.mean_function.MeanFunctionEngine`. A
least-squares engine reproduces the linear AR and shows the contract:

.. testcode::

   from cultivars.univariate.mean_function import MeanFunctionEngine

   class LinearEngine:
       def fit(self, features, target):
           design = np.column_stack([np.ones(len(target)), features])
           beta = np.linalg.lstsq(design, target, rcond=None)[0]

           class Predictor:
               n_parameters = design.shape[1]

               def predict(self, new):
                   return np.column_stack([np.ones(len(new)), new]) @ beta

           return Predictor()

   print(isinstance(LinearEngine(), MeanFunctionEngine))
   ols = ARNN(v, order=1, engine=LinearEngine()).fit()
   print(ols.engine, round(ols.llf, 3), round(AR(v, order=1).fit().llf, 3))

.. testoutput::

   True
   LinearEngine -40.18 -40.18

Specification choices
~~~~~~~~~~~~~~~~~~~~~

- ``ARNN(endog, *, order, engine=None)``: ``engine=None`` uses the
  reference perceptron.
- ``TARNN(endog, *, order, engine=None, threshold_variable=None,
  delay=1, threshold=None, trim=0.15)``: ``threshold=None`` uses the
  median of the lagged threshold variable; ``fit`` raises
  ``NumericalError`` if a stated threshold leaves either side with less
  than ``trim`` of the sample. The threshold is never searched: a grid
  would retrain both networks at every candidate.

Information criteria penalize the nominal weight count, which overstates
the effective degrees of freedom of a weight-decayed network. Use them to
choose capacity *within* one engine configuration, never against a
linear model. ``llf`` is a concentrated Gaussian likelihood at a
penalized fit, a goodness-of-fit summary. ``likelihood_ratio_test``
against a linear null is refused.

Simulation and state-space emission
-----------------------------------

The parametric results simulate at the fitted parameters with Gaussian
innovations. ``seed`` takes an integer or a ``Generator``; ``burn``
initial periods are discarded.

.. testcode::

   path = hard.simulate(300, seed=3)
   print(path.shape, bool(np.array_equal(path, hard.simulate(300, seed=3))))
   print(smooth.simulate(5, seed=0).round(3))
   draw, regimes = ms.simulate_regimes(1000, seed=1)
   print(draw.shape, np.bincount(regimes) / 1000)

.. testoutput::

   (300,) True
   [ 0.357 -0.12   0.026 -0.469  0.456]
   (1000,) [0.739 0.261]

A TAR fit cannot simulate (``SpecificationError``): its driver's future is
not part of the model. The ARNN and TARNN results do not simulate.

``nonlinear_state_space(measurement_variance)`` on SETAR, STAR, ARNN and
self-exciting TARNN results returns a
:class:`~cultivars.state_space.nonlinear.NonlinearSSM` whose state is the
lag stack of the true series and whose observation adds Gaussian noise of
the stated (strictly positive) variance. Use the particle filter for the
hard-switch emitters and any of the three filters for the smooth ones
(see :doc:`state_space`).

.. testcode::

   ssm = smooth.nonlinear_state_space(0.05)
   noisy = s + np.sqrt(0.05) * np.random.default_rng(9).standard_normal(500)
   cleaned = ssm.unscented_filter(noisy[:, None]).filtered_state[:, 0]
   print(ssm.k_states, bool(np.mean((cleaned - s) ** 2) < np.mean((noisy - s) ** 2)))

.. testoutput::

   1 True

How the pieces connect
----------------------

- **Diagnostics first.** :mod:`cultivars.diagnostics.nonlinearity`
  supplies the valid tests of linearity (Tsay, Hansen sup-F, Teräsvirta,
  RESET) and :func:`~cultivars.diagnostics.nonlinearity.bds` for residual
  independence; see :doc:`diagnostics`.
- **Comparison.** Every parametric result here and the linear
  :class:`~cultivars.univariate.autoregression.AR` share ``compare`` and
  ``information_criteria`` through the comparison mixin. ``compare``
  raises ``DimensionError`` unless all fits have the same ``nobs``.
- **State space.** ``nonlinear_state_space`` hands the fitted mean to
  :mod:`cultivars.state_space.nonlinear`; ``MarkovSwitchingARResult``'s
  ``state_space`` is the same Hamilton-filter engine that
  :class:`~cultivars.state_space.regime_switching.MarkovSwitchingSSM`
  exposes for general measurements.
- **Vector versions.** :class:`~cultivars.multivariate.nonlinear.threshold.TVAR`,
  :class:`~cultivars.multivariate.nonlinear.smooth_transition.STVAR` and
  :class:`~cultivars.multivariate.regime_switching.markov_switching.MarkovSwitchingVAR`
  in :doc:`nonlinear_multivariate`.
- **Forecast evaluation.** None of these results has a ``forecast``
  method. For a forecast comparison (:doc:`forecasting`), build point
  forecasts from ``simulate`` paths or from ``predict`` on a lag matrix
  you construct.

Nuances and gotchas
-------------------

- **No forecast method.** ``SETARResult``, ``STARResult`` and
  ``MarkovSwitchingARResult`` have no ``forecast``; ``ARNNResult`` and
  ``TARNNResult`` have ``predict`` (one step, given lags you build) but no
  ``simulate``. The multi-step conditional mean of a nonlinear model is a
  simulation problem.
- **No standard errors.** None of the estimators on this page reports
  standard errors; every summary says so.
- **Effective samples differ.** SETAR and STAR drop ``max(order, delay)``
  observations, MS-AR drops ``order``. ``LSTAR(y, order=1, delay=2)`` has
  one fewer observation than ``AR(y, order=1)``, and ``compare`` between
  them raises ``DimensionError``. Trim the series yourself to align
  samples.
- **Parameter counts are not uniform.** ``SETARResult.n_params`` and
  ``STARResult.n_params`` count the coefficient blocks, the threshold and
  (STAR) ``gamma``, but *not* the variance, while
  :class:`~cultivars.univariate.autoregression.AR` counts its variance.
  In the BIC table above SETAR(1) has ``k = 5`` and AR(1) has ``k = 3``:
  the threshold model is penalized one parameter less than a consistent
  count would.
- **The threshold sits on a data point.** Candidates are sample quantiles
  of the lagged transition variable, and ``z <= r`` is assigned to the
  lower regime everywhere (estimation, ``regime_weight``, ``simulate``,
  ``nonlinear_state_space``).
- **Large ``gamma`` is a lower bound.** When ``is_effectively_abrupt`` is
  ``True``, refit as SETAR. ``gamma`` is per standard deviation of the
  transition variable; divide by ``transition_scale`` for the raw-unit
  speed.
- **ESTAR labels.** For ESTAR the "lower" block is the region near the
  threshold, the "upper" block is both tails.
- **TAR refusals.** A TAR result refuses ``simulate`` and
  ``nonlinear_state_space``; so does an externally driven TARNN for the
  latter. ``TAR`` requires a fixed ``delay`` (default 1).
- **Markov-switching seeds.** Different seeds can land on different local
  optima. Every start, including the first, draws a perturbed transition
  matrix from the generator, so even ``n_init=1`` depends on ``seed``.
  Always pass ``seed`` and check ``converged``.
- **``converged=False`` with ``n_iter == screen_iter``.** If the refining
  EM run ends below the best screened log-likelihood, the result falls
  back to the screened start and reports that run's iteration count and a
  ``False`` convergence flag, and the summary leads with a
  non-convergence warning. Refit with a different ``seed`` or larger
  ``n_init`` before trusting the numbers.
- **Testing the number of regimes.** ``likelihood_ratio_test`` refuses any
  comparison that changes :math:`K`, including against a non-switching
  result.
- **Fitted values are mixtures.** MS-AR ``fittedvalues`` and ``resid`` are
  expectations under the smoothed posterior, which uses the full sample;
  they are not real-time quantities. Use ``probabilities("filtered")``
  for anything that must be real time.
- **Configuring the reference network.** The default engine class,
  ``NumpyMLPEngine``, lives in the private module
  ``cultivars.engine._internals._engines``; there is no public import path
  for it. To change ``hidden_units``, ``alpha``, ``n_restarts`` or the
  training ``seed`` you either import that private name or write your own
  engine. ``engine_config`` on the result records the full configuration.
- **Network fits are slow.** The reference engine takes a few seconds per
  fit on a few hundred observations; TARNN trains two networks.
- **ARNN comparisons.** ``compare`` does not stop you ranking an ARNN
  against an AR by BIC; the result is not meaningful. Two ARNN fits with
  different engines share the label ``AR-NN(p)`` in the table; read the
  ``k`` column to tell them apart.

See also
--------

- :doc:`diagnostics` for the linearity tests and BDS.
- :doc:`univariate_linear` for the linear AR/ARMA baselines.
- :doc:`volatility` for conditional-variance models and ARCH tests.
- :doc:`nonlinear_multivariate` for TVAR, STVAR and MS-VAR.
- :doc:`state_space` for filtering the emitted nonlinear systems.
- :doc:`nuances` for package-wide conventions on refusals and
  comparisons.
- API: :mod:`cultivars.univariate.threshold`,
  :mod:`cultivars.univariate.smooth_transition`,
  :mod:`cultivars.univariate.regime_switching`,
  :mod:`cultivars.univariate.mean_function`,
  :mod:`cultivars.diagnostics.nonlinearity`.
