.. meta::
   :description: VECM, VARMA, panel, global, mixed-frequency, functional and Nelson-Siegel systems in cultivars, and which of them structural schemes accept.

Beyond the basic VAR
====================

This page covers the reduced-form systems that keep the VAR's
post-estimation surface but change the design or the solve: cointegrated
systems (VECM, VECMX), moving-average terms (VARMA), pooled panels
(PanelVAR), linked country models (GVAR), mixed sampling frequencies
(MFVAR, MIDASVAR), curve-valued data (FunctionalVAR) and yield-curve
factor models (Dynamic and time-varying Nelson-Siegel). Read :doc:`var`
first: the coefficient layout, ``params`` keys, array index conventions
and test records described there apply unchanged here. Declaring
structural shocks is :doc:`structural_identification`; shrinkage and
large systems are :doc:`bayesian_vars` and :doc:`large_dimensional`.

.. contents:: On this page
   :local:
   :depth: 1

The family at a glance
----------------------

All classes live in :mod:`cultivars.multivariate.reduced_form`; the
module column gives the leaf module to import from.

.. list-table::
   :header-rows: 1
   :widths: 22 18 34 26

   * - Class
     - Module
     - Model in one line
     - Use when
   * - :class:`~cultivars.multivariate.reduced_form.error_correction.VECM`
     - ``error_correction``
     - :math:`\Delta y_t = \alpha\beta' y_{t-1} + \sum \Gamma_i \Delta
       y_{t-i} + u_t`, Johansen reduced-rank ML
     - integrated series with long-run relations
   * - :class:`~cultivars.multivariate.reduced_form.error_correction.VECMX`
     - ``error_correction``
     - the VECM for a block :math:`y` given weakly exogenous I(1)
       :math:`x` (Pesaran, Shin and Smith)
     - a small open economy, or a GVAR unit
   * - :class:`~cultivars.multivariate.reduced_form.moving_average.VARMA`
     - ``moving_average``
     - :math:`y_t = \sum A_i y_{t-i} + u_t + \sum M_j u_{t-j}`,
       Hannan-Rissanen regressions
     - a parsimonious representation with MA dynamics
   * - :class:`~cultivars.multivariate.reduced_form.panel.PanelVAR`
     - ``panel``
     - common :math:`A_l, \Sigma_u` across units, unit intercepts
       (LSDV)
     - many short units with homogeneous dynamics
   * - :class:`~cultivars.multivariate.reduced_form.closed_global.GVAR`
     - ``closed_global``
     - VARX or VECMX units linked by a weight matrix and solved into one
       system
     - many countries, too many variables for one VAR
   * - :class:`~cultivars.multivariate.reduced_form.mixed_frequency.MFVAR`
     - ``mixed_frequency``
     - latent high-frequency VAR observed through temporal aggregation;
       Kalman smoother, parameters supplied
     - nowcasting a quarterly series from monthly data
   * - :class:`~cultivars.multivariate.reduced_form.mixed_frequency.MIDASVAR`
     - ``mixed_frequency``
     - low-frequency VAR plus exponential-Almon MIDAS regressors,
       profiled ML
     - high-frequency regressors for a low-frequency system
   * - :class:`~cultivars.multivariate.reduced_form.functional.FunctionalVAR`
     - ``functional``
     - VAR on basis scores (FPCA, Nelson-Siegel, B-spline), mapped back
       to curves
     - each observation is a curve on a grid
   * - :class:`~cultivars.multivariate.reduced_form.term_structure.DynamicNelsonSiegel`,
       :class:`~cultivars.multivariate.reduced_form.term_structure.TimeVaryingNelsonSiegel`
     - ``term_structure``
     - Diebold-Rudebusch-Aruoba state space, exact (or approximate) ML
     - yield-curve factors with per-maturity noise and missing data

Cointegrated systems: ``VECM`` and ``VECMX``
--------------------------------------------

.. math::

   \Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta y_{t-i}
   + \mu_t + u_t, \qquad \Pi = \alpha\beta', \quad
   \operatorname{rank}(\Pi) = r.

Johansen's (1991, 1995) reduced-rank regression estimates
:math:`\beta` from an eigenvalue problem and then takes :math:`\alpha`,
the :math:`\Gamma_i` and the deterministic terms from one least-squares
regression on ``[deterministic | lagged differences | error-correction
terms]``. ``order`` counts lags of the *levels* VAR, so ``order=p``
carries :math:`p - 1` lagged differences and the inherited
``lag_order_selection`` chooses :math:`p` on the unrestricted levels VAR.

The rank is chosen on the model, before fitting, with ``rank_test()``,
which returns a
:class:`~cultivars.diagnostics.cointegration.JohansenRankTest` (trace and
maximum-eigenvalue sequences with p-values simulated from the asymptotic
null for the deterministic case in hand). The test ignores the ``rank``
the model was built with.

.. testcode::

   import numpy as np
   from cultivars.multivariate.reduced_form.error_correction import VECM

   rng = np.random.default_rng(7)
   common = np.cumsum(rng.standard_normal(300))
   levels = np.column_stack([
       common + 0.5 * rng.standard_normal(300),
       0.5 * common + 0.5 * rng.standard_normal(300),
       np.cumsum(rng.standard_normal(300)),
   ])
   model = VECM(levels, order=2, rank=1, names=("p", "q", "s"))
   test = model.rank_test(simulations=1000)
   print(test)
   print(test.selected_rank(), test.selected_rank(statistic="max_eigenvalue"))
   print(test.trace_pvalue.round(3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   JohansenRankTest(k=3, nobs=298, deterministic='constant', trace_rank=1, max_eigenvalue_rank=1)
   1 1
   [0.    0.096 0.577]

``selected_rank(alpha=0.05, statistic="trace")`` reads the sequence down
from :math:`r = 0` and stops at the first non-rejection; it returns
:math:`k` when every null is rejected (stationary in levels: fit a
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
instead). ``to_table()`` prints the full sequence. ``small_sample=True``
applies the Reinsel-Ahn scaling; ``simulations`` (default 25 000) sets
the p-value resolution and ``seed`` is fixed by default so p-values
reproduce.

.. testcode::

   vecm = model.fit()
   print(vecm.alpha.shape, vecm.beta.shape, vecm.gamma.shape, vecm.coefficients.shape)
   print(vecm.normalized_beta().ravel().round(2), vecm.normalized_alpha().ravel().round(2))
   print(vecm.n_common_trends, vecm.stability_check().n_unit_roots, vecm.is_stable)
   print(list(vecm.params)[:5])
   print(vecm.weak_exogeneity("s").reject(), vecm.weak_exogeneity("q").reject())
   print(vecm.irf(8).shape, vecm.forecast(2).shape)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   (3, 1) (3, 1) (1, 3, 3) (2, 3, 3)
   [ 1.   -2.02  0.02] [-0.16  0.39 -0.01]
   2 2 True
   ['p: const', 'p: D.p.L1', 'p: D.q.L1', 'p: D.s.L1', 'p: ec1']
   False True
   (9, 3, 3) (2, 3)

.. list-table:: VECM specification
   :header-rows: 1
   :widths: 24 20 56

   * - Argument
     - Default
     - Meaning
   * - ``order``
     - required
     - lags of the levels VAR (:math:`p - 1` lagged differences)
   * - ``rank``
     - required
     - cointegrating rank :math:`r`, ``0 <= r < k`` for a closed VECM
   * - ``cointegration_trend``
     - ``"constant"``
     - Johansen's case: ``"none"``, ``"restricted_constant"`` (constant
       inside :math:`\beta`), ``"constant"`` (unrestricted constant,
       drift in levels), ``"restricted_trend"`` (trend inside
       :math:`\beta`), ``"trend"``. Changes the rank test's null
       distribution, not only the display.
   * - ``names``
     - ``y1 ... yk``
     - variable labels (Cholesky order for propagation)
   * - ``exog``, ``exog_names``, ``contemporaneous``
     - ``None``, ``x1 ...``, ``True``
     - accepted by the ``VECM`` constructor, but the resulting
       ``VECMResult`` is broken (see Nuances); use ``VECMX`` for a
       conditional system

What the result offers. Two coordinate systems are kept apart. Inference
(``params``, ``bse``, ``pvalues``, ``conf_int``, ``granger_causality``,
``weak_exogeneity``, residual diagnostics) reads the short-run regression
conditional on :math:`\hat\beta`; its regressors are named
``"D.{name}.L{i}"`` for lagged differences and ``"ec{j}"`` for the
error-correction terms, and the summary appends ``beta[j]: name`` rows
without standard errors. Propagation (``irf``, ``fevd``,
``historical_decomposition``, ``forecast``, ``companion``) reads the
implied levels VAR in ``coefficients`` (also returned by ``to_var()``),
and ``stability_check`` allows exactly :math:`k - r` unit roots.
Further members: ``cointegrating_matrix`` (:math:`\Pi`, ``(k, k)``),
``normalized_beta(on=0)`` and ``normalized_alpha(on=0)`` (the readable
basis and its matching loadings), ``error_correction_terms()``
(``(nobs, r)``), ``eigenvalues``, ``short_run_deterministic`` and the
levels-form ``deterministic`` and ``trend``. ``granger_causality`` in a
VECM tests the short-run and long-run channels jointly.

``VECMX(endog, exog, *, order, rank, ...)`` models only :math:`y` given
weakly exogenous integrated :math:`x`, with :math:`\beta` spanning
:math:`(y', x')'`. It keeps every short-run member and a conditional
``forecast(steps, *, exog_future)``, and refuses the closed-system
members (``irf``, ``fevd``, ``ma_representation``,
``historical_decomposition``, ``companion``, ``stability_check``) with
``SpecificationError``. Its ``to_varx()`` returns the
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.ConditionalLevels`
record a GVAR links. ``rank_test`` on a ``VECMX`` simulates the
conditional null of Pesaran, Shin and Smith (2000).

.. testcode::

   from cultivars.multivariate.reduced_form.error_correction import VECMX

   rng = np.random.default_rng(0)
   x = np.cumsum(rng.standard_normal(300))
   y1 = np.zeros(300)
   for t in range(1, 300):
       y1[t] = y1[t - 1] + 0.4 * (x[t - 1] - y1[t - 1]) + 0.5 * rng.standard_normal()
   block = np.column_stack([y1, np.cumsum(rng.standard_normal(300))])
   cond = VECMX(block, x[:, None], order=2, rank=1,
                names=("a", "b"), exog_names=("x",)).fit()
   print(cond.beta.shape, cond.n_common_trends, list(cond.params)[:6])
   path = cond.forecast(3, exog_future=np.full((3, 1), x[-1]))
   print(path.shape, cond.to_varx().phi.shape)
   try:
       cond.irf(4)
   except Exception as error:
       print(type(error).__name__)

.. testoutput::

   (3, 1) 2 ['a: const', 'a: D.a.L1', 'a: D.b.L1', 'a: D.x.L1', 'a: D.x', 'a: ec1']
   (3, 2) (2, 2, 2)
   SpecificationError

Moving-average terms: ``VARMA``
-------------------------------

.. math::

   y_t = D d_t + \sum_{i=1}^{p} A_i y_{t-i} + u_t + \sum_{j=1}^{q} M_j u_{t-j}.

Estimated by the three-pass Hannan-Rissanen (1982) procedure: a long
autoregression of order :math:`h \approx \sqrt{T}` (``long_order``)
estimates the innovations, a regression on own lags and lagged
innovations estimates :math:`(A, M)`, and a third pass re-derives the
innovations through the model and re-estimates, but only when the
second-stage MA block is invertible (``refined`` records whether it
ran). No optimizer is involved. ``order`` is the pair ``(p, q)`` with
``q >= 1``; ``trend`` and ``names`` are as for ``VAR``.

.. testcode::

   from cultivars.multivariate.reduced_form.moving_average import VARMA

   rng = np.random.default_rng(0)
   A = np.array([[0.5, 0.1], [0.0, 0.4]])
   M = np.array([[0.4, 0.0], [0.2, 0.3]])
   u = rng.standard_normal((402, 2))
   z = np.zeros((402, 2))
   for t in range(1, 402):
       z[t] = A @ z[t - 1] + u[t] + M @ u[t - 1]
   varma = VARMA(z[2:], order=(1, 1), names=("a", "b")).fit()
   print(varma.order, varma.ma_order, varma.long_order, varma.refined, varma.nobs)
   print(varma.ma_coefficients.shape, list(varma.params)[:5])
   psi = varma.ma_representation(3)
   print(np.allclose(psi[1], varma.coefficients[0] + varma.ma_coefficients[0]))
   print(varma.is_stable, varma.is_invertible)

.. testoutput::

   1 1 20 True 398
   (1, 2, 2) ['a: const', 'a: a.L1', 'a: b.L1', 'a: u[a].L1', 'a: u[b].L1']
   True
   True True

The result is a full VAR-style record: ``coefficients`` (:math:`A_i`,
``(p, k, k)``), ``ma_coefficients`` (:math:`M_j`, ``(q, k, k)``,
``ma_coefficients[j-1][r, c]`` the effect of :math:`u_{c, t-j}` on
equation ``r``), ``invertibility_check()`` and ``is_invertible``, plus
the whole inference and propagation surface. ``ma_representation`` is
overridden to :math:`\Psi_h = M_h + \sum_i A_i\Psi_{h-i}`, so ``irf``,
``fevd`` and the decompositions are correct for the VARMA, and
``forecast`` carries the last innovations across the sample boundary.
Standard errors condition on the estimated innovations (they understate
uncertainty modestly), and no echelon-form identification is imposed:
:math:`(A, M)` are not unique, :math:`\Psi_h` is.
``lag_order_selection`` raises; fit candidate ``(p, q)`` pairs on a
common sample and rank them with ``compare``.

Pooled panels: ``PanelVAR``
---------------------------

.. math::

   y_{it} = \mu_i + \sum_{l=1}^{p} A_l y_{i,t-l} + u_{it},
   \qquad i = 1, \dots, N.

Slopes and :math:`\Sigma_u` are common to all units; only the intercept
varies (Holtz-Eakin, Newey and Rosen, 1988). ``panel`` is a balanced
``(n_units, nobs, k)`` array or a list of ``(nobs_i, k)`` arrays
(unbalanced). Lags are built inside each unit, so the effective sample
is :math:`\sum_i (T_i - p)`.

.. list-table:: PanelVAR specification
   :header-rows: 1
   :widths: 20 16 64

   * - Argument
     - Default
     - Meaning
   * - ``effects``
     - ``"unit"``
     - ``"unit"``: one intercept per unit (LSDV, Nickell-biased of order
       :math:`1/T`); ``"none"``: no unit effects
   * - ``trend``
     - ``"n"``
     - pooled deterministic terms; must be ``"n"`` with
       ``effects="unit"`` (collinear otherwise, raises)
   * - ``names``, ``unit_names``
     - ``y1 ...``, ``unit1 ...``
     - variable and unit labels

.. testcode::

   from cultivars.multivariate.reduced_form.panel import PanelVAR

   rng = np.random.default_rng(0)
   A = np.array([[0.5, 0.2], [0.1, 0.4]])
   units = []
   for i in range(5):
       mu = rng.normal(0.0, 2.0, size=2)
       series = np.zeros((60 + 10 * i, 2))
       series[0] = mu
       for t in range(1, series.shape[0]):
           series[t] = mu + A @ (series[t - 1] - mu) + rng.standard_normal(2)
       units.append(series)
   pvar = PanelVAR(units, order=1, names=("g", "i"),
                   unit_names=[f"c{j}" for j in range(5)]).fit()
   print(pvar.n_units, pvar.nobs, pvar.min_time_dimension, pvar.unit_lengths)
   print(pvar.coefficients[0].round(2))
   print(list(pvar.params)[:3], list(pvar.slopes)[:2])
   print(pvar.forecast(2, unit="c3").round(2))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   5 395 60 (60, 70, 80, 90, 100)
   [[0.51 0.15]
    [0.15 0.35]]
   ['g: const[c0]', 'g: const[c1]', 'g: const[c2]'] ['g: g.L1', 'g: i.L1']
   [[-4.41  2.12]
    [-3.79  2.35]]

The result adds ``n_units``, ``unit_names``, ``unit_lengths``,
``min_time_dimension``, ``effects``, ``unit_effects`` (dict of
``(k,)`` intercepts), ``slopes`` (``params`` without the unit
intercepts), ``unit_series(unit)`` and ``unit_residuals(unit)``.
``forecast(steps, *, unit)`` requires a unit name. Residual tests
(portmanteau, ARCH) read within units. ``irf``, ``fevd`` and Granger
tests are those of the pooled dynamics. GMM-in-differences estimators
are not implemented.

Global VARs: ``GVAR``
---------------------

.. math::

   y_{it} = \sum_{l} \Phi_{il} y_{i,t-l} + \Lambda_{i0} y^*_{it}
   + \sum_{l} \Lambda_{il} y^*_{i,t-l} + d_{it} + u_{it},
   \qquad y^*_{it} = \sum_j w_{ij} y_{jt}.

Each unit is a VARX (levels) or VECMX (error correction) fitted against
its own foreign aggregate; ``GVAR(units, weights, *, variable_labels,
unit_names=None).solve()`` stacks the units through the link matrices
and inverts :math:`G_0` (Pesaran, Schuermann and Weiner, 2004). There is
no ``fit``: the step is algebra. ``weights`` must have a zero diagonal,
non-negative entries and unit row sums; ``variable_labels`` gives one
shared label per global column so foreign aggregates average like with
like. Each unit's exogenous block must be exactly the aggregate built
from its weight row, in the unit's own variable order.

.. testcode::

   from cultivars.multivariate.reduced_form.vector_autoregression import VARX
   from cultivars.multivariate.reduced_form.closed_global import GVAR

   rng = np.random.default_rng(0)
   world = np.zeros((300, 3))
   A = np.array([[0.5, 0.2, 0.1], [0.1, 0.5, 0.2], [0.2, 0.1, 0.5]])
   for t in range(1, 300):
       world[t] = A @ world[t - 1] + rng.standard_normal(3)
   W = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
   unit_fits = [
       VARX(world[:, [i]], (world @ W[i])[:, None], order=1, exog_order=1).fit()
       for i in range(3)
   ]
   gvar = GVAR(unit_fits, W, variable_labels=["gdp"] * 3,
               unit_names=("us", "eu", "jp")).solve()
   print(gvar.names, gvar.coefficients.shape, gvar.contemporaneous.shape)
   print(gvar.is_stable, round(gvar.linkage_condition, 3), gvar.unit_columns("eu"))
   fitted = gvar.endog[:-1] @ gvar.coefficients[0].T + gvar.deterministic[0]
   print(np.allclose(gvar.endog[1:], fitted + gvar.resid))
   print(gvar.generalized_irf(4).shape, gvar.forecast(2).shape)

.. testoutput::

   ('us.gdp', 'eu.gdp', 'jp.gdp') (1, 3, 3) (3, 3)
   True 1.034 (1,)
   True
   (5, 3, 3) (2, 3)

``GVARResult`` carries the propagation surface (``irf``,
``generalized_irf``, ``fevd``, ``generalized_fevd``, ``forecast``,
``simulate``, ``historical_decomposition``, ``ma_representation``,
``stability_check``) plus ``linkage_condition`` (condition number of
:math:`G_0`), ``weak_exogeneity_test(unit)`` and ``exogeneity_table()``
for VECMX units, ``units``, ``weights`` and ``unit_columns(unit)``.
It has no ``llf``, ``params``, ``bse`` or ``compare``: an assembled
system has no likelihood or coefficient table. The summary recommends
the generalized responses because a recursive order across countries is
rarely defensible. ``GVARResult`` feeds
:class:`~cultivars.multivariate.large_dim.spillover.Spillover`
(:doc:`large_dimensional`).

Mixed frequencies: ``MFVAR`` and ``MIDASVAR``
---------------------------------------------

The two classes put the model at different frequencies and differ in
whether they estimate.

``MFVAR`` (Mariano and Murasawa, 2003) is a latent high-frequency VAR
:math:`z_t = c + \sum A_i z_{t-i} + u_t` observed directly for
``"high"`` variables and through aggregation for ``"stock"``
(end-of-period value) and ``"flow"`` (weighted sum over the period)
variables. The calendar is encoded as ``nan`` in the sample. **It does
not estimate**: the docstring documents that the Gaussian likelihood
with an aggregated variable converges to a wrong limit. You supply
``coefficients``, ``sigma_u`` (and optionally ``intercept`` and
``weights``), or take them from a VAR fitted on a high-frequency sample
with ``MFVAR.from_result(result, kinds=..., period=...)``; then
``smooth(endog)`` returns an ``MFVARResult``, ``filter(endog)`` the raw
Kalman filter output and ``nowcast(endog)`` the last smoothed period.

.. testcode::

   from cultivars.multivariate.reduced_form.vector_autoregression import VAR
   from cultivars.multivariate.reduced_form.mixed_frequency import MFVAR

   rng = np.random.default_rng(0)
   A = np.array([[0.5, 0.2], [0.3, 0.6]])
   monthly_truth = np.zeros((240, 2))
   for t in range(1, 240):
       monthly_truth[t] = A @ monthly_truth[t - 1] + rng.standard_normal(2)
   observed = monthly_truth.copy()
   observed[:, 1] = np.nan
   quarter_ends = np.arange(2, 240, 3)
   observed[quarter_ends, 1] = [monthly_truth[t - 2:t + 1, 1].mean() for t in quarter_ends]

   high_freq_fit = VAR(monthly_truth[:120], order=1, names=("ip", "gdp")).fit()
   mf = MFVAR.from_result(high_freq_fit, kinds=["high", "flow"], period=3)
   latent = mf.smooth(observed)
   print(latent.endog.shape, latent.observed_fraction.round(3), latent.trend)
   gap = latent.endog[:, 1] - monthly_truth[:, 1]
   print(round(float(np.sqrt(np.mean(gap ** 2))), 2), round(float(monthly_truth[:, 1].std()), 2))
   print({key: round(value, 3) for key, value in latent.nowcast().items()})
   print(latent.irf(3).shape, latent.forecast(2).shape)

.. testoutput::

   (240, 2) [1.    0.333] n
   0.66 1.48
   {'ip': -0.649, 'gdp': 0.411}
   (4, 2, 2) (2, 2)

``MFVARResult`` holds the smoothed path ``endog``, ``latent_cov``,
``latent_stderr``, ``observed``, ``observed_fraction``,
``implied_low_frequency()``, ``nowcast(index=-1)``, ``loglikelihood``
and the full propagation surface of the supplied system. ``from_result``
does not carry the deterministic block over (``trend`` is ``"n"``
above); pass ``intercept=`` to the constructor when levels matter.

``MIDASVAR`` (Ghysels, Santa-Clara and Valkanov, 2004) is a
low-frequency VAR with each high-frequency regressor compressed into one
column per period by a normalized exponential-Almon polynomial with two
parameters, estimated by profiled Gaussian likelihood.
``MIDASVAR(endog, exog_high, *, order, period, midas_lags=None,
trend="c", names, exog_names)`` takes ``exog_high`` with ``period`` rows
per low-frequency row.

.. testcode::

   from cultivars.multivariate.reduced_form.mixed_frequency import MIDASVAR

   months = monthly_truth[:, 0].reshape(80, 3)
   spending = months @ [0.2, 0.3, 0.5] + 0.2 * rng.standard_normal(80)
   quarterly = np.column_stack([spending, observed[quarter_ends, 1]])
   midas = MIDASVAR(quarterly, monthly_truth[:, :1], order=1, period=3,
                    names=["spend", "gdp"], exog_names=["ip"]).fit()
   print(midas.midas_weights.ravel().round(2), midas.midas_coefficients.ravel().round(2))
   print(list(midas.params)[:4], midas.nobs)
   print(midas.forecast(1, exog_high_future=np.zeros((3, 1))).round(3))

.. testoutput::

   [0.47 0.27 0.25] [1.01 0.49]
   ['spend: const', 'spend: spend.L1', 'spend: gdp.L1', 'spend: ip'] 79
   [[ 0.069 -0.029]]

``midas_weights`` is ``(midas_lags, m)``, ordered **most recent
sub-period first** (the true weights above were 0.5, 0.3, 0.2 from the
quarter's last month back) and summing to one, so
``midas_coefficients`` (``(k, m)``) is the total effect of the
regressor. ``theta`` holds the polynomial parameters; ``bse`` conditions
on them, and ``joint_stderr()`` gives full-information standard errors
including ``theta``. The rest of the VAR surface (``irf``, ``fevd``,
Granger tests, diagnostics, ``lag_order_selection``) is inherited.
``forecast`` requires ``exog_high_future`` with ``steps * period``
rows.

Curves: ``FunctionalVAR``
-------------------------

.. math::

   y_t(s) \approx \bar y(s) + \sum_{j=1}^{r} \xi_{tj}\phi_j(s), \qquad
   \xi_t = c + \sum_{i=1}^{p} A_i\xi_{t-i} + u_t.

Each curve is projected on a basis, a VAR is fitted to the scores, and
fitted values, forecasts and responses are mapped back through the
basis matrix :math:`\Phi` (Bosq, 2000; Diebold and Li, 2006).
``FunctionalVAR(curves, grid=None, *, order, basis="fpca",
n_components=None, decay=None, df=None, degree=3, trend="c")``:
``basis="fpca"`` keeps ``n_components`` principal components (default:
enough for 95 percent of variance), ``"nelson-siegel"`` uses level,
slope and curvature loadings with ``decay`` fixed or profiled over the
grid, ``"bspline"`` uses ``df`` B-spline functions.

.. testcode::

   from cultivars.multivariate.reduced_form.functional import FunctionalVAR

   rng = np.random.default_rng(0)
   taus = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
   lam_tau = 0.6 * taus
   slope_load = (1 - np.exp(-lam_tau)) / lam_tau
   loadings = np.column_stack([np.ones(8), slope_load, slope_load - np.exp(-lam_tau)])
   mu = np.array([5.0, -1.5, 1.0])
   ns_factors = np.tile(mu, (120, 1))
   for t in range(1, 120):
       ns_factors[t] = mu + 0.9 * (ns_factors[t - 1] - mu) + 0.2 * rng.standard_normal(3)
   yields = ns_factors @ loadings.T + 0.02 * rng.standard_normal((120, 8))

   fvar = FunctionalVAR(yields, taus, order=1, basis="nelson-siegel").fit()
   print(round(fvar.decay, 3), fvar.factors.names, round(fvar.reconstruction_rmse, 3))
   print(fvar.fitted_curves().shape, fvar.forecast_curves(3).shape, fvar.irf_curves(4).shape)
   pcs = FunctionalVAR(yields, taus, order=1).fit()
   print(pcs.k_factors, pcs.explained_variance.round(3), pcs.factors.names)

.. testoutput::

   0.597 ('level', 'slope', 'curvature') 0.015
   (119, 8) (3, 8) (5, 8, 3)
   2 [0.94  0.049] ('pc1', 'pc2')

``FunctionalVARResult`` carries ``basis_matrix`` (``(n_points, r)``),
``mean_curve`` (zero for Nelson-Siegel, which is not centred),
``scores``, ``explained_variance`` (FPCA only), ``decay``
(Nelson-Siegel only), ``reconstruction_rmse`` (the floor no curve-level
number can beat), ``projected_curves()``, ``fitted_curves()`` (one row
shorter per lag), ``forecast_curves(steps)`` (``(steps, n_points)``) and
``irf_curves(horizon, orthogonalized=True)`` (``(H+1, n_points, r)``,
``[h, :, j]`` the curve's response to factor shock ``j``). The factor
VAR itself is the ordinary ``VARResult`` in ``.factors``, with every
test and decomposition of :doc:`var`.

Yield curves: ``DynamicNelsonSiegel`` and ``TimeVaryingNelsonSiegel``
---------------------------------------------------------------------

.. math::

   y_t(\tau) = L_t + S_t\frac{1 - e^{-\lambda\tau}}{\lambda\tau}
   + C_t\Bigl(\frac{1 - e^{-\lambda\tau}}{\lambda\tau} - e^{-\lambda\tau}\Bigr)
   + \varepsilon_t(\tau), \qquad
   f_t - \mu = \operatorname{diag}(\phi)(f_{t-1} - \mu) + \eta_t.

``DynamicNelsonSiegel(panel, maturities, *, decay=None)`` is the
Diebold, Rudebusch and Aruoba (2006) state space estimated by exact
Kalman-filter ML, with diagonal factor dynamics and one measurement
variance per maturity; ``nan`` entries are handled element-wise. It
needs at least 4 maturities and 30 dates. ``decay=None`` estimates
:math:`\lambda`; a number fixes it.

.. testcode::

   from cultivars.multivariate.reduced_form.term_structure import DynamicNelsonSiegel

   rng = np.random.default_rng(0)
   short_taus = np.array([0.5, 2.0, 5.0, 10.0])
   lt = 0.6 * short_taus
   sl = (1 - np.exp(-lt)) / lt
   basis = np.column_stack([np.ones(4), sl, sl - np.exp(-lt)])
   f = np.tile(mu, (40, 1))
   for t in range(1, 40):
       f[t] = mu + 0.9 * (f[t - 1] - mu) + 0.2 * rng.standard_normal(3)
   curves = f @ basis.T + 0.05 * rng.standard_normal((40, 4))

   dns = DynamicNelsonSiegel(curves, short_taus, decay=0.6).fit()
   print(dns.decay, dns.decay_fixed, dns.n_params, dns.factors.shape)
   mean, std = dns.forecast(6)
   print(mean.shape, std.shape, bool(np.all(np.diff(std, axis=0) >= -1e-12)))
   print(dns.curve(-1, maturities=[1.0, 30.0]).shape)

.. testoutput::

   0.6 True 16.0 (40, 3)
   (6, 4) (6, 4) True
   (2,)

``DynamicNelsonSiegelResult`` holds ``mu``, ``ar`` (diagonal
persistence), ``state_innovation_cov``, ``measurement_var``, smoothed
``factors`` and ``factor_cov``, the ``level``/``slope``/``curvature``
paths, ``loadings``, ``fitted_curves()``, ``curve(date_index,
maturities=None)`` on any grid, ``state_space`` (a
:class:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM`) and
``forecast(steps)``, which returns ``(mean, std)`` arrays of shape
``(steps, n_maturities)``; ``std`` includes factor and measurement
uncertainty but not parameter uncertainty. It also has ``compare`` and
``likelihood_ratio_test`` (fixed against estimated decay is a valid
nested pair).

``TimeVaryingNelsonSiegel(panel, maturities).fit(filter="unscented")``
adds :math:`\log\lambda_t` as a fourth AR(1) state (Koopman, Mallee and
van der Wel, 2010) and maximizes the extended (``filter="extended"``) or
unscented Kalman approximation of the likelihood. Its result exposes
``decay`` (the smoothed path), ``decay_mean``, ``hump_maturity``,
``loadings(t)``, ``fitted``, ``log_decay``, ``log_decay_std`` and the
factor paths. It has no ``forecast``. Its likelihood is approximate and
filter-specific, which the summary and comparison label state.
Estimation is slow (tens of seconds for 60 dates), so this example is
not run in the documentation build:

.. code-block:: python

   from cultivars.multivariate.reduced_form.term_structure import TimeVaryingNelsonSiegel

   tv = TimeVaryingNelsonSiegel(yields[:60], taus).fit(filter="extended")
   tv.decay.shape          # (60,)
   tv.n_params             # 23.0: DNS's 21 with the constant decay replaced by
                           # an AR(1) in log-decay (mean, persistence, sd)
   dns_full = DynamicNelsonSiegel(yields[:60], taus).fit()
   dns_full.compare(tv)    # rows labelled 'DNS(lambda estimated)', 'DNS-TV[extended]'

How the pieces connect
----------------------

Structural identification schemes in
:mod:`cultivars.multivariate.structural` read a fitted reduced form
through the ``ClosedSystemResult`` protocol
(``cultivars.engine._core._protocols``): ``names``, ``nobs``,
``sigma_u``, ``resid``, ``coefficients``, ``k_endog`` and
``ma_representation(horizon)``. The scheme takes the result in its
constructor, e.g. ``RecursiveSVAR(result).identify()``.

.. list-table::
   :header-rows: 1
   :widths: 26 16 18 40

   * - Result
     - Closed system
     - ``forecast``
     - Notes
   * - ``VECMResult``
     - yes
     - points
     - propagation on the implied levels VAR (unit roots allowed)
   * - ``VECMXResult``
     - **no**
     - points, needs ``exog_future``
     - refuses ``irf``/``fevd``/``ma_representation``; passes the
       protocol check anyway (see Nuances)
   * - ``VARMAResult``
     - yes
     - points
     - schemes read the VARMA :math:`\Psi_h`
   * - ``PanelVARResult``
     - yes
     - points, needs ``unit``
     - pooled dynamics
   * - ``GVARResult``
     - yes
     - points
     - also feeds ``Spillover``
   * - ``MFVARResult``
     - yes
     - points
     - parameters supplied, not estimated
   * - ``MIDASVARResult``
     - yes
     - points, needs ``exog_high_future``
     - responses of the low-frequency block
   * - ``FunctionalVARResult``
     - no (use ``.factors``)
     - curves
     - ``.factors`` is a ``VARResult`` and is closed
   * - ``DynamicNelsonSiegelResult``
     - no
     - mean and std
     - state-space model; see :doc:`state_space`
   * - ``TimeVaryingNelsonSiegelResult``
     - no
     - none
     -

.. testcode::

   from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR

   for fitted in (vecm, varma, pvar, gvar, latent, midas, fvar.factors):
       print(type(fitted).__name__, RecursiveSVAR(fitted).identify().impact.shape)

.. testoutput::

   VECMResult (3, 3)
   VARMAResult (2, 2)
   PanelVARResult (2, 2)
   GVARResult (3, 3)
   MFVARResult (2, 2)
   MIDASVARResult (2, 2)
   VARResult (3, 3)

Other connections: the unit-root and cointegration tests to run before
choosing levels, differences or a VECM are in :doc:`diagnostics`;
``VARXResult.to_varx()`` and ``VECMXResult.to_varx()`` produce the
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.ConditionalLevels`
records ``GVAR`` links; ``MFVAR.from_result`` reads any fitted result
with ``coefficients``, ``sigma_u`` and ``names``; the Nelson-Siegel
models expose their fitted state space for the tools in
:doc:`state_space`.

Nuances and gotchas
-------------------

* **VECMX passes the closed-system check.** ``VECMXResult`` subclasses
  ``VECMResult``, so ``RecursiveSVAR(vecmx_result).identify()`` succeeds
  and returns an impact matrix; the error only appears when the
  structural result's ``irf`` calls the refused
  ``ma_representation``. Do not identify conditional models.

  .. testcode::

     svar = RecursiveSVAR(cond).identify()
     try:
         svar.irf(2)
     except Exception as error:
         print(type(error).__name__)

  .. testoutput::

     SpecificationError

* **Do not pass exog to VECM.** ``VECM(..., exog=x)`` fits, but the
  returned ``VECMResult`` has an empty ``coefficients`` stack and
  ``params``, ``summary()``, ``irf`` and ``forecast`` all fail with
  ``DimensionError``, ``ValueError`` or ``IndexError``. The conditional
  model is ``VECMX``.

  .. testcode::

     bad = VECM(block, order=2, rank=1, exog=x[:, None]).fit()
     print(bad.coefficients.shape)
     try:
         bad.params
     except Exception as error:
         print(type(error).__name__)

  .. testoutput::

     (0, 2, 2)
     DimensionError

* **VECM order counts levels lags.** ``order=2`` means one lagged
  difference. ``coefficients`` is the ``(p, k, k)`` levels stack, not
  :math:`\Gamma`; the short-run matrices are ``gamma``.
* **Only the cointegrating space is identified.** Read ``beta`` through
  ``normalized_beta(on=...)``; the ``ec`` coefficients in ``params``
  pair with the raw ``beta``, ``normalized_alpha`` with the normalized
  one. ``beta`` has no standard errors. ``normalized_beta`` raises when
  the pivot variable has a zero loading.
* **VECM Granger tests reject whenever the effect adjusts.** With one
  relation the long-run restriction's Wald contribution equals the
  squared z-statistic of :math:`\alpha_{e}` for any non-zero
  :math:`\hat\beta_c`; check :math:`\beta_c` before reading a rejection
  as causality.
* **Do not compare VECMs of different rank by likelihood ratio.** The
  rank statistic is not chi-squared; use ``rank_test()``.
* **VARMA simulation ignores the MA block.** ``VARMAResult.simulate``
  is the inherited VAR simulator and draws from the AR part only, so
  simulated paths do not have the fitted autocovariances: below, the
  fitted VARMA implies a lag-one cross-correlation of 0.24, the AR part
  alone 0.13, and ``simulate`` reproduces the latter. ``forecast`` and
  ``irf`` do use the MA terms.

  .. testcode::

     def lag1_corr(psi, sigma):
         # corr(y2_t, y1_{t-1}) implied by MA weights psi
         g0 = sum(p @ sigma @ p.T for p in psi)
         g1 = sum(psi[h + 1] @ sigma @ psi[h].T for h in range(len(psi) - 1))
         return round(float(g1[1, 0] / np.sqrt(g0[0, 0] * g0[1, 1])), 2)

     ar_only = np.array([np.linalg.matrix_power(varma.coefficients[0], h)
                         for h in range(201)])
     sim = varma.simulate(50000, seed=1)
     print(lag1_corr(varma.ma_representation(200), varma.sigma_u),
           lag1_corr(ar_only, varma.sigma_u),
           round(float(np.corrcoef(sim[1:, 1], sim[:-1, 0])[0, 1]), 2))

  .. testoutput::

     0.24 0.13 0.13

* **PanelVAR historical decomposition crosses unit boundaries.** It is
  the inherited single-series recursion over the stacked residuals, so
  a unit's early contributions include the previous unit's shocks.
  Compute it per unit from ``unit_residuals`` if you need it.

  .. testcode::

     hd = pvar.historical_decomposition()
     first_of_c1 = pvar.unit_lengths[0] - 1   # first effective row of unit c1
     print(round(float(np.abs(hd[first_of_c1].sum(axis=1) - pvar.resid[first_of_c1]).max()), 2))

  .. testoutput::

     0.93

* **PanelVAR cannot simulate with unit effects.** ``simulate`` raises
  ``DimensionError`` because the unit dummies do not fit the single-path
  simulator. ``effects="unit"`` also forbids ``trend="c"``.
* **Panel forecasts are per unit.** ``forecast`` without ``unit=``
  raises ``SpecificationError``.
* **VECMX simulate fails with IndexError** rather than the
  ``SpecificationError`` the other refused members raise.
* **GVAR is not an estimate.** No ``llf``, ``params``, ``bse``,
  ``compare`` or ``likelihood_ratio_test``; weak-exogeneity tests exist
  only for VECMX units (levels units are listed as untestable).
* **MFVAR takes parameters, it does not fit them.** ``from_result``
  drops the intercept; ``nowcast`` reads the smoother's last row. All
  ``kinds`` equal to ``"high"`` raises (use ``VAR``).
* **MIDAS weights run backwards in time.** ``midas_weights[0]`` is the
  last sub-period of the low-frequency period. ``joint_stderr()`` keys
  are ordered regressor-major (``'spend: const', 'gdp: const', ...``),
  unlike ``params``.
* **Conditional forecasts need paths.** ``VECMX``, ``MIDASVAR`` (and
  ``VARX``) refuse to forecast without the future exogenous or
  high-frequency path; ``MIDASVAR`` needs ``steps * period`` rows.
* **FunctionalVAR fitted curves are shorter.** They lose ``order``
  rows; the Nelson-Siegel basis has ``mean_curve`` equal to zero, the FPCA basis the sample
  mean. ``irf_curves`` defaults to Cholesky-orthogonalized factor shocks
  (one standard deviation), not unit shocks.
* **Nelson-Siegel estimation is slow.** Each likelihood evaluation runs
  a Kalman filter; expect tens of seconds for a few dozen dates. Fixing
  ``decay`` removes one dimension. ``TimeVaryingNelsonSiegelResult``
  has no ``forecast``, and ``fitted`` is an array property, not a
  method.

See also
--------

* :doc:`var` -- the base surface every model here inherits.
* :doc:`structural_identification`, :doc:`diagnostics`,
  :doc:`state_space`, :doc:`large_dimensional`, :doc:`bayesian_vars`,
  :doc:`forecasting`, :doc:`nuances`.
* :mod:`cultivars.multivariate.reduced_form.error_correction`,
  :mod:`cultivars.multivariate.reduced_form.moving_average`,
  :mod:`cultivars.multivariate.reduced_form.panel`,
  :mod:`cultivars.multivariate.reduced_form.closed_global`,
  :mod:`cultivars.multivariate.reduced_form.mixed_frequency`,
  :mod:`cultivars.multivariate.reduced_form.functional`,
  :mod:`cultivars.multivariate.reduced_form.term_structure`,
  :class:`~cultivars.diagnostics.cointegration.JohansenRankTest`.

References: Johansen (1991, 1995); Pesaran, Shin and Smith (2000);
Hannan and Rissanen (1982); Holtz-Eakin, Newey and Rosen (1988); Nickell
(1981); Pesaran, Schuermann and Weiner (2004); Mariano and Murasawa
(2003); Ghysels, Santa-Clara and Valkanov (2004); Bosq (2000); Diebold
and Li (2006); Diebold, Rudebusch and Aruoba (2006); Koopman, Mallee and
van der Wel (2010).
