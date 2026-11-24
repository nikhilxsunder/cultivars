.. meta::
   :description: Reduced-form VAR and VARX in cultivars: specification, lag selection, coefficient layout, inference, diagnostics, IRF, FEVD and forecasts.

Vector autoregressions
======================

This page covers the reduced-form vector autoregression and its
exogenous-regressor variant: how to specify and estimate them, how the
fitted coefficients are laid out, and what the result answers --
inference, residual diagnostics, stability, forecasts, impulse responses,
variance and historical decompositions, Granger causality and model
comparison. It is not about declaring structural shocks (see
:doc:`structural_identification`), full posterior simulation for
Bayesian VARs (see :doc:`bayesian_vars`), or cointegrated, VARMA, panel,
global and mixed-frequency systems (see :doc:`var_extensions`).

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
     - Role in one line
     - Use when
   * - :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
     - :mod:`~cultivars.multivariate.reduced_form.vector_autoregression`
     - :math:`y_t = D d_t + \sum_{i=1}^p A_i y_{t-i} + u_t` by
       multivariate least squares, optionally shrunk by a prior
     - every variable is modelled and gets an equation
   * - :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARX`
     - same
     - the VAR plus a distributed lag :math:`\sum_{j=0}^s B_j x_{t-j}` of
       weakly exogenous regressors
     - some series are conditioned on (oil price, foreign rate) rather
       than modelled
   * - :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`,
       :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARXResult`
     - same
     - frozen fitted records returned by ``fit()``
     - always; this is where every answer lives
   * - :class:`~cultivars.multivariate.reduced_form.vector_autoregression.CoefficientInference`
     - same
     - protocol of the object behind ``coefficient_covariance``
       (sampling or posterior covariance of the coefficients)
     - custom Wald tests and the full :math:`\operatorname{vec}(B)`
       covariance
   * - :class:`~cultivars.multivariate.reduced_form.vector_autoregression.ConditionalLevels`
     - same
     - record returned by ``VARXResult.to_varx()``: a unit's equations in
       levels
     - feeding a VARX unit to
       :class:`~cultivars.multivariate.reduced_form.closed_global.GVAR`

The model
---------

.. math::

   y_t = D d_t + A_1 y_{t-1} + \cdots + A_p y_{t-p} + u_t,
   \qquad \mathrm{E}(u_t) = 0, \quad \mathrm{E}(u_t u_t') = \Sigma_u,

with :math:`y_t` of dimension :math:`k` and :math:`d_t` holding a
constant, a constant and a linear trend, or nothing. Writing the
regressors of all equations as one matrix :math:`Z` gives :math:`Y = Z B
+ U`, and multivariate least squares :math:`\hat B = (Z'Z)^{-1}Z'Y` is
equation-by-equation OLS and, under Gaussian innovations, the
conditional maximum-likelihood estimator at the same time (Lütkepohl,
2005, ch. 3; Hamilton, 1994, ch. 11). There is no separate ``method``
argument: OLS and ML coincide for the coefficients, and the result
stores both residual covariances (see `Estimation and coefficient
layout`_). The reduced form originates with Sims (1980).

A first fit
-----------

The examples on this page share one simulated three-variable system with
a constant and correlated innovations.

.. testcode::

   import numpy as np
   from cultivars.multivariate.reduced_form.vector_autoregression import VAR

   rng = np.random.default_rng(42)
   A1 = np.array([[0.5, 0.1, 0.0],
                  [0.2, 0.4, 0.1],
                  [0.0, 0.3, 0.3]])
   impact = np.array([[1.0, 0.0, 0.0],
                      [0.5, 1.0, 0.0],
                      [0.2, 0.3, 1.0]])
   y = np.zeros((350, 3))
   for t in range(1, 350):
       y[t] = [0.2, 0.1, -0.1] + A1 @ y[t - 1] + impact @ rng.standard_normal(3)
   y = y[50:]

   res = VAR(y, order=1, names=("gdp", "infl", "rate")).fit()
   print(res.coefficients.shape, res.deterministic.shape, res.nobs, res.n_params)
   print(res.coefficients[0].round(2))
   print(res.is_stable, round(res.stability_check().max_modulus, 3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   (1, 3, 3) (1, 3) 299 18.0
   [[ 0.54  0.02  0.13]
    [ 0.21  0.38  0.11]
    [-0.07  0.29  0.26]]
   True 0.643

``print(res)`` (or ``res.summary()``) renders a coefficient table with
the information criteria, the stability verdict and a standing note that
orthogonalized responses rest on the ordering of ``names``.

Specification
-------------

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - Argument
     - Default
     - Meaning
   * - ``endog``
     - required
     - ``(nobs, k)`` array, time down the rows. A ``pandas.DataFrame``
       is accepted as an array; its column names are **not** read.
   * - ``order``
     - required, keyword
     - lag length :math:`p \ge 0`. ``order=0`` is allowed (a constant
       plus white noise); Granger tests then refuse.
   * - ``trend``
     - ``"c"``
     - ``"n"`` (none), ``"c"`` (constant), ``"ct"`` (constant and linear
       trend). The trend regressor counts observations from 1 at the
       first sample row, so on the first usable row of a VAR(p) it
       equals :math:`p + 1`.
   * - ``names``
     - ``y1 ... yk``
     - one unique label per column. The order of ``names`` is the
       Cholesky order used by ``irf``, ``fevd``, ``structural_shocks``
       and ``historical_decomposition``.
   * - ``prior``
     - ``None``
     - a prior from :mod:`cultivars.bayes.priors` (for example
       :class:`~cultivars.bayes.priors.MinnesotaPrior`). Replaces the
       least-squares solve by the posterior mean and switches inference
       to posterior moments; see `Shrinkage through prior=`_.

:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARX`
takes ``(endog, exog)`` positionally plus ``order``, ``exog_order``
(default ``0``: contemporaneous term only), ``trend``, ``names`` and
``exog_names`` (default ``x1 ... xm``). It does not accept ``prior``.

The constructor checks that the sample can support the design: without
a prior it needs at least :math:`p + (d + kp) + 1` rows, where :math:`d`
is the number of deterministic columns.

Choosing the lag order
----------------------

``lag_order_selection`` lives on the *model*, so it runs before any fit.
Every order from 0 to ``max_lags`` is scored on the same effective
sample, trimmed at ``max_lags``, so a column of the table is comparable
across orders.

.. testcode::

   model = VAR(y, order=1, names=("gdp", "infl", "rate"))
   selection = model.lag_order_selection(max_lags=6)
   print(selection)
   print(selection.selected, selection.consensus, selection.nobs)
   print(selection.curve("bic").round(3)[:3], selection.best("aic"))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   LagOrderSelection(max_lags=6, nobs=294, AIC=1, BIC=1, HQIC=1, FPE=1)
   {'aic': 1, 'bic': 1, 'hqic': 1, 'fpe': 1} 1 294
   [0.779 0.177 0.315] 1

The record carries the four whole curves (``aic``, ``bic``, ``hqic``,
``fpe``, indexed by order from zero), ``orders``, ``best(criterion)``,
``selected``, ``consensus`` (``None`` when the criteria disagree) and
``to_table()``. ``max_lags=None`` uses the largest order the sample
supports; a larger request raises ``SpecificationError``.

Estimation and coefficient layout
---------------------------------

The fitted record stores the estimate in two shapes.

``coefficients``
   ``(p, k, k)``. ``coefficients[i][r, c]`` is the effect of variable
   ``c`` at lag ``i + 1`` on equation ``r``, so ``coefficients[0]`` is
   :math:`A_1` in the usual row-equation orientation.
``deterministic``
   ``(d, k)``, one row per deterministic term in the order constant,
   trend; zero rows when ``trend="n"``. Column ``r`` belongs to equation
   ``r``.
``design``
   ``(nobs, d + kp)`` regressor matrix in the column order
   ``[const, trend, y1.L1 ... yk.L1, y1.L2, ...]``.

The labelled views follow the design order. ``params`` is a flat
``dict`` keyed ``"{equation}: {regressor}"``, equation-major;
``equation(name)`` returns one equation's coefficients keyed by
regressor. Lag regressors are named ``"{variable}.L{lag}"``, the
deterministic ones ``"const"`` and ``"trend"``.

.. testcode::

   print(list(res.params)[:5], len(res.params))
   print({key: round(value, 3) for key, value in res.equation("infl").items()})
   print(res.params["infl: gdp.L1"] == res.coefficients[0][1, 0])

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   ['gdp: const', 'gdp: gdp.L1', 'gdp: infl.L1', 'gdp: rate.L1', 'infl: const'] 12
   {'const': 0.084, 'gdp.L1': 0.211, 'infl.L1': 0.381, 'rate.L1': 0.111}
   True

Two residual covariances are stored. ``sigma_u`` divides
:math:`\hat U'\hat U` by :math:`T - (d + kp)` and feeds every
inferential quantity (standard errors, Wald tests, the Cholesky impact
matrix); ``sigma_ml`` divides by :math:`T` and feeds ``llf`` and the
information criteria. ``nobs`` is the effective sample :math:`n - p`,
``resid`` and ``fittedvalues`` are ``(nobs, k)`` and aligned with
``endog[p:]``. ``n_params`` counts :math:`k(d + kp) + k(k+1)/2`, the
covariance included, and ``information_criteria`` returns
``InformationCriteria(aic, bic, hqic)`` on the total-likelihood scale
:math:`-2\ell + \text{penalty}`.

Coefficient inference
---------------------

``bse``, ``tvalues`` and ``pvalues`` are dicts keyed exactly like
``params``; ``conf_int(alpha=0.05)`` maps the same keys to
``(lower, upper)`` tuples. The statistics are referred to the standard
normal, not a :math:`t`, and the summary heads the column ``z``.

.. testcode::

   print(round(res.bse["gdp: rate.L1"], 4), round(res.tvalues["gdp: rate.L1"], 3),
         round(res.pvalues["gdp: rate.L1"], 3))
   low, high = res.conf_int(alpha=0.10)["gdp: rate.L1"]
   print(round(low, 3), round(high, 3))

.. testoutput::

   0.0583 2.16 0.031
   0.03 0.222

``coefficient_covariance`` returns the object these are computed from,
which satisfies the
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.CoefficientInference`
protocol: ``coefficients`` and ``stderr``, ``tstat``, ``pvalue`` as
``(d + kp, k)`` arrays (regressor rows, equation columns),
``conf_int(alpha=...)`` returning ``(lower, upper)`` arrays,
``to_matrix()`` for the full covariance of :math:`\operatorname{vec}(B)`
stacked equation by equation (index ``equation * width + regressor``),
``wald(cells, null=...)`` for a joint zero restriction on
``(equation, regressor)`` index pairs, and ``wald_restriction(R,
null=...)`` for a general :math:`R\operatorname{vec}(B) = 0`.

.. testcode::

   cov = res.coefficient_covariance
   print(cov.stderr.shape, cov.to_matrix().shape)
   # rate.L1 (design column 3) in the gdp and infl equations (0 and 1)
   joint = cov.wald([(0, 3), (1, 3)], null="rate.L1 is zero in gdp and infl")
   print(round(joint.statistic, 3), joint.df, round(joint.pvalue, 4))

.. testoutput::

   (4, 3) (12, 12)
   5.112 2 0.0776

Every test in this family returns a Wald or likelihood-ratio record with
``statistic``, ``df``, ``pvalue``, ``null``, ``reject(alpha=0.05)`` and
``summary()``.

Shrinkage through ``prior=``
----------------------------

Passing a prior from :mod:`cultivars.bayes.priors` keeps the VAR a point
estimator but replaces least squares by the posterior mean under that
prior. Three things change on the result: ``coefficient_covariance`` is
a posterior covariance (so ``bse`` are posterior standard deviations and
the summary drops the p-value column and relabels the others
``post. sd`` and ``m/sd``); ``sigma_u`` is corrected by the *effective*
number of parameters; and ``n_params`` becomes that effective, generally
fractional, count, which is what the information criteria then charge.
``posterior`` holds the posterior object (``None`` without a prior) and
``prior_label`` describes it. A proper prior also lifts the sample-size
check, so a system with more regressors than observations estimates.

.. testcode::

   from cultivars.bayes.priors import MinnesotaPrior

   shrunk = VAR(y, order=1, names=("gdp", "infl", "rate"),
                prior=MinnesotaPrior(tightness=0.1)).fit()
   print(shrunk.prior_label)
   print(round(shrunk.n_params, 2), shrunk.coefficients[0].diagonal().round(2))
   wide = VAR(y[:20], order=8, prior=MinnesotaPrior(tightness=0.2)).fit()
   print(wide.coefficients.shape, wide.nobs)

.. testoutput::

   minnesota(l1=0.1, l2=0.5, l3=1, l4=100)
   13.81 [0.66 0.62 0.5 ]
   (8, 3, 3) 12

The default ``MinnesotaPrior`` centres own first lags at one
(``persistence=1.0``, the random-walk prior of Litterman, 1986), so on
stationary data it pulls own-lag coefficients up, as above. Full
posterior simulation, marginal likelihoods and posterior predictive
checks are the job of the Bayesian VAR classes in :doc:`bayesian_vars`.

Residual diagnostics
--------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Method
     - Test
   * - ``portmanteau_test(lags=10)``
     - multivariate Ljung-Box statistic on the residual autocovariances
       through ``lags``; rejection means the lag order is too short
   * - ``normality_test(variable)``
     - Jarque-Bera on one equation's residuals, 2 degrees of freedom
   * - ``arch_test(variable, lags=5)``
     - Engle's LM test, :math:`(T - \text{lags}) R^2` from a regression
       of squared residuals on their lags
   * - ``residual_diagnostics(portmanteau_lags=10, arch_lags=5)``
     - all of the above in one table, one Jarque-Bera and one ARCH row
       per equation

.. testcode::

   print(res.residual_diagnostics(portmanteau_lags=8, arch_lags=4))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

                                 Residual diagnostics
   ================================================================================
   Observations:                      299  Equations:                           3
   --------------------------------------------------------------------------------
   test                  statistic   df   p-value
   portmanteau(8)           65.529   60    0.2910
   jarque-bera[gdp]          0.898    2    0.6381
   jarque-bera[infl]         0.085    2    0.9585
   jarque-bera[rate]         0.109    2    0.9471
   arch-lm(4)[gdp]           2.622    4    0.6229
   arch-lm(4)[infl]          7.796    4    0.0993
   arch-lm(4)[rate]          0.335    4    0.9875
   ================================================================================
   Small p-values reject the null named by each test. A rejected portmanteau means
   the lag order is too short, which invalidates every other quantity the result
   reports.

The portmanteau degrees of freedom are :math:`k^2 h - k(d + kp)`: the
deterministic columns are charged as well as the lags (60 above, where
the textbook :math:`k^2(h - p)` gives 63). A ``lags`` value that leaves no
degrees of freedom raises ``SpecificationError``.

Stability and the companion form
--------------------------------

``companion`` is the ``(kp, kp)`` companion matrix and
``stability_check()`` its eigenvalue verdict, a record with
``eigenvalues`` (complex), ``max_modulus``, ``is_stable``,
``n_unit_roots``, ``n_explosive`` and ``tol``. ``is_stable`` is the
shortcut. Every propagation method below presumes stability; none of
them refuses an unstable system, and the summary prints a warning first
when the system is not stable.

.. testcode::

   check = res.stability_check()
   print(res.companion.shape, check.n_unit_roots, check.n_explosive)
   print(np.sort(np.abs(check.eigenvalues)).round(3))

.. testoutput::

   (3, 3) 0 0
   [0.23  0.307 0.643]

Forecasting and simulation
--------------------------

``forecast(steps)`` iterates the fitted system forward from the last
``p`` observations with future innovations at zero and the
deterministic terms extrapolated. It returns a ``(steps, k)`` array of
point forecasts; row ``h`` is the forecast of :math:`y_{T+h+1}`. **No
forecast interval is returned**, and there is no method for one on the
reduced-form result. Forecast densities and evaluation are covered in
:doc:`forecasting`.

``simulate(n=200, seed=None, burn=100)`` draws a fresh ``(n, k)`` path
from the fitted coefficients with Gaussian innovations of covariance
``sigma_u``, starting from zeros and discarding ``burn`` periods.

.. testcode::

   print(res.forecast(3).round(3))
   hand = res.deterministic[0] + res.coefficients[0] @ y[-1]
   print(np.allclose(res.forecast(1)[0], hand))
   print(res.simulate(100, seed=1).shape)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   [[ 0.556  0.288 -0.16 ]
    [ 0.505  0.293 -0.137]
    [ 0.48   0.287 -0.126]]
   True
   (100, 3)

Impulse responses
-----------------

All propagation arrays share one index convention:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Method
     - Shape
     - Entry
   * - ``ma_representation(horizon=20)``
     - ``(H+1, k, k)``
     - ``[h, i, j]`` is :math:`\Psi_h[i, j]`, with :math:`\Psi_0 = I`
   * - ``irf(horizon=20, orthogonalized=True, cumulative=False)``
     - ``(H+1, k, k)``
     - ``[h, i, j]``: response of variable ``i`` at horizon ``h`` to
       shock ``j``
   * - ``generalized_irf(horizon=20, cumulative=False)``
     - ``(H+1, k, k)``
     - same reading, Pesaran-Shin shocks
   * - ``fevd(horizon=20)``, ``generalized_fevd(horizon=20)``
     - ``(H+1, k, k)``
     - ``[h, i, j]``: share of variable ``i``'s ``(h+1)``-step forecast
       error variance due to shock ``j``; sums to one over ``j``
   * - ``historical_decomposition()``
     - ``(nobs, k, k)``
     - ``[t, i, j]``: cumulative contribution of shock ``j`` to variable
       ``i`` at effective-sample date ``t``
   * - ``structural_shocks()``
     - ``(nobs, k)``
     - Cholesky-orthogonalized residuals :math:`P^{-1}\hat u_t`

``horizon`` is the largest lead, so every array has ``horizon + 1``
rows with the impact at index 0.

With ``orthogonalized=True`` (the default) the response is
:math:`\Psi_h P` with :math:`P` the lower Cholesky factor of
``sigma_u``: a one-standard-deviation shock under the recursive ordering
of ``names``. This is a recursive SVAR whose identifying assumption is
the column order; permute ``names`` and the answer changes. With
``orthogonalized=False`` the response is :math:`\Psi_h`, a unit
reduced-form innovation, whose columns are not interpretable one at a
time because the innovations are correlated. ``cumulative=True`` returns
running sums over ``h`` (levels responses of differenced data).
``generalized_irf`` (Pesaran and Shin, 1998) is
:math:`\Psi_h \Sigma_u e_j / \sqrt{\sigma_{jj}}`; it needs no ordering
and coincides with the orthogonalized response for the first variable
only.

.. testcode::

   irf = res.irf(10)
   print(irf.shape)
   print(np.allclose(irf[0], np.linalg.cholesky(res.sigma_u)))
   print(np.allclose(res.irf(10, orthogonalized=False), res.ma_representation(10)))
   print(np.allclose(res.irf(10, cumulative=True)[-1], irf.sum(axis=0)))
   girf = res.generalized_irf(10)
   print(np.allclose(girf[:, :, 0], irf[:, :, 0]), np.allclose(girf[:, :, 1], irf[:, :, 1]))
   # response of rate (i=2) at horizon 2 to an infl shock (j=1)
   print(round(irf[2, 2, 1], 3))

.. testoutput::

   (11, 3, 3)
   True
   True
   True
   True False
   0.211

Confidence bands for impulse responses are not produced by the reduced
form. Declared identification and its inference live in
:doc:`structural_identification`; posterior bands come from
:doc:`bayesian_vars`.

Variance and historical decompositions
--------------------------------------

``fevd`` uses the orthogonalized responses and inherits their ordering
dependence; at ``h = 0`` it is exactly triangular, so the first variable
is attributed all of its own impact variance by construction.
``generalized_fevd`` normalizes the squared generalized responses so
rows sum to one (Diebold and Yilmaz, 2012); because generalized shocks
are correlated, read it as a relative ranking, not as an exact
decomposition.

.. testcode::

   shares = res.fevd(10)
   print(shares[0].round(3))
   print(shares[10].round(3))
   print(np.allclose(shares.sum(axis=2), 1.0))
   print(res.generalized_fevd(10)[10].round(3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   [[1.    0.    0.   ]
    [0.286 0.714 0.   ]
    [0.038 0.097 0.866]]
   [[0.967 0.014 0.019]
    [0.396 0.589 0.015]
    [0.077 0.21  0.713]]
   True
   [[0.689 0.241 0.07 ]
    [0.261 0.62  0.119]
    [0.061 0.225 0.714]]

``historical_decomposition()`` attributes the stochastic part of each
observation to the Cholesky shocks. Summing over the last axis recovers
:math:`\sum_{s=0}^{t}\Psi_s\hat u_{t-s}`, *not* the observed series: the
deterministic terms and the effect of pre-sample initial conditions are
excluded. Its cost is :math:`O(\text{nobs}^2)`.

.. testcode::

   hd = res.historical_decomposition()
   psi = res.ma_representation(res.nobs - 1)
   t = 50
   by_hand = sum(psi[s] @ res.resid[t - s] for s in range(t + 1))
   print(hd.shape, np.allclose(hd[t].sum(axis=1), by_hand))
   print(np.cov(res.structural_shocks().T).round(2).diagonal())

.. testoutput::

   (299, 3, 3) True
   [0.99 0.99 0.99]

The structural shocks have sample variance :math:`(T - d - kp)/(T-1)`
rather than one, because ``sigma_u`` is degrees-of-freedom corrected.

Granger causality
-----------------

``granger_causality(cause, effect)`` is the Wald test that all ``p``
lags of ``cause`` are zero in the ``effect`` equation (Granger, 1969),
using ``coefficient_covariance``. Both arguments are variable names.

.. testcode::

   test = res.granger_causality("rate", "gdp")
   print(test.null)
   print(round(test.statistic, 3), test.df, round(test.pvalue, 4), test.reject())
   print(res.granger_causality("infl", "gdp").reject())

.. testoutput::

   rate does not Granger-cause gdp
   4.665 1 0.0308 True
   False

The test is pairwise and per equation. Block exogeneity of several
variables, or restrictions across equations, go through
``coefficient_covariance.wald`` or ``wald_restriction``.

Comparing specifications
------------------------

``compare(*others, criterion="aic")`` ranks results by ``"aic"``,
``"bic"`` or ``"hqic"`` and returns a table, best first;
``likelihood_ratio_test(unrestricted)`` treats the calling result as the
restricted model. Both require the same ``nobs``. A VAR(1) and a VAR(2)
on the same array do *not* share a sample (299 and 298 rows), so drop
the extra leading row from the shorter model first.

.. testcode::

   var1 = VAR(y[1:], order=1).fit()
   var2 = VAR(y, order=2).fit()
   print(var1.nobs, var2.nobs)
   print(var1.compare(var2, criterion="bic"))
   lr = var1.likelihood_ratio_test(var2)
   print(round(lr.statistic, 3), lr.df, round(lr.pvalue, 3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   298 298
                               Model comparison by BIC
   ================================================================================
   Observations:                      298  Models:                              2
   --------------------------------------------------------------------------------
   model            llf    k        BIC     dBIC
   VAR(1)     -1261.124   18   2624.795    0.000
   VAR(2)     -1256.007   27   2665.835   41.040
   ================================================================================
   Ranked best-first; the final column is the gap to the best model.
   10.234 9 0.332

For choosing ``p`` the model-level ``lag_order_selection`` is the better
tool; ``compare`` is for comparing different trends, variable sets or
families on one sample.

Exogenous regressors: ``VARX``
------------------------------

.. math::

   y_t = D d_t + \sum_{i=1}^{p} A_i y_{t-i} + \sum_{j=0}^{s} B_j x_{t-j} + u_t.

The distributed lag of :math:`x` is appended to the design after the
endogenous lags. The result adds ``exog``, ``exog_names``,
``exog_order``, ``k_exog`` and ``exog_coefficients``, a ``(s + 1, k,
m)`` stack with ``exog_coefficients[j][r, q]`` the effect of
:math:`x_{q, t-j}` on equation ``r`` (``j = 0`` is the impact). In
``params`` the contemporaneous term is keyed by the bare exogenous name
and lag ``j`` by ``"{name}.L{j}"``. The effective sample is
:math:`n - \max(p, s)`.

.. testcode::

   from cultivars.multivariate.reduced_form.vector_autoregression import VARX

   rng = np.random.default_rng(3)
   oil = rng.standard_normal((250, 1))
   z = np.zeros((250, 2))
   for t in range(1, 250):
       z[t] = (np.array([[0.5, 0.1], [0.2, 0.3]]) @ z[t - 1]
               + np.array([0.8, 0.0]) * oil[t, 0]
               + np.array([0.0, 0.4]) * oil[t - 1, 0]
               + 0.5 * rng.standard_normal(2))

   resx = VARX(z, oil, order=1, exog_order=1,
               names=("output", "prices"), exog_names=("oil",)).fit()
   print(resx.exog_coefficients.shape, resx.nobs, resx.design.shape)
   print(list(resx.equation("prices")))
   print(resx.exog_coefficients[:, :, 0].round(2))
   print(resx.forecast(2, exog_future=np.zeros((2, 1))).round(3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   (2, 2, 1) 249 (249, 5)
   ['const', 'output.L1', 'prices.L1', 'oil', 'oil.L1']
   [[0.81 0.01]
    [0.07 0.32]]
   [[0.439 0.263]
    [0.253 0.251]]

``forecast`` is the only method whose signature changes:
``forecast(steps, *, exog_future)`` requires a ``(steps, m)`` path for
:math:`x`, uses the last ``s`` sample rows of ``exog`` for the lags that
reach back into the sample, and raises ``SpecificationError`` without
the path. Stability, ``irf``, ``fevd``, ``historical_decomposition`` and
``granger_causality`` read the endogenous block only: :math:`x` is
conditioned on, never shocked, so it does not enter the moving-average
representation, and naming an exogenous variable in
``granger_causality`` raises. ``lag_order_selection`` varies ``p`` with
the exogenous block held fixed.

``to_varx()`` repackages the unit in levels as a
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.ConditionalLevels`
record (``phi`` the endogenous stack, ``impact`` :math:`= B_0`,
``exog_lags`` :math:`= B_1, \dots, B_s`, ``deterministic``, ``names``,
``exog_names``), the currency
:class:`~cultivars.multivariate.reduced_form.closed_global.GVAR` links.

.. testcode::

   levels = resx.to_varx()
   print(levels.phi.shape, levels.impact.shape, levels.exog_lags.shape, levels.exog_names)

.. testoutput::

   (1, 2, 2) (2, 1) (1, 2, 1) ('oil',)

How the pieces connect
----------------------

* **Structural identification.** A fitted ``VARResult`` (and
  ``VARXResult``) satisfies the ``ClosedSystemResult`` protocol in
  ``cultivars.engine._core._protocols`` -- ``names``, ``nobs``,
  ``sigma_u``, ``resid``, ``coefficients``, ``k_endog``,
  ``ma_representation`` -- which is everything an identification scheme
  reads. Schemes take the result in their constructor and identify with
  a no-argument ``identify()``:

  .. testcode::

     from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR

     svar = RecursiveSVAR(res, order=("rate", "gdp", "infl")).identify()
     print(svar.impact.round(3))
     print(np.allclose(RecursiveSVAR(res).identify().irf(5), res.irf(5)))

  .. testoutput::
     :options: +NORMALIZE_WHITESPACE

     [[0.202 1.023 0.   ]
      [0.421 0.544 0.922]
      [1.057 0.    0.   ]]
     True

  The declared recursive scheme in the variables' own order reproduces
  ``irf()``; the full menu (long-run, sign, external instruments,
  heteroskedasticity, non-Gaussianity) is in
  :doc:`structural_identification`.
* **Bayesian estimation.** ``prior=`` gives a shrunk point estimate;
  the posterior samplers that take the same priors are in
  :doc:`bayesian_vars` and :doc:`large_dimensional`.
* **Extensions.** VECM, VARMA, panel and MIDAS results present the same
  surface (same ``params`` keys pattern, same array conventions); see
  :doc:`var_extensions`.
* **Global systems.** ``VARXResult.to_varx()`` is what
  :class:`~cultivars.multivariate.reduced_form.closed_global.GVAR`
  consumes.
* **Mixed frequency.** ``MFVAR.from_result`` takes ``coefficients``,
  ``sigma_u`` and ``names`` from a fitted VAR to filter a
  mixed-frequency panel (:doc:`var_extensions`).
* **Diagnostics and forecasting.** Unit-root and cointegration tests to
  run before choosing levels, differences or a VECM are in
  :doc:`diagnostics`; forecast evaluation is in :doc:`forecasting`.

Nuances and gotchas
-------------------

* **The identify method does not work with the shipped schemes.**
  ``VARResult.identify(strategy)`` calls ``strategy.identify(self)``,
  but every scheme in :mod:`cultivars.multivariate.structural` takes the result
  in its constructor and defines ``identify()`` with no argument.
  Passing an instance raises ``TypeError``; passing the class raises
  ``AttributeError``. Use ``Scheme(result, ...).identify()``.

  .. testcode::

     try:
         res.identify(RecursiveSVAR(res))
     except TypeError as error:
         print(error)

  .. testoutput::

     RecursiveSVAR.identify() takes 1 positional argument but 2 were given

* **DataFrame column names are ignored.** ``endog`` is converted to an
  array; ``names`` defaults to ``y1 ... yk`` regardless. Pass
  ``names=tuple(df.columns)`` yourself.

  .. testcode::

     import pandas as pd

     frame = pd.DataFrame(y, columns=["gdp", "infl", "rate"])
     print(VAR(frame, order=1).fit().names)

  .. testoutput::

     ('y1', 'y2', 'y3')

* **No DataFrame export and no AIC attribute.** There is no
  ``to_pandas``, ``to_dict`` or ``aic`` on the result; it returns
  NumPy arrays and dicts; criteria are under
  ``information_criteria.aic`` (and ``.bic``, ``.hqic``).
* **Two information-criterion scales.** ``lag_order_selection`` reports
  Lütkepohl's per-observation criteria (:math:`\log\det\hat\Sigma +
  \text{penalty}/T`, covariance parameters excluded);
  ``information_criteria`` on a result is on the :math:`-2\ell +
  \text{penalty}` scale with covariance parameters included. The same
  model scores differently in the two places; compare within one.
* **The effective sample shrinks with the order.** ``nobs = n - p``
  (VARX: :math:`n - \max(p, s)`), and ``resid[t]`` belongs to ``endog[t + p]``.
  ``compare`` and ``likelihood_ratio_test`` raise ``DimensionError``
  across different ``nobs``.
* **Variable order is an identifying assumption.** The order of
  ``names`` drives ``irf`` (default ``orthogonalized=True``), ``fevd``,
  ``structural_shocks`` and ``historical_decomposition``. Forecasts,
  Granger tests, ``ma_representation`` and ``generalized_irf`` are
  invariant to it.
* **Normal, not t, reference distribution.** ``pvalues`` and
  ``conf_int`` use the standard normal.
* **Under a prior, p-values are still returned.** ``pvalues`` gives
  normal tail areas of posterior mean over posterior sd even though the
  summary drops that column as having no frequentist meaning. Do not
  report them as p-values.
* **Likelihood-ratio test against a shrunk model.**
  ``shrunk.likelihood_ratio_test(ols_fit)`` runs with the fractional
  parameter difference truncated to an integer (18 - 13.81 gives
  ``df=4``) instead of refusing; a chi-squared LR test is not defined
  for that pair.

  .. testcode::

     print(shrunk.likelihood_ratio_test(res).df)

  .. testoutput::

     4

* **Point forecasts only.** ``forecast`` returns points, and there is
  no ``forecast_interval``. ``simulate`` gives paths from the point
  estimates (no parameter uncertainty).
* **VARX simulation ignores the exogenous block.**
  ``VARXResult.simulate`` is the inherited VAR simulator, so it draws the endogenous system with
  :math:`x \equiv 0`.
* **Granger tests need lags and endogenous names.** ``order=0`` raises
  ``SpecificationError("a VAR(0) has no lags to test.")``; an exogenous
  name in a VARX raises too.
* **The portmanteau degrees of freedom** subtract :math:`k` times the
  full regressor count, deterministic columns included (see
  `Residual diagnostics`_).
* **No shrinkage for VARX.** ``VARX`` takes no ``prior``; passing one
  raises ``TypeError``.

See also
--------

* :doc:`var_extensions` -- VECM, VARMA, panel, global, mixed-frequency,
  functional and yield-curve systems on the same surface.
* :doc:`structural_identification` -- declaring the shocks.
* :doc:`bayesian_vars` and :doc:`large_dimensional` -- posterior
  simulation and high-dimensional systems.
* :doc:`diagnostics`, :doc:`forecasting`, :doc:`conventions`,
  :doc:`nuances`.
* :mod:`cultivars.multivariate.reduced_form.vector_autoregression`,
  :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`,
  :class:`~cultivars.bayes.priors.MinnesotaPrior`.

References: Sims (1980); Granger (1969); Litterman (1986); Hamilton
(1994, ch. 11); Pesaran and Shin (1998); Lütkepohl (2005, chs. 2-4 and
10); Diebold and Yilmaz (2012).
