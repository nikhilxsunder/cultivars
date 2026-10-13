.. meta::
   :description: GARCH, GJR, EGARCH, FIGARCH, ARMA-GARCH and stochastic-volatility models in cultivars: estimation, persistence, simulation and variance forecasts.

Volatility models
=================

This page covers models for the time-varying variance of a single
series. The observation-driven family in
:mod:`cultivars.univariate.conditional_variance` (GARCH, GJR, EGARCH,
FIGARCH, with constant or ARMA means) makes the variance a function of
past shocks; the latent family in
:mod:`cultivars.univariate.stochastic_volatility` (SV, UCSV) makes it a
state with its own innovation. They answer "how volatile is the series
now, how persistent are variance shocks, and is there a leverage
effect?". Constant-variance mean models are on
:doc:`univariate_linear`; volatility in a system of series (BVAR with
stochastic volatility, factor SV, identification through
heteroskedasticity) is on :doc:`bayesian_vars`,
:doc:`large_dimensional` and :doc:`structural_identification`.

.. contents:: On this page
   :local:
   :depth: 1

The family at a glance
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 24 34 22

   * - Class
     - Module
     - Model in one line
     - Use when
   * - :class:`~cultivars.univariate.conditional_variance.GARCH`
     - ``conditional_variance``
     - :math:`\sigma_t^2 = \omega + \sum\alpha_i\varepsilon_{t-i}^2 +
       \sum\beta_j\sigma_{t-j}^2`
     - symmetric volatility clustering; ARCH(p) with ``q=0``
   * - :class:`~cultivars.univariate.conditional_variance.GJR`
     - ``conditional_variance``
     - GARCH plus :math:`\gamma\,\varepsilon_{t-1}^2\,1\{\varepsilon_{t-1}<0\}`
     - leverage in a level recursion
   * - :class:`~cultivars.univariate.conditional_variance.EGARCH`
     - ``conditional_variance``
     - recursion in :math:`\log\sigma_t^2` driven by :math:`|z_t|` and
       :math:`z_t`
     - leverage without positivity constraints
   * - :class:`~cultivars.univariate.conditional_variance.FIGARCH`
     - ``conditional_variance``
     - FIGARCH(1, d, 1), hyperbolic ARCH(:math:`\infty`) weights
     - long memory in volatility
   * - :class:`~cultivars.univariate.conditional_variance.ARGARCH`,
       :class:`~cultivars.univariate.conditional_variance.ARMAGARCH`,
       :class:`~cultivars.univariate.conditional_variance.ARMAGJR`,
       :class:`~cultivars.univariate.conditional_variance.ARMAEGARCH`,
       :class:`~cultivars.univariate.conditional_variance.ARMAFIGARCH`
     - ``conditional_variance``
     - the same variances under an AR or ARMA mean, estimated jointly
     - the mean dynamics matter as much as the variance
   * - :class:`~cultivars.univariate.stochastic_volatility.SV`
     - ``stochastic_volatility``
     - :math:`y_t = c + e^{h_t/2}\varepsilon_t`, AR(1) log variance
     - latent volatility; Student-t tails or leverage; Bayesian posterior
   * - :class:`~cultivars.univariate.stochastic_volatility.UCSV`
     - ``stochastic_volatility``
     - random-walk trend plus irregular, both with random-walk log
       variances
     - trend inflation (Stock and Watson, 2007)
   * - :class:`~cultivars.multivariate.structural.stochastic_volatility.LongMemorySV`
     - ``multivariate.structural.stochastic_volatility``
     - fractionally integrated log variance, Whittle estimation
     - long memory in a latent volatility

GARCH-type models
-----------------

Every model in :mod:`~cultivars.univariate.conditional_variance` writes

.. math::

   y_t = \mu_t + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t,
   \qquad z_t \sim N(0, 1),

with :math:`\mu_t` zero, a constant, or an AR/ARMA recursion, and
:math:`\sigma_t^2` one of the recursions of Engle (1982), Bollerslev
(1986), Glosten, Jagannathan and Runkle (1993), Nelson (1991) or Baillie,
Bollerslev and Mikkelsen (1996). Mean and variance are estimated
*jointly* by maximizing the Gaussian likelihood; the mean uses conditional
sum of squares, and the variance recursion starts from an exponentially
weighted backcast. ``fit()`` takes no arguments.

.. testcode::

   import numpy as np
   from cultivars.univariate.conditional_variance import GARCH

   rng = np.random.default_rng(1)
   n = 2000
   h, r = np.ones(n), np.zeros(n)
   for t in range(1, n):
       h[t] = 0.05 + 0.08 * (r[t - 1] - 0.05) ** 2 + 0.9 * h[t - 1]
       r[t] = 0.05 + np.sqrt(h[t]) * rng.standard_normal()

   garch = GARCH(r).fit()
   print(garch.vol, garch.order, garch.nobs, garch.n_params)
   print({k: round(v, 3) for k, v in garch.params.items()})
   print(round(garch.persistence, 3), round(garch.half_life, 1),
         round(garch.unconditional_variance, 3))

.. testoutput::

   GARCH (1, 0, 1) 2000 4
   {'const': 0.048, 'omega': 0.041, 'alpha[1]': 0.087, 'beta[1]': 0.899}
   0.986 50.1 3.016

Specification choices (constant-mean fronts):

.. list-table::
   :header-rows: 1
   :widths: 16 54 30

   * - Argument
     - Meaning
     - Default
   * - ``p``
     - lags of :math:`\varepsilon_t^2` (EGARCH: of :math:`|z_t| - E|z|`)
     - 1
   * - ``o``
     - asymmetry lags; GJR and EGARCH only, must be at least 1
     - 1
   * - ``q``
     - lags of :math:`\sigma_t^2` (EGARCH: of :math:`\log\sigma_t^2`)
     - 1
   * - ``ar_lags``
     - AR lags in the mean (GARCH, GJR, EGARCH accept it directly)
     - 0
   * - ``mean``
     - ``"constant"`` or ``"zero"``
     - ``"constant"``
   * - ``truncation``
     - FIGARCH only: ARCH(:math:`\infty`) lags retained
     - 1000

What the result
(:class:`~cultivars.univariate.conditional_variance.GARCHResult`) gives
you:

- ``params``: ``const``, ``ar.L1`` .., ``omega``, ``alpha[1]`` ..,
  ``gamma[1]`` .., ``beta[1]`` .. (empty blocks omitted). Note the
  bracket naming, unlike ``ar.L1``.
- Fields ``vol`` (``"GARCH"``, ``"GJR"`` or ``"EGARCH"``), ``mean``
  (``"constant"``/``"zero"``), ``const``, ``ar_params``, ``omega``,
  ``alpha``, ``gamma``, ``beta``; property ``order`` = ``(p, o, q)``.
- Paths ``conditional_variance``, ``conditional_volatility``
  (its square root), ``standardized_resid`` (``resid`` divided by the
  volatility), ``fittedvalues`` and ``resid`` of the mean.
- ``persistence``, ``is_covariance_stationary``, ``half_life``,
  ``unconditional_variance``, ``has_leverage``.
- ``llf``, ``nobs``, ``n_params``, ``information_criteria``,
  ``compare``, ``likelihood_ratio_test``, ``simulate``,
  ``to_pandas`` / ``to_polars`` / ``to_dict``.

.. testcode::

   print(list(garch.to_pandas().columns))
   eps, s2 = garch.resid, garch.conditional_variance
   one_step = garch.omega + garch.alpha[0] * eps[:-1] ** 2 + garch.beta[0] * s2[:-1]
   print(bool(np.allclose(s2[1:], one_step)))

.. testoutput::

   ['observed', 'fitted', 'resid', 'conditional_variance', 'conditional_volatility', 'standardized_resid']
   True

Asymmetry: ``GJR`` and ``EGARCH``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{GJR:}\quad
   \sigma_t^2 = \omega + \sum_i \alpha_i \varepsilon_{t-i}^2
   + \sum_k \gamma_k \varepsilon_{t-k}^2\,1\{\varepsilon_{t-k} < 0\}
   + \sum_j \beta_j \sigma_{t-j}^2,

   \text{EGARCH:}\quad
   \log\sigma_t^2 = \omega + \sum_i \alpha_i\bigl(|z_{t-i}| - \sqrt{2/\pi}\bigr)
   + \sum_k \gamma_k z_{t-k} + \sum_j \beta_j \log\sigma_{t-j}^2.

The two use **opposite sign conventions** for leverage: in GJR,
:math:`\gamma > 0` means bad news raises volatility more; in EGARCH the
same effect is :math:`\gamma < 0`. GJR's :math:`\gamma` is searched
unconstrained, so a negative estimate is reported as reverse asymmetry.

.. testcode::

   from cultivars.univariate.conditional_variance import EGARCH, GJR

   rng = np.random.default_rng(0)
   h, lev = np.ones(2500), np.zeros(2500)
   for t in range(1, 2500):
       h[t] = 0.02 + 0.05 * lev[t - 1] ** 2 + 0.85 * h[t - 1]
       h[t] += 0.10 * lev[t - 1] ** 2 * (lev[t - 1] < 0)
       lev[t] = np.sqrt(h[t]) * rng.standard_normal()

   fits = [GARCH(lev).fit(), GJR(lev).fit(), EGARCH(lev).fit()]
   print(round(fits[1].gamma[0], 3), round(fits[2].gamma[0], 3))
   table = fits[0].compare(*fits[1:], criterion="bic")
   print([row[0] for row in table.rows])
   print(fits[2].unconditional_variance, round(fits[2].persistence, 3))

.. testoutput::

   0.073 -0.057
   ['EGARCH(1, 1, 1)', 'GJR(1, 1, 1)', 'GARCH(1, 1)']
   None 0.966

For EGARCH, ``persistence`` is :math:`\sum_j \beta_j` (the AR root of the
log variance) and ``unconditional_variance`` is ``None``: the stationary
level is the mean of a lognormal and has no closed form in the reported
parameters. For the level families ``persistence`` is
:math:`\sum\alpha_i + \tfrac12\sum\gamma_k + \sum\beta_j`.

ARMA means: ``ARGARCH`` to ``ARMAFIGARCH``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \varepsilon_t
   + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j},
   \qquad \varepsilon_t = \sigma_t z_t.

The ``AR*``/``ARMA*`` fronts take ``ar_lags`` and ``ma_lags`` (both
default to 1 on the ``ARMA*`` fronts) and return
:class:`~cultivars.univariate.conditional_variance.ARMAGARCHResult` or
:class:`~cultivars.univariate.conditional_variance.ARMAFIGARCHResult`,
which add ``mean_order``, ``ma_params``, the mean-side ``stability`` /
``is_stationary`` and ``invertibility`` / ``is_invertible`` verdicts, and
``is_structurally_constrained``. ``const`` is the recursion intercept
:math:`c`, as in :class:`~cultivars.univariate.autoregression.ARResult`.

.. testcode::

   from cultivars.univariate.conditional_variance import ARGARCH, ARMAGARCH

   argarch = ARGARCH(r, ar_lags=2).fit()
   plain = GARCH(r, ar_lags=2).fit()
   print(argarch.nobs, argarch.mean_order, round(argarch.llf - plain.llf, 8))
   print(argarch.is_stationary, argarch.is_structurally_constrained)

   armagarch = ARMAGARCH(r, ar_lags=1, ma_lags=1).fit()
   print(list(armagarch.params))

.. testoutput::

   1998 (2, 0) 0.0
   True False
   ['const', 'ar.L1', 'ma.L1', 'omega', 'alpha[1]', 'beta[1]']

``ARGARCH(y, ar_lags=k)`` and ``GARCH(y, ar_lags=k)`` are the same
estimator; the first returns the richer record. With a moving-average
block the mean is searched inside the stationary-invertible region, so
``is_stationary`` then verifies the transform rather than describing the
data (``is_structurally_constrained`` is ``True``).

Long memory: ``FIGARCH``
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \sigma_t^2 = \frac{\omega}{1 - \beta}
   + \Bigl[1 - \frac{(1 - \phi L)(1 - L)^d}{1 - \beta L}\Bigr]
   \varepsilon_t^2.

The order is fixed at (1, d, 1); the parameters are ``omega``, ``phi``,
``d`` and ``beta``, each of ``phi``, ``d``, ``beta`` inside
:math:`(0, 1)`. The ARCH(:math:`\infty`) weights sum to one for every
:math:`d > 0`, so
:class:`~cultivars.univariate.conditional_variance.FIGARCHResult`
reports ``persistence = 1.0``, ``is_covariance_stationary = False`` and
``half_life = inf``, has no ``unconditional_variance``, and refuses
``simulate``.

.. testcode::

   from cultivars.univariate.conditional_variance import FIGARCH

   fig = FIGARCH(r[:1000], truncation=200).fit()
   print(list(fig.params), fig.truncation)
   print(fig.persistence, fig.is_covariance_stationary, fig.half_life)

.. testoutput::

   ['const', 'omega', 'phi', 'd', 'beta'] 200
   1.0 False inf

Innovation distributions
------------------------

The conditional-variance models are **Gaussian only**: there is no
``dist`` argument, and the reported ``llf`` is the Gaussian likelihood.
With fat-tailed data, read the estimates as Gaussian quasi-maximum
likelihood (consistent for the variance parameters under correct mean and
variance specification) and inspect ``standardized_resid`` for the tail
shape. ``simulate`` also draws Gaussian :math:`z_t`.

Heavy tails and leverage are available on the latent side:
``SV(y, dist="t")`` makes :math:`\varepsilon_t` Student-t with estimated
degrees of freedom, and ``SV(y, leverage=True)`` correlates the return and
log-variance innovations. The two cannot be combined.

Persistence and stationarity
----------------------------

Stationarity on the variance side is ``persistence < 1`` alone
(``is_covariance_stationary``), and ``half_life`` is
:math:`\log 0.5 / \log(\text{persistence})`. The constant-mean results
carry no mean-stationarity verdict; the ``ARMA*`` results report it
separately through ``is_stationary``.

The level families' estimator rejects any draw with persistence at or
above 0.999, so an integrated or explosive GARCH comes back just below
the bound and ``is_covariance_stationary`` is ``True`` for every GARCH
and GJR fit. Read a persistence above about 0.99 as "integrated", not as
a finite half-life to trust.

.. testcode::

   rng = np.random.default_rng(7)
   h, x = np.ones(1500), np.zeros(1500)
   for t in range(1, 1500):
       h[t] = 0.01 + 0.2 * x[t - 1] ** 2 + 0.82 * h[t - 1]   # true sum 1.02
       x[t] = np.sqrt(h[t]) * rng.standard_normal()
   explosive = GARCH(x, mean="zero").fit()
   print(round(explosive.persistence, 3), explosive.is_covariance_stationary)

.. testoutput::

   0.995 True

Stochastic volatility: ``SV``
-----------------------------

.. math::

   y_t = c + \exp(h_t / 2)\,\varepsilon_t,
   \qquad
   h_{t+1} = \mu + \phi\,(h_t - \mu) + \sigma\,\eta_t,
   \qquad \varepsilon_t, \eta_t \sim N(0, 1)

(Taylor, 1986). The likelihood is an integral over the latent path, and
:class:`~cultivars.univariate.stochastic_volatility.SV` offers four ways
to do it:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Call
     - Method
     - Returns
   * - ``fit()`` (``method="qml"``)
     - Harvey, Ruiz and Shephard (1994) quasi-ML: Kalman filter on
       :math:`\log(y_t - c)^2`
     - :class:`~cultivars.univariate.stochastic_volatility.SVResult`
   * - ``fit(method="particle", n_particles=500, seed=None)``
     - simulated ML on a bootstrap particle likelihood
     - ``SVResult``
   * - ``sample()`` (``method="gibbs"``)
     - Kim, Shephard and Chib (1998) mixture Gibbs sampler
     - :class:`~cultivars.univariate.stochastic_volatility.SVPosterior`
   * - ``sample(method="particle")``
     - particle marginal Metropolis-Hastings (Andrieu, Doucet and
       Holenstein, 2010)
     - ``SVPosterior``

.. testcode::

   from cultivars.univariate.stochastic_volatility import SV

   rng = np.random.default_rng(11)
   m = 300
   lv = np.empty(m)
   lv[0] = -1.0
   for t in range(1, m):
       lv[t] = -1.0 + 0.95 * (lv[t - 1] + 1.0) + np.sqrt(0.05) * rng.standard_normal()
   ret = np.exp(lv / 2) * rng.standard_normal(m)

   sv = SV(ret, mean="zero").fit()
   print(sv.method, sv.nobs, sv.n_params)
   print(round(sv.mu, 3), round(sv.phi, 3), round(sv.sigma2, 4))
   print(round(sv.unconditional_variance, 3), round(sv.half_life, 1))
   print(sv.forecast_variance(3).round(3))
   lower, upper = sv.volatility_bands(alpha=0.10)
   print(bool(np.all(lower < sv.volatility) and np.all(sv.volatility < upper)))

.. testoutput::

   qml 300 3.0
   -0.85 0.946 0.0093
   0.447 12.6
   [0.514 0.51  0.507]
   True

``SVResult`` carries ``mean``, ``mu``, ``phi``, ``sigma2``, ``nu``
(``None`` without ``dist="t"``), ``rho`` (``0.0`` without leverage),
``method``, the smoothed ``log_variance`` and ``log_variance_std``, and
``volatility`` (:math:`e^{\hat h_t/2}`). ``volatility_bands(alpha=0.05)``
exponentiates normal bands on the smoothed log variance.
``forecast_variance(steps)`` propagates the last smoothed log-variance
moments through the AR(1) with the log-normal correction and converges to
``unconditional_variance``. ``state_space`` is the fitted model as a
:class:`~cultivars.state_space.nonlinear.NonlinearSSM` (read by the
particle filter); ``quasi_state_space`` is the linearized
:class:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM`.
``to_pandas`` columns are ``observed``, ``log_variance``, ``volatility``,
``volatility_lower``, ``volatility_upper`` and ``standardized``.

The Gibbs posterior:

.. testcode::

   post = SV(ret, mean="zero").sample(n_draws=250, n_burn=50, seed=0)
   print(post.method, post.n_kept, post.phi_draws.shape)
   print(round(post.phi_draws.mean(), 3), post.volatility_quantiles().shape)

.. testoutput::

   gibbs 200 (200,)
   0.94 (3, 300)

``n_draws`` counts *all* iterations, burn-in included, so
``n_kept = (n_draws - n_burn) / thin``. ``SVPosterior`` exposes
``mu_draws``, ``phi_draws``, ``sigma2_draws``, ``mean_draws``, ``h_draws``
(``(S, n)`` under Gibbs), ``log_variance`` and ``volatility`` (posterior
means), ``volatility_quantiles(levels)``, ``convergence()``,
``posterior_replications()`` and ``simulate(n, draw=None)``; the prior is
:class:`~cultivars.bayes.priors.VolatilityPrior`, passed as
``sample(prior=...)``. Convergence and predictive checks are covered in
:doc:`bayesian_workflow`.

Trend with stochastic volatility: ``UCSV``
------------------------------------------

.. math::

   y_t = \tau_t + e^{h_t/2}\varepsilon_t,
   \qquad \tau_{t+1} = \tau_t + e^{q_t/2}\eta_t,

with :math:`h_t` and :math:`q_t` random walks of standard deviation
:math:`\gamma` (Stock and Watson, 2007). ``UCSV(y, gamma=0.2)`` holds
:math:`\gamma` at the Stock-Watson value; ``gamma=None`` estimates both
vol-of-vols under :class:`~cultivars.bayes.priors.RandomWalkVolatilityPrior`.
``fit()`` runs the Gibbs sampler and returns a
:class:`~cultivars.univariate.stochastic_volatility.UCSVPosterior`.

.. testcode::

   from cultivars.univariate.stochastic_volatility import UCSV

   rng = np.random.default_rng(0)
   infl = np.cumsum(0.2 * rng.standard_normal(150)) + 0.7 * rng.standard_normal(150)
   ucsv = UCSV(infl).fit(n_draws=200, n_burn=50, seed=0)
   print(ucsv.n_kept, ucsv.gamma_fixed, ucsv.trend.shape)
   paths = ucsv.forecast(4, n_paths=500, seed=0)
   print(paths.shape)
   print(list(ucsv.to_pandas().columns))

.. testoutput::

   150 True (150,)
   (500, 4)
   ['observed', 'trend', 'trend_q05', 'trend_q95', 'gap', 'irregular_volatility', 'trend_volatility']

``UCSVPosterior`` carries ``trend``, ``trend_quantiles(levels)``,
``gap``, ``irregular_volatility``, ``trend_volatility``, the draw arrays
``trend_draws``, ``h_draws``, ``q_draws``, ``gamma2_irregular_draws``,
``gamma2_trend_draws``, and ``forecast(steps, n_paths=2000, seed=None)``,
which returns predictive *draws of the series*, shape
``(n_paths, steps)``.

Variance forecasts and VaR-style use
------------------------------------

What exists:

- ``SVResult.forecast_variance(steps)``: expected variance
  :math:`E[e^{h_{T+k}} \mid y]` for :math:`k = 1, \ldots, steps`.
- ``UCSVPosterior.forecast(steps, ...)``: predictive draws of the series.
- **Nothing on the GARCH-type results.** ``GARCHResult``,
  ``FIGARCHResult`` and the ARMA variants have no ``forecast`` or
  ``forecast_variance`` method, and ``SVPosterior`` has none either.

For a GARCH(1, 1) the forecasts follow from the fitted recursion: the
one-step variance is
:math:`\sigma^2_{T+1} = \omega + \alpha\varepsilon_T^2 + \beta\sigma_T^2`,
and for :math:`h > 1`,
:math:`E[\sigma^2_{T+h}] = \bar\sigma^2 + \pi^{h-1}(\sigma^2_{T+1} -
\bar\sigma^2)` with :math:`\pi` the persistence and :math:`\bar\sigma^2`
the unconditional variance.

.. testcode::

   s2_next = (garch.omega + garch.alpha[0] * garch.resid[-1] ** 2
              + garch.beta[0] * garch.conditional_variance[-1])
   ubar, pi = garch.unconditional_variance, garch.persistence
   path = ubar + pi ** np.arange(5) * (s2_next - ubar)
   print(round(s2_next, 3), path.round(3))

.. testoutput::

   1.005 [1.005 1.033 1.06  1.087 1.114]

A one-step value-at-risk is a quantile of the predictive distribution,
:math:`\hat c + \sigma_{T+1} q_\alpha`. Under the Gaussian likelihood
:math:`q_\alpha` is the normal quantile; the empirical quantile of
``standardized_resid`` (filtered historical simulation) is the usual
remedy for fat tails. Quantile forecasts are scored with
:func:`~cultivars.forecast.scoring.pinball_loss`; here the in-sample 1%
quantiles of the GARCH fit beat a constant-variance quantile:

.. testcode::

   from scipy.stats import norm
   from cultivars.forecast.scoring import pinball_loss

   z01 = norm.ppf(0.01)
   var_gauss = garch.const + np.sqrt(s2_next) * z01
   var_fhs = garch.const + np.sqrt(s2_next) * np.quantile(garch.standardized_resid, 0.01)
   print(round(var_gauss, 3), round(var_fhs, 3))

   q_garch = garch.fittedvalues + garch.conditional_volatility * z01
   q_const = np.full(r.size, r.mean() + r.std() * z01)
   print(round(pinball_loss(r, q_garch, 0.01).mean(), 4),
         round(pinball_loss(r, q_const, 0.01).mean(), 4))

.. testoutput::

   -2.285 -2.36
   0.0427 0.0509

Out-of-sample evaluation over rolling origins is
:class:`~cultivars.forecast.backtest.Backtest`; because the GARCH results
have no ``forecast``, pass a ``predict=`` hook built from the recursion
above. See :doc:`forecasting`.

Simulating from a fit
---------------------

``GARCHResult.simulate(n=200, *, seed=None, burn=...)`` runs the family's
own recursion from the unconditional variance with Gaussian shocks and
applies the mean on top; ``ARMAGARCHResult`` drives the full ARMA mean.
``SVResult.simulate`` and ``SVPosterior.simulate`` start the log variance
from its stationary law. ``FIGARCHResult.simulate`` (and the ARMA
variant) always raises
:class:`~cultivars.exceptions.SpecificationError`, because no finite
burn-in makes a long-memory variance forget its start.

.. testcode::

   from cultivars.exceptions import SpecificationError

   sim = garch.simulate(20000, seed=0)
   print(sim.shape, round(sim.var(), 2))
   print(sv.simulate(5, seed=0).round(3))
   try:
       fig.simulate(10)
   except SpecificationError as err:
       print(str(err)[:40])

.. testoutput::

   (20000,) 2.62
   [ 0.075 -0.081  0.417  0.072 -0.352]
   FIGARCH results do not simulate: the lon

How the pieces connect
----------------------

- **Mean first, jointly.** A GARCH error on an ARMA mean is fitted in one
  likelihood by the ``ARMA*`` fronts; there is no two-step "fit ARIMA,
  then GARCH on its residuals" API, and ``ARMAGARCHResult`` is the joint
  analogue of :class:`~cultivars.univariate.box_jenkins.ARMAResult`
  (:doc:`univariate_linear`).
- **Long memory.** FIGARCH is the variance counterpart of
  :class:`~cultivars.univariate.fractional_integration.ARFIMA`; the latent
  counterpart is
  :class:`~cultivars.multivariate.structural.stochastic_volatility.LongMemorySV`,
  estimated by Whittle in the frequency domain.
- **Decomposition.** ``UCSV`` relaxes the constant variances of
  :class:`~cultivars.univariate.unobserved_components.UnobservedComponents`
  with ``trend="level"``.
- **State space.** ``SVResult.state_space`` is a
  :class:`~cultivars.state_space.nonlinear.NonlinearSSM` whose
  ``particle_filter`` estimates the exact likelihood or reads new data
  with the fitted law (:doc:`state_space`).
- **Bayesian tools.** ``SVPosterior`` and ``UCSVPosterior`` provide
  ``convergence()``; ``SVPosterior`` also provides
  ``posterior_replications`` for predictive checks
  (:doc:`bayesian_workflow`).
- **Multivariate volatility.** Stochastic volatility in VAR innovations
  (:class:`~cultivars.multivariate.large_dim.volatility.BVARSV`), factor
  stochastic volatility
  (:class:`~cultivars.multivariate.large_dim.factor_volatility.FactorSV`)
  and time-varying-parameter VARs with SV are on :doc:`bayesian_vars`,
  :doc:`large_dimensional` and :doc:`nonlinear_multivariate`;
  identification through heteroskedasticity
  (:class:`~cultivars.multivariate.structural.stochastic_volatility.StochasticVolatilitySVAR`)
  is on :doc:`structural_identification`.
- **Forecast evaluation.** Quantile and density forecasts built from
  these results are scored by :mod:`cultivars.forecast.scoring` and
  :mod:`cultivars.forecast.calibration` (:doc:`forecasting`).

Nuances and gotchas
-------------------

- **No variance forecast on GARCH-type results.** Compute it from
  ``omega``, ``alpha``, ``gamma``, ``beta``, the last ``resid`` and the
  last ``conditional_variance`` as shown above. ``SVResult`` is the only
  record with ``forecast_variance``.
- **No standard errors.** Every summary says "Standard errors are not yet
  available for this estimator."; use ``likelihood_ratio_test`` between
  nested fits (for example GARCH inside GJR) for inference.
- **Gaussian innovations only in the GARCH family.** No Student-t or
  skewed-t option; the likelihood is Gaussian QML.
- **Persistence is capped below 0.999** for GARCH and GJR (and
  :math:`|\sum\beta_j| < 0.999` for EGARCH), so ``is_covariance_stationary``
  cannot be ``False`` for those fits even on explosive data. The
  :class:`~cultivars.univariate.conditional_variance.GARCH` docstring
  states otherwise; trust the behaviour shown above.
- **EGARCH and GJR sign conventions differ.** Leverage is
  ``gamma > 0`` in GJR and ``gamma < 0`` in EGARCH. GJR persistence counts
  ``gamma`` at half weight.
- **Effective sample.** ``nobs`` is :math:`n - \text{ar\_lags}` for any
  front with an AR mean (``GARCH(y, ar_lags=2)`` has ``nobs = n - 2``)
  and :math:`n` otherwise; ``conditional_variance``, ``resid`` and
  ``fittedvalues`` are aligned with the last ``nobs`` observations, and
  ``compare`` refuses to rank fits with different ``nobs``.

  .. testcode::

     from cultivars.exceptions import DimensionError
     try:
         garch.compare(plain)
     except DimensionError as err:
         print(err)

  .. testoutput::

     information criteria are only comparable across a common sample; got differing observation counts [1998, 2000].

- **The first conditional variance is a backcast**, not
  ``unconditional_variance``; ``conditional_variance[0]`` comes from an
  exponentially weighted mean square of the residuals.
- **FIGARCH.** The order is fixed at (1, d, 1);
  ``has_long_memory`` is ``True`` for nearly every fit because ``d`` cannot
  reach zero; ``half_life`` is ``inf`` and ``simulate`` raises. The
  ARMA mean is ``ARMAFIGARCH``; ``FIGARCH`` itself accepts no ``ar_lags``.
- **SV: the QML likelihood is not the SV likelihood.** Under
  ``method="qml"`` ``llf`` is the Gaussian likelihood of the linearized
  model; information criteria compare only within one method (the
  comparison label includes the method). QML cannot identify leverage
  (``SV(y, leverage=True).fit()`` returns ``rho = 0.0``) and identifies
  ``nu`` weakly (on Gaussian data it drifted to about 500); use
  ``fit(method="particle")`` or ``sample()`` for either.
- **SV sizes and cost.** ``SV`` needs at least 50 observations and
  ``UCSV`` at least 40. QML runs a generic Kalman filter inside a
  numerical-gradient optimizer and takes several seconds on a few hundred
  observations; the particle methods are much slower.
- **Sampler bookkeeping.** ``n_draws`` includes ``n_burn``;
  ``sample(n_draws=100, n_burn=200)`` raises. ``UCSV.fit`` is a sampler
  (its signature is ``fit(n_draws=3000, n_burn=1000, thin=1, prior=None,
  seed=None)``), and its result has no ``llf`` or information criteria.
- ``LongMemorySV`` **lives in a multivariate module.** Import it from
  :mod:`cultivars.multivariate.structural.stochastic_volatility`, even
  though it is a single-series model.

See also
--------

- :doc:`univariate_linear` -- constant-variance mean models (AR, ARIMA,
  ARFIMA, unobserved components).
- :doc:`bayesian_vars` and :doc:`large_dimensional` -- multivariate
  stochastic volatility.
- :doc:`structural_identification` -- identification through
  heteroskedasticity.
- :doc:`forecasting` -- quantile and density scoring, backtests.
- :doc:`bayesian_workflow` -- convergence diagnostics and predictive
  checks for the SV posteriors.
- :doc:`state_space` -- the nonlinear and linear substrates behind SV.
- API: :mod:`cultivars.univariate.conditional_variance`,
  :mod:`cultivars.univariate.stochastic_volatility`.
