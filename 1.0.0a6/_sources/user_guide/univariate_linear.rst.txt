.. meta::
   :description: AR, ARMA, ARIMA, seasonal ARIMA, ARFIMA and unobserved-components models in cultivars: specification, estimation, results and quirks.

Univariate linear models
========================

This page covers the linear Gaussian models for the conditional mean of a
single series: autoregressions, the Box-Jenkins ARMA family with
differencing, seasonal blocks and regressors, fractionally integrated
ARMA, and Harvey's structural (unobserved-components) models. They answer
"what is the persistence, memory and deterministic structure of this
series, and which specification does the likelihood prefer?". They are
not for a time-varying variance (see :doc:`volatility`), regime or
threshold dynamics (see :doc:`univariate_nonlinear`), or several series
at once (see :doc:`var`).

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
     - Model in one line
     - Use when
   * - :class:`~cultivars.univariate.autoregression.AR`
     - :mod:`cultivars.univariate.autoregression`
     - :math:`y_t = c + \delta t + \sum_i \phi_i y_{t-i} + \varepsilon_t`,
       by OLS (CSS) or exact ML
     - a pure autoregression, a fast OLS fit, or a linear trend
   * - :class:`~cultivars.univariate.box_jenkins.ARMA`,
       :class:`~cultivars.univariate.box_jenkins.ARIMA`,
       :class:`~cultivars.univariate.box_jenkins.SARIMA`,
       :class:`~cultivars.univariate.box_jenkins.ARIMAX`,
       :class:`~cultivars.univariate.box_jenkins.SARIMAX`
     - :mod:`cultivars.univariate.box_jenkins`
     - multiplicative seasonal ARIMA, optionally a regression with
       ARIMA errors, by exact ML through the Kalman filter
     - moving-average terms, integer differencing, seasonality,
       exogenous regressors
   * - :class:`~cultivars.univariate.fractional_integration.ARFIMA`
     - :mod:`cultivars.univariate.fractional_integration`
     - :math:`\phi(L)(1-L)^d(y_t-\mu) = \theta(L)\varepsilon_t`,
       :math:`|d| < 0.5`
     - hyperbolically decaying autocorrelations (long memory)
   * - :class:`~cultivars.univariate.unobserved_components.UnobservedComponents`
     - :mod:`cultivars.univariate.unobserved_components`
     - level/trend + damped cycle + trigonometric seasonal + irregular,
       by exact ML and the Kalman smoother
     - a decomposition with bands, model-based detrending, and the only
       built-in forecast in this family

Every model follows the package's three-object discipline (see
:doc:`architecture`): the constructor validates and freezes a
specification, ``fit()`` estimates it, and the returned result is an
immutable record. Printing a result prints its summary table.

Autoregressions: ``AR``
-----------------------

.. math::

   y_t = c + \delta t + \sum_{i=1}^{p} \phi_i\, y_{t-i} + \varepsilon_t,
   \qquad \varepsilon_t \sim N(0, \sigma^2).

Two estimators sit behind ``method``. ``"css"`` (the default) conditions
on the first :math:`p` observations and solves the lagged regression by
OLS, so the likelihood is the conditional one on :math:`n - p`
observations. ``"exact"`` maximizes the exact Gaussian likelihood through
the companion state-space form, keeping the search inside the stationary
region with the partial-autocorrelation map of Monahan (1984); its
effective sample is all :math:`n` observations (Hamilton, 1994, ch. 5).

.. testcode::

   import numpy as np
   from cultivars.univariate.autoregression import AR

   rng = np.random.default_rng(0)
   e = rng.standard_normal(600)
   z = np.zeros(600)
   for t in range(2, 600):
       z[t] = 0.5 + 0.6 * z[t - 1] - 0.2 * z[t - 2] + e[t]
   y = z[100:]                      # drop the start-up transient; n = 500

   css = AR(y, order=2).fit()
   exact = AR(y, order=2, method="exact").fit()
   print(css.nobs, exact.nobs)
   print({k: round(v, 3) for k, v in css.params.items()})
   print(css.is_stationary, round(css.stability.max_modulus, 3))

.. testoutput::

   498 500
   {'const': 0.49, 'ar.L1': 0.556, 'ar.L2': -0.205, 'sigma2': 1.003}
   True 0.453

Specification choices:

``order``
   Non-negative integer :math:`p`. ``order=0`` fits the deterministic
   terms alone.
``trend``
   ``"n"`` (none), ``"c"`` (constant, default) or ``"ct"`` (constant and
   linear trend). ``"ct"`` is available under CSS only; combining it
   with ``method="exact"`` raises
   :class:`~cultivars.exceptions.SpecificationError` at construction.
``method``
   ``"css"`` (OLS, :math:`n - p` observations) or ``"exact"`` (exact ML,
   :math:`n` observations).

What the result
(:class:`~cultivars.univariate.autoregression.ARResult`) gives you:

- ``params``: a dict in table order. Names are ``const``, ``trend``,
  ``ar.L1`` .. ``ar.Lp`` and ``sigma2``; absent terms are omitted.
- The same numbers as fields: ``const`` (``None`` when ``trend="n"``),
  ``trend_coeff``, ``ar_params`` (an array) and ``sigma2``.
- ``llf``, ``nobs``, ``n_params`` (counts ``sigma2``),
  ``information_criteria``, ``compare``, ``likelihood_ratio_test``.
- ``stability`` (companion eigenvalues, ``max_modulus``,
  ``n_unit_roots``, ``n_explosive``) and ``is_stationary``.
- ``fittedvalues`` and ``resid`` on the effective sample, ``endog`` in
  full, ``to_pandas`` / ``to_polars`` / ``to_dict``.
- ``simulate(n, seed=..., burn=...)``.

``const`` is the intercept *inside* the recursion. The implied mean is
:math:`c / (1 - \sum_i \phi_i)`:

.. testcode::

   print(round(css.const / (1 - css.ar_params.sum()), 3), round(y.mean(), 3))

.. testoutput::

   0.756 0.759

Box-Jenkins models: ``ARMA`` to ``SARIMAX``
-------------------------------------------

.. math::

   \phi(L)\,\Phi(L^s)\,\bigl(w_t - x_t'\beta\bigr)
   = \theta(L)\,\Theta(L^s)\,\varepsilon_t,
   \qquad w_t = (1 - L)^d (1 - L^s)^D y_t,

with :math:`\phi(L) = 1 - \phi_1 L - \cdots`, the moving average in the
*plus* convention :math:`\theta(L) = 1 + \theta_1 L + \cdots`, and
:math:`x_t` the deterministic terms followed by any exogenous
regressors (Box, Jenkins, Reinsel and Ljung, 2015). Differencing is
applied before estimation; the ARMA part is cast in Harvey's state-space
form and the exact likelihood is evaluated by the Kalman filter (Harvey,
1989; Durbin and Koopman, 2012). Stationarity and invertibility are
imposed through the partial-autocorrelation reparameterization.

:class:`~cultivars.univariate.box_jenkins.SARIMAX` is the one real
specification; the other four classes are narrowing constructors that
fix or forbid arguments. All five return
:class:`~cultivars.univariate.box_jenkins.ARMAResult`.

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Class
     - Signature
     - Restriction
   * - ``ARMA``
     - ``ARMA(endog, *, order=(p, q), trend="c")``
     - ``d = 0``, no seasonal block, no regressors
   * - ``ARIMA``
     - ``ARIMA(endog, *, order=(p, d, q), trend="c")``
     - no seasonal block, no regressors
   * - ``SARIMA``
     - ``SARIMA(endog, *, order, seasonal_order=(P, D, Q, s), trend="c")``
     - seasonal block required, no regressors
   * - ``ARIMAX``
     - ``ARIMAX(endog, *, order, exog, trend="c")``
     - regressors required (``None`` or zero columns raise)
   * - ``SARIMAX``
     - ``SARIMAX(endog, *, order, seasonal_order=(0, 0, 0, 0), trend="c",
       exog=None)``
     - none

An ARMA(1, 1) on simulated data, with the moving-average sign convention
visible:

.. testcode::

   from cultivars.univariate.box_jenkins import ARIMA, ARIMAX, ARMA, SARIMA

   rng = np.random.default_rng(2)
   e = rng.standard_normal(500)
   x = np.zeros(500)
   for t in range(1, 500):
       x[t] = 0.8 * x[t - 1] + e[t] - 0.4 * e[t - 1]

   arma = ARMA(x, order=(1, 1)).fit()
   print(arma.order, arma.nobs, arma.n_params)
   print({k: round(v, 3) for k, v in arma.params.items()})
   print(arma.is_stationary, arma.is_invertible)

.. testoutput::

   (1, 0, 1) 500 4
   {'const': -0.158, 'ar.L1': 0.739, 'ma.L1': -0.345, 'sigma2': 1.022}
   True True

A simulated :math:`e_t - 0.4 e_{t-1}` comes back as ``ma.L1`` near
:math:`-0.4`.

Specification choices:

``order``
   ``(p, d, q)`` (``(p, q)`` for ``ARMA``), non-negative integers.
``seasonal_order``
   ``(P, D, Q, s)``; ``s`` must be at least 2 whenever ``P``, ``D`` or
   ``Q`` is positive.
``trend``
   ``"n"``, ``"c"`` or ``"ct"``, applied on the *differenced* scale. With
   ``d = 1`` and ``trend="c"`` the constant is the drift of the levels.
``exog``
   ``(n,)`` or ``(n, k)`` regressors, differenced alongside ``endog``.
   This is a regression with ARIMA errors, not a transfer function.

Parameter names extend the AR scheme: ``const``, ``trend``, ``x1`` ..
``xk`` (in the order of the ``beta`` array), ``ar.L1`` .., ``ma.L1`` ..,
seasonal terms at their lag multiples (``ar.S.L12``, ``ma.S.L12``,
``ma.S.L24`` ..), then ``sigma2``. The fields are ``ar_params``,
``ma_params``, ``seasonal_ar_params``, ``seasonal_ma_params``, ``beta``,
``sigma2``, plus ``order``, ``seasonal_order``, ``seasonal_period``,
``trend`` and ``k_exog``.

A seasonal model and an integrated one:

.. testcode::

   s = np.zeros(500)
   for t in range(13, 500):
       s[t] = 0.5 * s[t - 1] + 0.4 * s[t - 12] - 0.2 * s[t - 13] + e[t]
   seasonal = SARIMA(s, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)).fit()
   print(list(seasonal.params))
   print(seasonal.stability.eigenvalues.shape)

   levels = np.cumsum(0.3 + e)            # random walk with drift 0.3
   rw = ARIMA(levels, order=(0, 1, 0)).fit()
   print(rw.nobs, round(rw.beta[0], 3))

.. testoutput::

   ['const', 'ar.L1', 'ar.S.L12', 'sigma2']
   (13,)
   499 0.246

Stationarity and invertibility are judged on the *multiplied*
polynomials :math:`\phi(L)\Phi(L^s)` and :math:`\theta(L)\Theta(L^s)`, so
the seasonal AR(1) at :math:`s = 12` over an AR(1) has 13 companion
eigenvalues. ``stability`` and ``invertibility`` are
:class:`~cultivars.diagnostics.stability.StabilityAssessment` records.

A regression with ARIMA errors recovers a levels relationship even when
the error is integrated:

.. testcode::

   X = rng.standard_normal((500, 2))
   ylev = X @ [2.0, -1.0] + np.cumsum(rng.standard_normal(500))
   reg = ARIMAX(ylev, order=(0, 1, 0), exog=X, trend="n").fit()
   print(list(reg.params), reg.beta.round(2))

.. testoutput::

   ['x1', 'x2', 'sigma2'] [ 2.01 -0.91]

Long memory: ``ARFIMA``
-----------------------

.. math::

   \phi(L)\,(1 - L)^d\,(y_t - \mu) = \theta(L)\,\varepsilon_t,
   \qquad
   (1 - L)^d = \sum_{k \ge 0}
   \frac{\Gamma(k - d)}{\Gamma(-d)\,\Gamma(k + 1)} L^k.

For :math:`0 < d < 0.5` the process is stationary with autocorrelations
decaying as :math:`k^{2d-1}` (Granger and Joyeux, 1980; Hosking, 1981).
:class:`~cultivars.univariate.fractional_integration.ARFIMA` estimates
:math:`\mu`, :math:`d`, the short-memory block and :math:`\sigma^2`
jointly: the series is fractionally differenced with the filter cut at
``truncation`` terms and the ARMA block's exact Kalman likelihood is
evaluated on the result. The search starts at a local Whittle estimate
of :math:`d` and confines :math:`d` to :math:`(-0.499, 0.499)` through a
scaled ``tanh``.

.. testcode::

   from scipy.special import gammaln
   from cultivars.univariate.fractional_integration import ARFIMA

   rng = np.random.default_rng(3)
   n, d = 500, 0.3
   k = np.arange(n + 300)
   psi = np.exp(gammaln(k + d) - gammaln(d) - gammaln(k + 1))  # (1-L)^(-d)
   lm = np.convolve(rng.standard_normal(n + 300), psi)[300:300 + n]

   arfima = ARFIMA(lm, order=(0, 0)).fit()
   print({k: round(v, 3) for k, v in arfima.params.items()})
   print(arfima.truncation, arfima.is_stationary, arfima.has_long_memory)

   short = ARMA(lm, order=(1, 0)).fit()
   table = arfima.compare(short, criterion="bic")
   print([row[0] for row in table.rows])

.. testoutput::

   {'mean': 0.181, 'd': 0.319, 'sigma2': 0.983}
   500 True True
   ['ARFIMA(0, d, 0)', 'ARIMA(1, 0, 0)']

Specification choices: ``order=(p, q)`` for the short-memory block;
``trend`` is ``"c"`` (estimate :math:`\mu`, reported as ``mean``) or
``"n"``; ``truncation`` defaults to the sample size. The result
(:class:`~cultivars.univariate.fractional_integration.ARFIMAResult`)
carries ``d``, ``mean``, ``ar_params``, ``ma_params``, ``sigma2``,
``truncation``, the properties ``is_stationary`` (requires
:math:`|d| < 0.5` *and* a stable AR block), ``is_mean_reverting``
(:math:`d < 1`), ``has_long_memory`` (:math:`|d| > 10^{-3}`, a threshold
not a test), plus ``stability``, ``invertibility`` and the comparison
methods. ``fittedvalues`` and ``resid`` are on the fractionally
differenced scale. There is no ``simulate``.

Structural decomposition: ``UnobservedComponents``
--------------------------------------------------

.. math::

   y_t = \mu_t + \psi_t + \gamma_t + \varepsilon_t,
   \qquad \varepsilon_t \sim N(0, \sigma^2_\varepsilon),

with the trend :math:`\mu_t` a local level, a local linear trend or a
smooth trend, :math:`\psi_t` a damped stochastic cycle with estimated
period, and :math:`\gamma_t` a trigonometric seasonal whose harmonics
share one variance (Harvey, 1989; Harvey and Jaeger, 1993). The variances
(and the cycle's damping and frequency) are estimated by exact ML through
the Kalman filter under an approximate-diffuse prior (variance
:math:`10^6`) on the nonstationary states; every component is reported as
the full-sample smoothed estimate with its smoothed standard deviation.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Argument
     - Meaning
     - Default
   * - ``trend="level"``
     - local level :math:`\mu_{t+1} = \mu_t + \eta_t`
     - default
   * - ``trend="lltrend"``
     - local linear trend: level and slope both random walks
     -
   * - ``trend="smooth"``
     - integrated random walk (slope moves, level has no innovation); the
       smoothed level is the HP trend at
       :math:`\lambda = \sigma^2_\varepsilon / \sigma^2_\zeta`
     -
   * - ``cycle=True``
     - damped cycle, :math:`\rho` and :math:`\lambda_c` estimated;
       the period is confined to roughly 2 to 126 observations
     - ``False``
   * - ``seasonal=s``
     - trigonometric seasonal of period ``s`` (at least 2)
     - ``None``

.. testcode::

   from cultivars.univariate.unobserved_components import UnobservedComponents

   rng = np.random.default_rng(5)
   n = 200
   level = np.cumsum(0.1 * rng.standard_normal(n))
   pattern = np.tile([1.0, -0.5, -1.0, 0.5], n // 4)
   q = level + pattern + 0.3 * rng.standard_normal(n)

   uc = UnobservedComponents(q, seasonal=4).fit()
   print(uc.n_params, round(uc.sigma2_irregular, 3), round(uc.sigma2_level, 3))
   print(list(uc.to_pandas().columns))
   print(bool(np.allclose(uc.signal + uc.irregular, q)))

   mean, std = uc.forecast(4)
   print(mean.round(2), std.round(2))

.. testoutput::

   3.0 0.067 0.018
   ['observed', 'level', 'seasonal', 'irregular']
   True
   [-0.49 -1.92 -2.37 -1.04] [0.35 0.37 0.39 0.41]

The result
(:class:`~cultivars.univariate.unobserved_components.UnobservedComponentsResult`)
exposes:

- The estimated variances as fields: ``sigma2_irregular``,
  ``sigma2_level``, ``sigma2_slope``, ``sigma2_cycle``,
  ``sigma2_seasonal``, and ``cycle_rho``, ``cycle_freq`` (each ``None``
  when the component is absent), plus ``cycle_period``.
- Smoothed paths ``level``, ``level_std``, ``slope``, ``cycle``,
  ``seasonal``, ``signal`` (everything but the irregular) and
  ``irregular``. Asking for a component the specification lacks raises
  :class:`~cultivars.exceptions.SpecificationError`.
- ``forecast(steps)`` returning a ``(mean, std)`` pair of arrays; the
  standard errors include state uncertainty and the irregular, not
  parameter uncertainty.
- ``state_space``: the fitted system as a
  :class:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM`,
  re-applicable to new data (see :doc:`state_space`).
- ``simulate(n, seed=..., burn=0)``: unit-root components start at zero,
  not from the diffuse prior.
- ``llf``, ``nobs``, ``n_params``, ``information_criteria``, ``compare``,
  ``likelihood_ratio_test``, ``summary``.

Comparing, testing and exporting
--------------------------------

Every result on this page shares one comparison surface.
``information_criteria`` is a named tuple ``(aic, bic, hqic)`` computed
from ``llf``, ``nobs`` and ``n_params``. ``compare(*others,
criterion="aic")`` returns a ranked :class:`~cultivars.summary.SummaryTable`
(columns ``model``, ``llf``, ``k``, the criterion and its gap to the
best); there are deliberately no ``<`` / ``>`` operators.
``likelihood_ratio_test(unrestricted)`` treats the caller as the
restricted model and returns a record with ``statistic``, ``df`` and
``pvalue``. Both refuse results with different ``nobs``.

Order selection the way the package intends it, on a common sample:

.. testcode::

   fits = [AR(y, order=p, method="exact").fit() for p in (1, 2, 3)]
   table = fits[0].compare(*fits[1:], criterion="bic")
   print(table.columns)
   for row in table.rows:
       print(row[0], row[3], row[4])

   lr = fits[1].likelihood_ratio_test(fits[2])
   print(lr.df, round(lr.statistic, 3), round(lr.pvalue, 3))

.. testoutput::

   ('model', 'llf', 'k', 'BIC', 'dBIC')
   AR(2) 1445.784 0.000
   AR(3) 1451.739 5.956
   AR(1) 1460.832 15.048
   1 0.259 0.611

``to_pandas(index=None)``, ``to_polars()`` and ``to_dict()`` return the
aligned per-observation series. For AR, ARMA-family and ARFIMA results the
columns are ``observed``, ``fitted`` and ``resid``, trimmed to the
effective sample; for ``UnobservedComponents`` they are the decomposition.
The default index is ``0 .. nobs - 1`` of the *effective* sample, so pass
``index=`` to align with dates:

.. testcode::

   frame = css.to_pandas(index=np.arange(2, 500))
   print(frame.shape, frame.index[0])

.. testoutput::

   (498, 3) 2

Simulating from a fit
---------------------

``AR``, the Box-Jenkins results and ``UnobservedComponents`` provide
``simulate(n=200, *, seed=None, burn=...)``, which returns an ``(n,)``
array at the fitted parameters with Gaussian innovations. ``AR`` adds the
mean path implied by the deterministic terms and refuses a fit whose AR
coefficients sum to one. The Box-Jenkins simulator generates the
stationary ARMA part, adds the deterministic block on the differenced
scale and integrates from zero, so the first :math:`d + sD` entries are
exactly zero; it refuses fits with regressors.

.. testcode::

   print(css.simulate(3, seed=0).round(3))
   path = ARIMA(levels, order=(1, 1, 0)).fit().simulate(4, seed=0)
   print(path[0], path.shape)

.. testoutput::

   [0.34  1.983 1.36 ]
   0.0 (4,)

How the pieces connect
----------------------

- **Choosing** ``d``. Run :func:`~cultivars.diagnostics.unit_roots.adf`,
  :func:`~cultivars.diagnostics.unit_roots.kpss` and friends before
  setting ``d`` (:doc:`diagnostics`). For fractional :math:`d`,
  :func:`~cultivars.diagnostics.long_memory.local_whittle` and
  :func:`~cultivars.diagnostics.long_memory.gph` estimate it without a
  short-memory specification; seasonal unit roots are tested by
  :func:`~cultivars.diagnostics.seasonality.hegy`.
- **Decompositions.** An ARIMA(p, 1, q) result feeds
  :class:`~cultivars.spectral.filters.BeveridgeNelsonDecomposition`, and
  ``UnobservedComponents(trend="smooth")`` is the likelihood-based
  counterpart of :class:`~cultivars.spectral.filters.HodrickPrescottFilter`
  (:doc:`spectral`).
- **State space.** ``UnobservedComponentsResult.state_space`` is a
  :class:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM`
  whose ``loglikelihood`` on the estimation sample reproduces ``llf``;
  filter new data through it (:doc:`state_space`).
- **Forecast evaluation.** :class:`~cultivars.forecast.backtest.Backtest`
  re-fits a model at each origin and calls ``forecast(H)`` on the result
  unless you pass ``predict=``. In this family only
  ``UnobservedComponentsResult`` has ``forecast``, and it returns a
  ``(mean, std)`` tuple, so always pass a ``predict`` hook (see
  :doc:`forecasting`):

  .. testcode::

     from cultivars.forecast.backtest import Backtest

     bt = Backtest(q, lambda w: UnobservedComponents(w, seasonal=4).fit(),
                   horizons=2, start=180, step=10,
                   predict=lambda res, h, seed: res.forecast(h)[0])
     out = bt.run()
     print(out.n_origins, out.is_density)

  .. testoutput::

     2 False

- **Volatility.** A mean model with GARCH errors is estimated jointly by
  the ``AR*``/``ARMA*`` fronts in
  :mod:`~cultivars.univariate.conditional_variance`, not by chaining a
  fit here into a variance model (:doc:`volatility`). Time-varying
  trend variances are :class:`~cultivars.univariate.stochastic_volatility.UCSV`.

Nuances and gotchas
-------------------

- **No** ``forecast()`` **on AR, ARMA-family or ARFIMA results.**
  ``ARResult``, ``ARMAResult`` and ``ARFIMAResult`` have no ``forecast``
  or ``predict`` method. Compute point forecasts from the estimates (for
  an AR with ``trend="c"``, iterate :math:`c + \sum_i \phi_i y_{T+h-i}`),
  or fit :class:`~cultivars.univariate.unobserved_components.UnobservedComponents`
  when you need forecasts with standard errors out of the box.

  .. testcode::

     print(hasattr(css, "forecast"), hasattr(arma, "forecast"),
           hasattr(arfima, "forecast"), hasattr(uc, "forecast"))

  .. testoutput::

     False False False True

- **No standard errors.** None of the four results reports standard
  errors, t-statistics or confidence intervals; each summary says
  "Standard errors are not yet available for this estimator." Use
  ``likelihood_ratio_test`` for nested hypotheses.
- **Effective sample sizes differ.** ``nobs`` is :math:`n - p` for
  ``AR(method="css")``, :math:`n` for ``AR(method="exact")``,
  :math:`n - d - sD` for the Box-Jenkins models (MA and AR lags do not
  shorten it; the exact likelihood uses every differenced observation),
  :math:`n` for ARFIMA and ``UnobservedComponents``. ``compare`` and
  ``likelihood_ratio_test`` raise
  :class:`~cultivars.exceptions.DimensionError` on differing ``nobs``, so
  AR orders fitted by CSS cannot be ranked against each other: use
  ``method="exact"``.

  .. testcode::

     from cultivars.exceptions import DimensionError
     try:
         css.compare(AR(y, order=3).fit())
     except DimensionError as err:
         print(err)

  .. testoutput::

     information criteria are only comparable across a common sample; got differing observation counts [497, 498].

- **Mixed scales in** ``to_pandas`` **after differencing.** For an
  ARIMA with ``d > 0`` (or ``D > 0``) the ``observed`` column is the
  *level* series trimmed to the last ``nobs`` rows, while ``fitted`` and
  ``resid`` are on the *differenced* scale, so ``observed - fitted`` is
  not ``resid``. Difference ``endog`` yourself to compare like with like.

  .. testcode::

     df = rw.to_pandas()
     print(round(df["observed"].iloc[-1], 2), round(df["fitted"].iloc[-1], 2))
     print(bool(np.allclose(np.diff(levels) - rw.fittedvalues, rw.resid)))

  .. testoutput::

     123.29 0.25
     True

- **Two intercept conventions.** ``ARResult.const`` is the recursion
  intercept :math:`c`; ``ARMAResult.beta[0]`` (``params["const"]``) is
  the mean of the differenced series. On the same AR(1) data the two
  differ by the factor :math:`1 - \phi`:

  .. testcode::

     ar1 = AR(y, order=1).fit()
     arma10 = ARMA(y, order=(1, 0)).fit()
     print(round(ar1.const / (1 - ar1.ar_params[0]), 2), round(arma10.beta[0], 2))

  .. testoutput::

     0.76 0.76

- **Stability verdicts on constrained fits verify the transform.** The
  Box-Jenkins, exact-AR and ARFIMA estimators cannot leave the stationary
  and invertible region, so ``is_stationary`` is ``True`` even on a random
  walk; read ``stability.max_modulus`` (close to one) instead. Only ``AR(method="css")``
  can report a non-stationary fit.

  .. testcode::

     walk = ARMA(np.cumsum(rng.standard_normal(400)), order=(1, 0)).fit()
     print(walk.is_stationary, round(walk.stability.max_modulus, 3))

  .. testoutput::

     True 0.984

- **Near-cancelling ARMA roots are weakly identified, and the optimizer
  has one start.** When :math:`\phi_1 + \theta_1` is small the ARMA(1, 1)
  likelihood is nearly flat along :math:`\phi_1 = -\theta_1`, and the fit
  can stop at a common-factor point. On the series below (true
  :math:`\phi_1 = 0.6`, :math:`\theta_1 = -0.4`) the estimate is a pair of
  almost cancelling roots; an independent exact-likelihood maximization
  of the same data reaches a slightly higher ``llf`` (by about 0.03) at
  :math:`(0.71, -0.62)`. Compare against the AR(1) and white-noise fits
  before reading an ARMA(1, 1) coefficient.

  .. testcode::

     rng1 = np.random.default_rng(1)
     e1 = rng1.standard_normal(500)
     x1 = np.zeros(500)
     for t in range(1, 500):
         x1[t] = 0.6 * x1[t - 1] + e1[t] - 0.4 * e1[t - 1]
     weak = ARMA(x1, order=(1, 1)).fit()
     print(weak.ar_params.round(3), weak.ma_params.round(3))

  .. testoutput::

     [0.986] [-0.965]

- **ARFIMA's** :math:`d` **is boundary-capped.** :math:`d` is confined to
  :math:`(-0.499, 0.499)`, so a unit-root series returns ``d`` near 0.499
  with ``is_stationary=True``. Treat :math:`d` at the boundary as a request
  for integer differencing (fit ``ARIMA`` with ``d=1``, or ARFIMA on the
  differences). :math:`d` and a near-unit AR root trade off in short
  samples; keep ``order`` small.
- **Exogenous regressors are differenced too.** In ``ARIMAX`` /
  ``SARIMAX`` with ``d > 0`` the regressors are differenced with
  ``endog``; supply them in levels when the relationship is in levels.
  A result with regressors refuses ``simulate``.
- **Linear trend under exact ML is refused.**
  ``AR(y, order=p, trend="ct", method="exact")`` raises
  :class:`~cultivars.exceptions.SpecificationError`.
- **Unobserved-components likelihoods are approximate-diffuse.** ``llf``
  is comparable across specifications on one sample, not with software
  that uses an exact-diffuse initialization. The result has no ``params``
  dict, ``fittedvalues`` or ``resid``; read the named variance fields.
  Fits with ``cycle=True`` are noticeably slower (two optimizer starts).
- **Likelihood-ratio tests on variances are non-standard.** A test of
  ``UnobservedComponents(y)`` against ``UnobservedComponents(y,
  seasonal=4)`` puts a variance on the boundary under the null; the
  reported :math:`\chi^2` p-value is conservative.

See also
--------

- :doc:`conventions` -- shapes, parameter naming and sign conventions
  across the package.
- :doc:`volatility` -- conditional-variance and stochastic-volatility
  models, including ARMA means with GARCH errors.
- :doc:`univariate_nonlinear` -- threshold, smooth-transition and
  Markov-switching autoregressions.
- :doc:`state_space` -- the linear-Gaussian substrate behind ARIMA and
  unobserved components.
- :doc:`diagnostics` -- unit-root, long-memory and seasonality tests.
- :doc:`forecasting` -- backtesting and forecast evaluation.
- API: :mod:`cultivars.univariate.autoregression`,
  :mod:`cultivars.univariate.box_jenkins`,
  :mod:`cultivars.univariate.fractional_integration`,
  :mod:`cultivars.univariate.unobserved_components`.
