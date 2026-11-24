.. meta::
   :description: Producing forecasts from cultivars results and grading them: backtests, proper scores, calibration, comparison tests, and the model confidence set.

Forecasting and forecast evaluation
===================================

This page covers two things: how a fitted result in cultivars produces a
forecast, and how :mod:`cultivars.forecast` grades forecasts out of sample.
The evaluation tools answer the questions a forecasting exercise asks: which
model forecasts better, whether the gap is significant, whether the
predictive densities are calibrated, and which models a window of data
cannot rank. Model weighting by marginal likelihood or stacking is part of
:doc:`bayesian_workflow`. In-sample residual diagnostics are covered in
:doc:`diagnostics`.

.. contents:: On this page
   :local:
   :depth: 1

The family at a glance
----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 22 30 24

   * - Tool
     - Module
     - What it does
     - Use when
   * - :class:`~cultivars.forecast.backtest.Backtest`
     - :mod:`~cultivars.forecast.backtest`
     - Re-estimates a model at every origin of an expanding or rolling
       schedule and records forecasts next to outcomes
     - Every out-of-sample exercise; all other tools read its record
   * - :class:`~cultivars.forecast.scoring.DensityScore`,
       :func:`~cultivars.forecast.scoring.pinball_loss`
     - :mod:`~cultivars.forecast.scoring`
     - CRPS, energy score, log score, point losses for one origin; the
       quantile loss
     - Grading one predictive sample, or quantile forecasts
   * - :class:`~cultivars.forecast.calibration.Calibration`
     - :mod:`~cultivars.forecast.calibration`
     - PIT series, histogram, Berkowitz (2001) test
     - Checking whether densities are too narrow, too wide, or biased
   * - :class:`~cultivars.forecast.comparison.ForecastComparison`,
       :func:`~cultivars.forecast.comparison.giacomini_white`
     - :mod:`~cultivars.forecast.comparison`
     - Diebold-Mariano, Harvey-Leybourne-Newbold, Giacomini-White
     - Two non-nested forecasters on the same origins
   * - :func:`~cultivars.forecast.comparison.clark_west`
     - :mod:`~cultivars.forecast.comparison`
     - Adjusted MSPE test for nested models
     - The larger model nests the smaller one
   * - :func:`~cultivars.forecast.comparison.mincer_zarnowitz`,
       :func:`~cultivars.forecast.comparison.encompassing`
     - :mod:`~cultivars.forecast.comparison`
     - Efficiency regression; forecast encompassing
     - Asking whether one forecast can be improved, or adds to another
   * - :func:`~cultivars.forecast.confidence_set.model_confidence_set`
     - :mod:`~cultivars.forecast.confidence_set`
     - Hansen-Lunde-Nason (2011) set of models not worse than the best
     - Three or more forecasters
   * - :func:`~cultivars.forecast.conditional.conditional_forecast`
     - :mod:`~cultivars.forecast.conditional`
     - Waggoner-Zha (1999) forecast given stated paths for some variables
     - Policy or scenario forecasts from a VAR-type system
   * - :func:`~cultivars.forecast.fan.fan_chart`
     - :mod:`~cultivars.forecast.fan`
     - Pointwise quantile bands of predictive paths
     - Handing bands to a plotting library

Every evaluation tool takes arrays (paths, outcomes, losses), not models.
A forecaster from outside the package is graded on the same terms as one
from inside it. Every loss is negatively oriented: smaller is better,
including the log score, which is the negative log predictive density.

Where forecasts come from
-------------------------

There is no forecasting protocol that every result implements. Three
conventions coexist, and which one a result follows decides how the
:class:`~cultivars.forecast.backtest.Backtest` reads it.

**Point forecasters.** The reduced-form multivariate results expose
``forecast(steps=1)`` returning a ``(steps, k)`` array of conditional
means.

.. testcode::

   import numpy as np
   from cultivars.multivariate.reduced_form.vector_autoregression import VAR

   rng = np.random.default_rng(0)
   A = np.array([[0.5, 0.2], [0.0, 0.6]])
   y = np.zeros((200, 2))
   for t in range(1, 200):
       y[t] = A @ y[t - 1] + rng.standard_normal(2)

   var = VAR(y, order=1, names=("gdp", "infl")).fit()
   print(var.forecast(3).round(3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   [[ 0.53  -0.716]
    [ 0.09  -0.362]
    [-0.092 -0.172]]

**Density forecasters.** The sampled Bayesian VARs expose
``forecast_paths(steps=8, *, seed=None)``, one simulated path per retained
posterior draw, so the paths integrate parameter and shock uncertainty.
Their ``forecast(steps=8, *, seed=None)`` is a summary of those paths with
shape ``(steps, k, 3)``: 16th percentile, mean, 84th percentile.

.. testcode::

   from cultivars.multivariate.large_dim.bayesian import BVAR

   bvar = BVAR(y, order=1, names=("gdp", "infl")).fit(n_draws=200, seed=0)
   paths = bvar.forecast_paths(3, seed=0)
   summary = bvar.forecast(3, seed=0)
   print(paths.shape, summary.shape)
   print(bool(np.allclose(summary[..., 1], paths.mean(axis=0))))

.. testoutput::

   (200, 3, 2) (3, 2, 3)
   True

The table lists every forecasting method on a public result. Signatures
were read from the source; defaults differ between families.

.. list-table::
   :header-rows: 1
   :widths: 38 34 28

   * - Result
     - Method
     - Returns
   * - ``VARResult``, ``VARMAResult``, ``VECMResult``, ``GVARResult``,
       ``MFVARResult``
     - ``forecast(steps=1)``
     - ``(steps, k)`` point path
   * - ``VARXResult``, ``VECMXResult``, ``MIDASVARResult``
     - ``forecast(steps=1, *, exog_future=...)``
       (``exog_high_future`` for MIDAS)
     - ``(steps, k)``; the future exogenous path is required
   * - ``PanelVARResult``
     - ``forecast(steps=1, *, unit=None)``
     - ``(steps, k)`` for one unit
   * - ``TVARResult``, ``STVARResult``
     - ``forecast(steps=1, *, transition_future=None)``
     - ``(steps, k)`` point path
   * - ``DFMResult``
     - ``forecast(steps=8)``
     - ``(steps, n_series)`` in original units
   * - ``BVARResult``, ``GibbsBVARResult``, ``StudentBVARResult``,
       ``BVARSVResult``, ``HierarchicalBVARResult``
     - ``forecast_paths(steps=8, *, seed=None)``;
       ``forecast(steps=8, *, seed=None)``
     - ``(n_kept, steps, k)`` draws; ``(steps, k, 3)`` summary
   * - :class:`~cultivars.forecast.conditional.ConditionalForecastResult`
     - ``forecast_paths(steps=None, *, seed=None)``
     - the stored ``(S, steps, k)`` paths
   * - :class:`~cultivars.bayes.combination.ModelCombinationSelection`
     - ``forecast_paths(steps=8, *, n_draws=None, seed=None)``;
       ``forecast(...)``
     - mixture draws; ``(steps, k, 3)`` summary
   * - ``QVARResult``
     - ``forecast(quantile)``
     - ``(k,)`` one-step quantile forecast
   * - ``FunctionalCoefficientVARResult``
     - ``forecast()``
     - ``(k,)`` one-step forecast
   * - ``FunctionalVARResult``
     - ``forecast_curves(steps=1)``
     - forecast curves
   * - ``DynamicNelsonSiegelResult``
     - ``forecast(steps)``
     - tuple ``(mean, std)``, each ``(steps, p)``
   * - ``UnobservedComponentsResult``
     - ``forecast(steps)``
     - tuple ``(mean, std)``, each ``(steps,)``
   * - ``UCSVPosterior``
     - ``forecast(steps, *, n_paths=2000, seed=None)``
     - ``(n_paths, steps)`` predictive draws
   * - ``SVResult``
     - ``forecast_variance(steps)``
     - ``(steps,)`` expected variance

**No forecast method.** The univariate linear and nonlinear results do not
forecast: ``ARResult``, ``ARMAResult`` (returned by ``ARMA``, ``ARIMA``,
``SARIMA``), the GARCH family (``GARCHResult``, ``FIGARCHResult``,
``ARMAGARCHResult``, ``ARMAFIGARCHResult``), ``SETARResult``,
``STARResult``, ``MarkovSwitchingARResult`` and ``SVPosterior`` have
neither ``forecast`` nor ``forecast_paths``. ``ARNNResult`` and
``TARNNResult`` have ``predict(features)``, a one-step map from a design
matrix you build. The section `Backtesting a model without forecast()`_
shows how to evaluate these models anyway.

Backtests
---------

A backtest is the loop behind every out-of-sample evaluation: estimate on
the data available at an origin, forecast :math:`H` steps, store the
forecasts next to the outcomes, advance (Tashman 2000; West 2006). With
``start`` the size of the first estimation window and ``step`` the spacing,
:class:`~cultivars.forecast.backtest.Backtest` visits

.. math::

   o_t = \texttt{start} + t \cdot \texttt{step}, \qquad
   T = \Bigl\lfloor \frac{n - H - \texttt{start}}{\texttt{step}} \Bigr\rfloor + 1,

and stores, for every horizon :math:`h \le H` and series :math:`j`,

.. math::

   \texttt{point}[t, h-1, j] = \hat y_{o_t + h - 1 \mid o_t,\, j}, \qquad
   \texttt{realized}[t, h-1, j] = y_{o_t + h - 1,\, j}.

The model enters as a *fitting function* from a sample window to a fitted
result. At each origin the harness calls it, then asks the result to
forecast:

- if ``predict`` was given, ``predict(result, steps, seed)``;
- else, if the result has ``forecast_paths``, it is a density forecaster
  and ``forecast_paths(steps, seed=seed)`` is recorded as draws;
- else ``forecast(steps)`` is recorded as a point forecast; a
  ``(steps, k, 3)`` summary is read at its middle (mean) column.

A point VAR and a density BVAR on the same schedule:

.. testcode::

   from cultivars.forecast.backtest import Backtest

   names = ("gdp", "infl")
   point = Backtest(
       y, lambda w: VAR(w, order=1).fit(), horizons=2, start=140, names=names
   ).run()
   fit_bvar = lambda w: BVAR(w, order=1).fit(n_draws=200, seed=0)
   density = Backtest(y, fit_bvar, horizons=2, start=140, names=names).run(seed=0)

   print(point.n_origins, point.origins[:3], point.is_density)
   print(density.n_draws, density.paths.shape)
   print(point.rmse.round(3))
   print(density.rmse.round(3))
   print(density.mean_crps.round(3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   59 [140 141 142] False
   200 (59, 200, 2, 2)
   [[0.932 0.976]
    [0.989 1.108]]
   [[0.929 0.978]
    [0.995 1.101]]
   [[0.529 0.535]
    [0.568 0.617]]

Rows of ``rmse``, ``mae``, ``mean_crps`` and ``mean_log_score`` are
horizons, columns are series. ``summary()`` prints the same numbers with
the schedule in the notes.

Specification choices
~~~~~~~~~~~~~~~~~~~~~

``start`` (required, keyword-only)
   Observations in the first estimation window. Must satisfy
   ``1 <= start <= n - horizons``. The first origin forecasts rows
   ``start .. start + horizons - 1``.
``horizons=1``
   Longest horizon :math:`H`. Every horizon ``1 .. H`` is kept.
``scheme="expanding"``
   ``"expanding"`` fits on rows ``0 .. o_t - 1``; ``"rolling"`` fits on the
   last ``window`` of them.
``window=None``
   Rolling window length; defaults to ``start``. Must not exceed
   ``start``. Ignored when expanding, but recorded on the result.
``step=1``
   Origins between evaluations. Estimation still happens at every visited
   origin; ``step`` only thins the schedule. There is no
   "re-estimate every :math:`r` origins" mode.
``predict=None``
   ``(result, steps, seed) -> array`` hook. Return ``(steps, k)`` for a
   point forecast or ``(n_draws, steps, k)`` for draws; with one series,
   ``(steps,)`` and ``(n_draws, steps)`` are accepted.
``names=None``
   Series labels; default ``y1 .. yk``, or ``y`` for a single series.
``run(*, seed=None)``
   Origin ``t`` passes ``seed + t`` to ``forecast_paths`` or the hook.
   With ``seed=None`` a density backtest draws fresh shocks on every run.

A rolling schedule that evaluates every fifth origin on 100-observation
windows:

.. testcode::

   rolling = Backtest(
       y, lambda w: VAR(w, order=1).fit(), start=140,
       scheme="rolling", window=100, step=5,
   )
   print(rolling.origins)
   print(rolling.run().window)

.. testoutput::

   [140 145 150 155 160 165 170 175 180 185 190 195]
   100

The record
~~~~~~~~~~

:class:`~cultivars.forecast.backtest.BacktestResult` holds ``realized`` and
``point`` as ``(T, H, k)``, ``paths`` as ``(T, S, H, k)`` or ``None``, and
``errors`` as ``point - realized`` (positive means over-forecast). The
methods that feed the other modules:

``losses(kind="squared", *, horizon=1, name=None, tau=None)``
   One loss per origin: ``"squared"``, ``"absolute"``, ``"crps"``,
   ``"log"``, or ``"pinball"`` (requires ``tau``). Returns ``(T,)`` with
   ``name`` and ``(T, k)`` without. ``"crps"`` and ``"log"`` need draws.
``quantile(tau, *, horizon=1)``
   ``(T, k)`` empirical quantiles of the draws.
``record(horizon=1)``
   ``(paths, realized)`` as ``(T, S, k)`` and ``(T, k)``, the input of
   :class:`~cultivars.forecast.calibration.Calibration`. Needs draws.

.. testcode::

   crps = density.losses("crps", name="gdp")
   log_score = density.losses("log", name="gdp")
   pinball = density.losses("pinball", tau=0.9, name="gdp")
   print(crps.shape, round(crps.mean(), 3), round(log_score.mean(), 3))
   print(pinball.shape, density.quantile(0.9).shape)

.. testoutput::

   (59,) 0.529 1.355
   (59,) (59, 2)

Backtesting a model without forecast()
--------------------------------------

The ``predict`` hook covers any result that does not follow the
conventions, including the univariate models that have no forecast method.
The hook receives the fitted result, the horizon, and the origin's seed.
For an AR(:math:`p`) with intercept, the point forecast iterates
:math:`\hat y_{T+h} = c + \sum_i \phi_i \hat y_{T+h-i}` from the result's
``const``, ``ar_params`` and ``endog``:

.. testcode::

   from cultivars.univariate.autoregression import AR

   rng = np.random.default_rng(1)
   x = np.zeros(200)
   for t in range(1, 200):
       x[t] = 0.6 * x[t - 1] + rng.standard_normal()

   def ar_point(res, steps, seed):
       history = list(res.endog[-res.order:])
       out = []
       for _ in range(steps):
           lags = history[::-1][: res.order]
           nxt = res.const + float(np.dot(res.ar_params, lags))
           out.append(nxt)
           history.append(nxt)
       return np.array(out)

   ar1 = Backtest(
       x, lambda w: AR(w, order=1).fit(), horizons=2, start=140, predict=ar_point
   ).run()
   print(ar1.names, ar1.point.shape)
   print(ar1.rmse.round(3).ravel())

.. testoutput::

   ('y',) (59, 2, 1)
   [1.005 1.099]

A hook can also return draws. Here Gaussian shocks with the fitted
``sigma2`` give a ``(n_draws, steps)`` array, which the harness reads as a
density forecast for a single series. These draws carry shock uncertainty
only, not parameter uncertainty.

.. testcode::

   def ar_density(res, steps, seed, n_draws=500):
       gen = np.random.default_rng(seed)
       p = res.order
       history = np.tile(res.endog[-p:], (n_draws, 1))
       draws = np.empty((n_draws, steps))
       for h in range(steps):
           mean = res.const + history[:, ::-1][:, :p] @ res.ar_params
           draws[:, h] = mean + np.sqrt(res.sigma2) * gen.standard_normal(n_draws)
           history = np.column_stack([history[:, 1:], draws[:, h]])
       return draws

   ar1_density = Backtest(
       x, lambda w: AR(w, order=1).fit(), horizons=2, start=140, predict=ar_density
   ).run(seed=0)
   print(ar1_density.is_density, ar1_density.paths.shape)
   print(ar1_density.mean_crps.round(3).ravel())

.. testoutput::

   True (59, 500, 2, 1)
   [0.565 0.619]

The fitting function need not return a package result. The historical-mean
benchmark is a window mean and a hook that repeats it:

.. testcode::

   hist_mean = Backtest(
       x, lambda w: w.mean(), horizons=2, start=140,
       predict=lambda m, steps, seed: np.full(steps, m),
   ).run()
   print(hist_mean.rmse.round(3).ravel())

.. testoutput::

   [1.117 1.108]

Scoring rules
-------------

A strictly proper score is minimized in expectation only by the true
predictive distribution (Gneiting and Raftery 2007).
:class:`~cultivars.forecast.scoring.DensityScore` scores one origin's
predictive sample :math:`x^{(1)}, \dots, x^{(S)}` against one outcome path:

.. math::

   \mathrm{CRPS} = \frac{1}{S}\sum_s |x^{(s)} - y|
   - \frac{1}{2S^2}\sum_{s,s'} |x^{(s)} - x^{(s')}|,

the energy score (the same with the Euclidean norm over all :math:`k`
series), the log score :math:`-\log \hat f(y)` with :math:`\hat f` a
Gaussian kernel density of the draws (Silverman bandwidth), and the
squared error of the predictive mean and absolute error of the predictive
median. The CRPS and energy score are exact functions of the sample; the
log score is estimated.

.. testcode::

   from cultivars.forecast.scoring import DensityScore, pinball_loss

   held_out = BVAR(y[:196], order=1, names=names).fit(n_draws=200, seed=0)
   scores = DensityScore(
       held_out.forecast_paths(4, seed=0), y[196:], names=names
   ).compute()
   print(scores.crps.shape, scores.energy.shape, scores.n_draws)
   print(scores.crps.round(3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   (4, 2) (4,) 200
   [[0.567 0.334]
    [0.326 0.566]
    [0.448 0.313]
    [1.367 0.892]]

``paths`` must be ``(n_draws, h, k)`` or ``(n_draws, k)``. A
one-dimensional ``(n_draws,)`` sample is refused; pass a single series as
``(n_draws, 1)``.

:func:`~cultivars.forecast.scoring.pinball_loss` is the proper score for a
:math:`\tau`-quantile forecast :math:`q` (Koenker and Bassett 1978;
Gneiting 2011), :math:`\ell_\tau(y, q) = (y - q)(\tau - \mathbb{1}\{y < q\})`,
applied elementwise:

.. testcode::

   print(pinball_loss([1.0, -1.0], [0.0, 0.0], 0.9))

.. testoutput::

   [0.9 0.1]

Calibration
-----------

A forecaster can score well and still misstate its uncertainty. The
probability integral transform :math:`u_t = \hat F_t(y_t)` of a calibrated
forecaster is i.i.d. uniform (Diebold, Gunther and Tay 1998). A U-shaped
histogram means the densities are too narrow, a hump means too wide, a
slope means bias. Berkowitz (2001) tests :math:`z_t = \Phi^{-1}(u_t)`
against :math:`z_t = \mu + \rho z_{t-1} + \varepsilon_t` with
:math:`H_0: \mu = 0, \rho = 0, \sigma^2 = 1`, a :math:`\chi^2_3`
likelihood ratio.

:class:`~cultivars.forecast.calibration.Calibration` takes the
``(T, n_draws, k)`` stack that ``BacktestResult.record(h)`` returns and
needs at least 10 origins:

.. testcode::

   from cultivars.forecast.calibration import Calibration

   calib = Calibration(*density.record(1), names=names).compute()
   print(calib.pit.shape)
   print(calib.histogram(4))
   print(round(calib.berkowitz("gdp").pvalue, 3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   (59, 2)
   [[17. 13.]
    [13. 20.]
    [13. 18.]
    [16.  8.]]
   0.133

``histogram(bins=10)`` returns counts with one column per series.
``berkowitz(name)`` returns a likelihood-ratio test record with
``statistic``, ``pvalue`` and ``reject``. Read the formal test at horizon
one: multi-step PITs overlap and their serial correlation reads as
miscalibration.

Comparing two forecasters
-------------------------

Diebold-Mariano and Giacomini-White
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a loss differential :math:`d_t = L_{A,t} - L_{B,t}` the
Diebold-Mariano (1995) statistic is

.. math::

   \mathrm{DM} = \frac{\bar d}{\sqrt{\hat\sigma^2_{LR}/T}}, \qquad
   \hat\sigma^2_{LR} = \hat\gamma_0 + 2\sum_{j=1}^{h-1}
   \Bigl(1 - \frac{j}{h}\Bigr)\hat\gamma_j .

:class:`~cultivars.forecast.comparison.ForecastComparison` reports it with
the Harvey, Leybourne and Newbold (1997) correction (read against
:math:`t_{T-1}`) and the Giacomini and White (2006) conditional test with
instruments :math:`(1, d_{t-h})`. The horizon is an argument of
``compute``, not of the constructor:

.. testcode::

   from cultivars.forecast.comparison import ForecastComparison

   verdict = ForecastComparison(
       point.losses(name="gdp"), density.losses(name="gdp")
   ).compute()
   print(verdict.nobs, verdict.horizon, round(verdict.mean_differential, 4))
   print(round(verdict.dm_statistic, 3), round(verdict.dm_pvalue, 3))
   print(round(verdict.hln_pvalue, 3), round(verdict.gw_pvalue, 3))

   two_step = ForecastComparison(
       point.losses(horizon=2, name="gdp"), density.losses(horizon=2, name="gdp")
   ).compute(horizon=2)
   print(two_step.horizon, round(two_step.hln_pvalue, 3))

.. testoutput::

   59 1 0.0059
   0.289 0.773
   0.776 0.099
   2 0.63

A positive mean differential favors the second forecaster. Here the BVAR's
one-step squared error for ``gdp`` is lower by 0.006 on average, and no
test separates the two.

:func:`~cultivars.forecast.comparison.giacomini_white` runs the
conditional test alone with ``lags`` and user ``instruments``; it loses
``horizon`` origins to the lagged differential (58 here).

Nested models: Clark-West
~~~~~~~~~~~~~~~~~~~~~~~~~

When the larger model nests the smaller one, the larger model's MSPE is
inflated by estimation noise under the null and the Diebold-Mariano
statistic is not asymptotically normal (Clark and McCracken 2001). Clark
and West (2007) add back the squared forecast gap:

.. math::

   \hat f_t = e_{r,t}^2 - \bigl[e_{u,t}^2 - (\hat y_{r,t} - \hat y_{u,t})^2\bigr],

and test its mean against the upper tail of the standard normal. The
historical mean is nested in the AR(1) with intercept:

.. testcode::

   from cultivars.forecast.comparison import clark_west

   cw = clark_west(
       ar1.realized[:, 0, 0],   # outcomes, horizon 1
       hist_mean.point[:, 0, 0],  # restricted forecast
       ar1.point[:, 0, 0],        # unrestricted forecast
   )
   print(round(cw.statistic, 3), round(cw.pvalue, 4), bool(cw.reject()))
   print(round(cw.mspe_restricted, 3), round(cw.mspe_unrestricted, 3))

.. testoutput::

   2.866 0.0021 True
   1.248 1.01

The functions take forecasts and outcomes, not losses. Pass the same
``horizon`` keyword you backtested at; it sets the HAC window.

Efficiency and encompassing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~cultivars.forecast.comparison.mincer_zarnowitz` regresses
:math:`y_t = a + b f_t + u_t` and tests :math:`(a, b) = (0, 1)` with a HAC
Wald statistic, :math:`W/2` read against :math:`F(2, T-2)` (Mincer and
Zarnowitz 1969). :func:`~cultivars.forecast.comparison.encompassing` tests
whether forecast A already contains what B knows, one-sided, with the
least-squares combination weight on B (Harvey, Leybourne and Newbold
1998):

.. testcode::

   from cultivars.forecast.comparison import encompassing, mincer_zarnowitz

   mz = mincer_zarnowitz(point.realized[:, 0, 0], point.point[:, 0, 0])
   print(round(mz.intercept, 3), round(mz.slope, 3), bool(mz.reject()))

   enc = encompassing(
       point.realized[:, 0, 0], point.point[:, 0, 0], density.point[:, 0, 0]
   )
   print(round(enc.weight, 3), round(enc.pvalue, 3))

.. testoutput::

   -0.059 0.816 False
   0.84 0.242

The encompassing test is directional: swap the forecasts to ask whether B
encompasses A. A large weight with no rejection, as here, says the window
cannot tell the two highly correlated forecasts apart.

The model confidence set
------------------------

With more than two forecasters, Hansen, Lunde and Nason (2011) eliminate
the worst model while the hypothesis of equal predictive ability among the
survivors is rejected, using a stationary bootstrap of the loss panel
(Politis and Romano 1994). The result is a set that contains the best
model with probability at least :math:`1 - \alpha`. The input is a
``(T, M)`` panel of losses on the same origins:

.. testcode::

   from cultivars.forecast.confidence_set import model_confidence_set

   ar2 = Backtest(
       x, lambda w: AR(w, order=2).fit(), horizons=2, start=140, predict=ar_point
   ).run()
   panel = np.column_stack([m.losses(name="y") for m in (hist_mean, ar1, ar2)])
   mcs = model_confidence_set(
       panel, names=("mean", "AR(1)", "AR(2)"), n_bootstrap=500, seed=0
   )
   print(mcs.included, mcs.best)
   print(mcs.pvalues.round(3), mcs.mean_losses.round(3))

.. testoutput::

   ('AR(1)', 'AR(2)', 'mean') AR(1)
   [0.102 1.    0.37 ] [1.248 1.01  1.031]

The defaults are ``alpha=0.1`` (a 90% set), ``statistic="range"``
(alternative ``"max"``), ``n_bootstrap=1000`` (at least 100) and
``block_length=max(2, round(T ** (1/3)))``. ``pvalues`` follow the input
column order; ``included`` is ordered best first. The historical mean
survives at the 10% level with an MCS p-value of 0.102, although
Clark-West rejects it against the AR(1): the set answers a different and
more conservative question.

Conditional forecasts
---------------------

A conditional forecast fixes the future path of some variables and asks
what the rest of the system does (Doan, Litterman and Sims 1984; Waggoner
and Zha 1999). With the moving-average form
:math:`y_{T+h} = \mu_h + \sum_{j<h} \Psi_j P \varepsilon_{T+h-j}`, each
conditioned cell is a linear restriction :math:`R\varepsilon = r` on the
future standardized shocks. The paths are drawn from the restricted
Gaussian centred on the minimum-norm shocks
:math:`\varepsilon^\ast = R^\top(RR^\top)^{-1} r`.

.. testcode::

   from cultivars.forecast.conditional import conditional_forecast

   scenario = conditional_forecast(var, 4, {"infl": [1.0, 1.0]}, seed=0)
   print(scenario.paths.shape, scenario.n_conditions)
   print(scenario.mean.round(3))
   print(round(scenario.modesty, 2), round(scenario.modesty_pvalue, 3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   (1000, 4, 2) 2
   [[0.622 1.   ]
    [0.345 1.   ]
    [0.229 0.638]
    [0.104 0.372]]
   1.5 0.222

``conditions`` maps a variable name to its path from horizon one; ``None``
or ``nan`` frees a cell, and a short path frees the remaining horizons.
Conditions are hard: every path meets them exactly; soft (interval)
conditions are not implemented. ``modesty`` is
:math:`\|\varepsilon^\ast\|^2/q` for :math:`q` conditions, and
``modesty_pvalue`` reads :math:`\|\varepsilon^\ast\|^2` against
:math:`\chi^2_q` (Leeper and Zha 2003): a small p-value means the scenario
needs shocks the model deems unusual.

The function reads its input by duck typing. A point result needs
``names``, ``forecast(steps)``, ``ma_representation(horizon)`` and
``sigma_u`` (every VAR-family result), and its paths carry shock
uncertainty at fixed parameters. A result with ``beta_draws`` and
``sigma_draws`` (the Bayesian VARs) is propagated draw by draw, so the
paths also carry parameter uncertainty:

.. testcode::

   sampled = conditional_forecast(bvar, 4, {"infl": [1.0, 1.0]}, n_draws=100, seed=0)
   print(sampled.source, sampled.parameter_uncertainty, sampled.paths.shape)

.. testoutput::

   BVARResult True (100, 4, 2)

The result has ``forecast_paths()``, so it can be scored, fanned, or
combined like any density forecaster. ``forecast_paths(steps)`` refuses a
horizon other than the one it was built for.

Fan charts
----------

:func:`~cultivars.forecast.fan.fan_chart` returns pointwise quantiles of any
``(n_draws, steps, k)`` array as ``(len(quantiles), steps, k)``. The
default levels ``(0.05, 0.16, 0.5, 0.84, 0.95)`` give the 90% and 68% bands
around the median (Britton, Fisher and Whitley 1998). The bands are
pointwise per horizon, not joint over whole paths.

.. testcode::

   from cultivars.forecast.fan import fan_chart

   fan = fan_chart(bvar.forecast_paths(8, seed=0))
   print(fan.shape)
   print(fan[:, 0, 0].round(2))
   lower, upper = fan_chart(scenario.paths, quantiles=(0.16, 0.84))
   print(lower[:2, 1], upper[:2, 1])

.. testoutput::

   (5, 8, 2)
   [-0.95 -0.34  0.59  1.5   2.31]
   [1. 1.] [1. 1.]

The package draws nothing; pass the rows to a plotting library.

How the pieces connect
----------------------

- The :class:`~cultivars.engine._core._protocols.PredictiveResult` protocol
  (one method, ``forecast_paths(steps, *, seed)``) is what makes a result a
  density forecaster. ``Backtest`` checks it with ``isinstance``; the
  Bayesian VARs, ``ConditionalForecastResult`` and
  ``ModelCombinationSelection`` satisfy it, and so does any object of your
  own with that method.
- ``BacktestResult.losses`` feeds
  :class:`~cultivars.forecast.comparison.ForecastComparison`,
  :func:`~cultivars.forecast.comparison.giacomini_white` and, stacked as
  columns, :func:`~cultivars.forecast.confidence_set.model_confidence_set`.
  ``point`` and ``realized`` feed the forecast tests.
- ``BacktestResult.record(h)`` feeds
  :class:`~cultivars.forecast.calibration.Calibration`.
- Negated ``losses("log")`` are the held-out log predictive densities that
  :func:`~cultivars.bayes.combination.stacking` weights; see
  :doc:`bayesian_workflow`.
- ``forecast_paths`` output from any source feeds
  :class:`~cultivars.forecast.scoring.DensityScore` and
  :func:`~cultivars.forecast.fan.fan_chart`.

Nuances and gotchas
-------------------

- **Default horizons differ.** ``VARResult.forecast()`` returns one step;
  ``BVARResult.forecast()`` and ``forecast_paths()`` return eight. Always
  pass ``steps``.
- **The univariate families do not forecast.** AR, ARMA/ARIMA/SARIMA,
  GARCH-type, SETAR, STAR and Markov-switching results have no
  ``forecast``. A ``Backtest`` on them raises ``SpecificationError``
  ("offers neither forecast_paths() nor forecast(); pass predict=").
  Write a hook as shown above. There is no conditional-variance forecast on
  GARCH results; ``SVResult.forecast_variance`` is the only variance
  forecast among univariate results.
- **Tuple-returning forecasts are misread by Backtest.**
  ``UnobservedComponentsResult.forecast`` and
  ``DynamicNelsonSiegelResult.forecast`` return ``(mean, std)``. The
  harness converts the tuple to an array and records it as a density
  forecast with two "draws" (the mean and the standard deviation), without
  an error. ``n_draws`` is 2 and every density loss is meaningless. Use
  ``predict=lambda r, h, s: r.forecast(h)[0]`` for the point forecast.
- **UCSVPosterior is backtested unseeded.** Its ``forecast(steps)``
  returns ``(n_paths, steps)`` draws and is recorded as a density, but the
  harness does not pass the seed to ``forecast``. Use
  ``predict=lambda r, h, s: r.forecast(h, seed=s)`` for reproducible runs.
- **Exogenous paths.** ``VARXResult.forecast`` requires ``exog_future``,
  so a default backtest raises. The ``predict`` hook receives only
  ``(result, steps, seed)``, not the origin; recover the origin from the
  fitted result (for example ``len(result.endog)`` under the expanding
  scheme) to slice the exogenous path.
- **QVAR needs a hook and a reshape.** ``QVARResult.forecast(quantile)``
  takes a level, not a horizon, so the default call ``forecast(1)`` fails
  with "quantile 1 was not estimated". Its ``(k,)`` return is also refused
  for ``k > 1``; use ``predict=lambda r, h, s: r.forecast(0.9)[None, :]``
  with ``horizons=1``.
- **A point backtest's quantile is its point forecast.**
  ``BacktestResult.quantile(tau)`` and ``losses("pinball", tau=...)`` on a
  point backtest return the recorded forecasts whatever ``tau`` is. This is
  intended for quantile forecasters, but on a mean forecaster it silently
  scores the mean as every quantile.
- **Pass one series to the comparison tests.**
  ``ForecastComparison`` flattens its inputs. ``losses()`` without
  ``name`` is ``(T, k)``, and passing two such arrays runs the test on
  ``T * k`` interleaved values with no error. Select a series with
  ``name=`` or aggregate across series yourself.
- **Minimum window lengths.** ``ForecastComparison`` and the forecast tests
  need at least 8 origins; ``Calibration`` needs 10;
  ``model_confidence_set`` needs ``n_bootstrap >= 100``.
- **Density backtests need a seed.** ``run()`` without ``seed`` gives
  different paths on every run, even when the fitting function seeds the
  posterior sampler, because the predictive shocks are drawn separately.
- **Every origin must produce the same kind of forecast.** A density
  backtest whose hook returns a different number of draws at different
  origins raises ``DimensionError`` rather than padding.
- **Shape detection by size.** The harness recognizes a ``(steps, k, 3)``
  summary by shape. With ``k = 3``, ``horizons = 3`` and exactly 3
  predictive draws, a density forecast is misread as a summary and
  recorded as a point forecast. Use more than 3 draws.
- **Multi-step calibration and Mincer-Zarnowitz over-reject.** Overlapping
  multi-step errors are serially correlated; the summaries of
  ``Calibration`` and ``MincerZarnowitzTest`` say so. Read the formal
  verdicts at horizon one.

See also
--------

- :doc:`bayesian_workflow` -- marginal likelihoods, model averaging and
  stacking of the density forecasters graded here.
- :doc:`bayesian_vars` -- the Bayesian VARs that produce
  ``forecast_paths``.
- :doc:`var` and :doc:`var_extensions` -- the point forecasters.
- :doc:`univariate_linear` and :doc:`volatility` -- the univariate models
  that need a ``predict`` hook.
- :doc:`nuances` -- package-wide conventions and quirks.
- :mod:`cultivars.forecast.backtest`, :mod:`cultivars.forecast.scoring`,
  :mod:`cultivars.forecast.calibration`,
  :mod:`cultivars.forecast.comparison`,
  :mod:`cultivars.forecast.confidence_set`,
  :mod:`cultivars.forecast.conditional`, :mod:`cultivars.forecast.fan`
