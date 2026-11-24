.. meta::
   :description: A ten-minute tour of cultivars: unit-root tests, ARIMA fitting and comparison, then a VAR with order selection, forecasts, Granger causality and impulse responses.

Quickstart
==========

This page runs one small analysis from start to finish: decide whether
a series needs differencing, fit and compare two ARIMA models for it,
then move to a three-variable VAR and read forecasts, causality tests
and impulse responses from it. Every block runs as written, and the
output under it is what it prints. The data are simulated so that the
page needs no network connection and the true model is known; the
:doc:`examples <../auto_examples/index>` repeat the same steps on FRED
data.

Imports always name the leaf module, never the subpackage. The path
says which family a model belongs to, and the subpackages re-export
nothing.

A series and its unit root
--------------------------

Simulate quarterly log output: growth is an AR(1) around 0.5% a
quarter, and the level is its running sum.

.. testcode::

   import numpy as np

   rng = np.random.default_rng(7)
   growth = np.zeros(240)
   for t in range(1, 240):
       growth[t] = 0.5 + 0.4 * (growth[t - 1] - 0.5) + 0.8 * rng.standard_normal()
   log_gdp = 100 + np.cumsum(growth)

The augmented Dickey-Fuller test keeps the unit-root null for the level
and rejects it for the first difference, so the level is integrated of
order one:

.. testcode::

   from cultivars.diagnostics.unit_roots import adf

   level = adf(log_gdp)
   change = adf(np.diff(log_gdp))
   print(level.reject(), change.reject())

.. testoutput::

   False True

Every test returns a record. ``reject()`` gives the verdict at 5% by
default (``reject(alpha=0.01)`` for another level), and ``summary()``
prints the whole result: the null, the lag order the test chose, the
critical values, and the tail the rejection lies in.

.. testcode::

   print(change.summary())

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

                                       ADF Test
   ================================================================================
   Null:                        unit root  Trend:                               c
   Lags:                                0  Method:                   aic (max 14)
   Observations:                      238  Verdict at 5%:        reject unit root
   --------------------------------------------------------------------------------
   statistic        value   p-value       1%       5%      10%
   ADF           -10.3191    0.0000   -3.458   -2.874   -2.573
   ================================================================================
   Rejection lies in the lower tail.

Fit and compare ARIMA models
----------------------------

A model is constructed with the data and the specification, and
``fit()`` returns the result. Fit an ARIMA(1, 1, 0) and an
ARIMA(1, 1, 1) to the level:

.. testcode::

   from cultivars.univariate.box_jenkins import ARIMA

   ar1 = ARIMA(log_gdp, order=(1, 1, 0)).fit()
   arma11 = ARIMA(log_gdp, order=(1, 1, 1)).fit()
   print(ar1.summary())

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

                                ARIMA(1, 1, 0) Results
   ================================================================================
   Model:                  ARIMA(1, 1, 0)  Log-likelihood:               -259.446
   Trend:                               c  AIC:                           524.891
   Exog regressors:                     0  BIC:                           535.321
   Observations:                      239  HQIC:                          529.094
   --------------------------------------------------------------------------------
                coef
   const      0.3081
   ar.L1      0.3764
   sigma2     0.5130
   ================================================================================
   Stationary: True   max |AR root| = 0.3764
   Standard errors are not yet available for this estimator.

The AR coefficient is near its true 0.4, and the constant is near the
true intercept of the differenced series, the mean growth rate scaled
by :math:`1 - \phi`: :math:`0.5 \times (1 - 0.4) = 0.3`. The closing
notes are part of the result. Here they report the
stationarity check on the AR polynomial, and that this estimator does
not report standard errors yet.

Results fitted to the same data compare directly. ``compare`` ranks
them on one information criterion and reports each model's distance
from the best:

.. testcode::

   print(ar1.compare(arma11, criterion="bic"))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

                               Model comparison by BIC
   ================================================================================
   Observations:                      239  Models:                              2
   --------------------------------------------------------------------------------
   model                   llf   k       BIC    dBIC
   ARIMA(1, 1, 0)     -259.446   3   535.321   0.000
   ARIMA(1, 1, 1)     -258.918   4   539.743   4.422
   ================================================================================
   Ranked best-first; the final column is the gap to the best model.

The MA term buys half a log-likelihood point for one parameter, and BIC
prefers the AR(1), which is the true model. The numbers behind any
summary are attributes of the result:

.. testcode::

   print({name: round(value, 3) for name, value in ar1.params.items()})
   print(ar1.is_stationary, round(ar1.information_criteria.bic, 3))

.. testoutput::

   {'const': 0.308, 'ar.L1': 0.376, 'sigma2': 0.513}
   True 535.321

A vector autoregression
-----------------------

Now three series that move together: output growth, inflation and a
policy rate, with the rate feeding back on growth one quarter later.

.. testcode::

   A = np.array([
       [0.5, 0.1, -0.3],   # growth
       [0.1, 0.6, 0.0],    # inflation
       [0.2, 0.3, 0.7],    # rate
   ])
   y = np.zeros((240, 3))
   for t in range(1, 240):
       y[t] = A @ y[t - 1] + rng.standard_normal(3) * [0.8, 0.4, 0.3]
   names = ("growth", "inflation", "rate")

Choose the lag order before fitting. Every candidate order is scored on
the same sample, so the criteria can be compared down each column:

.. testcode::

   from cultivars.multivariate.reduced_form.vector_autoregression import VAR

   selection = VAR(y, order=1, names=names).lag_order_selection(max_lags=6)
   print(selection.to_table())

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

                                 Lag order selection
   ================================================================================
   Observations:                      234  Maximum lags:                        6
   Consensus:                           1
   --------------------------------------------------------------------------------
   p          AIC        BIC       HQIC       FPE
   0      -2.5409    -2.4966    -2.5230    0.0788
   1     -4.9782*   -4.8010*   -4.9068*   0.0069*
   2      -4.9430    -4.6329    -4.8179    0.0071
   3      -4.9424    -4.4994    -4.7638    0.0071
   4      -4.8776    -4.3018    -4.6454    0.0076
   5      -4.8691    -4.1604    -4.5834    0.0077
   6      -4.8348    -3.9931    -4.4954    0.0080
   ================================================================================
   An asterisk marks each criterion's minimum. Every order was scored on the same
   sample, so entries within a column are comparable; FPE is on a determinant
   scale, so entries across columns are not.

All four criteria pick one lag. Fit the VAR(1), check that it is stable,
and forecast four quarters ahead. Rows are horizons and columns follow
``names``:

.. testcode::

   var = VAR(y, order=1, names=names).fit()
   print(var.is_stable)
   print(var.forecast(steps=4).round(3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   True
   [[ 0.431  0.469  0.367]
    [ 0.18   0.325  0.453]
    [ 0.014  0.209  0.42 ]
    [-0.074  0.116  0.329]]

Causality and identification
----------------------------

Does the policy rate help predict growth? The Wald test of the lag
exclusion restriction answers in the same record form as the unit-root
test:

.. testcode::

   print(var.granger_causality("rate", "growth").summary())

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

                                      Wald Test
   ================================================================================
   Verdict at 5%:                  reject
   --------------------------------------------------------------------------------
   null                                   statistic   df   p-value
   rate does not Granger-cause growth       10.5894    1    0.0011
   ================================================================================

Impulse responses need structural shocks, and those need an
identifying assumption. Identification is a second step that takes the
fitted reduced form and returns a structural result. The fitted VAR is
never changed. A recursive scheme orders the rate last, so the policy
rate can respond to growth and inflation within the quarter while
neither responds to the rate until the next one:

.. testcode::

   from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR

   svar = RecursiveSVAR(var, order=("growth", "inflation", "rate")).identify()
   irf = svar.irf(horizon=12)
   print(irf.shape)
   print(irf[[0, 4, 8], :, 2].round(3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   (13, 3, 3)
   [[ 0.     0.     0.273]
    [-0.056 -0.012  0.027]
    [-0.002 -0.008 -0.014]]

The response array is indexed ``[horizon, variable, shock]``, from
impact to twelve quarters. The slice above is the rate shock at
horizons 0, 4 and 8. Growth and inflation do not move on impact, as the
ordering requires; growth falls after a rate rise, as the simulated
system says it should; and the responses die out because the VAR is
stable. The forecast-error variance decomposition uses the same index
order:

.. testcode::

   fevd = svar.fevd(horizon=12)
   print(fevd[12, 0].round(3))

.. testoutput::

   [0.951 0.019 0.03 ]

At three years, 95% of the uncertainty in growth comes from its own
shock and 3% from the rate.

Where next
----------

- :doc:`concepts` explains the pattern behind every step above: what a
  result record holds, how tests and summaries are shaped, how data are
  passed in, and what the error classes mean.
- The :doc:`examples <../auto_examples/index>` run identification by
  long-run, sign, narrative and proxy restrictions, Bayesian VARs,
  regime switching and forecast evaluation on FRED data.
- The :doc:`API reference <../api/index>` documents each model with its
  mathematics and references.
