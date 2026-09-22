cultivars
=========

Research-grade time series econometrics for Python 3.12+: ARIMA to state
space, VAR to structural identification, Bayesian and spectral methods,
on numpy and scipy alone.

.. warning::

   ``cultivars`` |release| is a **prerelease**. The public API is stable
   in shape but not yet frozen; expect argument and result-field renames
   before 1.0.0. Pin the exact version if you depend on it.

Install
-------

.. code-block:: bash

   pip install --pre cultivars

``pandas`` and ``polars`` are optional; ``pip install --pre "cultivars[pandas]"``
enables ``result.to_pandas()`` and dataframe input.

Quick start
-----------

The example pulls real GDP, the GDP deflator, the federal funds rate and the
NBER recession indicator from FRED with `fedfred <https://nikhilxsunder.github.io/fedfred/>`_
(``pip install fedfred``; a free API key from
`research.stlouisfed.org <https://fredaccount.stlouisfed.org/apikeys>`_)
and runs one representative tool from each subpackage.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import fedfred as fd

   from cultivars.diagnostics import adf, kpss
   from cultivars.univariate import ARIMA
   from cultivars.multivariate import VAR, RecursiveSVAR
   from cultivars.spectral import (
       HamiltonFilter, MultitaperSpectrum, SpectralDensity, TurningPoints, concordance,
   )

   fred = fd.FredAPI("YOUR_FRED_API_KEY", cache_mode=True)

   def fetch(series_id: str, **kwargs) -> pd.Series:
       df = fred.get_series_observations(series_id, observation_start="1960-01-01", **kwargs)
       return df["value"].astype(float).rename(series_id)

   gdp  = fetch("GDPC1")                                               # real GDP, quarterly
   defl = fetch("GDPDEF")                                              # GDP deflator
   ffr  = fetch("FEDFUNDS", frequency="q", aggregation_method="avg")   # monthly -> quarterly mean
   rec  = fetch("USREC",    frequency="q", aggregation_method="avg")   # share of months in recession

   log_gdp = np.log(gdp).dropna()
   growth  = (400 * log_gdp.diff()).dropna()          # annualised, %
   infl    = (400 * np.log(defl).diff()).dropna()
   panel   = pd.concat({"growth": growth, "inflation": infl, "ffr": ffr}, axis=1, sort=True).dropna()

   # 1. Integration order: ADF keeps the unit root on the level, KPSS keeps stationarity on growth.
   print(adf(log_gdp.values, trend="ct").summary())
   print(kpss(growth.values, trend="c").summary())

   # 2. A univariate model of growth.
   arma = ARIMA(growth.values, order=(2, 0, 1)).fit()
   print(arma.summary())

   # 3. Trend and cycle, one-sided (Hamilton 2018), then a Bry-Boschan chronology against NBER.
   cycle = HamiltonFilter(horizon=8, lags=4).filter(log_gdp.values)
   turns = TurningPoints(convention="quarterly").date(log_gdp.values)
   nber  = rec.reindex(log_gdp.index).ffill().fillna(0).values >= 0.5
   print(turns.summary())
   print("concordance with NBER:", round(concordance(turns, nber), 3))

   # 4. A VAR, a Granger test, a recursive SVAR and its impulse responses.
   var  = VAR(panel.values, order=4, trend="c", names=list(panel.columns)).fit()
   print(var.granger_causality("ffr", "growth").summary())
   svar = RecursiveSVAR(var, order=["growth", "inflation", "ffr"]).identify()
   irf  = svar.irf(horizon=20)                        # (horizon + 1, variable, shock)
   print(pd.DataFrame(irf[:, :, 2], columns=panel.columns).round(3).head(8))   # responses to an ffr shock
   print(pd.DataFrame(var.forecast(steps=8), columns=panel.columns).round(2))

   # 5. Does the VAR reproduce the data's spectrum at business-cycle frequencies?
   data_spec  = MultitaperSpectrum(panel["growth"].values, bandwidth=4.0).compute()
   model_spec = SpectralDensity(var).compute()
   print(data_spec.summary())

Every model follows the same pattern: construct with the data and the
specification, call ``fit()`` (or ``identify()``, ``filter()``,
``compute()``, ``date()`` for the non-likelihood tools), and read the
result object. ``print(res)`` renders a summary table with a verdict and
the notes that qualify it; ``res.to_pandas()`` returns the coefficient
table as a frame when ``pandas`` is installed, ``res.to_polars()`` when
``polars`` is. Inputs are anything ``numpy.asarray`` accepts, so a
``pandas.Series`` or a ``polars`` column passes straight through; results
carry integer positions, and you keep the index.

A few things the example is deliberately showing:

- The two unit-root tests have opposite nulls, and agreement between them
  is the evidence, not either one alone.
- ``HamiltonFilter`` is one-sided and never revised; ``HodrickPrescottFilter``
  is there for comparability with the literature, and its summary says why
  you should prefer the former for inference.
- ``RecursiveSVAR`` states the Cholesky ordering as an identifying
  assumption in its summary; permuting ``order`` changes the impulse
  responses, and that is the point.
- ``MultitaperSpectrum`` is model-free; overlaying it on ``SpectralDensity``
  of the fitted VAR reads misspecification frequency by frequency.

.. toctree::
   :maxdepth: 2
   :hidden:

   api/index
