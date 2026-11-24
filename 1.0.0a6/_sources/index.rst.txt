.. meta::
   :description: cultivars is a research-grade time series econometrics library for Python: ARIMA to state space, VAR to structural identification, Bayesian and spectral methods on numpy and scipy alone.

cultivars
=========

**Research-grade time series econometrics for Python.** ARIMA to state
space, VAR to structural identification, Bayesian and spectral methods,
on :mod:`numpy` and :mod:`scipy` alone, behind one estimation surface:
construct a model with the data and its specification, call ``fit()``,
read a result that states its verdict and the assumptions behind it.

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting_started/index
      :link-type: doc
      :text-align: center

      Install, fetch a series, fit a first model, read its summary.

   .. grid-item-card:: User guide
      :link: user_guide/index
      :link-type: doc
      :text-align: center

      The mathematics, the models, the decisions, family by family.

   .. grid-item-card:: Examples
      :link: auto_examples/index
      :link-type: doc
      :text-align: center

      Research workflows on live FRED data, as scripts and notebooks.

   .. grid-item-card:: API reference
      :link: api/index
      :link-type: doc
      :text-align: center

      Every public class, function and result, with its sources.

Release status
--------------

.. admonition:: cultivars |release| is a pre-release
   :class: warning

   :bdg-warning:`API: stable in shape, not frozen` :bdg-secondary:`numerics: settling`

   The model-fit-result pattern, the subpackage layout and the names of
   the public classes will not change before 1.0.0. Argument names, result
   fields and summary wording may still be renamed; each rename is listed
   in the :doc:`release notes <project/release_notes>`. The public API
   freezes at ``1.0.0b1``, after which only fixes land. Pin the exact
   version in anything that must not move:

   .. parsed-literal::

      pip install --pre "cultivars==\ |release|\ "

Install
-------

Python 3.12 or newer. The core depends on :mod:`numpy`, :mod:`scipy`
and :mod:`fedfred`; :mod:`pandas` and :mod:`polars` are optional and
switch on dataframe input and ``to_pandas()`` / ``to_polars()`` on
every result.

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install --pre cultivars
         pip install --pre "cultivars[pandas]"      # or [polars]

   .. tab-item:: uv

      .. code-block:: bash

         uv add --prerelease allow cultivars
         uv add --prerelease allow "cultivars[pandas]"

   .. tab-item:: From source

      .. code-block:: bash

         git clone https://github.com/nikhilxsunder/cultivars.git
         cd cultivars
         uv sync --group dev

A conda-forge package follows the first stable release.

In sixty seconds
----------------

Every tool in the library follows the same three lines. The example is
synthetic so it runs without a network connection; the
:doc:`examples <auto_examples/index>` do the same on FRED data.

.. testcode::

   import numpy as np
   from cultivars.multivariate.reduced_form.vector_autoregression import VAR
   from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR

   rng = np.random.default_rng(0)
   A = np.array([[0.5, 0.3], [0.0, 0.4]])          # rate feeds gdp, not the reverse
   y = np.zeros((300, 2))
   for t in range(1, 300):
       y[t] = A @ y[t - 1] + rng.standard_normal(2)

   var = VAR(y, order=1, names=("gdp", "rate")).fit()           # 1. model, fit
   svar = RecursiveSVAR(var, order=("gdp", "rate")).identify()  # 2. identify
   irf = svar.irf(horizon=8)                                    # 3. read the result

   print(var.is_stable, svar.scheme, irf.shape)
   print(var.granger_causality("rate", "gdp").summary())

.. testoutput::
   :hide:
   :options: +ELLIPSIS, +NORMALIZE_WHITESPACE

   True recursive (9, 2, 2)
   ...Wald Test...
   ...reject...
   ...rate does not Granger-cause gdp...

.. code-block:: text

   True recursive (9, 2, 2)
                                      Wald Test
   ================================================================================
   Verdict at 5%:                  reject
   --------------------------------------------------------------------------------
   null                                statistic   df   p-value
   rate does not Granger-cause gdp       15.9250    1    0.0001
   ================================================================================

``print(result)`` renders a summary table: the estimates, a verdict where
there is one, and the notes that qualify it. A result is a frozen
dataclass, so everything in the table is also a field, and ``to_pandas()``
returns the coefficient table as a frame when :mod:`pandas` is installed.

What is inside
--------------

Seven subpackages, 74 modules, roughly one hundred model classes and forty
test functions. The subpackages hold modules only; every public name is
imported from the module that defines it, and the :doc:`API reference
<api/index>` is organised the same way.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :mod:`cultivars.univariate`
      :link: api/generated/cultivars.univariate
      :link-type: doc

      ARIMA and seasonal ARIMA, AR, GARCH, GJR, EGARCH, FIGARCH,
      stochastic volatility and UC-SV, Markov switching, threshold and
      smooth-transition autoregressions, ARFIMA, unobserved components.

   .. grid-item-card:: :mod:`cultivars.multivariate`
      :link: api/generated/cultivars.multivariate
      :link-type: doc

      VAR, VARX, VECM, VARMA, panel, global, functional and mixed-frequency
      VARs, dynamic Nelson-Siegel; zero, sign, narrative, set, proxy,
      heteroskedasticity and non-Gaussian identification; large Bayesian
      VARs, dynamic factor models; Markov-switching, threshold,
      smooth-transition, time-varying and quantile VARs.

   .. grid-item-card:: :mod:`cultivars.state_space`
      :link: api/generated/cultivars.state_space
      :link-type: doc

      The linear Gaussian representation with Kalman filter and smoother,
      extended, unscented and particle filters for nonlinear models, Kim's
      filter and smoother for switching systems. Every state-space model in
      the library exposes its representation through ``state_space``.

   .. grid-item-card:: :mod:`cultivars.bayes`
      :link: api/generated/cultivars.bayes
      :link-type: doc

      Minnesota, Normal-inverse-Wishart, independent Normal-Wishart,
      sum-of-coefficients, dummy-initial-observation, horseshoe,
      Dirichlet-Laplace, normal-gamma and spike-and-slab priors; split-R-hat
      and effective sample sizes; marginal likelihood by bridge sampling and
      modified harmonic mean; prior and posterior predictive checks; model
      averaging and stacking.

   .. grid-item-card:: :mod:`cultivars.diagnostics`
      :link: api/generated/cultivars.diagnostics
      :link-type: doc

      Unit roots (ADF, Phillips-Perron, KPSS, DF-GLS, Ng-Perron,
      Zivot-Andrews), Johansen cointegration rank with simulated p-values,
      structural breaks (sup-Wald, Bai-Perron, CUSUM), nonlinearity (Tsay,
      Terasvirta, BDS, Hansen), seasonality (HEGY, Canova-Hansen), long
      memory (GPH, local Whittle).

   .. grid-item-card:: :mod:`cultivars.forecast`
      :link: api/generated/cultivars.forecast
      :link-type: doc

      A rolling and expanding backtest harness that keeps point forecasts
      and predictive draws aligned with outcomes; Diebold-Mariano with the
      Harvey-Leybourne-Newbold correction, Giacomini-White, Clark-West,
      encompassing and Mincer-Zarnowitz tests; the model confidence set;
      PIT calibration and Berkowitz; CRPS, log and energy scores;
      conditional forecasts and fan charts.

   .. grid-item-card:: :mod:`cultivars.spectral`
      :link: api/generated/cultivars.spectral
      :link-type: doc

      Periodogram, Daniell, Welch and multitaper spectra; the spectral
      density implied by a fitted system; Hodrick-Prescott, Baxter-King,
      Christiano-Fitzgerald, Hamilton, Butterworth and Beveridge-Nelson
      decompositions; Bry-Boschan turning points and concordance; wavelets;
      spectral Granger causality.

A worked analysis
-----------------

The library is built for the workflow of an empirical macro paper, so the
representative example is one. It pulls real GDP, the GDP deflator, the
federal funds rate and the NBER recession indicator from FRED with
:mod:`fedfred` (``pip install fedfred``; a free key from
`fredaccount.stlouisfed.org <https://fredaccount.stlouisfed.org/apikeys>`_,
read from the ``FRED_API_KEY`` environment variable), then runs one tool
from each subpackage on the result.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import fedfred as fd

   from cultivars.diagnostics.unit_roots import adf, kpss
   from cultivars.univariate.box_jenkins import ARIMA
   from cultivars.multivariate.reduced_form.vector_autoregression import VAR
   from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR
   from cultivars.spectral.cycles import TurningPoints, concordance
   from cultivars.spectral.density import SpectralDensity
   from cultivars.spectral.filters import HamiltonFilter
   from cultivars.spectral.periodogram import MultitaperSpectrum

   fred = fd.FredAPI(cache_mode=True)              # key from FRED_API_KEY

   def fetch(series_id: str, **kwargs) -> pd.Series:
       frame = fred.get_series_observations(series_id, observation_start="1960-01-01", **kwargs)
       return frame["value"].astype(float).rename(series_id)

   gdp = fetch("GDPC1")                                               # real GDP, quarterly
   deflator = fetch("GDPDEF")                                         # GDP deflator
   ffr = fetch("FEDFUNDS", frequency="q", aggregation_method="avg")   # monthly -> quarterly mean
   rec = fetch("USREC", frequency="q", aggregation_method="avg")      # share of months in recession

   log_gdp = np.log(gdp).dropna()
   growth = (400 * log_gdp.diff()).dropna()                           # annualised percent
   inflation = (400 * np.log(deflator).diff()).dropna()
   panel = pd.concat({"growth": growth, "inflation": inflation, "ffr": ffr}, axis=1).dropna()

   # 1. Integration order. The two tests have opposite nulls; agreement is the evidence.
   print(adf(log_gdp.to_numpy(), trend="ct").summary())
   print(kpss(growth.to_numpy(), trend="c").summary())

   # 2. A univariate model of growth.
   arma = ARIMA(growth.to_numpy(), order=(2, 0, 1)).fit()
   print(arma.summary())

   # 3. Trend and cycle, one-sided and never revised, then a Bry-Boschan chronology against NBER.
   cycle = HamiltonFilter(horizon=8, lags=4).filter(log_gdp.to_numpy())
   turns = TurningPoints(convention="quarterly").date(log_gdp.to_numpy())
   nber = rec.reindex(log_gdp.index).ffill().fillna(0).to_numpy() >= 0.5
   print(turns.summary())
   print("concordance with NBER:", round(concordance(turns, nber), 3))

   # 4. A VAR, a Granger test, a recursive SVAR and its impulse responses.
   var = VAR(panel.to_numpy(), order=4, names=tuple(panel.columns)).fit()
   print(var.granger_causality("ffr", "growth").summary())
   svar = RecursiveSVAR(var, order=("growth", "inflation", "ffr")).identify()
   irf = svar.irf(horizon=20)                                         # (horizon + 1, variable, shock)
   print(pd.DataFrame(irf[:, :, 2], columns=panel.columns).round(3).head(8))
   print(pd.DataFrame(var.forecast(steps=8), columns=panel.columns).round(2))

   # 5. Does the VAR reproduce the data's spectrum at business-cycle frequencies?
   data_spectrum = MultitaperSpectrum(panel["growth"].to_numpy(), bandwidth=4.0).compute()
   model_spectrum = SpectralDensity(var).compute()
   print(data_spectrum.summary())

Four things the example is showing on purpose. The unit-root tests have
opposite nulls, and agreement between them is the evidence, not either one
alone. :class:`~cultivars.spectral.filters.HamiltonFilter` is one-sided
and never revised, which is what makes it usable for dating in real time;
:class:`~cultivars.spectral.filters.HodrickPrescottFilter` is there for
comparability with the literature and its summary says why to prefer the
former for inference. :class:`~cultivars.multivariate.structural.zero_restrictions.RecursiveSVAR`
states the Cholesky ordering as an identifying assumption in its summary,
and permuting ``order`` changes the responses; that is the point, not an
artifact. :class:`~cultivars.spectral.periodogram.MultitaperSpectrum`
is model-free, so overlaying it on the
:class:`~cultivars.spectral.density.SpectralDensity` of the fitted VAR
reads misspecification frequency by frequency.

How the library is designed
---------------------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: One pattern

      A model is a validated specification; ``fit()``, ``identify()``,
      ``filter()``, ``compute()`` or ``date()`` returns a frozen result.
      Results carry integer positions and leave your index alone; inputs
      are anything :func:`numpy.asarray` accepts.

   .. grid-item-card:: Refusals, not silent failure

      Malformed shapes raise :class:`~cultivars.exceptions.DimensionError`,
      inconsistent specifications
      :class:`~cultivars.exceptions.SpecificationError`, and estimators
      that cannot certify their answer
      :class:`~cultivars.exceptions.NumericalError`, each with the numbers
      that triggered it.

   .. grid-item-card:: Summaries that argue

      Every summary names its identifying assumption, its p-value source
      (analytic, simulated or bootstrapped), the convention behind its
      degrees of freedom, and the critique a referee would raise.

   .. grid-item-card:: Documented to the equation

      Each class docstring carries the model in notation, the estimator,
      the references with page numbers and a doctest that runs in CI. The
      :doc:`user guide <user_guide/index>` is where the families are
      compared; the :doc:`engine pages <engine/index>` are where the
      numerics are explained.

Citing
------

If cultivars contributes to published work, cite the software; a Zenodo
DOI will be attached at the first stable release.

.. parsed-literal::

   @software{sunder_cultivars,
     author  = {Sunder, Nikhil},
     title   = {cultivars: research-grade time series econometrics for Python},
     year    = {2026},
     version = {\ |release|\ },
     url     = {https://github.com/nikhilxsunder/cultivars},
     license = {MIT}
   }

The :github:`CITATION.cff` file in the repository carries the same entry in
a form GitHub and Zenodo read directly.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   user_guide/index
   auto_examples/index
   api/index
   engine/index
   project/index
