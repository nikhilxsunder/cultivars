.. meta::
   :description: How cultivars is organized: specifications, frozen result records, the schemes and views that consume them, and the attribute contracts that connect the families.

Architecture
============

cultivars has roughly sixty estimators, and they all fit one small
design. Once you know what kind of object you are holding and which verb
it answers to, you can work out what it connects to without reading its
reference page. This page describes that design: the kinds of objects,
the verbs, the contracts that let one result feed many consumers, and
which result families support which capabilities.

.. contents:: On this page
   :local:
   :depth: 1

The layout follows the analysis
-------------------------------

The subpackages are ordered the way an empirical project runs:

.. code-block:: text

   diagnostics            is the series stationary, cointegrated, nonlinear, broken?
        |
        v
   univariate             one series: ARIMA, GARCH, SV, threshold, switching ...
   multivariate           a system: reduced_form, structural, large_dim,
        |                 nonlinear, regime_switching
        |                 (state_space: the filtering engines several of them use)
        v
   multivariate.structural   what are the shocks?   (identification)
   spectral                  at which frequencies?  (frequency-domain views)
   forecast                  how good are the forecasts?   (evaluation)
   bayes                     is the posterior trustworthy, which model?

Two supporting modules sit beside them. :mod:`cultivars.typing` holds the
literal option sets (``Trend = "n" | "c" | "ct"`` and so on), and
:mod:`cultivars.exceptions` holds the error hierarchy. A third,
:mod:`cultivars.summary`, holds the two records every result renders
through. :mod:`cultivars.engine` contains the private numerical layers;
nothing in it is public API.

Five kinds of objects
---------------------

Every public name is one of five kinds. The kind tells you how to use
the object.

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Kind
     - How to recognize it
     - Examples
   * - **Specification** (a model)
     - Constructed with data and options; has ``fit()``
     - :class:`~cultivars.univariate.box_jenkins.ARIMA`,
       :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`,
       :class:`~cultivars.multivariate.large_dim.bayesian.BVAR`,
       :class:`~cultivars.univariate.conditional_variance.GARCH`
   * - **Result**
     - Returned by ``fit()``; a frozen dataclass named ``...Result``
       (``...Posterior`` for some samplers)
     - ``ARMAResult``, ``VARResult``, ``BVARResult``, ``SVPosterior``
   * - **Consumer**
     - Constructed with a *result* and options; has one action verb
     - :class:`~cultivars.multivariate.structural.zero_restrictions.RecursiveSVAR`,
       :class:`~cultivars.spectral.density.SpectralDensity`,
       :class:`~cultivars.forecast.backtest.Backtest`
   * - **Test or record**
     - Returned by a function such as ``adf()`` or a result method such
       as ``granger_causality()``; has ``statistic``, ``pvalue``,
       ``reject()``, ``summary()``
     - ``UnitRootTest``, ``WaldTest``, ``LinearityTest``
   * - **Option object**
     - Passed *into* a specification to configure it; has no verb of its
       own
     - the priors in :mod:`cultivars.bayes.priors`
       (:class:`~cultivars.bayes.priors.MinnesotaPrior`, ...)

The verbs
---------

Each kind answers to a fixed verb, and the verb is always a method on
an object constructed with everything it needs. Arguments that change
*what* is computed go to the constructor. Arguments that change *how*
it is computed (number of draws, random seed) go to the verb.

.. list-table::
   :header-rows: 1
   :widths: 16 34 50

   * - Verb
     - Called on
     - Returns
   * - ``fit()``
     - every specification (56 classes)
     - a result record
   * - ``identify()``
     - every identification scheme in
       :mod:`cultivars.multivariate.structural`, plus ``TVPSVAR``
     - a structural result, or the accepted set for set-identified
       schemes
   * - ``compute()``
     - views and evaluators: spectral density and causality, spillover
       tables, spectral estimators, calibration, forecast comparison,
       density scores, Beveridge-Nelson
     - a result record for that view
   * - ``run()``
     - :class:`~cultivars.forecast.backtest.Backtest`
     - a ``BacktestResult`` of aligned forecasts and outcomes
   * - ``filter()`` / ``smooth()``
     - state-space engines, trend and band-pass filters
     - filtered or smoothed states, or the filtered series
   * - ``solve()``, ``transform()``
     - :class:`~cultivars.multivariate.reduced_form.closed_global.GVAR`,
       the DSGE perturbation model; :class:`~cultivars.spectral.wavelets.MODWT`
     - the solved system; the wavelet transform

Two families of plain functions sit outside this pattern. Hypothesis
tests (``adf``, ``kpss``, ``terasvirta``, ``bds`` and the rest) take
data and return a test record. Scoring and chain diagnostics
(``pinball_loss``, ``rhat``, ``ess_bulk``) take arrays and return
numbers.

One result, many consumers
--------------------------

A consumer never asks *which* estimator produced its input. It reads a
fixed set of attributes, and any result that has them is accepted.
This duck-typed contract is what lets one fitted VAR be identified,
viewed in the frequency domain, decomposed into spillovers and
backtested, with no adapter code:

.. testcode::

   import numpy as np
   from cultivars.multivariate.reduced_form.vector_autoregression import VAR
   from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR
   from cultivars.spectral.density import SpectralDensity
   from cultivars.multivariate.large_dim.spillover import Spillover
   from cultivars.forecast.backtest import Backtest

   rng = np.random.default_rng(3)
   A = np.array([[0.6, 0.2], [0.0, 0.5]])
   y = np.zeros((300, 2))
   for t in range(1, 300):
       y[t] = A @ y[t - 1] + rng.standard_normal(2)
   names = ("output", "prices")

   var = VAR(y, order=1, names=names).fit()

   svar = RecursiveSVAR(var).identify()           # structural view
   density = SpectralDensity(var).compute()       # frequency-domain view
   spill = Spillover(var, horizon=10).compute()   # connectedness view
   backtest = Backtest(                           # evaluation
       y, lambda w: VAR(w, order=1, names=names).fit(),
       horizons=4, start=200, names=names,
   ).run()

   print(svar.irf(8).shape, density.frequencies.shape, round(spill.total, 1))
   print(backtest.n_origins, backtest.rmse.round(3)[0])

.. testoutput::

   (9, 2, 2) (256,) 3.7
   97 [0.98  0.954]

A result that lacks an attribute a consumer needs is refused at
construction. The message names the missing attribute, so you know
which capability is absent rather than getting a traceback deep inside
the computation:

.. testcode::

   from cultivars.multivariate.large_dim.bayesian import BVAR
   from cultivars.exceptions import SpecificationError

   bvar = BVAR(y, order=1, names=names).fit(n_draws=200, seed=0)
   try:
       RecursiveSVAR(bvar)
   except SpecificationError as error:
       print(str(error)[:111])

.. testoutput::

   identification needs a closed reduced-form result exposing ('names', 'nobs', 'sigma_u', 'resid', 'coefficients'

Here the Bayesian VAR result has no ``ma_representation``, so it is
not a closed system in the sense the identification schemes require.
It does satisfy the *predictive* contract, so the same backtest harness
records it as a density forecaster.

The contracts
-------------

These are the attribute sets the consumers check. They are structural
(duck-typed) protocols. Any object with the attributes qualifies,
including a class you write yourself. Only ``Identification`` is
importable from the public :mod:`cultivars.typing`; the others live in
the private engine today.

.. list-table::
   :header-rows: 1
   :widths: 20 38 42

   * - Contract
     - Requires
     - Accepted by
   * - **Closed system**
     - ``names``, ``nobs``, ``sigma_u``, ``resid``, ``coefficients``,
       ``k_endog``, ``ma_representation(horizon)``
     - every identification scheme;
       :class:`~cultivars.spectral.density.SpectralDensity`;
       :class:`~cultivars.spectral.causality.SpectralCausality`;
       :class:`~cultivars.multivariate.large_dim.spillover.Spillover`
   * - **Structural result**
     - ``source`` (the closed system it came from), ``irf(horizon)``
     - :class:`~cultivars.multivariate.structural.factor_augmented.FactorAugmentedSVAR`,
       Markov-switching structural models
   * - **Predictive result**
     - ``forecast_paths(steps, seed=...)`` returning simulated paths
     - :class:`~cultivars.forecast.backtest.Backtest` (as a density
       forecaster), conditional forecasts, model combination in
       :mod:`cultivars.bayes.combination`
   * - **Replicating result / model**
     - ``observed`` plus ``posterior_replications(...)`` (result) or
       ``prior_replications(...)`` (model)
     - posterior and prior predictive checks in
       :mod:`cultivars.bayes.checks`
   * - **Point forecaster**
     - ``forecast(steps)``
     - :class:`~cultivars.forecast.backtest.Backtest` (as a point
       forecaster); anything else can be backtested through its
       ``predict=`` hook

Shared capabilities
-------------------

Results share behaviour through a small set of capabilities. If a result
has one of these methods, the method means the same thing wherever it
appears:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Capability
     - Members
   * - Summary
     - ``summary()``, ``print(result)``, notebook HTML. On every model
       result and most test records; the Johansen rank test and lag
       selection records use ``to_table()`` instead, and the state-space
       filter records have neither yet.
   * - Comparison
     - ``information_criteria``, ``compare(*others, criterion=...)``,
       ``likelihood_ratio_test(unrestricted)``. On likelihood-based fits
       to the same sample.
   * - Stability
     - ``stability`` / ``stability_check()``, ``is_stationary`` /
       ``is_stable``, ``invertibility`` / ``is_invertible``. Each returns a
       record with the roots and their moduli.
   * - Series export
     - ``to_dict()``, ``to_pandas(index=...)``, ``to_polars()``. On
       univariate results with per-observation output.
   * - Propagation
     - ``irf``, ``fevd``, ``ma_representation``, ``forecast``,
       ``simulate``, ``historical_decomposition``. On closed linear
       systems.
   * - Convergence
     - ``convergence()``, returning chain diagnostics. On sampled
       posteriors.
   * - Replication
     - ``posterior_replications()``, ``simulate()``. On sampled
       posteriors, for predictive checks.

What each family supports
-------------------------

The matrix below is the practical consequence of the contracts. A dot
means the capability is absent. Several of these gaps are on the
roadmap; :doc:`nuances` lists the ones that most often surprise users.

.. list-table::
   :header-rows: 1
   :widths: 34 9 9 9 9 10 10 10

   * - Result family
     - forecast
     - paths
     - simulate
     - irf
     - compare
     - closed
     - export
   * - AR, ARMA/ARIMA/SARIMA
     - .
     - .
     - yes
     - .
     - yes
     - .
     - yes
   * - GARCH family
     - .
     - .
     - yes
     - .
     - yes
     - .
     - yes
   * - SV (QML)
     - variance
     - .
     - yes
     - .
     - yes
     - .
     - yes
   * - SETAR, STAR, Markov-switching AR
     - .
     - .
     - yes
     - .
     - yes
     - .
     - yes
   * - Unobserved components
     - yes
     - .
     - yes
     - .
     - yes
     - .
     - yes
   * - VAR, VARX, VARMA, VECM
     - yes
     - .
     - yes
     - yes
     - yes
     - yes
     - .
   * - Panel VAR, GVAR, MF-VAR, MIDAS-VAR
     - yes
     - .
     - yes
     - yes
     - some
     - yes
     - .
   * - BVAR, Gibbs, Student-t, BVAR-SV
     - yes
     - yes
     - yes
     - yes
     - .
     - .
     - .
   * - TVAR, STVAR
     - yes
     - .
     - .
     - yes
     - yes
     - .
     - .
   * - Time-varying VARs
     - .
     - .
     - .
     - .
     - .
     - .
     - .
   * - Structural results (any scheme)
     - .
     - .
     - .
     - yes
     - .
     - .
     - .

"paths" is ``forecast_paths`` (a predictive sample); "closed" is the
closed-system contract above; "export" is ``to_pandas``. Bayesian
results are compared by marginal likelihood through
:mod:`cultivars.bayes.evidence` rather than ``compare``; see
:doc:`bayesian_workflow`.

How the families are built
--------------------------

The families share engines, which explains why they behave alike:

- **Univariate models** share one likelihood and optimizer layer, so
  every result has ``llf``, ``information_criteria`` and ``compare``,
  and ``params`` is always a dictionary keyed by parameter name. The
  key style follows each family's literature: ``ar.L1`` and ``sigma2``
  for Box-Jenkins models, ``omega``, ``alpha[1]`` and ``beta[1]`` for
  GARCH.
- **Closed linear systems** (VAR, VARX, VARMA, VECM, panel, global and
  mixed-frequency VARs) share one propagation layer, built on the
  moving-average representation. That is why their IRF, FEVD, forecast
  and simulation methods have identical signatures and index
  conventions, and why every identification scheme accepts all of them.
- **Bayesian VARs** share one posterior-draws layer: credible
  intervals, IRF bands, ``forecast_paths``, ``convergence()`` and
  predictive replications behave the same across the conjugate, Gibbs,
  Student-t and stochastic-volatility samplers.
- **State-space engines** in :mod:`cultivars.state_space` sit
  underneath several families: unobserved components and dynamic
  Nelson-Siegel use the Kalman filter, Markov-switching models use the
  Hamilton and Kim filters, and DSGE perturbation uses the nonlinear
  filters. You can call the engines directly; see :doc:`state_space`.

Extending the library
---------------------

Because consumers check attributes rather than classes, your own code
plugs in at the same points:

- **A forecaster from outside the package** can be backtested by
  passing ``predict=`` to :class:`~cultivars.forecast.backtest.Backtest`
  (see :doc:`forecasting`).
- **A reduced form of your own** can be identified, viewed spectrally
  or decomposed into spillovers if it exposes the closed-system
  attributes.
- **A sampler of your own** can be checked with
  :func:`~cultivars.bayes.checks.replication_check` and the chain
  functions in :mod:`cultivars.bayes.chains`, which take plain arrays.

See also
--------

- :doc:`conventions` for shapes, names and index order.
- :doc:`nuances` for the places where families still differ.
- :doc:`../getting_started/concepts` for the result-record design in
  brief.
