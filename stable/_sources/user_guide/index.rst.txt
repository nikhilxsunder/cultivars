.. meta::
   :description: The cultivars user guide: how the library is organized, the conventions every estimator shares, and a page per model family and workflow.

User guide
==========

The user guide explains how to do econometrics with cultivars, one
family of models or one task at a time. Each page gives the model in its
standard notation, runs a verified example, lists the specification
choices that matter, and ends with the nuances that are easy to get
wrong. The :doc:`API reference <../api/index>` has the exhaustive
argument lists; the guide is where to learn which class to reach for and
how its parts fit together.

New to the library? Read :doc:`../getting_started/quickstart` first,
then :doc:`architecture` and :doc:`conventions`. Those two pages explain
the object model and the shape and index rules that every other page
assumes.

Find the right page
-------------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Question
     - Page
   * - How are models, results, schemes and evaluators related?
     - :doc:`architecture`
   * - What shape do inputs take? How are returned arrays indexed?
     - :doc:`conventions`
   * - Which model fits my data and question?
     - :doc:`choosing_a_model`
   * - What behaves differently from what I would expect?
     - :doc:`nuances`
   * - Is my series stationary, cointegrated, nonlinear, broken?
     - :doc:`diagnostics`
   * - AR, ARIMA, seasonal ARIMA, ARFIMA, unobserved components
     - :doc:`univariate_linear`
   * - GARCH, GJR, EGARCH, FIGARCH, stochastic volatility
     - :doc:`volatility`
   * - Threshold, smooth-transition, Markov-switching, neural AR
     - :doc:`univariate_nonlinear`
   * - VAR and VARX: estimation, inference, IRFs, forecasts
     - :doc:`var`
   * - VECM, VARMA, panel, global, mixed-frequency, functional, yield curve
     - :doc:`var_extensions`
   * - Recursive, long-run, sign, narrative, proxy and other identification
     - :doc:`structural_identification`
   * - Bayesian VARs and their priors
     - :doc:`bayesian_vars`
   * - Factor models, FAVAR, sparse and network VARs, spillovers
     - :doc:`large_dimensional`
   * - Threshold, smooth-transition, time-varying and switching VARs
     - :doc:`nonlinear_multivariate`
   * - Kalman, particle and regime-switching filters
     - :doc:`state_space`
   * - Spectra, band-pass and trend filters, wavelets
     - :doc:`spectral`
   * - Backtests, scoring rules, forecast comparison tests
     - :doc:`forecasting`
   * - Convergence, predictive checks, marginal likelihood, model averaging
     - :doc:`bayesian_workflow`

.. toctree::
   :caption: Foundations
   :maxdepth: 1

   architecture
   conventions
   choosing_a_model
   nuances

.. toctree::
   :caption: Before estimation
   :maxdepth: 1

   diagnostics

.. toctree::
   :caption: Univariate models
   :maxdepth: 1

   univariate_linear
   volatility
   univariate_nonlinear

.. toctree::
   :caption: Multivariate models
   :maxdepth: 1

   var
   var_extensions
   structural_identification
   bayesian_vars
   large_dimensional
   nonlinear_multivariate

.. toctree::
   :caption: Engines and frequency-domain views
   :maxdepth: 1

   state_space
   spectral

.. toctree::
   :caption: After estimation
   :maxdepth: 1

   forecasting
   bayesian_workflow
