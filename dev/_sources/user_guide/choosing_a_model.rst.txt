.. meta::
   :description: A decision guide from your data and research question to the cultivars class that fits it, with the user-guide page to read next.

Choosing a model
================

Start from the question, not the class list. The tables below map
common research questions to the class that answers them and to the
page that explains it. Run the :doc:`diagnostics` first: whether a
series has a unit root, is cointegrated or behaves nonlinearly decides
half of the choices below.

.. contents:: On this page
   :local:
   :depth: 1

One series
----------

.. list-table::
   :header-rows: 1
   :widths: 42 30 28

   * - You want to ...
     - Use
     - Read
   * - model the conditional mean of a stationary series
     - :class:`~cultivars.univariate.autoregression.AR`,
       :class:`~cultivars.univariate.box_jenkins.ARMA`
     - :doc:`univariate_linear`
   * - model an integrated or seasonal series
     - :class:`~cultivars.univariate.box_jenkins.ARIMA`,
       :class:`~cultivars.univariate.box_jenkins.SARIMA`
       (``...X`` variants for regressors)
     - :doc:`univariate_linear`
   * - estimate long memory, :math:`d` between 0 and 1/2
     - :class:`~cultivars.univariate.fractional_integration.ARFIMA`;
       semiparametric :math:`d` in :mod:`~cultivars.diagnostics.long_memory`
     - :doc:`univariate_linear`
   * - decompose into trend, cycle and seasonal components
     - :class:`~cultivars.univariate.unobserved_components.UnobservedComponents`;
       filters in :mod:`cultivars.spectral`
     - :doc:`univariate_linear`, :doc:`spectral`
   * - model time-varying volatility of returns
     - :class:`~cultivars.univariate.conditional_variance.GARCH`,
       :class:`~cultivars.univariate.conditional_variance.GJR`,
       :class:`~cultivars.univariate.conditional_variance.EGARCH`,
       :class:`~cultivars.univariate.conditional_variance.FIGARCH`
     - :doc:`volatility`
   * - treat volatility as a latent process
     - :class:`~cultivars.univariate.stochastic_volatility.SV`,
       :class:`~cultivars.univariate.stochastic_volatility.UCSV`
     - :doc:`volatility`
   * - let dynamics switch at an observed threshold
     - :class:`~cultivars.univariate.threshold.SETAR`
     - :doc:`univariate_nonlinear`
   * - let dynamics change smoothly with a transition variable
     - :class:`~cultivars.univariate.smooth_transition.LSTAR`,
       :class:`~cultivars.univariate.smooth_transition.ESTAR`
     - :doc:`univariate_nonlinear`
   * - let dynamics switch with an unobserved Markov state
     - :class:`~cultivars.univariate.regime_switching.MarkovSwitchingAR`
     - :doc:`univariate_nonlinear`
   * - fit a flexible nonlinear mean
     - :class:`~cultivars.univariate.mean_function.ARNN`,
       :class:`~cultivars.univariate.mean_function.TARNN`
     - :doc:`univariate_nonlinear`

Several series, reduced form
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 42 30 28

   * - You want to ...
     - Use
     - Read
   * - model joint dynamics of a few stationary series
     - :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
     - :doc:`var`
   * - condition on series you do not model (oil, foreign rates)
     - :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARX`
     - :doc:`var`
   * - model cointegrated series in levels
     - :class:`~cultivars.multivariate.reduced_form.error_correction.VECM`
     - :doc:`var_extensions`
   * - allow moving-average terms
     - :class:`~cultivars.multivariate.reduced_form.moving_average.VARMA`
     - :doc:`var_extensions`
   * - pool the same system across many units
     - :class:`~cultivars.multivariate.reduced_form.panel.PanelVAR`
     - :doc:`var_extensions`
   * - link country models through trade weights
     - :class:`~cultivars.multivariate.reduced_form.closed_global.GVAR`
     - :doc:`var_extensions`
   * - mix monthly and quarterly series
     - :class:`~cultivars.multivariate.reduced_form.mixed_frequency.MFVAR`,
       :class:`~cultivars.multivariate.reduced_form.mixed_frequency.MIDASVAR`
     - :doc:`var_extensions`
   * - model a whole curve each period (yields, distributions)
     - :class:`~cultivars.multivariate.reduced_form.term_structure.DynamicNelsonSiegel`,
       :class:`~cultivars.multivariate.reduced_form.functional.FunctionalVAR`
     - :doc:`var_extensions`

Many series
-----------

.. list-table::
   :header-rows: 1
   :widths: 42 30 28

   * - You want to ...
     - Use
     - Read
   * - shrink a medium or large VAR toward a prior
     - :class:`~cultivars.multivariate.large_dim.bayesian.BVAR`,
       :class:`~cultivars.multivariate.large_dim.hierarchical.HierarchicalBVAR`
     - :doc:`bayesian_vars`
   * - use a non-conjugate or global-local shrinkage prior
     - :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR`
     - :doc:`bayesian_vars`
   * - handle fat tails or time-varying volatility in a BVAR
     - :class:`~cultivars.multivariate.large_dim.student.StudentBVAR`,
       :class:`~cultivars.multivariate.large_dim.volatility.BVARSV`
     - :doc:`bayesian_vars`
   * - summarize a large panel with a few factors
     - :class:`~cultivars.multivariate.large_dim.dynamic_factor.DFM`,
       :class:`~cultivars.multivariate.large_dim.factor_augmented.FAVAR`
     - :doc:`large_dimensional`
   * - select a sparse lag structure or a network
     - :class:`~cultivars.multivariate.large_dim.sparse.SparseVAR`,
       :class:`~cultivars.multivariate.large_dim.graphical.GraphicalVAR`
     - :doc:`large_dimensional`
   * - measure connectedness between markets
     - :class:`~cultivars.multivariate.large_dim.spillover.Spillover`
     - :doc:`large_dimensional`

Structural questions
--------------------

Pick the scheme by the identifying information you are willing to
assume. All of them take a fitted reduced form; see
:doc:`structural_identification`.

.. list-table::
   :header-rows: 1
   :widths: 42 30 28

   * - Your identifying assumption
     - Use
     - Read
   * - a timing order: some variables do not respond within the period
     - :class:`~cultivars.multivariate.structural.zero_restrictions.RecursiveSVAR`
     - :doc:`structural_identification`
   * - some shocks have no long-run effect on some variables
     - :class:`~cultivars.multivariate.structural.zero_restrictions.LongRunSVAR`
     - :doc:`structural_identification`
   * - a general pattern of zeros on impact (A/B model)
     - :class:`~cultivars.multivariate.structural.zero_restrictions.ShortRunSVAR`,
       :class:`~cultivars.multivariate.structural.zero_restrictions.MixedSVAR`
     - :doc:`structural_identification`
   * - only the signs of responses
     - :class:`~cultivars.multivariate.structural.sign_restrictions.SignRestrictedSVAR`,
       :class:`~cultivars.multivariate.structural.set_identification.SetIdentifiedSVAR`
     - :doc:`structural_identification`
   * - signs plus facts about particular historical episodes
     - ``NarrativeSignRestrictedSVAR`` in
       :mod:`~cultivars.multivariate.structural.sign_restrictions`
     - :doc:`structural_identification`
   * - an external instrument correlated with one shock
     - :class:`~cultivars.multivariate.structural.external_instruments.ProxySVAR`
     - :doc:`structural_identification`
   * - volatility regimes or non-Gaussian shocks do the identifying
     - :class:`~cultivars.multivariate.structural.heteroskedacity.HeteroskedasticSVAR`,
       :class:`~cultivars.multivariate.structural.non_gaussian.NonGaussianSVAR`
     - :doc:`structural_identification`
   * - shocks whose effects drift over time
     - ``TVPSVAR`` in :mod:`~cultivars.multivariate.nonlinear.time_varying`
     - :doc:`nonlinear_multivariate`

Nonlinear and changing systems
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 42 30 28

   * - You want to ...
     - Use
     - Read
   * - a VAR whose dynamics depend on a threshold variable
     - :class:`~cultivars.multivariate.nonlinear.threshold.TVAR`,
       :class:`~cultivars.multivariate.nonlinear.smooth_transition.STVAR`
     - :doc:`nonlinear_multivariate`
   * - coefficients that drift every period
     - :class:`~cultivars.multivariate.nonlinear.time_varying.TVPVAR`,
       :class:`~cultivars.multivariate.nonlinear.time_varying.TVPVARSV`
     - :doc:`nonlinear_multivariate`
   * - regimes driven by a hidden Markov chain
     - :class:`~cultivars.multivariate.regime_switching.markov_switching.MarkovSwitchingVAR`
     - :doc:`nonlinear_multivariate`
   * - responses at quantiles of the distribution
     - :class:`~cultivars.multivariate.nonlinear.quantile.QVAR`
     - :doc:`nonlinear_multivariate`

Frequency-domain questions
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 42 30 28

   * - You want to ...
     - Use
     - Read
   * - estimate a spectrum from data
     - :mod:`cultivars.spectral.periodogram`
     - :doc:`spectral`
   * - read the spectrum, coherence and phase a fitted system implies
     - :class:`~cultivars.spectral.density.SpectralDensity`
     - :doc:`spectral`
   * - test causality at business-cycle frequencies
     - :class:`~cultivars.spectral.causality.SpectralCausality`
     - :doc:`spectral`
   * - extract a cycle (HP, band-pass, Butterworth)
     - :mod:`cultivars.spectral.filters`, :mod:`cultivars.spectral.band_pass`
     - :doc:`spectral`
   * - localize co-movement in time and frequency
     - :mod:`cultivars.spectral.wavelets`
     - :doc:`spectral`

Point estimates or posteriors?
------------------------------

Most reduced forms come in a frequentist and a Bayesian version, and
the choice changes what you get back:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - Frequentist (``VAR``, ``ARIMA``, ...)
     - Bayesian (``BVAR``, ``GibbsBVAR``, ``SV``, ...)
   * - Fit returns
     - point estimates, sampling covariance
     - posterior draws, credible intervals
   * - Model choice
     - ``compare`` on information criteria, LR tests
     - marginal likelihood, model averaging (:doc:`bayesian_workflow`)
   * - Forecasts
     - ``forecast`` (point)
     - ``forecast_paths`` (predictive sample)
   * - Identification
     - every scheme accepts the result
     - not yet: Bayesian results are not closed systems
   * - Cost
     - milliseconds
     - seconds to minutes; set ``n_draws`` and ``seed``

A ``prior=`` argument on
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
sits between the two. It shrinks the point estimate toward the prior
mean but keeps a VAR result, which can still be identified; see
:doc:`bayesian_vars`.
