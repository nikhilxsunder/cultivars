.. meta::
   :description: Behaviour in cultivars that differs from what users of other packages expect, and the known issues in the current pre-release, each with what to do instead.

Nuances and known issues
========================

This page lists the places where cultivars behaves differently from what
experience with other packages would suggest, and the defects known in
the current pre-release. Every item says what to do instead and links
to the page that demonstrates it.

The first part describes deliberate design that will stay. The second
part, :ref:`known-issues`, lists bugs and gaps found while writing this
guide against ``1.0.0a4``; items leave the list as they are fixed, and
the :doc:`release notes <../project/release_notes>` record each fix.

.. contents:: On this page
   :local:
   :depth: 2

By design
---------

Data and samples
~~~~~~~~~~~~~~~~

**Column names are not read.** ``VAR(df)`` labels its variables ``y1``,
``y2``, ... even when ``df`` is a labelled dataframe. Pass
``names=tuple(df.columns)``. See :doc:`conventions`.

**Outputs cover the effective sample.** ``resid``, ``fittedvalues`` and
every per-period decomposition start after the lags and differences, and
the default ``to_pandas()`` index runs from 0 on that sample, not from
the original positions. Drop the first ``p`` (or ``d``) dates before
attaching them. See :doc:`conventions`.

**Comparisons need a common sample.** ``compare`` and
``likelihood_ratio_test`` raise
:class:`~cultivars.exceptions.DimensionError` when two results have
different ``nobs``, as a VAR(1) and a VAR(2) on the same data do. Trim
the data so both effective samples coincide. See :doc:`diagnostics`.

**Missing values are refused, not dropped.** Only the state-space
engines accept gaps, and they differ in which pattern they accept: the
linear Gaussian engine takes gaps in any element, the nonlinear and Kim
engines only whole missing rows, the Hamilton filter none. See
:doc:`state_space`.

Results and what they offer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Univariate results do not forecast.** ``AR``, ``ARMA``/``ARIMA``/
``SARIMA``, the GARCH family, ``ARFIMA``, ``SETAR``, ``STAR`` and the
Markov-switching AR return results with ``simulate`` but no
``forecast``. Unobserved components forecast, returning
``(mean, std)``. ``SV`` has ``forecast_variance``. To evaluate a
univariate model out of sample, backtest it through the ``predict=``
hook; see :doc:`forecasting`. :doc:`volatility` shows how to compute
GARCH variance forecasts from the fitted recursion.

**Bayesian VARs are not closed systems.** ``BVAR``, ``GibbsBVAR``,
``StudentBVAR`` and ``BVARSV`` results have no ``ma_representation``.
The identification schemes, ``SpectralDensity``, ``SpectralCausality``
and ``Spillover`` therefore refuse them. For a shrunk estimate that can
still be identified, pass a prior to
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
(``VAR(y, order=4, prior=MinnesotaPrior())``). See :doc:`bayesian_vars`
and :doc:`architecture`.

**Identification is constructed, not requested.** Write
``RecursiveSVAR(var, order=...).identify()``. The method
``var.identify(scheme)`` exists on reduced-form results but does not
work with the shipped schemes; see :ref:`known-issues`.

**Several result families are deliberately narrow.** ``SparseVAR`` and
``GraphicalVAR`` results have no ``forecast`` or ``irf``. The
time-varying VARs have no ``forecast`` and, except ``TVPSVAR``, no
``irf``. ``MarkovSwitchingVAR`` has no ``forecast``, ``simulate`` or
``irf``. :doc:`architecture` has the full capability matrix.

**Summaries can be uneven.** Box-Jenkins summaries state that standard
errors are not yet available. The state-space filter records have no
``summary()`` and print as full dataclass reprs, so print the fields you
need. The Johansen rank test and lag-order selection records use
``to_table()`` instead of ``summary()``.

Specification defaults
~~~~~~~~~~~~~~~~~~~~~~

**``lag_order_selection()`` without ``max_lags`` can select absurd
orders.** The default scores every order the sample supports, and on
short samples the criteria can then favour very long lags. Always pass
``max_lags``. The method scores *linear, unshrunk* VARs even when called
from a nonlinear or Bayesian model, and it refuses orders a wide system
cannot fit by least squares. See :doc:`var` and :doc:`bayesian_vars`.

**``n_draws`` means different things.** It is the number of draws kept
for ``BVAR`` and ``HierarchicalBVAR``. For ``GibbsBVAR``,
``StudentBVAR``, ``BVARSV`` and ``FactorSV`` it is total iterations,
which must exceed ``n_burn`` (500 for ``GibbsBVAR``, 1000 for the
others). Default thinning is 1 for ``GibbsBVAR`` and ``FactorSV`` and 2
for ``StudentBVAR`` and ``BVARSV``. Read ``n_kept`` on the result. See :doc:`conventions`.

**Default seeds differ.** Sign-restriction samplers default to
``seed=0``. Posterior samplers and ``StochasticVolatilitySVAR`` default
to ``None`` and are not reproducible unless you pass a seed.

**Forecast horizons default differently.** ``VARResult.forecast()``
defaults to one step; ``BVARResult.forecast()`` and ``forecast_paths()``
to eight. Pass ``steps`` explicitly.

**``BVAR`` is conjugate.** It refuses priors that break conjugacy, such
as a Minnesota prior with ``cross_equation`` different from one. Use
:class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR` for the
sampled posterior under that prior, or ``VAR(prior=...)`` for the
shrunk point estimate. See :doc:`bayesian_vars`.

**Parameter counts differ by family.** SETAR and STAR leave the
innovation variance out of ``n_params``; AR, VARs, threshold VARs and
Markov-switching models include it. Information criteria are therefore
offset by one parameter across those families. See :doc:`conventions`.

Structural analysis
~~~~~~~~~~~~~~~~~~~

**"One standard deviation" depends on the scheme.** Proxy and
non-Gaussian schemes scale shocks to the full sample, heteroskedastic
identification to the first regime, structural stochastic volatility to
average volatility. Sign conventions differ too: recursive and long-run
schemes make the diagonal positive, others make each column's largest
entry positive. Compare schemes on responses normalized to a unit
impact. See :doc:`structural_identification`.

**Unrestricted columns in a sign-restricted set have no fixed sign.**
A shock you did not restrict can enter the accepted set with either
orientation, so its quantile bands mix both. Restrict at least one sign
of every shock you intend to read.

**A sign-restricted set can be smaller than requested.** When the
attempt budget runs out with some but not all draws accepted, the result
holds fewer draws and no warning is raised. Check ``n_accepted`` and
``acceptance_rate`` on the result. When no draw is
accepted the sampler raises
:class:`~cultivars.exceptions.SpecificationError`. The remedies are to
shorten the horizons, loosen a sign or raise ``max_attempts``.
``SetIdentifiedSVAR`` signals an empty set with
:class:`~cultivars.exceptions.NumericalError` instead.

**Narrative events are indexed on the effective sample.** Subtract the
VAR order from the raw position of the event date. See
:doc:`structural_identification`.

**A VECM reports itself stable.** ``VECMResult.is_stable`` is ``True``
although the system has unit roots by construction, so long-run and
mixed schemes accept it at construction and refuse only at
``identify()``.

Forecast evaluation
~~~~~~~~~~~~~~~~~~~

**Point backtests have no quantiles.** On a point forecaster,
``BacktestResult.quantile(tau)`` and the pinball loss return the point
forecast for every ``tau``. Use a density forecaster
(``forecast_paths``) for quantile evaluation. See :doc:`forecasting`.

**Pass one series to comparison tests.**
:class:`~cultivars.forecast.comparison.ForecastComparison` flattens its
inputs. Passing ``losses()`` for all series at once, with shape
``(T, k)``, tests ``T * k`` interleaved values. Select one series with
``losses(..., name=...)``.

Frequency domain
~~~~~~~~~~~~~~~~

**Phase signs differ by module.** In
:class:`~cultivars.spectral.density.SpectralDensity`, a negative phase
means the cause leads. In the wavelet results, a positive phase and
``lead`` mean the first series leads.

**Band arguments are named differently.** The band-pass filters take
``low``/``high``, ``SpectralCausalityResult.band`` takes
``low_period``/``high_period``, and the wavelet results take
``scale_band=(low, high)``. All are in periods.
See :doc:`spectral`.

.. _known-issues:

Known issues in 1.0.0a4
-----------------------

These are defects, not design. Each has a workaround.

.. list-table::
   :header-rows: 1
   :widths: 34 36 30

   * - Issue
     - Effect
     - Do instead
   * - ``ARIMA`` with :math:`d > 0`: ``to_pandas()``
     - ``observed`` holds levels while ``fitted`` and ``resid`` are on
       the differenced scale
     - difference the data yourself and fit ``ARMA``, or read
       ``resid`` and ``fittedvalues`` directly
   * - ``result.identify(scheme)``
     - raises ``TypeError``
     - ``Scheme(result, ...).identify()``
   * - Christiano-Fitzgerald band-pass filter
     - endpoint weights are wrong; the cycle leaks trend throughout the
       sample
     - use :class:`~cultivars.spectral.band_pass.BaxterKingFilter`
   * - ``SpectralDensity`` and ``SpectralCausality`` on a ``VARMAResult``
     - the moving-average block is ignored
     - fit a long VAR instead, or compute the spectrum from the VARMA
       polynomials yourself
   * - ``HeteroskedasticSVAR`` results
     - ``fevd``, ``structural_shocks`` and ``historical_decomposition``
       are mis-scaled (FEVD rows do not sum to one)
     - use ``irf`` only, normalized to a unit impact
   * - ``simulate`` on VARMA, VARX, MIDAS-VAR, panel VAR
     - VARMA ignores the MA block; VARX and MIDAS-VAR set the regressors
       to zero; panel VAR raises
     - simulate from the coefficients yourself
   * - Panel VAR ``historical_decomposition``
     - crosses unit boundaries
     - decompose unit by unit
   * - ``VECM(..., exog=...)``
     - accepted, but returns a result whose methods fail
     - use :class:`~cultivars.multivariate.reduced_form.error_correction.VECMX`
   * - Adaptive priors (horseshoe, normal-gamma, Dirichlet-Laplace) in
       ``BVAR``, ``StudentBVAR``, ``BVARSV`` and ``VAR``
     - accepted, but used as a fixed Gaussian prior at their starting
       scales
     - use :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR`,
       which samples the scales
   * - ``HierarchicalBVAR`` marginal likelihood
     - evaluated at the hyperparameter mode, which favours it in
       ``compare`` and model averaging
     - compare it only with care, or fix the hyperparameters
   * - ``kpss(...).reject(alpha)``
     - the p-value is clipped to [0.01, 0.10], so verdicts at
       ``alpha`` below 0.01 or above 0.10 are wrong
     - compare ``statistic`` with ``critical_values``
   * - VAR portmanteau test
     - degrees of freedom also subtract the deterministic columns
     - recompute the p-value with :math:`k^2(h - p)` degrees of freedom
   * - Markov-switching ``converged`` flag
     - can report non-convergence for a fit that converged, and
       ``n_init=1`` still depends on ``seed``
     - check ``llf`` across seeds rather than the flag
   * - GARCH-family persistence
     - the optimizer caps persistence just below 0.999, so
       ``is_covariance_stationary`` is always ``True``
     - check the persistence value itself near the cap
   * - Box-Jenkins optimizer
     - one starting point; ARMA fits with near-cancelling roots can
       stop at a local optimum
     - check ``invertibility`` and ``stability`` for near-cancelling
       roots; refit a smaller order
   * - ``Backtest`` on unobserved components or dynamic Nelson-Siegel
     - their ``(mean, std)`` forecast is recorded as a two-draw density
     - pass ``predict=lambda res, h, seed: res.forecast(h)[0]``
   * - ``NarrativeSignRestrictedSVAR``
     - missing from ``__all__``, so absent from the API reference
     - import it from
       :mod:`~cultivars.multivariate.structural.sign_restrictions`
       directly; it works

See also
--------

- :doc:`conventions` for the rules these items are exceptions to.
- :doc:`architecture` for the capability matrix.
- `Open issues <https://github.com/nikhilxsunder/cultivars/issues>`_ to
  report anything not listed here.
