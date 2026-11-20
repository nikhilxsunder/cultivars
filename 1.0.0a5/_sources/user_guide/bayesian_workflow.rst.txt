.. meta::
   :description: The Bayesian workflow in cultivars: priors, MCMC convergence, predictive checks, marginal likelihoods, Bayes factors, model averaging and stacking.

The Bayesian workflow
=====================

:mod:`cultivars.bayes` holds what surrounds a Bayesian fit, in the order
an analysis runs (Gelman et al. 2020): state a prior, check what it
implies for the data, fit, check the sampler converged, check the fit
reproduces the data, weigh competing models by their evidence, and combine
them when no model deserves the whole verdict. The models themselves live
with their frequentist counterparts; the Bayesian VARs are described in
:doc:`bayesian_vars`. Out-of-sample scoring of density forecasts is in
:doc:`forecasting`.

.. contents:: On this page
   :local:
   :depth: 1

The family at a glance
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 28 22

   * - Tool
     - Module
     - What it does
     - Use when
   * - Prior classes (``NormalInverseWishartPrior``, ``HorseshoePrior``, ...)
     - :mod:`~cultivars.bayes.priors`
     - State a prior as a record, composable with ``+``
     - Fitting any Bayesian VAR
   * - :func:`~cultivars.bayes.chains.rhat`,
       :func:`~cultivars.bayes.chains.ess_bulk`,
       :func:`~cultivars.bayes.chains.ess_tail`,
       :func:`~cultivars.bayes.chains.mcse`,
       :func:`~cultivars.bayes.chains.geweke`,
       :func:`~cultivars.bayes.chains.convergence`
     - :mod:`~cultivars.bayes.chains`
     - Split-:math:`\widehat R`, effective sample sizes, Monte Carlo error,
       Geweke z, and a report bundling them
     - Before reading any MCMC output as a posterior
   * - :func:`~cultivars.bayes.checks.prior_predictive_check`,
       :func:`~cultivars.bayes.checks.posterior_predictive_check`,
       :func:`~cultivars.bayes.checks.replication_check`
     - :mod:`~cultivars.bayes.checks`
     - Locate the data among data sets simulated from the prior or
       posterior
     - Choosing a prior; checking a fit
   * - :func:`~cultivars.bayes.evidence.marginal_likelihood`,
       :func:`~cultivars.bayes.evidence.compare`,
       :func:`~cultivars.bayes.evidence.modified_harmonic_mean`,
       :func:`~cultivars.bayes.evidence.bridge_sampling`
     - :mod:`~cultivars.bayes.evidence`
     - Log marginal likelihood records, Bayes factors, posterior model
       probabilities
     - Ranking models fitted to the same sample
   * - :func:`~cultivars.bayes.combination.bayesian_model_average`,
       :func:`~cultivars.bayes.combination.stacking`
     - :mod:`~cultivars.bayes.combination`
     - Mixture predictive weighted by posterior probability or by held-out
       log score
     - Forecasting from several models at once

Which results qualify
---------------------

Every tool reads its input by duck typing against a protocol in
``cultivars.engine._core._protocols``. A result of your own that exposes
the same members is accepted.

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Tool
     - Requires
     - Satisfied by
   * - ``result.convergence()``
     - a method on the result (``_ConvergenceMixin``)
     - ``BVARResult``, ``GibbsBVARResult``, ``StudentBVARResult``,
       ``BVARSVResult``, ``HierarchicalBVARResult``, ``FactorSVResult``,
       ``TVPVARResult``, ``TVPVARSVResult``, ``TVPSVARResult``,
       ``StochasticVolatilitySVARResult``, ``SVPosterior``,
       ``UCSVPosterior``, ``PerturbationDSGEPosterior``
   * - :func:`~cultivars.bayes.checks.prior_predictive_check`
     - ``ReplicatingModel``: ``observed`` and ``prior_replications()`` on
       the unfitted model
     - ``BVAR``, ``StudentBVAR``, ``HierarchicalBVAR``, ``SV``, and
       ``GibbsBVAR`` under a static prior without dummy observations
   * - :func:`~cultivars.bayes.checks.posterior_predictive_check`
     - ``ReplicatingResult``: ``observed`` and ``posterior_replications()``
     - ``BVARResult``, ``GibbsBVARResult``, ``StudentBVARResult``,
       ``BVARSVResult``, ``HierarchicalBVARResult``, ``SVPosterior``
   * - :func:`~cultivars.bayes.evidence.marginal_likelihood`
     - a ``marginal_likelihood()`` method returning the record
     - ``BVARResult`` (analytic), ``HierarchicalBVARResult`` (analytic, at
       the hyperparameter mode), ``GibbsBVARResult`` under a static prior
       (Chib), ``PerturbationDSGEPosterior`` (modified harmonic mean)
   * - :func:`~cultivars.bayes.combination.bayesian_model_average`,
       :func:`~cultivars.bayes.combination.stacking`
     - ``PredictiveResult``: ``forecast_paths(steps, *, seed)``
     - the five Bayesian VAR results,
       ``ConditionalForecastResult``, ``ModelCombinationSelection``

The running example is a bivariate VAR(1):

.. testcode::

   import numpy as np

   rng = np.random.default_rng(0)
   A = np.array([[0.5, 0.2], [0.0, 0.6]])
   y = np.zeros((160, 2))
   for t in range(1, 160):
       y[t] = A @ y[t - 1] + rng.standard_normal(2)
   names = ("gdp", "infl")

Priors
------

A prior is a frozen record passed as ``prior=``. It states the coefficient
mean, the coefficient variance, and any dummy observations; the model reads
nothing else. :class:`~cultivars.bayes.priors.NormalInverseWishartPrior`
is the conjugate Minnesota prior with an exact posterior and marginal
likelihood (Kadiyala and Karlsson 1997);
:class:`~cultivars.bayes.priors.IndependentNormalWishartPrior` frees the
cross-equation weight and needs the Gibbs sampler;
:class:`~cultivars.bayes.priors.SumOfCoefficientsPrior` and
:class:`~cultivars.bayes.priors.DummyInitialObservationPrior` add
pseudo-observations (Sims and Zha 1998); the horseshoe, spike-and-slab,
Dirichlet-Laplace and Normal-Gamma priors are adaptive hierarchies for
:class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR`. Priors compose
with ``+``, and the result records the composition in ``prior_label``:

.. testcode::

   from cultivars.bayes.priors import (
       NormalInverseWishartPrior,
       SumOfCoefficientsPrior,
   )
   from cultivars.multivariate.large_dim.bayesian import BVAR

   prior = NormalInverseWishartPrior(tightness=0.1) + SumOfCoefficientsPrior()
   res = BVAR(y, order=1, prior=prior, names=names).fit(n_draws=200, seed=0)
   print(res.prior_label, res.n_dummy)

.. testoutput::

   niw(l1=0.1, l3=1, l4=100) + soc(1) 2

Hyperparameters are validated at ``fit``, not at construction. The full
catalogue, with the Minnesota variance formula and the pairing rules, is in
:doc:`bayesian_vars`.

Prior predictive checks
-----------------------

Before fitting, simulate data sets from the prior and see whether the
data's features are plausible under it (Gabry et al. 2019). For each of
``n_replications`` prior draws a sample of the shape of the effective
sample is simulated, and each discrepancy statistic :math:`T` is located
by the tail probability :math:`p_T = \Pr(T(y^{rep}) \ge T(y))` with ties
counted as one half. The statistics are ``mean``, ``sd``, ``min``,
``max``, ``skewness``, ``kurtosis`` (excess), ``acf1`` and ``arch1``; the
default set is ``("mean", "sd", "acf1", "kurtosis", "arch1")``.

.. testcode::

   from cultivars.bayes.checks import prior_predictive_check

   candidates = {
       "random walk": NormalInverseWishartPrior(),
       "persistence 0.5": NormalInverseWishartPrior(persistence=0.5),
   }
   for label, candidate in candidates.items():
       check = prior_predictive_check(
           BVAR(y, order=1, prior=candidate, names=names),
           statistics=["sd", "acf1"], seed=0,
       )
       bands = check.replicated_quantiles()  # (levels, statistics, series)
       print(label, np.log10(bands[:, 0, 0]).round(1), bands[:, 1, 0].round(2))
   print(check.observed[:, 0].round(2))

.. testoutput::

   random walk [ 0.6  5.6 22.7] [0.68 0.91 0.98]
   persistence 0.5 [-0.2  0.1  1.2] [0.1  0.52 0.89]
   [1.23 0.57]

``replicated_quantiles(levels=(0.05, 0.5, 0.95))`` is
``(len(levels), m, k)``, rows following ``statistics``. For a prior check
read these bands, not the p-values. Under the default prior, centred on a
random walk with ``tightness=0.2``, the median replicated standard
deviation of ``gdp`` is :math:`10^{5.6}` against an observed 1.23: the
prior centres the own lags on one, so many of its draws are explosive and
the replications diverge over 159 periods. The check's ``notes`` report that share (68% of
replications here). Centring the own lags on 0.5 brings the bands around
the data. Model-specific prior settings pass through as keywords
(``df`` for ``StudentBVAR``; ``prior_mu``, ``prior_phi``,
``prior_sigma2`` for ``SV``), and an unknown keyword is refused by name.

Two models on a common sample
-----------------------------

The marginal likelihood scores the observations after the presample, so a
VAR(1) and a VAR(2) fitted to the same array score different data. To
compare them, drop leading rows from the shorter-lag model so both score
the same 158 observations:

.. testcode::

   p1 = BVAR(y[1:], order=1, names=names).fit(n_draws=200, seed=0)
   p2 = BVAR(y, order=2, names=names).fit(n_draws=200, seed=0)
   print(p1.nobs, p2.nobs, p1.n_kept)

.. testoutput::

   158 158 200

Convergence
-----------

MCMC draws are a posterior only once the chain has forgotten its start and
explores the distribution in proportion. :mod:`cultivars.bayes.chains`
follows Vehtari et al. (2021): rank-normalized split-:math:`\widehat R`
with folding, bulk and tail effective sample sizes by Geyer's (1992)
initial monotone sequence, the Monte Carlo standard error of the mean, and
the Geweke (1992) z-score of the first 10% against the last 50% of a
chain. A quantity is flagged when

.. math::

   \widehat R > \texttt{rhat\_tol} = 1.01
   \quad\text{or}\quad
   \min(\widehat S_{\text{bulk}}, \widehat S_{\text{tail}})
   < \texttt{min\_ess} \cdot C, \qquad \texttt{min\_ess} = 100,

for :math:`C` chains.

On a result
~~~~~~~~~~~

Every sampled result has ``convergence(*chains, include=None, exclude=(),
rhat_tol=1.01, min_ess=100.0)``. Further results of the same class from
other seeds are pooled as extra chains, which is what lets
:math:`\widehat R` detect a chain stuck in one mode; a single run is
assessed by its two halves. The Gibbs sampler under an independent
Normal-Wishart prior is a real Markov chain. For the Gibbs samplers
``n_draws`` counts all iterations including ``n_burn``, so
``n_draws=600, n_burn=200`` keeps 400 draws:

.. testcode::

   from cultivars.bayes.priors import IndependentNormalWishartPrior
   from cultivars.multivariate.large_dim.gibbs import GibbsBVAR

   gibbs = GibbsBVAR(y[1:], order=1, prior=IndependentNormalWishartPrior(), names=names)
   run_a = gibbs.fit(n_draws=600, n_burn=200, seed=0)
   run_b = gibbs.fit(n_draws=600, n_burn=200, seed=1)
   report = run_a.convergence(run_b)
   print(report.n_chains, report.n_draws, report.converged)
   print(report.worst(3))
   print(report.rhat.max().round(3), report.ess_bulk.min().round(0))

.. testoutput::

   2 400 True
   ('beta[1,1]', 'beta[0,0]', 'beta[2,1]')
   1.004 618.0

The report is a :class:`~cultivars.bayes.chains.ConvergenceTest`. Its
arrays ``mean``, ``sd``, ``mcse``, ``ess_bulk``, ``ess_tail``, ``rhat``
and ``geweke`` follow ``names`` (``beta[i,j]`` is regressor ``i`` in
equation ``j``; ``sigma[i,j]`` the innovation covariance). ``flagged``
lists failing quantities, ``degenerate`` those whose draws never moved and
that are therefore not judged, ``worst(n)`` the largest
:math:`\widehat R`, and ``summary()`` the table. ``include`` and
``exclude`` select draw arrays by name (``"beta"``, ``"sigma"``, ...).

On draws of your own
~~~~~~~~~~~~~~~~~~~~

The functions take one or more chains, each ``(S,)`` for a scalar or
``(S, d)`` for ``d`` quantities, as positional arguments. Use them for a
function of the draws or a single coefficient pooled across runs, here
the own-lag coefficient of inflation (row 2 is ``infl`` at lag one, column
1 the ``infl`` equation):

.. testcode::

   from cultivars.bayes.chains import convergence, ess_bulk, geweke, mcse, rhat

   persistence_a = run_a.beta_draws[:, 2, 1]
   persistence_b = run_b.beta_draws[:, 2, 1]
   print(round(float(rhat(persistence_a, persistence_b)), 3))
   print(round(float(ess_bulk(persistence_a, persistence_b)), 1))
   print(round(float(mcse(persistence_a, persistence_b)), 4))
   print(geweke(persistence_a, persistence_b).round(2))
   own = convergence(persistence_a, persistence_b, names=["infl.L1"])
   print(own.converged, own.names)

.. testoutput::

   1.002
   761.3
   0.0024
   [-0.53 -0.09]
   True ('infl.L1',)

``geweke`` returns one z-score per chain (and per quantity). With a
single ``(S,)`` chain the scalar functions return a float; with ``(S, d)``
chains they return ``(d,)`` arrays.

Posterior predictive checks
---------------------------

A posterior predictive check simulates replicated samples from retained
posterior draws and asks whether the fitted model reproduces the data's
features (Rubin 1984; Gelman, Meng and Stern 1996). The p-values are not a
test: under a correct model they concentrate near 0.5 rather than being
uniform (Meng 1994). ``extreme(level=0.05)`` names the cells in either
tail, ``adequate()`` is true when there are none. ``pvalues`` is
``(m, k)``, rows following ``statistics``.

.. testcode::

   from cultivars.bayes.checks import posterior_predictive_check

   ppc = posterior_predictive_check(p1, seed=0)
   print(ppc.statistics)
   print(ppc.pvalues.round(2))
   print(ppc.adequate(), ppc.extreme())
   print(round(ppc.pvalue("kurtosis", "gdp"), 3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   ('mean', 'sd', 'acf1', 'kurtosis', 'arch1')
   [[0.54 0.52]
    [0.6  0.65]
    [0.57 0.6 ]
    [0.06 0.62]
    [0.36 0.76]]
   True ()
   0.065

The VAR(1) reproduces every default feature; the excess kurtosis of
``gdp``, with a tail probability of 0.065, comes closest to a tail.

:func:`~cultivars.bayes.checks.replication_check` is the comparison alone,
for replications simulated elsewhere: ``observed`` ``(n, k)`` and
``replicated`` ``(R, n, k)`` with at least 20 replications. Heavy-tailed
data against Gaussian replications fail on ``kurtosis`` and nothing else:

.. testcode::

   from cultivars.bayes.checks import replication_check

   gen = np.random.default_rng(3)
   heavy = gen.standard_t(3, size=(200, 1))
   heavy = (heavy - heavy.mean()) / heavy.std()
   check = replication_check(heavy, gen.standard_normal((300, 200, 1)), names=["r"])
   print(check.extreme())

.. testoutput::

   ('kurtosis[r]',)

Marginal likelihood and Bayes factors
-------------------------------------

The marginal likelihood :math:`m(y) = \int f(y \mid \theta)\pi(\theta)\,d\theta`
penalizes prior mass spent where the data do not want it. Differences of
log values on the same sample are log Bayes factors, and with prior model
probabilities they give posterior model probabilities (Kass and Raftery
1995). :func:`~cultivars.bayes.evidence.marginal_likelihood` dispatches to
the result's own route and returns a
:class:`~cultivars.bayes.evidence.MarginalLikelihoodSelection` with
``log_value``, ``mcse`` (zero when exact), ``method``, ``source``,
``nobs`` and ``n_draws``:

.. testcode::

   from cultivars.bayes.evidence import compare, marginal_likelihood

   ml1, ml2 = marginal_likelihood(p1), marginal_likelihood(p2)
   print(ml1.method, ml1.mcse, ml1.nobs)
   print(round(ml1.log_value, 3), round(ml2.log_value, 3))
   print(round(ml1.log_bayes_factor(ml2), 3), ml1.bayes_factor_strength(ml2))
   table = compare(ml1, ml2, names=("BVAR(1)", "BVAR(2)"))
   print(table)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   analytic 0.0 158
   -482.25 -483.14
   0.891 not worth more than a bare mention
                                          Marginal likelihood comparison
   ===========================================================================================================
   Models:                              2  Prior:                           equal
   -----------------------------------------------------------------------------------------------------------
   model         log ML    mcse   log BF vs best                             evidence   post. prob.     method
   BVAR(1)     -482.250   0.000            0.000                                 best         0.709   analytic
   BVAR(2)     -483.140   0.000           -0.891   not worth more than a bare mention         0.291   analytic
   ===========================================================================================================
   Differences of log ML are log Bayes factors; the strength column is Kass and Raftery's reading of 2 log BF
   against the best model.

``compare(*records, names=None, prior_probabilities=None)`` returns a
summary table whose cells are formatted strings; read numbers from the
records. Its notes warn when records score different numbers of
observations or when a model lies within two Monte Carlo standard errors
of the best.

The Gibbs BVAR under a static prior uses Chib's (1995) identity
:math:`\log m(y) = \log f(y \mid \theta^\ast) + \log\pi(\theta^\ast) -
\log\pi(\theta^\ast \mid y)` with a simulated ordinate, so its record
carries a Monte Carlo error:

.. testcode::

   chib = marginal_likelihood(run_a)
   print(chib.method, chib.nobs, bool(chib.mcse > 0))

.. testoutput::

   chib 158 True

Generic estimators
~~~~~~~~~~~~~~~~~~

For draws from another sampler, two estimators take posterior draws and the
unnormalized log posterior. :func:`~cultivars.bayes.evidence.modified_harmonic_mean`
(Geweke 1999) uses a truncated Gaussian envelope with mass ``tau=0.9``.
:func:`~cultivars.bayes.evidence.bridge_sampling` (Meng and Wong 1996)
also needs the log kernel as a callable, because it scores proposal
draws. Both return the same record type, with ``nobs = 0``. On a
bivariate standard normal with kernel :math:`-\tfrac12\|z\|^2` the true
value is :math:`\log 2\pi \approx 1.838`:

.. testcode::

   from cultivars.bayes.evidence import bridge_sampling, modified_harmonic_mean

   z = np.random.default_rng(2).standard_normal((3000, 2))
   log_kernel = -0.5 * (z**2).sum(axis=1)
   mhm = modified_harmonic_mean(z, log_kernel)
   bridge = bridge_sampling(z, log_kernel, lambda v: -0.5 * float(v @ v), seed=0)
   print(round(mhm.log_value, 2), round(bridge.log_value, 2), bridge.nobs)

.. testoutput::

   1.84 1.84 0

Transform bounded parameters to the real line and include the Jacobian in
the kernel; the plain harmonic mean is not offered, because its variance is
infinite.

Model combination
-----------------

Bayesian model averaging
~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~cultivars.bayes.combination.bayesian_model_average` weights models
by posterior probability, :math:`w_i \propto \pi_i\, m_i(y)`, normalized on
the log scale (Hoeting et al. 1999). It computes the records through
``marginal_likelihood`` unless ``records=`` is given:

.. testcode::

   from cultivars.bayes.combination import bayesian_model_average

   bma = bayesian_model_average(p1, p2, names=("BVAR(1)", "BVAR(2)"))
   print(bma.method, bma.weights.round(3), bma.scores.round(3))
   mixed = bma.forecast_paths(4, seed=0)
   print(mixed.shape, bma.forecast(4, seed=0).shape)

.. testoutput::

   bayesian model average [0.709 0.291] [-482.25 -483.14]
   (200, 4, 2) (4, 2, 3)

The returned
:class:`~cultivars.bayes.combination.ModelCombinationSelection` draws a
model in proportion to its weight and then one of that model's paths.
``forecast_paths(steps=8, *, n_draws=None, seed=None)`` returns as many
mixed paths as the largest model supplies unless ``n_draws`` is given.

Records that score different numbers of observations are refused, which is
why the example fitted on a common sample:

.. testcode::

   from cultivars.exceptions import SpecificationError

   naive = BVAR(y, order=1).fit(n_draws=200, seed=0)
   try:
       bayesian_model_average(naive, p2)
   except SpecificationError as err:
       print(str(err)[:62])

.. testoutput::

   The marginal likelihoods score different numbers of observatio

Stacking
~~~~~~~~

BMA concentrates on one model as the sample grows even when no model is
true. Stacking (Yao, Vehtari, Simpson and Gelman 2018; Geweke and Amisano
2011) instead chooses the simplex weights that maximize the held-out log
score of the mixture,

.. math::

   \max_{w \in \Delta^{M-1}} \frac{1}{T}\sum_{t=1}^{T}
   \log \sum_{m=1}^{M} w_m\, p_m(y_t \mid y_{<t}),

so it needs genuinely held-out log predictive densities, one row per
origin and one column per model. A density backtest supplies them: the
log loss is the negative log predictive density, so negate it. The
backtest's log score is per series; summing across series treats the
series as independent given the model.

.. testcode::

   from cultivars.bayes.combination import stacking
   from cultivars.forecast.backtest import Backtest

   fits = {
       "BVAR(1)": lambda w: BVAR(w, order=1).fit(n_draws=200, seed=0),
       "BVAR(2)": lambda w: BVAR(w, order=2).fit(n_draws=200, seed=0),
   }
   log_density = np.column_stack([
       -Backtest(y, f, start=120, step=2).run(seed=0).losses("log").sum(axis=1)
       for f in fits.values()
   ])
   stacked = stacking(log_density, (p1, p2), names=tuple(fits))
   print(log_density.shape, stacked.n_origins)
   print(stacked.weights.round(3), stacked.scores.round(3))

.. testoutput::

   (20, 2) 20
   [0. 1.] [-3.689 -3.541]

``scores`` holds each model's mean held-out log density. On these 20
origins the BVAR(2) has the higher held-out score and takes the whole
weight, while the marginal likelihood gave the BVAR(1) 0.709: the two
criteria answer different questions, and 20 origins is a short window. The
stacked
weights are optimized on the same origins they are reported on, so the
gain over the best single model is optimistic. The ``results`` argument
supplies the models the mixture predicts from, typically the full-sample
fits.

How the pieces connect
----------------------

- ``forecast_paths`` is the common currency. Every result that combination
  accepts can also be backtested as a density forecaster, scored with
  :class:`~cultivars.forecast.scoring.DensityScore` and fanned with
  :func:`~cultivars.forecast.fan.fan_chart`
  (:doc:`forecasting`). A ``ModelCombinationSelection`` itself has
  ``forecast_paths``, so a combination can be backtested.
- ``losses("log")`` from :class:`~cultivars.forecast.backtest.BacktestResult`
  is the input of :func:`~cultivars.bayes.combination.stacking` after a
  sign change.
- The convergence machinery is shared: ``result.convergence()`` and
  :func:`~cultivars.bayes.chains.convergence` build the same
  ``ConvergenceTest``.
- The marginal-likelihood record is shared by ``compare``,
  ``bayesian_model_average`` and each result's own
  ``marginal_likelihood()``; ``BVARResult`` also exposes the bare float
  ``log_marginal_likelihood``.

Nuances and gotchas
-------------------

- **Conjugate draws can be flagged.** ``BVAR`` draws are i.i.d. from the
  exact posterior, yet with 200 draws a single-chain report flags
  quantities whose split-:math:`\widehat R` exceeds 1.01 by noise; the
  report's notes say so. Pass a second run or raise ``n_draws``.
- **Not every sampled result has a marginal likelihood.**
  ``StudentBVARResult``, ``BVARSVResult``, ``SVPosterior``,
  ``UCSVPosterior``, the TVP models and ``GibbsBVARResult`` under an
  adaptive prior or with dummy observations raise
  ``SpecificationError``. Compare them by held-out score (``stacking`` or a
  backtest). The Gibbs refusal for adaptive priors names the reason:
  Chib's estimator would need reduced runs.
- **The hierarchical BVAR's evidence is at the mode.**
  ``HierarchicalBVARResult.marginal_likelihood()`` returns
  :math:`p(y \mid \hat\theta)` at the hyperparameter mode, reported as
  ``method="analytic"`` with ``mcse=0`` and source ``"BVAR(1), glp(...)"``.
  It is not integrated over the hyperprior, and the mode maximizes it, so a
  Bayes factor against a fixed-prior BVAR favors the hierarchical model by
  construction.
- **Different presamples are not comparable.** ``compare`` warns and
  ``bayesian_model_average`` refuses when records score different
  ``nobs``. Records from ``modified_harmonic_mean`` and
  ``bridge_sampling`` carry ``nobs = 0`` and escape this check.
- **No prior predictive check under adaptive priors or for BVARSV.**
  ``prior_predictive_check`` on a ``GibbsBVAR`` with a horseshoe-type
  prior raises, because the marginal prior of the latent scales has no
  closed form; ``BVARSV`` has no ``prior_replications``.
- **Prior replications start at the observed presample.** For data in
  levels, the ``mean`` statistic reads the drift the prior admits over the
  sample, not a prior on the level.
- **Tail probabilities are resolved to 1/R.** With the default 200
  replications the smallest non-zero p-value is 0.005; at least 20
  replications are required.
- **compare returns strings.** The table's cells are formatted text. Read
  ``log_value`` and ``log_bayes_factor`` from the records.
- **Stacking needs held-out densities.** In-sample log densities reward
  overfitting. ``stacking`` cannot check this; produce the matrix with a
  :class:`~cultivars.forecast.backtest.Backtest`.
- **Two weights near 0.5 are a finding.** Stacking weights do not
  concentrate when two models err in different directions.

See also
--------

- :doc:`bayesian_vars` -- the Bayesian VAR family, its samplers and the
  prior catalogue.
- :doc:`forecasting` -- backtests and proper scores for the density
  forecasts combined here.
- :doc:`volatility` -- the stochastic-volatility posteriors with
  ``convergence()`` and posterior checks.
- :doc:`nuances` -- package-wide conventions.
- :mod:`cultivars.bayes.priors`, :mod:`cultivars.bayes.chains`,
  :mod:`cultivars.bayes.checks`, :mod:`cultivars.bayes.evidence`,
  :mod:`cultivars.bayes.combination`
