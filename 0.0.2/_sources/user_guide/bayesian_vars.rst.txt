.. meta::
   :description: Bayesian VARs in cultivars: prior objects and composition, the conjugate BVAR, hierarchical, Gibbs, Student-t and stochastic-volatility BVARs.

Bayesian VARs
=============

A Bayesian VAR replaces the least-squares estimate of a vector
autoregression with a posterior under a prior that shrinks the
coefficients toward a stated belief, usually independent random walks.
Shrinkage is what makes a system with more regressors than observations
estimable, and the posterior draws give credible bands for every
propagated object: coefficients, impulse responses and the predictive
distribution. This page covers the prior objects in
:mod:`~cultivars.bayes.priors`, the five Bayesian VAR estimators in
:mod:`cultivars.multivariate.large_dim`, and the ``prior=`` argument of the
point-estimate :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`.
Penalized and factor models for wide panels are on
:doc:`large_dimensional`; the unrestricted VAR is on :doc:`var`; marginal
likelihoods, predictive checks and model averaging in general are on
:doc:`bayesian_workflow`.

.. contents:: On this page
   :local:
   :depth: 1

The family at a glance
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 30 30 30

   * - Class
     - Module
     - Model in one line
     - Use when
   * - :class:`~cultivars.multivariate.large_dim.bayesian.BVAR`
     - :mod:`~cultivars.multivariate.large_dim.bayesian`
     - Conjugate Normal-inverse-Wishart prior; exact posterior, independent
       draws, closed-form marginal likelihood.
     - The default. Any width, including more regressors than rows; prior
       choice by marginal likelihood.
   * - :class:`~cultivars.multivariate.large_dim.hierarchical.HierarchicalBVAR`
     - :mod:`~cultivars.multivariate.large_dim.hierarchical`
     - The conjugate BVAR with its tightness (and dummy-prior looseness)
       estimated, after Giannone, Lenza and Primiceri (2015).
     - You do not want to defend a hand-picked tightness.
   * - :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR`
     - :mod:`~cultivars.multivariate.large_dim.gibbs`
     - Independent Normal-Wishart prior, two-block Gibbs sampler; also runs
       the adaptive shrinkage priors.
     - You want Litterman's cross-equation shrinkage, a horseshoe or a
       spike-and-slab selection; moderate width.
   * - :class:`~cultivars.multivariate.large_dim.student.StudentBVAR`
     - :mod:`~cultivars.multivariate.large_dim.student`
     - Conjugate prior, Student-t innovations through a scale mixture.
     - A few dates (2008, 2020) should not own the error bands.
   * - :class:`~cultivars.multivariate.large_dim.volatility.BVARSV`
     - :mod:`~cultivars.multivariate.large_dim.volatility`
     - Minnesota prior, random-walk stochastic volatility on triangularized
       innovations (Carriero, Clark and Marcellino 2019).
     - Innovation variances drift over the sample.
   * - :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
       with ``prior=``
     - :mod:`~cultivars.multivariate.reduced_form.vector_autoregression`
     - Posterior-mean point estimate under any prior; the full
       ``VARResult`` surface.
     - You need shrinkage *and* the closed-system tools: structural
       identification, ``fevd``, spillover tables.

All examples on this page share one simulated system: three stationary
variables with non-zero means and three cross-variable links.

.. testcode::

   import numpy as np
   from cultivars.multivariate.large_dim.bayesian import BVAR

   rng = np.random.default_rng(7)
   A = np.array([[0.6, 0.2, 0.0],
                 [0.0, 0.5, 0.1],
                 [0.1, 0.0, 0.4]])
   y = np.zeros((200, 3))
   for t in range(1, 200):
       y[t] = A @ y[t - 1] + rng.standard_normal(3)
   y += np.array([2.0, 3.0, 4.0])
   names = ("output", "prices", "rate")

Priors are objects
------------------

A prior is a frozen dataclass from :mod:`~cultivars.bayes.priors`. It
states three things about the ``(w, k)`` coefficient matrix :math:`B`
(rows in design order: deterministic terms, then lag blocks; columns are
equations): its mean (``coefficient_mean``), its variance
(``coefficient_variance``), and any pseudo-observations stacked under the
sample (``dummy_observations``). Each hook takes a
:class:`~cultivars.bayes.priors.PriorContext` describing the sample, so a
prior knows nothing about the estimator that reads it.

The Minnesota variance of the coefficient on variable :math:`j` at lag
:math:`l` in equation :math:`i`, with :math:`s_i` the residual scale of a
univariate autoregression of variable :math:`i`, is (Litterman 1986)

.. math::

   \operatorname{Var}(A_{l,ij}) =
   \Bigl(\frac{\lambda_1}{l^{\lambda_3}}\Bigr)^{2}
   \times
   \begin{cases} 1 & i = j \\ \lambda_2^2\, s_i^2 / s_j^2 & i \ne j \end{cases},
   \qquad
   \operatorname{Var}(c_i) = (\lambda_1 \lambda_4 s_i)^2 .

.. list-table:: The prior classes
   :header-rows: 1
   :widths: 30 34 36

   * - Class (``prior_label``)
     - Hyperparameters (defaults)
     - What it is
   * - :class:`~cultivars.bayes.priors.NormalInverseWishartPrior`
       (``niw(...)``)
     - ``tightness=0.2`` (:math:`\lambda_1`), ``decay=1.0``
       (:math:`\lambda_3`), ``exogenous=100.0`` (:math:`\lambda_4`),
       ``persistence=1.0``
     - Minnesota moments with :math:`\lambda_2` pinned at one, so the
       variance factors as :math:`\Sigma \otimes \Omega`. Conjugate.
   * - :class:`~cultivars.bayes.priors.MinnesotaPrior`
       (``minnesota(...)``)
     - as above plus ``cross_equation=0.5`` (:math:`\lambda_2`) and
       ``sum_of_coefficients=None`` (:math:`\lambda_5`)
     - Litterman's full prior. Not conjugate unless
       ``cross_equation=1``.
   * - :class:`~cultivars.bayes.priors.IndependentNormalWishartPrior`
       (``inw(...)``)
     - same fields as ``MinnesotaPrior``
     - The same moments, stated independently of :math:`\Sigma`; the
       default of ``GibbsBVAR``.
   * - :class:`~cultivars.bayes.priors.SumOfCoefficientsPrior`
       (``soc(...)``)
     - ``tightness=1.0``
     - :math:`k` dummy rows: own lags sum to one (Doan, Litterman and
       Sims 1984). Improper alone; a component.
   * - :class:`~cultivars.bayes.priors.DummyInitialObservationPrior`
       (``dio(...)``)
     - ``tightness=1.0``
     - One dummy row: the system starts at its presample mean (Sims 1993).
       Improper alone; a component.
   * - :class:`~cultivars.bayes.priors.HorseshoePrior`,
       :class:`~cultivars.bayes.priors.SpikeAndSlabPrior`,
       :class:`~cultivars.bayes.priors.DirichletLaplacePrior`,
       :class:`~cultivars.bayes.priors.NormalGammaPrior`
     - ``deterministic_scale=10.0``; plus ``spike``, ``slab``,
       ``inclusion`` / ``concentration`` / ``shape``
     - Adaptive hierarchies with latent per-coefficient scales, centred
       at zero. Sampled only by ``GibbsBVAR``.
   * - :class:`~cultivars.bayes.priors.NoPrior` (``none``)
     - none
     - The identity of composition; on ``VAR`` it means least squares.

Priors compose with ``+``: means and variances combine by precision
weighting, dummy blocks stack, and ``NoPrior`` changes nothing. The sum
of a conjugate base and the two dummy components is the prior of Sims
and Zha (1998) and Giannone, Lenza and Primiceri (2015):

.. testcode::

   from cultivars.bayes.priors import (
       DummyInitialObservationPrior,
       MinnesotaPrior,
       NormalInverseWishartPrior,
       PriorContext,
       SumOfCoefficientsPrior,
   )

   base = NormalInverseWishartPrior(tightness=0.1)
   print(base)
   sims_zha = base + SumOfCoefficientsPrior() + DummyInitialObservationPrior()
   fitted = BVAR(y, order=2, names=names, prior=sims_zha).fit(n_draws=50, seed=0)
   print(fitted.prior_label)
   print(fitted.n_dummy)

.. testoutput::

   NormalInverseWishartPrior(tightness=0.1, decay=1.0, exogenous=100.0, persistence=1.0)
   niw(l1=0.1, l3=1, l4=100) + soc(1) + dio(1)
   4

The moments can be read directly on a hand-built context. With two
variables at one lag and residual scales 1 and 4, the cross-lag
variances of the conjugate prior carry only the scale ratio, while the
Minnesota default multiplies them by :math:`\lambda_2^2 = 0.25`:

.. testcode::

   ctx = PriorContext(k_endog=2, order=1, scales=np.array([1.0, 4.0]),
                      presample_mean=np.zeros(2))
   print(NormalInverseWishartPrior().coefficient_variance(ctx).round(4))
   print(MinnesotaPrior().coefficient_variance(ctx).round(4))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   [[4.0e+02 6.4e+03]
    [4.0e-02 6.4e-01]
    [2.5e-03 4.0e-02]]
   [[4.0e+02 6.4e+03]
    [4.0e-02 1.6e-01]
    [6.0e-04 4.0e-02]]

Hyperparameters are validated when the prior is used, not when it is
built: ``NormalInverseWishartPrior(tightness=-1)`` constructs, and so
does a model holding it; the ``SpecificationError`` surfaces at ``fit``.

Which model accepts which prior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verified against the code; "refused" means a ``SpecificationError`` at
``fit``.

.. list-table::
   :header-rows: 1
   :widths: 28 12 12 12 12 12 12

   * - Prior
     - ``BVAR``
     - ``StudentBVAR``
     - ``GibbsBVAR``
     - ``BVARSV``
     - ``VAR``
     - Notes
   * - ``NormalInverseWishartPrior``
     - default
     - default
     - yes
     - yes
     - yes
     -
   * - ``MinnesotaPrior(cross_equation != 1)``
     - refused
     - refused
     - yes
     - default
     - yes
     - breaks the Kronecker factorization
   * - ``IndependentNormalWishartPrior``
     - refused
     - refused
     - default
     - yes
     - yes
     - accepted by the conjugate models only at ``cross_equation=1``
   * - ``+ SumOfCoefficientsPrior``, ``+ DummyInitialObservationPrior``
     - yes
     - yes
     - yes
     - refused
     - yes
     - a dummy row has no date to be volatility-weighted by
   * - dummy component alone
     - refused
     - refused
     - refused
     - refused
     - yes
     - improper: infinite coefficient variances
   * - adaptive (horseshoe, ...)
     - see note
     - see note
     - yes
     - see note
     - see note
     - only ``GibbsBVAR`` samples the scales; see
       `Nuances and gotchas`_
   * - ``NoPrior``
     - refused
     - refused
     - refused
     - refused
     - least squares
     -

The conjugate BVAR
------------------

:class:`~cultivars.multivariate.large_dim.bayesian.BVAR` estimates

.. math::

   y_t = c + \sum_{l=1}^{p} A_l\, y_{t-l} + u_t,
   \qquad u_t \sim \mathcal{N}(0, \Sigma),

stacked as :math:`Y = XB + U`, under

.. math::

   \Sigma \sim \mathcal{IW}(S_0, k + 2),
   \qquad
   \operatorname{vec}(B) \mid \Sigma \sim
   \mathcal{N}\bigl(\operatorname{vec}(B_0),\ \Sigma \otimes \Omega\bigr),

with :math:`S_0 = \operatorname{diag}(s_1^2, \ldots, s_k^2)`. The
Kronecker form makes the posterior Normal-inverse-Wishart in closed form
(Kadiyala and Karlsson 1997; Banbura, Giannone and Reichlin 2010):
:math:`K = \Omega^{-1} + X^\top X`,
:math:`\bar B = K^{-1}(\Omega^{-1} B_0 + X^\top Y)`, and the draws are
independent samples, not a Markov chain.

.. testcode::

   res = BVAR(y, order=2, names=names).fit(n_draws=500, seed=0)
   print(res.nobs, res.n_kept, res.n_dummy, res.posterior_df)
   print(res.beta_draws.shape, res.sigma_draws.shape, res.coefficients.shape)
   print(res.credible_interval("output", "prices.L1").round(3))

.. testoutput::

   198 500 0 203.0
   (500, 7, 3) (500, 3, 3) (2, 3, 3)
   [0.074 0.138 0.209]

``credible_interval(equation, regressor)`` returns
``(16th percentile, mean, 84th percentile)`` of one coefficient across
draws; regressor labels are ``"const"``, ``"trend"`` and
``"{name}.L{lag}"``. Every three-number summary on Bayesian VAR results
uses the same 68% convention. The summary reports the own-first-lag
coefficient of each equation and the log marginal likelihood:

.. testcode::

   print(res)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

                                   BVAR(2) Results
   ================================================================================
   Model:                          BVAR(2)  Prior:        niw(l1=0.2, l3=1, l4=100)
   Draws:                              500  Variables:                            3
   Observations:                       198  Dummy rows:                           0
   Trend:                                c  log ML:                        -837.410
   --------------------------------------------------------------------------------
   equation     own L1 mean       68% interval
   output            0.5572   [0.4946, 0.6193]
   prices            0.6073   [0.5409, 0.6735]
   rate              0.4081   [0.3388, 0.4743]
   ================================================================================
   The posterior is exact Normal-inverse-Wishart; draws are independent, not a
   Markov chain, so no convergence diagnostics apply.
   log_marginal_likelihood is the sample's, with the dummy-observation contribution
   divided out; differences across priors are log Bayes factors, and hyperparameter
   choice should maximize it (Giannone-Lenza-Primiceri).
   Posterior probability of stability: 1.00. Mass near the unit circle is the
   random-walk prior speaking, not a defect.
   No llf, parameter count, or information criteria are reported: a posterior has
   none, and the marginal likelihood is the honest replacement.
   forecast() is the full posterior predictive -- parameter and shock uncertainty
   jointly.

Specification choices
~~~~~~~~~~~~~~~~~~~~~

``order`` (required)
   Positive integer; ``order=0`` is refused because the Minnesota prior
   centres the own first lag.
``prior`` (default ``NormalInverseWishartPrior()``)
   Any conjugacy-compatible prior or composition; see the table above.
``trend`` (default ``"c"``)
   ``"n"``, ``"c"`` or ``"ct"``. The deterministic block gets standard
   deviation :math:`\lambda_1 \lambda_4 s_i`, effectively unshrunk.
``names`` (default ``y1 ... yk``)
   Also the recursive ordering of every orthogonalized impulse response.
``fit(n_draws=1000, seed=None)``
   ``n_draws`` independent draws, all kept (no burn-in, no thinning).
   The posterior mean, ``sigma_u`` and ``log_marginal_likelihood`` do not
   depend on ``n_draws`` or ``seed``; only Monte Carlo bands do.

What the result gives you
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~cultivars.multivariate.large_dim.bayesian.BVARResult` carries
``beta_mean`` (``(w, k)``, exact :math:`\bar B`), ``coefficients``
(``(p, k, k)``, ``coefficients[l-1][i, j]`` is the effect of variable
``j`` at lag ``l`` on equation ``i``), ``deterministic``, ``sigma_u``
(exact posterior mean :math:`\bar S / (\bar\nu - k - 1)`), ``beta_draws``,
``sigma_draws``, ``resid`` and ``fittedvalues`` at the posterior mean,
``nobs`` (:math:`T - p`, dummy rows excluded), ``n_dummy``,
``posterior_df``, ``log_marginal_likelihood``, ``prior_label`` and
``stable_share`` (share of draws with companion spectral radius below
one). Methods: ``credible_interval``, ``irf``, ``irf_bands``,
``forecast``, ``forecast_paths``, ``marginal_likelihood``,
``convergence``, ``simulate``, ``posterior_replications`` and
``summary``. There is no ``llf``, ``aic`` or ``n_params``.

Impulse responses with bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``irf(horizon=20, orthogonalized=True, cumulative=False)`` is the
posterior mean of the responses computed draw by draw, each draw
rotated by its own Cholesky factor; ``irf_bands`` adds the 16th and 84th
percentiles on a last axis. Entry ``[h, i, j]`` is the response of
variable ``i`` at lead ``h`` to shock ``j``.

.. testcode::

   bands = res.irf_bands(12)
   print(bands.shape)
   print(bands[[0, 1, 4], 0, 1].round(3))   # output's response to a prices shock

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   (13, 3, 3, 3)
   [[0.    0.    0.   ]
    [0.029 0.089 0.151]
    [0.026 0.092 0.156]]

Rows are leads 0, 1 and 4; columns are ``(16th, mean, 84th)``. The
impact response is exactly zero because ``output`` is ordered before
``prices``: the recursive ordering of ``names`` is an identifying
assumption. The posterior mean response is not the response at the
posterior mean coefficients beyond lead one, because :math:`\Psi_h` is a polynomial of
degree :math:`h` in the lag matrices.

Forecasts and the predictive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``forecast_paths(steps, seed=)`` simulates one future per draw, each
under its own :math:`(B, \Sigma)` and shocks, so the paths sample the
joint posterior predictive. ``forecast`` is their pointwise
``(16th, mean, 84th)`` summary, shape ``(steps, k, 3)``.

.. testcode::

   paths = res.forecast_paths(4, seed=1)
   fc = res.forecast(4, seed=1)
   print(paths.shape, fc.shape)
   print(fc[:, 0].round(2))
   # a path-dependent probability: output stays below 2 for four quarters
   print(round(float((paths[:, :, 0] < 2.0).all(axis=1).mean()), 3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   (500, 4, 3) (4, 3, 3)
   [[1.08 2.01 2.96]
    [0.95 1.97 3.04]
    [0.58 1.81 2.98]
    [0.5  1.66 2.85]]
   0.186

Pass the paths to :func:`~cultivars.forecast.fan.fan_chart` for quantile
fans or to the scoring tools on :doc:`forecasting`. Without a ``seed``
every call draws fresh shocks and returns different numbers.

Marginal likelihood and prior choice
------------------------------------

``log_marginal_likelihood`` is :math:`\log p(Y)` in closed form, with any
dummy-observation content divided out, :math:`\log p(Y \mid Y_d) =
\log p(Y, Y_d) - \log p(Y_d)`, so priors with and without dummies are
comparable. Differences across priors on the same sample are log Bayes
factors (Giannone, Lenza and Primiceri 2015). ``marginal_likelihood()``
wraps the number in the record that
:func:`~cultivars.bayes.evidence.compare` ranks:

.. testcode::

   from cultivars.bayes.evidence import compare

   candidates = {
       "tight": NormalInverseWishartPrior(tightness=0.05),
       "default": NormalInverseWishartPrior(),
       "loose": NormalInverseWishartPrior(tightness=1.0),
       "stationary": NormalInverseWishartPrior(persistence=0.0),
       "sims-zha": NormalInverseWishartPrior() + SumOfCoefficientsPrior()
                   + DummyInitialObservationPrior(),
   }
   records = [
       BVAR(y, order=2, names=names, prior=p).fit(n_draws=10, seed=0).marginal_likelihood()
       for p in candidates.values()
   ]
   print(records[1].method, records[1].mcse, records[1].source)
   table = compare(*records, names=tuple(candidates))
   for row in table.rows:
       print(row[0], row[1], row[3], row[5])

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   analytic 0.0 BVAR(2), niw(l1=0.2, l3=1, l4=100)
   stationary -834.748 0.000 0.931
   default -837.410 -2.662 0.065
   sims-zha -840.227 -5.480 0.004
   loose -854.985 -20.237 0.000
   tight -875.533 -40.785 0.000

On stationary data centred away from the random walk, the
``persistence=0.0`` prior wins and the dummy-observation prior, which
pulls toward unit roots, loses. The full treatment of evidence,
predictive checks and model averaging is on :doc:`bayesian_workflow`.

Comparing lag orders
~~~~~~~~~~~~~~~~~~~~

A fit of order :math:`p` scores :math:`T - p` observations, so marginal
likelihoods of different orders are not Bayes factors as fitted;
``compare`` adds a note to the table rather than refusing. Trim the
sample so every order conditions on the same presample:

.. testcode::

   p_max = 4
   by_order = [
       BVAR(y[p_max - p:], order=p).fit(n_draws=10, seed=0).marginal_likelihood()
       for p in range(1, p_max + 1)
   ]
   print({r.nobs for r in by_order})
   print(compare(*by_order).rows[0][0])

.. testoutput::

   {196}
   BVAR(1), niw(l1=0.2, l3=1, l4=100)

``BVAR.lag_order_selection(max_lags)`` exists but scores unshrunk
least-squares information criteria, so it is bounded by the same sample
constraints as the unrestricted VAR (see `Nuances and gotchas`_).

Refused priors
~~~~~~~~~~~~~~

The conjugate model checks that the prior variance factors as
:math:`\sigma_i^2 \omega_j` and refuses a prior that does not, rather
than pinning :math:`\lambda_2` itself:

.. testcode::

   from cultivars.exceptions import SpecificationError

   try:
       BVAR(y, order=2, prior=MinnesotaPrior()).fit(n_draws=10)
   except SpecificationError as err:
       print(str(err)[:68])

.. testoutput::

   the prior variance does not factor as sigma_i^2 * omega_col, so Norm

Use ``GibbsBVAR`` for a sampled posterior under that prior, or
``VAR(..., prior=MinnesotaPrior())`` for the shrunk point estimate.

Estimated tightness: HierarchicalBVAR
-------------------------------------

:class:`~cultivars.multivariate.large_dim.hierarchical.HierarchicalBVAR`
treats :math:`\theta = (\lambda, \mu, \delta)` (overall tightness,
sum-of-coefficients and dummy-initial-observation *looseness*, larger is
looser, as in Giannone, Lenza and Primiceri 2015) as unknowns with Gamma
hyperpriors (mode 0.2, sd 0.4 for :math:`\lambda`; mode 1, sd 1 for
:math:`\mu, \delta`), and uses the conjugate marginal likelihood as their
likelihood:

.. math::

   p(\theta \mid y) \propto p(y \mid \theta)\, p(\theta).

``method="empirical"`` conditions on the mode; ``method="full"`` (the
default) runs a random-walk Metropolis chain over :math:`\log\theta` and
draws :math:`(B, \Sigma)` at every kept state, so bands integrate over
the tightness.

.. testcode::

   from cultivars.multivariate.large_dim.hierarchical import HierarchicalBVAR

   model = HierarchicalBVAR(y, order=2, names=names)
   eb = model.fit(method="empirical", n_draws=200, seed=0)
   full = model.fit(method="full", n_draws=200, n_burn=200, seed=0)
   print(eb.prior_label, eb.hyper_names)
   print(eb.hyper_mode.round(3))
   print(full.hyper_draws.shape, round(full.acceptance_rate, 2))
   print(full.hyperparameter_interval("lambda").round(3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   glp(lambda, mu, delta) ('lambda', 'mu', 'delta')
   [0.292 1.696 0.232]
   (200, 3) 0.49
   [0.253 0.34  0.415]

Constructor arguments: ``sum_of_coefficients=True`` and
``dummy_initial_observation=True`` switch the two dummy components (and
their hyperparameters) on; ``decay``, ``exogenous`` and ``persistence``
are held fixed. ``fit(method="full", n_draws=1000, n_burn=500, seed=None)``:
``n_draws`` is the number of kept draws, ``n_burn`` the Metropolis
burn-in. The result is a ``BVARResult`` subclass with ``hyper_names``,
``hyper_mode``, ``hyper_draws`` (empty under empirical Bayes),
``acceptance_rate`` (``nan`` under empirical Bayes), ``method`` and
``hyperparameter_interval(name)``. Its ``log_marginal_likelihood`` is
:math:`p(y \mid \hat\theta)` at the mode, not integrated over the
hyperprior.

Non-conjugate priors: GibbsBVAR
-------------------------------

:class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR` states the prior
independently of the covariance (Koop and Korobilis 2010),

.. math::

   \operatorname{vec}(B) \sim \mathcal{N}(\underline\beta, \underline V),
   \qquad
   \Sigma \sim \mathcal{IW}(\underline S, \underline\nu),

and samples :math:`\operatorname{vec}(B) \mid \Sigma` by one joint GLS
draw with precision :math:`\underline V^{-1} + \Sigma^{-1} \otimes
X^\top X`, then :math:`\Sigma \mid B` from an inverse-Wishart. Under an
adaptive prior a third block redraws the latent scales. Each sweep
factors a :math:`(k^2 p + k\,n_{det})`-square matrix, so wide systems
belong to ``BVAR``.

.. testcode::

   from cultivars.multivariate.large_dim.gibbs import GibbsBVAR
   from cultivars.bayes.priors import HorseshoePrior, SpikeAndSlabPrior

   gibbs = GibbsBVAR(y, order=1, names=names).fit(n_draws=600, n_burn=200, seed=0)
   print(gibbs.prior_label, gibbs.n_draws, gibbs.n_kept)
   evidence = gibbs.marginal_likelihood()
   print(evidence.method, round(evidence.log_value, 1), evidence.mcse < 0.1)

.. testoutput::

   inw(l1=0.2, l2=0.5, l3=1, l4=100) 600 400
   chib -836.9 True

The default prior is
:class:`~cultivars.bayes.priors.IndependentNormalWishartPrior`, Litterman's
moments with ``cross_equation=0.5``. The marginal likelihood is Chib's
(1995) estimate from the Gibbs output; it is refused under an adaptive
prior and under a prior with dummy observations.

Adaptive shrinkage reports what it learned. A spike-and-slab prior
(George, Sun and Ni 2008) gives posterior inclusion probabilities; a
global-local prior (horseshoe, Carvalho, Polson and Scott 2010;
Dirichlet-Laplace; Normal-Gamma, Griffin and Brown 2010) gives the
posterior mean local scale of each standardized coefficient. Both are
``(p, k, k)`` stacks aligned with ``coefficients``:

.. testcode::

   ssvs = GibbsBVAR(y, order=1, names=names, prior=SpikeAndSlabPrior()).fit(
       n_draws=600, n_burn=200, seed=0)
   print(ssvs.inclusion_probabilities()[0].round(2))
   horseshoe = GibbsBVAR(y, order=1, names=names, prior=HorseshoePrior()).fit(
       n_draws=600, n_burn=200, seed=0)
   print(horseshoe.shrinkage_scales()[0].round(2))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   [[1.   0.65 0.89]
    [0.12 1.   0.07]
    [0.74 0.04 1.  ]]
   [[0.73 0.38 0.39]
    [0.13 0.82 0.23]
    [0.36 0.24 0.59]]

Selection reads the sample, not the data-generating process. The true
lag matrix is non-zero on the diagonal and at ``[0, 1]`` (0.2), ``[1, 2]``
and ``[2, 0]`` (0.1 each). In this sample least squares puts ``[0, 2]``
at 0.23 with a t-ratio of 3.2, so spike-and-slab includes it; the weak
``[1, 2]`` link is excluded. The horseshoe scales rank the same way. Calling
``inclusion_probabilities`` on a global-local fit, or
``shrinkage_scales`` on a spike-and-slab fit, raises
``SpecificationError``. Adaptive priors do not compose:
``HorseshoePrior() + SumOfCoefficientsPrior()`` is refused at ``fit``.

``fit(n_draws=2000, n_burn=500, thin=1, seed=None)``: ``n_draws`` counts
*all* sweeps, burn-in included, and ``n_kept`` is
``ceil((n_draws - n_burn) / thin)``. The result adds ``shrinkage``,
``shrinkage_label``, ``n_draws``, ``n_burn`` and ``thin`` to the shared
surface (``credible_interval``, ``irf``, ``irf_bands``, ``forecast``,
``forecast_paths``, ``stable_share``, ``convergence``).

Fat tails: StudentBVAR
----------------------

:class:`~cultivars.multivariate.large_dim.student.StudentBVAR` keeps the
conjugate prior surface and replaces Gaussian innovations with
:math:`u_t \sim t_\nu(0, \Sigma)`, estimated through the scale mixture
:math:`u_t \mid w_t \sim \mathcal{N}(0, \Sigma / w_t)`,
:math:`w_t \sim \mathrm{Gamma}(\nu/2, \nu/2)` (Geweke 1993; Chiu, Mumtaz
and Pinter 2017). Given the weights the model is the conjugate VAR on
reweighted rows; :math:`\nu` is either fixed by the caller or drawn on the
grid :math:`2, \ldots, 29, 30, 35, 40, 50, 60, 80, 100`.

.. testcode::

   from cultivars.multivariate.large_dim.student import StudentBVAR

   rng_t = np.random.default_rng(11)
   fat = np.zeros((200, 2))
   for t in range(1, 200):
       w = rng_t.gamma(2.0, 0.5)            # t_4 innovations as a scale mixture
       fat[t] = 0.5 * fat[t - 1] + rng_t.standard_normal(2) / np.sqrt(w)
   heavy = StudentBVAR(fat, order=1).fit(n_draws=400, n_burn=200, seed=0)
   print(heavy.n_kept, heavy.df_estimated)
   print(heavy.df_interval().round(1))
   print(heavy.outlier_dates().size, heavy.weight_mean.shape)
   ratio = heavy.innovation_covariance / heavy.sigma_u
   print(np.allclose(ratio, heavy.df / (heavy.df - 2.0)))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   100 True
   [5.  6.2 8. ]
   14 (199,)
   True

``fit(df=None, n_draws=2000, n_burn=1000, thin=2, seed=None)``; pass
``df=5.0`` to fix the tail index (``df_interval`` then raises).
``sigma_u`` and ``sigma_draws`` are the t *scale* matrix; the innovation
covariance is ``innovation_covariance``. ``outlier_dates(threshold=0.5)``
returns row indices into ``resid`` (aligned with ``endog[p:]``) whose
posterior mean weight is below the threshold. There is no marginal
likelihood.

Drifting volatility: BVARSV
---------------------------

:class:`~cultivars.multivariate.large_dim.volatility.BVARSV` holds the
coefficients constant and lets innovation variances move,

.. math::

   u_t = A^{-1} \operatorname{diag}\bigl(e^{h_{1t}/2}, \ldots,
   e^{h_{kt}/2}\bigr)\, \varepsilon_t,
   \qquad
   h_{it} = h_{i,t-1} + \eta_{it},

with :math:`A` unit lower triangular, after Carriero, Clark and
Marcellino (2019). The coefficients are drawn jointly by GLS under a moment
prior (default ``MinnesotaPrior()``) and the log variances by the
Kim-Shephard-Chib mixture.

.. testcode::

   from cultivars.multivariate.large_dim.volatility import BVARSV

   rng_v = np.random.default_rng(5)
   drift = np.zeros((200, 2))
   for t in range(1, 200):
       scale = 0.5 + 1.5 * t / 200          # volatility rises through the sample
       drift[t] = 0.5 * drift[t - 1] + scale * rng_v.standard_normal(2)
   sv = BVARSV(drift, order=1).fit(n_draws=300, n_burn=100, seed=0)
   print(sv.prior_label, sv.n_kept, sv.h_draws.shape)
   path = sv.volatility_path("y1")           # (nobs, 3): low, mean, high
   print(path.shape, path[0, 1] < path[-1, 1])

.. testoutput::

   minnesota(l1=0.2, l2=0.5, l3=1, l4=100) 100 (100, 199, 2)
   (199, 3) True

``fit(n_draws=2000, n_burn=1000, thin=2, k_vol=0.01, seed=None)``;
``k_vol`` is the prior scale of the log-volatility random walk (raise it
toward 0.1 when volatility drifts a lot). ``sigma_u`` and ``sigma_draws``
are the covariance at the *end* of the sample,
:math:`\Sigma_T = A^{-1} \operatorname{diag}(e^{h_T}) A^{-\top}`, which
is what ``irf`` and the predictive use; the predictive continues each
draw's random walk forward, so its bands widen with the horizon. The
names order is a structural declaration made at estimation time.
Dummy-observation priors are refused, and there is no marginal
likelihood.

Shrinkage on the point path: ``VAR(prior=...)``
-----------------------------------------------

:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
accepts any prior and returns the posterior mean of the coefficients as
a point estimate (Theil-Goldberger mixed estimation, one equation at a
time), with their Gaussian posterior covariance in ``posterior``
(``stderr``, ``conf_int``, ``wald``). Each equation's error variance is
held at the univariate residual scale :math:`s_i^2`, Litterman's
procedure, so those posterior standard deviations do not reflect the
VAR's own residual variance. It is not a sampler: there are no
draws, and impulse responses, forecasts and decompositions propagate the
point estimate. What it offers in exchange is the whole
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
surface, including ``identify``, ``fevd``, ``generalized_fevd`` and the
closed-system contract that :doc:`structural_identification` and
:class:`~cultivars.multivariate.large_dim.spillover.Spillover` require.

.. testcode::

   from cultivars.multivariate.reduced_form.vector_autoregression import VAR

   ols = VAR(y, order=2, names=names).fit()
   shrunk = VAR(y, order=2, names=names, prior=MinnesotaPrior(tightness=0.1)).fit()
   print(shrunk.prior_label)
   print(round(ols.n_params, 1), round(shrunk.n_params, 1))
   conj = VAR(y, order=2, names=names, prior=NormalInverseWishartPrior()).fit()
   print(np.allclose(conj.coefficients, res.coefficients))

.. testoutput::

   minnesota(l1=0.1, l2=0.5, l3=1, l4=100)
   27.0 15.2
   True

Under a conjugate prior the point estimate equals ``BVAR``'s exact
posterior mean (the last line). ``n_params`` charges the *effective*
number of coefficients the prior left free, and ``llf``, ``aic`` and
``bic`` are evaluated at the posterior mean with that count. The
coefficient table in ``summary()`` reports posterior standard deviations
in place of standard errors.

Convergence
-----------

Every sampled result has
``convergence(*chains, include=None, exclude=(), rhat_tol=1.01, min_ess=100)``,
which returns a
:class:`~cultivars.bayes.chains.ConvergenceTest` over every ``*_draws``
array (split R-hat, bulk and tail ESS, MCSE, Geweke). Pass further
results from runs with different seeds as extra chains:

.. testcode::

   second = GibbsBVAR(y, order=1, names=names).fit(n_draws=600, n_burn=200, seed=1)
   check = gibbs.convergence(second)
   print(check.n_chains, check.n_draws, check.converged)
   print(check.worst(2))

.. testoutput::

   2 400 True
   ('sigma[1,1]', 'sigma[0,1]')

Quantities are labelled by array position, ``beta[row, col]`` in
design-row order and ``sigma[i, j]``. On a ``BVAR`` result the draws are
independent, R-hat is one by construction, and the method exists for
interface parity: a flag there means only that ``n_draws`` was small.

How the pieces connect
----------------------

- **Evidence.** ``BVAR``, ``HierarchicalBVAR`` (at the mode) and
  ``GibbsBVAR`` under a static prior return
  :class:`~cultivars.bayes.evidence.MarginalLikelihoodSelection` records
  from ``marginal_likelihood()``; :func:`~cultivars.bayes.evidence.marginal_likelihood`
  dispatches to them and :func:`~cultivars.bayes.evidence.compare` ranks
  them. ``StudentBVAR`` and ``BVARSV`` have no such method; compare them
  by predictive score. See :doc:`bayesian_workflow`.
- **Predictive checks.** ``BVAR``, ``HierarchicalBVAR``, ``GibbsBVAR`` and
  ``StudentBVAR`` models offer ``prior_replications``; all five results
  offer ``posterior_replications``, which
  :func:`~cultivars.bayes.checks.posterior_predictive_check` consumes.
  ``BVARSV`` has no ``prior_replications``.
- **Forecast evaluation.** ``forecast_paths`` is the predictive sample that
  :func:`~cultivars.forecast.fan.fan_chart`, the density scores and
  :func:`~cultivars.bayes.combination.bayesian_model_average` take; see
  :doc:`forecasting`.
- **Structural analysis.** Bayesian VAR results are not closed-system
  results: they have no ``identify``, ``ma_representation`` or
  ``fevd``, and ``Spillover`` refuses them. Orthogonalized ``irf`` /
  ``irf_bands`` are the recursive identification in the order of
  ``names``. For other identification schemes, either rotate
  ``beta_draws`` and ``sigma_draws`` yourself or use
  ``VAR(prior=...)`` and :doc:`structural_identification`.
- **Wide panels.** ``BVAR`` is the shrinkage answer to dimensionality;
  the selection and factor answers are on :doc:`large_dimensional`.

Nuances and gotchas
-------------------

- **Adaptive priors are silently mis-used outside GibbsBVAR.**
  ``BVAR``, ``StudentBVAR``, ``BVARSV`` and ``VAR`` all accept
  ``HorseshoePrior()`` (and the other adaptive priors) without error, but
  none of them samples the latent scales: they use the prior's variance at
  its *initial* scales as a fixed Gaussian prior, while ``prior_label``
  still reads ``horseshoe``. ``BVAR`` even reports a closed-form marginal
  likelihood for it. Use adaptive priors with ``GibbsBVAR`` only.
- **``n_draws`` means different things.** ``BVAR`` and
  ``HierarchicalBVAR``: draws kept. ``GibbsBVAR``, ``StudentBVAR``,
  ``BVARSV``: total iterations, burn-in included, then thinned
  (``thin`` defaults 1, 2, 2). ``StudentBVAR(...).fit(n_draws=400,
  n_burn=200)`` keeps 100 draws.
- **The default prior differs by model.** ``NormalInverseWishartPrior()``
  for ``BVAR`` and ``StudentBVAR``, ``IndependentNormalWishartPrior()``
  for ``GibbsBVAR``, ``MinnesotaPrior()`` for ``BVARSV``, least squares
  for ``VAR``. All centre own first lags at ``persistence=1.0``, the
  random walk; set ``persistence=0.0`` for growth rates or other
  stationary data.
- **Dummy priors depend on the data's level.** The sum-of-coefficients
  rows are scaled by the presample means; on demeaned data or growth
  rates they carry almost no weight whatever ``tightness`` is.
  ``tightness`` in the package means larger is tighter; the
  Giannone-Lenza-Primiceri :math:`\mu` and :math:`\delta` used by
  ``HierarchicalBVAR`` are reciprocals (larger is looser).
- **Composition double counts.** ``NormalInverseWishartPrior() +
  NormalInverseWishartPrior()`` is accepted and precision-weights the two,
  halving every variance. ``MinnesotaPrior(sum_of_coefficients=1.0) +
  SumOfCoefficientsPrior()`` adds the same rows twice.
- **Orthogonalized responses depend on column order**, for every model
  here. ``BVARResult.irf`` and ``HierarchicalBVARResult.irf`` take only
  ``horizon``, ``orthogonalized`` and ``cumulative``; the
  ``GibbsBVAR``, ``StudentBVAR`` and ``BVARSV`` results additionally take
  ``shocks=`` and ``responses=`` name subsets to bound memory.
- **Bands are pointwise 68%.** ``(16th, mean, 84th)`` at each cell, not a
  joint band over horizons; the middle value is a Monte Carlo mean, not
  ``beta_mean``. Take other quantiles from ``beta_draws`` or
  ``forecast_paths``.
- **Wide systems.** ``BVAR`` needs more rows than columns and at least
  ``order + 2`` rows, nothing more, so 20 variables at two lags fit on 40
  observations. ``lag_order_selection`` on the same model is computed by
  least squares and refuses ``max_lags`` beyond what an unrestricted VAR
  supports on that sample; compare marginal likelihoods on a trimmed
  common sample instead (above).
- **Random-walk priors put mass past the unit circle.** On persistent data
  ``stable_share`` is often well below one; long-horizon responses and
  forecasts then include explosive draws.
- **The hierarchical evidence is optimistic.** ``HierarchicalBVAR``
  reports :math:`p(y \mid \hat\theta)` at the maximizing hyperparameters;
  putting it in a ``compare`` table beside fixed-prior fits favours it by
  construction.
- **``compare`` warns, it does not refuse.** Records with different
  ``nobs`` produce a note under the table; the probabilities are still
  printed.

See also
--------

- :doc:`bayesian_workflow` -- marginal likelihood, predictive checks,
  model averaging and stacking.
- :doc:`large_dimensional` -- sparse, graphical and factor models, and
  spillover tables.
- :doc:`var` and :doc:`var_extensions` -- the unrestricted VAR these
  models shrink.
- :doc:`structural_identification` -- identification schemes for closed
  reduced forms.
- :doc:`forecasting` -- scoring and comparing predictive paths.
- :mod:`cultivars.bayes.priors`, :mod:`cultivars.bayes.evidence`,
  :mod:`cultivars.bayes.chains`
- :class:`~cultivars.multivariate.large_dim.bayesian.BVAR`,
  :class:`~cultivars.multivariate.large_dim.hierarchical.HierarchicalBVAR`,
  :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR`,
  :class:`~cultivars.multivariate.large_dim.student.StudentBVAR`,
  :class:`~cultivars.multivariate.large_dim.volatility.BVARSV`
