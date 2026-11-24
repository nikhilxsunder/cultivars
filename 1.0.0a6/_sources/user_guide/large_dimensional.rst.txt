.. meta::
   :description: Large VARs in cultivars: dynamic factor models, FAVAR, factor stochastic volatility, sparse and graphical VARs, and spillover tables.

Large and high-dimensional systems
==================================

An unrestricted VAR on :math:`k` series with :math:`p` lags has
:math:`p k^2` slope coefficients, and past a handful of variables a
macroeconomic sample cannot pin them down. :mod:`cultivars.multivariate.large_dim`
offers three answers: *shrink* every coefficient toward a prior (the
Bayesian VARs, documented on :doc:`bayesian_vars`), *select* a sparse
support by penalized estimation, or *factor* the panel into a few common
components and model those. This page covers the selection and factor
models and the Diebold-Yilmaz spillover view that reads any closed
reduced form as a network. It is not the place for small systems
(:doc:`var`) or for identification (:doc:`structural_identification`).

.. contents:: On this page
   :local:
   :depth: 1

The family at a glance
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 28 32 30

   * - Class
     - Module
     - Model in one line
     - Use when
   * - :class:`~cultivars.multivariate.large_dim.dynamic_factor.DFM`
     - :mod:`~cultivars.multivariate.large_dim.dynamic_factor`
     - Principal components of the standardized panel, then a VAR on the
       factor scores (Stock-Watson two-step).
     - Forecasting or summarizing a wide panel driven by a few common
       factors.
   * - :class:`~cultivars.multivariate.large_dim.factor_augmented.FAVAR`
     - :mod:`~cultivars.multivariate.large_dim.factor_augmented`
     - A VAR on factors plus observed variables, with an observation
       equation back to every panel series (Bernanke, Boivin and Eliasz
       2005).
     - Responses of hundreds of series to a shock identified in a small
       system.
   * - :class:`~cultivars.multivariate.large_dim.factor_volatility.FactorSV`
     - :mod:`~cultivars.multivariate.large_dim.factor_volatility`
     - Static factor model with stochastic volatility on every factor and
       idiosyncratic error; Gibbs sampler.
     - A time-varying covariance or correlation matrix for a panel of
       returns.
   * - :class:`~cultivars.multivariate.large_dim.sparse.SparseVAR`
     - :mod:`~cultivars.multivariate.large_dim.sparse`
     - Penalized VAR: lasso, adaptive lasso, SCAD, MCP or lag-group.
     - The zero pattern is the finding: a directed Granger network.
   * - :class:`~cultivars.multivariate.large_dim.graphical.GraphicalVAR`
     - :mod:`~cultivars.multivariate.large_dim.graphical`
     - Lasso VAR plus a sparse residual precision by nodewise regression
       (Barigozzi and Brownlees 2019).
     - You also want the contemporaneous partial-correlation network.
   * - :class:`~cultivars.multivariate.large_dim.spillover.Spillover`
     - :mod:`~cultivars.multivariate.large_dim.spillover`
     - Not an estimator: the generalized variance decomposition of a
       fitted closed reduced form as a connectedness table.
     - Diebold-Yilmaz total, directional and net spillovers.
   * - ``BVAR``, ``HierarchicalBVAR``, ``GibbsBVAR``, ``StudentBVAR``,
       ``BVARSV``
     - :mod:`~cultivars.multivariate.large_dim.bayesian` and siblings
     - Shrinkage toward a prior, with posterior draws.
     - See :doc:`bayesian_vars`.

Choosing among them
~~~~~~~~~~~~~~~~~~~

- **Forecast every series, few common drivers:** ``DFM``. It is one SVD
  and one least-squares VAR, fast at any width.
- **Structural shock, panel-wide responses:** ``FAVAR``, then
  :class:`~cultivars.multivariate.structural.factor_augmented.FactorAugmentedSVAR`.
- **Who Granger-causes whom among many series:** ``SparseVAR``; add
  ``GraphicalVAR`` for the within-period network.
- **Moving covariances of many returns:** ``FactorSV``.
- **Connectedness of any of the above:** ``Spillover`` on a closed
  reduced form (``VARResult``, ``SparseVARResult``,
  ``GraphicalVARResult``, the ``factors`` of a DFM or FAVAR).
- **A wide VAR with bands on everything and no sparsity claim:**
  ``BVAR`` (:doc:`bayesian_vars`).

Dynamic factor model
--------------------

:class:`~cultivars.multivariate.large_dim.dynamic_factor.DFM` estimates

.. math::

   x_t = \Lambda F_t + e_t,
   \qquad
   F_t = \sum_{i=1}^{p} A_i F_{t-i} + u_t,

on the standardized panel :math:`x_t = (y_t - \bar y) / s`. The first
:math:`r` principal components estimate the factor space (Stock and
Watson 2002), and a VAR without deterministic terms is fitted to the
scores. Left unspecified, :math:`r` minimizes the Bai and Ng (2002)
criterion

.. math::

   IC_{p2}(r) = \ln V(r) + r\,\frac{N + T}{N T}\,\ln\min(N, T).

.. testcode::

   import numpy as np
   from cultivars.multivariate.large_dim.dynamic_factor import DFM

   rng = np.random.default_rng(1)
   n, N = 240, 40
   f = np.zeros((n, 2))
   for t in range(1, n):
       f[t] = np.array([[0.7, 0.1], [0.0, 0.5]]) @ f[t - 1] + rng.standard_normal(2)
   loadings = rng.standard_normal((N, 2))
   panel = 10.0 + f @ loadings.T + rng.standard_normal((n, N))

   dfm = DFM(panel, order=1).fit()
   print(dfm.n_factors, dfm.criterion_values.round(2))
   print(dfm.variance_shares.round(3), dfm.factors.names)
   print(dfm.loadings.shape, dfm.common_component.shape)
   print(dfm.forecast(4).shape)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   2 [-0.38 -0.84 -0.81 -0.79 -0.74 -0.7  -0.66 -0.62]
   [0.386 0.266] ('f1', 'f2')
   (40, 2) (240, 40)
   (4, 40)

Specification choices
~~~~~~~~~~~~~~~~~~~~~

``order`` (required)
   Lag order of the factor VAR, at least 0.
``n_factors`` (default ``None``)
   Fix :math:`r`, or leave ``None`` to minimize :math:`IC_{p2}` over
   ``1 .. max_factors``.
``max_factors`` (default 8)
   Upper end of the search; silently capped at ``min(nobs - 1, N)``.
``series_names`` (default ``x1 ... xN``)
   Labels used by ``r_squared()`` and the summary.

What the result gives you
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~cultivars.multivariate.large_dim.dynamic_factor.DFMResult`:
``factors`` is a full
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
on the scores, with variables ``f1 ... fr`` (``irf``, ``fevd``,
``forecast``, ``identify`` all run there). ``loadings`` are the
``(N, r)`` orthonormal loadings on the *standardized* panel;
``means`` and ``scales`` undo the standardization; ``idiosyncratic`` is
the standardized remainder; ``variance_shares`` the share of standardized
variance each component explains; ``criterion_values`` the
:math:`IC_{p2}` path (empty when ``n_factors`` was fixed). The property
``common_component`` and the method ``forecast(steps)`` are in original
units; ``r_squared()`` returns a ``{series: R^2}`` dict.

.. testcode::

   fit_share = dfm.r_squared()
   print(len(fit_share), round(np.mean(list(fit_share.values())), 3))
   print(np.allclose(dfm.common_component + dfm.idiosyncratic * dfm.scales, panel))

.. testoutput::

   40 0.652
   True

Factors are identified only up to rotation, so ``f1`` and ``f2`` carry no
structural names; the common component, the ``R^2`` values and the panel
forecast are rotation invariant. The panel forecast maps the factor
forecast through the loadings with the idiosyncratic part at zero, so it
decays to the series means at the factor VAR's rate.

Factor-augmented VAR
--------------------

:class:`~cultivars.multivariate.large_dim.factor_augmented.FAVAR` puts
observed variables :math:`Y_t` (a policy rate, say) beside the factors,

.. math::

   \begin{pmatrix} F_t \\ Y_t \end{pmatrix}
   = \sum_{i=1}^{p} A_i \begin{pmatrix} F_{t-i} \\ Y_{t-i} \end{pmatrix} + u_t,
   \qquad
   x_t = \Lambda^F F_t + \Lambda^Y Y_t + e_t .

Principal components of the full panel absorb the observed block's
contemporaneous influence. Declaring the ``slow`` series (those that do
not react within the period) runs the Bernanke-Boivin-Eliasz cleaning
:math:`\hat F_t = C_t - \hat b_Y' Y_t`, which a recursive identification
with the observed block ordered last needs.

.. testcode::

   from cultivars.multivariate.large_dim.factor_augmented import FAVAR
   from cultivars.multivariate.structural.factor_augmented import FactorAugmentedSVAR

   rng = np.random.default_rng(0)
   n = 300
   factor = np.zeros(n)
   for t in range(1, n):
       factor[t] = 0.7 * factor[t - 1] + rng.standard_normal()
   shock = rng.standard_normal(n)
   rate = 0.5 * factor + shock
   lam = rng.uniform(0.5, 1.5, 30)
   slow = np.outer(factor, lam[:15]) + 0.5 * rng.standard_normal((n, 15))
   fast = np.outer(factor, lam[15:]) + 0.5 * rng.standard_normal((n, 15))
   fast += 0.8 * np.outer(shock, rng.uniform(0.5, 1.5, 15))
   info = np.column_stack([slow, fast])
   labels = tuple(f"s{i}" for i in range(15)) + tuple(f"q{i}" for i in range(15))

   favar = FAVAR(info, rate, n_factors=1, order=1, panel_names=labels,
                 slow=labels[:15], observed_names=("rate",)).fit()
   print(favar.cleaned, favar.factors.names, favar.loadings.shape)
   print(round(float(favar.r2.mean()), 3))
   svar = FactorAugmentedSVAR(favar).identify()
   print(svar.shock_names, svar.irf(8).shape)

.. testoutput::

   True ('f1', 'rate') (30, 2)
   0.897
   ('f1', 'rate') (9, 30, 2)

``FAVAR(panel, observed, *, n_factors, order, slow=None, trend="c",
observed_names=None, panel_names=None)``: ``n_factors`` is required
(no criterion search here), and ``slow`` needs at least
``n_factors + 1`` names. The result carries ``factors`` (a ``VARResult``
on ``[F, Y]``), ``loadings`` ``(N, r + m)``, ``scores`` ``(T, r)``,
``r2`` (an array), ``explained_variance`` (before cleaning), ``cleaned``,
and the method ``common_component()``. Identification happens afterwards:
:class:`~cultivars.multivariate.structural.factor_augmented.FactorAugmentedSVAR`
factorizes ``factors`` recursively with the observed block last by
default, or maps any point-identified structural result on
``favar.factors``, and returns responses of all :math:`N` panel series;
see :doc:`structural_identification`.

Factor stochastic volatility
----------------------------

:class:`~cultivars.multivariate.large_dim.factor_volatility.FactorSV`
models a demeaned panel as

.. math::

   y_t = \Lambda f_t + \varepsilon_t,
   \quad f_t \sim \mathcal{N}(0, H_t),
   \quad \varepsilon_t \sim \mathcal{N}(0, D_t),
   \qquad
   h_{j,t+1} = \mu_j + \phi_j (h_{jt} - \mu_j) + \sigma_j \xi_{jt},

with a log-AR(1) variance on each of the :math:`r` factors and :math:`k`
idiosyncratic errors, so :math:`\Sigma_t = \Lambda H_t \Lambda' + D_t`
moves through :math:`k + r` paths (Pitt and Shephard 1999; Chib, Nardari
and Shephard 2006). Estimation is a Gibbs sampler with Kim-Shephard-Chib
mixture blocks. :math:`\Lambda` is lower triangular with a unit diagonal
in its first :math:`r` rows, so the first ``n_factors`` columns anchor
the factors' scale and sign.

.. testcode::

   from cultivars.multivariate.large_dim.factor_volatility import FactorSV

   rng = np.random.default_rng(2)
   n = 200
   h = np.zeros(n)
   for t in range(1, n):
       h[t] = 0.95 * h[t - 1] + 0.3 * rng.standard_normal()
   common = np.exp(h / 2) * rng.standard_normal(n)
   returns = np.outer(common, [1.0, 0.8, 0.6, 0.4]) + 0.5 * rng.standard_normal((n, 4))

   fsv = FactorSV(returns, n_factors=1).fit(n_draws=300, n_burn=100, seed=0)
   print(fsv.n_kept, fsv.loadings[:, 0].round(2))
   print(fsv.communality().round(2))
   print(fsv.factor_volatility(0).shape, fsv.conditional_correlation().shape)

.. testoutput::

   200 [1.   0.84 0.63 0.45]
   [0.86 0.71 0.62 0.45]
   (3, 200) (200, 4, 4)

``FactorSV(panel, *, n_factors, series_names=None)`` requires at least 50
rows and ``n_factors < k``. ``fit(n_draws=3000, n_burn=1000, thin=1,
prior=None, loading_prior_precision=1.0, seed=None)`` counts ``n_draws``
including burn-in; ``prior`` is a
:class:`~cultivars.bayes.priors.VolatilityPrior` for the log-variance
laws. The result stores ``loading_draws``, ``factor_draws``,
``h_factor_draws``, ``h_idio_draws`` and the AR parameters' draws;
``loadings`` and ``factors`` are posterior-mean properties;
``factor_volatility(j)`` and ``idiosyncratic_volatility(series)`` return
``(len(quantiles), T)`` quantiles of the standard deviation;
``conditional_covariance()`` and ``conditional_correlation()`` are
``(T, k, k)`` posterior means; ``communality()`` is the time-averaged
factor share per series. The sampler does not use ancillarity-sufficiency
interweaving (Kastner, Fruehwirth-Schnatter and Lopes 2017), so loadings
and volatility paths mix slowly; read ``convergence()`` before the point
summaries.

Sparse VAR
----------

:class:`~cultivars.multivariate.large_dim.sparse.SparseVAR` solves, on the
standardized panel,

.. math::

   \hat A = \arg\min_A\ \tfrac{1}{2}\sum_t
   \Bigl\lVert y_t - c - \sum_{i=1}^{p} A_i y_{t-i} \Bigr\rVert^2
   + \lambda\, P(A),

with :math:`P` the lasso (Basu and Michailidis 2015), the adaptive lasso
(ridge-pilot weights), SCAD or MCP (local linear approximation, Zou and
Li 2008), or the lag-group penalty
:math:`\sum_l \sqrt{k^2}\,\lVert A_l \rVert_F` that keeps or drops whole
lag matrices (Nicholson, Matteson and Bien 2017). Deterministic terms are
never penalized. When ``lam`` is not given it is chosen along a
geometric path by rolling-origin one-step forecast cross-validation with a
refit at every origin.

.. testcode::

   from cultivars.multivariate.large_dim.sparse import SparseVAR

   rng = np.random.default_rng(3)
   k, n = 8, 300
   A = 0.4 * np.eye(k)
   A[1, 0], A[2, 0], A[5, 4] = 0.3, 0.3, -0.3
   y = np.zeros((n, k))
   for t in range(1, n):
       y[t] = A @ y[t - 1] + rng.standard_normal(k)

   lasso = SparseVAR(y, order=1).fit(penalty="lasso")
   print(round(lasso.lam, 4), lasso.n_nonzero, round(lasso.sparsity, 3))
   print(lasso.lambda_path.shape, lasso.cv_errors.shape)
   edges = lasso.granger_adjacency(threshold=0.1)
   print([(i, j) for i in range(k) for j in range(k) if edges[i, j]])

.. testoutput::

   0.0357 35 0.453
   (20,) (20,)
   [(1, 0), (2, 0), (5, 4)]

``granger_adjacency(threshold)`` returns a ``(k, k)`` boolean matrix with
``[i, j]`` true when some lag of variable ``j`` enters equation ``i``
above the threshold; the diagonal is always false. At the default
``threshold=0.0`` every surviving coefficient is an edge, and the
forecast-optimal penalty keeps many small ones.

``fit(penalty="lasso", lam=None, n_lambdas=20, lambda_min_ratio=1e-3)``;
``penalty`` is one of ``"lasso"``, ``"adaptive"``, ``"scad"``, ``"mcp"``,
``"group"``. The result has ``coefficients`` ``(p, k, k)``,
``deterministic``, ``sigma_u`` (residual scatter over ``nobs`` minus the
average active regressors per equation), ``resid``, ``fittedvalues``,
``penalty``, ``lam`` (on the standardized scale), ``lambda_path`` and
``cv_errors`` (both empty when ``lam`` was given), ``n_nonzero``,
``sparsity`` and ``ma_representation(horizon)``. It reports no
likelihood, information criteria or standard errors: post-selection
inference is not attempted.

.. testcode::

   grouped = SparseVAR(y, order=2).fit(penalty="group", n_lambdas=10)
   print(round(grouped.lam, 4), [int(np.count_nonzero(b)) for b in grouped.coefficients])
   stated = SparseVAR(y, order=2).fit(penalty="group", lam=0.05)
   print([int(np.count_nonzero(b)) for b in stated.coefficients])

.. testoutput::

   0.0455 [64, 64]
   [64, 0]

Each lag block is either entirely zero or entirely free. The
cross-validated level (0.0455) keeps the second lag although the
simulated system has only one; a slightly larger stated level, 0.05,
drops it.

Graphical VAR
-------------

:class:`~cultivars.multivariate.large_dim.graphical.GraphicalVAR` adds the
second network. Stage one is the lasso VAR above; stage two estimates the
precision of the residuals by Meinshausen-Buhlmann nodewise lasso
regressions, each with its own penalty chosen by K-fold
cross-validation, symmetrized by averaging. A zero in the precision is a
zero partial correlation: conditional independence of two innovations
given the others.

.. testcode::

   from cultivars.multivariate.large_dim.graphical import GraphicalVAR

   rng = np.random.default_rng(0)
   k, n = 5, 400
   theta = np.eye(k)
   theta[0, 1] = theta[1, 0] = -0.5
   theta[2, 3] = theta[3, 2] = 0.5
   chol = np.linalg.cholesky(np.linalg.inv(theta))
   coef = 0.5 * np.eye(k)
   coef[1, 0] = coef[2, 1] = 0.4
   z = np.zeros((n, k))
   for t in range(1, n):
       z[t] = coef @ z[t - 1] + chol @ rng.standard_normal(k)

   graph = GraphicalVAR(z, order=1).fit(seed=0)
   within = graph.contemporaneous_adjacency(threshold=0.3)
   print([(i, j) for i in range(k) for j in range(i + 1, k) if within[i, j]])
   across = graph.granger_adjacency(threshold=0.2)
   print([(i, j) for i in range(k) for j in range(k) if across[i, j]])
   print(graph.partial_correlations.round(2)[0, 1], graph.lam_nodes.shape)

.. testoutput::

   [(0, 1), (2, 3)]
   [(1, 0), (2, 1)]
   0.47 (5,)

``fit(lam=None, n_lambdas=15, lambda_min_ratio=1e-3, n_folds=5,
seed=None)``; the first stage is always the lasso (there is no
``penalty`` argument), and ``seed`` drives the fold shuffle of the
second stage. The result is a ``SparseVARResult`` plus ``precision``,
``partial_correlations``, ``lam_nodes`` and
``contemporaneous_adjacency(threshold=0.05)``. Neither network is a
causal graph or a test.

Spillover tables
----------------

:class:`~cultivars.multivariate.large_dim.spillover.Spillover` reads the
generalized forecast-error variance decomposition (Koop, Pesaran and
Potter 1996; Pesaran and Shin 1998) of a fitted closed reduced form as a
directed network (Diebold and Yilmaz 2009, 2012):

.. math::

   \theta_{ij}(H) = \frac{\sigma_{jj}^{-1} \sum_{h=0}^{H}
   (e_i' \Psi_h \Sigma_u e_j)^2}
   {\sum_{h=0}^{H} e_i' \Psi_h \Sigma_u \Psi_h' e_i},
   \qquad
   \tilde\theta_{ij}(H) = 100\,\frac{\theta_{ij}(H)}{\sum_j \theta_{ij}(H)} .

It needs only ``names``, ``k_endog``, ``sigma_u`` and
``ma_representation``, so it takes a ``VARResult`` (shrunk or not), a
``SparseVARResult``, a ``GraphicalVARResult``, or the ``factors`` of a
DFM or FAVAR. Running it on the lasso VAR is the Demirer, Diebold, Liu
and Yilmaz (2018) pipeline:

.. testcode::

   from cultivars.multivariate.large_dim.spillover import Spillover

   table = Spillover(lasso, horizon=10).compute()
   print(round(table.total, 2), table.transmitters())
   print(table.table.shape, np.allclose(table.table.sum(axis=1), 100.0))
   print(table.net.round(1))
   print(round(Spillover(dfm.factors, horizon=10).compute().total, 2))

.. testoutput::

   6.26 ('y1', 'y5', 'y8')
   (8, 8) True
   [ 22.6 -13.6 -10.5  -1.2   9.4  -5.2  -1.5   0.1]
   1.13

``Spillover(result, *, horizon=10)`` then ``compute()``. ``horizon`` is the
largest moving-average lead, so :math:`H + 1` terms are summed and the
table equals ``100 * result.generalized_fevd(horizon)[-1]`` on a
``VARResult``. :class:`~cultivars.multivariate.large_dim.spillover.SpilloverResult`
has ``table`` (rows sum to 100; entry ``[i, j]`` is the share of ``i``'s
variance due to shocks to ``j``), ``total`` (average off-diagonal mass,
in percent), ``to_others`` (column sums), ``from_others`` (row sums),
``net``, ``net_pairwise`` and ``transmitters()`` (names with positive
``net``, largest first). Because generalized shocks are correlated the
table is a relative attribution, not an additive decomposition. For the
rolling-window picture, refit on each window and call ``Spillover``
again; there is no built-in rolling helper.

How this relates to the Bayesian VARs
-------------------------------------

The three answers differ in what they claim and what they return.

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * -
     - Shrink (``BVAR`` family)
     - Select (``SparseVAR``, ``GraphicalVAR``)
     - Factor (``DFM``, ``FAVAR``, ``FactorSV``)
   * - Claim
     - Coefficients are near a prior mean.
     - Most coefficients are exactly zero.
     - A few factors drive the comovement.
   * - Uncertainty
     - Posterior draws, 68% bands.
     - None reported.
     - Two-step: none for the factor step; ``FactorSV`` is a posterior.
   * - Closed system for ``Spillover`` / identification
     - No (except ``VAR(prior=...)``).
     - Yes.
     - Yes, through ``.factors`` (DFM, FAVAR).
   * - Model comparison
     - Marginal likelihood (``BVAR``, ``GibbsBVAR``).
     - Cross-validation errors on the result.
     - :math:`IC_{p2}` (DFM).

A BVAR result propagates through its own draws and does not satisfy the
closed-system protocol, so ``Spillover`` refuses it. For a shrunk
network, fit the point-estimate path with the same prior:

.. testcode::

   from cultivars.bayes.priors import NormalInverseWishartPrior
   from cultivars.exceptions import SpecificationError
   from cultivars.multivariate.large_dim.bayesian import BVAR
   from cultivars.multivariate.reduced_form.vector_autoregression import VAR

   prior = NormalInverseWishartPrior(persistence=0.0)
   posterior = BVAR(y, order=1, prior=prior).fit(n_draws=50, seed=0)
   try:
       Spillover(posterior)
   except SpecificationError as err:
       print(str(err)[:60])
   shrunk = VAR(y, order=1, prior=prior).fit()
   print(np.allclose(shrunk.coefficients, posterior.coefficients))
   print(round(Spillover(shrunk, horizon=10).compute().total, 2))

.. testoutput::

   Spillover reads a fitted closed reduced-form result exposing
   True
   7.92

How the pieces connect
----------------------

- **Factor VARs are VARs.** ``DFMResult.factors`` and
  ``FAVARResult.factors`` are ordinary ``VARResult`` objects: forecast,
  ``irf``, ``fevd``, Granger tests, and every identification model on
  :doc:`structural_identification` apply. ``FactorAugmentedSVAR`` maps
  the identified factor-space shocks onto all panel series.
- **Closed-system contract.** ``SparseVARResult`` and
  ``GraphicalVARResult`` expose ``names``, ``k_endog``, ``sigma_u`` and
  ``ma_representation``, which is what ``Spillover`` and the structural
  layer read.
- **Bayesian tools.** ``FactorSVResult.convergence()`` is the same
  diagnostic as on the Bayesian VARs (:doc:`bayesian_workflow`). The
  penalized and two-step factor models have no posterior and no marginal
  likelihood.
- **Forecasting.** ``DFMResult.forecast`` gives point paths for the
  whole panel; the factor VAR's ``forecast`` gives the factor paths. For
  predictive densities on a wide system use ``BVAR.forecast_paths``
  (:doc:`bayesian_vars`, :doc:`forecasting`).

Nuances and gotchas
-------------------

- **SparseVAR and GraphicalVAR have no forecast or irf.** Their results
  offer only ``ma_representation`` among the propagation methods; build
  forecasts from ``coefficients`` and ``deterministic``, or impulse
  responses from ``ma_representation`` and ``sigma_u``.
- **FAVAR's observation equation has no intercept.** ``loadings`` come
  from regressing the *standardized* (mean-zero) panel on ``[F, Y]``
  without a constant, while ``Y`` is used as supplied. Shifting the
  observed block by a constant changes ``loadings``, ``r2`` and
  ``common_component()``. Demean the observed variables before passing
  them if you read the observation equation.
- **Inconsistent accessors across the factor results.**
  ``DFMResult.common_component`` is a property and ``r_squared()`` a
  method returning a dict; ``FAVARResult.common_component()`` is a method
  and ``r2`` an array attribute. The panel labels are ``series_names`` on
  ``DFM`` and ``FactorSV`` and ``panel_names`` on ``FAVAR``.
- **Factor labels are a basis.** In ``DFM`` and ``FAVAR`` the factors are
  principal components, identified up to rotation; in ``FactorSV`` they
  are pinned by the triangular convention, so reordering the columns of
  the panel changes the posterior.
- **The Bai-Ng count can overselect** when the common component explains
  nearly all of the panel; fix ``n_factors`` if the selected count looks
  generous next to ``variance_shares``.
- **Selected support is not inference.** Edges depend on ``lam`` and on
  ``threshold``. The cross-validated level is prediction-optimal and keeps
  more than the true support; the nonconvex penalties are designed for
  support recovery, but on any one sample they can keep spurious links
  too. Inspect ``lambda_path`` and ``cv_errors``.
- **lam is not comparable across penalties.** It is on the standardized
  scale and each penalty's path starts from its own zeroing level.
- **Spillover horizon counts from zero.** ``horizon=10`` sums
  :math:`\Psi_0, \ldots, \Psi_{10}`, eleven terms.
- **Run time.** ``SparseVAR`` refits at every rolling origin for every
  path point. With defaults, the 8-variable example above takes about
  0.2 s at one lag and about 1.5 s at two lags; ``penalty="mcp"`` roughly
  doubles that.
  ``FactorSV`` at 300 iterations on four series takes a few seconds; real
  applications need thousands.

See also
--------

- :doc:`bayesian_vars` -- the shrinkage answer: BVAR and its relatives.
- :doc:`var` -- the reduced-form VAR that factor models and spillover
  tables build on.
- :doc:`structural_identification` -- identification for closed systems,
  including ``FactorAugmentedSVAR``.
- :doc:`volatility` -- univariate stochastic volatility, the law each
  ``FactorSV`` path follows.
- :doc:`forecasting` -- evaluating panel forecasts.
- :mod:`cultivars.multivariate.large_dim.dynamic_factor`,
  :mod:`cultivars.multivariate.large_dim.factor_augmented`,
  :mod:`cultivars.multivariate.large_dim.factor_volatility`,
  :mod:`cultivars.multivariate.large_dim.sparse`,
  :mod:`cultivars.multivariate.large_dim.graphical`,
  :mod:`cultivars.multivariate.large_dim.spillover`
