.. meta::
   :description: Structural identification in cultivars: zero, sign, narrative, proxy, heteroskedasticity, non-Gaussian, SV and FAVAR schemes on a fitted VAR.

Structural identification
=========================

A fitted reduced form gives you the innovation covariance
:math:`\Sigma_u` and nothing about which economic shocks produced it.
Every factorization :math:`\Sigma_u = B B'` tells a different story, and
:mod:`cultivars.multivariate.structural` is the set of ways to choose one:
declared zeros, declared signs, declared history, an outside instrument,
or a statistical property of the shocks. This page covers every scheme in
that package, how to read what each returns, and where they disagree. It
is not about estimating the reduced form itself (see :doc:`var` and
:doc:`var_extensions`), nor about time-varying or regime-dependent
structural models (see :doc:`nonlinear_multivariate`).

.. contents:: On this page
   :local:
   :depth: 1

The family at a glance
----------------------

All classes live under ``cultivars.multivariate.structural``; the column
"module" gives the leaf module to import from.

.. list-table::
   :header-rows: 1
   :widths: 22 20 34 24

   * - Class
     - Module
     - Identifying information
     - Returns
   * - :class:`~cultivars.multivariate.structural.zero_restrictions.RecursiveSVAR`
     - ``zero_restrictions``
     - Causal ordering on impact (Cholesky), Sims (1980)
     - ``SVARResult``, complete
   * - :class:`~cultivars.multivariate.structural.zero_restrictions.LongRunSVAR`
     - ``zero_restrictions``
     - Ordering of permanence at the infinite horizon, Blanchard and
       Quah (1989)
     - ``SVARResult``, complete
   * - :class:`~cultivars.multivariate.structural.zero_restrictions.ShortRunSVAR`
     - ``zero_restrictions``
     - Zeros anywhere in the AB model :math:`A u_t = B\varepsilon_t`,
       by maximum likelihood, Amisano and Giannini (1997)
     - ``SVARResult``, complete, with an over-identification LR test
   * - :class:`~cultivars.multivariate.structural.zero_restrictions.MixedSVAR`
     - ``zero_restrictions``
     - :math:`k(k-1)/2` zeros split between impact and long run, Galí
       (1999)
     - ``SVARResult``, complete
   * - :class:`~cultivars.multivariate.structural.sign_restrictions.SignRestrictedSVAR`
     - ``sign_restrictions``
     - Signs of responses at chosen horizons, Uhlig (2005)
     - ``SignRestrictedSVARResult`` (a set of rotations)
   * - :class:`~cultivars.multivariate.structural.sign_restrictions.NarrativeSignRestrictedSVAR`
     - ``sign_restrictions``
     - Signs plus dated events, Antolín-Díaz and Rubio-Ramírez (2018)
     - ``SignRestrictedSVARResult``
   * - :class:`~cultivars.multivariate.structural.set_identification.SetIdentifiedSVAR`
     - ``set_identification``
     - Signs (and optional zeros) on one shock; exact bounds,
       Giacomini and Kitagawa (2021), Gafarov, Meier and Montiel Olea
       (2018)
     - ``SetIdentifiedSVARResult`` (lower/upper bounds)
   * - :class:`~cultivars.multivariate.structural.external_instruments.ProxySVAR`
     - ``external_instruments``
     - External instrument for one shock, Mertens and Ravn (2013)
     - ``SVARResult``, one column
   * - :class:`~cultivars.multivariate.structural.heteroskedacity.HeteroskedasticSVAR`
     - ``heteroskedacity``
     - Declared variance regimes, Rigobon (2003)
     - ``SVARResult``, complete, statistically labelled
   * - :class:`~cultivars.multivariate.structural.non_gaussian.NonGaussianSVAR`
     - ``non_gaussian``
     - Independent, non-Gaussian shocks, Comon (1994), Lanne, Meitz and
       Saikkonen (2017)
     - ``SVARResult``, complete, statistically labelled
   * - :class:`~cultivars.multivariate.structural.stochastic_volatility.StochasticVolatilitySVAR`
     - ``stochastic_volatility``
     - Non-proportional latent volatility paths, Lewis (2021), Bertsche
       and Braun (2022)
     - ``StochasticVolatilitySVARResult`` (posterior draws)
   * - :class:`~cultivars.multivariate.structural.factor_augmented.FactorAugmentedSVAR`
     - ``factor_augmented``
     - Any point scheme on a FAVAR's factor VAR, mapped onto the panel,
       Bernanke, Boivin and Eliasz (2005)
     - ``FactorAugmentedSVARResult``
   * - :class:`~cultivars.multivariate.structural.perturbation.PerturbationDSGE`
     - ``perturbation``
     - Not a scheme: a DSGE model solved by perturbation and estimated
       by likelihood
     - ``PerturbationDSGEResult``

The two-stage design
--------------------

Structural analysis in cultivars is always two steps. First fit a reduced
form. Then hand the *fitted result* to a scheme object together with the
restrictions, and call ``identify()`` on the scheme:

.. code-block:: text

   reduced = VAR(y, order=p).fit()                       # stage 1
   structural = Scheme(reduced, <restrictions>).identify()   # stage 2

The scheme validates the restrictions at construction, so a malformed
declaration fails before any computation. ``identify()`` takes no
arguments (the one exception is ``StochasticVolatilitySVAR``, whose
``identify`` takes the sampler settings) and returns a new, frozen result whose ``source`` attribute points back at the reduced form.
The reduced form is never modified; one fit can be identified any number
of ways.

The running example is a bivariate VAR(1) whose true impact matrix is
lower triangular, so a recursive ordering is the correct one:

.. testcode::

   import numpy as np
   from cultivars.multivariate.reduced_form.vector_autoregression import VAR
   from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR

   rng = np.random.default_rng(0)
   A = np.array([[0.5, 0.1], [0.0, 0.4]])
   B_true = np.array([[1.0, 0.0], [0.5, 1.0]])
   eps = rng.standard_normal((402, 2))
   y = np.zeros((402, 2))
   for t in range(1, 402):
       y[t] = A @ y[t - 1] + B_true @ eps[t]

   var = VAR(y, order=1, names=("rate", "output")).fit()
   svar = RecursiveSVAR(var).identify()
   print(svar.scheme, svar.shock_names, svar.is_complete)
   print(svar.impact.round(2))
   print(svar.source is var)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   recursive ('rate', 'output') True
   [[1.   0.  ]
    [0.51 1.  ]]
   True

Every result type is a frozen dataclass; attributes cannot be reassigned.
Printing a result (or calling ``summary()``) renders a table with the
identifying assumption restated as a sentence and a note on the shock
normalization:

.. testcode::

   print(svar.summary())

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

                              Structural VAR (recursive)
   ================================================================================
   Scheme:                      recursive  Identified shocks:              2 of 2
   Variables:                           2  Observations:                      401
   --------------------------------------------------------------------------------
   impact of       rate   output
   rate          1.0046   0.0000
   output        0.5053   0.9959
   ================================================================================
   Recursive ordering rate -> output: each variable responds on impact only to
   shocks at or before its own position. Permuting the ordering changes the answer;
   that is the identifying assumption, not a numerical artifact.
   Shocks are normalized to unit variance, so impact-column entries are responses
   to a one-standard-deviation structural shock.

Which reduced forms are accepted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A scheme reads only the propagation surface of the reduced form, defined
by the ``ClosedSystemResult`` protocol in
``cultivars/engine/_core/_protocols.py``: ``names``, ``nobs``,
``sigma_u``, ``resid``, ``coefficients``, ``k_endog`` and
``ma_representation(horizon)``. The constructor checks for these members
(``k_endog`` aside) and raises
:class:`~cultivars.exceptions.SpecificationError` naming the missing ones.
In practice:

* accepted: ``VARResult``, ``VARXResult``, ``VECMResult``, ``VARMAResult``
  and the other closed results of :doc:`var` and :doc:`var_extensions`
  (``PanelVARResult``, ``GVARResult``, ``MFVARResult``, ...), plus any
  user-defined object that exposes the same members;
* rejected: the conjugate Bayesian ``BVARResult``, which has no
  ``ma_representation``; its ``irf_bands`` give recursive bands
  directly (see :doc:`bayesian_vars`).

.. testcode::

   from cultivars.exceptions import SpecificationError
   from cultivars.multivariate.large_dim.bayesian import BVAR

   bvar = BVAR(y, order=1, names=("rate", "output")).fit()
   try:
       RecursiveSVAR(bvar)
   except SpecificationError as err:
       print(str(err)[:58])

.. testoutput::

   identification needs a closed reduced-form result exposing

``result.identify(scheme)`` does not work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Closed reduced-form results carry a method ``identify(strategy)`` that
calls ``strategy.identify(self)``. Every scheme in this package takes the
result in its constructor and defines ``identify()`` with no argument, so
that route fails. Use ``Scheme(result, ...).identify()``:

.. testcode::

   try:
       var.identify(RecursiveSVAR(var))
   except TypeError as err:
       print(err)

.. testoutput::

   RecursiveSVAR.identify() takes 1 positional argument but 2 were given

Choosing a scheme
-----------------

.. list-table::
   :header-rows: 1
   :widths: 46 28 26

   * - Identifying information you are willing to defend
     - Scheme
     - Identification
   * - A contemporaneous causal ordering
     - ``RecursiveSVAR``
     - point, closed form
   * - Which shocks have permanent effects on which variables (data in
       differences, stationary system)
     - ``LongRunSVAR``
     - point, closed form
   * - Zeros on impact that are not a triangle, or more zeros than needed
       (testable)
     - ``ShortRunSVAR``
     - point, ML; LR test if over-identified
   * - Some zeros on impact, some at the long run
     - ``MixedSVAR``
     - point, rotation solve
   * - Only the direction of responses, for one or more shocks
     - ``SignRestrictedSVAR``
     - set, sampled under a uniform (Haar) prior
   * - Signs, plus what happened in specific dated episodes
     - ``NarrativeSignRestrictedSVAR``
     - set, sampled
   * - Signs on one shock, and you want the set's exact endpoints with
       no prior
     - ``SetIdentifiedSVAR``
     - set, exact bounds
   * - An outside series correlated with one shock only
     - ``ProxySVAR``
     - point, one column
   * - Known dates at which shock variances changed
     - ``HeteroskedasticSVAR``
     - point, statistical labels
   * - Independent, non-Gaussian shocks (fat tails, skew)
     - ``NonGaussianSVAR``
     - point, statistical labels
   * - Volatility that drifts, without known break dates
     - ``StochasticVolatilitySVAR``
     - posterior draws, statistical labels
   * - A large panel summarized by factors (FAVAR)
     - ``FactorAugmentedSVAR``
     - whatever the factor-space scheme gives

Reading a point-identified result
---------------------------------

:class:`~cultivars.multivariate.structural.zero_restrictions.SVARResult`
is returned by every point scheme (recursive, long-run, short-run, mixed,
proxy, heteroskedasticity, non-Gaussian, and the posterior-median view of
the SV scheme). Its conventions:

``impact``
   ``(k, s)``: row ``i`` is variable ``i`` in the reduced form's
   ``names`` order, column ``j`` is shock ``shock_names[j]``. ``s = k``
   for a complete scheme, ``s = 1`` for a proxy.
``irf(horizon=20, *, cumulative=False)``
   ``(horizon + 1, k, s)``; entry ``[h, i, j]`` is the response of
   variable ``i`` at lead ``h`` to **one standard deviation** of shock
   ``j``. ``cumulative=True`` returns running sums (levels of differenced
   data).
``fevd(horizon=20)``
   ``(horizon + 1, k, s)``; entry ``[h, i, j]`` is shock ``j``'s share of
   variable ``i``'s ``(h+1)``-step forecast-error variance. Rows sum to one
   only under complete identification.
``structural_shocks()``
   ``(nobs, s)``, recovered as :math:`b_j' \Sigma_u^{-1} u_t`, which needs
   only the identified columns.
``historical_decomposition()``
   ``(nobs, k, s)``; entry ``[t, i, j]`` is shock ``j``'s cumulative
   contribution to variable ``i`` at ``t``. Cost is
   :math:`O(\text{nobs}^2)`.
``long_run_impact``
   ``(k, s)`` cumulated (infinite-horizon) response.
``scheme``, ``restriction``, ``diagnostics``
   Short scheme name, the identifying assumption as a sentence, and a
   tuple of ``(label, value)`` string pairs; ``dict(result.diagnostics)``
   makes it a lookup.
``k_shocks``, ``k_endog``, ``is_complete``, ``names``, ``source``
   Bookkeeping.

All responses are to one-standard-deviation shocks. To get responses to
a shock that moves variable ``i`` by one unit on impact, divide the column
by its impact entry:

.. testcode::

   theta = svar.irf(8)
   print(theta.shape, svar.fevd(8).shape)
   unit_rate = theta[:, :, 0] / svar.impact[0, 0]
   print(unit_rate[:3].round(3))
   print(bool(np.allclose(svar.fevd(8).sum(axis=2), 1.0)))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   (9, 2, 2) (9, 2, 2)
   [[1.    0.503]
    [0.574 0.246]
    [0.327 0.128]]
   True

Point-identified results carry no confidence bands: no bootstrap or
delta-method interval is attached to ``SVARResult.irf``. Uncertainty is
reported only where it is the object of the scheme (the sign, set, and
SV schemes below).

Zero restrictions
-----------------

Recursive (Cholesky) ordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   u_t = B\varepsilon_t,\qquad B = \operatorname{chol}(P\Sigma_u P')
   \text{ mapped back through } P,

with :math:`P` the permutation that puts the declared ordering first.
``order`` lists the variables most exogenous first; ``None`` uses the
column order. Shock columns follow the declared ordering and take the
variables' names; impact rows stay in the reduced form's variable order.
The diagonal of the triangular factor is positive, so each shock raises
its own variable on impact.

.. testcode::

   flipped = RecursiveSVAR(var, order=("output", "rate")).identify()
   print(flipped.shock_names)
   print(flipped.impact.round(2))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   ('output', 'rate')
   [[0.45 0.9 ]
    [1.12 0.  ]]

Under the reversed ordering the ``rate`` shock (column 1) has no impact on
``output`` (row 1). ``RecursiveSVAR(var).identify().irf(h)`` equals the
reduced form's ``var.irf(h, orthogonalized=True)``; the scheme exists so
that the ordering is an explicit argument recorded in the summary.

Long-run (Blanchard-Quah)
~~~~~~~~~~~~~~~~~~~~~~~~~

The cumulated response matrix is lower triangular in the declared
ordering. With :math:`F = A(1)^{-1}M(1)` the long-run multiplier of the
innovations,

.. math::

   F\Sigma_u F' = \Theta\Theta',\qquad \Theta\ \text{lower triangular},
   \qquad B = F^{-1}\Theta .

The example builds a system whose long-run impact of the structural
shocks is triangular by construction, then recovers it:

.. testcode::

   from cultivars.multivariate.structural.zero_restrictions import LongRunSVAR

   rng = np.random.default_rng(0)
   F = np.array([[1.0, 0.0], [0.5, 1.0]])
   B_lr = (np.eye(2) - A) @ F
   eps_lr = rng.standard_normal((602, 2))
   y_lr = np.zeros((602, 2))
   for t in range(1, 602):
       y_lr[t] = A @ y_lr[t - 1] + B_lr @ eps_lr[t]
   var_lr = VAR(y_lr, order=1, names=("dy", "u")).fit()

   bq = LongRunSVAR(var_lr).identify()
   print(bq.scheme, bq.shock_names)
   print(bq.long_run_impact.round(2))
   print(bq.impact.round(2), B_lr.round(2), sep="\n")
   print(bool(np.allclose(bq.irf(200, cumulative=True)[-1], bq.long_run_impact)))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   long-run ('dy', 'u')
   [[ 1.06 -0.  ]
    [ 0.67  0.94]]
   [[ 0.44 -0.11]
    [ 0.31  0.58]]
   [[ 0.45 -0.1 ]
    [ 0.3   0.6 ]]
   True

The constructor refuses a result with ``is_stable`` false; ``identify()``
raises :class:`~cultivars.exceptions.SpecificationError` when the AR
polynomial has a (near-)unit root, which is what an error-correction
model in levels has (difference the integrated variables first), and
:class:`~cultivars.exceptions.NumericalError` if the factorization fails
to reproduce :math:`\Sigma_u` because :math:`F` is ill-conditioned.

Short-run AB model
~~~~~~~~~~~~~~~~~~

.. math::

   A u_t = B\varepsilon_t,\qquad \Sigma_u = A^{-1}BB'A^{-\top},
   \qquad \text{impact} = A^{-1}B .

Patterns are ``(k, k)`` arrays: ``np.nan`` marks a free cell, a finite
number fixes it. ``a=None`` sets :math:`A = I` (the B-model);
``b=None`` frees the diagonal of :math:`B` and fixes the rest at zero (the
A-model). The diagonal of ``a`` must be fixed. At most :math:`k(k+1)/2`
cells may be free (order condition, checked at construction); fewer free
cells over-identify the model and the likelihood-ratio statistic against
the unrestricted covariance is put in ``diagnostics``. The rank condition
is checked at ``identify()`` time: a just-identified pattern that cannot
reproduce :math:`\Sigma_u` raises
:class:`~cultivars.exceptions.NumericalError`.

.. testcode::

   from cultivars.multivariate.structural.zero_restrictions import ShortRunSVAR

   rng = np.random.default_rng(0)
   B3 = np.array([[0.45, -0.1, 0.0], [0.27, 0.58, -0.1], [0.11, 0.14, 0.7]])
   eps3 = rng.standard_normal((802, 3))
   y3 = np.zeros((802, 3))
   for t in range(1, 802):
       y3[t] = 0.4 * y3[t - 1] + B3 @ eps3[t]
   var3 = VAR(y3, order=1, names=("prod", "hours", "rate")).fit()

   nan = np.nan
   exact = ShortRunSVAR(var3, b=[[nan, 0, 0], [nan, nan, 0], [nan, nan, nan]])
   print(exact.n_free, exact.overidentifying_restrictions)
   ab = exact.identify()
   print(dict(ab.diagnostics))
   print(bool(np.allclose(ab.impact, RecursiveSVAR(var3).identify().impact)))

   over = ShortRunSVAR(var3, b=[[nan, 0, 0], [nan, nan, 0], [nan, 0, nan]]).identify()
   print(dict(over.diagnostics))

.. testoutput::

   6 0
   {'Free parameters': '6', 'Identification': 'exact'}
   True
   {'Free parameters': '5', 'Over-ID restrictions': '1', 'Over-ID LR': '1.251', 'Over-ID p-value': '0.2634'}

Each AB impact column is signed so that its largest-magnitude entry is
positive (not its diagonal entry, as in the recursive scheme).

Mixed short- and long-run zeros
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``MixedSVAR`` takes an ``impact`` pattern and a ``long_run`` pattern
(same ``nan``/value convention) with exactly :math:`k(k-1)/2` fixed cells
in total, at least one of them at the long run, and solves for the
rotation :math:`Q` of the Cholesky factor that satisfies all of them:

.. math::

   B = LQ,\quad Q'Q = I,\quad B_{ij} = c_{ij}\ \text{(impact cells)},\quad
   (FB)_{ij} = d_{ij}\ \text{(long-run cells)} .

The Galí (1999) style example: only the technology shock moves
productivity in the long run, and the policy shock cannot move
productivity on impact.

.. testcode::

   from cultivars.multivariate.structural.zero_restrictions import MixedSVAR

   rng = np.random.default_rng(0)
   A3 = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
   F3 = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.3, 0.2, 1.0]])
   eps_m = rng.standard_normal((802, 3))
   y_m = np.zeros((802, 3))
   for t in range(1, 802):
       y_m[t] = A3 @ y_m[t - 1] + (np.eye(3) - A3) @ F3 @ eps_m[t]
   var_m = VAR(y_m, order=1, names=("prod", "hours", "rate")).fit()

   gali = MixedSVAR(
       var_m,
       impact=[[nan, nan, 0.0], [nan, nan, nan], [nan, nan, nan]],
       long_run=[[nan, 0.0, 0.0], [nan, nan, nan], [nan, nan, nan]],
       shock_names=("technology", "demand", "policy"),
   ).identify()
   print(gali.impact.round(2))
   print(gali.long_run_impact[0].round(2))
   print(dict(gali.diagnostics)["Identification"])

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   [[ 0.45 -0.09  0.  ]
    [ 0.26  0.57 -0.05]
    [ 0.08  0.07  0.73]]
   [ 1.02  0.   -0.  ]
   exact

Fewer or more than :math:`k(k-1)/2` restrictions is a
:class:`~cultivars.exceptions.SpecificationError` at construction (an
over-identified mixed pattern is not estimated); a pattern no rotation can
satisfy raises :class:`~cultivars.exceptions.NumericalError` at
``identify()``.

Sign restrictions
-----------------

Sampled identified set
~~~~~~~~~~~~~~~~~~~~~~

Any impact matrix consistent with :math:`\Sigma_u` is :math:`B = LQ` for
the Cholesky factor :math:`L` and an orthogonal :math:`Q`. The scheme
draws :math:`Q` uniformly (Haar, by QR of a Gaussian matrix with the sign
fix of Rubio-Ramírez, Waggoner and Zha 2010) and keeps the draws for which

.. math::

   \operatorname{sign}\big(e_i'\Psi_h L Q e_j\big) = s_{ij}
   \quad\text{for every declared }(i, j, s_{ij})\text{ and every }
   h \in \mathcal{H}.

``restrictions`` maps a shock label to a ``{variable: "+" | "-"}``
pattern. Declared shocks occupy the leading columns in declaration order;
the remaining columns are named ``unrestricted1``, ``unrestricted2``, ...
``horizons`` (default ``(0,)``) are the leads at which every declared sign
must hold. ``draws`` is the number of *accepted* rotations wanted,
``max_attempts`` the number of rotations to try (default
``1000 * draws``), ``seed`` (default ``0``) makes the set reproducible.

.. testcode::

   from cultivars.multivariate.structural.sign_restrictions import SignRestrictedSVAR

   rng = np.random.default_rng(0)
   B_s = np.array([[1.0, 0.5], [0.3, 1.0]])
   eps_s = rng.standard_normal((402, 2))
   y_s = np.zeros((402, 2))
   for t in range(1, 402):
       y_s[t] = A @ y_s[t - 1] + B_s @ eps_s[t]
   var_s = VAR(y_s, order=1, names=("gdp", "infl")).fit()

   signs = {
       "demand": {"gdp": "+", "infl": "+"},
       "supply": {"gdp": "+", "infl": "-"},
   }
   sset = SignRestrictedSVAR(var_s, signs, draws=200, seed=0).identify()
   print(sset.shock_names, sset.n_accepted, sset.requested, sset.attempts)
   print(round(sset.acceptance_rate, 3), sset.impacts.shape)

.. testoutput::

   ('demand', 'supply') 200 200 753
   0.266 (200, 2, 2)

The result holds the accepted impact matrices themselves. Each is a
complete model (:math:`B_nB_n' = \Sigma_u`); the summaries are pointwise
quantiles across the set:

``impacts``
   ``(n_accepted, k, k)``.
``irf_draws(horizon, *, cumulative=False)``
   ``(n_accepted, horizon + 1, k, k)``, entry ``[n, h, i, j]``.
``irf(horizon, *, quantiles=(0.16, 0.5, 0.84), cumulative=False)``
   ``(len(quantiles), horizon + 1, k, k)``, entry ``[q, h, i, j]``.
``fevd(horizon, *, quantiles=...)``
   same shape; shares are computed per rotation, then quantiles taken, so
   the quantile surfaces need not sum to one (they do with two variables).
``n_accepted``, ``requested``, ``attempts``, ``acceptance_rate``, ``horizons``
   The draw accounting.

.. testcode::

   bands = sset.irf(4)
   print(bands.shape, sset.irf_draws(4).shape)
   print(np.sign(bands[1, 0]))                   # median impact signs
   print(bands[:, 0, 0, 0].round(3))             # gdp on impact, demand shock

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   (3, 5, 2, 2) (200, 5, 2, 2)
   [[ 1.  1.]
    [ 1. -1.]]
   [0.858 1.017 1.112]

The set reflects identification uncertainty only: rotations are drawn at
the reduced-form point estimate, and the summary says so. The pointwise
median is not the response of any single rotation (Fry and Pagan 2011),
and the Haar draw is itself a prior over rotations (Baumeister and
Hamilton 2015). For the exact range of the set use
``SetIdentifiedSVAR`` below.

Signs at several horizons
~~~~~~~~~~~~~~~~~~~~~~~~~

``horizons=(0, 1, 2)`` requires each declared sign at every listed lead,
which can only shrink the set. On a system whose responses oscillate,
multi-horizon signs can become unsatisfiable. If no rotation is accepted
within ``max_attempts``, ``identify()`` raises
:class:`~cultivars.exceptions.SpecificationError`; if some but fewer than
``draws`` are accepted, it returns the partial set without an error, and
``n_accepted < requested`` is the only signal:

.. testcode::

   impact_only = SignRestrictedSVAR(var_s, signs, draws=100, seed=0).identify()
   three = SignRestrictedSVAR(var_s, signs, horizons=(0, 1, 2), draws=100, seed=0).identify()
   print(round(impact_only.acceptance_rate, 3), round(three.acceptance_rate, 3))

   short = SignRestrictedSVAR(var_s, signs, draws=200, max_attempts=100, seed=0).identify()
   print(short.n_accepted, short.requested, short.attempts)

   rng = np.random.default_rng(1)
   y_osc = np.zeros((402, 2))
   for t in range(1, 402):
       y_osc[t] = -0.9 * y_osc[t - 1] + B_s @ rng.standard_normal(2)
   var_osc = VAR(y_osc, order=1, names=("gdp", "infl")).fit()
   demand = {"demand": {"gdp": "+", "infl": "+"}}
   try:
       SignRestrictedSVAR(
           var_osc, demand, horizons=(0, 1), draws=10, max_attempts=200
       ).identify()
   except SpecificationError as err:
       print(str(err)[:54])

.. testoutput::

   0.292 0.139
   27 200 100
   no rotation satisfied the declared signs in 200 draws.

Remedies when no rotation is accepted: shorten ``horizons``, drop or
loosen a sign, raise ``max_attempts``, or reconsider whether the declared
shocks can coexist in this reduced form.

Narrative sign restrictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~cultivars.multivariate.structural.sign_restrictions.NarrativeSignRestrictedSVAR`
filters the same Haar draws by declared history, checked through each
candidate's own shocks :math:`\varepsilon_{t^\ast} = B^{-1}u_{t^\ast}`:

* ``shock_signs``: tuples ``(shock, period, "+"|"-")``, the sign of a shock
  in a period;
* ``contributions``: tuples ``(shock, variable, period, kind)`` with
  ``kind`` ``"most"`` (the shock's contribution to the variable's surprise
  exceeds every other shock's) or ``"overwhelming"`` (it exceeds all
  others combined).

``restrictions`` (traditional signs) is optional; at least one narrative
event is required. **Periods index the effective sample**: period ``t`` is
residual row ``t``, which is data row ``t + p`` for a VAR(:math:`p`).

.. testcode::

   from cultivars.multivariate.structural.sign_restrictions import (
       NarrativeSignRestrictedSVAR,
   )

   print(var_s.resid.shape[0], bool(np.allclose(var_s.resid + var_s.fittedvalues, y_s[1:])))
   period = int(np.argmax(np.abs(eps_s[1:, 0]) - np.abs(eps_s[1:, 1])))
   print(period, eps_s[period + 1].round(2))      # data row period + 1

   plain = SignRestrictedSVAR(var_s, demand, draws=300, seed=0).identify()
   story = NarrativeSignRestrictedSVAR(
       var_s,
       demand,
       shock_signs=(("demand", period, "-"),),
       contributions=(("demand", "gdp", period, "overwhelming"),),
       draws=300,
       seed=0,
   ).identify()
   print(round(plain.acceptance_rate, 2), round(story.acceptance_rate, 2))
   print(plain.irf(0)[0, 0, 0, 0].round(3), story.irf(0)[0, 0, 0, 0].round(3))

.. testoutput::

   401 True
   238 [-3.9   0.46]
   0.75 0.37
   0.387 0.948

The declared episode, a large negative demand shock that dominated output
in period 238, cuts the accepted set in half and lifts the 16th
percentile of output's impact response from 0.39 toward the true 1.0.
A period outside ``0 .. nobs - 1`` of the residuals raises
:class:`~cultivars.exceptions.SpecificationError` at construction.

A shock-sign event on a shock that has no traditional signs binds
nothing: the sampler may flip any column, and flipping a column flips its
recovered shock. The accepted set is the same as without the event:

.. testcode::

   vacuous = NarrativeSignRestrictedSVAR(
       var_s, demand, shock_signs=(("other", period, "+"),), draws=300, seed=0
   ).identify()
   print(vacuous.shock_names, vacuous.attempts == plain.attempts)
   print(bool(np.allclose(vacuous.impacts[:, :, 0], plain.impacts[:, :, 0])))

.. testoutput::

   ('demand', 'other') True
   True

Contribution events are flip-invariant and bind on their own.

Exact bounds (set identification)
---------------------------------

:class:`~cultivars.multivariate.structural.set_identification.SetIdentifiedSVAR`
takes the same kind of declaration for **one** shock and returns the exact
smallest and largest value of every response over all admissible
rotations, with no prior and no draws. The shock's rotation column
:math:`q` is confined to

.. math::

   \mathcal{Q} = \{q : \lVert q\rVert = 1,\ Zq = 0,\ Sq \ge 0\},

with rows of :math:`Z` (zeros) and :math:`S` (signs) of the form
:math:`e_i'\Psi_h L`; each response is linear in :math:`q`, so its
extrema are attained on faces of the region and are enumerated exactly.

Arguments: ``restrictions`` is a single ``{variable: sign}`` mapping (not
nested by shock), ``shock`` the label (default ``"restricted"``),
``horizons`` the sign leads, ``zeros`` an optional
``{variable: [leads]}`` mapping of exact zeros (at most :math:`k-1`).

.. testcode::

   from cultivars.multivariate.structural.set_identification import SetIdentifiedSVAR

   gk = SetIdentifiedSVAR(var_s, {"gdp": "+", "infl": "+"}, shock="demand").identify()
   print(gk.impact_bounds.round(3))
   bounds = gk.irf(4)
   print(bounds.shape)

   draws = plain.irf_draws(4)[:, :, :, 0]
   inside = (bounds[..., 0] - 1e-12 <= draws) & (draws <= bounds[..., 1] + 1e-12)
   print(bool(inside.all()))
   band = plain.irf(0)[:, 0, :, 0]
   print(band[[0, 2]].round(3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   [[0.    1.123]
    [0.    1.042]]
   (5, 2, 2)
   True
   [[0.387 0.451]
    [1.106 1.028]]

``irf(horizon, *, cumulative=False)`` returns ``(horizon + 1, k, 2)``:
``[h, i, 0]`` is the lower and ``[h, i, 1]`` the upper bound. Every
sampled rotation lies inside the bounds, and the 16-84 band of the
sampled set covers only part of the range. The bounds are pointwise: no
single rotation traces an edge of the box across horizons.

One exact zero in a two-variable system collapses the set to a point,
the recursive column with the zero-restricted variable ordered first:

.. testcode::

   point = SetIdentifiedSVAR(var_s, {"gdp": "+"}, zeros={"infl": [0]}).identify()
   print(point.impact_bounds.round(3))
   print(RecursiveSVAR(var_s, order=("infl", "gdp")).identify().impact[:, 1].round(3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   [[0.817 0.817]
    [0.    0.   ]]
   [0.817 0.   ]

Limits: joint declarations across several shocks have no face-enumeration
form and are not offered (use the sampled scheme); the face count is
capped, so many signs over many horizons in a large system raise
:class:`~cultivars.exceptions.SpecificationError` rather than being
approximated; an empty set raises
:class:`~cultivars.exceptions.NumericalError` at ``identify()``. The
bounds are computed at the reduced-form point estimate; they are not the
robust-Bayes credible region of Giacomini and Kitagawa (2021), which would
also integrate over reduced-form uncertainty.

External instruments (proxy SVAR)
---------------------------------

An instrument :math:`z_t` with :math:`\mathbb{E}[z_t\varepsilon_{1t}] =
\alpha \ne 0` and :math:`\mathbb{E}[z_t\varepsilon_{jt}] = 0` for
:math:`j \ne 1` gives :math:`\mathbb{E}[z_tu_t] = \alpha b_1`. The column
is estimated by two-stage least squares on the residuals and scaled so
that :math:`b_1'\Sigma_u^{-1}b_1 = 1` (a one-standard-deviation shock).
Only that column is identified, and the result carries one column.

``instruments`` is ``(nobs, m)`` (or 1-D for one instrument) and **must
have one row per residual row**, i.e. the effective sample after the
lags. ``normalize`` names the variable whose impact response is made
positive and that anchors the first stage (default: the first variable).

.. testcode::

   from cultivars.exceptions import DimensionError
   from cultivars.multivariate.structural.external_instruments import ProxySVAR

   rng = np.random.default_rng(7)
   proxy = eps[:, 0] + 0.5 * rng.standard_normal(402)
   iv = ProxySVAR(var, proxy[1:], shock="policy").identify()
   print(iv.scheme, iv.shock_names, iv.is_complete, iv.impact.ravel().round(2))
   print(dict(iv.diagnostics)["Normalization"], float(dict(iv.diagnostics)["First-stage F"]) > 100)
   print(iv.fevd(10)[-1].round(2).ravel(), iv.historical_decomposition().shape)

   try:
       ProxySVAR(var, proxy)                      # 402 rows, residuals have 401
   except DimensionError as err:
       print(str(err)[:52])

.. testoutput::

   proxy ('policy',) False [1.   0.49]
   rate > 0 on impact True
   [0.99 0.23] (401, 2, 1)
   instruments must have one row per residual row (401)

The variance shares of a single proxied shock do not sum to one; the
remainder belongs to shocks the instrument does not speak for. The
first-stage F is the classical homoskedastic statistic; F in single
digits means a poorly determined column. Exogeneity is not testable and
the summary says so. There is no weak-instrument-robust (Montiel Olea,
Stock and Watson 2021) interval.

Statistical identification
--------------------------

These schemes identify :math:`B` up to column order and sign from a
property of the shock distribution rather than an economic restriction.
They return complete ``SVARResult`` objects whose shocks are **labelled
by statistics**: default names are ``shock1 ... shockk`` (``shock_<var>``
for the SV scheme), and a diagnostic reports whether the identifying
condition holds in the data.

Heteroskedasticity (declared regimes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \Sigma_m = B\Lambda_m B',\qquad \Lambda_m\ \text{diagonal},
   \qquad \Lambda_1 = I .

With two regimes, :math:`L^{-1}\Sigma_2L^{-\top} = Q\Lambda_2Q'` is an
eigenproblem and :math:`B = LQ` (Rigobon 2003); with more regimes the
over-identified fit is a joint approximate diagonalization and its
residual is reported. ``regimes`` holds one label per **residual** row.
Shocks have unit variance in the first regime (first label to appear),
columns are ordered by descending variance ratio, and each column is
signed so its largest entry is positive.

.. testcode::

   from cultivars.multivariate.structural.heteroskedacity import HeteroskedasticSVAR

   rng = np.random.default_rng(0)
   eps_h = rng.standard_normal((402, 2))
   eps_h[201:, 0] *= 3.0
   y_h = np.zeros((402, 2))
   for t in range(1, 402):
       y_h[t] = 0.4 * y_h[t - 1] + B_s @ eps_h[t]
   var_h = VAR(y_h, order=1, names=("gdp", "infl")).fit()

   regimes = ["calm"] * 200 + ["volatile"] * 201
   rig = HeteroskedasticSVAR(var_h, regimes).identify()
   print(rig.shock_names)
   for key, value in rig.diagnostics:
       print(f"{key}: {value}")
   unit = lambda v: v / np.linalg.norm(v)
   print(unit(rig.impact[:, 0]).round(2), unit(B_s[:, 0]).round(2))

.. testoutput::

   ('shock1', 'shock2')
   Regimes: 2
   Variances in 'volatile' vs 'calm': 9.902, 0.941
   Min ratio separation: 0.8219
   Identification: distinct ratios
   [0.96 0.27] [0.96 0.29]

The shock whose standard deviation tripled comes first, its variance
ratio is about :math:`3^2 = 9`, and the direction of its impact column
matches the truth to within 0.02. If both shocks' variances shift by the same factor, the ratios tie and
the ``Identification`` entry reads ``WEAK: separation within sampling
noise``; an exact tie raises :class:`~cultivars.exceptions.NumericalError`.
The impact matrix reproduces the first regime's residual second moment,
not ``sigma_u``, so the scale is "one standard deviation in the first
regime". The inherited ``SVARResult`` methods do not account for this:
``fevd`` divides by the full-sample ``sigma_u``, so its rows do not sum to
one even though ``is_complete`` is true, and ``structural_shocks()``
projects with :math:`B'\Sigma_u^{-1}` rather than inverting :math:`B`, so
the recovered series has the right shape but the wrong scale (and
``historical_decomposition`` inherits that). Compute shares and shocks
from ``impact`` directly when the scale matters:

.. testcode::

   print(rig.fevd(8)[-1].sum(axis=1).round(3))       # should be [1, 1]
   own = np.linalg.solve(rig.impact, var_h.resid.T).T
   print(rig.structural_shocks().std(axis=0).round(2), own.std(axis=0).round(2))

.. testoutput::

   [0.225 0.752]
   [0.42 1.01] [2.34 0.99]

``own`` is :math:`B^{-1}u_t`: the first shock has standard deviation one
in the calm half and three in the volatile half, about 2.3 overall.

Non-Gaussianity (independent shocks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rotations of independent Gaussian shocks are again independent Gaussian
shocks; with at most one Gaussian shock the rotation is unique up to order
and sign (Comon 1994). The estimator whitens the residuals and jointly
diagonalizes third- and fourth-order cumulant slices, warm-started at the
FOBI solution. Columns are ordered by descending absolute excess kurtosis.
The constructor requires at least ``10 k`` residual rows.

.. testcode::

   from cultivars.multivariate.structural.non_gaussian import NonGaussianSVAR

   rng = np.random.default_rng(0)
   heavy = rng.laplace(size=602) / np.sqrt(2)
   flat = rng.uniform(-np.sqrt(3), np.sqrt(3), size=602)
   eps_g = np.column_stack([heavy, flat])
   y_g = np.zeros((602, 2))
   for t in range(1, 602):
       y_g[t] = 0.4 * y_g[t - 1] + B_s @ eps_g[t]
   var_g = VAR(y_g, order=1, names=("gdp", "infl")).fit()

   ica = NonGaussianSVAR(var_g, shock_names=("heavy", "flat")).identify()
   report = dict(ica.diagnostics)
   print(report["Excess kurtosis"], "|", report["Identification"])
   print(unit(ica.impact[:, 0]).round(2), unit(B_s[:, 0]).round(2))
   print(bool(np.allclose(ica.impact @ ica.impact.T, var_g.sigma_ml)))

.. testoutput::

   3.249, -1.196 | at most one Gaussian shock
   [0.94 0.34] [0.96 0.29]
   True

The Laplace shock (excess kurtosis 3) comes first and the uniform one
(excess kurtosis :math:`-1.2`) second; the direction of the first impact
column is recovered up to sampling error at 601 observations. On Gaussian data the scheme still returns a rotation, and the
``Identification`` entry reads ``WEAK: 2 shocks statistically Gaussian``.
The impact matrix reproduces the maximum-likelihood covariance
``sigma_ml`` (divisor ``nobs``), not the degrees-of-freedom-adjusted
``sigma_u``.

Stochastic volatility
~~~~~~~~~~~~~~~~~~~~~

Each structural shock gets its own latent log-AR(1) variance,

.. math::

   u_t = B\varepsilon_t,\quad \varepsilon_{jt} = e^{h_{jt}/2}\eta_{jt},\quad
   h_{jt} = \phi_j h_{j,t-1} + \sigma_j\xi_{jt},

and non-proportional volatility paths identify :math:`B` (Lewis 2021;
Bertsche and Braun 2022) without declared dates. Estimation is a Gibbs
sampler (Waggoner-Zha rows of :math:`B^{-1}`, Kim-Shephard-Chib volatility
step). Unlike every other scheme, ``identify`` takes the sampler
settings: ``n_draws`` (default 3000), ``n_burn`` (1000), ``thin``,
``prior`` (a :class:`~cultivars.bayes.priors.VolatilityPrior`) and
``seed``, which **defaults to** ``None``; pass a seed for reproducible
output.

.. testcode::

   from cultivars.multivariate.structural.stochastic_volatility import (
       StochasticVolatilitySVAR,
   )

   rng = np.random.default_rng(0)
   h = np.zeros((400, 2))
   for t in range(1, 400):
       h[t] = [0.97, 0.6] * h[t - 1] + [0.2, 0.6] * rng.standard_normal(2)
   shocks_sv = np.exp(h / 2) * rng.standard_normal((400, 2))
   y_sv = np.zeros((400, 2))
   for t in range(1, 400):
       y_sv[t] = 0.4 * y_sv[t - 1] + B_s @ shocks_sv[t]
   var_sv = VAR(y_sv, order=1, names=("gdp", "infl")).fit()

   post = StochasticVolatilitySVAR(var_sv).identify(n_draws=300, n_burn=100, seed=0)
   print(post.shock_names, post.n_kept, post.impact_draws.shape)
   print(post.irf(4).shape, post.volatility_path("shock_gdp").shape)
   point = post.to_point()
   print(point.scheme)
   print(dict(point.diagnostics)["Identification"])

.. testoutput::

   ('shock_gdp', 'shock_infl') 200 (200, 2, 2)
   (3, 5, 2, 2) (3, 399)
   stochastic volatility (posterior median)
   variance ratios move over time for every pair

The result holds posterior draws (``impact_draws``, ``h_draws``,
``phi_draws``, ``sigma2_draws``); ``impact`` is the pointwise posterior
median; ``irf`` and ``fevd`` return posterior quantiles with the same
``[q, h, i, j]`` layout as the sign-restricted set; ``to_point()`` gives an
``SVARResult`` at the median impact. ``identification_test()`` returns
pairwise ``(statistics, p_values)`` for constant variance ratios; with
proportional paths the posterior is still sharp, so this test, not the
posterior spread, is what tells you whether identification holds.
Shock ``j`` is the one that loads most on variable ``j``, signed to a
positive diagonal, and scaled to its in-sample average variance.
``relabel_rate`` reports how often draws had to be permuted to that
labelling. Convergence diagnostics are available through
``convergence()`` (see :doc:`bayesian_workflow`).

The module also contains ``LongMemorySV``, a univariate long-memory
stochastic-volatility model estimated by Whittle likelihood. It
identifies nothing structural; see :doc:`volatility`.

Factor-augmented SVAR
---------------------

A FAVAR (:doc:`large_dimensional`) is a VAR in factors and observed
variables, :math:`(F_t, Y_t)`, plus an observation equation
:math:`x_t = \Lambda^F F_t + \Lambda^Y Y_t + e_t`. Identification happens
on the factor VAR (``favar.factors``, a ``VARResult``);
:class:`~cultivars.multivariate.structural.factor_augmented.FactorAugmentedSVAR`
then maps the structural responses onto every panel series in its own
units, :math:`\operatorname{diag}(s)\Lambda\Psi_hB`. By default it applies
the Bernanke-Boivin-Eliasz recursive scheme with the observed block last;
``order=`` changes the ordering, or pass any point-identified result built
on ``favar.factors`` as the second argument.

.. testcode::

   from cultivars.multivariate.large_dim.factor_augmented import FAVAR
   from cultivars.multivariate.structural.factor_augmented import FactorAugmentedSVAR

   rng = np.random.default_rng(0)
   factor = np.zeros(300)
   for t in range(1, 300):
       factor[t] = 0.7 * factor[t - 1] + rng.standard_normal()
   weights = rng.uniform(0.5, 1.5, 30)
   panel = np.outer(factor, weights) + 0.3 * rng.standard_normal((300, 30))
   policy = 0.4 * factor + rng.standard_normal(300)
   names = [f"x{i + 1}" for i in range(30)]
   favar = FAVAR(
       panel, policy[:, None], n_factors=1, order=1,
       slow=names[:15], panel_names=names, observed_names=("ffr",),
   ).fit()

   fsvar = FactorAugmentedSVAR(favar).identify()
   print(fsvar.shock_names, fsvar.irf(12).shape, fsvar.factor_irf(12).shape)
   print(fsvar.factor_irf(0)[0].round(3))

   mine = RecursiveSVAR(favar.factors).identify()
   same = FactorAugmentedSVAR(favar, mine).identify()
   print(bool(np.allclose(same.irf(4), fsvar.irf(4))))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   ('f1', 'ffr') (13, 30, 2) (13, 2, 2)
   [[ 3.734  0.   ]
    [-0.411  0.99 ]]
   True

``irf`` has shape ``(horizon + 1, n_series, s)``; ``factor_irf`` gives the
factor-space responses; ``structural`` and ``favar`` hold the two
components. The policy shock does not move the factor on impact but does
move the panel through :math:`\Lambda^Y`. If the FAVAR was fitted without
``slow=`` (slow-variable cleaning), the summary carries a contamination
note. The supplied identification must be built on this fit's own
``favar.factors`` (checked by identity).

DSGE perturbation models
------------------------

:class:`~cultivars.multivariate.structural.perturbation.PerturbationDSGE`
is not an identification scheme: it does not take a reduced-form result.
The structure comes from a model. You supply a specification object with
``parameter_names``, ``bounds``, ``n_states``, ``n_controls`` and the
methods ``equations(theta, y_next, y, x_next, x)``,
``steady_state(theta)``, ``shock_loading(theta)``,
``observation(theta)`` (returning ``(Z, d, H)``) and ``log_prior(theta)``.
The engine solves

.. math::

   y_t = g(x_t;\theta),\qquad x_{t+1} = h(x_t;\theta) + \eta(\theta)\varepsilon_{t+1}

to first or (pruned) second order by Schmitt-Grohé and Uribe (2004)
perturbation with numerical derivatives, and estimates :math:`\theta` by
the likelihood of the resulting state space: ``filter="kalman"`` (exact,
``order=1`` only), ``"extended"``, ``"unscented"`` (the default, also at
``order=1``) or ``"particle"``. ``sample()`` runs particle marginal
Metropolis-Hastings.

.. testcode::

   from cultivars.multivariate.structural.perturbation import PerturbationDSGE

   class Latent:
       parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
       n_states, n_controls = 1, 1

       def equations(self, th, y_next, y, x_next, x):
           return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])

       def steady_state(self, th):
           return np.zeros(1), np.zeros(1)

       def shock_loading(self, th):
           return np.array([[th[1]]])

       def observation(self, th):
           return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])

       def log_prior(self, th):
           return 0.0

   rng = np.random.default_rng(0)
   x = np.zeros(200)
   for t in range(1, 200):
       x[t] = 0.8 * x[t - 1] + 0.5 * rng.standard_normal()

   dsge = PerturbationDSGE(Latent(), x, order=1).fit([0.5, 0.3], filter="kalman")
   print({k: round(v, 2) for k, v in dsge.params.items()}, dsge.filter)
   states, controls = dsge.impulse_responses(horizon=4)
   print(states.shape, controls[0, :, 0].round(3))

.. testoutput::

   {'rho': 0.82, 'sigma': 0.48} kalman
   (1, 4, 1) [0.48  0.394 0.324 0.267]

Note the different layout from ``SVARResult.irf``:
``impulse_responses(horizon=H)`` returns ``(states, controls)`` with shape
``(n_shocks, H, n)``, shock first, ``H`` periods starting at impact (not
``H + 1``). ``fit`` requires a starting point although ``theta0`` defaults
to ``None`` in its signature. More on the state-space side in
:doc:`state_space`.

Comparing schemes on one fit
----------------------------

One reduced form can be passed to several schemes. The data below have a
non-triangular impact matrix, a heavy-tailed first shock whose standard
deviation rises 2.5-fold halfway through, and a noisy proxy for that
shock, so every scheme has its identifying information. Compare the first
shock's impact column, normalized to a unit effect on ``gdp``, and its raw
scale:

.. testcode::

   rng = np.random.default_rng(42)
   T = 602
   eps_c = np.column_stack([
       rng.laplace(size=T) / np.sqrt(2),
       rng.uniform(-np.sqrt(3), np.sqrt(3), size=T),
   ])
   eps_c[301:, 0] *= 2.5
   y_c = np.zeros((T, 2))
   for t in range(1, T):
       y_c[t] = A @ y_c[t - 1] + B_s @ eps_c[t]
   z_c = eps_c[1:, 0] + 0.5 * rng.standard_normal(T - 1)
   fit = VAR(y_c, order=1, names=("gdp", "infl")).fit()

   columns = {
       "recursive": RecursiveSVAR(fit).identify().impact[:, 0],
       "proxy": ProxySVAR(fit, z_c).identify().impact[:, 0],
       "heteroskedastic": HeteroskedasticSVAR(
           fit, ["early"] * 300 + ["late"] * 301
       ).identify().impact[:, 0],
       "non-Gaussian": NonGaussianSVAR(fit).identify().impact[:, 0],
       "sign (median)": SignRestrictedSVAR(
           fit, demand, draws=200, seed=0
       ).identify().irf(0, quantiles=(0.5,))[0, 0, :, 0],
       "SV (median)": StochasticVolatilitySVAR(fit).identify(
           n_draws=300, n_burn=100, seed=0
       ).impact[:, 0],
   }
   print(f"{'truth':16s} infl/gdp = {B_s[1, 0] / B_s[0, 0]:.3f}")
   for label, col in columns.items():
       print(f"{label:16s} infl/gdp = {col[1] / col[0]:.3f}   gdp = {col[0]:.3f}")
   print(SetIdentifiedSVAR(fit, {"gdp": "+", "infl": "+"}).identify().impact_bounds.round(3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   truth            infl/gdp = 0.300
   recursive        infl/gdp = 0.441   gdp = 1.922
   proxy            infl/gdp = 0.329   gdp = 1.860
   heteroskedastic  infl/gdp = 0.305   gdp = 0.899
   non-Gaussian     infl/gdp = 0.338   gdp = 1.864
   sign (median)    infl/gdp = 0.634   gdp = 1.565
   SV (median)      infl/gdp = 0.313   gdp = 1.407
   [[0.    1.922]
    [0.    1.182]]

Three readings. First, the recursive ordering is wrong here and the
column shows it (0.44 against 0.30); the sign median is a summary of a
wide set, not an estimate of the truth. Second, the proxy, heteroskedastic,
non-Gaussian and SV columns agree on the direction. Third, they disagree
on **scale**, because "one standard deviation" means different things: the
proxy and non-Gaussian columns are full-sample standard deviations of a
shock whose variance changed (about :math:`\sqrt{(1 + 6.25)/2} = 1.9`),
the heteroskedastic column is one standard deviation in the first regime
(about 1), and the SV column is at the in-sample average log variance.
Compare schemes on unit-effect normalized responses, never on raw
one-s.d. responses.

How the pieces connect
----------------------

* **Reduced forms in.** Any result satisfying ``ClosedSystemResult``
  (:doc:`var`, :doc:`var_extensions`) can be identified. The same protocol
  feeds the spectral tools (:doc:`spectral`) and
  :class:`~cultivars.multivariate.large_dim.spillover.Spillover`
  (:doc:`large_dimensional`).
* **Structural results out.** Point results satisfy the
  ``StructuralResult`` protocol (``source`` plus
  ``irf(horizon, *, cumulative)``); that is what
  ``FactorAugmentedSVAR`` consumes, and what
  :class:`~cultivars.multivariate.regime_switching.markov_switching.MarkovSwitchingSVAR`
  packages per regime (:doc:`nonlinear_multivariate`).
* **Time-varying structure.**
  :class:`~cultivars.multivariate.nonlinear.time_varying.TVPSVAR`
  identifies a TVP-VAR recursively per period; see
  :doc:`nonlinear_multivariate`.
* **Bayesian tools.** ``StochasticVolatilitySVARResult`` and
  ``PerturbationDSGEPosterior`` expose ``convergence()``
  (:doc:`bayesian_workflow`). Sign-restricted sets do not; they are not
  Markov chains.

Nuances and gotchas
-------------------

* **Construct, then identify.** ``var.identify(Scheme(var, ...))`` raises
  ``TypeError`` (and ``var.identify(Scheme)`` raises ``AttributeError``);
  the working form is ``Scheme(var, ...).identify()``.
* **Effective sample.** Proxy instrument rows, heteroskedasticity regime
  labels and narrative event periods all index the residual rows,
  ``nobs`` of them, which start at data row ``p``. Passing a full-length
  series raises :class:`~cultivars.exceptions.DimensionError` (proxy) or
  :class:`~cultivars.exceptions.SpecificationError` (regimes); a narrative
  period ``t`` silently refers to data row ``t + p``.
* **Index order.** ``SVARResult.irf`` is ``[h, i, j]``; quantile surfaces
  (sign sets, SV posteriors) are ``[q, h, i, j]``; draws are
  ``[n, h, i, j]``; set bounds are ``[h, i, (lower, upper)]``; FAVAR panel
  responses are ``[h, series, j]``; DSGE responses are ``[shock, h, n]``.
* **Units.** Every scheme reports responses to one-standard-deviation
  shocks, but the standard deviation is the full sample's (most schemes),
  the first regime's (heteroskedasticity), or the in-sample average
  (SV). Divide a column by its impact on a reference variable to compare.
* **Sign normalizations differ by scheme.** Recursive and long-run:
  positive diagonal of the triangular factor. ``ShortRunSVAR``,
  ``MixedSVAR`` (all-zero columns), heteroskedastic and non-Gaussian:
  largest-magnitude entry of each column positive. Proxy: the
  ``normalize`` variable positive. SV: positive diagonal. Sign
  restrictions: declared signs; unrestricted columns carry an arbitrary
  sign, so their quantile bands mix both orientations:

  .. testcode::

     share = float((plain.impacts[:, 0, 1] > 0).mean())
     print(0.3 < share < 0.7)

  .. testoutput::

     True

* **Column flipping, never relabelling.** A sign-restricted column whose
  negative passes is flipped wholesale; columns are never permuted to
  rescue a draw. A shock's identity is its declared position.
* **No rotation accepted.** ``SpecificationError`` from
  ``SignRestrictedSVAR.identify()`` or
  ``NarrativeSignRestrictedSVAR.identify()`` means no draw passed within
  ``max_attempts``: shorten ``horizons``, loosen or drop a sign, check
  narrative dates against the effective sample, or raise
  ``max_attempts``. The same situation in ``SetIdentifiedSVAR`` (an empty
  set) is a ``NumericalError`` instead.
* **Partial sets are silent.** When the budget runs out after at least one
  acceptance, the result has ``n_accepted < requested`` and no warning.
  Check ``n_accepted`` before reading quantiles.
* **Seeds.** Sign and narrative schemes default to ``seed=0`` (same set
  every run); ``seed=None`` draws fresh rotations. The SV scheme defaults
  to ``seed=None``.
* **Impact-only vs multi-horizon signs.** ``horizons=(0,)`` restricts
  impact only; ``horizons=(0, 1, 2)`` requires the sign at every listed
  lead, not just at some. There is no per-variable horizon; declare the
  strictest common set or use ``SetIdentifiedSVAR`` with ``zeros``.
* **Statistical labels.** Heteroskedastic and non-Gaussian columns are
  reordered by variance ratio or kurtosis, so ``shock_names`` you pass
  label positions in that order, not variables. Check the impact columns
  before naming them.
* **Statistical schemes and the shared result surface.** For
  ``HeteroskedasticSVAR`` (and, by the degrees-of-freedom factor,
  ``NonGaussianSVAR``) ``impact @ impact.T`` is not ``sigma_u``;
  ``fevd`` rows then miss one and ``structural_shocks`` is mis-scaled.
  See the heteroskedasticity section for the direct computation.
* **No inference on point schemes.** ``SVARResult`` has no confidence
  bands. For recursive bands on a Bayesian VAR, use
  ``BVARResult.irf_bands`` (:doc:`bayesian_vars`).
* **Module spelling.** The heteroskedasticity module is
  ``cultivars.multivariate.structural.heteroskedacity``.
* **Narrative class import.** ``NarrativeSignRestrictedSVAR`` is not in
  the module's ``__all__``, so ``from ...sign_restrictions import *`` does
  not bring it in; import it by name.
* **FAVAR with a set.** ``FactorAugmentedSVAR`` accepts a
  sign-restricted set as its ``structural`` argument without complaint,
  but ``irf`` then fails with a NumPy ``ValueError``; pass point-identified
  results only.
* **Long-run schemes need stationarity.** ``LongRunSVAR`` and
  ``MixedSVAR`` reject explosive systems at construction and unit-root
  systems (for example a ``VECMResult``) at ``identify()``.

See also
--------

* :doc:`var` and :doc:`var_extensions` for the reduced forms these
  schemes consume.
* :doc:`bayesian_vars` for posterior impulse-response bands.
* :doc:`large_dimensional` for FAVAR and spillover tables.
* :doc:`nonlinear_multivariate` for Markov-switching and TVP structural
  VARs.
* :doc:`state_space` for the filters behind ``PerturbationDSGE``.
* :doc:`conventions` and :doc:`nuances` for package-wide conventions.
* API: :mod:`cultivars.multivariate.structural.zero_restrictions`,
  :mod:`cultivars.multivariate.structural.sign_restrictions`,
  :mod:`cultivars.multivariate.structural.set_identification`,
  :mod:`cultivars.multivariate.structural.external_instruments`,
  :mod:`cultivars.multivariate.structural.heteroskedacity`,
  :mod:`cultivars.multivariate.structural.non_gaussian`,
  :mod:`cultivars.multivariate.structural.stochastic_volatility`,
  :mod:`cultivars.multivariate.structural.factor_augmented`,
  :mod:`cultivars.multivariate.structural.perturbation`.
