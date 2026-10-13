.. meta::
   :description: Shapes, names, samples, array index order, trends, shock scales, seeds and frequency grids: the conventions every cultivars page assumes.

Conventions
===========

This page collects the rules that hold across the library: how data go
in, how samples shrink, how returned arrays are indexed, and what the
common options mean. The family pages assume them. Where a family breaks
a rule, :doc:`nuances` says so.

.. contents:: On this page
   :local:
   :depth: 1

Data in
-------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Rule
     - Detail
   * - Shape
     - One series: any 1-D array-like of length ``n``. A system: an
       ``(n, k)`` array-like, one column per variable. Rows are periods,
       oldest first.
   * - Types
     - Lists, NumPy arrays, pandas and polars objects are accepted and
       copied to ``float64``. The input is never modified.
   * - Names
     - Pass ``names=`` to label variables. Dataframe column names are
       **not** read; without ``names``, variables are ``y1``, ``y2``, ...
       Every method that takes a variable (``granger_causality``,
       ``equation``, ``spectrum``) takes these labels.
   * - Dates
     - Estimation is positional. Attach dates on the way out with
       ``to_pandas(index=...)`` where it exists, or index the returned
       arrays yourself.
   * - Missing values
     - NaN and infinity raise
       :class:`~cultivars.exceptions.NumericalError`. Only the
       state-space engines accept gaps; see :doc:`state_space` for which
       engine accepts which pattern.

The effective sample
--------------------

Lags and differences consume observations at the start of the sample.
Every per-observation output (``resid``, ``fittedvalues``, structural
shocks, historical decompositions) covers the *effective* sample only,
and ``nobs`` counts it:

.. testcode::

   import numpy as np
   from cultivars.univariate.autoregression import AR
   from cultivars.univariate.box_jenkins import ARIMA
   from cultivars.multivariate.reduced_form.vector_autoregression import VAR

   rng = np.random.default_rng(1)
   x = rng.standard_normal(100)
   y = np.zeros((200, 2))
   for t in range(1, 200):
       y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)

   print(AR(x, order=3).fit().nobs)                        # n - p
   print(ARIMA(np.cumsum(x), order=(1, 1, 0)).fit().nobs)  # n - d
   var = VAR(y, order=2, names=("a", "b")).fit()
   print(var.nobs, var.resid.shape)                         # n - p rows

.. testoutput::

   97
   99
   198 (198, 2)

Row ``i`` of ``resid`` therefore belongs to period ``i + p`` of the
input. To line results up with dates, drop the first ``p`` (or ``d``,
or ``p + d``) dates. Two consequences follow:

- **Comparisons need one sample.** ``compare`` and
  ``likelihood_ratio_test`` refuse results with different ``nobs``.
  To compare a VAR(1) with a VAR(2), fit both on data trimmed to the
  same effective sample; :doc:`diagnostics` shows how.
- **Event dates count on the effective sample.** A narrative sign
  restriction at "August 1990" takes the position of that month among
  the effective-sample rows, not among the raw rows; see
  :doc:`structural_identification`.

Index order of returned arrays
------------------------------

Arrays from closed linear systems and their structural results follow
one order, with time first:

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Method
     - Shape
     - Entry ``[a, b, c]`` is
   * - ``coefficients``
     - ``(p, k, k)``
     - lag ``a + 1``: coefficient of variable ``c`` in the equation for
       ``b``. ``coefficients[0]`` is :math:`A_1`.
   * - ``forecast(steps)``
     - ``(steps, k)``
     - horizon ``a + 1``, variable ``b``
   * - ``irf(h)``, ``ma_representation(h)``
     - ``(h + 1, k, k)``
     - horizon ``a`` (0 is impact), response of variable ``b`` to shock
       ``c``
   * - ``fevd(h)``
     - ``(h + 1, k, k)``
     - horizon ``a``, share of variable ``b``'s forecast-error variance
       due to shock ``c``; each ``[a, b, :]`` sums to one
   * - ``historical_decomposition()``
     - ``(nobs, k, k)``
     - period ``a`` of the effective sample, contribution of shock ``c``
       to variable ``b``
   * - ``simulate(n, seed=...)``
     - ``(n, k)``
     - a simulated path, after ``burn`` discarded periods

.. testcode::

   print(var.coefficients.shape, var.forecast(4).shape)
   print(var.irf(12).shape, var.fevd(12)[12].sum(axis=1))
   print(var.historical_decomposition().shape, var.simulate(50, seed=0).shape)

.. testoutput::

   (2, 2, 2) (4, 2)
   (13, 2, 2) [1. 1.]
   (198, 2, 2) (50, 2)

Univariate results return 1-D arrays of length ``nobs`` for
``resid`` and ``fittedvalues``. Parameter dictionaries (``params``)
keep insertion order: constant first, then lags in order, then variance
parameters.

Shock scale and ordering
------------------------

- ``irf(h)`` on a reduced form is *orthogonalized* by default: the
  shocks are the columns of the Cholesky factor of :math:`\Sigma_u` in
  the order of ``names``. A shock is one standard deviation, so the
  impact matrix ``irf(h)[0]`` is the Cholesky factor.
  ``orthogonalized=False`` gives responses to a unit innovation in one
  equation, with the identity matrix as the impact.
- Structural results label shocks with ``shock_names``. Recursive and
  long-run schemes name them after the variables. Schemes differ in
  what "one standard deviation" means (full sample, first regime,
  average volatility) and in how they fix each shock's sign. Compare
  schemes on normalized responses, not raw ones; see
  :doc:`structural_identification`.
- The ordering in ``names`` matters only where a scheme uses it
  (recursive identification, orthogonalized reduced-form responses,
  the TVP-SVAR). Everything else is invariant to it.

Deterministic terms
-------------------

``trend`` takes one of three codes across models
(:data:`cultivars.typing.Trend`):

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Code
     - Meaning
   * - ``"n"``
     - no deterministic term
   * - ``"c"``
     - a constant (the default almost everywhere)
   * - ``"ct"``
     - a constant and a linear time trend

Error-correction models use the five Johansen cases instead
(``"none"``, ``"restricted_constant"``, ``"constant"``,
``"restricted_trend"``, ``"trend"``), because the deterministic term can
sit inside or outside the cointegrating relation; see
:doc:`var_extensions`. A reported constant is an *intercept*, not the
mean: for an AR(1), the mean is ``const / (1 - phi)``.

Randomness and draws
--------------------

- Every stochastic routine takes ``seed``: an integer, a
  :class:`numpy.random.Generator`, or ``None`` for fresh entropy. Defaults
  differ: sign-restriction samplers default to ``seed=0`` and are
  reproducible out of the box, while posterior samplers default to
  ``None``. Pass ``seed`` explicitly in anything you publish.
- ``n_draws`` does not mean the same thing everywhere. For the
  conjugate :class:`~cultivars.multivariate.large_dim.bayesian.BVAR`
  and the hierarchical BVAR it is the number of draws *kept*. For the
  Gibbs, Student-t, stochastic-volatility and factor-SV samplers it is
  the total number of iterations, including burn-in and before
  thinning. Read ``n_kept`` on the result to see what you have.

Information criteria
--------------------

Likelihood-based results report AIC, BIC and HQIC through
``information_criteria``:

.. math::

   \mathrm{AIC} = -2\ell + 2k, \qquad
   \mathrm{BIC} = -2\ell + k \log n, \qquad
   \mathrm{HQIC} = -2\ell + 2k \log\log n,

with :math:`\ell` the maximized log-likelihood, :math:`n` = ``nobs`` and
:math:`k` = ``n_params``. Families count :math:`k` differently: VARs
and Markov-switching models include variance parameters, while SETAR
and STAR leave the innovation variance out. Compare across families
with that in mind, and within a family freely.

Frequencies
-----------

Frequency-domain results use angular frequency :math:`\omega` in
radians per period on :math:`[0, \pi]`, with 256 grid points by default
(``n_frequencies``). The period of a frequency is
:math:`2\pi / \omega` periods, so business-cycle frequencies in
quarterly data (6 to 32 quarters) are
:math:`\omega \in [2\pi/32, 2\pi/6]`. Band-pass filters take their band
in periods, not radians.

See also
--------

- :doc:`architecture` for which results expose which methods.
- :doc:`nuances` for the exceptions to these rules.
