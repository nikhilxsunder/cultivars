.. meta::
   :description: Unit-root, seasonal, long-memory, break, cointegration-rank, nonlinearity and residual tests in cultivars, and how to read their records.

Diagnostics and specification testing
=====================================

:mod:`cultivars.diagnostics` holds the tests you run before a model is
chosen: does a series carry a stochastic trend, is its seasonal pattern
fixed, how many cointegrating relations does a system have, did the
regression break, is a linear autoregression wrong in the way a regime
model would fix. Every test returns a record with one ``reject`` and one
``summary``, and the tests a fitted model runs on its own residuals
(portmanteau, normality, ARCH, Granger causality) return the same
records from the result object. This page does not cover forecast
comparison tests (see :doc:`forecasting`) or MCMC convergence
diagnostics (see :doc:`bayesian_workflow`).

.. contents:: On this page
   :local:
   :depth: 1

The family at a glance
----------------------

Each module is a leaf: import from the module, never from
``cultivars.diagnostics`` itself.

.. list-table::
   :header-rows: 1
   :widths: 22 18 30 30

   * - Producer
     - Module
     - Null hypothesis
     - Reference distribution / rejects in
   * - :func:`~cultivars.diagnostics.unit_roots.adf`,
       :func:`~cultivars.diagnostics.unit_roots.phillips_perron`
     - ``unit_roots``
     - unit root
     - MacKinnon (1996) response surface; lower tail
   * - :func:`~cultivars.diagnostics.unit_roots.dfgls`,
       :func:`~cultivars.diagnostics.unit_roots.ng_perron`
     - ``unit_roots``
     - unit root
     - asymptotic critical values only (``pvalue=None``); lower tail
   * - :func:`~cultivars.diagnostics.unit_roots.zivot_andrews`
     - ``unit_roots``
     - unit root without a break
     - asymptotic critical values only; lower tail
   * - :func:`~cultivars.diagnostics.unit_roots.kpss`
     - ``unit_roots``
     - level or trend stationarity
     - KPSS table, p-value interpolated in ``[0.01, 0.10]``; upper tail
   * - :func:`~cultivars.diagnostics.seasonality.hegy`
     - ``seasonality``
     - unit root at each seasonal frequency
     - simulated finite-sample law; ``t`` rows lower, ``F`` rows upper
   * - :func:`~cultivars.diagnostics.seasonality.canova_hansen`
     - ``seasonality``
     - stationary seasonality
     - simulated von Mises law; upper tail
   * - :func:`~cultivars.diagnostics.long_memory.gph`,
       :func:`~cultivars.diagnostics.long_memory.local_whittle`,
       :func:`~cultivars.diagnostics.long_memory.exact_local_whittle`
     - ``long_memory``
     - :math:`d = d_0` (``0`` as built)
     - normal, two-sided
   * - :func:`~cultivars.diagnostics.breaks.sup_wald`
     - ``breaks``
     - no break in the coefficients
     - simulated Brownian-bridge functionals; upper tail
   * - :func:`~cultivars.diagnostics.breaks.bai_perron`
     - ``breaks``
     - :math:`l` breaks against :math:`l + 1` (sequential)
     - simulated; selection record, not a test
   * - :func:`~cultivars.diagnostics.breaks.cusum`
     - ``breaks``
     - constant coefficients (and variance, ``squares=True``)
     - Brown-Durbin-Evans boundary (no p-value) or simulated; upper tail
   * - ``VECM(...).rank_test()``
     - ``cointegration`` (record)
     - cointegrating rank :math:`\le r`
     - simulated Johansen limit
   * - :func:`~cultivars.diagnostics.nonlinearity.terasvirta`,
       :func:`~cultivars.diagnostics.nonlinearity.tsay`,
       :func:`~cultivars.diagnostics.nonlinearity.ramsey_reset`
     - ``nonlinearity``
     - linear AR(:math:`p`)
     - :math:`F`; upper tail
   * - :func:`~cultivars.diagnostics.nonlinearity.hansen_threshold`
     - ``nonlinearity``
     - linear AR(:math:`p`)
     - fixed-regressor bootstrap; upper tail
   * - :func:`~cultivars.diagnostics.nonlinearity.bds`
     - ``nonlinearity``
     - i.i.d.
     - normal, two-sided
   * - ``result.portmanteau_test``, ``normality_test``, ``arch_test``,
       ``granger_causality``, ``likelihood_ratio_test``
     - fitted VAR-family results
     - named on the record
     - :math:`\chi^2`; upper tail

The test record
---------------

Every hypothesis test descends from
:class:`~cultivars.diagnostics.hypothesis.HypothesisTest`, which holds
two fields, ``statistic`` and ``pvalue`` (possibly ``None``), and two
methods: ``reject(*, alpha=0.05)`` and ``summary()``. ``summary()``
returns a :class:`~cultivars.summary.SummaryTable` (``to_text``,
``to_pandas``, ``to_polars``); ``print(record)`` renders that table and
``repr(record)`` is a one-line verdict. A study that collects a dozen
verdicts therefore holds one kind of object.

.. testcode::

   import numpy as np
   from cultivars.diagnostics.unit_roots import adf, dfgls, kpss, ng_perron

   rng = np.random.default_rng(0)
   walk = np.cumsum(rng.standard_normal(300))
   test = adf(walk)
   print(test)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

                                       ADF Test
   ================================================================================
   Null:                        unit root  Trend:                               c
   Lags:                                0  Method:                   aic (max 15)
   Observations:                      299  Verdict at 5%:          keep unit root
   --------------------------------------------------------------------------------
   statistic       value   p-value       1%       5%      10%
   ADF           -0.7203    0.8414   -3.452   -2.871   -2.572
   ================================================================================
   Rejection lies in the lower tail.

The unit-root and break records are
:class:`~cultivars.diagnostics.hypothesis.TabulatedTest` subclasses and
carry more: ``name``, ``null`` (a sentence), ``nobs`` (the effective
sample), ``critical_values`` (keys ``"1%"``, ``"5%"``, ``"10%"``),
``lower_tail`` (which way rejection lies) and ``companions`` (further
statistics of the same test on the same sample, each a full record).

.. testcode::

   print(test.name, test.null, test.lower_tail, test.nobs, test.lags)
   print({k: round(v, 3) for k, v in test.critical_values.items()})
   print(test.reject(), test.reject(alpha=0.10))

.. testoutput::

   ADF unit root True 299 0
   {'1%': -3.452, '5%': -2.871, '10%': -2.572}
   False False

A record known only through three tabulated critical values has
``pvalue=None``. Its ``reject`` reads the table, and refuses any level
that is not 1%, 5% or 10% rather than interpolate:

.. testcode::

   from cultivars.exceptions import SpecificationError

   gls = dfgls(walk)
   print(gls.pvalue, gls.reject(alpha=0.05))
   try:
       gls.reject(alpha=0.03)
   except SpecificationError as err:
       print(err)

.. testoutput::

   None False
   DF-GLS carries no p-value; alpha must be one of ('1%', '5%', '10%') to use the tabulated critical values, got 0.03.

Companions travel with the headline statistic. Ng-Perron's
:math:`MZ_t` carries :math:`MZ_\alpha`, :math:`MSB` and :math:`MP_T`;
the sup-Wald carries exp- and ave-Wald; HEGY carries one row per
frequency:

.. testcode::

   mz = ng_perron(walk)
   for rec in (mz, *mz.companions):
       print(f"{rec.name:<15} {rec.statistic:8.3f}  lower_tail={rec.lower_tail}  reject={rec.reject()}")

.. testoutput::

   Ng-Perron MZ_t    -0.922  lower_tail=True  reject=False
   MZ_a              -2.781  lower_tail=True  reject=False
   MSB                0.331  lower_tail=True  reject=False
   MP_T               8.106  lower_tail=True  reject=False

The record classes, by module:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Class
     - Base
     - Extra fields worth knowing
   * - :class:`~cultivars.diagnostics.hypothesis.ChiSquaredTest`
     - ``HypothesisTest``
     - ``df``
   * - :class:`~cultivars.diagnostics.hypothesis.WaldTest`
     - ``ChiSquaredTest``
     - ``df``, ``null`` (a sentence: ``"z does not Granger-cause x"``)
   * - :class:`~cultivars.diagnostics.hypothesis.LikelihoodRatioTest`
     - ``ChiSquaredTest``
     - ``df``; no ``null``
   * - :class:`~cultivars.diagnostics.unit_roots.UnitRootTest`
     - ``TabulatedTest``
     - ``trend``, ``lags``, ``method``, ``break_index``
   * - :class:`~cultivars.diagnostics.seasonality.SeasonalUnitRootTest`
     - ``UnitRootTest``
     - ``family``, ``period``
   * - :class:`~cultivars.diagnostics.breaks.BreakTest`
     - ``TabulatedTest``
     - ``break_index``, ``n_restrictions``, ``trimming``, ``path``,
       ``bounds``
   * - :class:`~cultivars.diagnostics.long_memory.LongMemoryEstimate`
     - ``HypothesisTest``
     - ``d``, ``se``, ``null_d``, ``bandwidth``; ``test(null)``,
       ``confidence_interval()``, ``is_stationary``
   * - :class:`~cultivars.diagnostics.nonlinearity.LinearityTest`
     - ``HypothesisTest``
     - ``df``, ``null``, ``alternative``, ``order``, ``delay``,
       ``threshold``, ``suggestion``, ``companions``
   * - :class:`~cultivars.diagnostics.nonlinearity.BDSTest`
     - ``HypothesisTest``
     - ``dimension``, ``epsilon``, ``companions``

Three records are *not* hypothesis tests and have no ``reject``:
:class:`~cultivars.diagnostics.breaks.MultipleBreakTest` (a selection
over break counts, with ``summary()``),
:class:`~cultivars.diagnostics.cointegration.JohansenRankTest` (two
sequences, read with ``selected_rank()`` and rendered with
``to_table()``) and the deterministic
:class:`~cultivars.diagnostics.stability.StabilityAssessment`.

.. testcode::

   from cultivars.diagnostics.breaks import sup_wald
   from cultivars.diagnostics.hypothesis import HypothesisTest, TabulatedTest
   from cultivars.diagnostics.long_memory import exact_local_whittle
   from cultivars.diagnostics.nonlinearity import tsay

   records = [
       adf(walk),
       kpss(walk),
       sup_wald(np.diff(walk), n_draws=1000, grid=500, seed=0),
       exact_local_whittle(walk),
       tsay(walk, order=1),
   ]
   for rec in records:
       print(f"{type(rec).__name__:<19} {isinstance(rec, HypothesisTest)} "
             f"{isinstance(rec, TabulatedTest)!s:<5} {rec.reject()}")

.. testoutput::

   UnitRootTest        True True  False
   UnitRootTest        True True  True
   BreakTest           True True  False
   LongMemoryEstimate  True False True
   LinearityTest       True False False

Which test for which question
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 35 25

   * - Question
     - Run
     - Read alongside
   * - Does the series have a stochastic trend?
     - ``adf`` (or ``dfgls`` for power near the null)
     - ``kpss`` with the same ``trend``
   * - Errors with a large negative MA root?
     - ``ng_perron`` (default modified AIC)
     - ``adf``; ``phillips_perron`` over-rejects here
   * - Unit root, or stationary around a broken trend?
     - ``zivot_andrews``
     - ``adf(trend="ct")``, ``sup_wald`` on the differences
   * - Fixed or wandering seasonal pattern?
     - ``hegy``
     - ``canova_hansen``
   * - :math:`d = 1` or :math:`d = 0.7`? Long memory in a stationary series?
     - ``exact_local_whittle``
     - ``local_whittle`` (:math:`d < 3/4`), ``gph``
   * - Did a regression break at an unknown date?
     - ``sup_wald``
     - ``cusum``
   * - How many breaks, and where?
     - ``bai_perron``
     - ``sup_wald`` per segment (``sequential``)
   * - How many cointegrating relations?
     - ``VECM(...).rank_test()``
     - unit-root tests on each series first
   * - Threshold autoregression?
     - ``tsay``, ``hansen_threshold``
     - ``arch_test`` on the linear residuals
   * - Smooth transition, logistic or exponential?
     - ``terasvirta`` (``suggestion``)
     - ``ramsey_reset``
   * - Anything left in the residuals at all?
     - ``bds``
     - an ARCH test first
   * - Lag order of a VAR?
     - ``model.lag_order_selection()``
     - ``result.portmanteau_test()``
   * - Is a fitted VAR stable?
     - ``result.stability_check()``
     - ``StabilityAssessment``

Unit roots and stationarity
---------------------------

The Dickey-Fuller family tests :math:`\gamma = 0` in

.. math::

   \Delta y_t = d_t + \gamma\, y_{t-1}
   + \sum_{j=1}^{p} \phi_j\, \Delta y_{t-j} + e_t,

where :math:`d_t` is nothing, a constant, or a constant and trend
(``trend="n"``, ``"c"``, ``"ct"``). Under the null the :math:`t` ratio
converges to a functional of Brownian motion whose quantiles sit far
left of the normal's (Dickey and Fuller 1979). KPSS reverses the null
(Kwiatkowski et al. 1992): it regresses :math:`y_t` on :math:`d_t` and
scales the squared partial sums of the residuals by a long-run variance
estimate,

.. math::

   \eta = \frac{T^{-2} \sum_{t=1}^{T} S_t^2}{\hat\lambda^2},
   \qquad S_t = \sum_{i \le t} \hat e_i ,

rejecting in the upper tail. The module offers:

- :func:`~cultivars.diagnostics.unit_roots.adf`: augmented
  Dickey-Fuller, augmentation chosen on a common sample, MacKinnon
  (1996) p-values (Said and Dickey 1984).
- :func:`~cultivars.diagnostics.unit_roots.phillips_perron`: the
  un-augmented :math:`t` with a kernel correction (Phillips and Perron
  1988).
- :func:`~cultivars.diagnostics.unit_roots.dfgls`: GLS-detrended
  Dickey-Fuller (Elliott, Rothenberg and Stock 1996), more powerful near
  the null; ``pvalue=None``.
- :func:`~cultivars.diagnostics.unit_roots.ng_perron`: the :math:`M`
  statistics (Ng and Perron 2001), sized correctly under a negative MA
  root; ``pvalue=None``.
- :func:`~cultivars.diagnostics.unit_roots.zivot_andrews`: unit root
  against a one-time break at an unknown date (Zivot and Andrews 1992);
  ``pvalue=None``.
- :func:`~cultivars.diagnostics.unit_roots.kpss`: stationarity null.
- :func:`~cultivars.diagnostics.unit_roots.long_run_variance`: the
  Bartlett or quadratic-spectral kernel estimator the corrected tests
  divide by.

There is no Phillips-Perron :math:`Z_\alpha`, no Lumsdaine-Papell or
Lee-Strazicich two-break test, and no panel unit-root test.

.. testcode::

   rng = np.random.default_rng(0)
   T = 300
   walk = np.cumsum(rng.standard_normal(T))
   ar = np.zeros(T)
   for t in range(1, T):
       ar[t] = 0.6 * ar[t - 1] + rng.standard_normal()
   trending = 0.05 * np.arange(T) + ar

   for f in (adf, kpss):
       for name, y in (("walk", walk), ("ar", ar)):
           rec = f(y)
           print(f"{rec.name:<5} {name:<5} stat={rec.statistic:7.3f}  "
                 f"p={rec.pvalue:.3f}  reject {rec.null}: {rec.reject()}")

.. testoutput::

   ADF   walk  stat= -0.720  p=0.841  reject unit root: False
   ADF   ar    stat= -9.570  p=0.000  reject unit root: True
   KPSS  walk  stat=  1.450  p=0.010  reject stationarity: True
   KPSS  ar    stat=  0.092  p=0.100  reject stationarity: False

Confirmatory analysis
~~~~~~~~~~~~~~~~~~~~~

Because the two families put opposite hypotheses under the null, read
them together. Rejection by one and not the other is a classification;
agreement in either direction is not.

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - ADF rejects unit root
     - KPSS rejects stationarity
     - Reading
   * - yes
     - no
     - I(0) around the deterministic terms tested
   * - no
     - yes
     - I(1)
   * - yes
     - yes
     - conflict: check for a break (``zivot_andrews``), fractional
       integration (``exact_local_whittle``), or a persistent
       stationary root, where KPSS over-rejects
   * - no
     - no
     - the sample cannot decide; low power, often a short sample

.. testcode::

   def classify(y, trend="c"):
       unit = adf(y, trend=trend).reject()
       stationary = kpss(y, trend=trend).reject()
       if unit and not stationary:
           return "I(0)"
       if stationary and not unit:
           return "I(1)"
       return "both reject" if unit else "neither rejects"

   for name, y in (("walk", walk), ("ar", ar), ("trending", trending)):
       print(f"{name:<9} trend='c': {classify(y):<6} trend='ct': {classify(y, 'ct')}")

.. testoutput::

   walk      trend='c': I(1)   trend='ct': I(1)
   ar        trend='c': I(0)   trend='ct': I(0)
   trending  trend='c': I(1)   trend='ct': I(0)

The ``trending`` row is the standard trap: a trend-stationary series
tested with ``trend="c"`` looks integrated. The deterministic terms must
include everything the stationary alternative needs.

A broken trend has the same effect (Perron 1989). Zivot-Andrews searches
the break date and rejects where the plain tests do not:

.. testcode::

   from cultivars.diagnostics.unit_roots import zivot_andrews

   t = np.arange(T)
   broken = 0.02 * t + 6.0 * (t >= 180) + ar
   za = zivot_andrews(broken, model="c")
   print(classify(broken, "ct"))
   print(za.reject(), za.break_index, za.pvalue, za.critical_values)

.. testoutput::

   I(1)
   True 180 None {'1%': -5.34, '5%': -4.8, '10%': -4.58}

Specification choices
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Argument
     - Default
     - Meaning
   * - ``trend``
     - ``"c"``
     - ``"n"``, ``"c"``, ``"ct"``; KPSS, DF-GLS and Ng-Perron accept
       only ``"c"`` and ``"ct"``. Zivot-Andrews takes ``model``
       (``"c"`` level break, ``"t"`` slope break, ``"ct"`` both) and
       always reports ``trend="ct"``.
   * - ``lags``
     - ``None`` (select)
     - augmentation for ADF, DF-GLS, Ng-Perron, Zivot-Andrews
   * - ``max_lags``
     - Schwert: :math:`\lfloor 12 (T/100)^{1/4} \rfloor`
     - ceiling for the search (15 at :math:`T = 300`)
   * - ``method``
     - ``"aic"`` (ADF), ``"maic"`` (DF-GLS, Ng-Perron)
     - ``"aic"``, ``"bic"``, ``"t-stat"``, ``"maic"``. Zivot-Andrews
       always uses the t-statistic rule on the no-break regression.
   * - ``kernel``
     - ``"bartlett"``
     - Phillips-Perron and KPSS; or ``"quadratic-spectral"``
   * - ``bandwidth``
     - Newey-West :math:`\lfloor 4 (T/100)^{2/9} \rfloor` (Bartlett),
       Andrews plug-in (QS)
     - kernel bandwidth; reported in the record's ``lags`` field
   * - ``trimming``
     - ``0.15``
     - Zivot-Andrews: fraction excluded at each end of the date search

The ``lags`` field means an augmentation count for the Dickey-Fuller
regressions and a kernel bandwidth for Phillips-Perron and KPSS;
``method`` says which. Note DF-GLS: on this plainly stationary AR(1)
the modified AIC picks six lags where AIC picks none, the known cost of
``"maic"`` far from the null.

.. testcode::

   from cultivars.diagnostics.unit_roots import long_run_variance, phillips_perron

   for rec in (adf(ar), phillips_perron(ar), dfgls(ar), kpss(ar)):
       print(f"{rec.name:<16} lags={rec.lags}  method={rec.method!r}")
   print(round(long_run_variance(ar), 2), round(long_run_variance(ar, bandwidth=20), 2))

.. testoutput::

   ADF              lags=0  method='aic (max 15)'
   Phillips-Perron  lags=5  method='bartlett kernel'
   DF-GLS           lags=6  method='maic (max 15)'
   KPSS             lags=5  method='bartlett kernel'
   3.02 4.17

The true long-run variance of this AR(1) is :math:`1/(1 - 0.6)^2 =
6.25`; the Newey-West default (5 lags here) reaches about half of it,
and a wider bandwidth moves closer. KPSS and Phillips-Perron inherit
that downward bias on persistent series.

Seasonal unit roots
-------------------

Seasonal differencing removes a unit root at every frequency
:math:`\omega_k = 2\pi k / s`,

.. math::

   1 - L^s = (1 - L)(1 + L) \prod_{k=1}^{s/2-1}
   \bigl(1 - 2\cos\omega_k\, L + L^2\bigr),

and a series rarely needs all of them. HEGY (Hylleberg, Engle, Granger
and Yoo 1990) tests each factor in one regression with the seasonal
random walk as the null: :math:`t` statistics at :math:`\omega = 0` and
:math:`\pi` (lower tail), :math:`F` statistics for each complex pair and
two joint :math:`F` statistics (upper tail). Canova and Hansen (1995)
reverse the null, as KPSS does at frequency zero. Both read p-values
from laws simulated at the sample size, period and deterministic set in
hand, so the seed matters and ``replications``/``n_draws`` set the
p-value resolution.

.. testcode::

   from cultivars.diagnostics.seasonality import canova_hansen, hegy

   rng = np.random.default_rng(5)
   T = 160
   pattern = np.tile([1.0, -0.5, 2.0, -2.5], T // 4)
   y = np.cumsum(rng.standard_normal(T)) + pattern

   root = hegy(y, period=4, replications=500, seed=0)
   for rec in (root, *root.companions):
       print(f"{rec.name:<38} lower_tail={rec.lower_tail!s:<5} "
             f"p={rec.pvalue:.3f}  reject={rec.reject()}")
   print(root.trend, root.lags, root.method)

.. testoutput::

   HEGY t, frequency 0                    lower_tail=True  p=0.396  reject=False
   HEGY t, frequency pi (Nyquist)         lower_tail=True  p=0.000  reject=True
   HEGY F, frequency 2 pi 1 / 4 (1.571)   lower_tail=False p=0.000  reject=True
   HEGY F, all seasonal frequencies       lower_tail=False p=0.000  reject=True
   HEGY F, all frequencies                lower_tail=False p=0.000  reject=True
   c+s 0 aic over 0..13

Keep at frequency zero and reject at every seasonal frequency reads
"difference once and model the season with dummies". Canova-Hansen
agrees from the other side; its headline is the joint statistic and the
companions locate any instability:

.. testcode::

   stable = canova_hansen(y, period=4, n_draws=1000, grid=500, seed=0)
   print(stable.name, round(stable.pvalue, 3), stable.reject())
   print([c.name for c in stable.companions])
   print(stable.lags, stable.method)

.. testoutput::

   Canova-Hansen, joint 0.997 False
   ['Canova-Hansen, frequency pi (Nyquist)', 'Canova-Hansen, frequency 2 pi 1 / 4 (1.571)']
   4 Bartlett kernel, 0 lag(s) of y

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Argument
     - Default
     - Meaning
   * - ``period``
     - required
     - even, at least 2; odd periods raise ``SpecificationError``
   * - ``trend``
     - ``"c"``
     - ``"n"``, ``"c"``, ``"ct"``
   * - ``seasonal``
     - ``True`` (HEGY)
     - seasonal dummies in the regression; the record's ``trend``
       becomes ``"c+s"``
   * - ``lags``, ``max_lags``, ``method``
     - select by ``"aic"`` up to Schwert's ceiling (HEGY)
     - augmentation of the seasonal difference; ``"aic"``, ``"bic"``,
       ``"t-stat"`` (no ``"maic"``)
   * - ``replications``
     - 2000 (HEGY), minimum 500
     - simulated seasonal random walks behind the p-values
   * - ``lags`` (Canova-Hansen)
     - ``0``
     - lags of :math:`y` in the regression, the paper's remedy for
       short-run dynamics
   * - ``bandwidth``, ``n_draws``, ``grid``
     - Newey-West, 5000, 2000 (Canova-Hansen)
     - long-run covariance bandwidth and the von Mises simulation

Long memory
-----------

For :math:`(1 - L)^d y_t = u_t` with short-memory :math:`u_t`, the
spectrum behaves as :math:`f(\lambda) \sim c\,\lambda^{-2d}` near the
origin, and the three semiparametric estimators read :math:`d` off the
:math:`m` lowest Fourier frequencies:
:func:`~cultivars.diagnostics.long_memory.gph` (Geweke and Porter-Hudak
1983; s.e. :math:`\pi/\sqrt{24 m}`, valid for :math:`d < 1/2`),
:func:`~cultivars.diagnostics.long_memory.local_whittle` (Robinson 1995;
s.e. :math:`1/(2\sqrt{m})`, limit theory for :math:`d < 3/4`) and
:func:`~cultivars.diagnostics.long_memory.exact_local_whittle` (Shimotsu
and Phillips 2005; valid for any :math:`d`). The record is an estimate
first: ``d`` and ``se``, with ``statistic`` and ``pvalue`` testing
:math:`d = 0`. ``test(null)`` re-aims the test without re-estimating.

.. testcode::

   from cultivars.diagnostics.long_memory import gph, local_whittle

   rng = np.random.default_rng(9)
   T, d = 800, 0.35
   psi = np.empty(T)
   psi[0] = 1.0
   for k in range(1, T):
       psi[k] = psi[k - 1] * (k - 1 + d) / k   # MA weights of (1 - L)^(-d)
   y = np.convolve(rng.standard_normal(T), psi)[:T]

   for f in (gph, local_whittle, exact_local_whittle):
       est = f(y)
       low, high = est.confidence_interval()
       print(f"{est.method:<20} d={est.d:6.3f} se={est.se:.3f} m={est.bandwidth:<3} "
             f"95% CI=({low:.2f}, {high:.2f})")

.. testoutput::

   gph                  d=-0.144 se=0.146 m=28  95% CI=(-0.43, 0.14)
   local_whittle        d= 0.286 se=0.057 m=77  95% CI=(0.17, 0.40)
   exact_local_whittle  d= 0.296 se=0.057 m=77  95% CI=(0.18, 0.41)

GPH at its default :math:`m = \lfloor T^{1/2} \rfloor` has a standard
error near 0.15 at :math:`T = 800`, wide enough to miss :math:`d = 0.35`
entirely on this draw. The canonical three nulls are one call each:

.. testcode::

   est = exact_local_whittle(y)
   for null in (0.0, 0.5, 1.0):
       rec = est.test(null)
       print(rec.null_d, round(rec.statistic, 2), rec.reject())
   print(est.is_stationary)

.. testoutput::

   0.0 5.19 True
   0.5 -3.58 True
   1.0 -12.36 True
   True

On a random walk, ``local_whittle`` stops at its search bound of one and
the summary says to switch estimators; ``exact_local_whittle`` searches
up to two:

.. testcode::

   walk = np.cumsum(rng.standard_normal(500))
   print(round(local_whittle(walk).d, 4), round(exact_local_whittle(walk).d, 4))

.. testoutput::

   1.0 1.0432

``f_bandwidth`` sets :math:`m` directly; otherwise
:math:`m = \lfloor T^{\text{exponent}} \rfloor` with ``exponent`` 0.5
(GPH) or 0.65 (Whittle). All three need at least 64 observations.

Structural breaks
-----------------

Each break function takes a target, optional regressors ``exog`` and a
``trend`` (``"n"``, ``"c"``, ``"ct"``). With ``exog=None`` the
regression is on the deterministic terms alone, a test for a break in
the mean or trend. Every coefficient is allowed to break, and serial
correlation and heteroskedasticity are not corrected: include lags of
the target in ``exog`` for a dynamic regression.

:func:`~cultivars.diagnostics.breaks.sup_wald` (Andrews 1993) computes
at each date :math:`T_b` in the trimmed window the Wald statistic

.. math::

   W(T_b) = \frac{S_0 - S_1(T_b)}{S_1(T_b) / (T - 2q)},

and reports the sup, with Andrews and Ploberger's (1994) exp-Wald and
ave-Wald as companions. The p-values come from ``n_draws`` simulated
Brownian-bridge paths at the trimming in use.

.. testcode::

   from cultivars.diagnostics.breaks import bai_perron, cusum

   rng = np.random.default_rng(3)
   y = np.concatenate([rng.standard_normal(120), 1.5 + rng.standard_normal(120)])

   sw = sup_wald(y, n_draws=1000, grid=500, seed=0)
   print(repr(sw))
   print([(c.name, round(c.statistic, 2)) for c in sw.companions])
   print(sw.n_restrictions, sw.trimming, sw.path.shape, int(np.isnan(sw.path).sum()))

.. testoutput::

   BreakTest(name='sup-Wald', statistic=143.4745, pvalue=0, break_index=122, nobs=240)
   [('exp-Wald', 67.54), ('ave-Wald', 57.72)]
   1 0.15 (240,) 71

``path`` holds :math:`W(T_b)` at every date (``nan`` outside the
window), so the Chow-type statistic at a date fixed in advance is
``sw.path[tb]``, asymptotically :math:`\chi^2_q` at a known date. There
is no separate Chow function.

:func:`~cultivars.diagnostics.breaks.bai_perron` (Bai and Perron 1998,
2003) finds the global least-squares partition for every break count up
to ``max_breaks`` by dynamic programming and selects the count by BIC
(default), LWZ (Liu, Wu and Zidek 1997) or the sequential
:math:`\sup F(l+1 \mid l)` procedure. All three are computed on every
call:

.. testcode::

   bp = bai_perron(y, max_breaks=3, n_draws=1000, grid=500, seed=0)
   print(bp)
   print(bp.partitions)
   print(bp.bic.round(3), int(np.argmin(bp.lwz)))
   print([(s.name, round(s.pvalue, 3), s.break_index) for s in bp.sequential])

.. testoutput::

   MultipleBreakTest(n_breaks=1, break_indices=(122,), criterion='bic', nobs=240)
   ((), (122,), (122, 203), (122, 163, 203))
   [0.519 0.093 0.11  0.149] 1
   [('sup F(1 | 0)', 0.0, 122), ('sup F(2 | 1)', 0.103, 203)]

:func:`~cultivars.diagnostics.breaks.cusum` cumulates the recursive
residuals (Brown, Durbin and Evans 1975) and compares the path with a
linear boundary. The CUSUM proper carries the tabulated boundary
multipliers and no p-value; ``squares=True`` gives the CUSUM of squares
with a simulated finite-sample boundary and a p-value.

.. testcode::

   cs = cusum(y)
   print(repr(cs))
   print(cs.critical_values, cs.path.shape, cs.bounds.shape)
   sq = cusum(y, squares=True, n_draws=1000, seed=0)
   print(repr(sq))

.. testoutput::

   BreakTest(name='CUSUM', statistic=2.3675, pvalue=None, break_index=142, nobs=240)
   {'1%': 1.143, '5%': 0.948, '10%': 0.85} (239,) (3, 239)
   BreakTest(name='CUSUM of squares', statistic=0.1584, pvalue=0.004, break_index=93, nobs=240)

``break_index`` on a CUSUM record is the first boundary crossing, not a
break-date estimate: the CUSUM crossing (142) lags the break at 120 and
the CUSUM-of-squares crossing (93) precedes it. The CUSUM of squares
also rejects here although the variance never changed: a mean shift
inflates the squared recursive residuals. Date breaks with ``sup_wald``
or ``bai_perron``.

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Argument
     - Default
     - Meaning
   * - ``trimming``
     - ``0.15``
     - fraction excluded at each end (``sup_wald``), minimum segment
       length (``bai_perron``); never fewer than ``q + 1``
       observations either way
   * - ``n_draws``, ``grid``
     - 5000, 2000 (minimum 500, 100)
     - simulated limit paths and their discretization
   * - ``max_breaks``
     - 5
     - ``bai_perron``; refused if ``max_breaks + 1`` minimum-length
       segments do not fit
   * - ``criterion``, ``alpha``
     - ``"bic"``, 0.05
     - ``bai_perron`` selection and the level of the sequential tests
   * - ``squares``
     - ``False``
     - ``cusum``: CUSUM of squares instead of CUSUM

Cointegration rank
------------------

Johansen's (1991) reduced-rank regression estimates the squared
canonical correlations :math:`\hat\lambda_1 \ge \cdots \ge \hat\lambda_k`
between differences and lagged levels and tests the rank with

.. math::

   \lambda_{\text{trace}}(r) = -T \sum_{i=r+1}^{k} \ln(1 - \hat\lambda_i),
   \qquad
   \lambda_{\max}(r) = -T \ln(1 - \hat\lambda_{r+1}).

The test is a method of the error-correction model,
``VECM(...).rank_test()``, and returns the
:class:`~cultivars.diagnostics.cointegration.JohansenRankTest` record.
It does not depend on the ``rank`` the model was built with, so build
with ``rank=0``. P-values are empirical over ``simulations`` draws of
the Johansen limit for the deterministic case
(``cointegration_trend``: ``"none"``, ``"restricted_constant"``,
``"constant"``, ``"restricted_trend"``, ``"trend"``; Johansen 1995) and,
for a :class:`~cultivars.multivariate.reduced_form.error_correction.VECMX`,
for the number of weakly exogenous I(1) regressors (Pesaran, Shin and
Smith 2000).

.. testcode::

   from cultivars.multivariate.reduced_form.error_correction import VECM

   rng = np.random.default_rng(7)
   T = 300
   common = np.cumsum(rng.standard_normal(T))
   levels = np.column_stack([
       common + rng.standard_normal(T),
       0.5 * common + rng.standard_normal(T),
       np.cumsum(rng.standard_normal(T)),
   ])
   rank_test = VECM(levels, order=2, rank=0).rank_test(simulations=1000, steps=200)
   print(rank_test)
   print(rank_test.trace_statistic.round(2), rank_test.trace_pvalue.round(3))
   print(rank_test.max_eigenvalue_pvalue.round(3))
   print(rank_test.selected_rank(), rank_test.selected_rank(statistic="max_eigenvalue"))

.. testoutput::

   JohansenRankTest(k=3, nobs=298, deterministic='constant', trace_rank=1, max_eigenvalue_rank=1)
   [142.03  16.23   1.65] [0.    0.074 0.629]
   [0.    0.056 0.629]
   1 1

``selected_rank`` stops at the first rank not rejected, returning ``k``
if all are rejected (stationary in levels). ``to_table()`` renders both
sequences with notes on the sequential size and the p-value resolution
(:math:`1/\text{simulations}`). ``small_sample=True`` applies the
Reinsel and Ahn (1992) scaling :math:`(T - kp)/T`, which pulls the
asymptotic test's over-rejection in short samples (about 7% at nominal
5% for three variables and 400 observations, per the method docstring)
back toward its level.

The selected rank feeds the model; see :doc:`var_extensions` for the
VECM itself.

.. testcode::

   res = VECM(levels, order=2, rank=rank_test.selected_rank(),
              names=["a", "b", "c"]).fit()
   print(res.normalized_beta().round(2).ravel())
   print(repr(res.weak_exogeneity("c")))

.. testoutput::

   [ 1.   -2.03  0.05]
   WaldTest(statistic=0.0148, df=1, pvalue=0.9033, keep at 5%: 'c is weakly exogenous for the cointegrating space')

Nonlinearity
------------

The four model-aimed tests fit the linear AR(``order``) internally, so
pass the series, not residuals. Each returns a
:class:`~cultivars.diagnostics.nonlinearity.LinearityTest` naming its
``alternative``.

- :func:`~cultivars.diagnostics.nonlinearity.terasvirta` (Teräsvirta
  1994): third-order Taylor expansion of a smooth transition in
  :math:`y_{t-d}`, an :math:`F(3p, n - 4p - 1)` test; its companions
  ``H04``, ``H03``, ``H02`` give ``suggestion`` (``"LSTAR"`` or
  ``"ESTAR"``).
- :func:`~cultivars.diagnostics.nonlinearity.tsay` (Tsay 1989):
  arranged autoregression sorted by :math:`y_{t-d}`; threshold
  alternative, no threshold estimate.
- :func:`~cultivars.diagnostics.nonlinearity.ramsey_reset` (Ramsey
  1969): powers :math:`2..k` of the fitted value; names no regime.
- :func:`~cultivars.diagnostics.nonlinearity.hansen_threshold` (Hansen
  1996): sup-:math:`F` over a threshold grid with a fixed-regressor
  bootstrap p-value; reports ``threshold``, the selected ``delay`` and
  simulated ``critical_values``, with ``df=None``.
- :func:`~cultivars.diagnostics.nonlinearity.bds` (Brock et al. 1996):
  independence against any dependence, run on residuals; record at
  ``dimension`` with lower dimensions as companions.

.. testcode::

   from cultivars.diagnostics.nonlinearity import bds, hansen_threshold, ramsey_reset, terasvirta

   rng = np.random.default_rng(21)
   T = 300
   y = np.zeros(T)
   for t in range(1, T):
       y[t] = (0.7 if y[t - 1] <= 0.0 else -0.4) * y[t - 1] + rng.standard_normal()

   for f in (terasvirta, tsay, ramsey_reset):
       rec = f(y, order=1)
       print(f"{rec.name:<11} {rec.alternative:<24} df={rec.df}  reject={rec.reject()}")
   tv = terasvirta(y, order=1)
   print(tv.suggestion, [(c.name, round(c.pvalue, 3)) for c in tv.companions])

.. testoutput::

   Teräsvirta  smooth transition        df=(3, 294)  reject=True
   Tsay        threshold                df=(2, 244)  reject=True
   RESET       smooth misspecification  df=(2, 295)  reject=True
   LSTAR [('H04', 0.797), ('H03', 0.005), ('H02', 0.0)]

``hansen_threshold`` defaults to ``replications=1000`` and
``n_grid=300``; a smaller run is enough to read a strong rejection:

.. testcode::

   h = hansen_threshold(y, order=1, replications=200, n_grid=100, seed=0)
   print(repr(h))
   print(round(h.threshold, 3), h.delay, h.df, round(h.critical_values["5%"], 2))

.. testoutput::

   LinearityTest(name='Hansen sup-F', statistic=40.4489, pvalue=0, order=1, nobs=299)
   -0.212 1 None 10.7

BDS rejects for any dependence, so read it after an ARCH test, not
instead of one. The model-aimed tests are not immune either: on this
ARCH(1) series, which has no regime in its mean, Tsay rejects linearity
as well as BDS, because neglected conditional heteroskedasticity
inflates it. Run the ARCH test on the linear residuals first:

.. testcode::

   e = np.zeros(600)
   for t in range(1, 600):
       e[t] = np.sqrt(0.2 + 0.7 * e[t - 1] ** 2) * rng.standard_normal()
   b = bds(e)
   print(tsay(e, order=1).reject(), b.reject(), b.dimension, [c.dimension for c in b.companions])
   print(round(b.epsilon / e.std(), 2))

.. testoutput::

   True True 3 [2]
   1.5

The record's summary warns when ``nobs < 500``, where the normal
reference over-rejects (about 8% at nominal 5% on 250 Gaussian
observations per the class docstring).

Stability and lag order
-----------------------

:mod:`cultivars.diagnostics.stability` documents two records that
fitted results produce. :class:`~cultivars.diagnostics.stability.StabilityAssessment`
holds the companion eigenvalues, ``max_modulus``, ``is_stable``,
``n_unit_roots`` and ``n_explosive`` at a tolerance ``tol`` (default
``1e-8``). It is deterministic: no null, no p-value. VAR-family results
return it from ``stability_check()``; ARMA-family results expose
``stability`` and ``invertibility`` properties. The class methods
``assess_stability``, ``assess_stability_from_companion``,
``is_stationary`` and ``is_invertible`` work on raw coefficients.

.. testcode::

   from cultivars.diagnostics.stability import StabilityAssessment

   A = np.array([[[0.5, 0.2], [0.0, 0.4]]])
   check = StabilityAssessment.assess_stability(A)
   print(check.is_stable, round(check.max_modulus, 2), check.n_unit_roots)
   walk_check = StabilityAssessment.assess_stability([1.0])
   print(walk_check.is_stable, walk_check.n_unit_roots, walk_check.n_explosive)
   print(StabilityAssessment.assess_stability([1.0], allow_unit_roots=True).is_stable)

.. testoutput::

   True 0.5 0
   False 1 0
   True

A VECM carries :math:`k - r` unit roots by construction and its
``stability_check()`` passes ``allow_unit_roots=True``; with the rank-one
model above:

.. testcode::

   vecm_check = res.stability_check()
   print(vecm_check.is_stable, vecm_check.n_unit_roots, vecm_check.n_explosive)

.. testoutput::

   True 2 0

:class:`~cultivars.diagnostics.stability.LagOrderSelection` comes from
``model.lag_order_selection(max_lags=...)`` on any VAR-family model. It
scores every order on one common sample and keeps the whole AIC, BIC,
HQIC and FPE curves; ``best(criterion)``, ``selected``, ``consensus``
(``None`` when the criteria disagree) and ``to_table()`` read them.

.. testcode::

   from cultivars.multivariate.reduced_form.vector_autoregression import VAR

   rng = np.random.default_rng(1)
   T = 250
   x = np.zeros((T, 2))
   for t in range(1, T):
       x[t] = A[0] @ x[t - 1] + rng.standard_normal(2)
   selection = VAR(x, order=1, names=["x", "z"]).lag_order_selection(max_lags=4)
   print(selection)
   print(selection.consensus, selection.nobs, selection.curve("bic").round(3))

.. testoutput::

   LagOrderSelection(max_lags=4, nobs=246, AIC=1, BIC=1, HQIC=1, FPE=1)
   1 246 [ 0.023 -0.261 -0.179 -0.11  -0.032]

Residual diagnostics on fitted results
--------------------------------------

Fitted VAR-family results (VAR, VARX, VECM, VECMX, VARMA, PanelVAR,
MIDASVAR; see :doc:`var`) run their own residual tests and return
:class:`~cultivars.diagnostics.hypothesis.WaldTest` records whose
``null`` is a sentence:

- ``portmanteau_test(lags=10)``: multivariate Ljung-Box statistic
  (Hosking 1980 adjustment) on the residual autocorrelations.
- ``normality_test(variable)``: Jarque-Bera on one equation's residuals,
  2 degrees of freedom. There is no multivariate (Lütkepohl) normality
  test.
- ``arch_test(variable, lags=5)``: Engle's (1982) LM test,
  :math:`(n - \text{lags}) R^2` from squared residuals on their lags.
- ``granger_causality(cause, effect)``: Wald test that ``cause``'s lags
  are jointly zero in ``effect``'s equation.
- ``likelihood_ratio_test(unrestricted)``: a
  :class:`~cultivars.diagnostics.hypothesis.LikelihoodRatioTest`
  against a larger model on the same sample.
- ``residual_diagnostics(portmanteau_lags=10, arch_lags=5)``: all of
  the first three in one ``SummaryTable`` (not a record).

.. testcode::

   fit = VAR(x, order=1, names=["x", "z"]).fit()
   print(fit.residual_diagnostics())
   print(repr(fit.granger_causality("z", "x")))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

                                 Residual diagnostics
   ================================================================================
   Observations:                      249  Equations:                           2
   --------------------------------------------------------------------------------
   test                statistic   df   p-value
   portmanteau(10)        28.390   34    0.7388
   jarque-bera[x]          0.321    2    0.8518
   jarque-bera[z]          4.272    2    0.1182
   arch-lm(5)[x]           6.471    5    0.2631
   arch-lm(5)[z]           6.812    5    0.2350
   ================================================================================
   Small p-values reject the null named by each test. A rejected portmanteau means
   the lag order is too short, which invalidates every other quantity the result
   reports.
   WaldTest(statistic=5.4301, df=1, pvalue=0.01979, reject at 5%: 'z does not Granger-cause x')

A likelihood-ratio test across lag orders needs the shorter model
estimated on the longer model's sample; drop the leading observations
yourself:

.. testcode::

   from cultivars.exceptions import DimensionError

   long_fit = VAR(x, order=2, names=["x", "z"]).fit()
   try:
       fit.likelihood_ratio_test(long_fit)
   except DimensionError as err:
       print(err)
   short_fit = VAR(x[1:], order=1, names=["x", "z"]).fit()
   print(repr(short_fit.likelihood_ratio_test(long_fit)))

.. testoutput::

   a likelihood-ratio test requires a common sample; got 249 and 248 observations.
   LikelihoodRatioTest(statistic=2.0384, df=4, pvalue=0.7287)

Univariate results (``ARResult``, ``ARMAResult``, ``GARCHResult``) carry
no residual tests of their own: no Ljung-Box, no Jarque-Bera, no ARCH
LM. The tools at hand are the ``stability``/``invertibility``
properties and :func:`~cultivars.diagnostics.nonlinearity.bds` on
``result.resid``:

.. testcode::

   from cultivars.univariate.box_jenkins import ARIMA

   rng = np.random.default_rng(0)
   z = np.zeros(600)
   for t in range(1, 600):
       z[t] = 0.7 * z[t - 1] + rng.standard_normal()
   arma = ARIMA(z, order=(1, 0, 0)).fit()
   print(arma.is_stationary, repr(bds(arma.resid)))

.. testoutput::

   True BDSTest(statistic=-1.4400, pvalue=0.1499, dimension=3, nobs=600)

A recommended pre-estimation workflow
-------------------------------------

The sequence below takes three level series to a validated VECM:
classify each series, check the differences for breaks, choose the lag
order, test the cointegrating rank, fit, and check the residuals and
the roots. It reuses ``classify`` from the confirmatory-analysis
section.

.. testcode::

   rng = np.random.default_rng(1)
   T = 400
   tau = np.cumsum(rng.standard_normal(T))
   u = np.zeros((T, 3))
   for t in range(1, T):
       u[t] = 0.5 * u[t - 1] + rng.standard_normal(3)
   data = np.column_stack([tau + u[:, 0], 0.5 * tau + u[:, 1], np.cumsum(u[:, 2])])
   names = ["y1", "y2", "y3"]

   # 1. Order of integration: levels, then first differences.
   print([classify(data[:, j]) for j in range(3)])
   print([classify(np.diff(data[:, j])) for j in range(3)])

   # 2. No break in the mean of any difference.
   print([sup_wald(np.diff(data[:, j]), n_draws=1000, grid=500, seed=0).reject()
          for j in range(3)])

   # 3. Lag order of the levels VAR, on a common sample.
   order = VAR(data, order=1, names=names).lag_order_selection(max_lags=6).best("bic")
   print(order)

   # 4. Cointegrating rank.
   ranks = VECM(data, order=order, rank=0, names=names).rank_test(
       simulations=2000, steps=200)
   rank = ranks.selected_rank()
   print(rank, ranks.trace_pvalue.round(3))

.. testoutput::

   ['I(1)', 'I(1)', 'I(1)']
   ['I(0)', 'I(0)', 'I(0)']
   [False, False, False]
   2
   1 [0.    0.084 0.691]

.. testcode::

   # 5. Fit at the selected rank and check what the fit left behind.
   vecm = VECM(data, order=order, rank=rank, names=names).fit()
   print(vecm.normalized_beta().round(2).ravel())
   print(round(vecm.portmanteau_test(10).pvalue, 3),
         [round(vecm.arch_test(n).pvalue, 3) for n in names])
   roots = vecm.stability_check()
   print(roots.n_unit_roots == 3 - rank, roots.n_explosive)

.. testoutput::

   [ 1.   -2.07  0.01]
   0.138 [0.818, 0.763, 0.382]
   True 0

The cointegrating vector recovers :math:`y_1 - 2 y_2` (the true
relation), the third series is excluded, the residuals show no
autocorrelation or ARCH at 5%, and the companion has exactly
:math:`k - r = 2` unit roots and none outside the circle. Add the
seasonal step (``hegy`` with ``canova_hansen``) before step 1 for
sub-annual data, and the nonlinearity step (``arch_test`` on the
residuals, then ``tsay``/``terasvirta``) after step 5 if a regime model
is a candidate (see :doc:`univariate_nonlinear` and
:doc:`nonlinear_multivariate`).

How the pieces connect
----------------------

- Unit-root and seasonal tests decide between a levels model, a
  differenced ARIMA/SARIMA (:doc:`univariate_linear`) and a cointegrated
  system (:doc:`var_extensions`).
- ``rank_test`` lives on
  :class:`~cultivars.multivariate.reduced_form.error_correction.VECM`
  and :class:`~cultivars.multivariate.reduced_form.error_correction.VECMX`;
  its record decides the ``rank`` argument of the model you then fit.
- Long-memory estimates point to
  :class:`~cultivars.univariate.fractional_integration.ARFIMA` or FIGARCH
  (:doc:`univariate_linear`, :doc:`volatility`).
- Linearity rejections license SETAR, LSTAR and ESTAR
  (:doc:`univariate_nonlinear`).
- Residual tests, Granger causality and lag selection sit on the fitted
  VAR-family results (:doc:`var`), and stability is presumed by every
  impulse response and forecast (:doc:`structural_identification`,
  :doc:`forecasting`).
- Forecast-comparison tests return records from the same family but
  live in :mod:`cultivars.forecast.comparison`; MCMC convergence reports
  live in :mod:`cultivars.bayes.chains` (:doc:`bayesian_workflow`).

Nuances and gotchas
-------------------

**Which tail rejects.** Dickey-Fuller-family records, Zivot-Andrews and
HEGY ``t`` rows have ``lower_tail=True``; KPSS, Canova-Hansen, HEGY
``F`` rows, every break test and every linearity test reject in the
upper tail. Ng-Perron's :math:`MSB` and :math:`MP_T` are positive yet
reject when *small*. ``reject()`` handles all of this; read
``lower_tail`` before comparing a statistic with a critical value by
hand.

**KPSS p-values are clipped, and reject() takes them literally.** The
p-value is interpolated in a four-point table and clipped to
``[0.01, 0.10]``. Two consequences follow, both verified:

.. testcode::

   rng = np.random.default_rng(42)
   strong = kpss(np.cumsum(rng.standard_normal(250)))
   print(round(strong.statistic, 2), strong.critical_values["1%"], strong.pvalue,
         strong.reject(alpha=0.01))
   weak = kpss(rng.standard_normal(250))
   print(round(weak.statistic, 3), weak.pvalue, weak.reject(alpha=0.20))

.. testoutput::

   3.69 0.739 0.01 False
   0.087 0.1 True

A statistic five times the 1% critical value does not "reject at 1%"
(``0.01 < 0.01`` is false), and any KPSS record rejects at a level above
10%, however small its statistic. Use ``alpha`` in ``(0.01, 0.10]`` with
KPSS, or compare ``statistic`` with ``critical_values`` directly.

**Simulated p-values have a resolution.** ``sup_wald``, ``bai_perron``,
``cusum(squares=True)``, ``hegy``, ``canova_hansen``,
``hansen_threshold`` and ``rank_test`` report the share of simulated
draws at or beyond the statistic. A p-value of ``0`` means below
``1 / n_draws`` (or ``replications``, ``simulations``). Pass ``seed`` for
reproducibility; the default ``seed=None`` gives a different p-value on
each call. ``rank_test`` alone fixes its seed (``20260819``) by default.

**Tables without p-values.** DF-GLS, Ng-Perron, Zivot-Andrews and the
CUSUM carry ``pvalue=None`` and accept only ``alpha`` in
``{0.01, 0.05, 0.10}``; any other level raises ``SpecificationError``.
DF-GLS's asymptotic critical values are mildly liberal (about 6% size at
nominal 5% with 100 observations, per its docstring).

**Small-sample size notes carried by the records.** Johansen's trace
test over-rejects in short samples (use ``small_sample=True``); BDS
over-rejects below 500 observations and its summary says so; the
long-memory standard errors are asymptotic in the bandwidth and the
95% interval covers 90 to 94% at 500 observations (low 80s for exact
local Whittle near :math:`d = 1/2`); KPSS over-rejects on persistent
stationary series, Phillips-Perron under a large negative MA root.

**Lag selection defaults differ across tests.** ADF and HEGY select by
AIC, DF-GLS and Ng-Perron by the modified AIC (``"maic"``), and
Zivot-Andrews by the t-statistic rule, all up to Schwert's ceiling.
The modified AIC selects long augmentations on series far from the
null (six lags on the AR(0.6) above, against none for AIC); pass
``method="aic"`` or a fixed ``lags`` there. KPSS and Phillips-Perron use
a Newey-West bandwidth; Canova-Hansen adds ``lags=0`` lags of the
series by default.

**Reported bandwidth under the quadratic-spectral kernel.** With
``kernel="quadratic-spectral"`` and ``bandwidth=None``, KPSS and
Phillips-Perron compute the long-run variance with Andrews' plug-in
bandwidth, but the record's ``lags`` field still reports the Newey-West
Bartlett rule. Pass ``bandwidth`` explicitly if the reported value
matters.

**The VAR portmanteau degrees of freedom.** ``portmanteau_test``
uses :math:`k^2 h - k \cdot m` degrees of freedom, where :math:`m`
counts *every* regressor per equation, deterministic terms included.
The textbook figure is :math:`k^2 (h - p)` (Lütkepohl 2005), so with a
constant the reported ``df`` is ``k`` smaller (34 instead of 36 for the
bivariate VAR(1) above) and the p-value slightly smaller. It also
requires ``k * k * lags`` to exceed that count; ``lags=1`` raises.

**No Engle-Granger test.** There is no residual-based cointegration
test. Running ``adf`` on OLS residuals from a cointegrating regression
does *not* give one: the residuals are fitted to look stationary, and
the Dickey-Fuller p-values are far too small for them (the correct
MacKinnon cointegration surfaces are not in the package). Use
``rank_test``.

**Breaks: what the records do not do.** No partial breaks (every
coefficient breaks), no HAC correction, no Chow function (use
``sup_wald(...).path[tb]`` against :math:`\chi^2_q`). ``bai_perron``'s
partitions are not nested across counts, and its sequential tests run
under every ``criterion``, so ``n_draws`` governs the cost even with
``criterion="bic"``. ``break_index`` is the first observation of the
new regime; for Zivot-Andrews and sup-Wald it is reported whether or
not the null is rejected.

**HEGY's simulated null uses no augmentation.** The HEGY draws run the
regression with zero lags on the full sample length, whatever ``lags``
the test selected. A brief check (80 seasonal random walks of 160
observations) gave rejection rates of 4% to 8% at nominal 5%.

**Long memory.** ``local_whittle`` cannot report :math:`d > 1`; an
estimate of exactly 1.0 is the search bound, and ``test(1.0)`` then
"keeps" a unit root by construction. ``reject()`` on a fresh
``LongMemoryEstimate`` tests :math:`d = 0`, not a unit root. GPH is
valid only for :math:`d < 1/2`; difference first or use
``exact_local_whittle``.

**Linearity tests.** Teräsvirta's ``suggestion`` is set even when
linearity is kept; read it only after ``reject()``. Neglected ARCH
inflates all four model-aimed tests, and ``hansen_threshold`` assumes
homoskedastic errors. The transition variable is always a lag of the
series; exogenous transition variables are not covered.
``hansen_threshold`` requires ``replications >= 200``.

**StabilityAssessment.is_invertible takes negated MA coefficients.**
The class methods test :math:`1 - c_1 z - \cdots - c_q z^q`, the AR
convention. For :math:`\theta(L) = 1 + \theta_1 L + \theta_2 L^2` pass
``-theta``:

.. testcode::

   theta = np.array([1.5, 0.6])   # 1 + 1.5 L + 0.6 L^2, invertible
   print(StabilityAssessment.is_invertible(theta), StabilityAssessment.is_invertible(-theta))

.. testoutput::

   False True

Fitted ARMA results negate internally; the trap is only in calling the
class method yourself.

**Records that are not tests.** ``JohansenRankTest`` and
``LagOrderSelection`` render with ``to_table()``, not ``summary()``.
``MultipleBreakTest`` has ``summary()`` but no ``reject``.
``residual_diagnostics()`` returns a table, not records; call the
individual methods to get objects you can ``reject``.

See also
--------

- :doc:`var` for the VAR results that run residual tests and lag
  selection.
- :doc:`var_extensions` for VECM and VECMX, where the rank test lives.
- :doc:`univariate_linear` and :doc:`univariate_nonlinear` for the models
  these tests choose between.
- :doc:`nuances` for package-wide conventions on samples and indexing.
- :mod:`cultivars.diagnostics.hypothesis`,
  :mod:`cultivars.diagnostics.unit_roots`,
  :mod:`cultivars.diagnostics.seasonality`,
  :mod:`cultivars.diagnostics.long_memory`,
  :mod:`cultivars.diagnostics.breaks`,
  :mod:`cultivars.diagnostics.cointegration`,
  :mod:`cultivars.diagnostics.nonlinearity`,
  :mod:`cultivars.diagnostics.stability`.
