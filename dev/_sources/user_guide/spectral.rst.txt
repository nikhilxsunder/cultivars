.. meta::
   :description: Spectra from data and from fitted VARs, Geweke causality, HP, Hamilton, Baxter-King and CF filters, turning points and wavelets in cultivars.

Spectral analysis and filtering
===============================

The :mod:`cultivars.spectral` package reads time series and fitted
systems by frequency. It answers four questions: where in the cycle a
series' variance sits (nonparametric spectra), what a fitted VAR claims
at each frequency (model-implied density, coherence, phase, Geweke
causality), how to separate trend from cycle (HP, Butterworth, Hamilton,
Beveridge-Nelson, Baxter-King, Christiano-Fitzgerald) and how to date
and compare cycles (turning points, concordance), with wavelets adding a
date axis. It does not estimate models: the model-implied views take a
fitted result from :doc:`var` or :doc:`var_extensions`, and long-memory
estimators that read the spectrum near zero are on :doc:`diagnostics`.

.. contents:: On this page
   :local:
   :depth: 1

The family at a glance
----------------------

.. list-table::
   :header-rows: 1
   :widths: 26 20 30 24

   * - Class
     - Module
     - What it computes
     - Use when
   * - :class:`~cultivars.spectral.periodogram.MultitaperSpectrum`,
       :class:`~cultivars.spectral.periodogram.DaniellSpectrum`,
       :class:`~cultivars.spectral.periodogram.WelchSpectrum`
     - :mod:`~cultivars.spectral.periodogram`
     - Smoothed periodogram matrix of a panel, with degrees of freedom
       and resolution
     - You want the data's spectrum, before or instead of a model.
   * - :class:`~cultivars.spectral.density.SpectralDensity`
     - :mod:`~cultivars.spectral.density`
     - :math:`\Psi(\omega)\Sigma_u\Psi(\omega)^*/2\pi` of a fitted VAR;
       coherence, partial coherence, gain, phase
     - You want what a fitted system implies by frequency.
   * - :class:`~cultivars.spectral.causality.SpectralCausality`,
       :class:`~cultivars.spectral.causality.ConditionalSpectralCausality`
     - :mod:`~cultivars.spectral.causality`
     - Geweke (1982, 1984) causality measures by frequency
     - You want Granger causality resolved by cycle length.
   * - :class:`~cultivars.spectral.filters.HodrickPrescottFilter`,
       :class:`~cultivars.spectral.filters.ButterworthFilter`,
       :class:`~cultivars.spectral.filters.HamiltonFilter`,
       :class:`~cultivars.spectral.filters.BeveridgeNelsonDecomposition`
     - :mod:`~cultivars.spectral.filters`
     - Trend-cycle decompositions
     - You need a trend removed, under a stated definition of trend.
   * - :class:`~cultivars.spectral.band_pass.BaxterKingFilter`,
       :class:`~cultivars.spectral.band_pass.ChristianoFitzgeraldFilter`
     - :mod:`~cultivars.spectral.band_pass`
     - Approximate ideal band-pass between two periods
     - You need the business-cycle band isolated.
   * - :class:`~cultivars.spectral.cycles.TurningPoints`,
       :func:`~cultivars.spectral.cycles.concordance`
     - :mod:`~cultivars.spectral.cycles`
     - Bry-Boschan / Harding-Pagan peaks and troughs; concordance index
     - You need a chronology, or agreement between two.
   * - :class:`~cultivars.spectral.wavelets.MODWT`,
       :class:`~cultivars.spectral.wavelets.WaveletCoherence`
     - :mod:`~cultivars.spectral.wavelets`
     - Variance by octave and date; Morlet coherence over scale and time
     - You suspect the frequency content changed during the sample.

Units and conventions
---------------------

Every object on this page uses the same units, so a data estimate and a
model density overlay without conversion.

Frequency
   :math:`\omega` in **radians per observation** on :math:`[0, \pi]`.
   :math:`\pi` is the Nyquist frequency, a cycle of two observations.
   Results store the grid as ``frequencies``.
Period
   :math:`2\pi / \omega` in **observations per cycle**. ``periods()`` on
   :class:`~cultivars.spectral.periodogram.SpectrumEstimate` and
   :class:`~cultivars.spectral.density.SpectralDensityResult` returns it,
   with ``inf`` at :math:`\omega = 0`. The package never converts to
   years: for quarterly data a period of 8 is two years, for monthly
   data it is eight months.
Bands
   Stated in **periods**, never in frequencies: ``low`` is the shortest
   period passed and ``high`` the longest. The Burns-Mitchell business
   cycle is 6 to 32 at quarterly frequency (:math:`\omega` from
   :math:`2\pi/32 = 0.196` to :math:`2\pi/6 = 1.047`) and 18 to 96 at
   monthly frequency. Periods below 2 do not exist at the sampling
   frequency.
Normalization
   Densities carry the :math:`1/2\pi` factor:
   :math:`\int_{-\pi}^{\pi} f(\omega)\,d\omega = \gamma(0)`, so twice the
   integral of a spectrum over :math:`[0, \pi]` is the variance, and
   white noise of variance :math:`\sigma^2` has the flat spectrum
   :math:`\sigma^2 / 2\pi`.
Matrix axes
   Spectral matrices are ``(n_frequencies, k, k)``; series are addressed
   by name (``"y1"``, ``"y2"``, ... unless ``names`` is given), not by
   index.

The argument names for bands differ by class:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Call
     - Band arguments
   * - ``BaxterKingFilter``, ``ChristianoFitzgeraldFilter``
     - ``low=6.0, high=32.0`` (periods)
   * - ``ButterworthFilter``
     - ``period=32.0`` (cutoff period), ``order=2``
   * - ``SpectralCausalityResult.band``
     - ``low_period=..., high_period=...`` (keyword-only)
   * - ``WaveletCoherenceResult.lead``
     - ``scale_band=(low, high)`` (periods, inclusive)

Nonparametric spectra from data
-------------------------------

The three estimators smooth the periodogram
:math:`I(\omega_j) = (2\pi N)^{-1} |\sum_t y_t e^{-i\omega_j t}|^2`,
which is inconsistent on its own, and each ordinate of the result is
approximately :math:`f(\omega)\chi^2_\nu/\nu`. They take the panel at
construction and estimate through ``compute()``.

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Estimator
     - Smoothing control (default)
     - Degrees of freedom :math:`\nu`
     - Resolution ``bandwidth`` (radians)
   * - ``MultitaperSpectrum``
       (Thomson 1982)
     - ``bandwidth`` :math:`NW` (4.0), ``n_tapers`` :math:`K`
       (:math:`2NW - 1`)
     - :math:`2K`
     - :math:`2\pi NW / N`
   * - ``DaniellSpectrum``
     - ``span`` :math:`m` (:math:`\operatorname{round}(\sqrt{N}/2)`)
     - :math:`16m^2/(4m-1)`
     - :math:`2\pi m / N`
   * - ``WelchSpectrum``
       (Welch 1967)
     - ``segment`` :math:`L` (:math:`\max(32, N/8)`, even)
     - :math:`36B/19` for :math:`B` half-overlapped segments
     - :math:`2\pi / L`, on the segment's coarser grid

All three take ``detrend`` (``"n"``, ``"c"`` the default, or ``"ct"``)
and ``names``, need at least 32 observations, and return a
:class:`~cultivars.spectral.periodogram.SpectrumEstimate`. Multitaper is
the default choice: least leakage at a given variance, full Fourier
grid. A three-variable system whose first series has an AR(2) cycle of
period 12:

.. testcode::

   import numpy as np
   from cultivars.spectral.periodogram import (
       DaniellSpectrum, MultitaperSpectrum, WelchSpectrum,
   )

   rng = np.random.default_rng(1)
   n = 400
   y = np.zeros((n, 3))
   for t in range(2, n):
       y[t, 0] = 1.3 * y[t - 1, 0] - 0.6 * y[t - 2, 0] + rng.standard_normal()
       y[t, 1] = 0.6 * y[t - 1, 0] + 0.3 * y[t - 1, 1] + rng.standard_normal()
       y[t, 2] = 0.5 * y[t - 1, 1] + 0.3 * y[t - 1, 2] + rng.standard_normal()
   names = ("cycle", "output", "prices")

   est = MultitaperSpectrum(y, names=names).compute()
   print(est.method, est.frequencies.shape, est.density.shape)
   print(est.degrees_of_freedom, round(est.bandwidth, 4))
   power = est.spectrum("cycle")
   peak = 1 + int(np.argmax(power[1:]))
   print(round(float(est.frequencies[peak]), 3), round(float(est.periods()[peak]), 1))
   print(round(float(2 * np.trapezoid(power, est.frequencies) / y[:, 0].var()), 2))

.. testoutput::

   multitaper (201,) (201, 3, 3)
   14.0 0.0628
   0.534 11.8
   1.01

The peak sits at period 11.8 against the AR(2)'s theoretical 12.0, and
twice the integral recovers the sample variance. The three estimators
trade resolution for stability differently on the same data:

.. testcode::

   for e in (est, DaniellSpectrum(y, names=names).compute(),
             WelchSpectrum(y, names=names).compute()):
       print(e.method, len(e.frequencies), round(e.degrees_of_freedom, 1),
             round(e.bandwidth, 4))

.. testoutput::

   multitaper 201 14.0 0.0628
   daniell 201 41.0 0.1571
   welch 26 28.4 0.1257

``SpectrumEstimate`` provides ``spectrum(name)``,
``coherence(first, second)`` (squared, biased upward by about
:math:`2/\nu` under zero coherence), ``confidence_interval(name,
alpha=0.05)`` (pointwise, constant ratio, so plot spectra on a log
scale), ``periods()`` and ``summary()``.

Model-implied spectra from fitted systems
-----------------------------------------

:class:`~cultivars.spectral.density.SpectralDensity` evaluates the
closed form

.. math::

   f(\omega) = \frac{1}{2\pi}\, \Psi(\omega)\, \Sigma_u\, \Psi(\omega)^\ast,
   \qquad
   \Psi(\omega) = \Bigl[I - \sum_{l=1}^{p} A_l e^{-i\omega l}\Bigr]^{-1},

of a fitted result on an inclusive grid of ``n_frequencies`` points
(default 256) on :math:`[0, \pi]` (Hamilton 1994, ch. 6). It
re-estimates nothing and reads no data: the input is the result's
``coefficients`` and ``sigma_u``. The result must satisfy the
closed-system protocol (``names``, ``nobs``, ``sigma_u``, ``resid``,
``coefficients``, ``k_endog``, ``ma_representation``), which
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
and :class:`~cultivars.multivariate.reduced_form.error_correction.VECM`
results do; anything else raises
:class:`~cultivars.exceptions.SpecificationError` at construction.

The model's grid matches a periodogram's Fourier grid when the sample
length :math:`N` is even and ``n_frequencies = N // 2 + 1``, so the two
overlay directly:

.. testcode::

   from cultivars.multivariate.reduced_form.vector_autoregression import VAR
   from cultivars.spectral.density import SpectralDensity

   fit = VAR(y, order=2, names=names).fit()
   model = SpectralDensity(fit, n_frequencies=201).compute()
   print(bool(np.allclose(model.frequencies, est.frequencies)))
   lower, upper = est.confidence_interval("cycle")
   implied = model.spectrum("cycle")
   print(round(float(np.mean((lower <= implied) & (implied <= upper))), 2))

.. testoutput::

   True
   0.97

The VAR(2) density lies inside the data's 95% band at 97% of the
ordinates. Where the two disagree is where the model is misspecified,
frequency by frequency.

Readings off the density
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Method
     - Returns
   * - ``spectrum(name)``
     - :math:`f_{ii}(\omega)`, real ``(n,)``
   * - ``coherence(first, second)``
     - Squared coherence :math:`|f_{ij}|^2 / f_{ii} f_{jj}`, symmetric
   * - ``partial_coherence(first, second)``
     - The same from :math:`f^{-1}`, conditioning on all other variables
   * - ``gain(effect, cause)``
     - :math:`|f_{ij}| / f_{jj}`
   * - ``phase(effect, cause)``
     - :math:`\arg f_{ij}` in :math:`(-\pi, \pi]`; negative means
       ``cause`` leads by :math:`-\phi/\omega` observations
   * - ``periods()``
     - :math:`2\pi/\omega`

In the simulated system ``cycle`` reaches ``prices`` only through
``output``. Coherence sees the link; partial coherence removes it:

.. testcode::

   w = model.frequencies
   business = (w >= 2 * np.pi / 32) & (w <= 2 * np.pi / 6)
   print(int(business.sum()))
   print(round(float(model.coherence("cycle", "prices")[business].mean()), 2),
         round(float(model.partial_coherence("cycle", "prices")[business].mean()), 2))
   k = int(np.argmin(np.abs(model.periods() - 12)))
   lead = -model.phase("output", "cycle")[k] / w[k]
   print(round(float(model.periods()[k]), 1), round(float(lead), 2))

.. testoutput::

   54
   0.52 0.01
   12.1 1.46

At period 12, ``cycle`` leads ``output`` by 1.46 quarters.

Spectral Granger causality
--------------------------

Geweke (1982) splits the effect's spectrum into an intrinsic part and a
part driven by the cause's innovations:

.. math::

   f_{x \to y}(\omega) = \ln \frac{S_{yy}(\omega)}
   {H_{yy}(\omega)\, \tilde\Sigma_{yy}\, \overline{H_{yy}(\omega)}}
   \;\ge\; 0,
   \qquad
   \frac{1}{\pi}\int_0^{\pi} f_{x \to y}(\omega)\,d\omega
   = \ln \frac{\sigma^2_{\text{restricted}}}{\sigma^2_{\text{full}}}.

:class:`~cultivars.spectral.causality.SpectralCausality` computes the
unconditional measure for every ordered pair from one fitted system.
For more than two variables each pair's marginal process is factored by
Wilson's (1972) algorithm, following Dhamala, Rangarajan and Ding
(2008); nothing is re-estimated.

.. testcode::

   from cultivars.spectral.causality import (
       ConditionalSpectralCausality, SpectralCausality,
   )

   geweke = SpectralCausality(fit, n_frequencies=201).compute()
   print(geweke.measure.shape)
   print(geweke.integrated().round(3))
   print(round(geweke.band("cycle", "output", low_period=6, high_period=32), 3),
         round(geweke.band("output", "cycle", low_period=6, high_period=32), 3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   (201, 3, 3)
   [[0.    0.005 0.003]
    [0.628 0.    0.007]
    [0.285 0.485 0.   ]]
   1.569 0.001

The axes of ``measure`` and ``integrated()`` are ``[effect, cause]``:
row 1, column 0 above is ``cycle -> output``. ``pair(cause, effect)``
and ``band(cause, effect, ...)`` take the spoken order and transpose for
you.

The unconditional measure credits ``cycle`` with causing ``prices``
(0.285), although the effect runs entirely through ``output``. Geweke's
(1984) conditional measure removes that, but needs a second, restricted
model fitted without the cause, which you supply:

.. testcode::

   restricted = VAR(y[:, 1:], order=6, names=names[1:]).fit()
   conditional = ConditionalSpectralCausality(
       fit, restricted, n_frequencies=201,
   ).compute()
   print(conditional.cause, conditional.effect_names, conditional.measure.shape)
   print(conditional.integrated().round(3).tolist(), round(conditional.consistency, 3))

.. testoutput::

   cycle ('output', 'prices') (201, 2)
   [0.615, 0.004] 0.255

The cause is inferred as the one name the restricted model lacks; the
remaining names must appear in the same order. ``consistency`` is the
largest relative gap between the two models' implied innovation
variances: zero in population, and large when the restricted model is
too short (the true restricted process is generally VARMA, hence the
generous ``order=6``).

Trend-cycle filters
-------------------

The four classes in :mod:`~cultivars.spectral.filters` define the trend
four ways and return one record,
:class:`~cultivars.spectral.filters.DecompositionFilterResult`, with
``trend`` and ``cycle`` of shape ``(rows, k)``. Each filter is applied
column by column; a 1-D series becomes one column.

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Class (defaults)
     - Trend is
     - Bill on the result
   * - ``HodrickPrescottFilter(penalty=1600.0)``
       (Hodrick and Prescott 1997)
     - Minimizer of :math:`\sum(y_t-\tau_t)^2 + \lambda\sum(\Delta^2\tau_t)^2`
     - ``provisional``: the first and last :math:`\lceil\text{cutoff}\rceil`
       rows (cutoff 39.7 at 1600) will be revised
   * - ``ButterworthFilter(period=32.0, order=2)`` (Gomez 2001)
     - Same family, stated by cutoff period and order
     - ``provisional`` as for HP; ``penalty`` reports the implied
       :math:`\lambda`
   * - ``HamiltonFilter(horizon=8, lags=4)`` (Hamilton 2018)
     - Fitted value of :math:`y_{t+h}` on a constant and
       :math:`y_t,\dots,y_{t-p+1}`
     - ``offset`` :math:`= h + p - 1` rows lost at the start
   * - ``BeveridgeNelsonDecomposition(result)``
       (Beveridge and Nelson 1981; Morley 2002)
     - Long-horizon forecast net of drift, from a fitted ARIMA(p, 1, q)
     - ``offset`` 1; conditional on the ARIMA

``filter(data)`` applies the first three; the Beveridge-Nelson
decomposition takes a fitted
:class:`~cultivars.univariate.box_jenkins.ARIMA` result with ``d = 1``,
no seasonal part, no exogenous regressors and trend ``"n"`` or ``"c"``,
and runs through ``compute()``. Row ``i`` of the output is input
observation ``offset + i``.

A quarterly log-level with drift, a random-walk component and a
period-20 cycle:

.. testcode::

   from cultivars.spectral.filters import (
       BeveridgeNelsonDecomposition, ButterworthFilter, HamiltonFilter,
       HodrickPrescottFilter,
   )
   from cultivars.univariate.box_jenkins import ARIMA

   rng = np.random.default_rng(4)
   n = 240
   t = np.arange(n)
   true_cycle = 0.02 * np.sin(2 * np.pi * t / 20)
   log_gdp = 0.005 * t + np.cumsum(0.002 * rng.standard_normal(n)) + true_cycle

   hp = HodrickPrescottFilter().filter(log_gdp)
   print(hp.trend.shape, hp.offset, round(hp.cutoff, 1), int(hp.provisional.sum()))
   bw = ButterworthFilter(period=32, order=2).filter(log_gdp)
   print(round(bw.penalty, 1), int(bw.provisional.sum()))
   ham = HamiltonFilter().filter(log_gdp)
   print(ham.offset, ham.cycle.shape, bool(ham.provisional.any()))

   def fit_to_truth(res):
       c = res.cycle[:, 0]
       aligned = true_cycle[res.offset:res.offset + len(c)]
       return round(float(np.corrcoef(c, aligned)[0, 1]), 2)

   print(fit_to_truth(hp), fit_to_truth(bw), fit_to_truth(ham))

.. testoutput::

   (240, 1) 0 39.7 80
   677.1 64
   11 (229, 1) False
   0.97 0.97 0.66

Butterworth at ``period=32, order=2`` is HP at :math:`\lambda = 677`.
The Hamilton cycle is the eight-quarter forecast error, so it carries the
random-walk component's eight-step variation as well as the cycle; that
is its definition, not a failure. The point of Hamilton (2018) shows on
a pure random walk, which has no cycle: HP manufactures one, about as
large as the increments themselves, and the summary prints that ratio:

.. testcode::

   walk = np.cumsum(rng.standard_normal(n))
   hp_walk = HodrickPrescottFilter().filter(walk)
   print(round(float(hp_walk.cycle.std() / np.diff(walk).std()), 2))

   bn = BeveridgeNelsonDecomposition(ARIMA(log_gdp, order=(1, 1, 0)).fit()).compute()
   print(bn.method, bn.offset, bn.cycle.shape, bn.order)

.. testoutput::

   1.13
   beveridge-nelson 1 (239, 1) (1, 0)

Band-pass filters
-----------------

The ideal band-pass between periods ``low`` and ``high`` has weights
:math:`b_0 = (\omega_h - \omega_l)/\pi`,
:math:`b_j = (\sin j\omega_h - \sin j\omega_l)/(\pi j)`, with
:math:`\omega_l = 2\pi/\text{high}` and :math:`\omega_h = 2\pi/\text{low}`.
Both filters approximate it and return
:class:`~cultivars.spectral.band_pass.BandPassResult` with ``cycle``,
``trend`` (everything outside the band, low *and* high frequencies),
``low``, ``high``, ``method``, ``offset`` and ``symmetric``.

``BaxterKingFilter(low=6.0, high=32.0, truncation=12)``
   Baxter and King (1999): symmetric weights truncated at lag :math:`K`
   and recentred to sum to zero. Phase-neutral and final everywhere it
   is defined; loses ``truncation`` rows at each end.
``ChristianoFitzgeraldFilter(low=6.0, high=32.0)``
   Christiano and Fitzgerald (2003), random-walk version: keeps every
   row by letting the weights turn asymmetric near the ends.
   ``symmetric`` marks the rows with at least ``ceil(high)`` observations
   on both sides; elsewhere the value is phase-shifted and will be
   revised.

.. testcode::

   from cultivars.spectral.band_pass import (
       BaxterKingFilter, ChristianoFitzgeraldFilter,
   )

   bk = BaxterKingFilter().filter(log_gdp)
   print(bk.cycle.shape, bk.offset, bk.low, bk.high, bool(bk.symmetric.all()))
   print(fit_to_truth(bk))
   cf = ChristianoFitzgeraldFilter().filter(log_gdp)
   print(cf.cycle.shape, cf.offset, int(cf.symmetric.sum()))

.. testoutput::

   (216, 1) 12 6.0 32.0 True
   0.99
   (240, 1) 0 176

See `Nuances and gotchas`_ before relying on the Christiano-Fitzgerald
output.

Dating cycles
-------------

:class:`~cultivars.spectral.cycles.TurningPoints` applies the Bry and
Boschan (1971) rules in Harding and Pagan's (2002) form: a peak is a
local maximum over ``window`` observations on each side, turns
alternate, and phases and cycles must last at least ``min_phase`` and
``min_cycle`` observations. ``convention="quarterly"`` is
:math:`(2, 2, 5)` and ``"monthly"`` is :math:`(5, 6, 15)`; each element
can be overridden by keyword. ``date(endog)`` returns a
:class:`~cultivars.spectral.cycles.TurningPointsResult` with ``peaks``,
``troughs`` (indices into the input), a ``contraction`` indicator, a
``phases`` table of ``(start, end, is_contraction, duration, amplitude,
excess)`` and ``n_cycles``. The concordance index of Harding and Pagan
(2006) is the share of dates on which two contraction indicators agree.
Dating the Baxter-King cycle above against the true cycle's chronology
on the same rows:

.. testcode::

   from cultivars.spectral.cycles import TurningPoints, concordance

   dated = TurningPoints().date(bk.cycle[:, 0])
   print(dated.peaks.tolist(), dated.n_cycles)
   rows = slice(bk.offset, bk.offset + bk.cycle.shape[0])
   truth = TurningPoints().date(true_cycle[rows])
   print(truth.peaks.tolist())
   print(round(dated.concordance(truth), 2),
         round(concordance(dated, truth.contraction), 2))
   print(dated.phases[0][:4])

.. testoutput::

   [13, 32, 53, 73, 93, 114, 133, 153, 173, 193, 212] 10
   [13, 33, 53, 73, 93, 113, 133, 153, 173, 193, 213]
   0.96 0.96
   (3, 13, False, 10)

Indices are positions in the series passed to ``date``: add
``bk.offset`` to map a Baxter-King row back to the original sample.
``concordance`` also accepts a probability series (thresholded at one
half), such as smoothed recession probabilities from
:class:`~cultivars.univariate.regime_switching.MarkovSwitchingARResult`.

Wavelets
--------

:class:`~cultivars.spectral.wavelets.MODWT` (Percival and Walden 2000)
splits one series into octave bands, each aligned in time:
:math:`y_t = \sum_{j=1}^J D_{j,t} + S_{J,t}`, where level :math:`j`
covers periods :math:`[2^j, 2^{j+1})`. ``wavelet`` is ``"la8"``
(default), ``"d4"`` or ``"haar"``; ``levels`` defaults to the deepest
level that keeps a boundary-free coefficient. ``transform(endog)``
takes a 1-D series of at least 32 observations.

:class:`~cultivars.spectral.wavelets.WaveletCoherence` (Torrence and
Compo 1998; Grinsted, Moore and Jevrejeva 2004) computes Morlet
coherence over scale and date, with a cone of influence and an AR(1)
surrogate significance level. ``compute(first, second)`` takes two
aligned 1-D series. The surrogates dominate the cost: the default
``surrogates=300`` takes seconds; ``surrogates=0`` skips the
significance level. Pass ``seed`` for a reproducible level.

A period-24 cycle shared by two series in the first half of the sample
only, the second series lagging by three observations:

.. testcode::

   from cultivars.spectral.wavelets import MODWT, WaveletCoherence

   rng = np.random.default_rng(5)
   n = 256
   t = np.arange(n)
   x = np.sin(2 * np.pi * t / 24) + 0.5 * rng.standard_normal(n)
   shared = np.where(t < 128, np.sin(2 * np.pi * (t - 3) / 24), 0.0)
   z = shared + 0.5 * rng.standard_normal(n)

   mra = MODWT().transform(x)
   print(mra.levels, mra.wavelet, mra.details.shape, mra.periods[3])
   print(bool(np.allclose(mra.details.sum(axis=0) + mra.smooth, x)))
   print(int(np.argmax(mra.variance)) + 1, mra.variance.round(3).tolist())

   wc = WaveletCoherence(surrogates=50, seed=0).compute(x, z)
   print(wc.coherence.shape, wc.periods.shape)
   band = (wc.periods >= 20) & (wc.periods <= 28)
   significant = wc.significant()
   print(round(float(significant[band][:, 40:100].mean()), 2),
         round(float(significant[band][:, 160:220].mean()), 2))
   print(round(wc.lead((20, 28)), 1))

.. testoutput::

   5 la8 (5, 256) (16.0, 32.0)
   True
   4 [0.121, 0.058, 0.06, 0.398, 0.022]
   (57, 256) (57,)
   1.0 0.0
   3.4

Level 4 (periods 16 to 32) holds the cycle's variance; coherence is
significant throughout the first half and nowhere in the second, which
no full-sample spectrum can show; and ``lead`` recovers the three-period
delay. ``MODWTResult`` also gives ``coefficients``, ``scaling``,
``degrees_of_freedom``, ``variance_interval(alpha)``, ``scales`` and
``summary()``; ``WaveletCoherenceResult`` gives ``phase``, ``scales``,
``cone`` (``True`` inside the cone of influence), ``significance`` and
``summary()``.

How the pieces connect
----------------------

* **Fitted VARs feed the model-implied views.** ``SpectralDensity``,
  ``SpectralCausality`` and ``ConditionalSpectralCausality`` accept any
  result that satisfies the closed-system protocol, the same protocol
  the identification schemes on :doc:`structural_identification` read.
  The time-domain counterpart of the integrated Geweke measure is
  ``granger_causality(cause, effect)`` on a fitted VAR result
  (:doc:`var`).
* **Data and model overlay.** Fit a VAR, estimate a multitaper spectrum
  on the same sample with ``n_frequencies = N // 2 + 1``, and read the
  model density against the data's confidence band, as in
  `Model-implied spectra from fitted systems`_. This is a frequency-wise
  misspecification check to set beside the residual diagnostics on
  :doc:`diagnostics`.
* **Filters feed dating.** A band-pass or trend-cycle ``cycle`` column
  is the input for a growth-cycle chronology; the level itself is the
  input for a classical chronology.
* **Models and filters agree where they should.** The smooth-trend
  :class:`~cultivars.univariate.unobserved_components.UnobservedComponents`
  model reproduces the HP trend at the penalty
  :math:`\sigma^2_\varepsilon / \sigma^2_\zeta` it estimates
  (:doc:`univariate_linear`); its cycle period is
  :math:`2\pi/\lambda_c` in the units of this page.
* **Regime models feed concordance.** Smoothed recession probabilities
  from a Markov-switching model (:doc:`univariate_nonlinear`) go
  straight into :func:`~cultivars.spectral.cycles.concordance`.

Nuances and gotchas
-------------------

* **Christiano-Fitzgerald output is wrong in this release.** The
  endpoint weights of
  :class:`~cultivars.spectral.band_pass.ChristianoFitzgeraldFilter` are
  off by one lag, so the weights applied at each date do not sum to
  zero: a constant series returns a nonzero cycle, and a trending level
  leaks into the cycle everywhere, not only near the ends. Above, the CF
  cycle of ``log_gdp`` correlates 0.45 with the true cycle where
  Baxter-King reaches 0.99. Use ``BaxterKingFilter`` until this is
  fixed, and check any CF output on a constant first:

  .. testcode::

     flat = np.ones(100)
     cf_flat = ChristianoFitzgeraldFilter().filter(flat).cycle
     bk_flat = BaxterKingFilter().filter(flat).cycle
     print(round(float(np.abs(cf_flat).max()), 3),
           round(float(np.abs(bk_flat).max()), 3), fit_to_truth(cf))

  .. testoutput::

     0.215 0.0 0.45

* **VARMA results give the wrong density.** ``SpectralDensity`` and
  ``SpectralCausality`` build :math:`\Psi(\omega)` from ``coefficients``
  alone. For a
  :class:`~cultivars.multivariate.reduced_form.moving_average.VARMA`
  result that is the autoregressive block only, so the moving-average
  part is silently dropped (a VARMA(1, 1) with MA coefficient 0.8
  returns about half the true variance). Pass VAR or VECM results.
* **Not every multivariate result is accepted.**
  :class:`~cultivars.multivariate.large_dim.bayesian.BVAR` results lack
  ``ma_representation`` and are refused with
  :class:`~cultivars.exceptions.SpecificationError`.
* **Unit roots.** A near-unit root gives a legitimately huge density at
  :math:`\omega = 0` (a fitted VECM in levels does this); an exact unit
  root makes the lag polynomial singular and ``compute()`` raises
  :class:`~cultivars.exceptions.NumericalError`. Summaries exclude
  :math:`\omega = 0` from the peak search for this reason.
* **Grids align only by construction.** A periodogram's grid is
  :math:`2\pi j/N`, :math:`j = 0,\dots,\lfloor N/2 \rfloor`, ending
  short of :math:`\pi` when :math:`N` is odd; the model grid always ends
  at :math:`\pi`. Welch uses the segment's grid, ``segment // 2 + 1``
  points. Check ``np.allclose`` on the two ``frequencies`` arrays before
  comparing ordinates.
* **Phase signs differ between modules.** ``SpectralDensityResult.phase``
  is negative when ``cause`` leads ``effect``. ``WaveletCoherenceResult``
  ``phase`` and ``lead`` are positive when the *first* series leads.
* **Axis order of causality arrays is** ``[effect, cause]``.
  ``measure[:, i, j]`` is ``j -> i``; use ``pair(cause, effect)`` to
  avoid transposition errors. ``band`` raises if no grid point falls in
  the band; refine ``n_frequencies`` for narrow bands.
* **Conditional causality needs matching names.** The restricted model
  must contain the full model's names minus exactly one, in the same
  order; give it explicit ``names``. Nothing checks that the two models
  were fitted on the same sample; ``consistency`` will be large if they
  were not.
* **Rows lost and rows provisional.** Hamilton loses ``horizon + lags -
  1`` rows at the start, Baxter-King ``truncation`` at each end,
  Beveridge-Nelson one. HP and Butterworth keep every row but flag
  ``ceil(cutoff)`` rows at each edge as ``provisional`` (80 of 240 at
  ``penalty=1600``); a sample shorter than twice the cutoff is
  provisional throughout.
* **The band-pass "trend" is not a trend.** ``BandPassResult.trend`` is
  the input minus the cycle, so it holds the frequencies above the band
  as well as below it. Its summary's "cycle share of variance" uses that
  trend, so for a trending level it is close to zero; filter a detrended
  or differenced series to read it as a variance share.
* **Turning points near the ends.** The first and last ``window``
  observations can never be turns, and a phase open at the sample end is
  absent from ``phases``. The quarterly triple dates spurious short
  phases on noisy series; use ``convention="monthly"``, a larger
  ``min_phase``, or a smoothed input.
* **Concordance base rate.** Two independent chronologies with
  contraction shares :math:`p, p'` agree at rate
  :math:`pp' + (1-p)(1-p')`, not one half. A probability of exactly 0.5
  counts as expansion.
* **Wavelet edges.** MODWT is circular: the first
  :math:`(2^j - 1)(L - 1)` coefficients at level :math:`j` wrap around,
  and the variance excludes them, so the deepest level has very few
  degrees of freedom (``degrees_of_freedom`` shows it). Coherence inside
  ``cone`` is contaminated by padding; ``significant()`` and ``lead``
  ignore it, and ``significant()`` raises if ``surrogates=0``.
* **Univariate inputs.** ``MODWT``, ``TurningPoints`` and
  ``WaveletCoherence`` take 1-D series; the periodogram estimators and
  the filters take panels. ``SpectralCausality`` needs at least two
  variables.

See also
--------

* :doc:`var` and :doc:`var_extensions`: the fitted systems the
  model-implied views read.
* :doc:`structural_identification`: the same closed-system protocol,
  read for identification.
* :doc:`univariate_linear`: unobserved components, the model-based
  trend-cycle decomposition, and ARIMA for Beveridge-Nelson.
* :doc:`univariate_nonlinear`: Markov-switching regime probabilities
  for concordance.
* :doc:`diagnostics`: Granger tests, residual checks and long-memory
  estimators.
* :doc:`state_space`: the filtering engines behind unobserved
  components.
* API: :mod:`cultivars.spectral.periodogram`,
  :mod:`cultivars.spectral.density`, :mod:`cultivars.spectral.causality`,
  :mod:`cultivars.spectral.filters`, :mod:`cultivars.spectral.band_pass`,
  :mod:`cultivars.spectral.cycles`, :mod:`cultivars.spectral.wavelets`.

References: Thomson (1982); Welch (1967); Percival and Walden (1993,
2000); Hamilton (1994); Geweke (1982, 1984); Wilson (1972); Dhamala,
Rangarajan and Ding (2008); Hodrick and Prescott (1997); Ravn and Uhlig
(2002); Hamilton (2018); Gomez (2001); Beveridge and Nelson (1981);
Morley (2002); Baxter and King (1999); Christiano and Fitzgerald (2003);
Burns and Mitchell (1946); Bry and Boschan (1971); Harding and Pagan
(2002, 2006); Torrence and Compo (1998); Grinsted, Moore and Jevrejeva
(2004).
