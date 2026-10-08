# filepath: /src/cultivars/spectral/__init__.py
#
# Copyright (c) 2026 Nikhil Sunder
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
r"""Frequency-domain views of fitted systems and data, filters, cycle dating, and wavelets.

Everything in the time domain has a reading by frequency, and this
package is where that reading is taken. The workflow runs from the
data toward a model and back. A nonparametric spectrum says where a
series' variance sits, before any model is fitted; a filter isolates
the band that matters -- the business cycle, by the Burns-Mitchell
convention of six to thirty-two quarters -- or removes the trend so
that what remains can be dated into peaks and troughs and compared
across series for concordance; a fitted closed system's own spectral
density, coherence, phase and Geweke causality then say what the model
claims at each frequency, on the same axes as the data's estimate, so
the two can be overlaid and their disagreement read as misspecification
where it occurs; and the wavelet transforms add the date axis, for the
question of whether a comovement held throughout the sample or only in
part of it.

The package has one rule that every module keeps: a number from the
data and a number from a model are never the same type. The
periodogram estimators return :class:`~cultivars.spectral.periodogram.SpectrumEstimate`
with degrees of freedom and a resolution; the model's density returns
:class:`~cultivars.spectral.density.SpectralDensityResult` with neither,
because it has none -- it is a closed form of the fitted parameters and
inherits their every error. Both carry the :math:`1/2\pi`
normalization and the same grid on :math:`[0, \pi]`, so overlaying
them is a plot and not a conversion, and each result states its own
limits: the rows a filter lost or will revise, the cone of influence,
the surrogate level a coherence must beat, the boundary coefficients a
wavelet variance drops.

Each module is a leaf: names are imported from the module, not from
this package. The filters and estimators take arrays, so any series can
be filtered, dated, or estimated; the model-implied views take a fitted
closed reduced-form result -- a VAR, a VARMA, an error-correction model
in levels -- and re-estimate nothing.

Layout. :mod:`~cultivars.spectral.periodogram` is the nonparametric
estimate: :class:`~cultivars.spectral.periodogram.MultitaperSpectrum`,
:class:`~cultivars.spectral.periodogram.DaniellSpectrum` and
:class:`~cultivars.spectral.periodogram.WelchSpectrum`.
:mod:`~cultivars.spectral.density` is the model's:
:class:`~cultivars.spectral.density.SpectralDensity`, with coherence,
partial coherence, gain and phase on its result;
:mod:`~cultivars.spectral.causality` decomposes the same matrix into
Geweke's directed measures,
:class:`~cultivars.spectral.causality.SpectralCausality` and its
conditional form. :mod:`~cultivars.spectral.filters` holds the
trend-cycle definitions --
:class:`~cultivars.spectral.filters.HodrickPrescottFilter`,
:class:`~cultivars.spectral.filters.ButterworthFilter`,
:class:`~cultivars.spectral.filters.HamiltonFilter`,
:class:`~cultivars.spectral.filters.BeveridgeNelsonDecomposition` --
and :mod:`~cultivars.spectral.band_pass` the band isolators,
:class:`~cultivars.spectral.band_pass.BaxterKingFilter` and
:class:`~cultivars.spectral.band_pass.ChristianoFitzgeraldFilter`.
:mod:`~cultivars.spectral.cycles` dates the result,
:class:`~cultivars.spectral.cycles.TurningPoints`, and measures
:func:`~cultivars.spectral.cycles.concordance` between chronologies.
:mod:`~cultivars.spectral.wavelets` is
:class:`~cultivars.spectral.wavelets.MODWT` and
:class:`~cultivars.spectral.wavelets.WaveletCoherence`. The
frequency-domain long-memory estimators that read the spectrum's
behaviour at the origin live with the other diagnostics in
:mod:`~cultivars.diagnostics.long_memory`.

Example:
    An AR(2) with a cycle near 17 observations driving a second series.
    The spectrum finds the period, the band-pass isolates it, the dating
    counts its turns, and the fitted model's density agrees with the
    data's estimate and attributes the causality to the first series:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((400, 2))
    >>> for t in range(2, 400):
    ...     y[t, 0] = 1.5 * y[t - 1, 0] - 0.7 * y[t - 2, 0] + rng.standard_normal()
    ...     y[t, 1] = 0.5 * y[t - 1, 0] + rng.standard_normal()
    >>> est = periodogram.MultitaperSpectrum(y).compute()
    >>> power = est.spectrum("y1")
    >>> round(float(est.periods()[1 + int(np.argmax(power[1:]))]), 1)
    16.7
    >>> cycle = band_pass.BaxterKingFilter(low=6, high=32).filter(y[:, 0])
    >>> turns = cycles.TurningPoints().date(cycle.cycle[:, 0])
    >>> turns.n_cycles, int(np.median(np.diff(turns.peaks)))
    (32, 11)
    >>> fit = VAR(y, order=2).fit()
    >>> model = density.SpectralDensity(fit, n_frequencies=len(est.frequencies)).compute()
    >>> agreement = np.corrcoef(np.log(power[1:]), np.log(model.spectrum("y1")[1:]))[0, 1]
    >>> bool(agreement > 0.95)
    True
    >>> geweke = causality.SpectralCausality(fit).compute()
    >>> to_second = geweke.band("y1", "y2", low_period=6.0, high_period=32.0)
    >>> to_first = geweke.band("y2", "y1", low_period=6.0, high_period=32.0)
    >>> bool(to_second > 1.0), round(to_first, 2)
    (True, 0.0)
"""

from . import band_pass, causality, cycles, density, filters, periodogram, wavelets

__all__ = [
    "band_pass",
    "causality",
    "cycles",
    "density",
    "filters",
    "periodogram",
    "wavelets",
]
