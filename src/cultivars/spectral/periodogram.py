# filepath: /src/cultivars/spectral/periodogram.py
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
r"""Nonparametric spectra: the data's own frequency-domain story, model-free.

:mod:`~cultivars.spectral.density` reads the spectrum off a fitted
system and inherits every specification error the fit carries. This
module estimates it from the data alone, on the same Fourier grid and
with the same :math:`1/2\pi` normalization, so the two can be drawn on
one plot and their disagreement read as the model's misspecification at
each frequency. The starting point is the periodogram of a demeaned
sample,

.. math::

   I(\omega_j) = \frac{1}{2\pi N}\Bigl|\sum_{t=1}^{N} y_t e^{-i\omega_j t}\Bigr|^2,
   \qquad \omega_j = \frac{2\pi j}{N},

which is unbiased for :math:`f(\omega)` asymptotically but inconsistent
-- each ordinate is :math:`f(\omega_j)\chi^2_2/2` however long the
sample -- so every estimator here trades resolution for stability in a
stated way and reports both sides of the trade. The smoothed periodogram
averages :math:`2m + 1` adjacent ordinates with a modified Daniell
kernel; Welch averages periodograms of half-overlapped Hann-tapered
segments on the segment's coarser grid; Thomson's multitaper averages
periodograms of the same full sample under :math:`K` orthogonal Slepian
tapers, which is the least leakage-prone of the three at a given
variance and the default. Every ordinate of every estimator is
approximately :math:`f(\omega)\chi^2_\nu/\nu`, and the result carries
:math:`\nu` so a confidence band is one method call.

Two commitments shape the surface. First, the result is a distinct type
from :class:`~cultivars.spectral.density.SpectralDensityResult` on
purpose: a number from a periodogram and a number from a fitted model
answer different questions, and the package never lets one be mistaken
for the other -- the two overlay, they do not interconvert. Second,
resolution and degrees of freedom are stated on the result rather than
left to the reader's memory of the method: ``bandwidth`` says how close
two peaks can be before they merge and ``degrees_of_freedom`` says how
wide the band around each ordinate is, so the smoothing choice can be
audited from the printed summary.

Layout. :class:`MultitaperSpectrum`, :class:`DaniellSpectrum` and
:class:`WelchSpectrum` take a panel at construction, validate the
smoothing design, and estimate through ``compute``; all three return
:class:`SpectrumEstimate`, which exposes per-series spectra, squared
coherence, periods and the chi-squared band. The numerics live in
``_core``: ``_cross_periodogram`` forms the :math:`(n, k, k)` tapered
cross-periodogram, ``_fourier_frequencies`` the grid,
``_multitaper_density``, ``_daniell_density`` and ``_welch_density`` the
three averages with their degrees of freedom, ``_ols_detrend`` removes
the deterministic terms, ``_validate_spectrum_panel`` coerces the input
and resolves labels with the ``_MIN_SPECTRUM_OBS`` floor of 32, and
``_SPECTRAL_METHOD_TITLES`` names the methods in summaries.
``_SummaryMixin`` from ``_internals`` gives the result its printed form.
The model-implied density these estimates are meant to be overlaid on
is :class:`~cultivars.spectral.density.SpectralDensity`; the band-pass
filters that isolate whatever band the spectrum shows are in
:mod:`~cultivars.spectral.band_pass`.

References:
    Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
    *Proceedings of the IEEE*, 70(9), 1055-1096.

    Welch, P. D. (1967). The use of fast Fourier transform for the
    estimation of power spectra: A method based on time averaging over
    short, modified periodograms. *IEEE Transactions on Audio and
    Electroacoustics*, 15(2), 70-73.

    Bloomfield, P. (2000). *Fourier Analysis of Time Series: An
    Introduction*, 2nd ed. Wiley.

    Percival, D. B., & Walden, A. T. (1993). *Spectral Analysis for
    Physical Applications*. Cambridge University Press.

Example:
    An AR(2) with a spectral peak near 0.55 radians, estimated from the
    data and compared with two fitted models on the same grid. The
    VAR(1) is monotone by construction and its density falls inside the
    95% band at less than a third of the ordinates; the VAR(2) finds the
    peak and sits inside the band almost everywhere:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> from cultivars.spectral.density import SpectralDensity
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros(1024)
    >>> for t in range(2, 1024):
    ...     y[t] = 1.4 * y[t - 1] - 0.7 * y[t - 2] + rng.standard_normal()
    >>> est = MultitaperSpectrum(y).compute()
    >>> lower, upper = est.confidence_interval("y")
    >>> inside = {}
    >>> for order in (1, 2):
    ...     model = SpectralDensity(VAR(y, order=order).fit(), n_frequencies=513).compute()
    ...     power = model.spectrum("y1")
    ...     inside[order] = round(float(np.mean((lower <= power) & (power <= upper))), 2)
    >>> inside
    {1: 0.3, 2: 0.97}
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from .._core import (
    _MIN_SPECTRUM_OBS,
    _SPECTRAL_METHOD_TITLES,
    SummaryTable,
    _daniell_density,
    _multitaper_density,
    _ols_detrend,
    _validate_spectrum_panel,
    _welch_density,
    validate_order,
)
from .._internals import _SummaryMixin
from ..exceptions import SpecificationError

__all__ = ["DaniellSpectrum", "MultitaperSpectrum", "SpectrumEstimate", "WelchSpectrum"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SpectrumEstimate(_SummaryMixin):
    r"""A nonparametric spectral density matrix, with its resolution and degrees of freedom.

    Frequencies are radians per observation on the Fourier grid
    :math:`\omega_j = 2\pi j / N`, :math:`j = 0, \dots, \lfloor N/2
    \rfloor` (or the segment's grid under Welch), and the density carries
    the :math:`1/2\pi` normalization, exactly as
    :class:`~cultivars.spectral.density.SpectralDensityResult`, so

    .. math::

        2 \int_0^{\pi} \hat f_{ii}(\omega)\, d\omega \approx \hat\sigma_i^2

    and the two objects overlay on one set of axes. Every ordinate is
    approximately :math:`f(\omega)\,\chi^2_\nu / \nu` with :math:`\nu`
    the estimator's equivalent degrees of freedom, which is what
    :meth:`confidence_interval` inverts, and the estimate cannot
    separate two peaks closer than ``bandwidth`` radians.

    Attributes:
        names: Series labels, indexing the matrix axes.
        frequencies: The ``(n,)`` Fourier grid on ``[0, pi]``.
        density: The ``(n, k, k)`` complex Hermitian estimate.
        method: ``"daniell"``, ``"multitaper"`` or ``"welch"``.
        degrees_of_freedom: Equivalent chi-squared degrees of freedom of
            each ordinate.
        bandwidth: Half-width of the frequency window in radians, the
            resolution below which two peaks merge.
        nobs: Observations the estimate used.
        detrend: The deterministic terms removed first.

    Note:
        This is the data's periodogram, smoothed, and not a model's
        closed form; the two are kept as distinct types so that a
        disagreement between them can be read as misspecification by
        frequency rather than lost. The price of consistency is bias:
        the smoothing window averages the true spectrum over
        ``bandwidth`` radians, so a sharp peak is flattened and the
        variance identity above holds only approximately (the taper's
        leakage and the window's averaging both cost a little). The
        summary's "peak" is the argmax of a noisy estimate over the
        interior grid; for a monotone spectrum such as an AR(1)'s it
        lands wherever sampling noise puts it in the low band, so read
        it with the confidence band, not as a point.

    See Also:
        * :class:`MultitaperSpectrum`, :class:`DaniellSpectrum`,
          :class:`WelchSpectrum` -- the producers.
        * :class:`~cultivars.spectral.density.SpectralDensityResult` --
          the model-implied density on the same axes, for the overlay.
        * :class:`~cultivars.spectral.band_pass.BaxterKingFilter` --
          which band to isolate, once the spectrum says where the power
          sits.

    References:
        Percival, D. B., & Walden, A. T. (1993). *Spectral Analysis for
        Physical Applications*. Cambridge University Press.

        Priestley, M. B. (1981). *Spectral Analysis and Time Series*.
        Academic Press.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(1024)
        >>> for t in range(1, 1024):
        ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal()
        >>> est = MultitaperSpectrum(y).compute()
        >>> est.density.shape, est.degrees_of_freedom, round(est.bandwidth, 4)
        ((513, 1, 1), 14.0, 0.0245)

        The estimate integrates to the sample variance, and the true
        AR(1) spectrum :math:`\sigma^2 / 2\pi|1 - \phi e^{-i\omega}|^2`
        lies inside the 95% band at about 95% of the ordinates:

        >>> power = est.spectrum("y")
        >>> round(float(2.0 * np.trapezoid(power, est.frequencies) / y.var()), 2)
        0.99
        >>> theory = 1.0 / (2.0 * np.pi * np.abs(1.0 - 0.7 * np.exp(-1j * est.frequencies)) ** 2)
        >>> lower, upper = est.confidence_interval("y")
        >>> round(float(np.mean((lower <= theory) & (theory <= upper))), 2)
        0.97

        The grid is the one a fitted model's density can be evaluated
        on, so the two overlay directly:

        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> from cultivars.spectral.density import SpectralDensity
        >>> model = SpectralDensity(VAR(y, order=1).fit(), n_frequencies=513).compute()
        >>> bool(np.allclose(model.frequencies, est.frequencies))
        True
    """

    names: tuple[str, ...]
    """Series labels, indexing the matrix axes."""

    frequencies: npt.NDArray[np.float64] = field(repr=False)
    """``(n,)`` Fourier grid on ``[0, pi]``. Kept out of the repr."""

    density: npt.NDArray[np.complex128] = field(repr=False)
    """``(n, k, k)`` complex Hermitian estimate. Kept out of the repr."""

    method: str
    """``"daniell"``, ``"multitaper"`` or ``"welch"``."""

    degrees_of_freedom: float
    """Equivalent chi-squared degrees of freedom per ordinate."""

    bandwidth: float
    """Half-width of the frequency window in radians."""

    nobs: int
    """Observations the estimate used."""

    detrend: str
    """``"n"``, ``"c"`` or ``"ct"``: the deterministic terms removed first."""

    @property
    def k_endog(self) -> int:
        """Number of series.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> WelchSpectrum(rng.standard_normal((256, 3))).compute().k_endog
            3
        """
        return len(self.names)

    def _index(self, name: str) -> int:
        """Resolve a series label.

        Args:
            name: A label from ``names``.

        Returns:
            Its position on the matrix axes.

        Raises:
            SpecificationError: If the label is unknown.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> est = WelchSpectrum(rng.standard_normal((256, 2))).compute()
            >>> est._index("y2")
            1
            >>> est._index("z")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: unknown series 'z'; the estimate has ...
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise SpecificationError(
                f"unknown series {name!r}; the estimate has {self.names}."
            ) from None

    def periods(self) -> npt.NDArray[np.float64]:
        """Each grid frequency as a period in observations per cycle, zero mapping to infinity.

        Returns:
            ``(n,)`` periods ``2 pi / omega``, ``inf`` at the origin.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> est = DaniellSpectrum(rng.standard_normal(64)).compute()
            >>> est.periods()[[0, 1, 2, -1]].tolist()
            [inf, 64.0, 32.0, 2.0]
        """
        with np.errstate(divide="ignore"):
            return np.asarray(
                np.where(
                    self.frequencies > 0.0,
                    2.0 * np.pi / np.maximum(self.frequencies, 1e-300),
                    np.inf,
                ),
                dtype=np.float64,
            )

    def spectrum(self, name: str) -> npt.NDArray[np.float64]:
        r"""One series' estimated power spectrum, the real diagonal of the density.

        Args:
            name: The series label.

        Returns:
            ``(n,)`` non-negative power, normalized so that twice its
            integral over ``[0, pi]`` is the series' variance.

        Raises:
            SpecificationError: If the label is unknown.

        Example:
            White noise of unit variance has the flat spectrum
            :math:`1 / 2\pi`:

            >>> rng = np.random.default_rng(0)
            >>> est = MultitaperSpectrum(rng.standard_normal(2048)).compute()
            >>> round(float(2.0 * np.pi * est.spectrum("y").mean()), 1)
            1.0
        """
        i = self._index(name)
        return np.asarray(np.real(self.density[:, i, i]), dtype=np.float64)

    def coherence(self, first: str, second: str) -> npt.NDArray[np.float64]:
        r"""Squared coherence between two series at each frequency.

        .. math::

            \hat C_{xy}(\omega)
            = \frac{|\hat f_{xy}(\omega)|^2}{\hat f_{xx}(\omega)\,\hat f_{yy}(\omega)}
            \in [0, 1],

        the frequency-domain :math:`R^2` of one series on the other.
        With :math:`\nu` degrees of freedom the estimate is biased upward
        by about :math:`2 / \nu` under zero coherence, which the
        smoothing makes small; a raw periodogram (:math:`\nu = 2`) would
        report coherence one everywhere, which is why none is offered.

        Args:
            first: One series label.
            second: The other; order is immaterial.

        Returns:
            ``(n,)`` squared coherence.

        Raises:
            SpecificationError: If either label is unknown.

        Example:
            Two independent noises show the :math:`2 / \nu` floor, and a
            series against its own lag plus noise is coherent where the
            signal dominates:

            >>> rng = np.random.default_rng(0)
            >>> est = MultitaperSpectrum(rng.standard_normal((2048, 2))).compute()
            >>> floor = 2 / est.degrees_of_freedom
            >>> round(float(est.coherence("y1", "y2").mean()), 2), round(floor, 2)
            (0.14, 0.14)
            >>> y = np.zeros(1024)
            >>> for t in range(1, 1024):
            ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal()
            >>> z = np.column_stack([y, np.roll(y, 1) + rng.standard_normal(1024)])
            >>> coh = MultitaperSpectrum(z).compute().coherence("y1", "y2")
            >>> bool(coh[1:40].mean() > 0.8), bool(coh[-40:].mean() < 0.5)
            (True, True)
        """
        i, j = self._index(first), self._index(second)
        cross = np.abs(self.density[:, i, j]) ** 2
        auto = np.real(self.density[:, i, i]) * np.real(self.density[:, j, j])
        return np.asarray(cross / np.maximum(auto, 1e-300), dtype=np.float64)

    def confidence_interval(
        self, name: str, *, alpha: float = 0.05
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Pointwise ``1 - alpha`` band for one spectrum from the chi-squared approximation.

        From :math:`\nu \hat f / f \sim \chi^2_\nu`,

        .. math::

            \Bigl[\frac{\nu \hat f(\omega)}{\chi^2_{\nu, 1-\alpha/2}},\;
            \frac{\nu \hat f(\omega)}{\chi^2_{\nu, \alpha/2}}\Bigr],

        a band of constant *ratio* across frequencies, which is why
        spectra are plotted on a log scale. The band is pointwise, not
        simultaneous.

        Args:
            name: The series label.
            alpha: Tail mass, strictly inside ``(0, 1)``.

        Returns:
            ``(lower, upper)`` arrays of shape ``(n,)``.

        Raises:
            SpecificationError: If ``alpha`` is not inside ``(0, 1)`` or
                the label is unknown.

        Example:
            >>> from scipy.stats import chi2
            >>> rng = np.random.default_rng(0)
            >>> est = MultitaperSpectrum(rng.standard_normal(512)).compute()
            >>> lower, upper = est.confidence_interval("y")
            >>> nu = est.degrees_of_freedom
            >>> ratio = chi2.ppf(0.975, nu) / chi2.ppf(0.025, nu)
            >>> bool(np.allclose(upper / lower, ratio))
            True
            >>> est.confidence_interval("y", alpha=1.0)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: alpha must lie strictly inside (0, 1); got 1.0.
        """
        if not 0.0 < alpha < 1.0:
            raise SpecificationError(f"alpha must lie strictly inside (0, 1); got {alpha}.")
        power = self.spectrum(name)
        nu = self.degrees_of_freedom
        lower = power * nu / float(chi2.ppf(1.0 - alpha / 2.0, nu))
        upper = power * nu / float(chi2.ppf(alpha / 2.0, nu))
        return lower, upper

    def _summary_table(self) -> SummaryTable:
        """Peak and variance per series, with the estimate's resolution.

        One row per series: the interior frequency of largest power, its
        period, and twice the trapezoidal integral of the spectrum (the
        variance the estimate accounts for). The notes state the degrees
        of freedom and resolution, that the object is model-free, and
        that the peak search excludes the origin.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> t = np.arange(512.0)
            >>> y = np.cos(2 * np.pi * t / 16) + 0.3 * rng.standard_normal(512)
            >>> table = MultitaperSpectrum(y).compute()._summary_table()
            >>> table.title, table.columns
            ('Multitaper Spectrum', ('series', 'peak frequency', 'peak period', 'variance'))
            >>> table.rows[0][:3]
            ('y', '0.3927', '16.0')
        """
        rows = []
        interior = slice(1, None)
        for name in self.names:
            power = self.spectrum(name)
            peak = 1 + int(np.argmax(power[interior]))
            frequency = float(self.frequencies[peak])
            period = 2.0 * np.pi / frequency if frequency > 0.0 else np.inf
            variance = 2.0 * float(np.trapezoid(power, self.frequencies))
            rows.append((name, f"{frequency:.4f}", f"{period:.1f}", f"{variance:.4g}"))
        notes = [
            f"Equivalent degrees of freedom {self.degrees_of_freedom:.1f} per ordinate; the "
            f"frequency resolution is about {self.bandwidth:.4f} radians (periods closer than "
            "that merge).",
            "Model-free: this is the data's periodogram, smoothed, and not a fitted model's "
            "closed form; overlay it on SpectralDensity to read misspecification by "
            "frequency.",
            "Peak frequency excludes the zero-frequency point, where a trending series' "
            "power legitimately concentrates.",
        ]
        return SummaryTable(
            title=f"{_SPECTRAL_METHOD_TITLES[self.method]} Spectrum",
            metadata=(
                ("Series", str(self.k_endog)),
                ("Observations", str(self.nobs)),
                ("Grid", f"{len(self.frequencies)} Fourier frequencies on [0, pi]"),
                ("Detrend", self.detrend),
            ),
            columns=("series", "peak frequency", "peak period", "variance"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class MultitaperSpectrum:
    r"""Thomson's multitaper spectral estimate.

    :math:`K` discrete prolate spheroidal (Slepian) sequences of
    time-bandwidth product :math:`NW` are the orthogonal tapers that
    concentrate the most energy inside :math:`\pm W` cycles per
    observation; each gives a nearly leakage-free periodogram
    :math:`\hat f_k(\omega)`, and the estimate is their
    concentration-weighted average

    .. math::

        \hat f(\omega) = \frac{\sum_{k=1}^{K} \lambda_k\, \hat f_k(\omega)}
                              {\sum_{k=1}^{K} \lambda_k},

    with :math:`\lambda_k` the tapers' concentration ratios, close to one
    for :math:`K \le 2NW - 1` and falling off after. The average has
    :math:`2K` equivalent degrees of freedom and resolution
    :math:`2\pi NW / N` radians -- two peaks closer than that merge --
    so the trade is explicit: :math:`NW` buys smoothness at the cost of
    resolution, :math:`K` buys degrees of freedom up to the number of
    well-concentrated tapers. The usual :math:`NW = 4, K = 7` is the
    default; raise both for a smoother estimate, lower both to separate
    close peaks. Among the three estimators here it is the least biased
    at a given variance, and the default choice.

    Attributes:
        _panel: The ``(nobs, k)`` panel after detrending.
        _names: The series labels.
        _detrend: The validated detrending choice.
        _bandwidth: The time-bandwidth product :math:`NW`.
        _tapers: The taper count :math:`K`.

    Args:
        data: A ``(nobs, k)`` panel or one-dimensional series, at least
            32 observations.
        bandwidth: Time-bandwidth product :math:`NW`, at least 1 and
            below ``nobs / 2``; need not be an integer.
        n_tapers: Tapers :math:`K`; ``None`` takes :math:`2NW - 1`.
            At most :math:`2NW`.
        detrend: ``"n"``, ``"c"`` (demean) or ``"ct"`` (linear detrend),
            removed by least squares before tapering.
        names: Series labels; ``None`` reads them from a frame or uses
            ``y1 .. yk`` (``y`` for a single series).

    Raises:
        SpecificationError: If the bandwidth is below 1, the taper count
            is not an integer of at least 1 or exceeds :math:`2NW`, the
            detrend choice is unknown, the labels do not match the
            columns, or the panel is too short.
        DimensionError: If the panel is malformed.
        NumericalError: If the panel is not finite.

    Note:
        The :math:`2K` degrees of freedom are exact for an unweighted
        average of independent periodograms and slightly optimistic for
        the weighted one when the last tapers are poorly concentrated:
        at :math:`NW = 4` the eighth ratio is 0.70, and the effective
        count :math:`2 / \sum w_k^2` is 15.8 rather than 16. Keep
        :math:`K \le 2NW - 1` unless the extra taper is worth the
        leakage it admits. Detrending matters more here than for a
        model-implied density: an unremoved linear trend puts power at
        the origin that leaks through the tapers' sidelobes, and
        ``"ct"`` reduces the zero-frequency ordinate of a trending noise
        series by three orders of magnitude.

    See Also:
        * :class:`SpectrumEstimate` -- the result.
        * :class:`DaniellSpectrum` -- the smoothed periodogram, simpler
          and more biased.
        * :class:`WelchSpectrum` -- segment averaging, for long samples
          where resolution is not the constraint.

    References:
        Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
        *Proceedings of the IEEE*, 70(9), 1055-1096.

        Percival, D. B., & Walden, A. T. (1993). *Spectral Analysis for
        Physical Applications*, chapter 7. Cambridge University Press.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(400)
        >>> for t in range(1, 400):
        ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal()
        >>> est = MultitaperSpectrum(y).compute()
        >>> est.density.shape, bool(est.spectrum("y")[1] > est.spectrum("y")[-1])
        ((201, 1, 1), True)

        Degrees of freedom are :math:`2K` and the resolution is
        :math:`2\pi NW / N`; halving :math:`NW` halves both:

        >>> wide = MultitaperSpectrum(y, bandwidth=4).compute()
        >>> narrow = MultitaperSpectrum(y, bandwidth=2).compute()
        >>> wide.degrees_of_freedom, narrow.degrees_of_freedom
        (14.0, 6.0)
        >>> round(wide.bandwidth / narrow.bandwidth, 1), round(400 * wide.bandwidth / 2 / np.pi, 1)
        (2.0, 4.0)
    """

    __slots__ = ("_bandwidth", "_detrend", "_names", "_panel", "_tapers")

    def __init__(
        self,
        data: npt.ArrayLike,
        *,
        bandwidth: float = 4.0,
        n_tapers: int | None = None,
        detrend: str = "c",
        names: tuple[str, ...] | None = None,
    ) -> None:
        """Validate the panel and the taper design.

        Args:
            data: A ``(nobs, k)`` panel or one-dimensional series.
            bandwidth: Time-bandwidth product ``NW``.
            n_tapers: Tapers ``K``; ``None`` takes ``2 NW - 1``.
            detrend: ``"n"``, ``"c"`` or ``"ct"``.
            names: Series labels, or ``None``.

        Raises:
            SpecificationError: If ``bandwidth < 1``, ``n_tapers`` is not
                an integer of at least 1 or exceeds ``2 NW``, or the
                panel, detrend choice or labels fail validation.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(128)
            >>> MultitaperSpectrum(y)._tapers, MultitaperSpectrum(y, bandwidth=2.5)._tapers
            (7, 4)
            >>> MultitaperSpectrum(y, n_tapers=9)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: n_tapers must not exceed 2 NW = 8 ... got 9.
            >>> MultitaperSpectrum(y[:31])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: a nonparametric spectrum needs at least 32 ...
        """
        if not bandwidth >= 1.0:
            raise SpecificationError(f"bandwidth NW must be at least 1; got {bandwidth}.")
        tapers = int(2 * bandwidth - 1) if n_tapers is None else n_tapers
        self._tapers = validate_order(tapers, "n_tapers", minimum=1)
        if self._tapers > 2 * bandwidth:
            raise SpecificationError(
                f"n_tapers must not exceed 2 NW = {2 * bandwidth:g} for usable concentration; "
                f"got {self._tapers}."
            )
        self._bandwidth = float(bandwidth)
        self._detrend = detrend
        panel, self._detrend, self._names = _validate_spectrum_panel(
            data, detrend, names, minimum=_MIN_SPECTRUM_OBS
        )
        self._panel = _ols_detrend(panel, self._detrend)

    def compute(self) -> SpectrumEstimate:
        r"""Estimate the density matrix.

        Builds the :math:`K` Slepian tapers for the sample length,
        forms each taper's cross-periodogram on the Fourier grid, and
        averages them with concentration weights.

        Returns:
            The :class:`SpectrumEstimate` on ``nobs // 2 + 1`` Fourier
            frequencies, with ``degrees_of_freedom = 2 K`` and
            ``bandwidth = 2 pi NW / nobs``.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> est = MultitaperSpectrum(rng.standard_normal((256, 2)), bandwidth=3).compute()
            >>> est.method, est.frequencies.shape, est.degrees_of_freedom, est.nobs
            ('multitaper', (129,), 10.0, 256)
        """
        frequencies, density, nu = _multitaper_density(self._panel, self._bandwidth, self._tapers)
        n = self._panel.shape[0]
        return SpectrumEstimate(
            names=self._names,
            frequencies=frequencies,
            density=density,
            method="multitaper",
            degrees_of_freedom=nu,
            bandwidth=2.0 * np.pi * self._bandwidth / n,
            nobs=n,
            detrend=self._detrend,
        )


class DaniellSpectrum:
    r"""The periodogram smoothed with a modified Daniell kernel.

    The raw periodogram :math:`I(\omega_j)` at the Fourier frequencies
    is inconsistent -- each ordinate is :math:`f(\omega_j)\chi^2_2/2`
    however long the sample -- so adjacent ordinates are averaged with
    the flat kernel of half-width :math:`m` whose two end weights are
    halved,

    .. math::

        \hat f(\omega_j) = \sum_{|l| \le m} w_l\, I(\omega_{j+l}),
        \qquad
        w_l = \frac{1}{2m}\ (|l| < m),\quad w_{\pm m} = \frac{1}{4m},

    :math:`2m + 1` ordinates per estimate with the smoothing wrapped
    around the circle so the ends of the grid borrow from their mirror
    images. The equivalent degrees of freedom are
    :math:`\nu = 2/\sum_l w_l^2 = 16m^2/(4m - 1)`, about :math:`4m`, and
    the resolution is :math:`2\pi m / N` radians. The default span
    :math:`\lfloor \sqrt{N}/2 \rceil` is the usual compromise between
    resolution and variance. No taper is applied, so the estimate's
    leakage is the periodogram's own; it is the simplest of the three
    estimators and the one whose integral reproduces the sample variance
    exactly.

    Attributes:
        _panel: The ``(nobs, k)`` panel after detrending.
        _names: The series labels.
        _detrend: The detrending choice.
        _span: The kernel half-width :math:`m`.

    Args:
        data: A ``(nobs, k)`` panel or one-dimensional series, at least
            32 observations.
        span: Kernel half-width in ordinates, an integer of at least 1
            with ``2 span + 1`` at most ``nobs // 2``; ``None`` for the
            default.
        detrend: ``"n"``, ``"c"`` (demean) or ``"ct"`` (linear detrend),
            removed by least squares first.
        names: Series labels; ``None`` reads them from a frame or uses
            ``y1 .. yk`` (``y`` for a single series).

    Raises:
        SpecificationError: If the span is not an integer of at least 1,
            averages more ordinates than the grid holds, the detrend
            choice is unknown, the labels do not match the columns, or
            the panel is too short.
        DimensionError: If the panel is malformed.
        NumericalError: If the panel is not finite.

    Note:
        The span is the whole resolution-variance dial: at
        :math:`N = 1024` a span of 4 gives :math:`\nu = 17` and resolves
        peaks 0.025 radians apart, a span of 16 gives :math:`\nu = 65`
        and merges anything within 0.098 radians -- and flattens a sharp
        AR(1) peak at the origin to 97% of its height where the span-4
        estimate overshoots it by sampling noise. Read the printed
        ``bandwidth`` before reading a peak.

    See Also:
        * :class:`SpectrumEstimate` -- the result.
        * :class:`MultitaperSpectrum` -- tapered, leakage-resistant, the
          default.
        * :class:`WelchSpectrum` -- segment averaging on a coarser grid.

    References:
        Bloomfield, P. (2000). *Fourier Analysis of Time Series: An
        Introduction*, 2nd ed., chapter 8. Wiley.

        Percival, D. B., & Walden, A. T. (1993). *Spectral Analysis for
        Physical Applications*, chapter 6. Cambridge University Press.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> est = DaniellSpectrum(rng.standard_normal(400)).compute()
        >>> bool(abs(2.0 * np.pi * est.spectrum("y").mean() - 1.0) < 0.2)
        True

        The default span at :math:`N = 400` is 10, giving
        :math:`16 \cdot 100 / 39 = 41.0` degrees of freedom, and the
        estimate integrates to the sample variance exactly:

        >>> est.degrees_of_freedom, round(est.bandwidth, 4)
        (41.025641025641015, 0.1571)
        >>> y = rng.standard_normal(512)
        >>> est = DaniellSpectrum(y, span=4).compute()
        >>> round(float(2.0 * np.trapezoid(est.spectrum("y"), est.frequencies) / y.var()), 3)
        1.0
    """

    __slots__ = ("_detrend", "_names", "_panel", "_span")

    def __init__(
        self,
        data: npt.ArrayLike,
        *,
        span: int | None = None,
        detrend: str = "c",
        names: tuple[str, ...] | None = None,
    ) -> None:
        """Validate the panel and the span.

        Args:
            data: A ``(nobs, k)`` panel or one-dimensional series.
            span: Kernel half-width; ``None`` for ``round(sqrt(nobs) / 2)``.
            detrend: ``"n"``, ``"c"`` or ``"ct"``.
            names: Series labels, or ``None``.

        Raises:
            SpecificationError: If ``span`` is not an integer of at least
                1, ``2 span + 1`` exceeds ``nobs // 2``, or the panel,
                detrend choice or labels fail validation.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(400)
            >>> DaniellSpectrum(y)._span, DaniellSpectrum(y[:32])._span
            (10, 3)
            >>> DaniellSpectrum(y, span=100)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: a span of 100 averages more ordinates than ...
            >>> DaniellSpectrum(y, span=2.0)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: span must be an integer; got 2.0.
        """
        panel, self._detrend, self._names = _validate_spectrum_panel(
            data, detrend, names, minimum=_MIN_SPECTRUM_OBS
        )
        self._panel = _ols_detrend(panel, self._detrend)
        n = self._panel.shape[0]
        default = max(1, round(np.sqrt(n) / 2.0))
        self._span = validate_order(default if span is None else span, "span", minimum=1)
        if 2 * self._span + 1 > n // 2:
            raise SpecificationError(
                f"a span of {self._span} averages more ordinates than the {n // 2} available."
            )
        self._detrend = detrend

    def compute(self) -> SpectrumEstimate:
        r"""Estimate the density matrix.

        Forms the cross-periodogram on the Fourier grid and convolves it
        with the modified Daniell kernel around the circle.

        Returns:
            The :class:`SpectrumEstimate` on ``nobs // 2 + 1`` Fourier
            frequencies, with ``degrees_of_freedom = 16 m^2 / (4 m - 1)``
            and ``bandwidth = 2 pi m / nobs``.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> est = DaniellSpectrum(rng.standard_normal((256, 2)), span=1).compute()
            >>> est.method, est.frequencies.shape, round(est.degrees_of_freedom, 2)
            ('daniell', (129,), 5.33)
        """
        frequencies, density, nu = _daniell_density(self._panel, self._span)
        n = self._panel.shape[0]
        return SpectrumEstimate(
            names=self._names,
            frequencies=frequencies,
            density=density,
            method="daniell",
            degrees_of_freedom=nu,
            bandwidth=2.0 * np.pi * self._span / n,
            nobs=n,
            detrend=self._detrend,
        )


class WelchSpectrum:
    r"""Welch's averaged periodogram over half-overlapped Hann segments.

    The sample is cut into :math:`B` segments of :math:`L` observations
    starting every :math:`L/2`, each is demeaned, multiplied by the Hann
    taper :math:`h_t = \tfrac12 - \tfrac12\cos(2\pi t/L)`, and
    transformed on the segment's own Fourier grid; the estimate is the
    plain average

    .. math::

        \hat f(\omega_j) = \frac{1}{B}\sum_{b=1}^{B} I_b(\omega_j),
        \qquad \omega_j = \frac{2\pi j}{L},\ j = 0, \dots, \lfloor L/2 \rfloor,

    with :math:`\nu = 36B/19 \approx 1.9B` equivalent degrees of freedom,
    the Percival-Walden count for half-overlapped Hann segments, whose
    overlap makes adjacent periodograms correlated. The grid is the
    segment's, so the spacing is :math:`2\pi/L` and the number of
    ordinates :math:`L/2 + 1` regardless of the sample length; the
    default segment is an eighth of the sample rounded to even, at least
    32. Variance is bought by shortening segments, resolution by
    lengthening them, and only the sample length lets both improve.

    Attributes:
        _panel: The ``(nobs, k)`` panel after detrending.
        _names: The series labels.
        _detrend: The detrending choice.
        _segment: The segment length :math:`L`.

    Args:
        data: A ``(nobs, k)`` panel or one-dimensional series, at least
            32 observations.
        segment: Segment length :math:`L`, an integer from 8 to ``nobs``;
            ``None`` for the default. Even lengths put the last ordinate
            exactly at :math:`\pi`.
        detrend: ``"n"``, ``"c"`` (demean) or ``"ct"`` (linear detrend),
            removed from the whole sample first; each segment is then
            demeaned again.
        names: Series labels; ``None`` reads them from a frame or uses
            ``y1 .. yk`` (``y`` for a single series).

    Raises:
        SpecificationError: If the segment is not an integer of at least
            8 or exceeds the sample, the detrend choice is unknown, the
            labels do not match the columns, or the panel is too short.
        DimensionError: If the panel is malformed.
        NumericalError: If the panel is not finite.

    Note:
        The reported ``bandwidth`` is the grid spacing :math:`2\pi/L`,
        not the Hann taper's main lobe, which is twice as wide: two
        sinusoids two grid bins apart merge into one peak and three bins
        apart separate. Read peaks with that in mind. The per-segment
        Hann taper also costs about 3% of the variance (the taper's
        energy loss and the per-segment demeaning), so the integral of
        the spectrum falls slightly short of the sample variance where
        :class:`DaniellSpectrum` reproduces it exactly. Welch is the
        estimator for long samples where a coarse grid is acceptable and
        stationarity across segments is the question worth asking; for
        the sample lengths typical of macroeconomic data
        :class:`MultitaperSpectrum` keeps the full grid at the same
        degrees of freedom.

    See Also:
        * :class:`SpectrumEstimate` -- the result.
        * :class:`MultitaperSpectrum` -- full-grid resolution at the same
          variance, the default.
        * :class:`DaniellSpectrum` -- the smoothed periodogram.

    References:
        Welch, P. D. (1967). The use of fast Fourier transform for the
        estimation of power spectra: A method based on time averaging
        over short, modified periodograms. *IEEE Transactions on Audio
        and Electroacoustics*, 15(2), 70-73.

        Percival, D. B., & Walden, A. T. (1993). *Spectral Analysis for
        Physical Applications*, section 6.17. Cambridge University Press.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> est = WelchSpectrum(rng.standard_normal((800, 2)), segment=128).compute()
        >>> est.density.shape, bool(est.coherence("y1", "y2").mean() < 0.3)
        ((65, 2, 2), True)

        Eleven half-overlapped segments of 128 fit in 800 observations,
        giving :math:`36 \cdot 11 / 19` degrees of freedom on a grid of
        65 points spaced :math:`2\pi/128`:

        >>> round(est.degrees_of_freedom, 2), round(est.bandwidth, 4), est.nobs
        (20.84, 0.0491, 800)
        >>> WelchSpectrum(rng.standard_normal(4096)).compute().frequencies.shape
        (257,)
    """

    __slots__ = ("_detrend", "_names", "_panel", "_segment")

    def __init__(
        self,
        data: npt.ArrayLike,
        *,
        segment: int | None = None,
        detrend: str = "c",
        names: tuple[str, ...] | None = None,
    ) -> None:
        """Validate the panel and the segment length.

        Args:
            data: A ``(nobs, k)`` panel or one-dimensional series.
            segment: Segment length; ``None`` for ``max(32, nobs // 8)``
                rounded down to even.
            detrend: ``"n"``, ``"c"`` or ``"ct"``.
            names: Series labels, or ``None``.

        Raises:
            SpecificationError: If ``segment`` is not an integer of at
                least 8 or exceeds ``nobs``, or the panel, detrend choice
                or labels fail validation.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(1000)
            >>> WelchSpectrum(y)._segment, WelchSpectrum(y[:100])._segment
            (124, 32)
            >>> WelchSpectrum(y, segment=1001)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: segment 1001 exceeds the 1000 observations ...
            >>> WelchSpectrum(y, segment=7)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: segment must be >= 8; got 7.
        """
        panel, self._detrend, self._names = _validate_spectrum_panel(
            data, detrend, names, minimum=_MIN_SPECTRUM_OBS
        )
        self._panel = _ols_detrend(panel, self._detrend)
        n = self._panel.shape[0]
        default = max(32, (n // 8) - (n // 8) % 2)
        self._segment = validate_order(
            default if segment is None else segment, "segment", minimum=8
        )
        if self._segment > n:
            raise SpecificationError(
                f"segment {self._segment} exceeds the {n} observations available."
            )
        self._detrend = detrend

    def compute(self) -> SpectrumEstimate:
        r"""Estimate the density matrix.

        Walks the sample in steps of half a segment, demeans and
        Hann-tapers each block, and averages the blocks'
        cross-periodograms on the segment's Fourier grid.

        Returns:
            The :class:`SpectrumEstimate` on ``segment // 2 + 1``
            frequencies, with ``degrees_of_freedom = 36 B / 19`` for the
            ``B`` segments used and ``bandwidth = 2 pi / segment``.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> est = WelchSpectrum(rng.standard_normal((512, 2)), segment=64).compute()
            >>> est.method, est.frequencies.shape, round(est.degrees_of_freedom, 1)
            ('welch', (33,), 28.4)
        """
        frequencies, density, nu = _welch_density(self._panel, self._segment)
        return SpectrumEstimate(
            names=self._names,
            frequencies=frequencies,
            density=density,
            method="welch",
            degrees_of_freedom=nu,
            bandwidth=2.0 * np.pi / self._segment,
            nobs=self._panel.shape[0],
            detrend=self._detrend,
        )
