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

"""Nonparametric spectra: the data's own frequency-domain story, model-free.

:mod:`density` reads the spectrum off a fitted system and inherits every
specification error the fit carries. This module estimates it from the
data alone, on the same axes and with the same ``1 / (2 pi)``
normalization, so the two can be drawn on one plot and their disagreement
read as the model's misspecification at each frequency. The raw
periodogram is inconsistent -- its variance never shrinks -- so every
estimator here trades resolution for stability in a stated way. The
smoothed periodogram averages adjacent Fourier ordinates with a Daniell
kernel; Welch averages periodograms of overlapping tapered segments;
Thomson's multitaper averages periodograms of the same full sample under
orthogonal Slepian tapers, which is the least biased of the three at a
given variance and the default. Each result carries its equivalent
degrees of freedom, so a confidence band is one method call.

The result is a distinct type from :class:`SpectralDensityResult` on
purpose: a number from a periodogram and a number from a fitted model
answer different questions, and the package never lets one be mistaken
for the other.

References:
    Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
        *Proceedings of the IEEE*, 70(9), 1055-1096.
    Welch, P. D. (1967). The use of fast Fourier transform for the
        estimation of power spectra. *IEEE Transactions on Audio and
        Electroacoustics*, 15(2), 70-73.
    Percival, D. B., & Walden, A. T. (1993). *Spectral Analysis for
        Physical Applications*. Cambridge University Press.
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
    """A nonparametric spectral density matrix, with its resolution and degrees of freedom.

    Frequencies are radians per observation on ``[0, pi]`` and the
    density carries the ``1 / (2 pi)`` normalization, exactly as
    :class:`SpectralDensityResult`, so twice the integral of a spectrum
    over the grid is that series' sample variance and the two objects
    overlay. Every ordinate is approximately ``f(w) chi^2_nu / nu``, and
    :meth:`confidence_interval` uses that.

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
    """

    names: tuple[str, ...]
    frequencies: npt.NDArray[np.float64] = field(repr=False)
    density: npt.NDArray[np.complex128] = field(repr=False)
    method: str
    degrees_of_freedom: float
    bandwidth: float
    nobs: int
    detrend: str

    @property
    def k_endog(self) -> int:
        """Number of series."""
        return len(self.names)

    def _index(self, name: str) -> int:
        """Resolve a series label.

        Raises:
            SpecificationError: If the label is unknown.
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise SpecificationError(
                f"unknown series {name!r}; the estimate has {self.names}."
            ) from None

    def periods(self) -> npt.NDArray[np.float64]:
        """Each grid frequency as a period in observations per cycle, zero mapping to infinity."""
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
        """One series' estimated power spectrum, the real diagonal of the density."""
        i = self._index(name)
        return np.asarray(np.real(self.density[:, i, i]), dtype=np.float64)

    def coherence(self, first: str, second: str) -> npt.NDArray[np.float64]:
        """Squared coherence between two series at each frequency.

        ``|f_xy|^2 / (f_xx f_yy)``; with ``nu`` degrees of freedom the
        estimate is biased upward by about ``2 / nu`` under zero
        coherence, which the smoothing makes small.
        """
        i, j = self._index(first), self._index(second)
        cross = np.abs(self.density[:, i, j]) ** 2
        auto = np.real(self.density[:, i, i]) * np.real(self.density[:, j, j])
        return np.asarray(cross / np.maximum(auto, 1e-300), dtype=np.float64)

    def confidence_interval(
        self, name: str, *, alpha: float = 0.05
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Pointwise ``1 - alpha`` band for one spectrum from the chi-squared approximation.

        Raises:
            SpecificationError: If ``alpha`` is not inside ``(0, 1)``.
        """
        if not 0.0 < alpha < 1.0:
            raise SpecificationError(f"alpha must lie strictly inside (0, 1); got {alpha}.")
        power = self.spectrum(name)
        nu = self.degrees_of_freedom
        lower = power * nu / float(chi2.ppf(1.0 - alpha / 2.0, nu))
        upper = power * nu / float(chi2.ppf(alpha / 2.0, nu))
        return lower, upper

    def _summary_table(self) -> SummaryTable:
        """Peak and variance per series, with the estimate's resolution."""
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
    """Thomson's multitaper spectral estimate.

    ``K`` Slepian tapers of time-bandwidth product ``NW`` each give a
    leakage-free periodogram; their concentration-weighted average has
    ``2 K`` degrees of freedom and resolution ``2 pi NW / N``. The usual
    ``NW = 4, K = 7`` is the default; raise both for a smoother
    estimate, lower both to separate close peaks.

    Args:
        data: A ``(nobs, k)`` panel or one-dimensional series.
        bandwidth: Time-bandwidth product ``NW``.
        n_tapers: Tapers ``K``; ``None`` takes ``2 NW - 1``.
        detrend: ``"n"``, ``"c"`` (demean) or ``"ct"`` (linear detrend).
        names: Series labels; ``None`` reads them from a frame or uses
            ``y1 .. yk``.

    Raises:
        SpecificationError: If the bandwidth is below 1, the taper count
            is unusable, or the panel is too short.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(400)
        >>> for t in range(1, 400):
        ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal()
        >>> est = MultitaperSpectrum(y).compute()
        >>> est.density.shape, bool(est.spectrum("y")[1] > est.spectrum("y")[-1])
        ((201, 1, 1), True)
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
        """Validate the panel and the taper design."""
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
        """Estimate the density matrix."""
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
    """The periodogram smoothed with a modified Daniell kernel.

    Adjacent Fourier ordinates are averaged with a flat kernel of
    half-width ``span`` whose end weights are halved; ``2 span + 1``
    ordinates enter each estimate. Default ``span`` is ``sqrt(N) / 2``,
    the usual compromise between resolution and variance.

    Args:
        data: A ``(nobs, k)`` panel or one-dimensional series.
        span: Kernel half-width in ordinates; ``None`` for the default.
        detrend: ``"n"``, ``"c"`` (demean) or ``"ct"`` (linear detrend).
        names: Series labels.

    Raises:
        SpecificationError: If the span is below 1 or the panel is too
            short.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> est = DaniellSpectrum(rng.standard_normal(400)).compute()
        >>> bool(abs(2.0 * np.pi * est.spectrum("y").mean() - 1.0) < 0.2)
        True
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
        """Validate the panel and the span."""
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
        """Estimate the density matrix."""
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
    """Welch's averaged periodogram over half-overlapped Hann segments.

    The sample is cut into segments of ``segment`` observations
    overlapping by half, each is demeaned, Hann-tapered and
    transformed, and the periodograms are averaged. The grid is the
    segment's, so resolution is ``2 pi / segment``; the default segment
    is an eighth of the sample, at least 32.

    Args:
        data: A ``(nobs, k)`` panel or one-dimensional series.
        segment: Segment length; ``None`` for the default.
        detrend: ``"n"``, ``"c"`` (demean) or ``"ct"`` (linear detrend).
        names: Series labels.

    Raises:
        SpecificationError: If the segment is shorter than 8 or longer
            than the sample, or the panel is too short.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> est = WelchSpectrum(rng.standard_normal((800, 2)), segment=128).compute()
        >>> est.density.shape, bool(est.coherence("y1", "y2").mean() < 0.3)
        ((65, 2, 2), True)
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
        """Validate the panel and the segment length."""
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
        """Estimate the density matrix."""
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
