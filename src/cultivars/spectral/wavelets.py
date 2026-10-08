# filepath: /src/cultivars/spectral/wavelets.py
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
r"""Wavelet analysis: the MODWT multiresolution decomposition and Morlet wavelet coherence.

A spectrum answers "at which frequencies" and a filter "what is the
cycle"; a wavelet decomposition answers both at once, by frequency band
and by date. The maximal-overlap discrete wavelet transform of Percival
and Walden filters a series through a pyramid of upsampled wavelet and
scaling filters,

.. math::

   W_{j,t} = \sum_{l} \tilde h_l\, V_{j-1,\,t - 2^{j-1} l},
   \qquad
   y_t = \sum_{j=1}^{J} D_{j,t} + S_{J,t},

   \qquad
   \hat\nu^2_j = \frac{1}{M_j}\sum_{t \ge L_j - 1} W_{j,t}^2,

into details at dyadic period bands :math:`[2^j, 2^{j+1})` plus a
smooth, additively, aligned in time, and defined for any sample length;
the wavelet variance :math:`\hat\nu^2_j` over the boundary-free
coefficients is an analysis of variance by scale with a chi-squared
band, the usual way to ask which horizon carries a series'
variability. Wavelet coherence, on Torrence and Compo's continuous
Morlet transform, is the smoothed local squared coherence of two series

.. math::

   R^2(s, t) = \frac{|S(W_{xy}/s)|^2}{S(|W_x|^2/s)\, S(|W_y|^2/s)}

across scale :math:`s` and date :math:`t`, with the phase of the
cross-wavelet reading which series leads; it is the instrument behind
"co-movement of output and inflation at business-cycle frequencies
broke down after 1985" and its like.

Two commitments shape the surface. First, both objects are descriptive
and say so: the cone of influence marks where the transform's edge
padding is felt, the AR(1) surrogate level says how much coherence two
independent persistent series produce by chance, and the MODWT's
degrees of freedom shrink by half per level so the deepest scales
declare themselves uninformative -- the results state where the picture
can be trusted, not that a model holds. Second, the discrete and
continuous transforms are kept to the questions each answers: the MODWT
is additive and orthogonal-by-scale, so it decomposes one series;
Morlet coherence is redundant and smooth, so it compares two. Neither
is offered for the other's job.

Layout. :class:`MODWT` takes a series through ``transform`` into
:class:`MODWTResult`; :class:`WaveletCoherence` takes two through
``compute`` into :class:`WaveletCoherenceResult`. The numerics live in
``_core``: ``_modwt_filters`` holds the Haar, D4 and LA8 filters
rescaled for the MODWT, ``_modwt`` and ``_modwt_inverse`` run the
pyramid and its inverse, ``_modwt_details`` the multiresolution
analysis and ``_modwt_variance`` the unbiased variance with
Percival-Walden's :math:`\eta_3` degrees of freedom;
``_morlet_transform`` and ``_morlet_scales`` give the continuous
transform on its dyadic grid, ``_wavelet_smooth`` the Torrence-Webster
smoothing, ``_wavelet_coherence`` the coherence and phase,
``_cone_of_influence`` the edge mask, and ``_coherence_surrogates`` the
AR(1) significance level; ``_MORLET_OMEGA0``, ``_SCALES_PER_OCTAVE``,
``_COHERENCE_SURROGATES`` and ``_MIN_SPECTRUM_OBS`` are the defaults.
``_SummaryMixin`` from ``_internals`` gives both results their printed
form. The Fourier-domain counterparts -- a spectrum without a date axis
-- are :mod:`~cultivars.spectral.periodogram` and
:mod:`~cultivars.spectral.density`; the band-pass filters that isolate
a stated band rather than octaves are in
:mod:`~cultivars.spectral.band_pass`.

References:
    Percival, D. B., & Walden, A. T. (2000). *Wavelet Methods for Time
    Series Analysis*. Cambridge University Press.

    Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet
    analysis. *Bulletin of the American Meteorological Society*, 79(1),
    61-78.

    Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004). Application of
    the cross wavelet transform and wavelet coherence to geophysical
    time series. *Nonlinear Processes in Geophysics*, 11, 561-566.

    Gencay, R., Selcuk, F., & Whitcher, B. (2002). *An Introduction to
    Wavelets and Other Filtering Methods in Finance and Economics*.
    Academic Press.

Example:
    A common period-24 cycle that switches off halfway through the
    sample: coherence is significant in that band almost everywhere in
    the first half and almost nowhere in the second, which is the
    time-frequency statement no spectrum can make, and the MODWT of one
    series puts the cycle's variance in the level whose band holds it:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(512.0)
    >>> common = np.sin(2 * np.pi * t / 24) * (t < 256)
    >>> x = common + 0.5 * rng.standard_normal(512)
    >>> y = np.roll(common, 4) + 0.5 * rng.standard_normal(512)
    >>> res = WaveletCoherence(per_octave=4, surrogates=100, seed=0).compute(x, y)
    >>> hit = res.significant()[(res.periods > 18) & (res.periods < 32)]
    >>> round(float(hit[:, 40:216].mean()), 1), round(float(hit[:, 296:472].mean()), 1)
    (0.9, 0.1)
    >>> modwt = MODWT(levels=5).transform(x)
    >>> int(np.argmax(modwt.variance)) + 1, modwt.periods[3]
    (4, (16.0, 32.0))
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from .._core import (
    _COHERENCE_SURROGATES,
    _DEFAULT_ALPHA,
    _MIN_SPECTRUM_OBS,
    _MORLET_OMEGA0,
    _SCALES_PER_OCTAVE,
    SummaryTable,
    _coherence_surrogates,
    _cone_of_influence,
    _modwt,
    _modwt_details,
    _modwt_filters,
    _modwt_variance,
    _morlet_scales,
    _wavelet_coherence,
    validate_aligned,
    validate_choice,
    validate_endog,
    validate_open_interval,
    validate_order,
)
from .._internals import _SummaryMixin
from ..exceptions import SpecificationError

__all__ = ["MODWT", "MODWTResult", "WaveletCoherence", "WaveletCoherenceResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MODWTResult(_SummaryMixin):
    r"""A MODWT decomposition: coefficients, additive components, and variance by scale.

    The maximal-overlap transform keeps one coefficient per observation
    at every level, so level :math:`j` is a zero-phase band-pass of the
    series onto periods :math:`2^j` to :math:`2^{j+1}` observations and
    the scaling coefficients :math:`V_J` a low-pass onto everything
    longer. Two readings of the same coefficients are carried. The
    *multiresolution analysis* inverts each level alone to give details
    :math:`D_j` and a smooth :math:`S_J` that add back to the data
    exactly,

    .. math::

        y_t = \sum_{j=1}^{J} D_{j,t} + S_{J,t},

    each aligned in time with the series. The *wavelet variance*
    :math:`\hat\nu^2_j` is the mean square of the level-:math:`j`
    coefficients not touched by the periodic boundary, an unbiased
    estimate of the variance the series carries at that scale, with

    .. math::

        \sum_{j=1}^{J} \nu^2_j + \operatorname{var}(S_J) \approx \sigma^2_y,

    the analysis of variance by scale that the summary prints as shares.
    Each :math:`\hat\nu^2_j` is approximately
    :math:`\nu^2_j \chi^2_{\eta_j}/\eta_j` with :math:`\eta_j` the
    equivalent degrees of freedom, which :meth:`variance_interval`
    inverts.

    Attributes:
        coefficients: ``(J, T)`` wavelet coefficients ``W_j``, level by
            level; level ``j`` responds to periods of ``2^j`` to
            ``2^(j+1)`` observations.
        scaling: ``(T,)`` level-``J`` scaling coefficients ``V_J``.
        details: ``(J, T)`` multiresolution details ``D_j``; with
            ``smooth`` they sum to the data exactly.
        smooth: ``(T,)`` multiresolution smooth ``S_J``, the part of the
            series at periods beyond ``2^(J+1)``.
        variance: ``(J,)`` unbiased wavelet variance per level; the
            levels' sum plus the smooth's variance is the series variance.
        degrees_of_freedom: ``(J,)`` equivalent chi-squared degrees of
            freedom behind :meth:`variance_interval`.
        wavelet: The filter used.
        nobs: Observations.

    Note:
        The transform is circular: the first :math:`(2^j - 1)(L - 1)`
        coefficients at level :math:`j` (with :math:`L` the filter
        length) wrap around to the end of the sample, and the details
        near either end inherit that. The variance excludes those
        coefficients, which is why its sum falls a little short of the
        series variance, and why the degrees of freedom shrink by
        roughly half per level: at the deepest level a 256-observation
        series under ``la8`` leaves a single degree of freedom and a
        band that spans two orders of magnitude. The
        Percival-Walden :math:`\eta_3` count used here is the
        conservative one and assumes the level is stationary; a
        deterministic trend loads on the smooth, not on any detail.

    See Also:
        * :class:`MODWT` -- the producer.
        * :class:`WaveletCoherenceResult` -- the bivariate, continuous
          wavelet counterpart.
        * :class:`~cultivars.spectral.band_pass.BandPassResult` -- the
          Fourier band-pass on a user-chosen band, where the wavelet's
          bands are fixed at octaves.

    References:
        Percival, D. B., & Walden, A. T. (2000). *Wavelet Methods for
        Time Series Analysis*, chapters 5 and 8. Cambridge University
        Press.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = rng.standard_normal(256)
        >>> res = MODWT(wavelet="haar", levels=3).transform(y)
        >>> res.levels, res.periods
        (3, ((2.0, 4.0), (4.0, 8.0), (8.0, 16.0)))
        >>> bool(np.allclose(res.details.sum(axis=0) + res.smooth, y))
        True

        White noise puts half its variance in each successive octave,
        and the Haar level-1 coefficient is half the first difference:

        >>> res.variance.round(2).tolist()
        [0.49, 0.25, 0.14]
        >>> bool(np.allclose(res.coefficients[0, 1:], np.diff(y) / 2))
        True

        A sinusoid of period 12 lands in the level whose band holds it:

        >>> t = np.arange(1024.0)
        >>> wave = MODWT(levels=5).transform(np.sin(2 * np.pi * t / 12))
        >>> int(np.argmax(wave.variance)) + 1, wave.periods[int(np.argmax(wave.variance))]
        (3, (8.0, 16.0))
    """

    coefficients: npt.NDArray[np.float64] = field(repr=False)
    """``(J, T)`` wavelet coefficients by level. Kept out of the repr."""

    scaling: npt.NDArray[np.float64] = field(repr=False)
    """``(T,)`` level-``J`` scaling coefficients. Kept out of the repr."""

    details: npt.NDArray[np.float64] = field(repr=False)
    """``(J, T)`` multiresolution details. Kept out of the repr."""

    smooth: npt.NDArray[np.float64] = field(repr=False)
    """``(T,)`` multiresolution smooth. Kept out of the repr."""

    variance: npt.NDArray[np.float64]
    """``(J,)`` unbiased wavelet variance per level; ``nan`` where no interior coefficient."""

    degrees_of_freedom: npt.NDArray[np.float64] = field(repr=False)
    """``(J,)`` equivalent chi-squared degrees of freedom. Kept out of the repr."""

    wavelet: str
    """``"haar"``, ``"d4"`` or ``"la8"``."""

    nobs: int
    """Observations."""

    @property
    def levels(self) -> int:
        """Number of detail levels ``J``.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> MODWT(wavelet="d4", levels=4).transform(rng.standard_normal(128)).levels
            4
        """
        return int(self.coefficients.shape[0])

    @property
    def scales(self) -> npt.NDArray[np.float64]:
        r"""Physical scale ``2^(j-1)`` of each level, in observations.

        The scale :math:`\tau_j = 2^{j-1}` is the width of the averages
        the level-:math:`j` wavelet filter differences; the period band
        it responds to is :math:`[2\tau_j, 4\tau_j)`, which
        :attr:`periods` gives directly.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> MODWT(levels=4).transform(rng.standard_normal(256)).scales.tolist()
            [1.0, 2.0, 4.0, 8.0]
        """
        return np.asarray(2.0 ** np.arange(self.levels), dtype=np.float64)

    @property
    def periods(self) -> tuple[tuple[float, float], ...]:
        """The band of periods, in observations, each level responds to.

        Level :math:`j` covers :math:`[2^j, 2^{j+1})`; a sinusoid whose
        period sits on a band edge splits evenly between the two
        adjacent levels.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> MODWT(levels=2).transform(rng.standard_normal(64)).periods
            ((2.0, 4.0), (4.0, 8.0))
        """
        return tuple((float(2.0 ** (j + 1)), float(2.0 ** (j + 2))) for j in range(self.levels))

    def variance_interval(
        self, alpha: float = _DEFAULT_ALPHA
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Chi-squared confidence band for the wavelet variance at each level.

        From :math:`\eta_j \hat\nu^2_j / \nu^2_j \sim \chi^2_{\eta_j}`,

        .. math::

            \Bigl[\frac{\eta_j \hat\nu^2_j}{\chi^2_{\eta_j, 1-\alpha/2}},\;
            \frac{\eta_j \hat\nu^2_j}{\chi^2_{\eta_j, \alpha/2}}\Bigr].

        Args:
            alpha: Two-sided miscoverage.

        Returns:
            ``(lower, upper)`` each ``(J,)``, from ``nu v / chi2`` quantiles
            with the level's equivalent degrees of freedom.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> res = MODWT(wavelet="haar", levels=3).transform(rng.standard_normal(256))
            >>> lower, upper = res.variance_interval()
            >>> bool(np.all(lower < res.variance) and np.all(res.variance < upper))
            True
            >>> (upper / lower).round(2).tolist()
            [1.64, 2.02, 2.74]
        """
        nu = self.degrees_of_freedom
        lower = nu * self.variance / chi2.ppf(1.0 - alpha / 2.0, nu)
        upper = nu * self.variance / chi2.ppf(alpha / 2.0, nu)
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    def _summary_table(self) -> SummaryTable:
        """Variance by scale with its band and share.

        One row per level: period band, wavelet variance, its 95% band,
        its share of the total (levels' variances plus the smooth's),
        and the degrees of freedom; a closing row for the smooth. The
        notes state the circular boundary treatment, the conservatism
        of the bands, and the exact additivity of the components.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> table = MODWT(wavelet="haar", levels=3).transform(rng.standard_normal(256))
            >>> table = table._summary_table()
            >>> table.columns
            ('level', 'period band', 'variance', '95% band', 'share', 'dof')
            >>> table.rows[0]
            ('D1', '2-4', '0.4935', '[0.392, 0.641]', '50.3%', '128')
            >>> table.rows[-1][:2]
            ('S3', '>16')
        """
        lower, upper = self.variance_interval()
        total = float(np.nansum(self.variance)) + float(self.smooth.var())
        rows = []
        for j in range(self.levels):
            low, high = self.periods[j]
            share = self.variance[j] / total if total > 0.0 else np.nan
            rows.append(
                (
                    f"D{j + 1}",
                    f"{low:.0f}-{high:.0f}",
                    f"{self.variance[j]:.4g}",
                    f"[{lower[j]:.3g}, {upper[j]:.3g}]",
                    f"{100 * share:.1f}%",
                    f"{self.degrees_of_freedom[j]:.0f}",
                )
            )
        rows.append(
            (
                f"S{self.levels}",
                f">{self.periods[-1][1]:.0f}",
                f"{self.smooth.var():.4g}",
                "",
                f"{100 * self.smooth.var() / total:.1f}%" if total > 0.0 else "",
                "",
            )
        )
        notes = (
            "Periodic boundary treatment: the first (2^j - 1)(L - 1) coefficients at level j "
            "wrap the sample end and are excluded from the variance; details near either end "
            "carry that contamination.",
            "Bands are chi-squared with Percival-Walden's eta_3 degrees of freedom, which is "
            "conservative (over-covers) and assumes stationarity within the level; a trend "
            "loads on the smooth.",
            "Details and smooth sum to the data exactly and are aligned in time (zero phase).",
        )
        return SummaryTable(
            title="MODWT Decomposition",
            metadata=(
                ("Wavelet", self.wavelet),
                ("Levels", str(self.levels)),
                ("Observations", str(self.nobs)),
                ("Series variance", f"{total:.4g}"),
            ),
            columns=("level", "period band", "variance", "95% band", "share", "dof"),
            rows=tuple(rows),
            notes=notes,
        )


class MODWT:
    r"""Maximal-overlap discrete wavelet transform and multiresolution analysis.

    The pyramid filters the level-:math:`(j-1)` scaling coefficients
    with the wavelet and scaling filters upsampled by :math:`2^{j-1}`,

    .. math::

        W_{j,t} = \sum_{l=0}^{L-1} \tilde h_l\, V_{j-1,\,t - 2^{j-1} l \bmod T},
        \qquad
        V_{j,t} = \sum_{l=0}^{L-1} \tilde g_l\, V_{j-1,\,t - 2^{j-1} l \bmod T},

    starting from :math:`V_0 = y`, with the DWT filters rescaled by
    :math:`1/\sqrt 2` so that energy is preserved without decimation.
    Because nothing is decimated the transform is shift-invariant, every
    level has :math:`T` coefficients aligned with the data, and the
    multiresolution details are zero-phase -- the properties that make
    it the wavelet transform for time series rather than for
    compression. The cost is redundancy (:math:`J + 1` series of length
    :math:`T`) and the circular boundary, which touches
    :math:`(2^j - 1)(L - 1)` coefficients at level :math:`j`; the
    default depth is the largest :math:`J` at which that count still
    leaves a boundary-free coefficient, :math:`\lfloor \log_2(T/(L-1))
    \rfloor`.

    Attributes:
        _wavelet: The validated filter name.
        _levels: The requested depth, or ``None`` for the default.

    Args:
        wavelet: ``"haar"`` (2 taps, the first difference at level 1),
            ``"d4"`` (Daubechies extremal phase, 4 taps) or ``"la8"``
            (least asymmetric, 8 taps; the default -- near-zero phase
            and a sharper band split than Haar).
        levels: Detail levels ``J``, at least 1. By default the largest
            ``J`` for which every level keeps a boundary-free
            coefficient, ``floor(log2(T / (L - 1)))``.

    Raises:
        SpecificationError: If the wavelet is unknown or ``levels`` is
            not an integer of at least 1.

    Note:
        Shorter filters have less boundary contamination and a blurrier
        band split; ``haar`` at level 1 is exactly half the first
        difference, so its "band" leaks across the whole spectrum,
        while ``la8`` confines a period-12 sinusoid to level 3 with 98%
        of its energy. Deeper levels have fewer degrees of freedom by a
        factor of two each, so the default depth is an upper bound on
        what is estimable, not a recommendation.

    See Also:
        * :class:`MODWTResult` -- the result.
        * :class:`WaveletCoherence` -- the continuous Morlet transform
          for time-frequency comovement of two series.
        * :class:`~cultivars.spectral.band_pass.BaxterKingFilter` -- a
          band-pass on any stated band, when octaves are the wrong
          partition.

    References:
        Percival, D. B., & Walden, A. T. (2000). *Wavelet Methods for
        Time Series Analysis*, chapter 5. Cambridge University Press.

        Gencay, R., Selcuk, F., & Whitcher, B. (2002). *An Introduction
        to Wavelets and Other Filtering Methods in Finance and
        Economics*. Academic Press.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = rng.standard_normal(256)
        >>> res = MODWT(wavelet="haar", levels=3).transform(y)
        >>> res.levels, bool(np.allclose(res.details.sum(axis=0) + res.smooth, y))
        (3, True)

        The default depth depends on the filter length: 256 observations
        support 8 Haar levels, 6 of ``d4`` and 5 of ``la8``:

        >>> [MODWT(wavelet=w).transform(y).levels for w in ("haar", "d4", "la8")]
        [8, 6, 5]
    """

    __slots__ = ("_levels", "_wavelet")

    def __init__(self, *, wavelet: str = "la8", levels: int | None = None) -> None:
        """Validate the filter and depth.

        Args:
            wavelet: ``"haar"``, ``"d4"`` or ``"la8"``.
            levels: Detail levels, or ``None`` for the default.

        Raises:
            SpecificationError: If the wavelet is unknown or ``levels``
                is not an integer of at least 1.

        Example:
            >>> MODWT(wavelet="d4", levels=3)._wavelet
            'd4'
            >>> MODWT(wavelet="sym8")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: wavelet must be one of ... got 'sym8'.
            >>> MODWT(levels=0)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: levels must be >= 1; got 0.
        """
        self._wavelet = validate_choice(wavelet, ("haar", "d4", "la8"), "wavelet")
        self._levels = None if levels is None else validate_order(levels, "levels", minimum=1)

    def transform(self, endog: npt.ArrayLike) -> MODWTResult:
        """Decompose a series.

        Runs the pyramid to the requested depth, inverts each level for
        the multiresolution components, and computes the unbiased
        wavelet variance with its degrees of freedom.

        Args:
            endog: The ``(T,)`` series, at least 32 observations.

        Returns:
            The :class:`MODWTResult`.

        Raises:
            DimensionError: If the series is not one-dimensional.
            SpecificationError: If the series is shorter than 32, or
                shorter than the requested depth allows.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(256)
            >>> res = MODWT(levels=4).transform(y)
            >>> res.coefficients.shape, res.scaling.shape, res.wavelet, res.nobs
            ((4, 256), (256,), 'la8', 256)
            >>> MODWT(levels=6).transform(y)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: 6 levels of the la8 wavelet exceed what 256 ...
        """
        y = validate_endog(endog)
        n = y.shape[0]
        if n < _MIN_SPECTRUM_OBS:
            raise SpecificationError(
                f"a wavelet decomposition needs at least {_MIN_SPECTRUM_OBS} observations; got {n}."
            )
        taps = _modwt_filters(self._wavelet)[0].shape[0]
        deepest = max(int(np.floor(np.log2(n / (taps - 1)))), 1)
        levels = deepest if self._levels is None else self._levels
        if levels > deepest:
            raise SpecificationError(
                f"{levels} levels of the {self._wavelet} wavelet exceed what {n} observations "
                f"support without every coefficient touching the boundary (at most {deepest})."
            )
        coefficients, scaling = _modwt(y, self._wavelet, levels)
        details, smooth = _modwt_details(coefficients, scaling, self._wavelet)
        variance, degrees = _modwt_variance(coefficients, self._wavelet)
        return MODWTResult(
            coefficients=coefficients,
            scaling=scaling,
            details=details,
            smooth=smooth,
            variance=variance,
            degrees_of_freedom=degrees,
            wavelet=self._wavelet,
            nobs=n,
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class WaveletCoherenceResult(_SummaryMixin):
    r"""Squared wavelet coherence and phase of two series over scale and time.

    With :math:`W_x(s, t)` and :math:`W_y(s, t)` the Morlet transforms
    and :math:`W_{xy} = W_x \overline{W_y}` the cross-wavelet, the
    squared coherence is the smoothed, normalized cross-power

    .. math::

        R^2(s, t) = \frac{\bigl|S\bigl(W_{xy}(s, t)/s\bigr)\bigr|^2}
                         {S\bigl(|W_x(s, t)|^2/s\bigr)\,
                          S\bigl(|W_y(s, t)|^2/s\bigr)},

    a local :math:`R^2` between the two series at scale :math:`s` and
    date :math:`t`, and the phase difference
    :math:`\phi(s, t) = \arg S(W_{xy})` their local lead-lag, positive
    when the first series leads. :math:`S` is the Torrence-Webster
    smoothing (Gaussian in time with width :math:`s`, boxcar over 0.6
    octave in scale) without which :math:`R^2 \equiv 1`. The scale
    axis is converted to Fourier periods by
    :math:`\lambda = 4\pi s / (\omega_0 + \sqrt{2 + \omega_0^2})`, 1.03
    at :math:`\omega_0 = 6`; the cone of influence marks coefficients
    within :math:`\sqrt 2\, s` of either edge, where the zero padding
    behind the transform is felt; and the significance level is the
    :math:`1 - \alpha` quantile of the coherence of independent AR(1)
    surrogates, scale by scale, because smoothed coherence is high in
    places by chance and must be read against that, not against zero.

    Attributes:
        coherence: ``(S, T)`` squared coherence in ``[0, 1]``.
        phase: ``(S, T)`` phase difference in radians, ``arg(W_xy)``;
            positive means the first series leads at that scale.
        scales: ``(S,)`` wavelet scales, in observations.
        periods: ``(S,)`` equivalent Fourier periods.
        cone: ``(S, T)`` flags, ``True`` where the coefficient lies inside
            the cone of influence and edge padding contaminates it.
        significance: ``(S,)`` per-scale coherence level exceeded with
            probability ``alpha`` by independent AR(1) surrogates, or
            ``None`` when no surrogates were run.
        alpha: Level behind ``significance``.
        omega0: Morlet centre frequency.
        nobs: Observations.

    Note:
        Phase is meaningful only where coherence is: the lead-lag of two
        series that do not comove at a scale is the argument of a noisy
        complex number, and the summary's ``lead`` column is printed for
        every octave regardless. Read it alongside the ``share
        significant`` column, and prefer :meth:`lead` over a band you
        have already seen to be coherent. The largest scales reach the
        sample length, so their cone covers every date and their
        significance is ``nan``; :meth:`significant` is ``False`` there
        and :meth:`lead` returns ``nan`` for a band with no date outside
        the cone.

    See Also:
        * :class:`WaveletCoherence` -- the producer.
        * :class:`~cultivars.spectral.density.SpectralDensityResult` --
          coherence and phase by frequency alone, from a fitted model,
          when the relation is stable over time.
        * :class:`MODWTResult` -- the univariate discrete decomposition
          by octave.

    References:
        Torrence, C., & Compo, G. P. (1998). A practical guide to
        wavelet analysis. *Bulletin of the American Meteorological
        Society*, 79(1), 61-78.

        Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004). Application
        of the cross wavelet transform and wavelet coherence to
        geophysical time series. *Nonlinear Processes in Geophysics*,
        11, 561-566.

    Example:
        Two noisy copies of a period-24 sinusoid, the second delayed
        by three observations: coherence is significant in the band
        holding the signal and not at short periods, and the lead is
        about three:

        >>> rng = np.random.default_rng(0)
        >>> t = np.arange(256.0)
        >>> signal = np.sin(2 * np.pi * t / 24)
        >>> x = signal + 0.5 * rng.standard_normal(256)
        >>> y = np.roll(signal, 3) + 0.5 * rng.standard_normal(256)
        >>> res = WaveletCoherence(per_octave=4, surrogates=100, seed=1).compute(x, y)
        >>> res.coherence.shape, res.scales.shape, res.nobs
        ((29, 256), (29,), 256)
        >>> hit = res.significant()
        >>> band = (res.periods > 20) & (res.periods < 30)
        >>> bool(hit[band].mean() > 0.5), bool(hit[res.periods < 8].mean() < 0.15)
        (True, True)
        >>> bool(2.0 < res.lead((20.0, 30.0)) < 4.5)
        True
    """

    coherence: npt.NDArray[np.float64] = field(repr=False)
    """``(S, T)`` squared coherence. Kept out of the repr."""

    phase: npt.NDArray[np.float64] = field(repr=False)
    """``(S, T)`` phase difference in radians. Kept out of the repr."""

    scales: npt.NDArray[np.float64] = field(repr=False)
    """``(S,)`` wavelet scales in observations. Kept out of the repr."""

    periods: npt.NDArray[np.float64] = field(repr=False)
    """``(S,)`` equivalent Fourier periods. Kept out of the repr."""

    cone: npt.NDArray[np.bool_] = field(repr=False)
    """``(S, T)`` cone-of-influence flags. Kept out of the repr."""

    significance: npt.NDArray[np.float64] | None = field(repr=False)
    """``(S,)`` surrogate significance level, or ``None``. Kept out of the repr."""

    alpha: float
    """Level behind ``significance``."""

    omega0: float
    """Morlet centre frequency."""

    nobs: int
    """Observations."""

    def significant(self) -> npt.NDArray[np.bool_]:
        """Where coherence exceeds the surrogate level, outside the cone of influence.

        Returns:
            ``(S, T)`` flags; ``False`` throughout a scale whose level is
            ``nan`` (cone everywhere) and inside the cone at any scale.

        Raises:
            SpecificationError: If no surrogates were run.

        Example:
            Independent white noise is significant at about ``alpha`` of
            the dates outside the cone:

            >>> rng = np.random.default_rng(0)
            >>> a, b = rng.standard_normal(256), rng.standard_normal(256)
            >>> res = WaveletCoherence(per_octave=4, surrogates=100, seed=0).compute(a, b)
            >>> hit = res.significant()
            >>> bool(0.01 < hit[~res.cone].mean() < 0.1)
            True
            >>> WaveletCoherence(surrogates=0).compute(a, b).significant()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: no significance level: rerun ...
        """
        if self.significance is None:
            raise SpecificationError(
                "no significance level: rerun WaveletCoherence with surrogates > 0."
            )
        return np.asarray(
            (self.coherence > self.significance[:, None]) & ~self.cone, dtype=np.bool_
        )

    def lead(self, scale_band: tuple[float, float]) -> float:
        r"""Mean lead of the first series, in observations, over a band of periods.

        The phase difference at each date in the band is converted to
        time as :math:`\phi\,\lambda / 2\pi` and averaged over dates
        outside the cone of influence and scales whose period lies in
        ``scale_band``.

        Args:
            scale_band: ``(low, high)`` periods, inclusive.

        Returns:
            Positive when the first series leads, negative when it lags;
            ``nan`` if every date in the band is inside the cone.

        Raises:
            SpecificationError: If no scale falls in the band.

        Example:
            A sinusoid and its two-observation delay, both ways round:

            >>> t = np.arange(256.0)
            >>> x = np.sin(2 * np.pi * t / 16)
            >>> y = np.sin(2 * np.pi * (t - 2) / 16)
            >>> wc = WaveletCoherence(per_octave=4, surrogates=0)
            >>> round(wc.compute(x, y).lead((14.0, 18.0)), 1)
            2.1
            >>> round(wc.compute(y, x).lead((14.0, 18.0)), 1)
            -2.1
            >>> wc.compute(x, y).lead((1000.0, 2000.0))
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: no scale has a period in [1000.0, 2000.0].
        """
        low, high = scale_band
        rows = (self.periods >= low) & (self.periods <= high)
        if not np.any(rows):
            raise SpecificationError(f"no scale has a period in [{low}, {high}].")
        lag = self.phase[rows] * self.periods[rows][:, None] / (2.0 * np.pi)
        keep = ~self.cone[rows]
        return float(np.mean(lag[keep])) if np.any(keep) else np.nan

    def _summary_table(self) -> SummaryTable:
        """Mean coherence and lead per octave, outside the cone of influence.

        One row per octave of period (2-4, 4-8, ...): mean squared
        coherence, mean lead in observations, and -- when surrogates
        were run -- the share of dates above the significance level.
        Octaves with no date outside the cone are omitted.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> t = np.arange(256.0)
            >>> x = np.sin(2 * np.pi * t / 16)
            >>> y = np.sin(2 * np.pi * (t - 2) / 16)
            >>> res = WaveletCoherence(per_octave=4, surrogates=0).compute(x, y)
            >>> table = res._summary_table()
            >>> table.columns
            ('period band', 'mean coherence', 'lead', 'share significant')
            >>> [row[0] for row in table.rows]
            ['2-4', '4-8', '8-16', '16-32', '32-64', '64-128']
            >>> table.notes[-1]
            'No surrogate significance level was computed (surrogates=0).'
        """
        rows = []
        octaves = np.floor(np.log2(self.periods)).astype(int)
        for octave in np.unique(octaves):
            band = octaves == octave
            keep = ~self.cone[band]
            if not np.any(keep):
                continue
            mean_coherence = float(np.mean(self.coherence[band][keep]))
            lag = self.phase[band] * self.periods[band][:, None] / (2.0 * np.pi)
            lead = float(np.mean(lag[keep]))
            if self.significance is None:
                share = ""
            else:
                above = self.coherence[band] > self.significance[band][:, None]
                share = f"{100 * float(np.mean(above[keep])):.0f}%"
            rows.append(
                (
                    f"{2**octave:.0f}-{2 ** (octave + 1):.0f}",
                    f"{mean_coherence:.3f}",
                    f"{lead:+.2f}",
                    share,
                )
            )
        notes = [
            "Coherence is Torrence-Webster smoothed (Gaussian in time, 0.6 octave in scale) "
            "and is high in places by chance even for independent series; read it against "
            "the significance level, not against zero.",
            "Lead is the phase difference converted to observations at that period, averaged "
            "outside the cone of influence; positive when the first series leads.",
        ]
        if self.significance is None:
            notes.append("No surrogate significance level was computed (surrogates=0).")
        else:
            notes.append(
                f"Significance at {100 * (1 - self.alpha):.0f}% from independent AR(1) "
                "surrogates matched to each series' lag-one autocorrelation; the last column "
                "is the share of dates above it."
            )
        return SummaryTable(
            title="Wavelet Coherence",
            metadata=(
                ("Morlet omega0", f"{self.omega0:g}"),
                ("Scales", str(int(self.scales.shape[0]))),
                ("Period range", f"{self.periods[0]:.1f}-{self.periods[-1]:.1f}"),
                ("Observations", str(self.nobs)),
            ),
            columns=("period band", "mean coherence", "lead", "share significant"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class WaveletCoherence:
    r"""Morlet wavelet coherence of two series, with cone of influence and surrogate significance.

    The continuous transform of a series against the Morlet wavelet
    :math:`\psi(\eta) = \pi^{-1/4} e^{i\omega_0\eta} e^{-\eta^2/2}`
    at scale :math:`s`,

    .. math::

        W_x(s, t) = \sum_{u} x_u \sqrt{\frac{\delta}{s}}\;
        \overline{\psi}\Bigl(\frac{u - t}{s}\Bigr),

    is evaluated on a dyadic grid :math:`s_k = s_0 2^{k / v}` with
    :math:`v` voices per octave from ``smallest`` up to the sample
    length, for both series; the coherence and phase of the pair, the
    cone of influence, and (unless ``surrogates=0``) the per-scale
    significance level are assembled into a
    :class:`WaveletCoherenceResult`. The significance level is
    empirical: each series is replaced ``surrogates`` times by a
    Gaussian AR(1) with its own lag-one autocorrelation and variance,
    the two independent surrogates' coherence is pooled over dates
    outside the cone at every scale, and its :math:`1 - \alpha`
    quantile is the level a real coherence has to beat.

    Attributes:
        _omega0: The Morlet centre frequency.
        _per_octave: Voices per octave.
        _smallest: The smallest scale.
        _surrogates: Surrogate pairs for the significance level.
        _alpha: The significance level.
        _seed: The seed for the surrogate draws, or ``None``.

    Args:
        omega0: Morlet centre frequency, above 4; ``6`` makes scale and
            Fourier period nearly equal (ratio 1.03) and satisfies
            admissibility.
        per_octave: Voices per octave on the dyadic scale grid, at least
            1; 8 by default.
        smallest: Smallest scale, in observations, above 1; ``2`` is the
            Nyquist limit.
        surrogates: AR(1) surrogate pairs behind the significance level,
            300 by default; ``0`` skips it and leaves ``significance``
            as ``None``.
        alpha: Level of the significance threshold, inside ``(0, 1)``.
        seed: Seed for the surrogate draws; ``None`` draws fresh each
            call.

    Raises:
        SpecificationError: If a setting is out of range.

    Note:
        The surrogate step dominates the cost -- it runs the full
        bivariate transform ``surrogates`` times -- so the default 300
        pairs on a few hundred observations takes seconds, and the
        example below uses ``surrogates=0`` where only the point
        estimate is needed. The AR(1) null tests comovement beyond what
        two independent persistent series would show; it is not a test
        against a richer null, and a pair of series sharing a common
        deterministic seasonal will be "significantly" coherent at that
        period for the uninteresting reason.

    Warning:
        With ``seed=None`` the significance level differs between runs
        by surrogate sampling noise. Pass a seed for any result that
        will be reported.

    See Also:
        * :class:`WaveletCoherenceResult` -- the result.
        * :class:`MODWT` -- the discrete transform of one series.
        * :class:`~cultivars.spectral.causality.SpectralCausality` --
          directional dependence by frequency from a fitted model, the
          question coherence cannot answer.

    References:
        Torrence, C., & Compo, G. P. (1998). A practical guide to
        wavelet analysis. *Bulletin of the American Meteorological
        Society*, 79(1), 61-78.

        Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004). Application
        of the cross wavelet transform and wavelet coherence to
        geophysical time series. *Nonlinear Processes in Geophysics*,
        11, 561-566.

    Example:
        >>> t = np.arange(256.0)
        >>> x = np.sin(2 * np.pi * t / 16)
        >>> y = np.sin(2 * np.pi * (t - 2) / 16)
        >>> res = WaveletCoherence(per_octave=4, surrogates=0).compute(x, y)
        >>> bool(res.coherence[np.argmin(np.abs(res.periods - 16)), 128] > 0.95)
        True
        >>> bool(1.5 < res.lead((14.0, 18.0)) < 2.5)
        True

        The phase at the signal's period is the delay in radians,
        :math:`2\pi \cdot 2 / 16`:

        >>> row = int(np.argmin(np.abs(res.periods - 16)))
        >>> round(float(res.phase[row, 128]), 3), round(2 * np.pi * 2 / 16, 3)
        (0.785, 0.785)
    """

    __slots__ = ("_alpha", "_omega0", "_per_octave", "_seed", "_smallest", "_surrogates")

    def __init__(
        self,
        *,
        omega0: float = _MORLET_OMEGA0,
        per_octave: int = _SCALES_PER_OCTAVE,
        smallest: float = 2.0,
        surrogates: int = _COHERENCE_SURROGATES,
        alpha: float = _DEFAULT_ALPHA,
        seed: int | None = None,
    ) -> None:
        """Validate the transform and significance settings.

        Args:
            omega0: Morlet centre frequency, above 4.
            per_octave: Voices per octave, at least 1.
            smallest: Smallest scale, above 1.
            surrogates: Surrogate pairs, at least 0.
            alpha: Significance level inside ``(0, 1)``.
            seed: Seed for the surrogate draws.

        Raises:
            SpecificationError: If any setting is out of its range or of
                the wrong type.

        Example:
            >>> wc = WaveletCoherence(per_octave=4, surrogates=50, seed=3)
            >>> wc._omega0, wc._per_octave, wc._surrogates, wc._seed
            (6.0, 4, 50, 3)
            >>> WaveletCoherence(omega0=4)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: omega0 must lie in (4.0, inf); got 4.0.
            >>> WaveletCoherence(surrogates=-1)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: surrogates must be >= 0; got -1.
        """
        self._omega0 = validate_open_interval(omega0, "omega0", low=4.0, high=np.inf)
        self._per_octave = validate_order(per_octave, "per_octave", minimum=1)
        self._smallest = validate_open_interval(smallest, "smallest", low=1.0, high=np.inf)
        self._surrogates = validate_order(surrogates, "surrogates", minimum=0)
        self._alpha = validate_open_interval(alpha, "alpha", low=0.0, high=1.0)
        self._seed = seed

    def compute(self, first: npt.ArrayLike, second: npt.ArrayLike) -> WaveletCoherenceResult:
        """Compute coherence and phase between two aligned series.

        Builds the scale grid for the sample length, transforms both
        series, smooths and normalizes the cross-wavelet, marks the cone
        of influence, and -- if surrogates were requested -- draws the
        AR(1) surrogates and their per-scale significance level.

        Args:
            first: The ``(T,)`` reference series; a positive phase means
                it leads.
            second: The other ``(T,)`` series, of the same length.

        Returns:
            The :class:`WaveletCoherenceResult`.

        Raises:
            DimensionError: If either series is not one-dimensional or
                the second does not match the first's length.
            SpecificationError: If they are shorter than 32.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> a, b = rng.standard_normal(128), rng.standard_normal(128)
            >>> res = WaveletCoherence(per_octave=2, surrogates=0).compute(a, b)
            >>> res.coherence.shape, res.significance is None, bool(res.cone[-1].all())
            ((13, 128), True, True)
            >>> WaveletCoherence(surrogates=0).compute(a, b[:100])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.DimensionError: second must be one-dimensional with length 128; ...
        """
        x = validate_endog(first)
        n = x.shape[0]
        y = validate_aligned(second, n, "second")
        if n < _MIN_SPECTRUM_OBS:
            raise SpecificationError(
                f"wavelet coherence needs at least {_MIN_SPECTRUM_OBS} observations; got {n}."
            )
        scales = _morlet_scales(n, smallest=self._smallest, per_octave=self._per_octave)
        coherence, phase, _ = _wavelet_coherence(x, y, scales, self._omega0, self._per_octave)
        fourier = 4.0 * np.pi / (self._omega0 + np.sqrt(2.0 + self._omega0**2))
        significance = None
        if self._surrogates > 0:
            significance = _coherence_surrogates(
                x,
                y,
                scales,
                self._omega0,
                self._per_octave,
                self._surrogates,
                self._alpha,
                np.random.default_rng(self._seed),
            )
        return WaveletCoherenceResult(
            coherence=coherence,
            phase=phase,
            scales=scales,
            periods=fourier * scales,
            cone=_cone_of_influence(n, scales),
            significance=significance,
            alpha=self._alpha,
            omega0=self._omega0,
            nobs=n,
        )
