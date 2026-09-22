"""Wavelet analysis: the MODWT multiresolution decomposition and Morlet wavelet coherence.

A spectrum answers "at which frequencies" and a filter "what is the
cycle"; a wavelet decomposition answers both at once, by frequency band
and by date. The maximal-overlap discrete wavelet transform (MODWT) of
Percival and Walden splits a series into details at dyadic scales
``1, 2, 4, ...`` observations plus a smooth, additively, aligned in time
and defined for any sample length; its wavelet variance is an
analysis of variance by scale with a chi-squared band, and it is the
usual way to ask which horizon carries a series' variability. Wavelet
coherence, on Torrence and Compo's continuous Morlet transform, is the
local squared coherence of two series across scale and time, with the
phase difference reading which leads; it is the instrument behind
"co-movement of output and inflation at business-cycle frequencies
broke down after 1985" and its like. Both are descriptive: the cone of
influence and the surrogate significance level say where the picture
can be trusted, not that a model holds.

References:
    Percival, D. B., & Walden, A. T. (2000). *Wavelet Methods for Time
        Series Analysis*. Cambridge University Press.
    Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet
        analysis. *Bulletin of the American Meteorological Society*,
        79(1), 61-78.
    Grinsted, A., Moore, J. C., & Jevrejeva, S. (2004). Application of
        the cross wavelet transform and wavelet coherence to geophysical
        time series. *Nonlinear Processes in Geophysics*, 11, 561-566.
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
    """A MODWT decomposition: coefficients, additive components, and variance by scale.

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
    """

    coefficients: npt.NDArray[np.float64] = field(repr=False)
    scaling: npt.NDArray[np.float64] = field(repr=False)
    details: npt.NDArray[np.float64] = field(repr=False)
    smooth: npt.NDArray[np.float64] = field(repr=False)
    variance: npt.NDArray[np.float64]
    degrees_of_freedom: npt.NDArray[np.float64] = field(repr=False)
    wavelet: str
    nobs: int

    @property
    def levels(self) -> int:
        """Number of detail levels ``J``."""
        return int(self.coefficients.shape[0])

    @property
    def scales(self) -> npt.NDArray[np.float64]:
        """Physical scale ``2^(j-1)`` of each level, in observations."""
        return np.asarray(2.0 ** np.arange(self.levels), dtype=np.float64)

    @property
    def periods(self) -> tuple[tuple[float, float], ...]:
        """The band of periods, in observations, each level responds to."""
        return tuple((float(2.0 ** (j + 1)), float(2.0 ** (j + 2))) for j in range(self.levels))

    def variance_interval(
        self, alpha: float = _DEFAULT_ALPHA
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Chi-squared confidence band for the wavelet variance at each level.

        Args:
            alpha: Two-sided miscoverage.

        Returns:
            ``(lower, upper)`` each ``(J,)``, from ``nu v / chi2`` quantiles
            with the level's equivalent degrees of freedom.
        """
        nu = self.degrees_of_freedom
        lower = nu * self.variance / chi2.ppf(1.0 - alpha / 2.0, nu)
        upper = nu * self.variance / chi2.ppf(alpha / 2.0, nu)
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    def _summary_table(self) -> SummaryTable:
        """Variance by scale with its band and share."""
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
    """Maximal-overlap discrete wavelet transform and multiresolution analysis.

    Args:
        wavelet: ``"haar"``, ``"d4"`` (Daubechies extremal phase, 4 taps)
            or ``"la8"`` (least asymmetric, 8 taps; the default -- near-
            zero phase and a sharper band split than Haar).
        levels: Detail levels ``J``. By default the largest ``J`` for
            which every level keeps a boundary-free coefficient,
            ``floor(log2(T / (L - 1)))``.

    Raises:
        SpecificationError: If the wavelet is unknown or ``levels`` is
            not positive.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = rng.standard_normal(256)
        >>> res = MODWT(wavelet="haar", levels=3).transform(y)
        >>> res.levels, bool(np.allclose(res.details.sum(axis=0) + res.smooth, y))
        (3, True)
    """

    __slots__ = ("_levels", "_wavelet")

    def __init__(self, *, wavelet: str = "la8", levels: int | None = None) -> None:
        """Validate the filter and depth."""
        self._wavelet = validate_choice(wavelet, ("haar", "d4", "la8"), "wavelet")
        self._levels = None if levels is None else validate_order(levels, "levels", minimum=1)

    def transform(self, endog: npt.ArrayLike) -> MODWTResult:
        """Decompose a series.

        Args:
            endog: The ``(T,)`` series.

        Returns:
            The :class:`MODWTResult`.

        Raises:
            SpecificationError: If the series is too short for the depth.
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
    """Squared wavelet coherence and phase of two series over scale and time.

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
    """

    coherence: npt.NDArray[np.float64] = field(repr=False)
    phase: npt.NDArray[np.float64] = field(repr=False)
    scales: npt.NDArray[np.float64] = field(repr=False)
    periods: npt.NDArray[np.float64] = field(repr=False)
    cone: npt.NDArray[np.bool_] = field(repr=False)
    significance: npt.NDArray[np.float64] | None = field(repr=False)
    alpha: float
    omega0: float
    nobs: int

    def significant(self) -> npt.NDArray[np.bool_]:
        """Where coherence exceeds the surrogate level, outside the cone of influence.

        Raises:
            SpecificationError: If no surrogates were run.
        """
        if self.significance is None:
            raise SpecificationError(
                "no significance level: rerun WaveletCoherence with surrogates > 0."
            )
        return np.asarray(
            (self.coherence > self.significance[:, None]) & ~self.cone, dtype=np.bool_
        )

    def lead(self, scale_band: tuple[float, float]) -> float:
        """Mean lead of the first series, in observations, over a band of periods.

        The phase difference at each date in the band is converted to
        time as ``phase * period / (2 pi)`` and averaged over dates
        outside the cone of influence and scales whose period lies in
        ``scale_band``.

        Args:
            scale_band: ``(low, high)`` periods, inclusive.

        Returns:
            Positive when the first series leads, negative when it lags.

        Raises:
            SpecificationError: If no scale falls in the band.
        """
        low, high = scale_band
        rows = (self.periods >= low) & (self.periods <= high)
        if not np.any(rows):
            raise SpecificationError(f"no scale has a period in [{low}, {high}].")
        lag = self.phase[rows] * self.periods[rows][:, None] / (2.0 * np.pi)
        keep = ~self.cone[rows]
        return float(np.mean(lag[keep])) if np.any(keep) else np.nan

    def _summary_table(self) -> SummaryTable:
        """Mean coherence and lead per octave, outside the cone of influence."""
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
    """Morlet wavelet coherence of two series, with cone of influence and surrogate significance.

    Args:
        omega0: Morlet centre frequency; ``6`` makes scale and Fourier
            period nearly equal and satisfies admissibility.
        per_octave: Voices per octave on the dyadic scale grid.
        smallest: Smallest scale, in observations; ``2`` is the Nyquist
            limit.
        surrogates: AR(1) surrogate pairs behind the significance level;
            ``0`` skips it and leaves ``significance`` as ``None``.
        alpha: Level of the significance threshold.
        seed: Seed for the surrogate draws.

    Raises:
        SpecificationError: If a setting is out of range.

    Example:
        >>> t = np.arange(256.0)
        >>> x = np.sin(2 * np.pi * t / 16)
        >>> y = np.sin(2 * np.pi * (t - 2) / 16)
        >>> res = WaveletCoherence(per_octave=4, surrogates=0).compute(x, y)
        >>> bool(res.coherence[np.argmin(np.abs(res.periods - 16)), 128] > 0.95)
        True
        >>> bool(1.5 < res.lead((14.0, 18.0)) < 2.5)
        True
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
        """Validate the transform and significance settings."""
        self._omega0 = validate_open_interval(omega0, "omega0", low=4.0, high=np.inf)
        self._per_octave = validate_order(per_octave, "per_octave", minimum=1)
        self._smallest = validate_open_interval(smallest, "smallest", low=1.0, high=np.inf)
        self._surrogates = validate_order(surrogates, "surrogates", minimum=0)
        self._alpha = validate_open_interval(alpha, "alpha", low=0.0, high=1.0)
        self._seed = seed

    def compute(self, first: npt.ArrayLike, second: npt.ArrayLike) -> WaveletCoherenceResult:
        """Compute coherence and phase between two aligned series.

        Args:
            first: The ``(T,)`` reference series; a positive phase means
                it leads.
            second: The other ``(T,)`` series.

        Returns:
            The :class:`WaveletCoherenceResult`.

        Raises:
            DimensionError: If the series do not align.
            SpecificationError: If they are too short.
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
