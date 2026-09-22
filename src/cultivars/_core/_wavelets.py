"""Wavelet primitives: MODWT pyramid, multiresolution synthesis, Morlet transform, smoothing.

The maximal-overlap discrete wavelet transform keeps ``T`` coefficients
at every level (no decimation), which makes it shift-invariant and lets
the details of a multiresolution analysis line up with the data date by
date; its price is redundancy. The continuous Morlet transform is what
cross-wavelet coherence needs: complex, so phase is available, and
smooth in scale. Filters are Percival and Walden's tables.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter

from ..exceptions import SpecificationError
from ._defaults import _SCALING_FILTERS, _SQRT2


def _modwt_filters(wavelet: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """MODWT wavelet and scaling filters ``(h~, g~)`` for a named wavelet.

    The DWT scaling filter is rescaled by ``1 / sqrt 2``; the wavelet
    filter is its quadrature mirror ``h_l = (-1)^l g_{L-1-l}``.

    Raises:
        SpecificationError: If the wavelet is unknown.

    Example:
        >>> h, g = _modwt_filters("haar")
        >>> h.round(4).tolist(), g.round(4).tolist()
        ([0.5, -0.5], [0.5, 0.5])
    """
    if wavelet not in _SCALING_FILTERS:
        raise SpecificationError(
            f"unknown wavelet {wavelet!r}; choose from {tuple(_SCALING_FILTERS)}."
        )
    g = np.asarray(_SCALING_FILTERS[wavelet], dtype=np.float64)
    length = g.shape[0]
    h = np.array([(-1.0) ** k * g[length - 1 - k] for k in range(length)])
    return h / _SQRT2, g / _SQRT2


def _modwt(
    y: npt.NDArray[np.float64], wavelet: str, levels: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The MODWT pyramid with periodic boundary treatment.

    Level ``j`` filters the level ``j - 1`` scaling coefficients with the
    filters upsampled by ``2^(j-1)``:
    ``W_{j,t} = sum_l h_l V_{j-1, t - 2^(j-1) l mod T``.

    Args:
        y: ``(T,)`` series.
        wavelet: ``"haar"``, ``"d4"`` or ``"la8"``.
        levels: Depth ``J``.

    Returns:
        ``(W, V)`` with ``W`` of shape ``(J, T)`` -- wavelet coefficients
        by level -- and ``V`` the ``(T,)`` level-``J`` scaling coefficients.

    Example:
        >>> y = np.arange(8.0)
        >>> W, V = _modwt(y, "haar", 2)
        >>> W.shape, V.shape, bool(abs(W[0, 3] - 0.5) < 1e-12)
        ((2, 8), (8,), True)
    """
    h, g = _modwt_filters(wavelet)
    n = y.shape[0]
    index = np.arange(n)
    details = np.empty((levels, n))
    smooth = y.astype(np.float64)
    for j in range(levels):
        stride = 2**j
        w = np.zeros(n)
        v = np.zeros(n)
        for k, (hk, gk) in enumerate(zip(h, g, strict=True)):
            shifted = smooth[(index - stride * k) % n]
            w += hk * shifted
            v += gk * shifted
        details[j] = w
        smooth = v
    return details, smooth


def _modwt_inverse(
    details: npt.NDArray[np.float64], smooth: npt.NDArray[np.float64], wavelet: str
) -> npt.NDArray[np.float64]:
    """Invert the MODWT pyramid, ``V_{j-1,t} = sum_l h_l W_{j,t+2^(j-1)l} + g_l V_{j,t+2^(j-1)l}``.

    Example:
        >>> y = np.sin(np.arange(16.0))
        >>> W, V = _modwt(y, "la8", 2)
        >>> bool(np.allclose(_modwt_inverse(W, V, "la8"), y))
        True
    """
    h, g = _modwt_filters(wavelet)
    n = smooth.shape[0]
    index = np.arange(n)
    current = smooth.astype(np.float64)
    for j in range(details.shape[0] - 1, -1, -1):
        stride = 2**j
        previous = np.zeros(n)
        for k, (hk, gk) in enumerate(zip(h, g, strict=True)):
            positions = (index + stride * k) % n
            previous += hk * details[j][positions] + gk * current[positions]
        current = previous
    return current


def _modwt_details(
    details: npt.NDArray[np.float64], smooth: npt.NDArray[np.float64], wavelet: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Multiresolution analysis: the detail at each level and the final smooth, summing to the data.

    Detail ``D_j`` is the inverse transform with every coefficient but
    ``W_j`` set to zero; the smooth ``S_J`` keeps only ``V_J``.

    Example:
        >>> y = np.sin(np.arange(16.0))
        >>> W, V = _modwt(y, "d4", 3)
        >>> D, S = _modwt_details(W, V, "d4")
        >>> D.shape, bool(np.allclose(D.sum(axis=0) + S, y))
        ((3, 16), True)
    """
    levels, n = details.shape
    components = np.empty((levels, n))
    for j in range(levels):
        only = np.zeros_like(details)
        only[j] = details[j]
        components[j] = _modwt_inverse(only, np.zeros(n), wavelet)
    return components, _modwt_inverse(np.zeros_like(details), smooth, wavelet)


def _modwt_variance(
    details: npt.NDArray[np.float64], wavelet: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Unbiased wavelet variance per level with Percival and Walden's ``eta_3`` degrees of freedom.

    Coefficients within ``L_j - 1 = (2^j - 1)(L - 1)`` of the start are
    boundary-affected under periodic treatment and are dropped; the
    variance is the mean square of the remainder, and the equivalent
    chi-squared degrees of freedom is ``max(M_j / 2^j, 1)``.

    Returns:
        ``(variance, degrees_of_freedom)`` each ``(J,)``; a level with no
        interior coefficient reports ``nan`` and zero degrees of freedom.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> W, _ = _modwt(rng.standard_normal(512), "haar", 3)
        >>> v, nu = _modwt_variance(W, "haar")
        >>> bool(abs(v.sum() - 0.875) < 0.1), nu.round(0).tolist()
        (True, [256.0, 127.0, 63.0])
    """
    length = len(_SCALING_FILTERS[wavelet])
    levels = details.shape[0]
    variance = np.full(levels, np.nan)
    degrees = np.zeros(levels)
    for j in range(levels):
        boundary = (2 ** (j + 1) - 1) * (length - 1)
        interior = details[j][boundary:]
        count = interior.shape[0]
        if count > 0:
            variance[j] = float(interior @ interior) / count
            degrees[j] = max(count / 2 ** (j + 1), 1.0)
    return variance, degrees


def _morlet_transform(
    y: npt.NDArray[np.float64], scales: npt.NDArray[np.float64], omega0: float
) -> npt.NDArray[np.complex128]:
    """Continuous Morlet wavelet transform on the given scales, via the FFT.

    ``W(s, t) = sum_k Y_k Psi*(s w_k) e^{i w_k t}`` with the Fourier
    Morlet ``Psi(s w) = pi^(-1/4) sqrt(2 pi s / dt) e^{-(s w - w0)^2 / 2}``
    for ``w > 0``, Torrence and Compo's normalization, so that ``|W|^2``
    is comparable across scales.

    Args:
        y: ``(T,)`` series, zero-padded to the next power of two inside.
        scales: ``(S,)`` scales in observations.
        omega0: Morlet centre frequency; 6 is the standard choice.

    Returns:
        ``(S, T)`` complex coefficients.

    Example:
        >>> t = np.arange(256.0)
        >>> W = _morlet_transform(np.sin(2 * np.pi * t / 16), np.array([8.0, 16.0, 32.0]), 6.0)
        >>> W.shape, int(np.argmax(np.abs(W[:, 128])))
        ((3, 256), 1)
    """
    n = y.shape[0]
    padded = 1 << (n - 1).bit_length()
    transform = np.fft.fft(y - y.mean(), padded)
    omega = 2.0 * np.pi * np.fft.fftfreq(padded)
    out = np.empty((scales.shape[0], n), dtype=np.complex128)
    for i, scale in enumerate(scales):
        daughter = np.where(
            omega > 0.0,
            np.pi**-0.25
            * np.sqrt(2.0 * np.pi * scale)
            * np.exp(-0.5 * (scale * omega - omega0) ** 2),
            0.0,
        )
        out[i] = np.fft.ifft(transform * daughter)[:n]
    return out


def _morlet_scales(nobs: int, *, smallest: float, per_octave: int) -> npt.NDArray[np.float64]:
    """Dyadic scale grid ``s_0 2^(k / per_octave)`` up to the sample length.

    Example:
        >>> _morlet_scales(64, smallest=2.0, per_octave=1).tolist()
        [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    """
    largest = int(np.floor(per_octave * np.log2(nobs / smallest)))
    return smallest * 2.0 ** (np.arange(largest + 1) / per_octave)


def _wavelet_smooth(
    power: npt.NDArray[np.complex128] | npt.NDArray[np.float64],
    scales: npt.NDArray[np.float64],
    per_octave: int,
) -> npt.NDArray[np.complex128] | npt.NDArray[np.float64]:
    """Torrence-Webster smoothing of a wavelet field: Gaussian in time per scale, boxcar in scale.

    The time window at scale ``s`` is ``exp(-t^2 / (2 s^2))``; the scale
    window spans ``0.6`` of an octave, the decorrelation length of the
    Morlet wavelet, both as in Grinsted, Moore and Jevrejeva (2004).

    Example:
        >>> field = np.ones((4, 64))
        >>> smoothed = _wavelet_smooth(field, np.array([2.0, 4.0, 8.0, 16.0]), 1)
        >>> bool(np.allclose(smoothed[:, 32], 1.0, atol=1e-6))
        True
    """
    n = power.shape[1]
    padded = 1 << (n - 1).bit_length()
    omega = 2.0 * np.pi * np.fft.fftfreq(padded)
    out = np.empty(power.shape, dtype=np.complex128)
    for i, scale in enumerate(scales):
        kernel = np.exp(-0.5 * (scale * omega) ** 2)
        out[i] = np.fft.ifft(np.fft.fft(power[i], padded) * kernel)[:n]
    width = max(1, round(0.6 * per_octave))
    if width > 1:
        kernel_scale = np.ones(width) / width
        for t in range(n):
            out[:, t] = np.convolve(out[:, t], kernel_scale, mode="same")
    if np.isrealobj(power):
        return np.asarray(np.real(out), dtype=np.float64)
    return np.asarray(out, dtype=np.complex128)


def _wavelet_coherence(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    scales: npt.NDArray[np.float64],
    omega0: float,
    per_octave: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """Squared wavelet coherence and phase difference of two series.

    ``R^2 = |S(W_xy / s)|^2 / (S(|W_x|^2 / s) S(|W_y|^2 / s))`` with ``S``
    the smoothing of :func:`_wavelet_smooth`; the phase difference is the
    argument of the smoothed cross-wavelet.

    Returns:
        ``(coherence, phase, cross)`` each ``(S, T)``.

    Example:
        >>> t = np.arange(256.0)
        >>> x = np.sin(2 * np.pi * t / 16)
        >>> R2, phase, _ = _wavelet_coherence(x, x, np.array([8.0, 16.0, 32.0]), 6.0, 1)
        >>> bool(R2[1, 128] > 0.99)
        True
    """
    wx = _morlet_transform(x, scales, omega0)
    wy = _morlet_transform(y, scales, omega0)
    inverse = 1.0 / scales[:, None]
    cross = _wavelet_smooth(wx * np.conj(wy) * inverse, scales, per_octave)
    power_x = _wavelet_smooth(np.abs(wx) ** 2 * inverse, scales, per_octave)
    power_y = _wavelet_smooth(np.abs(wy) ** 2 * inverse, scales, per_octave)
    coherence = np.abs(cross) ** 2 / np.maximum(power_x * power_y, 1e-300)
    return (
        np.asarray(np.clip(coherence, 0.0, 1.0), dtype=np.float64),
        np.asarray(np.angle(cross), dtype=np.float64),
        np.asarray(cross, dtype=np.complex128),
    )


def _cone_of_influence(nobs: int, scales: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Mask of coefficients within ``sqrt 2`` scales of either edge, where padding contaminates.

    Example:
        >>> _cone_of_influence(8, np.array([1.0, 2.0])).sum(axis=1).tolist()
        [4, 6]
    """
    t = np.arange(nobs, dtype=np.float64)
    distance = np.minimum(t, nobs - 1 - t)
    return np.asarray(distance[None, :] < np.sqrt(2.0) * scales[:, None], dtype=np.bool_)


def _coherence_surrogates(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    scales: npt.NDArray[np.float64],
    omega0: float,
    per_octave: int,
    replications: int,
    alpha: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Per-scale ``1 - alpha`` quantile of coherence between independent AR(1) surrogates.

    Each series is replaced by a Gaussian AR(1) with its own lag-one
    autocorrelation and variance, as in Grinsted, Moore and Jevrejeva
    (2004); the coherence of the two independent surrogates is pooled
    over dates outside the cone of influence, scale by scale.

    Returns:
        ``(S,)`` significance thresholds; coherence above the threshold
        at a scale is unlikely under independence. A scale whose cone of
        influence covers every date reports ``nan``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> x, y = rng.standard_normal((2, 128))
        >>> level = _coherence_surrogates(x, y, np.array([4.0, 8.0]), 6.0, 1, 20, 0.05, rng)
        >>> level.shape, bool(np.all((level > 0.3) & (level < 1.0)))
        ((2,), True)
    """
    n = x.shape[0]
    inside = ~_cone_of_influence(n, scales)
    pooled: list[list[float]] = [[] for _ in scales]

    def surrogate(series: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        centred = series - series.mean()
        rho = float(centred[1:] @ centred[:-1] / (centred @ centred))
        rho = float(np.clip(rho, -0.99, 0.99))
        shocks = rng.standard_normal(n) * np.sqrt(centred.var() * (1.0 - rho**2))
        start = rng.standard_normal() * np.sqrt(centred.var())
        path, _ = lfilter([1.0], [1.0, -rho], shocks, zi=np.array([rho * start]))
        return np.asarray(path, dtype=np.float64)

    for _ in range(replications):
        coherence, _, _ = _wavelet_coherence(surrogate(x), surrogate(y), scales, omega0, per_octave)
        for s in range(scales.shape[0]):
            pooled[s].extend(coherence[s, inside[s]].tolist())
    return np.asarray(
        [float(np.quantile(values, 1.0 - alpha)) if values else np.nan for values in pooled]
    )
