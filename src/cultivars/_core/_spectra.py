# filepath: /src/cultivars/_core/_spectra.py
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

"""Frequency-domain primitives: grids, transfer functions, spectral matrices.

Stated once so no two frequency-domain consumers can disagree about a
convention. The conventions are: frequencies live on ``[0, pi]`` in radians
per observation (the process is real, so the negative half-circle is the
conjugate mirror and reporting it would be decoration); the transfer
function uses ``e**(-i omega l)`` for lag ``l``; and the spectral density
carries the ``1 / (2 pi)`` normalization, so integrating it over
``[-pi, pi]`` -- twice its real part over ``[0, pi]`` -- returns the
autocovariance at lag zero. Everything here is a closed-form map from a
fitted model's ``(coefficients, sigma_u)``; nothing touches data.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..exceptions import NumericalError, SpecificationError


def _seasonal_frequencies(period: int) -> tuple[float, ...]:
    """Harmonic seasonal frequencies ``2 pi k / s`` for ``k = 1 .. s/2 - 1``.

    Example:
        >>> np.round(_seasonal_frequencies(4), 4)
        array([1.5708])
    """
    return tuple(2.0 * np.pi * k / period for k in range(1, period // 2))


def frequency_grid(n_frequencies: int) -> npt.NDArray[np.float64]:
    """An inclusive uniform grid on ``[0, pi]``.

    Args:
        n_frequencies: Grid points, at least two; both endpoints included.

    Returns:
        A ``(n_frequencies,)`` array from ``0`` to ``pi``.

    Raises:
        SpecificationError: If fewer than two points are asked for.

    Example:
        >>> grid = frequency_grid(5)
        >>> bool(grid[0] == 0.0 and abs(grid[-1] - np.pi) < 1e-15)
        True
    """
    if n_frequencies < 2:
        raise SpecificationError(f"n_frequencies must be at least 2; got {n_frequencies}.")
    return np.linspace(0.0, np.pi, n_frequencies)


def transfer_function(
    coefficients: npt.NDArray[np.float64], frequencies: npt.NDArray[np.float64]
) -> npt.NDArray[np.complex128]:
    """The moving-average transfer function ``Psi(omega)`` of a lag stack.

    ``Psi(omega) = [I - sum_l A_l e**(-i omega l)]**(-1)``: the frequency
    response mapping innovations to observables. A (near) unit root makes
    the lag polynomial (near) singular at frequency zero, in which case the
    transfer function is legitimately enormous there -- that is the model
    speaking, not an error -- and only an exactly singular solve raises.

    Args:
        coefficients: The ``(p, k, k)`` lag stack.
        frequencies: Radian frequencies, any shape ``(n,)``.

    Returns:
        A ``(n, k, k)`` complex array.

    Raises:
        NumericalError: If the lag polynomial is exactly singular at some
            frequency.
    """
    order = int(coefficients.shape[0])
    k = int(coefficients.shape[-1]) if coefficients.ndim == 3 else 0
    if coefficients.ndim != 3 or coefficients.shape != (order, k, k):
        raise SpecificationError(
            f"coefficients must be a (p, k, k) lag stack; got shape {coefficients.shape}."
        )
    n = int(frequencies.shape[0])
    polynomial = np.tile(np.eye(k, dtype=np.complex128), (n, 1, 1))
    for lag in range(1, order + 1):
        phase = np.exp(-1j * frequencies * lag)
        polynomial -= phase[:, None, None] * coefficients[lag - 1]
    try:
        return np.asarray(np.linalg.inv(polynomial), dtype=np.complex128)
    except np.linalg.LinAlgError as error:
        raise NumericalError(
            "the lag polynomial is exactly singular at some frequency; the "
            "system has an exact unit root there and no spectral density."
        ) from error


def spectral_matrix(
    transfer: npt.NDArray[np.complex128], sigma_u: npt.NDArray[np.float64]
) -> npt.NDArray[np.complex128]:
    """The spectral density matrix ``Psi Sigma Psi* / (2 pi)``.

    Hermitian at every frequency by construction, and verified rather than
    assumed, because a broken Hermitian symmetry upstream would silently
    corrupt every coherence and causality number built on top.

    Args:
        transfer: ``(n, k, k)`` transfer function values.
        sigma_u: ``(k, k)`` innovation covariance.

    Returns:
        A ``(n, k, k)`` complex Hermitian array.

    Raises:
        NumericalError: If the result loses Hermitian symmetry.
    """
    density = np.asarray(
        transfer @ sigma_u @ np.conj(np.swapaxes(transfer, -1, -2)) / (2.0 * np.pi),
        dtype=np.complex128,
    )
    drift = float(np.abs(density - np.conj(np.swapaxes(density, -1, -2))).max())
    scale = float(np.abs(density).max())
    if drift > 1e-8 * max(scale, 1.0):
        raise NumericalError(
            "the spectral density lost Hermitian symmetry; the transfer "
            "function or covariance upstream is corrupt."
        )
    return density


def _pairwise_measure(
    transfer: npt.NDArray[np.complex128], sigma: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Geweke's bivariate measure from an innovation representation.

    Ordering is ``[effect, cause]``. The instantaneous correlation is
    rotated out of the cause's innovation (equivalently, shared variance is
    attributed to the effect), the effect's spectrum splits into an
    intrinsic and a causal part, and the measure is the log of total over
    intrinsic -- computed in ``log1p`` form so a zero causal part is an
    exact zero.

    Args:
        transfer: ``(n, 2, 2)`` transfer function of the pair.
        sigma: ``(2, 2)`` innovation covariance of the pair.

    Returns:
        The ``(n,)`` measure of cause -> effect.
    """
    s_yy, s_xy, s_xx = float(sigma[0, 0]), float(sigma[0, 1]), float(sigma[1, 1])
    rotated = transfer[:, 0, 0] + (s_xy / s_yy) * transfer[:, 0, 1]
    residual_xx = max(s_xx - s_xy**2 / s_yy, 0.0)
    intrinsic = np.abs(rotated) ** 2 * s_yy
    causal = np.abs(transfer[:, 0, 1]) ** 2 * residual_xx
    return np.asarray(np.log1p(causal / np.maximum(intrinsic, 1e-300)), dtype=np.float64)


def _butterworth_penalty(period: float, order: int) -> float:
    """Penalty of the Butterworth sine filter with cutoff ``period`` and ``order`` differences.

    Gomez's (2001) sine-form Butterworth low-pass has gain ``1 / (1 + (sin(w / 2)
    / sin(w_c / 2))^(2n))``, which is the gain of the penalized smoother
    ``min sum (y - tau)^2 + lambda sum (Delta^n tau)^2`` at ``lambda = (2
    sin(w_c / 2))^(-2n)``; the Hodrick-Prescott filter is the ``n = 2`` case.

    Example:
        >>> round(_butterworth_penalty(39.7, 2))
        1601
    """
    return float((2.0 * np.sin(np.pi / period)) ** (-2 * order))


def _penalty_cutoff_period(penalty: float, order: int) -> float:
    """Cutoff period at which the penalized smoother's gain is one half; the inverse of the above.

    Example:
        >>> round(_penalty_cutoff_period(1600.0, 2), 1)
        39.7
    """
    return float(np.pi / np.arcsin(0.5 * penalty ** (-1.0 / (2 * order))))


def _fourier_frequencies(length: int) -> npt.NDArray[np.float64]:
    """Fourier frequencies ``2 pi j / N`` for ``j = 0 .. N // 2``, on ``[0, pi]``.

    Example:
        >>> _fourier_frequencies(8).round(4)
        array([0.    , 0.7854, 1.5708, 2.3562, 3.1416])
    """
    return 2.0 * np.pi * np.arange(length // 2 + 1) / length


def _cross_periodogram(
    block: npt.NDArray[np.float64], taper: npt.NDArray[np.float64] | None = None
) -> npt.NDArray[np.complex128]:
    """Tapered cross-periodogram matrix of a ``(N, k)`` block on the Fourier frequencies.

    ``I(w) = X(w) X(w)* / (2 pi sum h_t^2)`` with ``X`` the discrete Fourier
    transform of the tapered columns, normalized so that twice the integral
    over ``[0, pi]`` of a diagonal recovers that column's variance.

    Args:
        block: ``(N, k)`` data, already demeaned or detrended.
        taper: ``(N,)`` window, or ``None`` for the rectangular one.

    Returns:
        ``(N // 2 + 1, k, k)`` complex Hermitian array.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal((400, 1))
        >>> I = _cross_periodogram(x)
        >>> freqs = _fourier_frequencies(400)
        >>> bool(abs(2.0 * np.trapezoid(I[:, 0, 0].real, freqs) - x.var()) < 0.05)
        True
    """
    weights = np.ones(block.shape[0]) if taper is None else taper
    transform = np.fft.rfft(block * weights[:, None], axis=0)
    scale = 2.0 * np.pi * float(weights @ weights)
    return np.asarray(
        transform[:, :, None] * np.conj(transform[:, None, :]) / scale, dtype=np.complex128
    )


def _daniell_density(
    panel: npt.NDArray[np.float64], span: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128], float]:
    """Periodogram smoothed with the modified Daniell kernel of half-width ``span``.

    The kernel weights ``1 / (2 span)`` the ``2 span - 1`` interior
    ordinates and ``1 / (4 span)`` the two end ordinates; the smoothing is
    applied over the full circle so that frequencies near ``0`` and ``pi``
    borrow from their mirror images rather than from a truncated window.
    Equivalent degrees of freedom are ``2 / sum w^2``.

    Args:
        panel: ``(N, k)`` demeaned data.
        span: Half-width ``m``; the kernel averages ``2 m + 1`` ordinates.

    Returns:
        ``(frequencies, density, degrees_of_freedom)``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> f, d, nu = _daniell_density(rng.standard_normal((400, 2)), 4)
        >>> f.shape, d.shape, round(nu, 1)
        ((201,), (201, 2, 2), 17.1)
    """
    n = panel.shape[0]
    raw = _cross_periodogram(panel)
    half = n // 2
    circle = np.concatenate([raw, np.conj(raw[n - half - 1 : 0 : -1])], axis=0)  # length n
    kernel = np.full(2 * span + 1, 1.0 / (2.0 * span))
    kernel[0] = kernel[-1] = 1.0 / (4.0 * span)
    padded = np.concatenate([circle[-span:], circle, circle[:span]], axis=0)
    smoothed = np.zeros_like(raw)
    for j, weight in enumerate(kernel):
        smoothed += weight * padded[j : j + n][: half + 1]
    return _fourier_frequencies(n), smoothed, float(2.0 / (kernel @ kernel))


def _multitaper_density(
    panel: npt.NDArray[np.float64], bandwidth: float, n_tapers: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128], float]:
    """Thomson's multitaper estimate with ``n_tapers`` Slepian sequences of time-bandwidth ``NW``.

    Each taper gives a nearly leakage-free periodogram; averaging ``K``
    of them, weighted by their concentration ratios, trades resolution
    of ``2 NW / N`` cycles per observation for ``2 K`` equivalent degrees
    of freedom without the bias of a single smooth window.

    Args:
        panel: ``(N, k)`` demeaned data.
        bandwidth: Time-bandwidth product ``NW``; 4 is the usual choice.
        n_tapers: Tapers ``K``, at most ``2 NW - 1`` for good concentration.

    Returns:
        ``(frequencies, density, degrees_of_freedom)``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> f, d, nu = _multitaper_density(rng.standard_normal((400, 1)), 4.0, 7)
        >>> f.shape, d.shape, nu
        ((201,), (201, 1, 1), 14.0)
    """
    from scipy.signal.windows import dpss

    n = panel.shape[0]
    tapers, ratios = dpss(n, bandwidth, n_tapers, return_ratios=True)
    weights = np.asarray(ratios, dtype=np.float64) / float(np.sum(ratios))
    density = np.zeros((n // 2 + 1, panel.shape[1], panel.shape[1]), dtype=np.complex128)
    for taper, weight in zip(np.atleast_2d(tapers), weights, strict=True):
        density += weight * _cross_periodogram(panel, np.asarray(taper, dtype=np.float64))
    return _fourier_frequencies(n), density, float(2 * n_tapers)


def _welch_density(
    panel: npt.NDArray[np.float64], segment: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128], float]:
    """Welch's averaged periodogram over Hann-tapered segments with half overlap.

    Args:
        panel: ``(N, k)`` demeaned data.
        segment: Segment length ``L``; the grid has ``L // 2 + 1`` points.

    Returns:
        ``(frequencies, density, degrees_of_freedom)`` with the Percival-
        Walden equivalent degrees of freedom ``36 B / 19`` for ``B``
        half-overlapped Hann segments.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> f, d, nu = _welch_density(rng.standard_normal((400, 1)), 100)
        >>> f.shape, d.shape, round(nu, 1)
        ((51,), (51, 1, 1), 13.3)
    """
    n = panel.shape[0]
    step = segment // 2
    starts = range(0, n - segment + 1, step)
    taper = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(segment) / segment)
    density = np.zeros((segment // 2 + 1, panel.shape[1], panel.shape[1]), dtype=np.complex128)
    count = 0
    for start in starts:
        block = panel[start : start + segment]
        density += _cross_periodogram(block - block.mean(axis=0), taper)
        count += 1
    return _fourier_frequencies(segment), density / count, float(36.0 * count / 19.0)


def _ideal_weights(count: int, low: float, high: float) -> npt.NDArray[np.float64]:
    """The ideal band-pass filter's weights at lags ``0 .. count``.

    ``b_0 = (b - a) / pi`` and ``b_j = (sin(j b) - sin(j a)) / (pi j)``
    with ``a = 2 pi / high`` and ``b = 2 pi / low`` the band's edge
    frequencies.
    """
    a = 2.0 * np.pi / high
    b = 2.0 * np.pi / low
    lags = np.arange(1, count + 1, dtype=np.float64)
    weights = np.empty(count + 1, dtype=np.float64)
    weights[0] = (b - a) / np.pi
    weights[1:] = (np.sin(lags * b) - np.sin(lags * a)) / (np.pi * lags)
    return weights


def _fractional_spectrum(
    freqs: npt.NDArray[np.float64], *, d: float, sigma2: float, phi: float = 0.0
) -> npt.NDArray[np.float64]:
    """Spectral density of an ARFIMA(1, d, 0) process at the given frequencies.

    ``f(lambda) = sigma2 / (2 pi) * |1 - phi e^{-i lambda}|**-2 * (2 sin(lambda / 2))**-2d``,
    the long-memory kernel times a first-order short-memory factor. At
    ``lambda = 0`` with ``d > 0`` the density is infinite; callers that
    include the zero frequency must handle it.

    Args:
        freqs: Frequencies in ``[0, pi]``.
        d: Fractional differencing order, ``|d| < 0.5``.
        sigma2: Innovation variance.
        phi: Short-memory AR(1) coefficient, ``|phi| < 1``.

    Returns:
        The density at each frequency.
    """
    lam = np.asarray(freqs, dtype=np.float64)
    short = 1.0 / (1.0 - 2.0 * phi * np.cos(lam) + phi**2)
    with np.errstate(divide="ignore"):
        long_memory = (2.0 * np.sin(0.5 * lam)) ** (-2.0 * d)
    return np.asarray(sigma2 / (2.0 * np.pi) * short * long_memory, dtype=np.float64)
