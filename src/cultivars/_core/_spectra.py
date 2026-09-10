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
