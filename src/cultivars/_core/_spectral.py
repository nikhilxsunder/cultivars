# filepath: /src/cultivars/_core/_spectral.py
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

"""Frequency-domain primitives."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..exceptions import SpecificationError


def periodogram(
    y: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Positive Fourier frequencies and the periodogram, with the DC term dropped.

    The series is mean-centered first, which makes the zero frequency
    uninformative; it is discarded so that log-periodogram regressions are not
    anchored by a structurally zero ordinate.

    Args:
        y: The series, shape ``(n,)``.

    Returns:
        A tuple ``(freqs, ordinates)``, each of length ``floor(n / 2)``.

    Raises:
        SpecificationError: If the series has fewer than two observations.
    """
    n = y.shape[0]
    if n < 2:
        raise SpecificationError(f"periodogram requires at least 2 observations; got {n}.")
    transform = np.fft.rfft(y - y.mean())
    ordinates = (np.abs(transform) ** 2) / n
    freqs = 2.0 * np.pi * np.arange(transform.shape[0]) / n
    return freqs[1:], ordinates[1:]


def bandwidth(nobs: int, m: int | None, exponent: float) -> int:
    """Resolve the number of Fourier frequencies for a semiparametric estimator.

    Args:
        nobs: Series length.
        m: Explicit bandwidth, or ``None`` to derive it from ``exponent``.
        exponent: Exponent in the default rule ``m = floor(n ** exponent)``.

    Returns:
        The bandwidth, never below 2.

    Raises:
        SpecificationError: If an explicit ``m`` is below 2, or ``exponent``
            does not lie in ``(0, 1)``.

    Example:
        >>> bandwidth(400, None, 0.5)
        20
    """
    if m is not None:
        if m < 2:
            raise SpecificationError(f"bandwidth m must be >= 2; got {m}.")
        return int(m)
    if not (0.0 < exponent < 1.0):
        raise SpecificationError(f"bandwidth_exponent must lie in (0, 1); got {exponent}.")
    return max(2, int(np.floor(nobs**exponent)))
