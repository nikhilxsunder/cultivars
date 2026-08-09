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


def local_whittle_d(
    y: npt.NDArray[np.float64], m: int | None = None, exponent: float = 0.65
) -> tuple[float, int]:
    """Local Whittle estimate of the fractional differencing parameter.

    Args:
        y: The series.
        m: Explicit bandwidth, or ``None`` for the default rule.
        exponent: Exponent in the default bandwidth rule.

    Returns:
        A tuple ``(d_hat, m_eff)``.
    """
    from scipy.optimize import minimize_scalar

    from ._defaults import _D_MAX

    freqs, ordinates = periodogram(y)
    m_eff = min(bandwidth(y.shape[0], m, exponent), freqs.shape[0])
    lam = freqs[:m_eff]
    power = ordinates[:m_eff]
    log_lam_mean = float(np.log(lam).mean())

    def objective(d: float) -> float:
        g = float(np.mean(lam ** (2.0 * d) * power))
        if not np.isfinite(g) or g <= 0.0:
            return 1e10
        return float(np.log(g) - 2.0 * d * log_lam_mean)

    result = minimize_scalar(objective, bounds=(-_D_MAX, _D_MAX), method="bounded")
    return float(result.x), m_eff


def gph_d(
    y: npt.NDArray[np.float64], m: int | None = None, exponent: float = 0.5
) -> tuple[float, float, int]:
    """Geweke-Porter-Hudak log-periodogram estimate of ``d``.

    Args:
        y: The series.
        m: Explicit bandwidth, or ``None`` for the default rule.
        exponent: Exponent in the default bandwidth rule.

    Returns:
        A tuple ``(d_hat, se, m_eff)``.

    Raises:
        NumericalError: If the periodogram has non-positive ordinates, or the
            regressor has zero variance.
    """
    from ..exceptions import NumericalError

    freqs, ordinates = periodogram(y)
    m_eff = min(bandwidth(y.shape[0], m, exponent), freqs.shape[0])
    lam = freqs[:m_eff]
    power = ordinates[:m_eff]
    if np.any(power <= 0.0):
        raise NumericalError("periodogram has non-positive ordinates; cannot take logs.")
    regressor = -2.0 * np.log(2.0 * np.sin(lam / 2.0))
    centered = regressor - regressor.mean()
    denom = float(centered @ centered)
    if denom <= 0.0:
        raise NumericalError("degenerate GPH regression (zero regressor variance).")
    response = np.log(power)
    d_hat = float(centered @ (response - response.mean()) / denom)
    se = float(np.pi / np.sqrt(24.0 * denom))
    return d_hat, se, m_eff
