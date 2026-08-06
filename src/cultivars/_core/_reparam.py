# filepath: /src/cultivars/_core/_reparam.py
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

"""Unconstrained reparameterization of stationary autoregressive coefficients.

Optimizers work on the real line; stationarity is a constraint on the
companion eigenvalues. The Monahan mapping resolves this: any vector in
``R**p`` maps through ``tanh`` to partial autocorrelations in ``(-1, 1)``,
and the Durbin-Levinson recursion maps those to AR coefficients that are
stationary by construction. The optimizer therefore never proposes an
explosive parameter vector, and no penalty term is needed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._defaults import _PACF_CLIP


def pacf_to_coeffs(pacf: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Map partial autocorrelations to AR coefficients (Durbin-Levinson).

    Args:
        pacf: Partial autocorrelations, each in ``(-1, 1)``.

    Returns:
        The implied AR coefficients ``phi_1, ..., phi_p``.

    Example:
        >>> np.round(pacf_to_coeffs(np.array([0.5, -0.3])), 4)
        array([ 0.65, -0.3 ])
    """
    p = pacf.shape[0]
    out = np.zeros(p, dtype=np.float64)
    for k in range(p):
        r = pacf[k]
        updated = out.copy()
        updated[k] = r
        for j in range(k):
            updated[j] = out[j] - r * out[k - 1 - j]
        out = updated
    return out


def coeffs_to_pacf(coeffs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Map AR coefficients back to partial autocorrelations (inverse recursion).

    Args:
        coeffs: AR coefficients ``phi_1, ..., phi_p``.

    Returns:
        The implied partial autocorrelations.
    """
    p = coeffs.shape[0]
    work = np.asarray(coeffs, dtype=np.float64).copy()
    pacf = np.zeros(p, dtype=np.float64)
    for k in range(p - 1, -1, -1):
        r = work[k]
        pacf[k] = r
        if k > 0:
            denom = 1.0 - r * r
            if abs(denom) < 1e-12:
                denom = 1e-12
            previous = np.empty(k, dtype=np.float64)
            for j in range(k):
                previous[j] = (work[j] + r * work[k - 1 - j]) / denom
            work[:k] = previous
    return pacf


def pack_stationary(coeffs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Map stationary AR coefficients to unconstrained optimizer coordinates.

    Args:
        coeffs: Stationary AR coefficients.

    Returns:
        Unconstrained values in ``R**p``.
    """
    if coeffs.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.arctanh(np.clip(coeffs_to_pacf(coeffs), -_PACF_CLIP, _PACF_CLIP))


def unpack_stationary(psi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Map unconstrained optimizer coordinates to stationary AR coefficients.

    Args:
        psi: Unconstrained values in ``R**p``.

    Returns:
        AR coefficients that are stationary by construction.

    Example:
        >>> phi = np.array([0.5, -0.3])
        >>> bool(np.allclose(unpack_stationary(pack_stationary(phi)), phi))
        True
    """
    if psi.size == 0:
        return np.zeros(0, dtype=np.float64)
    return pacf_to_coeffs(np.tanh(psi))
