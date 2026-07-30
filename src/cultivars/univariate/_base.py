# filepath: /src/cultivars/univariate/_base.py
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
"""Shared internals for the univariate family.

Holds machinery reused across univariate models so that individual model
modules stay focused on their estimator: the stationarity/invertibility
reparameterization (Monahan 1984 / Durbin-Levinson) and information criteria.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt


def pacf_to_coeffs(pacf: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Durbin-Levinson map from partial autocorrelations in (-1, 1) to the
    coefficients of a stationary/invertible polynomial ``1 - a_1 L - ...``."""
    p = pacf.shape[0]
    out = np.zeros(p, dtype=np.float64)
    for k in range(p):
        rk = pacf[k]
        new = out.copy()
        new[k] = rk
        for j in range(k):
            new[j] = out[j] - rk * out[k - 1 - j]
        out = new
    return out


def coeffs_to_pacf(coeffs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Inverse of :func:`pacf_to_coeffs` (reverse Durbin-Levinson)."""
    p = coeffs.shape[0]
    work = coeffs.astype(np.float64).copy()
    pacf = np.zeros(p, dtype=np.float64)
    for k in range(p - 1, -1, -1):
        rk = work[k]
        pacf[k] = rk
        if k > 0:
            denom = 1.0 - rk * rk
            if abs(denom) < 1e-12:
                denom = 1e-12
            prev = np.empty(k, dtype=np.float64)
            for j in range(k):
                prev[j] = (work[j] + rk * work[k - 1 - j]) / denom
            work[:k] = prev
    return pacf


class InformationCriteria(NamedTuple):
    """Model-selection criteria.

    Attributes:
        aic: Akaike information criterion.
        bic: Bayesian (Schwarz) information criterion.
        hqic: Hannan-Quinn information criterion.
    """

    aic: float
    bic: float
    hqic: float


def information_criteria(llf: float, nobs: int, n_params: int) -> InformationCriteria:
    """Compute AIC/BIC/HQIC from a log-likelihood and parameter count."""
    aic = -2.0 * llf + 2.0 * n_params
    bic = -2.0 * llf + n_params * np.log(nobs)
    hqic = -2.0 * llf + 2.0 * n_params * np.log(np.log(nobs))
    return InformationCriteria(float(aic), float(bic), float(hqic))