# filepath: /src/cultivars/engine/_internals/_moments.py
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
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["_CointegrationMoments", "_VectorMoments"]


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorMoments:
    """Everything a multivariate least-squares fit yields before it is named.

    The families in this group differ in how they build a design matrix and in
    how they slice the coefficients back apart. They do not differ in what
    happens between those two steps, and holding that middle as one record is
    what keeps the concentrated Gaussian likelihood written once. A family that
    reimplemented it would eventually disagree with its siblings by a
    degrees-of-freedom convention, which is exactly the kind of divergence that
    does not announce itself.

    Attributes:
        coef: The ``(width, k)`` coefficient matrix as estimated.
        resid: Residuals aligned with the design rows.
        fittedvalues: ``design @ coef``, computed rather than differenced.
        sigma_u: Residual covariance with the degrees-of-freedom correction.
        sigma_ml: Residual covariance divided by the sample size.
        llf: Gaussian log-likelihood concentrated over ``sigma_ml``.
        nobs: Design rows.
        width: Design columns, that is regressors per equation.
    """

    coef: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    sigma_u: npt.NDArray[np.float64]
    sigma_ml: npt.NDArray[np.float64]
    llf: float
    nobs: int
    width: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _CointegrationMoments:
    """The reduced-rank problem's solution, before a rank is chosen.

    Holds what the eigenvalue problem produces plus the three blocks the
    short-run regression is built from. Separated from the fit because none of
    it depends on the rank: every candidate rank reads the same eigenvectors,
    which is what makes a rank test cheap once the decomposition is done.

    Attributes:
        eigenvalues: Squared canonical correlations, descending.
        eigenvectors: Candidate cointegrating vectors, normalized so that
            ``beta' S11 beta`` is the identity. Columns match
            :attr:`eigenvalues`.
        levels: The lagged levels block, with any restricted deterministic
            column appended.
        differences: The contemporaneous differences being explained.
        short_run: Unrestricted deterministic terms and lagged differences.
        s00: Residual second moment of the differences.
        nobs: Effective sample.
    """

    eigenvalues: npt.NDArray[np.float64]
    eigenvectors: npt.NDArray[np.float64]
    levels: npt.NDArray[np.float64]
    differences: npt.NDArray[np.float64]
    short_run: npt.NDArray[np.float64]
    s00: npt.NDArray[np.float64]
    nobs: int
