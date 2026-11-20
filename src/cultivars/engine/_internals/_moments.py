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
"""Moment records: what a multivariate regression yields before a family names it.

Two records. :class:`_VectorMoments` is the output of a multivariate
least-squares fit: coefficients, residuals, both residual covariances, and
the concentrated Gaussian log likelihood. Every reduced-form vector family
passes through it between building its design and slicing its coefficients
apart. :class:`_CointegrationMoments` is the output of the reduced-rank
eigenvalue problem behind Johansen's procedure, held apart from any fit
because it does not depend on the cointegrating rank.

References:
    Johansen, S. (1991). Estimation and hypothesis testing of cointegration
    vectors in Gaussian vector autoregressive models. *Econometrica*,
    59(6), 1551-1580.
"""

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

    See Also:
        * :meth:`._BayesianVectorAutoRegressionModel._gaussian_moments` --
          returns an instance.
        * :meth:`._ExogenousVectorAutoRegressionModel._gaussian_moments` --
          returns an instance.
        * :meth:`._ExogenousVectorErrorCorrectionModel._gaussian_moments` --
          returns an instance.
        * :meth:`._FunctionalCoefficientVectorAutoRegressionModel._gaussian_moments`
          -- returns an instance.
    """

    coef: npt.NDArray[np.float64]
    """Coefficient matrix as estimated, ``(width, k)``: design columns by equations."""
    resid: npt.NDArray[np.float64]
    """Residuals, ``(nobs, k)``, aligned with the design rows."""
    fittedvalues: npt.NDArray[np.float64]
    """Fitted values ``design @ coef``, ``(nobs, k)``."""
    sigma_u: npt.NDArray[np.float64]
    """Residual covariance with the degrees-of-freedom correction, ``(k, k)``."""
    sigma_ml: npt.NDArray[np.float64]
    """Residual covariance divided by the sample size, ``(k, k)``; the maximum-likelihood one."""
    llf: float
    """Gaussian log likelihood, concentrated over ``sigma_ml``."""
    nobs: int
    """Number of design rows."""
    width: int
    """Number of design columns, the regressors of each equation."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _CointegrationMoments:
    r"""The reduced-rank problem's solution, before a rank is chosen.

    Holds what the eigenvalue problem produces plus the three blocks the
    short-run regression is built from. Separated from the fit because none of
    it depends on the rank: every candidate rank reads the same eigenvectors,
    which is what makes a rank test cheap once the decomposition is done.

    Attributes:
        eigenvalues: Squared canonical correlations, descending.
        eigenvectors: Candidate cointegrating vectors, normalized so that
            :math:`\beta^\top S_{11} \beta` is the identity. Columns match
            :attr:`eigenvalues`.
        levels: The lagged levels block, with any restricted deterministic
            column appended.
        differences: The contemporaneous differences being explained.
        short_run: Unrestricted deterministic terms and lagged differences.
        s00: Residual second moment of the differences.
        nobs: Effective sample.

    See Also:
        * :meth:`._ExogenousVectorErrorCorrectionModel._cointegration_moments`
          -- returns an instance.
        * :meth:`._VectorErrorCorrectionModel._cointegration_moments` --
          returns an instance.
    """

    eigenvalues: npt.NDArray[np.float64]
    """Squared canonical correlations between differences and lagged levels, descending."""
    eigenvectors: npt.NDArray[np.float64]
    r"""Candidate cointegrating vectors, one per column, in the order of ``eigenvalues``.

    Normalized so that :math:`\beta^\top S_{11} \beta` is the identity.
    """
    levels: npt.NDArray[np.float64]
    """The lagged-levels block, with any restricted deterministic column appended."""
    differences: npt.NDArray[np.float64]
    """The contemporaneous differences being explained."""
    short_run: npt.NDArray[np.float64]
    """Unrestricted deterministic terms and lagged differences."""
    s00: npt.NDArray[np.float64]
    """Residual second-moment matrix of the differences."""
    nobs: int
    """Effective sample size."""
