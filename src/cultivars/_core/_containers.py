# filepath: /src/cultivars/_core/_containers.py
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

"""Reusable value objects shared by internal mechanisms and public results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..exceptions import SpecificationError


class InformationCriteria(NamedTuple):
    """Model-selection criteria computed from a fitted log-likelihood.

    Attributes:
        aic: Akaike information criterion, ``-2 * llf + 2 * k``.
        bic: Bayesian (Schwarz) criterion, ``-2 * llf + k * log(n)``.
        hqic: Hannan-Quinn criterion, ``-2 * llf + 2 * k * log(log(n))``.
    """

    aic: float
    bic: float
    hqic: float


def information_criteria(llf: float, nobs: int, n_params: int) -> InformationCriteria:
    """Compute all three information criteria from a fit summary.

    Args:
        llf: Maximized log-likelihood.
        nobs: Number of observations the likelihood was evaluated on.
        n_params: Number of free parameters, including the innovation variance.

    Returns:
        The populated :class:`InformationCriteria`.

    Raises:
        SpecificationError: If ``nobs < 3`` (``log(log(n))`` is undefined or
            negative below that) or ``n_params`` is negative.

    Example:
        >>> ic = information_criteria(-100.0, 200, 3)
        >>> round(ic.aic, 2)
        206.0
    """
    if nobs < 3:
        raise SpecificationError(f"nobs must be >= 3 to form criteria; got {nobs}.")
    if n_params < 0:
        raise SpecificationError(f"n_params must be non-negative; got {n_params}.")
    penalty = 2.0 * n_params
    return InformationCriteria(
        aic=float(-2.0 * llf + penalty),
        bic=float(-2.0 * llf + n_params * np.log(nobs)),
        hqic=float(-2.0 * llf + penalty * np.log(np.log(nobs))),
    )


@dataclass(frozen=True, slots=True)
class _Layout:
    """Column bookkeeping for the responsibility-weighted regression.

    Switching and non-switching blocks share one design matrix; this object
    is the single source of truth for which column belongs to which regime,
    so the M-step never has to recompute offsets inline.
    """

    n_regimes: int
    order: int
    switching_mean: bool
    switching_ar: bool

    @property
    def n_intercept(self) -> int:
        """Number of intercept columns."""
        return self.n_regimes if self.switching_mean else 1

    @property
    def n_ar(self) -> int:
        """Number of autoregressive columns."""
        return self.n_regimes * self.order if self.switching_ar else self.order

    @property
    def width(self) -> int:
        """Total design width."""
        return self.n_intercept + self.n_ar

    def intercept_col(self, regime: int) -> int:
        """Index of the intercept column serving ``regime``."""
        return regime if self.switching_mean else 0

    def ar_slice(self, regime: int) -> slice:
        """Column slice of the AR block serving ``regime``."""
        base = self.n_intercept
        if self.switching_ar:
            return slice(base + regime * self.order, base + (regime + 1) * self.order)
        return slice(base, base + self.order)
