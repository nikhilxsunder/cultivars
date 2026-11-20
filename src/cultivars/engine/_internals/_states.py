# filepath: /src/cultivars/engine/_internals/_states.py
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
"""EM state records: what one run of a Markov-switching EM leaves behind.

Two records, one for the switching autoregression and one for the switching
VAR. Each holds the parameters at the last iteration, the three sets of
regime probabilities from the final E-step, the log likelihood, and whether
the run converged. The estimation engines run several starts and compare
these records to choose among them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "_ExpectationMaximizationState",
    "_VectorExpectationMaximizationState",
]


@dataclass(frozen=True, slots=True)
class _ExpectationMaximizationState:
    """Internal state carried between EM restarts.

    The end point of one EM run for a Markov-switching autoregression with
    :math:`K` regimes, order ``p``, and an effective sample of :math:`T` periods.

    See Also:
        * :meth:`._MarkovSwitchingModel._run_em` -- returns an instance.
    """

    transition: npt.NDArray[np.float64]
    """Row-stochastic transition matrix, ``(K, K)``."""
    intercepts: npt.NDArray[np.float64]
    """Regime intercepts, ``(K,)``; equal across regimes when the mean does not switch."""
    ar_params: npt.NDArray[np.float64]
    """Regime autoregressive coefficients, ``(K, p)``; equal rows when they do not switch."""
    sigma2: npt.NDArray[np.float64]
    """Regime innovation variances, ``(K,)``; equal when the variance does not switch."""
    filtered_prob: npt.NDArray[np.float64]
    r"""Filtered regime probabilities :math:`\Pr(S_t \mid y_{1:t})`, ``(T, K)``."""
    predicted_prob: npt.NDArray[np.float64]
    r"""One-step-predicted regime probabilities :math:`\Pr(S_t \mid y_{1:t-1})`, ``(T, K)``."""
    smoothed_prob: npt.NDArray[np.float64]
    r"""Smoothed regime probabilities :math:`\Pr(S_t \mid y_{1:T})`, ``(T, K)``."""
    llf: float
    """Log likelihood at the last completed iteration."""
    n_iter: int
    """EM iterations run."""
    converged: bool
    """Whether the run met its convergence tolerance before the iteration cap."""


@dataclass(frozen=True, kw_only=True)
class _VectorExpectationMaximizationState:
    """Internal state carried between EM restarts of a switching VAR.

    The end point of one EM run with :math:`M` regimes and :math:`k` variables.

    See Also:
        * :meth:`._MarkovSwitchingVectorAutoRegressionModel._run_em` -- returns
          an instance.
    """

    transition: npt.NDArray[np.float64]
    """Row-stochastic transition matrix, ``(M, M)``."""
    coefficients: npt.NDArray[np.float64]
    """Regime coefficient matrices, regimes on the leading axis."""
    sigmas: npt.NDArray[np.float64]
    """Regime innovation covariances, ``(M, k, k)``."""
    filtered_prob: npt.NDArray[np.float64]
    """Filtered regime probabilities, ``(T, M)``."""
    predicted_prob: npt.NDArray[np.float64]
    """One-step-predicted regime probabilities, ``(T, M)``."""
    smoothed_prob: npt.NDArray[np.float64]
    """Smoothed regime probabilities, ``(T, M)``."""
    llf: float
    """Log likelihood at the last completed iteration."""
    n_iter: int
    """EM iterations run."""
    converged: bool
    """Whether the run met its convergence tolerance before the iteration cap."""
