# filepath: /src/cultivars/state_space/regime_switching.py
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
"""Discrete Markov-switching inference: the Hamilton filter and Kim smoother.

Regime-switching models in cultivars separate two concerns:

- The *regime chain* — a first-order ``K``-state Markov process with transition
  matrix ``P`` — and the recursions that infer regime probabilities from a
  sequence of per-regime conditional data densities. The Hamilton (1989) forward
  filter and the Kim (1994) backward smoother are generic: they do not know what
  model produced the densities. They live here.
- The *observation model* — how ``Pr(y_t | S_t = j, past)`` is formed — is
  model-specific (a switching AR mean, a switching variance, a switching VAR).
  Each model supplies the ``(T, K)`` matrix of log conditional densities and
  consumes the filter/smoother output. See :mod:`cultivars.univariate.ms_ar`.

This factoring is what lets one filter serve MS-AR today and MS-VAR tomorrow:
only the density map changes. A future continuous-state
``RegimeSwitchingStateSpace`` — the Kim-Nelson filter, which runs one Kalman
filter per regime and collapses the state across regimes at each step — layers
directly on top of these same two recursions.

Conventions:

- The transition matrix is **row-stochastic**:
  ``transition[i, j] = Pr(S_t = j | S_{t-1} = i)`` and each row sums to one.
- Regime-probability vectors are length ``K`` and the one-step-ahead prediction
  is ``xi_pred = xi_filt @ transition``.
- Conditional densities are passed in **logarithms** so that near-zero regime
  likelihoods (small variances, outliers) do not underflow; the filter combines
  them with a log-sum-exp.

References:
    Hamilton, J. D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. *Econometrica*, 57(2).
    Hamilton, J. D. (1994). *Time Series Analysis*, ch. 22.
    Kim, C.-J. (1994). Dynamic linear models with Markov-switching.
    *Journal of Econometrics*, 60(1-2).
    Kim, C.-J. & Nelson, C. R. (1999). *State-Space Models with Regime
    Switching*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError, NumericalError, SpecificationError

# --------------------------------------------------------------------------
# Transition-matrix utilities
# --------------------------------------------------------------------------

def validate_transition(transition: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Validate and return a row-stochastic transition matrix.

    Args:
        transition: A ``(K, K)`` array-like with non-negative rows that each sum
            to one, where ``transition[i, j] = Pr(S_t = j | S_{t-1} = i)``.

    Returns:
        The validated matrix as a ``float64`` array.

    Raises:
        DimensionError: If ``transition`` is not a square 2-D array.
        SpecificationError: If any entry is negative or a row does not sum to one.
        NumericalError: If ``transition`` contains non-finite values.
    """
    p = np.asarray(transition, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise DimensionError(
            f"transition must be a square (K, K) matrix; got shape {p.shape}."
        )
    if p.shape[0] < 1:
        raise SpecificationError("transition must have at least one regime.")
    if not np.all(np.isfinite(p)):
        raise NumericalError("transition contains non-finite values.")
    if np.any(p < -_ROW_SUM_ATOL):
        raise SpecificationError("transition contains negative probabilities.")
    row_sums = p.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=_ROW_SUM_ATOL):
        raise SpecificationError(
            f"transition rows must sum to 1; got row sums {np.round(row_sums, 6)}."
        )
    return p


def ergodic_distribution(transition: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the stationary (ergodic) distribution of a Markov chain.

    Solves ``pi = pi @ P`` subject to ``sum(pi) = 1`` for the row-stochastic
    transition matrix ``P``. For an irreducible chain this is unique; for a
    reducible chain it returns one valid stationary distribution.

    Args:
        transition: A row-stochastic ``(K, K)`` transition matrix.

    Returns:
        The stationary distribution as a length-``K`` probability vector.

    Raises:
        NumericalError: If no non-negative stationary distribution can be formed.

    Example:
        >>> p = np.array([[0.9, 0.1], [0.2, 0.8]])
        >>> np.round(ergodic_distribution(p), 4)
        array([0.6667, 0.3333])
    """
    p = validate_transition(transition)
    k = p.shape[0]
    # Stack (P' - I) pi = 0 with the normalization sum(pi) = 1 and solve in the
    # least-squares sense so the system is well posed even at a defective P.
    a = np.vstack([p.T - np.eye(k), np.ones(k)])
    b = np.concatenate([np.zeros(k), [1.0]])
    pi, *_ = np.linalg.lstsq(a, b, rcond=None)
    pi = np.clip(pi, 0.0, None)
    total = pi.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise NumericalError("failed to compute a valid ergodic distribution.")
    return pi / total


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class HamiltonFilterResult:
    """Output of the Hamilton forward filter.

    Attributes:
        filtered_prob: Contemporaneous regime probabilities
            ``Pr(S_t = j | y_{1..t})``, shape ``(T, K)``.
        predicted_prob: One-step-ahead regime probabilities
            ``Pr(S_t = j | y_{1..t-1})``, shape ``(T, K)``.
        loglikelihood: Total log-likelihood ``sum_t log Pr(y_t | y_{1..t-1})``.
        loglikelihood_contributions: Per-period contributions, shape ``(T,)``.
    """

    filtered_prob: npt.NDArray[np.float64]
    predicted_prob: npt.NDArray[np.float64]
    loglikelihood: float
    loglikelihood_contributions: npt.NDArray[np.float64]


@dataclass(frozen=True)
class KimSmootherResult:
    """Output of the Kim backward smoother.

    Attributes:
        smoothed_prob: Full-sample regime probabilities
            ``Pr(S_t = j | y_{1..T})``, shape ``(T, K)``.
        smoothed_joint_prob: Consecutive-pair probabilities
            ``Pr(S_t = i, S_{t+1} = j | y_{1..T})``, shape ``(T - 1, K, K)``;
            entry ``[t]`` links period ``t`` to ``t + 1``. Empty when ``T == 1``.
            These are the expected-transition weights of the EM M-step.
    """

    smoothed_prob: npt.NDArray[np.float64]
    smoothed_joint_prob: npt.NDArray[np.float64]


# --------------------------------------------------------------------------
# Hamilton filter
# --------------------------------------------------------------------------

def hamilton_filter(
    log_conditional_density: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    initial_prob: npt.ArrayLike | None = None,
) -> HamiltonFilterResult:
    """Run the Hamilton forward filter over a discrete regime chain.

    At each period the filter predicts the regime distribution one step forward
    through ``transition``, weights it by the per-regime conditional densities,
    and renormalizes; the normalizing constant is the period's likelihood
    contribution. Densities are supplied in logarithms and combined with a
    log-sum-exp for numerical stability.

    Args:
        log_conditional_density: Log conditional densities
            ``log Pr(y_t | S_t = j, past)``, shape ``(T, K)``. ``-inf`` entries
            (a regime that assigns zero density) are permitted as long as some
            regime is finite at every period.
        transition: Row-stochastic ``(K, K)`` transition matrix.
        initial_prob: Optional initial regime distribution ``Pr(S_1 = j)``,
            length ``K``. Defaults to the ergodic distribution of ``transition``.

    Returns:
        A :class:`HamiltonFilterResult`.

    Raises:
        DimensionError: If shapes are inconsistent.
        SpecificationError: If ``initial_prob`` is not a length-``K`` distribution.
        NumericalError: If every regime density vanishes at some period, or a
            non-finite (NaN) density is supplied.

    Example:
        >>> p = np.array([[0.95, 0.05], [0.10, 0.90]])
        >>> logd = np.log(np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]))
        >>> res = hamilton_filter(logd, p)
        >>> res.filtered_prob.shape
        (3, 2)
        >>> bool(np.allclose(res.filtered_prob.sum(axis=1), 1.0))
        True
    """
    logd = np.asarray(log_conditional_density, dtype=np.float64)
    if logd.ndim != 2:
        raise DimensionError(
            f"log_conditional_density must be 2-D (T, K); got shape {logd.shape}."
        )
    if np.isnan(logd).any():
        raise NumericalError("log_conditional_density contains NaN values.")
    p = validate_transition(transition)
    n, k = logd.shape
    if p.shape[0] != k:
        raise DimensionError(
            f"transition is {p.shape[0]}x{p.shape[0]} but densities imply K={k}."
        )

    if initial_prob is None:
        xi0 = ergodic_distribution(p)
    else:
        xi0 = np.asarray(initial_prob, dtype=np.float64)
        if xi0.shape != (k,):
            raise SpecificationError(
                f"initial_prob must have length K={k}; got shape {xi0.shape}."
            )
        if np.any(xi0 < 0.0) or not np.isclose(xi0.sum(), 1.0, atol=_ROW_SUM_ATOL):
            raise SpecificationError("initial_prob must be a probability vector.")

    filtered = np.empty((n, k), dtype=np.float64)
    predicted = np.empty((n, k), dtype=np.float64)
    contributions = np.empty(n, dtype=np.float64)

    prev_filtered = xi0
    for t in range(n):
        xi_pred = xi0 if t == 0 else prev_filtered @ p
        predicted[t] = xi_pred
        log_joint = np.log(np.clip(xi_pred, _TINY, None)) + logd[t]
        max_log = float(np.max(log_joint))
        if not np.isfinite(max_log):
            raise NumericalError(
                f"all regime densities vanished at period {t}; the model cannot "
                "explain this observation under any regime."
            )
        weights = np.exp(log_joint - max_log)
        denom = float(weights.sum())
        contributions[t] = max_log + np.log(denom)
        xi_filt = weights / denom
        filtered[t] = xi_filt
        prev_filtered = xi_filt

    return HamiltonFilterResult(
        filtered_prob=filtered,
        predicted_prob=predicted,
        loglikelihood=float(contributions.sum()),
        loglikelihood_contributions=contributions,
    )


# --------------------------------------------------------------------------
# Kim smoother
# --------------------------------------------------------------------------

def kim_smoother(
    filter_result: HamiltonFilterResult, transition: npt.ArrayLike
) -> KimSmootherResult:
    """Run the Kim backward smoother given a Hamilton-filter pass.

    Implements Kim's (1994) exact smoother for a discrete chain:

    ``Pr(S_t = i | y_{1..T}) = Pr(S_t = i | y_{1..t})
    * sum_j [ P_ij * Pr(S_{t+1} = j | y_{1..T}) / Pr(S_{t+1} = j | y_{1..t}) ]``.

    Args:
        filter_result: The output of :func:`hamilton_filter` on the same data.
        transition: The same row-stochastic ``(K, K)`` transition matrix used
            for filtering.

    Returns:
        A :class:`KimSmootherResult` with smoothed marginal and consecutive-pair
        regime probabilities.

    Raises:
        DimensionError: If ``transition`` is not conformable with the filter.

    Example:
        >>> p = np.array([[0.95, 0.05], [0.10, 0.90]])
        >>> logd = np.log(np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]))
        >>> sm = kim_smoother(hamilton_filter(logd, p), p)
        >>> bool(np.allclose(sm.smoothed_prob.sum(axis=1), 1.0))
        True
    """
    p = validate_transition(transition)
    filtered = filter_result.filtered_prob
    predicted = filter_result.predicted_prob
    n, k = filtered.shape
    if p.shape[0] != k:
        raise DimensionError(
            f"transition is {p.shape[0]}x{p.shape[0]} but filter implies K={k}."
        )

    smoothed = np.empty((n, k), dtype=np.float64)
    smoothed[-1] = filtered[-1]
    joint = np.zeros((max(n - 1, 0), k, k), dtype=np.float64)

    for t in range(n - 2, -1, -1):
        pred_next = np.clip(predicted[t + 1], _TINY, None)
        ratio = smoothed[t + 1] / pred_next               # length K over j
        # Pr(S_t=i, S_{t+1}=j | Y_T) = filt[t, i] * P_ij * ratio_j.
        joint_t = filtered[t][:, None] * p * ratio[None, :]
        joint[t] = joint_t
        smoothed[t] = joint_t.sum(axis=1)
        total = smoothed[t].sum()
        if total > 0.0:
            smoothed[t] /= total

    return KimSmootherResult(smoothed_prob=smoothed, smoothed_joint_prob=joint)
