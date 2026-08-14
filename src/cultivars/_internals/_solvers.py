# filepath: /src/cultivars/_internals/_solvers.py
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

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from .._core import (
    _ROW_SUM_ATOL,
    _TINY,
    ergodic_distribution,
    validate_transition,
)
from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._objectives import _Objective
from ._results import _HamiltonFilterResult, _KimSmootherResult


def _solve[P](objective: _Objective[P]) -> tuple[P, float]:
    """Minimize an objective from every starting point and keep the best.

    Args:
        objective: The surface to minimize. Its ``method`` and ``options``
            select the algorithm; its ``starts`` supply the initial points.

    Returns:
        A tuple ``(parameters, criterion)`` where ``criterion`` is the minimized
        value of :meth:`_Objective.__call__` at the winning start.

    Raises:
        NumericalError: If ``starts`` yields no points at all.
    """
    best_x: npt.NDArray[np.float64] | None = None
    best_f = np.inf
    for theta0 in objective.starts():
        result = minimize(objective, theta0, method=objective.method, options=objective.options)
        if float(result.fun) < best_f:
            best_f = float(result.fun)
            best_x = np.asarray(result.x, dtype=np.float64)
    if best_x is None:
        raise NumericalError("objective supplied no starting points.")
    return objective.unpack(best_x), best_f


def _maximize_likelihood[P](objective: _Objective[P]) -> tuple[P, float]:
    """Solve a negative-log-likelihood objective, returning the log-likelihood.

    A thin sign-flip over :func:`_solve`, kept as a separate name so that no
    call site has to remember which surfaces are negated and which are not.

    Args:
        objective: A surface whose criterion is a negative log-likelihood.

    Returns:
        A tuple ``(parameters, llf)`` with the maximized log-likelihood.
    """
    parameters, criterion = _solve(objective)
    return parameters, -criterion


def hamilton_filter(
    log_conditional_density: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    initial_prob: npt.ArrayLike | None = None,
) -> _HamiltonFilterResult:
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
        A :class:`_HamiltonFilterResult`.

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
        raise DimensionError(f"log_conditional_density must be 2-D (T, K); got shape {logd.shape}.")
    if np.isnan(logd).any():
        raise NumericalError("log_conditional_density contains NaN values.")
    n, k = logd.shape
    p = validate_transition(transition, k)

    if initial_prob is None:
        xi0 = ergodic_distribution(p)
    else:
        xi0 = np.asarray(initial_prob, dtype=np.float64)
        if xi0.shape != (k,):
            raise SpecificationError(f"initial_prob must have length K={k}; got shape {xi0.shape}.")
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

    return _HamiltonFilterResult(
        filtered_prob=filtered,
        predicted_prob=predicted,
        loglikelihood=float(contributions.sum()),
        loglikelihood_contributions=contributions,
    )


def kim_smoother(
    filter_result: _HamiltonFilterResult, transition: npt.ArrayLike
) -> _KimSmootherResult:
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
    filtered = filter_result.filtered_prob
    predicted = filter_result.predicted_prob
    n, k = filtered.shape
    p = validate_transition(transition, k)
    if p.shape[0] != k:
        raise DimensionError(f"transition is {p.shape[0]}x{p.shape[0]} but filter implies K={k}.")

    smoothed = np.empty((n, k), dtype=np.float64)
    smoothed[-1] = filtered[-1]
    joint = np.zeros((max(n - 1, 0), k, k), dtype=np.float64)

    for t in range(n - 2, -1, -1):
        pred_next = np.clip(predicted[t + 1], _TINY, None)
        ratio = smoothed[t + 1] / pred_next  # length K over j
        # Pr(S_t=i, S_{t+1}=j | Y_T) = filt[t, i] * P_ij * ratio_j.
        joint_t = filtered[t][:, None] * p * ratio[None, :]
        joint[t] = joint_t
        smoothed[t] = joint_t.sum(axis=1)
        total = smoothed[t].sum()
        if total > 0.0:
            smoothed[t] /= total

    return _KimSmootherResult(smoothed_prob=smoothed, smoothed_joint_prob=joint)
