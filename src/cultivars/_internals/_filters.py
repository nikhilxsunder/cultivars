# filepath: /src/cultivars/_internals/_filters.py
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

from .._core import (
    _ROW_SUM_ATOL,
    _TINY,
    ergodic_distribution,
    validate_transition,
)
from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._results import _HamiltonFilterResult


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
