# filepath: /src/cultivars/engine/_internals/_filters.py
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
r"""The Hamilton filter: forward regime probabilities for a discrete Markov chain.

One function, :func:`_hamilton_filter`. It takes the per-regime log
conditional densities of the observations as a ``(T, K)`` array, with the
transition matrix, and returns the filtered and one-step-predicted regime
probabilities and the log likelihood. It knows nothing about what produced
the densities. A switching autoregression and a switching VAR both reduce
to that array, which is why the filter is written once here and not inside
either model. The backward pass is
:func:`~cultivars.engine._internals._smoothers._kim_smoother`.

In symbols, with :math:`P_{ij} = \Pr(S_t = j \mid S_{t-1} = i)`, the vector
:math:`f(y_t)` of regime-conditional densities and :math:`\odot` the
elementwise product, one step of the recursion and the likelihood it
accumulates are

.. math::

   \xi_{t \mid t-1} = P^\top \xi_{t-1 \mid t-1}, \qquad
   \xi_{t \mid t} = \frac{\xi_{t \mid t-1} \odot f(y_t)}
   {\mathbf{1}^\top \bigl( \xi_{t \mid t-1} \odot f(y_t) \bigr)}, \qquad
   \log L = \sum_{t} \log \mathbf{1}^\top \bigl( \xi_{t \mid t-1} \odot f(y_t) \bigr).

The two probability paths are returned as a
:class:`~cultivars.engine._internals._results._HamiltonFilterResult`.

References:
    Hamilton, J. D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. *Econometrica*,
    57(2), 357-384.
"""

import numpy as np
import numpy.typing as npt

from ...exceptions import DimensionError, NumericalError, SpecificationError
from .._core._defaults import (
    _ROW_SUM_ATOL,
    _TINY,
)
from .._core._estimators import (
    _ergodic_distribution,
)
from .._core._validators import (
    _validate_transition,
)
from ._results import _HamiltonFilterResult

__all__ = [
    "_hamilton_filter",
]


def _hamilton_filter(
    log_conditional_density: npt.ArrayLike,
    transition: npt.ArrayLike,
    *,
    initial_prob: npt.ArrayLike | None = None,
) -> _HamiltonFilterResult:
    r"""Run the Hamilton forward filter over a discrete regime chain.

    At each period the filter predicts the regime distribution one step forward
    through ``transition``, weights it by the per-regime conditional densities,
    and renormalizes; the normalizing constant is the period's likelihood
    contribution. Densities are supplied in logarithms and combined with a
    log-sum-exp for numerical stability.

    Args:
        log_conditional_density: Log conditional densities
            :math:`\log \Pr(y_t \mid S_t = j, \text{past})`, shape ``(T, K)``. ``-inf`` entries
            (a regime that assigns zero density) are permitted as long as some
            regime is finite at every period. ``+inf`` is not: it is
            reported with the same error as a period in which every density
            vanished.
        transition: Row-stochastic ``(K, K)`` transition matrix.
        initial_prob: Optional initial regime distribution :math:`\Pr(S_1 = j)`,
            length :math:`K`. Defaults to the ergodic distribution of ``transition``.

    Returns:
        A :class:`~cultivars.engine._internals._results._HamiltonFilterResult`.

    Raises:
        DimensionError: If shapes are inconsistent.
        SpecificationError: If ``initial_prob`` is not a length-:math:`K` distribution.
        NumericalError: If every regime density vanishes at some period, or a
            non-finite (NaN) density is supplied.

    Example:
        >>> p = np.array([[0.95, 0.05], [0.10, 0.90]])
        >>> logd = np.log(np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]))
        >>> res = _hamilton_filter(logd, p)
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
    p = _validate_transition(transition, k)

    if initial_prob is None:
        xi0 = _ergodic_distribution(p)
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
