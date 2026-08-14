# filepath: /src/cultivars/_internals/_smoothers.py
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
    _TINY,
    validate_transition,
)
from ..exceptions import DimensionError
from ._results import _HamiltonFilterResult, _KimSmootherResult


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

