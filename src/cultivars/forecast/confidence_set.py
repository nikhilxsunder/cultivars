# filepath: /src/cultivars/forecast/confidence_set.py
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

"""The model confidence set: which forecasters the data cannot rank below the best.

A pairwise test says whether one forecaster beats another. With many
forecasters the question is different -- which of them could be the best
-- and answering it with pairwise tests multiplies the size distortion.
Hansen, Lunde and Nason (2011) answer it the way a confidence interval
answers a point estimate: a *set* of models that contains the best one
with probability at least the confidence level, built by eliminating the
worst model while the hypothesis that the remaining ones are equally good
is rejected, with the null distribution taken from a stationary bootstrap
of the loss panel so that the serial correlation of forecast losses is
respected.

The set is the honest multi-model answer, in the same spirit as the
set-identified impulse responses elsewhere in the package: on a short or
noisy evaluation window it keeps many models, and that is the finding.

References:
    Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence
        set. *Econometrica*, 79(2), 453-497.
    Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
        *Journal of the American Statistical Association*, 89(428),
        1303-1313.

Example:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> losses = rng.standard_normal((150, 3)) ** 2
    >>> losses[:, 2] += 1.0
    >>> mcs = model_confidence_set(losses, names=("A", "B", "C"), n_bootstrap=300, seed=0)
    >>> "C" in mcs.included, mcs.pvalue("C") < 0.1
    (False, True)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .._core import (
    _MCS_ALPHA,
    _MCS_BOOTSTRAP,
    _model_confidence_set,
    _validate_names,
    _variable_names,
)
from .._internals import _ModelConfidenceSetSelection as ModelConfidenceSetSelection
from ..exceptions import DimensionError, NumericalError, SpecificationError

__all__ = ["ModelConfidenceSetSelection", "model_confidence_set"]


def model_confidence_set(
    losses: npt.ArrayLike,
    *,
    names: Sequence[str] | None = None,
    alpha: float = _MCS_ALPHA,
    statistic: str = "range",
    n_bootstrap: int = _MCS_BOOTSTRAP,
    block_length: float | None = None,
    seed: int | np.random.Generator | None = None,
) -> ModelConfidenceSetSelection:
    """Build the model confidence set from aligned loss series.

    Args:
        losses: ``(T, M)`` losses, one column per model and one row per
            evaluation origin, negatively oriented and aligned origin by
            origin -- ``BacktestResult.losses(...)`` from each model's
            backtest on the same schedule, stacked as columns.
        names: Model labels; default ``y1 .. yM``.
        alpha: Level of the set; the default gives the 90% set the authors
            report.
        statistic: ``"range"`` (largest studentized pairwise differential)
            or ``"max"`` (largest studentized deviation from the set
            average).
        n_bootstrap: Stationary-bootstrap replications.
        block_length: Expected block length of the bootstrap; default
            ``max(2, round(T ** (1 / 3)))``.
        seed: Seed or generator.

    Returns:
        The :class:`ModelConfidenceSetSelection`.

    Raises:
        DimensionError: If the panel is not ``(T, M)`` or the labels do
            not match.
        SpecificationError: If the level, statistic, or counts are
            unusable, or there are fewer than two models or origins.
        NumericalError: If a loss is not finite.
    """
    panel = np.asarray(losses, dtype=np.float64)
    if panel.ndim != 2:
        raise DimensionError(f"losses must be (T, M); got {panel.ndim}-D.")
    if not np.all(np.isfinite(panel)):
        raise NumericalError("losses must be finite.")
    count, n_models = panel.shape
    if n_models < 2:
        raise SpecificationError("a model confidence set needs at least two models.")
    if count < 2:
        raise SpecificationError("a model confidence set needs at least two evaluation origins.")
    if not 0.0 < alpha < 1.0:
        raise SpecificationError(f"alpha must lie strictly inside (0, 1); got {alpha}.")
    if n_bootstrap < 100:
        raise SpecificationError(f"n_bootstrap must be at least 100; got {n_bootstrap}.")
    labels = _validate_names(names, _variable_names(None, n_models))
    if len(labels) != n_models:
        raise DimensionError(f"{len(labels)} names for {n_models} models.")
    if len(set(labels)) != n_models:
        raise SpecificationError(f"model names must be distinct; got {labels}.")
    resolved_block = (
        float(max(2, round(count ** (1.0 / 3.0)))) if block_length is None else float(block_length)
    )
    if resolved_block < 1.0:
        raise SpecificationError(f"block_length must be at least 1; got {block_length}.")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    order, pvalues = _model_confidence_set(
        panel,
        alpha=alpha,
        n_bootstrap=int(n_bootstrap),
        block_length=resolved_block,
        statistic=statistic,
        rng=rng,
    )
    return ModelConfidenceSetSelection(
        names=labels,
        pvalues=pvalues,
        mean_losses=np.asarray(panel.mean(axis=0), dtype=np.float64),
        elimination_order=tuple(int(i) for i in order),
        alpha=float(alpha),
        statistic=statistic,
        n_bootstrap=int(n_bootstrap),
        block_length=resolved_block,
        nobs=int(count),
    )
