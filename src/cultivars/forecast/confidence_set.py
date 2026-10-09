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
r"""The model confidence set: which forecasters the data cannot rank below the best.

A pairwise test says whether one forecaster beats another. With many
forecasters the question is different -- which of them could be the best
-- and answering it with pairwise tests multiplies the size distortion.
Hansen, Lunde and Nason (2011) answer it the way a confidence interval
answers a point estimate: a *set* of models that contains the best one
with probability at least the confidence level,

.. math::

   \lim_{T \to \infty} \Pr\bigl(\mathcal{M}^\ast \subseteq
   \widehat{\mathcal{M}}^\ast_{1-\alpha}\bigr) \ge 1 - \alpha,

built by eliminating the worst model while the hypothesis that the
remaining ones are equally good, :math:`\mathbb{E}[L_{t,i} - L_{t,j}] =
0` for every pair in the set, is rejected. The test statistic is the
largest studentized pairwise differential (the range statistic) or the
largest studentized deviation from the set average (the max statistic),
and its null distribution is taken from a stationary bootstrap of the
loss panel (Politis & Romano, 1994) so that the serial correlation of
forecast losses is respected rather than assumed away.

Two commitments shape the surface. First, the input is a loss panel,
not a set of models: ``(T, M)`` losses aligned origin by origin, which
is what stacking each model's
:meth:`~cultivars.forecast.backtest.BacktestResult.losses` from
backtests on one schedule produces. The procedure never sees the
forecasters, so anything that yields a loss series can enter, and the
alignment it depends on is the backtest's responsibility. Second, the
set is the honest multi-model answer, in the same spirit as the
set-identified impulse responses elsewhere in the package: on a short
or noisy evaluation window it keeps many models, and that is the
finding. Each model carries an MCS p-value, the running maximum of the
elimination p-values up to its own, so that the set at any level is
read off the p-values rather than recomputed.

Layout. :func:`model_confidence_set` is the producer and
:class:`ModelConfidenceSetSelection` the record, the
``_ModelConfidenceSetSelection`` of ``_internals`` under its public
name, carrying the p-values, mean losses, elimination order, and the
``included`` / ``excluded`` / ``best`` readings. The numerics live in
``_core``: ``_model_confidence_set`` runs the elimination loop with the
stationary bootstrap and returns the order and the p-values;
``_MCS_ALPHA`` and ``_MCS_BOOTSTRAP`` are the 10% level and replication
count the defaults follow; ``_validate_names`` / ``_variable_names``
resolve the model labels.

References:
    Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence
    set. *Econometrica*, 79(2), 453-497.

    Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
    *Journal of the American Statistical Association*, 89(428),
    1303-1313.

    White, H. (2000). A reality check for data snooping. *Econometrica*,
    68(5), 1097-1126.

Example:
    Three loss series, the third a unit worse; the set excludes it and
    cannot separate the other two:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> losses = rng.standard_normal((150, 3)) ** 2
    >>> losses[:, 2] += 1.0
    >>> mcs = model_confidence_set(losses, names=("A", "B", "C"), n_bootstrap=300, seed=0)
    >>> "C" in mcs.included, mcs.pvalue("C") < 0.1
    (False, True)
    >>> mcs.included, mcs.best
    (('B', 'A'), 'B')

    The loss columns of three backtests on one schedule, which is the
    intended input:

    >>> from cultivars.forecast.backtest import Backtest
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> y = np.random.default_rng(1).standard_normal((160, 2))
    >>> fits = {p: (lambda w, p=p: VAR(w, order=p).fit()) for p in (1, 2, 3)}
    >>> panel = np.column_stack(
    ...     [Backtest(y, f, start=100).run().losses(name="y1") for f in fits.values()]
    ... )
    >>> panel.shape
    (60, 3)
    >>> model_confidence_set(panel, names=("VAR(1)", "VAR(2)", "VAR(3)"), seed=0).nobs
    60
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ..engine._core import (
    _MCS_ALPHA,
    _MCS_BOOTSTRAP,
    _model_confidence_set,
    _validate_names,
    _variable_names,
)
from ..engine._internals import _ModelConfidenceSetSelection as ModelConfidenceSetSelection
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
    r"""Build the model confidence set from aligned loss series.

    Hansen, Lunde and Nason's (2011) sequential procedure on a panel of
    losses :math:`L_{t,i}`. With :math:`d_{ij,t} = L_{t,i} - L_{t,j}` the
    pairwise differentials and :math:`\bar d_{ij}` their means, the
    hypothesis that every model in the current set :math:`\mathcal{M}`
    is equally good, :math:`H_0 : \mathbb{E}[d_{ij,t}] = 0` for all
    :math:`i, j \in \mathcal{M}`, is tested with

    .. math::

       T_R = \max_{i, j \in \mathcal{M}}
       \frac{|\bar d_{ij}|}{\sqrt{\widehat{\operatorname{var}}(\bar d_{ij})}}
       \qquad\text{or}\qquad
       T_{\max} = \max_{i \in \mathcal{M}}
       \frac{\bar d_{i\cdot}}{\sqrt{\widehat{\operatorname{var}}(\bar d_{i\cdot})}},

    where :math:`\bar d_{i\cdot}` is model :math:`i`'s mean deviation
    from the set average. The null distribution of the statistic and
    the variances in its denominator come from a stationary bootstrap
    of the loss panel (Politis & Romano, 1994), which keeps the serial
    correlation that overlapping forecast losses carry. While the test
    rejects at level :math:`\alpha`, the model with the largest
    studentized deviation is eliminated and the test repeated on what
    remains; the model confidence set is the set left when it stops.
    Each model's MCS p-value is the running maximum of the elimination
    p-values up to and including its own, which makes the set at any
    level exactly the models whose p-value is at least that level.

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

    Note:
        The columns must be losses of the *same* origins under the
        *same* loss, which is what stacking each model's
        :meth:`~cultivars.forecast.backtest.BacktestResult.losses` from
        backtests on one schedule guarantees. The set is the honest
        multi-model answer: on a short or noisy evaluation window it
        keeps many models, and that is the finding, not a failure of
        the procedure. At least 100 bootstrap replications are required
        because the p-values are bootstrap tail frequencies and fewer
        cannot resolve a 10% level. The default expected block length
        grows as :math:`T^{1/3}`, the rate at which the stationary
        bootstrap's block should grow for dependent data.

    See Also:
        * :class:`ModelConfidenceSetSelection` -- the record, with the
          set, the exclusions, and each model's p-value.
        * :class:`~cultivars.forecast.backtest.BacktestResult` -- produces
          the aligned loss columns.
        * :class:`~cultivars.forecast.comparison.ForecastComparison` --
          the pairwise question, for two models.

    References:
        Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model
        confidence set. *Econometrica*, 79(2), 453-497.

        Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
        *Journal of the American Statistical Association*, 89(428),
        1303-1313.

    Example:
        Three loss series, the third a unit worse on average; the set
        drops it and keeps the two the window cannot separate:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> losses = rng.standard_normal((150, 3)) ** 2
        >>> losses[:, 2] += 1.0
        >>> mcs = model_confidence_set(losses, names=("A", "B", "C"), n_bootstrap=300, seed=0)
        >>> mcs.included, mcs.excluded, mcs.best
        (('B', 'A'), ('C',), 'B')
        >>> mcs.elimination_order, mcs.pvalues.round(2)
        ((2, 0, 1), array([0.22, 1.  , 0.  ]))
        >>> mcs.block_length, mcs.nobs
        (5.0, 150)

        The max statistic with an explicit block length:

        >>> alt = model_confidence_set(losses, statistic="max", block_length=3, seed=1)
        >>> alt.names, alt.included
        (('y1', 'y2', 'y3'), ('y2', 'y1'))
        >>> model_confidence_set(losses, n_bootstrap=50)
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: n_bootstrap must be at least 100; got 50.
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
