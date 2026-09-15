# filepath: /src/cultivars/bayes/combination.py
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

"""Model combination: Bayesian model averaging and predictive stacking.

Two models fitted to the same sample rarely deserve a winner-take-all
verdict. A combination weights them and predicts from the mixture, and the
two principled ways to choose the weights answer different questions:

* **Bayesian model averaging** weights by posterior model probability --
  marginal likelihood times prior probability, normalized. It is the
  coherent answer when one of the models is true, and it concentrates on
  the best model as the sample grows. When none is true it still
  concentrates, on the model closest in Kullback-Leibler terms, which is
  the known weakness: an average over misspecified models ends up as a
  single model with extra steps.

* **Stacking** (Yao, Vehtari, Simpson, and Gelman, 2018) weights to
  maximize the out-of-sample log predictive score of the mixture. It does
  not assume any model is true and does not concentrate: two models that
  err in different directions keep sharing weight indefinitely, which is
  why stacking is the combination to use for forecasting. It needs held-out
  predictive densities -- one per model per evaluation origin -- which a
  rolling-origin backtest produces.

Both return the same :class:`ModelCombinationSelection`, whose
``forecast_paths`` draws from the mixture predictive by drawing a model in
proportion to its weight and then one of that model's paths.

References:
    Hoeting, J. A., Madigan, D., Raftery, A. E., & Volinsky, C. T. (1999).
        Bayesian model averaging: A tutorial. *Statistical Science*, 14(4),
        382-401.
    Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using stacking
        to average Bayesian predictive distributions. *Bayesian Analysis*,
        13(3), 917-1007.

Example:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> class Toy:
    ...     def __init__(self, level):
    ...         self.level = level
    ...     def forecast_paths(self, steps=8, *, seed=None):
    ...         return np.full((100, steps, 1), self.level)
    >>> good = -0.5 * rng.standard_normal(200) ** 2
    >>> weights = stacking(
    ...     np.column_stack([good, good - 3.0]), (Toy(0.0), Toy(1.0)), names=("a", "b")
    ... )
    >>> round(float(weights.weights[0]), 2)
    1.0
    >>> float(weights.forecast_paths(4, seed=0).mean())
    0.0
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from .._core import PredictiveResult, _stacking_weights, _validate_log_density_matrix
from .._internals import _MarginalLikelihoodSelection as MarginalLikelihoodSelection
from .._internals import _ModelCombinationSelection as ModelCombinationSelection
from ..exceptions import DimensionError, SpecificationError
from .evidence import marginal_likelihood

__all__ = ["ModelCombinationSelection", "bayesian_model_average", "stacking"]


def _check_predictive(results: Sequence[object]) -> tuple[PredictiveResult, ...]:
    """Every result must expose ``forecast_paths``, and there must be at least two."""
    if len(results) < 2:
        raise SpecificationError(f"A combination needs at least two models; got {len(results)}.")
    for result in results:
        if not isinstance(result, PredictiveResult):
            raise SpecificationError(
                f"{type(result).__name__} exposes no forecast_paths(); a combination can only "
                "mix results that simulate their posterior predictive."
            )
    return tuple(results)  # type: ignore[arg-type]


def _labels(names: Sequence[str] | None, fallback: Sequence[str]) -> tuple[str, ...]:
    labels = tuple(fallback) if names is None else tuple(names)
    if len(labels) != len(fallback):
        raise DimensionError(f"Got {len(labels)} names for {len(fallback)} models.")
    if len(set(labels)) != len(labels):
        raise SpecificationError(f"Model names must be distinct; got {labels}.")
    return labels


def bayesian_model_average(
    *results: object,
    records: Sequence[MarginalLikelihoodSelection] | None = None,
    prior_probabilities: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
) -> ModelCombinationSelection:
    """Weight models by posterior model probability.

    Args:
        *results: Two or more fitted results on the same sample, each
            exposing ``forecast_paths``.
        records: Their marginal likelihoods, in the same order; computed
            through :func:`~cultivars.bayes.evidence.marginal_likelihood`
            when omitted.
        prior_probabilities: Prior model probabilities; default equal.
        names: Row labels; default each record's ``source``.

    Returns:
        The combination, with weights ``p(M_m | y)``.

    Raises:
        SpecificationError: If fewer than two results are given, one cannot
            simulate its predictive, one has no marginal likelihood, the
            records score different numbers of observations, or a prior
            probability is negative.
        DimensionError: If ``records``, ``names`` or ``prior_probabilities``
            do not match the number of results.
    """
    predictive = _check_predictive(results)
    evidence = (
        tuple(marginal_likelihood(result) for result in predictive)
        if records is None
        else tuple(records)
    )
    if len(evidence) != len(predictive):
        raise DimensionError(f"Got {len(evidence)} records for {len(predictive)} results.")
    counts = {record.nobs for record in evidence if record.nobs > 0}
    if len(counts) > 1:
        raise SpecificationError(
            "The marginal likelihoods score different numbers of observations "
            f"({', '.join(str(c) for c in sorted(counts))}); posterior model probabilities "
            "are only defined over models fitted to the same sample. Drop leading rows so "
            "every model conditions on the same presample and refit."
        )
    if prior_probabilities is None:
        prior = np.full(len(evidence), 1.0 / len(evidence))
    else:
        prior = np.asarray(prior_probabilities, dtype=np.float64)
        if prior.shape != (len(evidence),):
            raise DimensionError(
                f"Got {prior.size} prior probabilities for {len(evidence)} models."
            )
        if np.any(prior < 0.0) or not prior.sum() > 0.0:
            raise SpecificationError(
                "Prior model probabilities must be non-negative, not all zero."
            )
        prior = prior / prior.sum()
    log_values = np.array([record.log_value for record in evidence])
    with np.errstate(divide="ignore"):
        log_post = log_values + np.log(prior)
    weights = np.exp(log_post - logsumexp(log_post))
    notes = [
        "Weights are posterior model probabilities; they concentrate on one model as the "
        "sample grows even when no model is true, so prefer stacking for forecasting "
        "under misspecification.",
    ]
    errors = np.array([record.mcse for record in evidence])
    if np.any(np.isfinite(errors) & (errors > 0.0)):
        notes.append(
            "Some marginal likelihoods are simulated; a weight ratio is only as precise as "
            "the difference of two log ML estimates with their Monte Carlo errors."
        )
    return ModelCombinationSelection(
        names=_labels(names, [record.source for record in evidence]),
        weights=np.asarray(weights, dtype=np.float64),
        method="bayesian model average",
        scores=log_values,
        results=predictive,
        notes=tuple(notes),
    )


def stacking(
    log_density: npt.ArrayLike,
    results: Sequence[object],
    *,
    names: Sequence[str] | None = None,
) -> ModelCombinationSelection:
    """Weight models to maximize the mixture's held-out log predictive score.

    Args:
        log_density: ``(T, M)`` log predictive densities of the realized
            outcome, one row per evaluation origin and one column per model,
            in the order of ``results``. These must be out-of-sample --
            from a rolling or expanding origin -- or the weights favour
            whichever model overfits most.
        results: The ``M`` fitted results, each exposing ``forecast_paths``.
        names: Row labels; default ``model[0] .. model[M-1]``.

    Returns:
        The combination, with the log-score-optimal weights and each
        model's own mean log score in ``scores``.

    Raises:
        DimensionError: If the matrix, results and names disagree in count.
        SpecificationError: If fewer than two models or too few origins.
    """
    matrix = _validate_log_density_matrix(log_density)
    predictive = _check_predictive(results)
    if matrix.shape[1] != len(predictive):
        raise DimensionError(
            f"log_density has {matrix.shape[1]} columns for {len(predictive)} results."
        )
    weights, score = _stacking_weights(matrix)
    own = matrix.mean(axis=0)
    notes = [
        f"Stacked mean log score {score:.3f} against the best single model's "
        f"{own.max():.3f}; weights need not concentrate, and a spread across models is "
        "the finding, not a failure to choose.",
    ]
    return ModelCombinationSelection(
        names=_labels(names, [f"model[{m}]" for m in range(len(predictive))]),
        weights=weights,
        method="stacking",
        scores=np.asarray(own, dtype=np.float64),
        results=predictive,
        n_origins=int(matrix.shape[0]),
        notes=tuple(notes),
    )
