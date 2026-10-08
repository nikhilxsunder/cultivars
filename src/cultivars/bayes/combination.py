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
r"""Model combination: Bayesian model averaging and predictive stacking.

Two models fitted to the same sample rarely deserve a winner-take-all
verdict. A combination weights them and predicts from the mixture
:math:`\sum_m w_m\, p_m(\tilde y \mid y)`, and the two principled ways to
choose :math:`w` on the simplex answer different questions. *Bayesian
model averaging* takes the posterior model probabilities,

.. math::

   w_m = \frac{\pi_m\, p(y \mid M_m)}{\sum_j \pi_j\, p(y \mid M_j)},

marginal likelihood times prior probability, normalized (Hoeting et al.
1999). It is the coherent answer when one of the models is true, and it
concentrates on the best model as the sample grows. When none is true it
still concentrates, on the model closest in Kullback-Leibler divergence,
which is the known weakness: an average over misspecified models ends up
as a single model with extra steps. *Stacking* takes instead the weights
that maximize the held-out log predictive score of the mixture,

.. math::

   \max_{w \in \Delta^{M-1}} \frac{1}{T} \sum_{t=1}^{T}
   \log \sum_{m=1}^{M} w_m\, p_m(y_t \mid y_{<t}),

over evaluation origins :math:`t` (Yao, Vehtari, Simpson & Gelman 2018).
It assumes no model is true and does not concentrate: two models that
err in different directions keep sharing weight indefinitely, which is
why stacking is the combination to use for forecasting. Its price is the
held-out densities, one per model per origin, which a rolling-origin
backtest produces and nothing else legitimately can.

Two commitments shape the surface. First, the two routes return the same
record, because whatever produced the weights, prediction is the same
act: draw a model in proportion to its weight, then one of that model's
predictive paths. The routes differ only in the record's ``method`` and
in what its ``scores`` column holds, so a consumer -- a fan chart, a
backtest -- never needs to know which was used. Second, averaging refuses
models whose marginal likelihoods score different numbers of
observations. Models of different autoregressive order condition on
different presamples, and a ratio of their marginal likelihoods is a
ratio of densities of different data, not a Bayes factor; the function
says so and asks for a common presample rather than silently comparing
them.

Layout. :func:`bayesian_model_average` and :func:`stacking` are the two
constructors. The former collects a
:class:`MarginalLikelihoodSelection` per model, through
:func:`~cultivars.bayes.evidence.marginal_likelihood` unless the caller
supplies them, and normalizes on the log scale; the latter hands the
``(T, M)`` density matrix to ``_core._estimators._stacking_weights``,
which maximizes the concave objective through a softmax parameterization
of the simplex. Both return a :class:`ModelCombinationSelection`, the
frozen record in ``_internals`` that owns the mixing --
``forecast_paths`` through ``_core._samplers._mix_predictive_paths``, and
the ``(16th percentile, mean, 84th percentile)`` ``forecast`` the family
convention expects -- and the summary table. The gatekeepers
``_validate_predictive``, ``_validate_log_density_matrix``, and
``_validate_names`` are ``_core`` validators shared with the rest of the
forecasting surface.

References:
    Hoeting, J. A., Madigan, D., Raftery, A. E., & Volinsky, C. T. (1999).
    Bayesian model averaging: A tutorial. *Statistical Science*, 14(4),
    382-401.

    Geweke, J., & Amisano, G. (2011). Optimal prediction pools. *Journal of
    Econometrics*, 164(1), 130-141.

    Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using stacking
    to average Bayesian predictive distributions. *Bayesian Analysis*,
    13(3), 917-1007.

Example:
    Stacking two stand-in models on held-out log densities in which the
    first is better at every origin; any object with ``forecast_paths``
    can be combined:

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
    >>> weights.method, round(float(weights.weights[0]), 2)
    ('stacking', 1.0)
    >>> float(weights.forecast_paths(4, seed=0).mean())
    0.0
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from .._core import (
    _stacking_weights,
    _validate_log_density_matrix,
    _validate_names,
    _validate_predictive,
)
from .._internals import _MarginalLikelihoodSelection as MarginalLikelihoodSelection
from .._internals import _ModelCombinationSelection as ModelCombinationSelection
from ..exceptions import DimensionError, SpecificationError
from .evidence import marginal_likelihood

__all__ = [
    "MarginalLikelihoodSelection",
    "ModelCombinationSelection",
    "bayesian_model_average",
    "stacking",
]


def bayesian_model_average(
    *results: object,
    records: Sequence[MarginalLikelihoodSelection] | None = None,
    prior_probabilities: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
) -> ModelCombinationSelection:
    r"""Weight models by posterior model probability.

    For models :math:`M_1, \ldots, M_m` fitted to the same sample
    :math:`y`, with marginal likelihoods :math:`p(y \mid M_i)` and prior
    probabilities :math:`\pi_i`, the weight on model :math:`i` is

    .. math::

       w_i = p(M_i \mid y)
           = \frac{\pi_i\, p(y \mid M_i)}{\sum_j \pi_j\, p(y \mid M_j)},

    computed on the log scale with a log-sum-exp normalization so that a
    spread of hundreds of nats between marginal likelihoods -- routine for
    a VAR -- does not underflow. The mixture predictive is then
    :math:`\sum_i w_i\, p(\tilde y \mid y, M_i)`, and the returned record
    draws from it by drawing a model in proportion to its weight and then
    one of that model's predictive paths (Hoeting et al., 1999).

    This is the coherent answer when one of the models is true, and as
    the sample grows the weights concentrate on it. When none is true the
    weights still concentrate, on whichever model is closest in
    Kullback-Leibler divergence, so the average collapses to a single
    model with extra steps; :func:`stacking` is the combination to prefer
    for forecasting under misspecification, and the record's notes say so.

    The marginal likelihoods must score the same observations. Two models
    of different autoregressive order condition on different presamples
    and therefore score samples of different length, and the ratio of
    their marginal likelihoods is then a ratio of densities of different
    data, not a Bayes factor. The function refuses that case; the remedy
    is to drop leading rows so every model conditions on the same
    presample and refit.

    Args:
        *results: Two or more fitted results on the same sample, each
            exposing ``forecast_paths(steps, *, seed)`` returning
            ``(n_paths, steps, k)``; the Bayesian VAR family, the
            stochastic-volatility posterior, and any point-estimate result
            with a simulated predictive. The predictive paths must agree in
            ``steps`` and ``k`` across models, which is checked when paths
            are drawn, not here.
        records: The models' marginal likelihoods as
            :class:`MarginalLikelihoodSelection` records, one per result in
            the same order. ``None`` computes them through
            :func:`~cultivars.bayes.evidence.marginal_likelihood`, which is
            exact for the conjugate BVAR and simulated (bridge sampling or
            Chib) for the sampled members of the family.
        prior_probabilities: :math:`\pi_i`, one per result, non-negative
            and not all zero; they are normalized to sum to one. ``None``
            gives every model the same prior probability.
        names: Row labels for the record, one per result, unique. ``None``
            uses each marginal-likelihood record's ``source``, which for a
            package result is the model and prior label, so two fits that
            differ only in prior are told apart.

    Returns:
        The :class:`ModelCombinationSelection` with ``method`` set to
        ``"bayesian model average"``, ``weights`` the posterior model
        probabilities, ``scores`` the log marginal likelihoods the weights
        came from, and ``n_origins`` zero, since no held-out evaluation was
        involved.

    Raises:
        SpecificationError: If fewer than two results are given; if a
            result exposes no ``forecast_paths``; if a result has no
            marginal likelihood (a point estimate, when ``records`` is
            omitted); if the records score different numbers of
            observations; if a prior probability is negative or all are
            zero; or if ``names`` repeats a label.
        DimensionError: If ``records``, ``names``, or
            ``prior_probabilities`` do not have one entry per result.

    Note:
        When some marginal likelihoods are simulated, each carries a Monte
        Carlo standard error, and a weight ratio :math:`w_i / w_j` is only
        as precise as the difference of two log marginal likelihood
        estimates; the record notes this when it applies. A gap of a few
        nats between two simulated estimates with errors of a nat each is
        not a decisive weight, however sharply the normalized numbers
        appear to split.

    See Also:
        * :func:`stacking` -- weights that maximize held-out predictive
          score, which do not concentrate under misspecification.
        * :func:`~cultivars.bayes.evidence.marginal_likelihood` -- the
          record this function computes for each result when ``records``
          is omitted.
        * :func:`~cultivars.bayes.evidence.compare` -- the same marginal
          likelihoods ranked in a table, with Bayes factors, when a
          winner rather than a mixture is wanted.
        * :class:`ModelCombinationSelection` -- the record returned, and
          how it draws from the mixture predictive.

    References:
        Hoeting, J. A., Madigan, D., Raftery, A. E., & Volinsky, C. T.
        (1999). Bayesian model averaging: A tutorial. *Statistical Science*,
        14(4), 382-401.

        Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *Journal of the
        American Statistical Association*, 90(430), 773-795.

    Example:
        Two conjugate BVARs on one sample, differing only in the tightness
        of the Minnesota prior; the marginal likelihoods are exact, so the
        weights are too:

        >>> import numpy as np
        >>> from cultivars.bayes.priors import NormalInverseWishartPrior
        >>> from cultivars.multivariate.large_dim.bayesian import BVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((120, 2))
        >>> for t in range(1, 120):
        ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
        >>> loose = BVAR(y, order=1).fit(n_draws=200, seed=0)
        >>> tight = BVAR(y, order=1, prior=NormalInverseWishartPrior(tightness=0.02))
        >>> tight = tight.fit(n_draws=200, seed=0)
        >>> bma = bayesian_model_average(loose, tight, names=("loose", "tight"))
        >>> bma.method, bma.weights.round(3)
        ('bayesian model average', array([0.998, 0.002]))
        >>> bma.forecast_paths(4, seed=0).shape, bma.forecast(4).shape
        ((200, 4, 2), (4, 2, 3))

        Prior model probabilities tilt the weights but a six-nat gap in
        log marginal likelihood is hard to overturn:

        >>> tilted = bayesian_model_average(loose, tight, prior_probabilities=(1.0, 99.0))
        >>> tilted.weights.round(3)
        array([0.824, 0.176])

        Models conditioning on different presamples score different
        samples and are refused:

        >>> two = BVAR(y, order=2).fit(n_draws=100, seed=0)
        >>> bayesian_model_average(loose, two)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: The marginal likelihoods score different ...
    """
    predictive = _validate_predictive(results)
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
        names=_validate_names(names, [record.source for record in evidence]),
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
    r"""Weight models to maximize the mixture's held-out log predictive score.

    Given the log predictive density :math:`\ell_{tm} = \log p_m(y_t \mid
    y_{<t})` that each model :math:`m` assigned to the realized outcome at
    each evaluation origin :math:`t`, the weights solve

    .. math::

       \max_{w \in \Delta^{M-1}} \;
       \frac{1}{T} \sum_{t=1}^{T} \log \sum_{m=1}^{M} w_m\, e^{\ell_{tm}},

    the mean log score of the mixture of predictive densities over the
    simplex (Yao, Vehtari, Simpson & Gelman, 2018). The objective is
    concave in :math:`w`, and it is maximized through a softmax map from
    :math:`M - 1` free coordinates whose stationary point is the simplex
    optimum, so the weights are non-negative and sum to one by
    construction rather than by projection.

    Nothing here assumes a model is true. Two models that err in
    different directions keep sharing weight however long the evaluation
    window, which is why stacking is the combination for forecasting and
    why a weight vector that does not concentrate is a finding about the
    models, not a failure to choose. The price is that the densities must
    be genuinely held out: from a rolling or expanding origin, each
    :math:`\ell_{tm}` computed from a fit that did not see :math:`y_t`.
    In-sample densities reward whichever model overfits most.

    Args:
        log_density: ``(T, M)`` log predictive densities of the realized
            outcome, one row per evaluation origin and one column per
            model, in the order of ``results``; anything
            :func:`numpy.asarray` accepts, finite throughout. A rolling
            backtest produces exactly this matrix. At least ``M`` origins
            are required, though a useful weight needs many more; the
            record reports the count as ``n_origins``.
        results: The ``M`` fitted results the columns correspond to, each
            exposing ``forecast_paths(steps, *, seed)``; typically the fits
            on the full sample, which the mixture predicts from.
        names: Row labels for the record, one per result, unique. ``None``
            uses ``model[0] ... model[M-1]``.

    Returns:
        The :class:`ModelCombinationSelection` with ``method`` set to
        ``"stacking"``, ``weights`` the optimal simplex point, ``scores``
        each model's own mean held-out log density (column means of
        ``log_density``), and ``n_origins`` the number of rows.

    Raises:
        DimensionError: If ``log_density`` is not two-dimensional, if its
            column count differs from the number of results, or if
            ``names`` does not have one label per result.
        SpecificationError: If there are fewer than two models, fewer
            origins than models, a result exposes no ``forecast_paths``, or
            ``names`` repeats a label.
        NumericalError: If a log density is not finite, or if the
            optimizer fails to converge to a finite objective.

    Note:
        The record's first note reports the stacked mean log score against
        the best single model's. The stacked score can never be below the
        best single model's on the evaluation origins, since a vertex of
        the simplex is feasible; how much it exceeds it is the gain from
        combining, on the data the weights were chosen on, and is
        therefore optimistic.

    See Also:
        * :func:`bayesian_model_average` -- weights from marginal
          likelihoods, for the case where one model is believed true.
        * :class:`~cultivars.forecast.backtest.Backtest` -- the
          rolling-origin machinery that produces held-out predictive
          densities.
        * :class:`ModelCombinationSelection` -- the record returned, and
          how it draws from the mixture predictive.

    References:
        Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Using
        stacking to average Bayesian predictive distributions. *Bayesian
        Analysis*, 13(3), 917-1007.

        Geweke, J., & Amisano, G. (2011). Optimal prediction pools.
        *Journal of Econometrics*, 164(1), 130-141.

    Example:
        Two stand-in models -- any object with ``forecast_paths`` will do
        -- and held-out log densities in which the first is better at
        every origin, so the optimum is the vertex:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> class Toy:
        ...     def __init__(self, level):
        ...         self.level = level
        ...     def forecast_paths(self, steps=8, *, seed=None):
        ...         return np.full((100, steps, 1), self.level)
        >>> good = -0.5 * rng.standard_normal(200) ** 2
        >>> logdens = np.column_stack([good, good - 0.3])
        >>> combined = stacking(logdens, (Toy(0.0), Toy(1.0)), names=("a", "b"))
        >>> combined.weights.round(3), combined.n_origins
        (array([1., 0.]), 200)
        >>> float(combined.forecast_paths(4, seed=0).mean())
        0.0

        Two models whose errors are unrelated share the weight, and that
        is the point:

        >>> other = -0.5 * rng.standard_normal(200) ** 2
        >>> shared = stacking(np.column_stack([good, other]), (Toy(0.0), Toy(1.0)))
        >>> shared.names, bool(np.all((shared.weights > 0.3) & (shared.weights < 0.7)))
        (('model[0]', 'model[1]'), True)
    """
    matrix = _validate_log_density_matrix(log_density)
    predictive = _validate_predictive(results)
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
        names=_validate_names(names, [f"model[{m}]" for m in range(len(predictive))]),
        weights=weights,
        method="stacking",
        scores=np.asarray(own, dtype=np.float64),
        results=predictive,
        n_origins=int(matrix.shape[0]),
        notes=tuple(notes),
    )
