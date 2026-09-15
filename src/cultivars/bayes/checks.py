# filepath: /src/cultivars/bayes/checks.py
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

"""Prior and posterior predictive checks.

A Bayesian model is a machine for generating data. The predictive checks
run the machine and compare what comes out with what came in: a
*posterior* predictive check simulates replicated data sets from the
posterior draws and asks whether the fitted model reproduces the features
of the sample it was fitted to; a *prior* predictive check simulates them
from the prior and shows what the prior deems plausible on the scale of
the data, before any sample is touched (Gelman, Meng & Stern, 1996;
Gabry et al., 2019).

Both compare *discrepancy statistics* -- per-variable features of a data
set, by default the mean, standard deviation, first-order autocorrelation,
excess kurtosis, and first-order autocorrelation of the squares -- and
report, for each, where the data sit in the replicated distribution as a
tail probability ``P(T(y_rep) >= T(y))``. The statistics are the ones
whose failure names the misspecification: a Gaussian innovation cannot
match a fat-tailed ``kurtosis``, a constant covariance cannot match the
volatility clustering in ``arch1``.

Every sampled result in the package that can replicate its sample
satisfies :class:`~cultivars._core.ReplicatingResult`: the conjugate,
Gibbs, Student-t, and stochastic-volatility BVARs, and the
stochastic-volatility posterior. Prior replications are offered by the
conjugate and Student-t BVARs (dummy observations folded into the prior),
the Gibbs BVAR under a static prior without dummies, and the
stochastic-volatility model with the samplers' prior hyperparameters.

Example:
    >>> import numpy as np
    >>> from cultivars.multivariate.large_dim import BVAR
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((160, 2))
    >>> for t in range(1, 160):
    ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(2)
    >>> check = posterior_predictive_check(BVAR(y, order=1).fit(n_draws=300, seed=0), seed=0)
    >>> check.kind, check.observed.shape, check.replicated.shape
    ('posterior', (5, 2), (200, 5, 2))
    >>> check.adequate()
    True
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from .._core import (
    _DISCREPANCY_NAMES,
    _DISCREPANCY_STATISTICS,
    _MIN_REPLICATIONS,
    ReplicatingModel,
    ReplicatingResult,
    _discrepancy_statistics,
    _source_label,
    _validate_replications,
    _validate_statistics,
    _variable_names,
)
from .._internals import _PredictiveCheckTest as PredictiveCheckTest
from ..exceptions import SpecificationError


def replication_check(
    observed: npt.ArrayLike,
    replicated: npt.ArrayLike,
    *,
    kind: str = "posterior",
    statistics: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
    source: str = "replications",
    notes: Sequence[str] = (),
) -> PredictiveCheckTest:
    """A predictive check on replications that came from elsewhere.

    The same comparison the two convenience functions run, on an observed
    panel and replicated data sets the user simulated -- from a sampler
    outside the package, or from a result whose replication rule is their
    own.

    Args:
        observed: ``(n,)`` or ``(n, k)`` data.
        replicated: ``(R, n)`` or ``(R, n, k)`` replicated data sets.
        kind: ``"posterior"`` or ``"prior"``; sets how the record reads.
        statistics: Names from :data:`DISCREPANCY_STATISTICS`; default
            mean, sd, acf1, kurtosis, arch1.
        names: Variable labels; default ``y1 ... yk``.
        source: Title for the record.
        notes: Remarks the record carries.

    Returns:
        The record.

    Raises:
        SpecificationError: If the kind or a statistic name is unknown, or
            the labels do not match the panel.
        DimensionError: If the shapes disagree or there are fewer than 20
            replications.
        NumericalError: If a replication is not finite.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = rng.standard_normal(100)
        >>> rep = rng.standard_normal((100, 100))
        >>> check = replication_check(y, rep, statistics=["mean", "sd"])
        >>> check.observed.shape, check.pvalues.shape
        ((2, 1), (2, 1))
    """
    if kind not in ("posterior", "prior"):
        raise SpecificationError(f"kind must be 'posterior' or 'prior'; got {kind!r}.")
    chosen = _validate_statistics(
        statistics, known=_DISCREPANCY_NAMES, default=_DISCREPANCY_STATISTICS
    )
    data, reps = _validate_replications(observed, replicated, minimum=_MIN_REPLICATIONS)
    k = data.shape[1]
    labels = (
        (tuple(f"y{i + 1}" for i in range(k)) if k > 1 else ("y",))
        if names is None
        else tuple(str(name) for name in names)
    )
    if len(labels) != k:
        raise SpecificationError(f"{len(labels)} names for {k} variables.")
    observed_stats = _discrepancy_statistics(data, chosen)
    replicated_stats = np.stack([_discrepancy_statistics(panel, chosen) for panel in reps])
    return PredictiveCheckTest(
        kind=kind,
        statistics=chosen,
        names=labels,
        observed=observed_stats,
        replicated=replicated_stats,
        n_replications=int(reps.shape[0]),
        nobs=int(data.shape[0]),
        source=source,
        notes=tuple(notes),
    )


def posterior_predictive_check(
    result: ReplicatingResult,
    *,
    statistics: Sequence[str] | None = None,
    n_replications: int = 200,
    seed: int | np.random.Generator | None = None,
) -> PredictiveCheckTest:
    """Replicate the sample from the posterior and compare its features with the data.

    Args:
        result: A fitted result that can replicate its sample -- one
            exposing ``observed`` and ``posterior_replications()``: the
            Bayesian VAR family and the stochastic-volatility posterior.
        statistics: Names from :data:`DISCREPANCY_STATISTICS`; default
            mean, sd, acf1, kurtosis, arch1.
        n_replications: Replicated data sets; draws are spread evenly
            across the kept sequence.
        seed: Seed or generator for the replications.

    Returns:
        The record, with the result's own remarks on how it replicates.

    Raises:
        SpecificationError: If the result cannot replicate its sample, or
            a statistic name is unknown. Results without retained draws
            -- point estimates -- have no posterior to replicate from.
        DimensionError: If fewer than 20 replications are asked for.

    Example:
        >>> import numpy as np
        >>> from cultivars.univariate import SV
        >>> rng = np.random.default_rng(1)
        >>> h = np.zeros(300)
        >>> for t in range(1, 300):
        ...     h[t] = 0.95 * h[t - 1] + 0.3 * rng.standard_normal()
        >>> y = np.exp(0.5 * h) * rng.standard_normal(300)
        >>> post = SV(y, mean="zero").sample(n_draws=600, n_burn=200, seed=0)
        >>> check = posterior_predictive_check(post, statistics=["sd", "arch1"], seed=0)
        >>> check.statistics, check.names
        (('sd', 'arch1'), ('y',))
    """
    if not isinstance(result, ReplicatingResult):
        raise SpecificationError(
            f"{type(result).__name__} cannot replicate its sample: a posterior predictive "
            "check needs a result exposing observed and posterior_replications(), which the "
            "Bayesian VAR family (BVAR, GibbsBVAR, StudentBVAR, BVARSV) and the "
            "stochastic-volatility posterior provide. A point estimate has no posterior to "
            "replicate from."
        )
    replicated = result.posterior_replications(n_replications, seed=seed)
    observed = np.asarray(result.observed, dtype=np.float64)
    k = 1 if observed.ndim == 1 else observed.shape[1]
    remarks = getattr(result, "_replication_notes", None)
    notes = tuple(remarks()) if callable(remarks) else ()
    return replication_check(
        observed,
        replicated,
        kind="posterior",
        statistics=statistics,
        names=_variable_names(result, k),
        source=_source_label(result),
        notes=notes,
    )


def prior_predictive_check(
    model: ReplicatingModel,
    *,
    statistics: Sequence[str] | None = None,
    n_replications: int = 200,
    seed: int | np.random.Generator | None = None,
    **hyperparameters: float | tuple[float, float] | None,
) -> PredictiveCheckTest:
    """Replicate the sample from the prior and locate the data in what it generates.

    Args:
        model: An unfitted model that can draw from its prior -- one
            exposing ``observed`` and ``prior_replications()``: ``BVAR``,
            ``StudentBVAR``, ``GibbsBVAR`` under a static prior without
            dummy observations, and ``SV``.
        statistics: Names from :data:`DISCREPANCY_STATISTICS`; default
            mean, sd, acf1, kurtosis, arch1.
        n_replications: Replicated data sets.
        seed: Seed or generator for the replications.
        **hyperparameters: Prior settings the model's ``prior_replications``
            takes by name -- ``df`` for the Student-t BVAR; ``prior_mu``,
            ``prior_phi``, ``prior_sigma2`` for the stochastic-volatility
            model, exactly as ``sample()`` takes them.

    Returns:
        The record.

    Raises:
        SpecificationError: If the model cannot draw from its prior, a
            statistic name is unknown, or a hyperparameter is not one the
            model takes.
        DimensionError: If fewer than 20 replications are asked for.

    Example:
        >>> import numpy as np
        >>> from cultivars.multivariate.large_dim import BVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(rng.standard_normal((120, 2)), axis=0)
        >>> check = prior_predictive_check(BVAR(y, order=2), statistics=["sd", "acf1"], seed=0)
        >>> check.kind, check.replicated.shape
        ('prior', (200, 2, 2))
    """
    if not isinstance(model, ReplicatingModel):
        raise SpecificationError(
            f"{type(model).__name__} cannot draw from its prior: a prior predictive check "
            "needs a model exposing observed and prior_replications(), which BVAR, "
            "StudentBVAR, GibbsBVAR (static prior, no dummy observations), and SV provide. "
            "The stochastic-volatility BVAR's random-walk volatility prior has no stationary "
            "distribution to replicate from; check it on the posterior side."
        )
    accepted = inspect.signature(model.prior_replications).parameters
    unknown = sorted(name for name in hyperparameters if name not in accepted)
    if unknown:
        raise SpecificationError(
            f"{type(model).__name__}.prior_replications does not take {unknown}; it takes "
            f"{[name for name in accepted if name not in ('n_replications', 'seed')]}."
        )
    replicated = cast(Any, model).prior_replications(n_replications, seed=seed, **hyperparameters)
    observed = np.asarray(model.observed, dtype=np.float64)
    k = 1 if observed.ndim == 1 else observed.shape[1]
    reach = np.abs(replicated).reshape(replicated.shape[0], -1).max(axis=1)
    wild = float((reach > 1e3 * np.abs(observed).max()).mean())
    notes = (
        (
            f"{wild:.0%} of the prior replications reach beyond a thousand times the largest "
            "observation: the prior admits explosive systems at that rate. A Minnesota prior "
            "centered on the unit root does so by construction; tighten the lag-one "
            "variance if that mass is unwanted.",
        )
        if wild > 0.0
        else ()
    )
    label = getattr(model, "prior", None)
    prior_label = getattr(label, "_label", None)
    source = (
        f"{type(model).__name__} under {prior_label()}"
        if callable(prior_label)
        else type(model).__name__
    )
    return replication_check(
        observed,
        replicated,
        kind="prior",
        statistics=statistics,
        names=_variable_names(model, k),
        source=source,
        notes=notes,
    )
