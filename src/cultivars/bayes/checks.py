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
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from .._core import (
    _DISCREPANCY_NAMES,
    _DISCREPANCY_STATISTICS,
    _EXTREME_PVALUE,
    _MIN_REPLICATIONS,
    ReplicatingModel,
    ReplicatingResult,
    SummaryTable,
    _discrepancy_statistics,
    _predictive_pvalues,
    _source_label,
    _validate_replications,
    _validate_statistics,
    _variable_names,
)
from ..exceptions import SpecificationError

__all__ = [
    "PredictiveCheckTest",
    "posterior_predictive_check",
    "prior_predictive_check",
    "replication_check",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class PredictiveCheckTest:
    """A prior or posterior predictive check: the data against replicated data.

    Each replication is a data set simulated from one draw of the
    parameters -- from the posterior, to ask whether the fitted model
    reproduces the features of the sample it was fitted to; from the
    prior, to ask what the prior says on the scale of the data before any
    sample touches it. A discrepancy statistic ``T`` is evaluated on the
    data and on every replication, and the tail probability ``P(T(y_rep)
    >= T(y))`` locates the data in the replicated distribution: near zero
    or one, the model does not produce data like these in that respect.

    The posterior version is not a frequentist test. Under a correctly
    specified model its p-value is concentrated near one half rather than
    uniform (Meng, 1994), because the same data fit the parameters and
    judge the fit, so a p-value in a tail understates the evidence of
    misspecification rather than overstating it. The prior version is
    not a test at all -- the data are one draw the prior may or may not
    cover -- and its table is read for the range of the replicated
    statistics as much as for the tail probability.

    Attributes:
        kind: ``"posterior"`` or ``"prior"``.
        statistics: Names of the discrepancy statistics, in row order.
        names: Variable labels, in column order.
        observed: ``(m, k)`` statistics of the data.
        replicated: ``(R, m, k)`` statistics of the replications.
        n_replications: Replicated data sets ``R``.
        nobs: Rows in the data set replicated -- the effective sample.
        source: What was checked, for the summary title.
        notes: Remarks from the replicating result or model.
    """

    kind: str
    statistics: tuple[str, ...]
    names: tuple[str, ...]
    observed: npt.NDArray[np.float64] = field(repr=False)
    replicated: npt.NDArray[np.float64] = field(repr=False)
    n_replications: int
    nobs: int
    source: str
    notes: tuple[str, ...] = ()

    @property
    def pvalues(self) -> npt.NDArray[np.float64]:
        """``(m, k)`` tail probabilities ``P(T(y_rep) >= T(y))``, ties counted as one half."""
        return _predictive_pvalues(self.observed, self.replicated)

    def pvalue(self, statistic: str, name: str | None = None) -> npt.NDArray[np.float64] | float:
        """One statistic's tail probabilities, for every variable or for one.

        Args:
            statistic: A name from :attr:`statistics`.
            name: A variable label; ``None`` returns the ``(k,)`` row.

        Raises:
            SpecificationError: If the statistic or the variable is unknown.
        """
        if statistic not in self.statistics:
            raise SpecificationError(
                f"unknown statistic {statistic!r}; this check computed {self.statistics}."
            )
        row = self.pvalues[self.statistics.index(statistic)]
        if name is None:
            return row
        if name not in self.names:
            raise SpecificationError(f"unknown variable {name!r}; expected one of {self.names}.")
        return float(row[self.names.index(name)])

    def replicated_quantiles(
        self, levels: tuple[float, ...] = (0.05, 0.5, 0.95)
    ) -> npt.NDArray[np.float64]:
        """Quantiles of each replicated statistic, ``(len(levels), m, k)``."""
        return np.asarray(np.quantile(self.replicated, levels, axis=0), dtype=np.float64)

    def _flags(self, level: float) -> npt.NDArray[np.bool_]:
        """Which ``(m, k)`` cells sit in a tail of the replicated distribution."""
        p = self.pvalues
        with np.errstate(invalid="ignore"):
            return np.asarray((p < level) | (p > 1.0 - level))

    def extreme(self, *, level: float = _EXTREME_PVALUE) -> tuple[str, ...]:
        """Labels ``statistic[variable]`` with a tail probability outside ``(level, 1 - level)``.

        Args:
            level: The tail size on each side.

        Raises:
            SpecificationError: If the level is not inside ``(0, 0.5)``.
        """
        if not 0.0 < level < 0.5:
            raise SpecificationError(f"level must lie strictly inside (0, 0.5); got {level}.")
        flags = self._flags(level)
        return tuple(
            f"{self.statistics[i]}[{self.names[j]}]"
            for i in range(len(self.statistics))
            for j in range(len(self.names))
            if flags[i, j]
        )

    def adequate(self, *, level: float = _EXTREME_PVALUE) -> bool:
        """Whether no statistic sits in a tail of its replicated distribution."""
        return not self.extreme(level=level)

    def summary(self, *, level: float = _EXTREME_PVALUE) -> SummaryTable:
        """Render as a table: one row per statistic and variable.

        Args:
            level: The tail size on each side that marks a row.

        Raises:
            SpecificationError: If the level is not inside ``(0, 0.5)``.
        """
        if not 0.0 < level < 0.5:
            raise SpecificationError(f"level must lie strictly inside (0, 0.5); got {level}.")
        p = self.pvalues
        bands = self.replicated_quantiles((0.05, 0.5, 0.95))
        flags = self._flags(level)
        rows = tuple(
            (
                self.statistics[i],
                self.names[j],
                f"{self.observed[i, j]:.4g}",
                f"{bands[0, i, j]:.4g}",
                f"{bands[1, i, j]:.4g}",
                f"{bands[2, i, j]:.4g}",
                "nan" if np.isnan(p[i, j]) else f"{p[i, j]:.3f}",
                "*" if flags[i, j] else "",
            )
            for i in range(len(self.statistics))
            for j in range(len(self.names))
        )
        extreme = self.extreme(level=level)
        verdict = "no statistic in a tail" if not extreme else f"{len(extreme)} in a tail"
        metadata = (
            ("Kind", f"{self.kind} predictive"),
            ("Replications", str(self.n_replications)),
            ("Observations", str(self.nobs)),
            ("Statistics", str(len(self.statistics))),
            ("Variables", str(len(self.names))),
            ("Verdict", verdict),
        )
        notes: list[str] = list(self.notes)
        if extreme:
            notes.append(
                f"* p-value below {level:g} or above {1.0 - level:g}: {', '.join(extreme[:8])}"
                + (" ..." if len(extreme) > 8 else "")
                + "."
            )
        if self.kind == "posterior":
            notes.append(
                "Posterior predictive p-values concentrate near 0.5 under a correctly "
                "specified model (Meng, 1994); a tail reading understates misspecification."
            )
        else:
            notes.append(
                "A prior predictive check locates the data in what the prior generates; the "
                "replicated quantiles say what the prior deems plausible on the data's scale."
            )
        if self.n_replications < 200:
            notes.append(
                f"{self.n_replications} replications resolve a tail probability to about "
                f"{1.0 / self.n_replications:.3f}; more replications sharpen the p-values."
            )
        return SummaryTable(
            title=f"{self.kind.capitalize()} predictive check: {self.source}",
            metadata=metadata,
            columns=("statistic", "variable", "observed", "q05", "median", "q95", "p", ""),
            rows=rows,
            notes=tuple(notes),
        )

    def __repr__(self) -> str:
        """Represent the predictive check as a string."""
        extreme = self.extreme()
        verdict = "no statistic in a tail" if not extreme else f"{len(extreme)} in a tail"
        return (
            f"PredictiveCheckTest({self.kind}, {self.n_replications} replications, "
            f"{len(self.statistics)} statistics x {len(self.names)} variables, {verdict})"
        )


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
    **hyperparameters: object,
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
