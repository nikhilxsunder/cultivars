from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from .._core import SummaryTable, _evidence_label
from ..exceptions import DimensionError, SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True)
class _LagOrderSelection:
    """Lag-order criteria evaluated over a common sample.

    Carries the whole curve for each criterion, not just the winner. The four
    routinely disagree -- Bayesian and Hannan-Quinn penalize harder than Akaike
    and final prediction error, so they pick shorter -- and a caller shown only
    four integers cannot tell a decisive choice from a coin flip between two
    orders whose criteria differ in the fourth decimal.

    Every order was scored on the same ``nobs`` observations, trimmed at
    ``max_lags``. That is what makes the entries of a curve comparable to each
    other; scoring each order on its own natural sample would bias the choice
    toward short lags, because a longer sample lowers the log-determinant for
    reasons that have nothing to do with fit.

    Attributes:
        max_lags: Longest order scored; the curves have ``max_lags + 1``
            entries, indexed by order from zero.
        nobs: The common effective sample every order was scored on.
        aic: Akaike criterion, ``log det Sigma + 2 m / T``.
        bic: Schwarz criterion, ``log det Sigma + log(T) m / T``.
        hqic: Hannan-Quinn criterion, ``log det Sigma + 2 log log(T) m / T``.
        fpe: Final prediction error. On a *determinant* scale rather than a
            log-determinant one, so its values are not comparable with the
            other three -- only its own minimum is meaningful.

    Note:
        The parameter count ``m`` here excludes the innovation covariance,
        following Lutkepohl: those terms are common across orders and cancel.
        A fitted result's ``information_criteria`` counts them, so the same
        model scores differently in the two places. Both are right in context;
        do not reconcile them.
    """

    max_lags: int
    nobs: int
    aic: npt.NDArray[np.float64]
    bic: npt.NDArray[np.float64]
    hqic: npt.NDArray[np.float64]
    fpe: npt.NDArray[np.float64]

    _CRITERIA: ClassVar[tuple[str, ...]] = ("aic", "bic", "hqic", "fpe")
    """The criteria scored, in display order."""

    @property
    def orders(self) -> tuple[int, ...]:
        """The orders scored, ``0`` through ``max_lags``."""
        return tuple(range(self.max_lags + 1))

    def curve(self, criterion: str) -> npt.NDArray[np.float64]:
        """The full criterion curve, indexed by order.

        Args:
            criterion: One of ``"aic"``, ``"bic"``, ``"hqic"``, ``"fpe"``.

        Returns:
            An array of length ``max_lags + 1``.

        Raises:
            SpecificationError: If the criterion is not recognized.
        """
        if criterion not in self._CRITERIA:
            raise SpecificationError(
                f"criterion must be one of {self._CRITERIA}; got {criterion!r}."
            )
        curve: npt.NDArray[np.float64] = getattr(self, criterion)
        return curve

    def best(self, criterion: str) -> int:
        """The order minimizing one criterion.

        Args:
            criterion: One of ``"aic"``, ``"bic"``, ``"hqic"``, ``"fpe"``.

        Returns:
            The selected order. Ties break toward the shorter order, which is
            the conservative direction.

        Raises:
            SpecificationError: If the criterion is not recognized.
        """
        return int(np.argmin(self.curve(criterion)))

    @property
    def selected(self) -> dict[str, int]:
        """Each criterion's chosen order, keyed by criterion name."""
        return {criterion: self.best(criterion) for criterion in self._CRITERIA}

    @property
    def consensus(self) -> int | None:
        """The order all four criteria agree on, or ``None`` when they differ.

        Deliberately refuses to arbitrate. Disagreement is the normal case and
        it is information: it says the evidence does not pin the order down,
        and the choice belongs to whoever knows what the model is for. A
        property that voted, or that quietly preferred one criterion, would
        hide exactly that.
        """
        picks = set(self.selected.values())
        return picks.pop() if len(picks) == 1 else None

    def to_table(self) -> SummaryTable:
        """The criterion curves as one table, minima marked.

        Returns:
            A :class:`SummaryTable` with one row per order and one column per
            criterion; each column's minimum carries a trailing asterisk.
        """
        chosen = self.selected
        return SummaryTable(
            title="Lag order selection",
            metadata=(
                ("Observations", f"{self.nobs}"),
                ("Maximum lags", f"{self.max_lags}"),
                ("Consensus", "none" if self.consensus is None else f"{self.consensus}"),
            ),
            columns=("p", *(criterion.upper() for criterion in self._CRITERIA)),
            rows=tuple(
                (
                    f"{order}",
                    *(
                        f"{self.curve(criterion)[order]:.4f}"
                        + ("*" if chosen[criterion] == order else "")
                        for criterion in self._CRITERIA
                    ),
                )
                for order in self.orders
            ),
            notes=(
                "An asterisk marks each criterion's minimum. Every order was scored "
                "on the same sample, so entries within a column are comparable; FPE "
                "is on a determinant scale, so entries across columns are not.",
            ),
        )

    def __repr__(self) -> str:
        """One-line summary naming each criterion's pick."""
        picks = ", ".join(f"{c.upper()}={order}" for c, order in self.selected.items())
        return f"LagOrderSelection(max_lags={self.max_lags}, nobs={self.nobs}, {picks})"


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _MarginalLikelihoodSelection:
    """A log marginal likelihood with its provenance.

    Attributes:
        log_value: The log marginal likelihood of the sample.
        mcse: Monte Carlo standard error of ``log_value``; ``0.0`` for an
            analytic value, ``nan`` when the estimator could not assess it.
        method: How it was obtained -- ``"analytic"``, ``"chib"``,
            ``"modified harmonic mean"``, or ``"bridge sampling"``.
        n_draws: Posterior draws the estimate used; ``0`` for analytic.
        source: The model, for tables.
        nobs: Observations the likelihood scored; ``0`` when unknown. Two
            records are comparable only when this agrees, and ``compare``
            says so when it does not -- a VAR with more lags conditions on
            more presample rows and scores fewer, which alone shifts its
            log marginal likelihood by roughly the per-observation log
            density times the difference.
        notes: Caveats that qualify the number.
    """

    log_value: float
    mcse: float
    method: str
    n_draws: int
    source: str
    nobs: int = 0
    notes: tuple[str, ...] = ()

    def log_bayes_factor(self, other: _MarginalLikelihoodSelection) -> float:
        """``log m_self(y) - log m_other(y)``: positive favours ``self``.

        Meaningful only when both were computed on the same sample; the
        record cannot check that, and the caller must.
        """
        return self.log_value - other.log_value

    def bayes_factor_strength(self, other: _MarginalLikelihoodSelection) -> str:
        """Verbal strength of the evidence for ``self`` over ``other``.

        Args:
            other: The record to compare against.

        Returns:
            A string describing the strength of the evidence for ``self`` over ``other``.
        """
        return _evidence_label(2.0 * self.log_bayes_factor(other))

    def compare(
        self,
        *others: _MarginalLikelihoodSelection,
        names: Sequence[str] | None = None,
        prior_probabilities: Sequence[float] | None = None,
    ) -> SummaryTable:
        """Rank models by evidence with posterior model probabilities.

        Args:
            *others: Further records on the same sample.
            names: Row labels; default each record's ``source``.
            prior_probabilities: Prior model probabilities, one per record;
                default equal.

        Returns:
            One row per model, best first: log marginal likelihood, its
            Monte Carlo error, the log Bayes factor against the best, the
            Kass-Raftery strength of that evidence, and the posterior model
            probability.

        Raises:
            DimensionError: If ``names`` or ``prior_probabilities`` do not
                match the number of records.
            SpecificationError: If a prior probability is negative or all
                are zero.
        """
        records = (self, *others)
        labels = tuple(r.source for r in records) if names is None else tuple(names)
        if len(labels) != len(records):
            raise DimensionError(f"Got {len(labels)} names for {len(records)} records.")
        if prior_probabilities is None:
            prior = np.full(len(records), 1.0 / len(records))
        else:
            prior = np.asarray(prior_probabilities, dtype=np.float64)
            if prior.shape != (len(records),):
                raise DimensionError(
                    f"Got {prior.shape[0] if prior.ndim == 1 else prior.shape} prior "
                    f"probabilities for {len(records)} records."
                )
            if np.any(prior < 0.0) or not prior.sum() > 0.0:
                raise SpecificationError(
                    "Prior model probabilities must be non-negative, not all zero."
                )
            prior = prior / prior.sum()
        values = np.array([r.log_value for r in records])
        with np.errstate(divide="ignore"):
            log_post = values + np.log(prior)
        post = np.exp(log_post - logsumexp(log_post))
        order = np.argsort(values)[::-1]
        best = values[order[0]]
        rows = []
        for i in order:
            record = records[i]
            delta = values[i] - best
            rows.append(
                (
                    labels[i],
                    f"{values[i]:.3f}",
                    f"{record.mcse:.3f}" if np.isfinite(record.mcse) else "nan",
                    f"{delta:.3f}",
                    "best" if i == order[0] else _evidence_label(-2.0 * delta),
                    f"{post[i]:.3f}",
                    record.method,
                )
            )
        notes = [
            "Differences of log ML are log Bayes factors; the strength column is Kass and "
            "Raftery's reading of 2 log BF against the best model.",
        ]
        counts = {r.nobs for r in records if r.nobs > 0}
        if len(counts) > 1:
            notes.append(
                "The records score different numbers of observations "
                f"({', '.join(str(c) for c in sorted(counts))}), so their differences are not "
                "Bayes factors: fit every model on the same effective sample (drop the "
                "leading rows so each conditions on the same presample) and recompute."
            )
        errors = np.array([r.mcse for r in records])
        if np.any(np.isfinite(errors) & (errors > 0.0)):
            close = [
                labels[i]
                for i in order[1:]
                if abs(values[i] - best)
                <= 2.0
                * np.sqrt(np.nan_to_num(errors[i]) ** 2 + np.nan_to_num(errors[order[0]]) ** 2)
            ]
            if close:
                notes.append(
                    "Within two Monte Carlo standard errors of the best, so not separated by "
                    f"this run: {', '.join(close)}."
                )
        for record, label in zip(records, labels, strict=True):
            notes.extend(f"{label}: {note}" for note in record.notes)
        return SummaryTable(
            title="Marginal likelihood comparison",
            metadata=(
                ("Models", str(len(records))),
                ("Prior", "equal" if prior_probabilities is None else "as given"),
            ),
            columns=(
                "model",
                "log ML",
                "mcse",
                "log BF vs best",
                "evidence",
                "post. prob.",
                "method",
            ),
            rows=tuple(rows),
            notes=tuple(notes),
        )

    def __repr__(self) -> str:
        error = f" (mcse {self.mcse:.3f})" if np.isfinite(self.mcse) and self.mcse > 0.0 else ""
        return (
            f"MarginalLikelihood({self.source}: log ML={self.log_value:.3f}{error}, {self.method})"
        )
