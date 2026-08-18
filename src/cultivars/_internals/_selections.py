
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ..exceptions import SpecificationError
from .._core import SummaryTable


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
