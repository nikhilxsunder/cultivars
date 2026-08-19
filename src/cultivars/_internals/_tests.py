
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..exceptions import SpecificationError
from .._core import SummaryTable


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _JohansenRankTest:
    """Sequential trace and maximum-eigenvalue tests for the cointegrating rank.

    Both sequences read down from ``r = 0``. The trace statistic tests
    ``rank <= r`` against ``rank = k``; the maximum-eigenvalue statistic tests
    ``rank = r`` against ``rank = r + 1``. The conventional reading stops at the
    first ``r`` that is not rejected, which :meth:`selected_rank` implements and
    which is a sequential procedure whose overall size is not the nominal level
    of any single step -- a caveat the table states rather than hides.

    Attributes:
        eigenvalues: All ``k`` squared canonical correlations, descending.
        trace_statistic: One entry per null ``rank <= r``, ``r = 0 .. k - 1``.
        max_eigenvalue_statistic: One entry per null ``rank = r``.
        trace_pvalue: Empirical p-values from the simulated null.
        max_eigenvalue_pvalue: Empirical p-values from the simulated null.
        nobs: Effective sample the statistics were computed on.
        deterministic: The Johansen case the null was simulated under.
        simulations: Replications behind each p-value, so its resolution is
            visible rather than implied.
        small_sample: Whether the Reinsel-Ahn degrees-of-freedom scaling was
            applied to the statistics.
    """

    eigenvalues: npt.NDArray[np.float64]
    trace_statistic: npt.NDArray[np.float64]
    max_eigenvalue_statistic: npt.NDArray[np.float64]
    trace_pvalue: npt.NDArray[np.float64]
    max_eigenvalue_pvalue: npt.NDArray[np.float64]
    nobs: int
    deterministic: str
    simulations: int
    small_sample: bool

    @property
    def k_endog(self) -> int:
        """Number of variables in the system."""
        return int(self.eigenvalues.shape[0])

    @property
    def ranks(self) -> tuple[int, ...]:
        """The null ranks tested, in order."""
        return tuple(range(self.k_endog))

    def selected_rank(self, *, alpha: float = 0.05, statistic: str = "trace") -> int:
        """First rank the sequence fails to reject.

        Args:
            alpha: Level for each step of the sequence.
            statistic: ``"trace"`` or ``"max_eigenvalue"``.

        Returns:
            The chosen rank, or ``k_endog`` if every null is rejected, which is
            the finding that the system is stationary in levels and does not
            want a vector error-correction representation at all.

        Raises:
            SpecificationError: If the level or the statistic is unrecognized.
        """
        if not 0.0 < alpha < 1.0:
            raise SpecificationError(f"alpha must lie strictly in (0, 1); got {alpha}.")
        if statistic == "trace":
            pvalues = self.trace_pvalue
        elif statistic == "max_eigenvalue":
            pvalues = self.max_eigenvalue_pvalue
        else:
            raise SpecificationError(
                f"statistic must be 'trace' or 'max_eigenvalue'; got {statistic!r}."
            )
        for rank, pvalue in enumerate(pvalues):
            if pvalue >= alpha:
                return rank
        return self.k_endog

    def to_table(self, *, alpha: float = 0.05) -> SummaryTable:
        """Render both sequences with the rank each one selects."""
        chosen = self.selected_rank(alpha=alpha)
        notes = [
            f"Read down from r = 0 and stop at the first row not rejected: trace selects "
            f"r = {chosen}, max-eigenvalue selects "
            f"r = {self.selected_rank(alpha=alpha, statistic='max_eigenvalue')}.",
            "This is a sequential procedure, so the overall probability of an incorrect "
            f"rank is not the {alpha:.0%} attached to any single row.",
            f"P-values are empirical over {self.simulations:,} draws from the simulated "
            f"asymptotic null for the '{self.deterministic}' case, so the smallest "
            f"resolvable value is {1 / self.simulations:.0e}.",
        ]
        if self.small_sample:
            notes.append(
                "Statistics carry the Reinsel-Ahn degrees-of-freedom scaling, which pulls "
                "the well-known upward size distortion of the asymptotic test back toward "
                "its nominal level in short samples."
            )
        return SummaryTable(
            title="Johansen cointegration rank test",
            metadata=(
                ("Observations", f"{self.nobs}"),
                ("Variables", f"{self.k_endog}"),
                ("Deterministic", self.deterministic),
                ("Selected rank", f"{chosen}"),
            ),
            columns=("H0", "eigenvalue", "trace", "P>trace", "max-eig", "P>max-eig"),
            rows=tuple(
                (
                    f"r <= {rank}",
                    f"{self.eigenvalues[rank]:.4f}",
                    f"{self.trace_statistic[rank]:.3f}",
                    f"{self.trace_pvalue[rank]:.4f}",
                    f"{self.max_eigenvalue_statistic[rank]:.3f}",
                    f"{self.max_eigenvalue_pvalue[rank]:.4f}",
                )
                for rank in self.ranks
            ),
            notes=tuple(notes),
        )

    def __repr__(self) -> str:
        """One line with the rank each sequence selects."""
        return (
            f"JohansenRankTest(k={self.k_endog}, nobs={self.nobs}, "
            f"deterministic={self.deterministic!r}, "
            f"trace_rank={self.selected_rank()}, "
            f"max_eigenvalue_rank={self.selected_rank(statistic='max_eigenvalue')})"
        )