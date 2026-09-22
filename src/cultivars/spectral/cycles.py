"""Business-cycle chronology: turning points dated from the data, and how two chronologies agree.

The Bry-Boschan rules, in Harding and Pagan's quarterly restatement
(BBQ), date peaks and troughs without a model: a peak is a local
maximum over a two-sided window, turns must alternate, every phase must
last a minimum number of observations and every full cycle a longer
minimum, and shorter episodes are deleted in pairs. That is the NBER
committee's judgement reduced to an algorithm, and the reason to have
it is comparison: against the committee's dates, against a Markov-
switching model's recession probabilities, or across countries. The
concordance index of Harding and Pagan is the share of dates two binary
chronologies agree on, and the phase table -- duration, amplitude, and
the excess of the actual path over the triangle joining its ends --
is the standard description of a cycle's shape.

References:
    Bry, G., & Boschan, C. (1971). *Cyclical Analysis of Time Series:
        Selected Procedures and Computer Programs*. NBER.
    Harding, D., & Pagan, A. (2002). Dissecting the cycle: A
        methodological investigation. *Journal of Monetary Economics*,
        49(2), 365-381.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core import (
    _TURNING_POINT_RULES,
    SummaryTable,
    _alternate_turns,
    _candidate_turns,
    _concordance,
    _enforce_durations,
    _phase_statistics,
    _validate_chronology,
    validate_endog,
    validate_order,
)
from .._internals import _SummaryMixin
from ..exceptions import DimensionError, SpecificationError

__all__ = ["TurningPoints", "TurningPointsResult", "concordance"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class TurningPointsResult(_SummaryMixin):
    """A dated chronology: peaks, troughs, the contraction indicator, and the phase table.

    Attributes:
        peaks: Indices of the peaks, in time order.
        troughs: Indices of the troughs, in time order.
        contraction: ``(T,)`` flags, ``True`` from the observation after a
            peak through the following trough. Before the first turn and
            after the last the state is carried from the nearest turn.
        phases: One row per phase ``(start, end, is_contraction,
            duration, amplitude, excess)``: the change in the series
            over the phase, and Harding-Pagan's excess -- the area
            between the path and the straight line joining its ends,
            divided by the triangle's area, positive when the path bows
            above the line.
        window: Half-width of the local-extremum test.
        min_phase: Shortest phase allowed.
        min_cycle: Shortest peak-to-peak or trough-to-trough cycle allowed.
        nobs: Observations.
    """

    peaks: npt.NDArray[np.int64]
    troughs: npt.NDArray[np.int64]
    contraction: npt.NDArray[np.bool_] = field(repr=False)
    phases: tuple[tuple[int, int, bool, int, float, float], ...] = field(repr=False)
    window: int
    min_phase: int
    min_cycle: int
    nobs: int

    @property
    def n_cycles(self) -> int:
        """Completed peak-to-peak cycles."""
        return max(int(self.peaks.shape[0]) - 1, 0)

    def concordance(self, other: TurningPointsResult | npt.ArrayLike) -> float:
        """Harding-Pagan concordance with another chronology.

        Args:
            other: A :class:`TurningPointsResult` on the same dates, or a
                ``(T,)`` boolean or probability series -- a Markov-
                switching model's recession probabilities, say -- which
                is thresholded at one half.

        Returns:
            The share of dates on which the two contraction indicators
            agree; one half is what independent chronologies of equal
            frequency would produce on average.

        Raises:
            DimensionError: If the other chronology has a different length.
        """
        return concordance(self.contraction, other)

    def _summary_table(self) -> SummaryTable:
        """Average duration, amplitude and excess by phase type."""
        rows = []
        for label, is_contraction in (("contraction", True), ("expansion", False)):
            matching = [p for p in self.phases if p[2] == is_contraction]
            if not matching:
                rows.append((label, "0", "", "", ""))
                continue
            duration = np.mean([p[3] for p in matching])
            amplitude = np.mean([p[4] for p in matching])
            excess = np.mean([p[5] for p in matching])
            rows.append(
                (
                    label,
                    str(len(matching)),
                    f"{duration:.1f}",
                    f"{amplitude:+.4f}",
                    f"{excess:+.3f}",
                )
            )
        share = float(self.contraction.mean())
        notes = (
            f"Rules: local extremum over +-{self.window}, phases of at least {self.min_phase} "
            f"and cycles of at least {self.min_cycle} observations, turns alternating; "
            f"{100 * share:.1f}% of dates are in contraction.",
            "Amplitude is the change in the series over the phase; excess is the area between "
            "the path and the line joining its ends relative to the triangle, positive for a "
            "path that bows above the line (a fast fall then flattening, in a contraction).",
            "The first and last `window` observations cannot be turns, and a phase still open "
            "at the sample end is not in the table.",
        )
        return SummaryTable(
            title="Turning Points",
            metadata=(
                ("Peaks", ", ".join(str(int(p)) for p in self.peaks) or "none"),
                ("Troughs", ", ".join(str(int(t)) for t in self.troughs) or "none"),
                ("Completed cycles", str(self.n_cycles)),
                ("Observations", str(self.nobs)),
            ),
            columns=("phase", "count", "mean duration", "mean amplitude", "mean excess"),
            rows=tuple(rows),
            notes=notes,
        )


class TurningPoints:
    """Date peaks and troughs with the Bry-Boschan / Harding-Pagan rules.

    Args:
        convention: ``"quarterly"`` for Harding-Pagan's ``(2, 2, 5)`` --
            window, minimum phase, minimum cycle -- or ``"monthly"`` for
            Bry-Boschan's ``(5, 6, 15)``.
        window: Override the local-extremum half-width.
        min_phase: Override the shortest phase.
        min_cycle: Override the shortest cycle.

    Raises:
        SpecificationError: If the convention is unknown or an override
            is not positive.

    Example:
        >>> t = np.arange(120.0)
        >>> y = 0.01 * t + 0.5 * np.sin(2 * np.pi * t / 24)
        >>> res = TurningPoints().date(y)
        >>> res.n_cycles, bool(res.contraction.mean() > 0.3)
        (4, True)
    """

    __slots__ = ("_min_cycle", "_min_phase", "_window")

    def __init__(
        self,
        *,
        convention: str = "quarterly",
        window: int | None = None,
        min_phase: int | None = None,
        min_cycle: int | None = None,
    ) -> None:
        """Resolve the rules."""
        if convention not in _TURNING_POINT_RULES:
            raise SpecificationError(
                f"convention must be one of {tuple(_TURNING_POINT_RULES)}; got {convention!r}."
            )
        base_window, base_phase, base_cycle = _TURNING_POINT_RULES[convention]
        self._window = validate_order(
            base_window if window is None else window, "window", minimum=1
        )
        self._min_phase = validate_order(
            base_phase if min_phase is None else min_phase, "min_phase", minimum=1
        )
        self._min_cycle = validate_order(
            base_cycle if min_cycle is None else min_cycle, "min_cycle", minimum=2
        )

    def date(self, endog: npt.ArrayLike) -> TurningPointsResult:
        """Date the chronology of a series.

        Args:
            endog: The series in levels (or logs); the rules act on the
                level, not on a cycle component.

        Returns:
            The :class:`TurningPointsResult`.

        Raises:
            SpecificationError: If the series is too short for the window.
        """
        y = validate_endog(endog)
        n = y.shape[0]
        if n < 2 * self._window + self._min_cycle:
            raise SpecificationError(
                f"a series of {n} observations is too short to date with window {self._window} "
                f"and minimum cycle {self._min_cycle}."
            )
        peaks, troughs = _candidate_turns(y, self._window)
        turns = _alternate_turns(y, peaks, troughs)
        turns = _enforce_durations(y, turns, self._min_phase, self._min_cycle)
        peak_index = np.asarray([i for i, p in turns if p], dtype=np.int64)
        trough_index = np.asarray([i for i, p in turns if not p], dtype=np.int64)
        contraction = np.zeros(n, dtype=np.bool_)
        if turns:
            # Before the first turn the economy is contracting if that turn is a trough.
            state = not turns[0][1]
            position = 0
            for index, is_peak in turns:
                contraction[position : index + 1] = state
                state = is_peak
                position = index + 1
            contraction[position:] = state
        return TurningPointsResult(
            peaks=peak_index,
            troughs=trough_index,
            contraction=contraction,
            phases=_phase_statistics(y, turns),
            window=self._window,
            min_phase=self._min_phase,
            min_cycle=self._min_cycle,
            nobs=n,
        )


def concordance(
    first: TurningPointsResult | npt.ArrayLike, second: TurningPointsResult | npt.ArrayLike
) -> float:
    """Harding-Pagan concordance of two chronologies.

    Each argument is a :class:`TurningPointsResult` or a ``(T,)`` array:
    booleans are taken as the contraction indicator, and a probability
    series -- a Markov-switching model's smoothed recession probability
    -- is thresholded at one half.

    Args:
        first: One chronology.
        second: The other, on the same dates.

    Returns:
        The share of dates on which the two indicators agree.

    Raises:
        DimensionError: If the lengths differ or an input is not a series.
        NumericalError: If a probability series is non-finite.

    Example:
        >>> a = np.array([True, True, False, False, False])
        >>> concordance(a, np.array([0.9, 0.6, 0.4, 0.1, 0.2]))
        1.0
    """
    a = first.contraction if isinstance(first, TurningPointsResult) else _validate_chronology(first)
    b = (
        second.contraction
        if isinstance(second, TurningPointsResult)
        else _validate_chronology(second)
    )
    if a.shape != b.shape:
        raise DimensionError(f"chronologies must align; got lengths {a.shape[0]} and {b.shape[0]}.")
    return _concordance(a, b)
