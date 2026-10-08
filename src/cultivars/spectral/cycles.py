# filepath: /src/cultivars/spectral/cycles.py
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
r"""Business-cycle chronology: turning points dated from the data, and how two chronologies agree.

The Bry-Boschan rules, in Harding and Pagan's quarterly restatement
(BBQ), date peaks and troughs without a model: a peak is a local
maximum over a two-sided window,

.. math::

   y_t > y_{t \pm k}, \qquad k = 1, \dots, \texttt{window},

turns must alternate, every phase must last a minimum number of
observations and every full cycle a longer minimum, and shorter
episodes are deleted in pairs. That is the NBER committee's judgement
reduced to an algorithm, and the reason to have it is comparison:
against the committee's dates, against a Markov-switching model's
recession probabilities, or across countries. The concordance index of
Harding and Pagan,

.. math::

   I = \frac{1}{T} \sum_{t=1}^{T}
   \bigl[ S_t S'_t + (1 - S_t)(1 - S'_t) \bigr],

is the share of dates two binary chronologies agree on, and the phase
table -- duration, amplitude, and the excess of the actual path over
the triangle joining its ends -- is the standard description of a
cycle's shape.

Two commitments shape the surface. First, the rules are stated and
overridable, and their consequences are reported rather than hidden:
the result carries the triple it was dated under, a contraction
indicator for every date, and a phase table in which a spurious short
phase is visible, because on unsmoothed data the quarterly triple will
date one and the user should see it rather than trust the peak list.
Second, a chronology is a description, not a test. No turn carries a
p-value, the last ``window`` observations cannot be turns, and a phase
open at the sample end is left out of the table; the recent end of a
chronology is provisional in the same way the recent end of a
Christiano-Fitzgerald cycle is, and for the same reason.

Layout. :class:`TurningPoints` resolves the rules and dates a series
into a :class:`TurningPointsResult`, which holds the peaks, troughs,
contraction indicator, and phase table and computes
:meth:`~TurningPointsResult.concordance` against another chronology;
:func:`concordance` is the same index as a function of two
chronologies or probability series. The numerics live in ``_core``:
``_candidate_turns`` runs the local-extremum test, ``_alternate_turns``
thins to an alternating sequence, ``_enforce_durations`` applies the
phase and cycle minima in Harding-Pagan's paired deletions,
``_phase_statistics`` computes the duration, amplitude, and excess of
each completed phase, ``_validate_chronology`` coerces a boolean or
probability series, ``_concordance`` is the index, and
``_TURNING_POINT_RULES`` holds the quarterly and monthly triples. The
band-pass filters in :mod:`~cultivars.spectral.band_pass` supply the
cycle component for a growth-cycle chronology, and
:class:`~cultivars.univariate.regime_switching.MSAR` the model-based
recession probabilities a chronology is most often compared against.

References:
    Bry, G., & Boschan, C. (1971). *Cyclical Analysis of Time Series:
    Selected Procedures and Computer Programs*. NBER.

    Harding, D., & Pagan, A. (2002). Dissecting the cycle: A
    methodological investigation. *Journal of Monetary Economics*,
    49(2), 365-381.

    Harding, D., & Pagan, A. (2006). Synchronization of cycles.
    *Journal of Econometrics*, 132(1), 59-79.

Example:
    A 24-period cycle on a slow trend, dated under the quarterly rules,
    and its concordance with a chronology lagged by three observations:

    >>> import numpy as np
    >>> t = np.arange(120.0)
    >>> y = 0.01 * t + 0.5 * np.sin(2 * np.pi * t / 24)
    >>> res = TurningPoints().date(y)
    >>> res.peaks, res.n_cycles
    (array([  6,  30,  54,  78, 102]), 4)
    >>> lagged = TurningPoints().date(np.roll(y, 3))
    >>> res.concordance(lagged), concordance(res, lagged)
    (0.75, 0.75)
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
    r"""A dated chronology: peaks, troughs, the contraction indicator, and the phase table.

    A peak at :math:`t` is a local maximum over :math:`\pm\texttt{window}`
    observations, a trough a local minimum, with the two alternating and
    every phase at least ``min_phase`` and every cycle at least
    ``min_cycle`` observations long. Each completed phase from one turn
    to the next carries Harding and Pagan's three numbers: its duration
    :math:`D`, its amplitude :math:`A = y_{\text{end}} - y_{\text{start}}`,
    and its excess

    .. math::

       E = \frac{\displaystyle \int_{\text{start}}^{\text{end}}
       \bigl(y_t - \ell_t\bigr)\, dt}{\tfrac{1}{2} D\, |A|},

    the area between the path and the straight line :math:`\ell` joining
    its ends, relative to the triangle that line closes, positive when
    the path bows above the line. A contraction with positive excess
    fell fast and then flattened; an expansion with negative excess
    started slowly and accelerated.

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

    Note:
        The chronology is a statement about the sample's shape under the
        stated rules, not a test: no turn carries a p-value, and a turn
        near either end of the sample is provisional in the same way a
        Christiano-Fitzgerald cycle is, since the local-extremum window
        cannot look past the last observation. The first and last
        ``window`` observations can never be turns, so a series that
        ends mid-phase has that phase open and absent from ``phases``,
        while ``contraction`` still carries the last known state to the
        end.

    See Also:
        * :class:`TurningPoints` -- the producer, with the quarterly and
          monthly rule sets.
        * :func:`concordance` -- the agreement of two chronologies, also
          reachable as :meth:`concordance`.
        * :class:`~cultivars.univariate.regime_switching.MSAR` -- the
          model-based alternative, whose smoothed regime probabilities
          this chronology can be compared against.

    References:
        Harding, D., & Pagan, A. (2002). Dissecting the cycle: A
        methodological investigation. *Journal of Monetary Economics*,
        49(2), 365-381.

        Bry, G., & Boschan, C. (1971). *Cyclical Analysis of Time Series:
        Selected Procedures and Computer Programs*. NBER.

    Example:
        A 24-period cycle on a slow upward trend; the quarterly rules
        find every turn, and the asymmetry of the trend shows in the
        amplitudes:

        >>> import numpy as np
        >>> t = np.arange(120.0)
        >>> y = 0.01 * t + 0.5 * np.sin(2 * np.pi * t / 24)
        >>> res = TurningPoints().date(y)
        >>> res.peaks, res.troughs
        (array([  6,  30,  54,  78, 102]), array([ 18,  42,  66,  90, 114]))
        >>> res.n_cycles, res.nobs, float(res.contraction.mean())
        (4, 120, 0.5)
        >>> start, end, is_contraction, duration, amplitude, excess = res.phases[0]
        >>> (start, end, is_contraction, duration), round(amplitude, 2), round(abs(excess), 3)
        ((6, 18, True, 12), -0.88, 0.0)
        >>> res.concordance(np.roll(res.contraction, 6))
        0.5
    """

    peaks: npt.NDArray[np.int64]
    """Indices of the peaks, ascending; a peak at ``t`` starts a contraction at ``t + 1``."""
    troughs: npt.NDArray[np.int64]
    """Indices of the troughs, ascending; a trough at ``t`` starts an expansion at ``t + 1``."""
    contraction: npt.NDArray[np.bool_] = field(repr=False)
    """``(T,)`` flags, ``True`` on every date in a contraction phase.

    Set from the observation after each peak through the following
    trough; the nearest turn's state is carried to the sample's ends.
    Kept out of the repr.
    """
    phases: tuple[tuple[int, int, bool, int, float, float], ...] = field(repr=False)
    """Completed phases as ``(start, end, is_contraction, duration, amplitude, excess)``.

    In time order, alternating; a phase open at the sample end is not
    included. Kept out of the repr.
    """
    window: int
    """Half-width of the local-extremum test, in observations."""
    min_phase: int
    """Shortest phase the rules allowed, in observations."""
    min_cycle: int
    """Shortest peak-to-peak or trough-to-trough cycle the rules allowed."""
    nobs: int
    """Observations in the dated series, :math:`T`."""

    @property
    def n_cycles(self) -> int:
        """Completed peak-to-peak cycles, one fewer than the number of peaks.

        Example:
            >>> import numpy as np
            >>> t = np.arange(120.0)
            >>> TurningPoints().date(np.sin(2 * np.pi * t / 40)).n_cycles
            2
        """
        return max(int(self.peaks.shape[0]) - 1, 0)

    def concordance(self, other: TurningPointsResult | npt.ArrayLike) -> float:
        r"""Harding-Pagan concordance with another chronology.

        .. math::

           I = \frac{1}{T} \sum_{t=1}^{T}
           \bigl[ S_t S'_t + (1 - S_t)(1 - S'_t) \bigr],

        the share of dates on which the two contraction indicators
        :math:`S` and :math:`S'` agree.

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

        Example:
            >>> import numpy as np
            >>> t = np.arange(120.0)
            >>> y = 0.01 * t + 0.5 * np.sin(2 * np.pi * t / 24)
            >>> res = TurningPoints().date(y)
            >>> lagged = TurningPoints().date(np.roll(y, 3))
            >>> res.concordance(lagged), res.concordance(np.where(res.contraction, 0.8, 0.2))
            (0.75, 1.0)
            >>> res.concordance(np.zeros(50))
            Traceback (most recent call last):
                ...
            cultivars.exceptions.DimensionError: chronologies must align; got lengths 120 and 50.
        """
        return concordance(self.contraction, other)

    def _summary_table(self) -> SummaryTable:
        """Average duration, amplitude and excess by phase type.

        One row for contractions and one for expansions, each with the
        count and the means over the completed phases of that type,
        under a header listing the turn dates and the cycle and
        observation counts; the notes restate the rules and the share
        of dates in contraction, define amplitude and excess, and say
        what the ends of the sample cannot show.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``summary()``,
            ``str()``, and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> t = np.arange(120.0)
            >>> res = TurningPoints().date(0.01 * t + 0.5 * np.sin(2 * np.pi * t / 24))
            >>> table = res._summary_table()
            >>> table.columns
            ('phase', 'count', 'mean duration', 'mean amplitude', 'mean excess')
            >>> table.rows[0][:3], table.metadata[2]
            (('contraction', '5', '12.0'), ('Completed cycles', '4'))
        """
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
    r"""Date peaks and troughs with the Bry-Boschan / Harding-Pagan rules.

    Three rules, applied in order, make a chronology from a series. A
    candidate peak is a local maximum over a two-sided window,

    .. math::

       y_t > y_{t \pm k}, \qquad k = 1, \dots, \texttt{window},

    and a candidate trough its mirror; candidates are then thinned to a
    strictly alternating sequence, the highest of consecutive peaks and
    the lowest of consecutive troughs surviving; and turns are removed
    in pairs until every phase lasts at least ``min_phase`` and every
    peak-to-peak or trough-to-trough cycle at least ``min_cycle``
    observations. The two named conventions are the triples the
    literature uses at quarterly and monthly frequency, and any of the
    three can be overridden.

    Args:
        convention: ``"quarterly"`` for Harding-Pagan's ``(2, 2, 5)`` --
            window, minimum phase, minimum cycle -- or ``"monthly"`` for
            Bry-Boschan's ``(5, 6, 15)``.
        window: Override the local-extremum half-width.
        min_phase: Override the shortest phase.
        min_cycle: Override the shortest cycle.

    Attributes:
        _window: The resolved local-extremum half-width.
        _min_phase: The resolved shortest phase.
        _min_cycle: The resolved shortest cycle.

    Raises:
        SpecificationError: If the convention is unknown or an override
            is not positive.

    Note:
        The rules are a smoothing device and their strength is the
        triple, not the frequency label. The quarterly defaults allow a
        two-observation phase and a five-observation cycle, which on an
        unsmoothed series with a little noise near a turn will date a
        spurious short phase inside the true one -- two peaks a few
        observations apart where there is one. The remedy is the
        ``"monthly"`` triple, a larger ``min_phase``, or a smoothing
        pass upstream, and the ``phases`` table of the result makes the
        short phase visible so the choice is informed. The rules act on
        the level: a series that trends will have longer expansions than
        contractions by construction, which is the classical-cycle
        reading and not a defect; a growth-cycle chronology needs the
        cycle component from a band-pass filter as input instead.

    See Also:
        * :class:`TurningPointsResult` -- the record :meth:`date`
          returns.
        * :func:`concordance` -- agreement between two chronologies.
        * :class:`~cultivars.spectral.band_pass.BaxterKingFilter` -- the
          cycle component for a growth-cycle chronology.

    References:
        Bry, G., & Boschan, C. (1971). *Cyclical Analysis of Time Series:
        Selected Procedures and Computer Programs*. NBER.

        Harding, D., & Pagan, A. (2002). Dissecting the cycle: A
        methodological investigation. *Journal of Monetary Economics*,
        49(2), 365-381.

    Example:
        >>> import numpy as np
        >>> t = np.arange(120.0)
        >>> y = 0.01 * t + 0.5 * np.sin(2 * np.pi * t / 24)
        >>> res = TurningPoints().date(y)
        >>> res.n_cycles, bool(res.contraction.mean() > 0.3)
        (4, True)

        A little noise on a 40-period cycle lets the quarterly triple
        date a spurious short phase at one trough; the monthly triple,
        or a longer minimum phase, does not:

        >>> rng = np.random.default_rng(0)
        >>> noisy = np.sin(2 * np.pi * t / 40) + 0.1 * rng.standard_normal(120)
        >>> TurningPoints().date(noisy).peaks
        array([11, 48, 53, 89])
        >>> TurningPoints(convention="monthly").date(noisy).peaks
        array([11, 48, 89])
        >>> TurningPoints(min_phase=6).date(noisy).peaks
        array([11, 53, 89])
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
        """Resolve the rules.

        The convention's triple is looked up in ``_TURNING_POINT_RULES``
        and each element replaced by its override when one is given;
        the window and minimum phase must be at least one and the
        minimum cycle at least two.

        Args:
            convention: Checked to be ``"quarterly"`` or ``"monthly"``.
            window: Override, checked to be at least one.
            min_phase: Override, checked to be at least one.
            min_cycle: Override, checked to be at least two.

        Raises:
            SpecificationError: If the convention is unknown or an override
                is below its minimum.

        Example:
            >>> rules = TurningPoints(convention="monthly")
            >>> rules._window, rules._min_phase, rules._min_cycle
            (5, 6, 15)
            >>> TurningPoints(convention="weekly")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: convention must be one of ('quarterly', ...
        """
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

        Candidates from the local-extremum test, thinned to alternate,
        then pruned for the duration rules; the contraction indicator is
        filled by walking the surviving turns, with the state before the
        first turn set to contraction when that turn is a trough, and the
        phase table computed over every completed phase.

        Args:
            endog: The series in levels (or logs); the rules act on the
                level, not on a cycle component.

        Returns:
            The :class:`TurningPointsResult`.

        Raises:
            SpecificationError: If the series is too short for the window
                and minimum cycle together.
            DimensionError: If the series is not one-dimensional.
            NumericalError: If the series is not finite.

        Example:
            >>> import numpy as np
            >>> t = np.arange(120.0)
            >>> res = TurningPoints().date(np.sin(2 * np.pi * t / 40))
            >>> res.peaks, res.troughs, res.nobs
            (array([10, 50, 90]), array([ 30,  70, 110]), 120)
            >>> bool(res.contraction[11]) and not bool(res.contraction[31])
            True
            >>> TurningPoints().date(np.zeros(20)).phases
            ()
            >>> TurningPoints().date(np.zeros(8))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: a series of 8 observations is too short ...
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
    r"""Harding-Pagan concordance of two chronologies.

    .. math::

       I = \frac{1}{T} \sum_{t=1}^{T}
       \bigl[ S_t S'_t + (1 - S_t)(1 - S'_t) \bigr],

    the share of dates on which two contraction indicators :math:`S`
    and :math:`S'` agree. Each argument is a :class:`TurningPointsResult`
    or a ``(T,)`` array: booleans are taken as the contraction
    indicator, and a probability series -- a Markov-switching model's
    smoothed recession probability -- is thresholded at one half.

    Args:
        first: One chronology.
        second: The other, on the same dates.

    Returns:
        The share of dates on which the two indicators agree.

    Raises:
        DimensionError: If the lengths differ or an input is not a series.
        NumericalError: If a probability series is non-finite.

    Note:
        One half is the expected concordance of two independent
        chronologies that each spend half their time in contraction; in
        general it is :math:`p p' + (1 - p)(1 - p')` for contraction
        shares :math:`p` and :math:`p'`, so two chronologies that are
        mostly expansion agree by chance well above one half, and a
        reading should be set against that base rate rather than
        against one half. The threshold is strict: a probability of
        exactly one half reads as expansion. An integer 0/1 series is
        read as probabilities and works as expected; a series outside
        ``[0, 1]`` is not refused and is thresholded like any other.

    See Also:
        * :meth:`TurningPointsResult.concordance` -- the same measure
          as a method of a chronology.
        * :class:`~cultivars.univariate.regime_switching.MSAR` -- whose
          smoothed regime probabilities are the usual second argument.

    References:
        Harding, D., & Pagan, A. (2002). Dissecting the cycle: A
        methodological investigation. *Journal of Monetary Economics*,
        49(2), 365-381.

        Harding, D., & Pagan, A. (2006). Synchronization of cycles.
        *Journal of Econometrics*, 132(1), 59-79.

    Example:
        >>> import numpy as np
        >>> a = np.array([True, True, False, False, False])
        >>> concordance(a, np.array([0.9, 0.6, 0.4, 0.1, 0.2]))
        1.0
        >>> concordance(a, ~a), concordance(a, [1, 1, 0, 0, 1])
        (0.0, 0.8)
        >>> concordance(a, np.zeros(4))
        Traceback (most recent call last):
            ...
        cultivars.exceptions.DimensionError: chronologies must align; got lengths 5 and 4.
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
