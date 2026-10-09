# filepath: /src/cultivars/spectral/filters.py
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
r"""Trend-cycle filters: four answers to "what is the trend", each with its bill.

A trend-cycle decomposition is a definition, not a discovery, and the
four here define the trend differently. Hodrick-Prescott and Butterworth
are penalized low-pass filters: the trend is the minimizer of

.. math::

   \sum_t (y_t - \tau_t)^2 + \lambda \sum_t (\Delta^n \tau_t)^2,

whose two-sided gain :math:`\bigl[1 + (\sin(\omega/2)/\sin(\omega_c/2))^{2n}
\bigr]^{-1}` is one half at the cutoff :math:`\omega_c = 2\pi/P`, with
:math:`\lambda = (2\sin(\pi/P))^{-2n}`; HP is the case :math:`n = 2`
stated by weight (1600 quarterly is :math:`P = 39.7`), Butterworth the
same family stated by period and order. Hamilton's regression filter
defines the trend as the fitted value of :math:`y_{t+h}` on a constant
and :math:`y_t, \dots, y_{t-p+1}`, so the cycle is what the level's own
recent history could not predict two years ahead; it exists because HP
manufactures dynamics -- a spurious cycle from a random walk, larger than
the increments themselves; end-point values that change as data arrive;
a penalty with no rationale at other frequencies -- and the HP result
prints that critique rather than assuming the user knows it.
Beveridge-Nelson is model-based: for a fitted ARIMA(p, 1, q) the trend is
the level the long-horizon forecast settles on net of drift,

.. math::

   \tau_t = y_t + \sum_{k \ge 1}\bigl(E_t \Delta y_{t+k} - \mu\bigr)
   = y_t + e_1' F (I - F)^{-1} x_t,

evaluated exactly in companion form, so the cycle is minus the
forecastable momentum of the differences -- which is why a BN cycle is
small, noisy, and negatively correlated with growth where the others are
smooth. Not a bug; a different question.

Two commitments shape the surface. First, every method's bill is
carried on the result: :class:`DecompositionFilterResult` records the
rows a one-sided method lost (``offset``) and flags the rows a two-sided
method will revise (``provisional``, the first and last ``cutoff``
observations of a penalized filter), and its summary prints the cycle's
standard deviation against that of the input's growth -- the diagnostic
on which Hamilton's critique rests -- followed by the method's own
caveats. Second, nothing here estimates a model: the three filters that
touch data are linear operations on a panel, column by column, and the
Beveridge-Nelson decomposition reads its answer off an ARIMA someone
else fitted, so a decomposition is never confused with an inference.

Layout. :class:`HodrickPrescottFilter`, :class:`ButterworthFilter` and
:class:`HamiltonFilter` take a panel through ``filter``;
:class:`BeveridgeNelsonDecomposition` takes a fitted
:class:`~cultivars.univariate.box_jenkins.ARIMA` result through
``compute``; all four return :class:`DecompositionFilterResult`, whose
``_penalized_result`` classmethod is the shared path of the first two.
The numerics live in ``_core``: ``_penalized_trend`` solves
:math:`(I + \lambda D'D)\tau = y` in banded form,
``_butterworth_penalty`` and ``_penalty_cutoff_period`` convert between
weight and cutoff period, ``_hamilton_regression`` runs the one-column
OLS, ``_beveridge_nelson`` forms the companion-form forecast sum,
``_METHOD_TITLES`` names the methods in summaries, and
``validate_endog_matrix`` and ``validate_order`` guard the inputs.
``_SummaryMixin`` from ``_internals`` gives the result its printed
form. The band-pass filters that isolate a cycle band rather than
remove a trend are in :mod:`~cultivars.spectral.band_pass`, and the
dating of the cycle those filters produce in
:mod:`~cultivars.spectral.cycles`.

References:
    Hodrick, R. J., & Prescott, E. C. (1997). Postwar U.S. business
    cycles: An empirical investigation. *Journal of Money, Credit and
    Banking*, 29(1), 1-16.

    Ravn, M. O., & Uhlig, H. (2002). On adjusting the Hodrick-Prescott
    filter for the frequency of observations. *Review of Economics and
    Statistics*, 84(2), 371-376.

    Hamilton, J. D. (2018). Why you should never use the Hodrick-Prescott
    filter. *Review of Economics and Statistics*, 100(5), 831-843.

    Gomez, V. (2001). The use of Butterworth filters for trend and cycle
    estimation in economic time series. *Journal of Business & Economic
    Statistics*, 19(3), 365-373.

    Beveridge, S., & Nelson, C. R. (1981). A new approach to
    decomposition of economic time series into permanent and transitory
    components. *Journal of Monetary Economics*, 7(2), 151-174.

    Morley, J. C. (2002). A state-space approach to calculating the
    Beveridge-Nelson decomposition. *Economics Letters*, 75(1), 123-127.

Example:
    The same random walk through three definitions. HP finds a cycle
    larger than the increments and flags 80 of 200 rows as provisional;
    Hamilton loses eleven rows and reports the eight-step forecast
    error; Beveridge-Nelson, told the truth about the process, finds no
    cycle at all:

    >>> import numpy as np
    >>> from cultivars.univariate.box_jenkins import ARIMA
    >>> rng = np.random.default_rng(0)
    >>> y = np.cumsum(rng.standard_normal(200))
    >>> hp = HodrickPrescottFilter().filter(y)
    >>> round(float(hp.cycle.std() / np.diff(y).std()), 1), int(hp.provisional.sum())
    (1.5, 80)
    >>> ham = HamiltonFilter().filter(y)
    >>> ham.offset, bool(abs(ham.cycle.std() / np.sqrt(8) - 1.0) < 0.1)
    (11, True)
    >>> bn = BeveridgeNelsonDecomposition(ARIMA(y, order=(0, 1, 0)).fit()).compute()
    >>> float(np.abs(bn.cycle).max())
    0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..engine._core import (
    _METHOD_TITLES,
    SummaryTable,
    _beveridge_nelson,
    _butterworth_penalty,
    _hamilton_regression,
    _penalized_trend,
    _penalty_cutoff_period,
    validate_endog_matrix,
    validate_order,
)
from ..engine._internals import _SummaryMixin
from ..exceptions import SpecificationError
from ..univariate.box_jenkins import ARMAResult

__all__ = [
    "BeveridgeNelsonDecomposition",
    "ButterworthFilter",
    "DecompositionFilterResult",
    "HamiltonFilter",
    "HodrickPrescottFilter",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class DecompositionFilterResult(_SummaryMixin):
    r"""A trend-cycle decomposition of a panel, with the bookkeeping the method needs.

    One container for four definitions of the trend. The penalized
    filters (Hodrick-Prescott, Butterworth) take the trend to be the
    minimizer of

    .. math::

        \sum_t (y_t - \tau_t)^2 + \lambda \sum_t (\Delta^n \tau_t)^2,

    a two-sided low-pass whose gain is one half at the period recorded
    in ``cutoff``. Hamilton's filter takes the trend to be the fitted
    value of the regression of :math:`y_{t+h}` on a constant and
    :math:`y_t, \dots, y_{t-p+1}`, so the cycle is what the level's own
    recent history could not predict :math:`h` steps ahead. The
    Beveridge-Nelson decomposition takes the trend of an ARIMA(p, 1, q)
    to be the level its long-horizon forecast settles on net of drift,

    .. math::

        \tau_t = y_t + \sum_{k \ge 1}\bigl(E_t \Delta y_{t+k} - \mu\bigr),

    so the cycle is minus the forecastable momentum of the differences.
    Whatever the definition, ``trend + cycle`` reproduces the input over
    the rows kept, and the result records which rows are missing
    (``offset``) and which will change as data arrive (``provisional``).

    The summary prints the cycle's standard deviation per series, in
    levels and relative to the standard deviation of the input's first
    difference, followed by the method's own caveats: the HP and
    Butterworth notes say how many rows are provisional and restate
    Hamilton's (2018) critique; the Hamilton note says the filter is
    one-sided; the Beveridge-Nelson note explains why its cycle is small.

    Attributes:
        trend: ``(rows, k)`` trend component; the first row aligns with
            observation ``offset`` of the input.
        cycle: ``(rows, k)`` cycle, the input minus the trend over the
            same rows.
        method: ``"hodrick-prescott"``, ``"butterworth"``, ``"hamilton"``
            or ``"beveridge-nelson"``.
        offset: Input observations lost before the first row.
        penalty: The smoothing weight of a penalized filter; ``None``
            otherwise.
        order: Differences penalized, or the ARMA ``(p, q)`` of a
            Beveridge-Nelson decomposition; ``None`` for Hamilton.
        cutoff: Period at which a penalized filter's gain is one half;
            ``None`` otherwise.
        horizon: Hamilton's forecast horizon; ``None`` otherwise.
        lags: Hamilton's regressors; ``None`` otherwise.
        provisional: ``(rows,)`` flags marking rows a two-sided filter
            will revise as data arrive -- the ends of a penalized
            filter, within ``cutoff`` observations of either edge; all
            ``False`` for the one-sided Hamilton and Beveridge-Nelson.

    Note:
        The ratio the summary prints is the diagnostic Hamilton (2018)
        builds his case on. A random walk has no cycle, yet the HP cycle
        of a random walk has a standard deviation *larger* than that of
        its increments (about 1.5 times at the quarterly penalty), because
        the filter manufactures a smooth component from the level's
        wandering. The same series through the Hamilton filter reports a
        cycle whose scale is the eight-step forecast error, which is what
        a random walk's unpredictable component actually is. Neither
        number is wrong; they answer different questions, and the ratio
        makes the question visible.

        ``provisional`` is a strong statement about the penalized
        filters: at HP's quarterly 1600 the cutoff is 39.7 periods, so
        the first and last forty observations of any sample -- ten years
        at each end of a fifty-year quarterly history -- are flagged.
        When the sample is shorter than twice the cutoff every row is
        provisional.

    See Also:
        * :class:`HodrickPrescottFilter`, :class:`ButterworthFilter`,
          :class:`HamiltonFilter`, :class:`BeveridgeNelsonDecomposition`
          -- the producers.
        * :class:`~cultivars.spectral.band_pass.BandPassResult` -- the
          same reporting convention for the band-pass filters, which
          isolate a cycle band rather than remove a trend.

    References:
        Hodrick, R. J., & Prescott, E. C. (1997). Postwar U.S. business
        cycles: An empirical investigation. *Journal of Money, Credit and
        Banking*, 29(1), 1-16.

        Hamilton, J. D. (2018). Why you should never use the
        Hodrick-Prescott filter. *Review of Economics and Statistics*,
        100(5), 831-843.

        Gomez, V. (2001). The use of Butterworth filters for trend and
        cycle estimation in economic time series. *Journal of Business &
        Economic Statistics*, 19(3), 365-373.

        Beveridge, S., & Nelson, C. R. (1981). A new approach to
        decomposition of economic time series into permanent and
        transitory components. *Journal of Monetary Economics*, 7(2),
        151-174.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(rng.standard_normal(200))
        >>> hp = HodrickPrescottFilter().filter(y)
        >>> hp.method, hp.offset, hp.n_series, int(hp.provisional.sum())
        ('hodrick-prescott', 0, 1, 80)
        >>> bool(np.allclose(hp.trend + hp.cycle, y[:, None]))
        True

        The random walk has no cycle; HP finds one larger than the
        increments, Hamilton reports the forecast error instead:

        >>> round(float(hp.cycle.std() / np.diff(y).std()), 1)
        1.5
        >>> ham = HamiltonFilter().filter(y)
        >>> ham.offset, ham.trend.shape, bool(ham.provisional.any())
        (11, (189, 1), False)
        >>> print(hp)  # doctest: +ELLIPSIS
                           Hodrick-Prescott Trend-Cycle Decomposition
        ===...
        Method:               hodrick-prescott  Rows:                              200
        Penalty:                          1600  Cutoff period:                    39.7
        Order:                               2
        ---...
                     cycle sd   cycle sd / growth sd
        series 1       1.4437                  1.498
        ===...
        Two-sided; the 80 rows flagged `provisional` lie within the cutoff period of an
        edge and will be revised as data arrive.
        Hamilton (2018): ...
    """

    trend: npt.NDArray[np.float64] = field(repr=False)
    """``(rows, k)`` trend component. Kept out of the repr."""

    cycle: npt.NDArray[np.float64] = field(repr=False)
    """``(rows, k)`` cycle, the input minus the trend. Kept out of the repr."""

    method: str
    """Which definition of the trend produced the rows."""

    offset: int
    """Input observations lost before the first row."""

    penalty: float | None = None
    """Smoothing weight of a penalized filter; ``None`` otherwise."""

    order: int | tuple[int, int] | None = None
    """Differences penalized, or the ARMA ``(p, q)``; ``None`` for Hamilton."""

    cutoff: float | None = None
    """Period at which a penalized filter's gain is one half; ``None`` otherwise."""

    horizon: int | None = None
    """Hamilton's forecast horizon; ``None`` otherwise."""

    lags: int | None = None
    """Hamilton's regressors; ``None`` otherwise."""

    provisional: npt.NDArray[np.bool_] = field(
        repr=False, default_factory=lambda: np.zeros(0, bool)
    )
    """``(rows,)`` flags for rows a two-sided filter will revise. Kept out of the repr."""

    @property
    def n_series(self) -> int:
        """Number of series filtered.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> panel = np.cumsum(rng.standard_normal((120, 3)), axis=0)
            >>> HodrickPrescottFilter(penalty=100).filter(panel).n_series
            3
        """
        return int(self.trend.shape[1])

    def _summary_table(self) -> SummaryTable:
        """Cycle scale per series, absolute and against the input's growth, with the caveats.

        One row per series: the cycle's standard deviation and its ratio
        to the standard deviation of the reconstructed level's first
        difference. The metadata block carries whichever of ``offset``,
        ``penalty``, ``cutoff``, ``order``, ``horizon`` and ``lags`` the
        method set; the notes come from :meth:`_method_notes`.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(rng.standard_normal(200))
            >>> table = HodrickPrescottFilter().filter(y)._summary_table()
            >>> table.title
            'Hodrick-Prescott Trend-Cycle Decomposition'
            >>> table.metadata[2:]
            (('Penalty', '1600'), ('Cutoff period', '39.7'), ('Order', '2'))
            >>> table.rows
            (('series 1', '1.4437', '1.498'),)
        """
        levels = self.trend + self.cycle
        growth = np.diff(levels, axis=0).std(axis=0)
        rows = tuple(
            (
                f"series {index + 1}",
                f"{self.cycle[:, index].std():.4f}",
                f"{self.cycle[:, index].std() / max(float(growth[index]), 1e-300):.3f}",
            )
            for index in range(self.n_series)
        )
        metadata = [("Method", self.method), ("Rows", str(self.trend.shape[0]))]
        if self.offset:
            metadata.append(("Offset", str(self.offset)))
        if self.penalty is not None:
            metadata.append(("Penalty", f"{self.penalty:g}"))
        if self.cutoff is not None:
            metadata.append(("Cutoff period", f"{self.cutoff:.1f}"))
        if self.order is not None:
            metadata.append(("Order", str(self.order)))
        if self.horizon is not None:
            metadata.append(("Horizon", str(self.horizon)))
        if self.lags is not None:
            metadata.append(("Lags", str(self.lags)))
        return SummaryTable(
            title=f"{_METHOD_TITLES[self.method]} Trend-Cycle Decomposition",
            metadata=tuple(metadata),
            columns=("", "cycle sd", "cycle sd / growth sd"),
            rows=rows,
            notes=self._method_notes(),
        )

    def _method_notes(self) -> tuple[str, ...]:
        """What the method's definition of the trend implies for reading the cycle.

        The penalized filters report the provisional row count and, for
        Hodrick-Prescott, Hamilton's (2018) critique; Hamilton's filter
        states its one-sidedness and the rows it discards;
        Beveridge-Nelson, the fall-through branch, explains why its cycle
        is small and volatile.

        Returns:
            One or two note strings for the summary's closing block.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(rng.standard_normal(200))
            >>> notes = HamiltonFilter().filter(y)._method_notes()
            >>> len(notes), notes[0][:47]
            (2, 'One-sided by construction: the cycle at t is wh')
            >>> ButterworthFilter(period=32).filter(y)._method_notes()[0][:60]
            'Two-sided Butterworth sine filter of Gomez (2001); the 64 ro'
        """
        edge = int(self.provisional.sum())
        if self.method == "hodrick-prescott":
            return (
                f"Two-sided; the {edge} rows flagged `provisional` lie within the cutoff "
                "period of an edge and will be revised as data arrive.",
                "Hamilton (2018): the HP filter produces cycles with no basis in the "
                "data-generating process (a random walk acquires a smooth cycle), its "
                "end-point values differ from its interior ones, and the penalty 1600 has "
                "no rationale at other frequencies. Prefer HamiltonFilter for inference; "
                "use HP where comparability with the literature is the point.",
            )
        if self.method == "butterworth":
            return (
                f"Two-sided Butterworth sine filter of Gomez (2001); the {edge} rows flagged "
                "`provisional` lie within the cutoff period of an edge. Higher order "
                "sharpens the cutoff and worsens the end-point problem; order 2 is HP.",
            )
        if self.method == "hamilton":
            return (
                "One-sided by construction: the cycle at t is what the level h steps "
                "earlier could not predict, so no row is revised; the first h + p - 1 "
                "observations carry no value.",
                "The regression on levels is valid for I(1) and I(2) series alike, "
                "and the cycle is stationary by construction.",
            )
        return (
            "Beveridge-Nelson: the trend is the long-horizon forecast net of drift, so the "
            "cycle is the forecastable momentum of the differences. It is typically small "
            "and volatile, and its sign follows the ARMA dynamics rather than a smooth "
            "hump; a large HP cycle and a small BN cycle on the same series are both "
            "correct answers to different questions.",
        )

    @classmethod
    def _penalized_result(
        cls, panel: npt.NDArray[np.float64], penalty: float, order: int, method: str
    ) -> DecompositionFilterResult:
        r"""Run the penalized smoother and package it.

        Solves :math:`(I + \lambda D'D)\tau = y` column by column, records
        the cutoff period :math:`\pi / \arcsin(\lambda^{-1/2n} / 2)`, and
        flags the first and last :math:`\lceil \text{cutoff} \rceil` rows
        (capped at half the sample) as provisional. Shared by
        :class:`HodrickPrescottFilter` and :class:`ButterworthFilter`,
        which differ only in how they arrive at ``penalty`` and ``order``.

        Args:
            panel: The validated ``(nobs, k)`` panel.
            penalty: The smoothing weight :math:`\lambda`.
            order: Differences penalized, :math:`n`.
            method: ``"hodrick-prescott"`` or ``"butterworth"``.

        Returns:
            The :class:`DecompositionFilterResult` with every row kept.

        Example:
            >>> t = np.arange(60.0)[:, None]
            >>> res = DecompositionFilterResult._penalized_result(
            ...     2.0 + 0.5 * t, 1600.0, 2, "hodrick-prescott"
            ... )
            >>> bool(np.abs(res.cycle).max() < 1e-8), round(res.cutoff, 1)
            (True, 39.7)
            >>> int(res.provisional.sum()), bool(res.provisional.all())
            (60, True)
        """
        trend = _penalized_trend(panel, penalty, order)
        cutoff = _penalty_cutoff_period(penalty, order)
        rows = panel.shape[0]
        edge = min(int(np.ceil(cutoff)), rows // 2)
        provisional = np.zeros(rows, dtype=np.bool_)
        provisional[:edge] = True
        provisional[rows - edge :] = True
        return cls(
            trend=trend,
            cycle=panel - trend,
            method=method,
            offset=0,
            penalty=penalty,
            order=order,
            cutoff=cutoff,
            provisional=provisional,
        )


class HodrickPrescottFilter:
    r"""The Hodrick-Prescott filter, with Hamilton's critique on the result.

    The trend is the minimizer of

    .. math::

        \sum_{t=1}^{T} (y_t - \tau_t)^2
        + \lambda \sum_{t=3}^{T} (\Delta^2 \tau_t)^2,

    the order-2 case of the penalized smoother, whose two-sided gain is

    .. math::

        G(\omega) = \frac{1}{1 + 4\lambda(1 - \cos\omega)^2},

    a low-pass that lets a linear trend through untouched and is one
    half at the period the result reports as ``cutoff``. The
    conventional penalties are 1600 quarterly, 100 annual and 14400
    monthly (Ravn-Uhlig's frequency-power rule gives 6.25 and 129600
    instead), which is to say cutoff periods of 39.7, 19.8, 68.8, 9.8
    and 119.2 observations; as :math:`\lambda \to \infty` the trend
    becomes the least-squares line.

    Attributes:
        _penalty: The smoothing weight :math:`\lambda`.

    Args:
        penalty: The smoothing weight :math:`\lambda`, positive.

    Raises:
        SpecificationError: If the penalty is not positive.

    Note:
        The result's summary carries Hamilton's (2018) critique in its
        notes rather than assuming the user knows it: the filter
        manufactures a smooth cycle from a random walk, its end-point
        values are revised as data arrive (the rows the result flags as
        ``provisional``), and 1600 has no rationale outside quarterly
        data. Use it where comparability with the literature is the
        point; use :class:`HamiltonFilter` where the cycle will be
        subject to inference. :class:`ButterworthFilter` at order 2 and
        the matching cutoff period is the same filter stated by
        frequency rather than by weight.

    See Also:
        * :class:`DecompositionFilterResult` -- the container returned.
        * :class:`ButterworthFilter` -- the same penalized family,
          parameterized by cutoff period and order.
        * :class:`HamiltonFilter` -- the one-sided alternative Hamilton
          (2018) proposes.

    References:
        Hodrick, R. J., & Prescott, E. C. (1997). Postwar U.S. business
        cycles: An empirical investigation. *Journal of Money, Credit and
        Banking*, 29(1), 1-16.

        Ravn, M. O., & Uhlig, H. (2002). On adjusting the
        Hodrick-Prescott filter for the frequency of observations.
        *Review of Economics and Statistics*, 84(2), 371-376.

        Hamilton, J. D. (2018). Why you should never use the
        Hodrick-Prescott filter. *Review of Economics and Statistics*,
        100(5), 831-843.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(rng.standard_normal(200))
        >>> res = HodrickPrescottFilter().filter(y)
        >>> res.trend.shape, round(res.cutoff, 1)
        ((200, 1), 39.7)

        The gain at the cutoff period is one half -- a cosine of that
        period comes out at half amplitude away from the edges -- and a
        linear trend passes through as trend, leaving no cycle:

        >>> t = np.arange(4000.0)
        >>> wave = np.cos(2 * np.pi * t / res.cutoff)
        >>> smoothed = HodrickPrescottFilter().filter(wave).trend[1000:3000, 0]
        >>> round(float(smoothed.std() / wave[1000:3000].std()), 2)
        0.5
        >>> line = HodrickPrescottFilter().filter(2.0 + 0.5 * t[:100])
        >>> bool(np.abs(line.cycle).max() < 1e-8)
        True
    """

    __slots__ = ("_penalty",)

    def __init__(self, *, penalty: float = 1600.0) -> None:
        """Validate the penalty.

        Args:
            penalty: The smoothing weight ``lambda``.

        Raises:
            SpecificationError: If the penalty is zero, negative or NaN.

        Example:
            >>> HodrickPrescottFilter(penalty=100.0)._penalty
            100.0
            >>> HodrickPrescottFilter(penalty=0)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: penalty must be positive; got 0.
        """
        if not penalty > 0.0:
            raise SpecificationError(f"penalty must be positive; got {penalty}.")
        self._penalty = float(penalty)

    def filter(self, data: npt.ArrayLike) -> DecompositionFilterResult:
        """Filter a panel.

        Each column is smoothed independently by the same banded solve;
        a one-dimensional series is treated as a one-column panel.

        Args:
            data: A ``(nobs, k)`` panel or one-dimensional series.

        Returns:
            The :class:`DecompositionFilterResult`, every row kept, with
            the first and last ``ceil(cutoff)`` rows flagged provisional.

        Raises:
            DimensionError: If the panel is malformed.
            SpecificationError: If it is shorter than four observations.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> panel = np.cumsum(rng.standard_normal((120, 2)), axis=0)
            >>> res = HodrickPrescottFilter(penalty=100.0).filter(panel)
            >>> res.n_series, int(res.provisional.sum()), round(res.cutoff, 1)
            (2, 40, 19.8)
            >>> HodrickPrescottFilter().filter(panel[:3, 0])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: ... needs at least 4 observations.
        """
        panel = validate_endog_matrix(data)
        if panel.shape[0] < 4:
            raise SpecificationError("the Hodrick-Prescott filter needs at least 4 observations.")
        return DecompositionFilterResult._penalized_result(
            panel, self._penalty, 2, "hodrick-prescott"
        )


class ButterworthFilter:
    r"""Gomez's (2001) Butterworth sine low-pass filter, stated by cutoff period and order.

    The trend is the minimizer of

    .. math::

        \sum_t (y_t - \tau_t)^2 + \lambda \sum_t (\Delta^n \tau_t)^2,
        \qquad
        \lambda = \bigl(2 \sin(\pi / P)\bigr)^{-2n},

    whose two-sided gain is the Butterworth form

    .. math::

        G(\omega) = \frac{1}{1 + \bigl(\sin(\omega/2) / \sin(\omega_c/2)\bigr)^{2n}},
        \qquad \omega_c = 2\pi / P,

    equal to one half at the cutoff period :math:`P` for every order
    and falling from one to zero more sharply as :math:`n` grows. Unlike
    :class:`HodrickPrescottFilter` the user states the frequency, not the
    weight: ``period=32, order=2`` is HP at :math:`\lambda = 677`, and
    ``period=39.7, order=2`` is HP at 1600. The penalty grows as
    :math:`P^{2n}`, so the same cutoff at order 3 is :math:`\lambda
    \approx 17620` and at order 4 :math:`\lambda \approx 4.6 \times 10^5`.

    Attributes:
        _period: The cutoff period :math:`P` in observations per cycle.
        _order: The differences penalized, :math:`n`.

    Args:
        period: Cutoff period in observations per cycle, above 2 (the
            Nyquist period); 32 quarterly for the Burns-Mitchell
            convention.
        order: Differences penalized, at least 1; 2 reproduces HP at
            the matching cutoff, higher is sharper.

    Raises:
        SpecificationError: If the period is 2 or below, or the order
            is not an integer of at least 1.

    Note:
        The order buys sharpness at two costs. The trend's null space is
        polynomials of degree :math:`n - 1`, so order 3 lets a quadratic
        through as trend where HP would bend it into the cycle. And the
        smoother is solved as :math:`(I + \lambda D'D)\tau = y` in banded
        form, whose conditioning degrades with :math:`\lambda`: the gain
        matches the closed form to three decimals through order 7 at
        period 32 and is visibly wrong at order 8 (0.86 where it should
        be 1.00), so treat orders above 6 as outside the filter's
        reliable range. The end-point problem worsens with order as
        well, and the result flags the same ``ceil(period)`` rows at
        each edge as provisional regardless.

    See Also:
        * :class:`DecompositionFilterResult` -- the container returned.
        * :class:`HodrickPrescottFilter` -- the order-2 member stated by
          weight.
        * :class:`~cultivars.spectral.band_pass.BaxterKingFilter` -- a
          band-pass rather than a low-pass, for isolating the cycle
          between two periods.

    References:
        Gomez, V. (2001). The use of Butterworth filters for trend and
        cycle estimation in economic time series. *Journal of Business &
        Economic Statistics*, 19(3), 365-373.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(rng.standard_normal(200))
        >>> res = ButterworthFilter(period=32, order=3).filter(y)
        >>> res.trend.shape, round(res.cutoff, 1), round(res.penalty)
        ((200, 1), 32.0, 17620)

        Order 2 at the HP cutoff period is the HP filter, and the gain
        at a period half the cutoff falls from 0.20 at order 1 to 0.004
        at order 4:

        >>> hp = HodrickPrescottFilter().filter(y)
        >>> same = ButterworthFilter(period=hp.cutoff, order=2).filter(y)
        >>> bool(np.allclose(same.trend, hp.trend))
        True
        >>> t = np.arange(4000.0)
        >>> wave = np.cos(2 * np.pi * t / 16)
        >>> gain = lambda n: ButterworthFilter(period=32, order=n).filter(wave).trend[1000:3000, 0]
        >>> [round(float(gain(n).std() / wave[1000:3000].std()), 3) for n in (1, 2, 4)]
        [0.202, 0.06, 0.004]
    """

    __slots__ = ("_order", "_period")

    def __init__(self, *, period: float = 32.0, order: int = 2) -> None:
        """Validate the cutoff and the order.

        Args:
            period: Cutoff period in observations per cycle.
            order: Differences penalized.

        Raises:
            SpecificationError: If ``period`` is 2 or below (or NaN), or
                ``order`` is not an integer of at least 1.

        Example:
            >>> f = ButterworthFilter(period=40, order=3)
            >>> f._period, f._order
            (40.0, 3)
            >>> ButterworthFilter(period=2)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: period must exceed 2 observations ... got 2.
            >>> ButterworthFilter(order=0)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: order must be >= 1; got 0.
        """
        if not period > 2.0:
            raise SpecificationError(f"period must exceed 2 observations per cycle; got {period}.")
        self._period = float(period)
        self._order = validate_order(order, "order", minimum=1)

    def filter(self, data: npt.ArrayLike) -> DecompositionFilterResult:
        """Filter a panel.

        Converts the cutoff period to the penalty, then runs the shared
        penalized smoother column by column; a one-dimensional series is
        treated as a one-column panel.

        Args:
            data: A ``(nobs, k)`` panel or one-dimensional series.

        Returns:
            The :class:`DecompositionFilterResult`, every row kept, with
            the first and last ``ceil(period)`` rows flagged provisional.

        Raises:
            DimensionError: If the panel is malformed.
            SpecificationError: If it is shorter than ``2 order + 2``.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> panel = np.cumsum(rng.standard_normal((150, 2)), axis=0)
            >>> res = ButterworthFilter(period=32).filter(panel)
            >>> res.n_series, res.method, int(res.provisional.sum())
            (2, 'butterworth', 64)
            >>> ButterworthFilter(order=3).filter(panel[:7, 0])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: a Butterworth filter of order 3 needs at ...
        """
        panel = validate_endog_matrix(data)
        if panel.shape[0] < 2 * self._order + 2:
            raise SpecificationError(
                f"a Butterworth filter of order {self._order} needs at least "
                f"{2 * self._order + 2} observations."
            )
        penalty = _butterworth_penalty(self._period, self._order)
        return DecompositionFilterResult._penalized_result(
            panel, penalty, self._order, "butterworth"
        )


class HamiltonFilter:
    r"""Hamilton's (2018) regression filter.

    Regress the level :math:`h` steps ahead on a constant and its own
    :math:`p` most recent values,

    .. math::

        y_{t+h} = \beta_0 + \beta_1 y_t + \beta_2 y_{t-1} + \cdots
        + \beta_p y_{t-p+1} + v_{t+h},

    by ordinary least squares over the whole sample; the fitted value is
    the trend at :math:`t + h` -- what the level's own recent history
    could predict that far ahead -- and the residual :math:`\hat v_{t+h}`
    is the cycle. Both begin at observation :math:`h + p - 1`, so the
    result is that many rows shorter than the input and reports the
    shortfall as ``offset``. The defaults are the paper's quarterly
    :math:`h = 8, p = 4`; monthly data want :math:`h = 24, p = 12`, and
    in either case :math:`h` should span the two years over which a
    business-cycle downturn resolves.

    The filter exists as the alternative to :class:`HodrickPrescottFilter`
    that Hamilton's critique calls for. The regression is valid for I(1)
    and I(2) series alike -- for a random walk the population
    coefficients are :math:`\beta_1 = 1` and the rest zero, so the cycle
    is the :math:`h`-step forecast error :math:`y_{t+h} - y_t` -- and the
    cycle is stationary by construction for any process whose
    :math:`d`-th difference is stationary with :math:`d \le p`. No
    future observations enter the regressors, so the filter does not
    manufacture dynamics from the two-sided smoothing that HP does.

    Attributes:
        _horizon: The forecast horizon :math:`h`.
        _lags: The number of own-lag regressors :math:`p`.

    Args:
        horizon: Forecast horizon :math:`h`, at least 1.
        lags: Regressors :math:`p`, at least 1.

    Raises:
        SpecificationError: If either is not an integer of at least 1.

    Note:
        "One-sided" describes the regressors, not the estimate. The
        coefficients are estimated on the full sample, so extending the
        data re-estimates :math:`\beta` and changes every fitted value a
        little; the result's ``provisional`` flags are all ``False``
        because no row depends on observations *after* :math:`t + h`
        through the design, which is the revision HP suffers from. For a
        strictly real-time cycle fix the coefficients at their random
        walk values and take :math:`y_{t+h} - y_t` directly.

        For a random walk with innovation variance :math:`\sigma^2` the
        cycle's standard deviation is about :math:`\sigma\sqrt{h}`, which
        is what an :math:`h`-step forecast error is; this is larger than
        the HP cycle's, and correctly so.

    See Also:
        * :class:`DecompositionFilterResult` -- the container returned.
        * :class:`HodrickPrescottFilter` -- the filter this one is the
          proposed replacement for.
        * :class:`BeveridgeNelsonDecomposition` -- the other definition
          in which the cycle is a forecast object.

    References:
        Hamilton, J. D. (2018). Why you should never use the
        Hodrick-Prescott filter. *Review of Economics and Statistics*,
        100(5), 831-843.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(rng.standard_normal(200))
        >>> res = HamiltonFilter().filter(y)
        >>> res.trend.shape, res.offset
        ((189, 1), 11)

        The result is the OLS regression it claims to be, and on a random
        walk the cycle has the scale of an eight-step forecast error:

        >>> design = np.column_stack([np.ones(189), y[3:192], y[2:191], y[1:190], y[0:189]])
        >>> beta = np.linalg.lstsq(design, y[11:], rcond=None)[0]
        >>> bool(np.allclose(design @ beta, res.trend[:, 0]))
        True
        >>> bool(abs(res.cycle.std() / np.sqrt(8) - 1.0) < 0.1)
        True
    """

    __slots__ = ("_horizon", "_lags")

    def __init__(self, *, horizon: int = 8, lags: int = 4) -> None:
        """Validate the horizon and lag count.

        Args:
            horizon: Forecast horizon ``h``.
            lags: Regressors ``p``.

        Raises:
            SpecificationError: If either is not an integer of at least 1.

        Example:
            >>> f = HamiltonFilter(horizon=24, lags=12)
            >>> f._horizon, f._lags
            (24, 12)
            >>> HamiltonFilter(horizon=0)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: horizon must be >= 1; got 0.
        """
        self._horizon = validate_order(horizon, "horizon", minimum=1)
        self._lags = validate_order(lags, "lags", minimum=1)

    def filter(self, data: npt.ArrayLike) -> DecompositionFilterResult:
        """Filter a panel, one regression per column.

        Each column is regressed on its own lags only; a one-dimensional
        series is treated as a one-column panel. The regression needs
        more observations than ``horizon + 2 lags``, so that at least one
        degree of freedom remains.

        Args:
            data: A ``(nobs, k)`` panel or one-dimensional series.

        Returns:
            The :class:`DecompositionFilterResult`, ``horizon + lags - 1`` rows shorter
            than the input, with no row provisional.

        Raises:
            DimensionError: If the panel is malformed.
            NumericalError: If it is too short for the regression.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> panel = np.cumsum(rng.standard_normal((300, 2)), axis=0)
            >>> res = HamiltonFilter(horizon=24, lags=12).filter(panel)
            >>> res.n_series, res.offset, res.trend.shape, bool(res.provisional.any())
            (2, 35, (265, 2), False)
            >>> HamiltonFilter().filter(panel[:16, 0])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.NumericalError: the Hamilton regression needs more than 16 ...
        """
        panel = validate_endog_matrix(data)
        offset = self._horizon + self._lags - 1
        rows = panel.shape[0] - offset
        trend = np.empty((max(rows, 0), panel.shape[1]))
        cycle = np.empty_like(trend)
        for column in range(panel.shape[1]):
            trend[:, column], cycle[:, column] = _hamilton_regression(
                panel[:, column], self._horizon, self._lags
            )
        return DecompositionFilterResult(
            trend=trend,
            cycle=cycle,
            method="hamilton",
            offset=offset,
            horizon=self._horizon,
            lags=self._lags,
            provisional=np.zeros(rows, dtype=np.bool_),
        )


class BeveridgeNelsonDecomposition:
    r"""The Beveridge-Nelson decomposition of a fitted ARIMA(p, 1, q).

    With :math:`\Delta y_t = \mu + \psi(L) e_t` the permanent component
    is the level the long-horizon forecast settles on once the drift is
    netted out,

    .. math::

        \tau_t = \lim_{H \to \infty}\bigl(E_t\, y_{t+H} - H\mu\bigr)
        = y_t + \sum_{k \ge 1}\bigl(E_t \Delta y_{t+k} - \mu\bigr),

    and the cycle :math:`c_t = y_t - \tau_t` is minus the sum of
    forecastable future changes. The trend is a random walk with drift
    :math:`\mu` and innovation :math:`\psi(1) e_t`, so the decomposition
    is the unique one in which the permanent component carries all the
    long-run variance. The sum is evaluated exactly in companion form
    (Morley 2002): with the state :math:`x_t` of the demeaned
    differences and their innovations and companion matrix :math:`F`,

    .. math::

        \sum_{k \ge 1} E_t \Delta y_{t+k} - \mu
        = e_1' F (I - F)^{-1} x_t,

    which is a linear function of the current state and needs no
    truncation. In closed form for the simplest cases: an ARIMA(1, 1, 0)
    with coefficient :math:`\phi` has cycle
    :math:`-\phi (\Delta y_t - \mu) / (1 - \phi)`, an ARIMA(0, 1, 1) with
    coefficient :math:`\theta` has cycle :math:`-\theta e_t`, and a pure
    random walk has none.

    Not an estimator: takes the fitted model and reads the decomposition
    off its parameters and residuals, so every number is conditional on
    that ARIMA being right.

    Attributes:
        _result: The fitted ARIMA(p, 1, q) result.

    Args:
        result: A fitted :class:`~cultivars.univariate.box_jenkins.ARIMA`
            with ``d = 1``, no seasonal part, no exogenous regressors and
            trend ``"n"`` or ``"c"``.

    Raises:
        SpecificationError: If the result is not fitted, or not an
            ARIMA(p, 1, q) of that form.

    Note:
        The BN cycle is typically small, volatile, and -- for an
        ARIMA(1, 1, 0) with :math:`\phi > 0` -- perfectly negatively
        correlated with growth, because it is a fixed multiple of the
        demeaned current change. Readers used to the smooth HP hump find
        this counterintuitive; it is the correct answer to "how far is
        the level from where it is heading", which is a different
        question from "what is the smooth part of the level". The
        decomposition is one-sided and exact, so no row is provisional,
        but it inherits the ARIMA's estimation error in full and the
        cycle changes with the model order chosen.

    See Also:
        * :class:`DecompositionFilterResult` -- the container returned.
        * :class:`~cultivars.univariate.box_jenkins.ARIMA` -- the model
          whose fitted result is the input.
        * :class:`HamiltonFilter` -- the other forecast-based definition
          of the cycle, model-free.
        * :class:`~cultivars.univariate.unobserved_components.UnobservedComponents`
          -- the structural alternative, in which trend and cycle are
          separate stochastic components with their own variances.

    References:
        Beveridge, S., & Nelson, C. R. (1981). A new approach to
        decomposition of economic time series into permanent and
        transitory components. *Journal of Monetary Economics*, 7(2),
        151-174.

        Morley, J. C. (2002). A state-space approach to calculating the
        Beveridge-Nelson decomposition. *Economics Letters*, 75(1),
        123-127.

    Example:
        >>> from cultivars.univariate.box_jenkins import ARIMA
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(0.2 + rng.standard_normal(300))
        >>> res = BeveridgeNelsonDecomposition(ARIMA(y, order=(1, 1, 0)).fit()).compute()
        >>> res.trend.shape, res.offset
        ((299, 1), 1)

        With genuine momentum in the differences the cycle is the closed
        form :math:`-\phi(\Delta y_t - \mu)/(1 - \phi)`, and much smaller
        than the HP cycle of the same series:

        >>> e = rng.standard_normal(400)
        >>> z = np.zeros(400)
        >>> for t in range(1, 400):
        ...     z[t] = 0.5 * z[t - 1] + e[t]
        >>> y = np.cumsum(0.2 + z)
        >>> fit = ARIMA(y, order=(1, 1, 0)).fit()
        >>> bn = BeveridgeNelsonDecomposition(fit).compute()
        >>> phi, mu = float(fit.ar_params[0]), float(fit.beta[0])
        >>> bool(np.allclose(bn.cycle[:, 0], -phi * (np.diff(y) - mu) / (1 - phi)))
        True
        >>> hp = HodrickPrescottFilter().filter(y)
        >>> bool(bn.cycle.std() < 0.6 * hp.cycle.std())
        True
    """

    __slots__ = ("_result",)

    def __init__(self, result: ARMAResult) -> None:
        """Check the fitted model is one the decomposition is defined for.

        Args:
            result: A fitted ARIMA result.

        Raises:
            SpecificationError: If ``result`` is not an
                :class:`~cultivars.univariate.box_jenkins.ARMAResult`,
                its ``d`` is not 1, it has a seasonal block or exogenous
                regressors, or its trend is other than ``"n"`` or ``"c"``.

        Example:
            >>> from cultivars.univariate.box_jenkins import ARIMA
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 + rng.standard_normal(200))
            >>> BeveridgeNelsonDecomposition(ARIMA(y, order=(1, 1, 0)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: BeveridgeNelson needs a fitted ARIMA ...
            >>> BeveridgeNelsonDecomposition(ARIMA(y, order=(1, 0, 0)).fit())  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: ... is defined for d = 1; got d = 0.
            >>> BeveridgeNelsonDecomposition(
            ...     ARIMA(y, order=(1, 1, 0), trend="ct").fit()
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: ... constant drift only; got trend 'ct'.
        """
        if not isinstance(result, ARMAResult):
            raise SpecificationError(
                f"BeveridgeNelson needs a fitted ARIMA result; got {type(result).__name__}."
            )
        _p, d, _q = result.order
        if d != 1:
            raise SpecificationError(
                f"the Beveridge-Nelson decomposition is defined for d = 1; got d = {d}."
            )
        if any(result.seasonal_order[:3]) or result.k_exog:
            raise SpecificationError(
                "the Beveridge-Nelson decomposition takes a non-seasonal ARIMA without "
                "exogenous regressors."
            )
        if result.trend not in ("n", "c"):
            raise SpecificationError(
                "the differenced series may carry a constant drift only; got trend "
                f"{result.trend!r}."
            )
        self._result = result

    def compute(self) -> DecompositionFilterResult:
        r"""Decompose the levels.

        Demeans the first differences by the fitted drift (zero under
        trend ``"n"``), evaluates the forecast sum
        :math:`e_1' F (I - F)^{-1} x_t` from the ARMA parameters and
        residuals, and adds it to the level to form the trend.

        Returns:
            The :class:`DecompositionFilterResult`, one row shorter than the series
            (the first difference has no value at ``t = 0``), with
            ``order`` set to the ARMA ``(p, q)`` and no row provisional.

        Raises:
            NumericalError: If the differenced series has a unit root,
                which makes :math:`I - F` singular and the trend
                undefined.

        Example:
            >>> from cultivars.univariate.box_jenkins import ARIMA
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 + rng.standard_normal(300))
            >>> res = BeveridgeNelsonDecomposition(ARIMA(y, order=(0, 1, 1)).fit()).compute()
            >>> res.method, res.order, bool(res.provisional.any())
            ('beveridge-nelson', (0, 1), False)
            >>> bool(np.allclose(res.trend + res.cycle, y[1:, None]))
            True

            A pure random walk has no forecastable momentum, hence no
            cycle:

            >>> walk = BeveridgeNelsonDecomposition(ARIMA(y, order=(0, 1, 0)).fit()).compute()
            >>> float(np.abs(walk.cycle).max())
            0.0
        """
        result = self._result
        y = result.endog
        differences = y[1:] - y[:-1]
        drift = float(result.beta[0]) if result.trend == "c" else 0.0
        z = differences - drift
        p, _d, q = result.order
        momentum = _beveridge_nelson(z, result.resid, result.ar_params, result.ma_params)
        levels = y[1:]
        trend = (levels + momentum)[:, None]
        return DecompositionFilterResult(
            trend=trend,
            cycle=levels[:, None] - trend,
            method="beveridge-nelson",
            offset=1,
            order=(p, q),
            provisional=np.zeros(levels.shape[0], dtype=np.bool_),
        )
