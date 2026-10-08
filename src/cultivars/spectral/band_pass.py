# filepath: /src/cultivars/spectral/band_pass.py
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
r"""Band-pass filters: the business-cycle component, with the costs stated.

The ideal band-pass filter -- keep cycles with periods inside a stated
band, kill everything else -- has the gain function :math:`\mathbb{1}\{
\omega_l \le |\omega| \le \omega_h\}` and therefore the weights

.. math::

   b_0 = \frac{\omega_h - \omega_l}{\pi},
   \qquad
   b_j = \frac{\sin j\omega_h - \sin j\omega_l}{\pi j},
   \qquad j = \pm 1, \pm 2, \dots,

infinitely many of them, so every usable filter is an approximation
with a bill attached, and the two standards pay it differently.
Baxter-King truncates the ideal weights symmetrically at lag :math:`K`
and recentres them to sum to zero: phase-neutral everywhere it is
defined, with exact removal of the zero frequency, at the price of
losing :math:`K` observations at *each* end, which is why a BK cycle
ends years before the data does. Christiano-Fitzgerald keeps every
observation by letting the weights turn asymmetric near the endpoints,
folding the weights that fall outside the sample into the endpoint
observations under their random-walk approximation of the series, at
the price of phase shift and revision-prone estimates exactly where
analysts look first -- the recent end.

Two commitments shape the surface. First, both bills are carried on
the result rather than in fine print: :class:`BandPassResult` records
the ``offset`` a method lost and a ``symmetric`` flag per row saying
where the estimate is phase-neutral and final, so a plot of the recent
end of a CF cycle can be marked as provisional by reading the record
rather than by remembering the paper. Second, bands are stated in
*periods* -- observations per cycle -- with the Burns-Mitchell
business-cycle convention at quarterly frequency being 6 to 32, because
periods are what an economist means and frequencies are what the
weights need; the conversion :math:`\omega = 2\pi / \text{period}` is
made once, inside, and the shortest admissible period is 2, the Nyquist
limit of the sampling.

These are the module's only members that touch data rather than fitted
models; they are model-free linear filters, and detrending choices
upstream (the filters assume no deterministic trend survives in the
input beyond what the CF random-walk premise absorbs) remain the
caller's.

Layout. :class:`BaxterKingFilter` and :class:`ChristianoFitzgeraldFilter`
are the producers, each validated at construction and applied through
``filter``, and :class:`BandPassResult` the record they share. The
numerics live in ``_core``: ``_ideal_weights`` gives :math:`b_0, \dots,
b_K` at any truncation, which BK mirrors and recentres and CF extends to
the sample length; ``_validate_band`` enforces ``2 <= low < high``; and
``validate_endog_matrix`` coerces a series or panel to ``(nobs, k)``.
The Hodrick-Prescott high-pass filter lives in
:mod:`~cultivars.spectral.filters`, and the spectral estimators that
show where a series' power sits before a band is chosen in
:mod:`~cultivars.spectral.periodogram`.

References:
    Baxter, M., & King, R. G. (1999). Measuring business cycles:
    Approximate band-pass filters for economic time series. *Review of
    Economics and Statistics*, 81(4), 575-593.

    Christiano, L. J., & Fitzgerald, T. J. (2003). The band pass filter.
    *International Economic Review*, 44(2), 435-465.

    Burns, A. F., & Mitchell, W. C. (1946). *Measuring Business Cycles*.
    NBER.

Example:
    A 12-period cycle riding on an 80-period swing, extracted both
    ways; BK loses twelve observations at each end and is final
    everywhere, CF keeps them all and is final only in the middle, and
    where both are final they agree:

    >>> import numpy as np
    >>> t = np.arange(300.0)
    >>> y = np.sin(2 * np.pi * t / 12) + np.sin(2 * np.pi * t / 80)
    >>> bk = BaxterKingFilter(low=6, high=32, truncation=12).filter(y)
    >>> cf = ChristianoFitzgeraldFilter(low=6, high=32).filter(y)
    >>> bk.cycle.shape, bk.offset, bool(bk.symmetric.all())
    ((276, 1), 12, True)
    >>> cf.cycle.shape, cf.offset, int(cf.symmetric.sum())
    ((300, 1), 0, 236)
    >>> agreement = np.corrcoef(bk.cycle[28:248, 0], cf.cycle[40:260, 0])[0, 1]
    >>> bool(agreement > 0.99)
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core import SummaryTable, _ideal_weights, _validate_band, validate_endog_matrix
from .._internals import _SummaryMixin
from ..exceptions import SpecificationError

__all__ = ["BandPassResult", "BaxterKingFilter", "ChristianoFitzgeraldFilter"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class BandPassResult(_SummaryMixin):
    r"""A filtered panel: the cycle, the remainder, and the bookkeeping.

    The cycle is the band-pass component,

    .. math::

       c_t = \sum_{j} \tilde b_j\, y_{t-j},

    the input convolved with an approximation to the ideal band-pass
    weights :math:`b_0 = (\omega_h - \omega_l)/\pi` and :math:`b_j =
    (\sin j\omega_h - \sin j\omega_l)/(\pi j)`, where :math:`\omega_l =
    2\pi/\texttt{high}` and :math:`\omega_h = 2\pi/\texttt{low}` are the
    band's edge frequencies; the trend is whatever the band did not
    pass, :math:`y_t - c_t`, so the two sum to the input on every row.
    Which approximation :math:`\tilde b` was used, what it cost in
    observations, and where its estimate is phase-neutral and final are
    carried on the record as ``method``, ``offset``, and ``symmetric``
    rather than left in fine print.

    Attributes:
        cycle: The ``(rows, k)`` band-pass component. For Baxter-King,
            ``rows = nobs - 2 * truncation`` and the first cycle value
            aligns with observation ``offset`` of the input; for
            Christiano-Fitzgerald, ``rows = nobs`` and ``offset`` is zero.
        trend: The input minus the cycle, over the same rows -- everything
            outside the band, low and high frequencies together.
        low: Shortest period in the band, in observations per cycle.
        high: Longest period in the band.
        method: ``"baxter-king"`` or ``"christiano-fitzgerald"``.
        offset: Input observations lost before the first cycle value.
        symmetric: ``(rows,)`` flags marking where the filter applied is
            symmetric, so the estimate is phase-neutral and final. All rows
            for Baxter-King by construction; for Christiano-Fitzgerald the
            flag is a stated convention -- at least ``high`` observations
            on each side -- because the asymmetric weights decay at the
            scale of the longest period in the band, and values outside the
            flag are both phase-shifted and subject to revision as data
            arrives.

    Note:
        ``trend`` is a misnomer the literature shares: it holds the
        frequencies *below* the band and those *above* it together, so
        for a series with a seasonal or a high-frequency component that
        part lands in ``trend`` too. A user who wants the low-frequency
        part alone should pass a band whose ``low`` is 2, the shortest
        representable period, so that nothing lies above it. The
        summary's "cycle share of variance" is
        :math:`\operatorname{var}(c) / (\operatorname{var}(c) +
        \operatorname{var}(y - c))`, which is a share only when the two
        are uncorrelated, as an ideal filter's components are and a
        truncated one's nearly are.

    See Also:
        * :class:`BaxterKingFilter` and :class:`ChristianoFitzgeraldFilter`
          -- the producers.
        * :class:`~cultivars.spectral.periodogram.SpectrumEstimate` -- reads
          the frequencies the band selects from the data itself.

    References:
        Baxter, M., & King, R. G. (1999). Measuring business cycles:
        Approximate band-pass filters for economic time series. *Review of
        Economics and Statistics*, 81(4), 575-593.

        Christiano, L. J., & Fitzgerald, T. J. (2003). The band pass
        filter. *International Economic Review*, 44(2), 435-465.

    Example:
        A 12-period cycle on top of an 80-period swing; the 6-32 band
        keeps the first and hands the second to ``trend``:

        >>> import numpy as np
        >>> t = np.arange(300.0)
        >>> y = np.sin(2 * np.pi * t / 12) + np.sin(2 * np.pi * t / 80)
        >>> res = BaxterKingFilter(low=6, high=32, truncation=12).filter(y)
        >>> res.cycle.shape, res.offset, res.method, bool(res.symmetric.all())
        ((276, 1), 12, 'baxter-king', True)
        >>> bool(np.allclose(res.cycle + res.trend, y[12:288, None]))
        True
        >>> round(float(np.corrcoef(res.cycle[:, 0], np.sin(2 * np.pi * t[12:288] / 12))[0, 1]), 2)
        0.99
        >>> full = ChristianoFitzgeraldFilter(low=6, high=32).filter(y)
        >>> full.cycle.shape, full.offset, int(full.symmetric.sum())
        ((300, 1), 0, 236)
    """

    cycle: npt.NDArray[np.float64] = field(repr=False)
    """``(rows, k)`` band-pass component; row ``i`` is input observation ``offset + i``.

    Kept out of the repr.
    """
    trend: npt.NDArray[np.float64] = field(repr=False)
    """``(rows, k)`` input minus cycle: every frequency outside the band. Kept out of the repr."""
    low: float
    """Shortest period passed, in observations per cycle; the band's upper edge frequency."""
    high: float
    """Longest period passed; the band's lower edge frequency is ``2 * pi / high``."""
    method: str
    """``"baxter-king"`` or ``"christiano-fitzgerald"``."""
    offset: int
    """Input rows lost before the first cycle value: ``truncation`` for BK, zero for CF."""
    symmetric: npt.NDArray[np.bool_] = field(repr=False)
    """``(rows,)`` flags where the applied filter is symmetric, hence phase-neutral and final.

    Kept out of the repr.
    """

    @property
    def n_series(self) -> int:
        """Number of series filtered, :math:`k`, the last axis of ``cycle`` and ``trend``."""
        return int(self.cycle.shape[1])

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary: the cycle's variance share per series.

        Under a header with the method, series and row counts, and the
        band; the notes restate the band as frequencies, say what the
        method cost and where its values are final, and remind that
        detrending is upstream.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``summary()``,
            ``str()``, and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> t = np.arange(300.0)
            >>> y = np.sin(2 * np.pi * t / 12) + np.sin(2 * np.pi * t / 80)
            >>> table = BaxterKingFilter().filter(y)._summary_table()
            >>> table.columns, table.rows
            (('series', 'cycle share of variance'), (('series 1', '55.3%'),))
            >>> table.metadata[3]
            ('Band', '[6, 32] periods')
        """
        share = self.cycle.var(axis=0) / np.maximum(
            self.cycle.var(axis=0) + self.trend.var(axis=0), 1e-300
        )
        rows = tuple(
            (f"series {index + 1}", f"{100.0 * float(value):.1f}%")
            for index, value in enumerate(share)
        )
        notes = [
            f"Band: periods {self.low:g} to {self.high:g} observations per "
            "cycle (frequencies "
            f"{2.0 * np.pi / self.high:.4f} to {2.0 * np.pi / self.low:.4f}).",
            (
                f"Baxter-King loses {self.offset} observations at each end "
                "by construction; every reported value is phase-neutral "
                "and final."
                if self.method == "baxter-king"
                else "Christiano-Fitzgerald keeps every observation; values "
                "where `symmetric` is False are phase-shifted and will be "
                "revised as data arrives -- which includes the recent end."
            ),
            "The filters are model-free; any deterministic trend beyond "
            "what the CF random-walk premise absorbs should be removed "
            "upstream.",
        ]
        return SummaryTable(
            title="Band-Pass Filter",
            metadata=(
                ("Method", self.method),
                ("Series", f"{self.n_series}"),
                ("Rows", f"{self.cycle.shape[0]}"),
                ("Band", f"[{self.low:g}, {self.high:g}] periods"),
            ),
            columns=("series", "cycle share of variance"),
            rows=rows,
            notes=tuple(notes),
        )


class BaxterKingFilter:
    r"""The Baxter-King symmetric truncated band-pass filter.

    The ideal band-pass weights :math:`b_j` truncated at lag :math:`K`
    and recentred,

    .. math::

       \tilde b_j = b_j - \frac{1}{2K + 1} \sum_{i=-K}^{K} b_i,
       \qquad |j| \le K,

    so that the weights sum to zero and the filter has zero gain at
    frequency zero: a constant, and a linear trend, are removed exactly
    rather than approximately. The filter is symmetric, hence
    phase-neutral everywhere it is defined, and it is defined only where
    :math:`K` observations exist on each side, which is why a
    Baxter-King cycle ends :math:`K` observations before the data does.

    Args:
        low: Shortest period passed, in observations per cycle. The
            quarterly business-cycle convention is 6.
        high: Longest period passed; 32 quarterly.
        truncation: Half-length of the moving average; 12 is the paper's
            quarterly recommendation. Larger approximates the ideal filter
            better and loses more sample.

    Attributes:
        _low: The validated shortest period.
        _high: The validated longest period.
        _truncation: The validated half-length :math:`K`.

    Raises:
        SpecificationError: If the band or truncation is malformed.

    Note:
        Larger ``truncation`` is not free in either direction: the
        approximation to the ideal gain improves as :math:`K` grows, but
        :math:`2K` observations are lost, and at the paper's
        recommendation of 12 a quarterly sample loses six years. The
        recentring means the weights are no longer the ideal filter's
        even inside the band; the gain at the band's edges leaks and
        ripples, which is the price of finite length and is the same
        for every truncated filter.

    See Also:
        * :class:`BandPassResult` -- the record :meth:`filter` returns.
        * :class:`ChristianoFitzgeraldFilter` -- keeps every observation
          at the cost of asymmetry near the ends.
        * :class:`~cultivars.spectral.filters.HodrickPrescottFilter` --
          the high-pass alternative.

    References:
        Baxter, M., & King, R. G. (1999). Measuring business cycles:
        Approximate band-pass filters for economic time series. *Review of
        Economics and Statistics*, 81(4), 575-593.

    Example:
        >>> import numpy as np
        >>> t = np.arange(300.0)
        >>> y = np.sin(2 * np.pi * t / 12) + np.sin(2 * np.pi * t / 80)
        >>> res = BaxterKingFilter(low=6, high=32, truncation=12).filter(y)
        >>> res.cycle.shape
        (276, 1)
        >>> bool(np.std(res.cycle) < np.std(y))
        True
        >>> float(np.abs(BaxterKingFilter().filter(3.0 + 0.1 * t).cycle).max()) < 1e-12
        True
    """

    __slots__ = ("_high", "_low", "_truncation")

    def __init__(self, *, low: float = 6.0, high: float = 32.0, truncation: int = 12) -> None:
        """Validate the band and the truncation.

        Args:
            low: Checked, with ``high``, to satisfy ``2 <= low < high``.
            high: Longest period passed.
            truncation: Checked to be at least one.

        Raises:
            SpecificationError: If the band is not ``2 <= low < high`` or
                the truncation is below one.

        Example:
            >>> BaxterKingFilter(low=32, high=6)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: the band needs 2 <= low < high in periods ...
        """
        _validate_band(low, high)
        if truncation < 1:
            raise SpecificationError(f"truncation must be at least 1; got {truncation}.")
        self._low = float(low)
        self._high = float(high)
        self._truncation = int(truncation)

    def filter(self, data: npt.ArrayLike) -> BandPassResult:
        """Filter a panel.

        The truncated ideal weights are recentred to sum to zero -- the
        Baxter-King adjustment that restores exact removal of the zero
        frequency, so a constant or linear trend cannot leak into the
        cycle. Each column is convolved with the ``2K + 1`` weights in
        ``"valid"`` mode, so the result has ``nobs - 2K`` rows aligned
        with input observations ``K .. nobs - K - 1``.

        Args:
            data: A ``(nobs, k)`` panel or one-dimensional series.

        Returns:
            The :class:`BandPassResult`, ``2 * truncation`` rows shorter
            than the input.

        Raises:
            DimensionError: If the panel is not one- or two-dimensional.
            SpecificationError: If the sample is too short for the
                truncation.
            NumericalError: If the panel is not finite.

        Example:
            >>> import numpy as np
            >>> t = np.arange(300.0)
            >>> panel = np.column_stack([np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 20)])
            >>> res = BaxterKingFilter().filter(panel)
            >>> res.cycle.shape, res.n_series, res.offset
            ((276, 2), 2, 12)
            >>> BaxterKingFilter(truncation=12).filter(np.zeros(24))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: a sample of 24 rows is too short for ...
        """
        panel = validate_endog_matrix(data)
        nobs = panel.shape[0]
        cut = self._truncation
        if nobs <= 2 * cut:
            raise SpecificationError(
                f"a sample of {nobs} rows is too short for truncation "
                f"{cut}; Baxter-King needs more than {2 * cut} rows."
            )
        half = _ideal_weights(cut, self._low, self._high)
        weights = np.concatenate([half[:0:-1], half])
        weights -= weights.mean()
        rows = nobs - 2 * cut
        cycle = np.empty((rows, panel.shape[1]))
        for column in range(panel.shape[1]):
            cycle[:, column] = np.convolve(panel[:, column], weights[::-1], "valid")
        middle = panel[cut : nobs - cut]
        return BandPassResult(
            cycle=cycle,
            trend=middle - cycle,
            low=self._low,
            high=self._high,
            method="baxter-king",
            offset=cut,
            symmetric=np.ones(rows, dtype=np.bool_),
        )


class ChristianoFitzgeraldFilter:
    r"""The Christiano-Fitzgerald asymmetric full-sample band-pass filter.

    The random-walk variant -- their recommended default -- which keeps
    every observation by letting the filter turn asymmetric toward the
    endpoints, where the missing ideal weights are absorbed into the
    endpoint observations under a random-walk premise for the series.
    At observation :math:`t` of a sample of :math:`T`,

    .. math::

       c_t = b_0\, y_t + \sum_{j=1}^{T-2-t} b_j\, y_{t+j} + \tilde b_{T-1-t}\, y_{T-1}
       + \sum_{j=1}^{t-1} b_j\, y_{t-j} + \tilde b_{t}\, y_{0},

    with :math:`\tilde b_m = -\tfrac{1}{2} b_0 - \sum_{j=1}^{m-1} b_j`
    the endpoint weight that stands in for every ideal weight beyond
    the sample, the optimal choice when the series is a random walk. The
    filter is symmetric only far from both ends; elsewhere the estimate
    is phase-shifted and revised as observations arrive, which is the
    price of having a value at the recent end at all.

    Args:
        low: Shortest period passed, in observations per cycle.
        high: Longest period passed.

    Attributes:
        _low: The validated shortest period.
        _high: The validated longest period.

    Raises:
        SpecificationError: If the band is malformed.

    Note:
        The random-walk premise is what makes a linear trend acceptable
        in the input -- a random walk with drift is the assumed process
        -- but it is a premise about the series' spectrum near zero, not
        a detrending step, and a series far from a random walk (a
        stationary series, or one with a strong deterministic trend and
        little else) is better served by removing the trend first. The
        ``symmetric`` flag on the result marks at least ``high``
        observations on each side as the region where the asymmetric
        weights have decayed enough for the estimate to be treated as
        final; that boundary is a stated convention, not a theorem.

    See Also:
        * :class:`BandPassResult` -- the record :meth:`filter` returns.
        * :class:`BaxterKingFilter` -- phase-neutral everywhere at the
          cost of the ends.

    References:
        Christiano, L. J., & Fitzgerald, T. J. (2003). The band pass
        filter. *International Economic Review*, 44(2), 435-465.

    Example:
        >>> import numpy as np
        >>> t = np.arange(300.0)
        >>> y = np.sin(2 * np.pi * t / 12) + np.sin(2 * np.pi * t / 80)
        >>> res = ChristianoFitzgeraldFilter(low=6, high=32).filter(y)
        >>> res.cycle.shape
        (300, 1)
        >>> bool(res.symmetric[150] and not res.symmetric[0])
        True
        >>> round(float(np.corrcoef(res.cycle[:, 0], np.sin(2 * np.pi * t / 12))[0, 1]), 2)
        1.0
    """

    __slots__ = ("_high", "_low")

    def __init__(self, *, low: float = 6.0, high: float = 32.0) -> None:
        """Validate the band.

        Args:
            low: Checked, with ``high``, to satisfy ``2 <= low < high``.
            high: Longest period passed.

        Raises:
            SpecificationError: If the band is not ``2 <= low < high``.

        Example:
            >>> ChristianoFitzgeraldFilter(low=1, high=32)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: the band needs 2 <= low < high in periods ...
        """
        _validate_band(low, high)
        self._low = float(low)
        self._high = float(high)

    def filter(self, data: npt.ArrayLike) -> BandPassResult:
        """Filter a panel, keeping every observation.

        The ideal weights are computed out to lag ``nobs`` once, the
        endpoint weights derived from their cumulative sums, and each
        observation's cycle assembled from the weights that fit inside
        the sample on each side plus the endpoint weight for what does
        not. The ``symmetric`` flag is set where at least ``ceil(high)``
        observations lie on both sides.

        Args:
            data: A ``(nobs, k)`` panel or one-dimensional series.

        Returns:
            The :class:`BandPassResult`, same length as the input.

        Raises:
            DimensionError: If the panel is not one- or two-dimensional.
            NumericalError: If the panel is not finite.

        Example:
            >>> import numpy as np
            >>> t = np.arange(300.0)
            >>> y = np.sin(2 * np.pi * t / 12) + np.sin(2 * np.pi * t / 80)
            >>> res = ChristianoFitzgeraldFilter().filter(y)
            >>> res.cycle.shape, res.offset, bool(np.allclose(res.cycle + res.trend, y[:, None]))
            ((300, 1), 0, True)
            >>> int(res.symmetric.argmax()), int(res.symmetric.sum())
            (32, 236)
        """
        panel = validate_endog_matrix(data)
        nobs, k = panel.shape
        weights = _ideal_weights(nobs, self._low, self._high)
        endpoint = -0.5 * weights[0] - np.concatenate([np.zeros(1), np.cumsum(weights[1:])])
        cycle = np.empty((nobs, k))
        for t in range(nobs):
            ahead = nobs - 1 - t
            value = weights[0] * panel[t]
            if ahead >= 1:
                value = value + weights[1:ahead] @ panel[t + 1 : nobs - 1]
                value = value + endpoint[ahead] * panel[nobs - 1]
            if t >= 1:
                value = value + weights[1:t] @ panel[t - 1 : 0 : -1]
                value = value + endpoint[t] * panel[0]
            cycle[t] = value
        span = int(np.ceil(self._high))
        positions = np.arange(nobs)
        symmetric = (positions >= span) & (positions <= nobs - 1 - span)
        return BandPassResult(
            cycle=cycle,
            trend=panel - cycle,
            low=self._low,
            high=self._high,
            method="christiano-fitzgerald",
            offset=0,
            symmetric=symmetric,
        )
