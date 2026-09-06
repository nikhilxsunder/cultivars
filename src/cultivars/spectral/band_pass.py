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

"""Band-pass filters: the business-cycle component, with the costs stated.

The ideal band-pass filter -- keep cycles with periods inside a stated
band, kill everything else -- has infinitely many weights, so every usable
filter is an approximation with a bill attached, and the two standards pay
it differently. Baxter-King truncates the ideal weights symmetrically:
phase-neutral everywhere it is defined, at the price of losing
``truncation`` observations at *each* end, which is why a BK cycle ends
years before the data does. Christiano-Fitzgerald keeps every observation
by letting the weights turn asymmetric near the endpoints (under their
random-walk approximation of the series), at the price of phase shift and
revision-prone estimates exactly where analysts look first -- the recent
end. Both bills are carried on the result rather than in fine print.

Bands are stated in *periods* -- observations per cycle -- with the
Burns-Mitchell business-cycle convention at quarterly frequency being 6 to
32. These are the module's only members that touch data rather than fitted
models; they are model-free linear filters, and detrending choices upstream
(the filters assume no deterministic trend survives in the input beyond
what the CF random-walk premise absorbs) remain the caller's.

References:
    Baxter, M., & King, R. G. (1999). Measuring business cycles:
        Approximate band-pass filters for economic time series. *Review of
        Economics and Statistics*, 81(4), 575-593.
    Christiano, L. J., & Fitzgerald, T. J. (2003). The band pass filter.
        *International Economic Review*, 44(2), 435-465.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core import SummaryTable, _ideal_weights, _validate_band, validate_endog_matrix
from .._internals import _SummaryMixin
from ..exceptions import SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class BandPassResult(_SummaryMixin):
    """A filtered panel: the cycle, the remainder, and the bookkeeping.

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
    """

    cycle: npt.NDArray[np.float64] = field(repr=False)
    trend: npt.NDArray[np.float64] = field(repr=False)
    low: float
    high: float
    method: str
    offset: int
    symmetric: npt.NDArray[np.bool_] = field(repr=False)

    @property
    def n_series(self) -> int:
        """Number of series filtered."""
        return int(self.cycle.shape[1])

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
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


class BaxterKing:
    """The Baxter-King symmetric truncated band-pass filter.

    Args:
        low: Shortest period passed, in observations per cycle. The
            quarterly business-cycle convention is 6.
        high: Longest period passed; 32 quarterly.
        truncation: Half-length of the moving average; 12 is the paper's
            quarterly recommendation. Larger approximates the ideal filter
            better and loses more sample.

    Raises:
        SpecificationError: If the band or truncation is malformed.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> t = np.arange(300.0)
        >>> y = np.sin(2 * np.pi * t / 12) + np.sin(2 * np.pi * t / 80)
        >>> res = BaxterKing(low=6, high=32, truncation=12).filter(y)
        >>> res.cycle.shape
        (276, 1)
        >>> bool(np.std(res.cycle) < np.std(y))
        True
    """

    __slots__ = ("_high", "_low", "_truncation")

    def __init__(self, *, low: float = 6.0, high: float = 32.0, truncation: int = 12) -> None:
        """Validate the band and the truncation."""
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
        cycle.

        Args:
            data: A ``(nobs, k)`` panel or one-dimensional series.

        Returns:
            The :class:`BandPassResult`, ``2 * truncation`` rows shorter
            than the input.

        Raises:
            DimensionError: If the panel is malformed or too short for the
                truncation.
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


class ChristianoFitzgerald:
    """The Christiano-Fitzgerald asymmetric full-sample band-pass filter.

    The random-walk variant -- their recommended default -- which keeps
    every observation by letting the filter turn asymmetric toward the
    endpoints, where the missing ideal weights are absorbed into the
    endpoint observations under a random-walk premise for the series.

    Args:
        low: Shortest period passed, in observations per cycle.
        high: Longest period passed.

    Raises:
        SpecificationError: If the band is malformed.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> t = np.arange(300.0)
        >>> y = np.sin(2 * np.pi * t / 12) + np.sin(2 * np.pi * t / 80)
        >>> res = ChristianoFitzgerald(low=6, high=32).filter(y)
        >>> res.cycle.shape
        (300, 1)
        >>> bool(res.symmetric[150] and not res.symmetric[0])
        True
    """

    __slots__ = ("_high", "_low")

    def __init__(self, *, low: float = 6.0, high: float = 32.0) -> None:
        """Validate the band."""
        _validate_band(low, high)
        self._low = float(low)
        self._high = float(high)

    def filter(self, data: npt.ArrayLike) -> BandPassResult:
        """Filter a panel, keeping every observation.

        Args:
            data: A ``(nobs, k)`` panel or one-dimensional series.

        Returns:
            The :class:`BandPassResult`, same length as the input.
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
