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

"""Trend-cycle filters: four answers to "what is the trend", each with its bill.

A trend-cycle decomposition is a definition, not a discovery, and the
four here define the trend differently. Hodrick-Prescott and Butterworth
are low-pass filters: the trend is what survives a penalty on its
``n``-th differences, and the penalty fixes the cutoff period at which
the filter's gain is one half (39.7 quarters for HP's 1600). Hamilton's
regression filter defines the trend as what could be predicted two years
ahead from the level's own recent history, and the cycle as what could
not; it exists because HP manufactures dynamics -- spurious cycles from
a random walk, end-point values that change as data arrive, a penalty
with no rationale at frequencies other than quarterly -- and the HP
result prints that critique rather than assuming the user knows it.
Beveridge-Nelson is a model-based definition: the trend of an ARIMA is
the level the long-horizon forecast converges to, net of drift, so the
cycle is the forecastable momentum, which is why a BN cycle is small,
noisy, and often negatively correlated with growth when the others are
smooth -- not a bug, a different question.

The filters that touch data take a panel and filter each column; the
Beveridge-Nelson decomposition takes a fitted ARIMA(p, 1, q). Every
result reports which rows are lost or provisional, in the manner of
:mod:`band_pass`.

References:
    Hodrick, R. J., & Prescott, E. C. (1997). Postwar U.S. business
        cycles: An empirical investigation. *Journal of Money, Credit and
        Banking*, 29(1), 1-16.
    Hamilton, J. D. (2018). Why you should never use the Hodrick-Prescott
        filter. *Review of Economics and Statistics*, 100(5), 831-843.
    Gomez, V. (2001). The use of Butterworth filters for trend and cycle
        estimation in economic time series. *Journal of Business &
        Economic Statistics*, 19(3), 365-373.
    Beveridge, S., & Nelson, C. R. (1981). A new approach to
        decomposition of economic time series into permanent and
        transitory components. *Journal of Monetary Economics*, 7(2),
        151-174.
    Morley, J. C. (2002). A state-space approach to calculating the
        Beveridge-Nelson decomposition. *Economics Letters*, 75(1),
        123-127.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core import (
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
from .._internals import _SummaryMixin
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
    """A trend-cycle decomposition of a panel, with the bookkeeping the method needs.

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
    """

    trend: npt.NDArray[np.float64] = field(repr=False)
    cycle: npt.NDArray[np.float64] = field(repr=False)
    method: str
    offset: int
    penalty: float | None = None
    order: int | tuple[int, int] | None = None
    cutoff: float | None = None
    horizon: int | None = None
    lags: int | None = None
    provisional: npt.NDArray[np.bool_] = field(
        repr=False, default_factory=lambda: np.zeros(0, bool)
    )

    @property
    def n_series(self) -> int:
        """Number of series filtered."""
        return int(self.trend.shape[1])

    def _summary_table(self) -> SummaryTable:
        """Cycle scale per series, absolute and against the input's growth, with the caveats."""
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
        """What the method's definition of the trend implies for reading the cycle."""
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
        """Run the penalized smoother and package it."""
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
    """The Hodrick-Prescott filter, with Hamilton's critique on the result.

    The trend minimizes ``sum (y - tau)^2 + penalty * sum (Delta^2 tau)^2``.
    The conventional penalties are 1600 quarterly, 100 annual and 14400
    monthly (Ravn-Uhlig's frequency-power rule gives 6.25 and 129600
    instead); each fixes the cutoff period the result reports.

    Args:
        penalty: The smoothing weight ``lambda``.

    Raises:
        SpecificationError: If the penalty is not positive.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(rng.standard_normal(200))
        >>> res = HodrickPrescottFilter().filter(y)
        >>> res.trend.shape, round(res.cutoff, 1)
        ((200, 1), 39.7)
    """

    __slots__ = ("_penalty",)

    def __init__(self, *, penalty: float = 1600.0) -> None:
        """Validate the penalty."""
        if not penalty > 0.0:
            raise SpecificationError(f"penalty must be positive; got {penalty}.")
        self._penalty = float(penalty)

    def filter(self, data: npt.ArrayLike) -> DecompositionFilterResult:
        """Filter a panel.

        Args:
            data: A ``(nobs, k)`` panel or one-dimensional series.

        Returns:
            The :class:`DecompositionFilterResult`, every row kept.

        Raises:
            DimensionError: If the panel is malformed.
            SpecificationError: If it is shorter than four observations.
        """
        panel = validate_endog_matrix(data)
        if panel.shape[0] < 4:
            raise SpecificationError("the Hodrick-Prescott filter needs at least 4 observations.")
        return DecompositionFilterResult._penalized_result(
            panel, self._penalty, 2, "hodrick-prescott"
        )


class ButterworthFilter:
    """Gomez's (2001) Butterworth sine low-pass filter, stated by cutoff period and order.

    The trend minimizes ``sum (y - tau)^2 + penalty * sum (Delta^order
    tau)^2`` with the penalty set so that the gain is one half at
    ``period``: ``penalty = (2 sin(pi / period))^(-2 order)``. Unlike HP
    the user states the frequency, not the weight, and the order sets
    how sharply the gain falls from one to zero around it.

    Args:
        period: Cutoff period in observations per cycle; 32 quarterly
            for the Burns-Mitchell convention.
        order: Differences penalized; 2 reproduces HP at the matching
            cutoff, higher is sharper.

    Raises:
        SpecificationError: If the period is below 2 or the order below 1.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(rng.standard_normal(200))
        >>> res = ButterworthFilter(period=32, order=3).filter(y)
        >>> res.trend.shape, round(res.cutoff, 1)
        ((200, 1), 32.0)
    """

    __slots__ = ("_order", "_period")

    def __init__(self, *, period: float = 32.0, order: int = 2) -> None:
        """Validate the cutoff and the order."""
        if not period > 2.0:
            raise SpecificationError(f"period must exceed 2 observations per cycle; got {period}.")
        self._period = float(period)
        self._order = validate_order(order, "order", minimum=1)

    def filter(self, data: npt.ArrayLike) -> DecompositionFilterResult:
        """Filter a panel.

        Args:
            data: A ``(nobs, k)`` panel or one-dimensional series.

        Returns:
            The :class:`DecompositionFilterResult`, every row kept.

        Raises:
            DimensionError: If the panel is malformed.
            SpecificationError: If it is shorter than ``2 order + 2``.
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
    """Hamilton's (2018) regression filter.

    Regress ``y_{t+h}`` on a constant and ``y_t, ..., y_{t-p+1}``; the
    residual is the cycle and the fitted value the trend. The defaults
    are the paper's quarterly ``h = 8, p = 4``; monthly data want
    ``h = 24, p = 12``.

    Args:
        horizon: Forecast horizon ``h``.
        lags: Regressors ``p``.

    Raises:
        SpecificationError: If either is below 1.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(rng.standard_normal(200))
        >>> res = HamiltonFilter().filter(y)
        >>> res.trend.shape, res.offset
        ((189, 1), 11)
    """

    __slots__ = ("_horizon", "_lags")

    def __init__(self, *, horizon: int = 8, lags: int = 4) -> None:
        """Validate the horizon and lag count."""
        self._horizon = validate_order(horizon, "horizon", minimum=1)
        self._lags = validate_order(lags, "lags", minimum=1)

    def filter(self, data: npt.ArrayLike) -> DecompositionFilterResult:
        """Filter a panel, one regression per column.

        Args:
            data: A ``(nobs, k)`` panel or one-dimensional series.

        Returns:
            The :class:`DecompositionFilterResult`, ``horizon + lags - 1`` rows shorter
            than the input.

        Raises:
            DimensionError: If the panel is malformed.
            NumericalError: If it is too short for the regression.
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
    """The Beveridge-Nelson decomposition of a fitted ARIMA(p, 1, q).

    With ``Delta y_t = mu + psi(L) e_t`` the trend is ``tau_t = y_t +
    sum_{k>=1} (E_t Delta y_{t+k} - mu)``, the level the forecast settles
    on once the drift is netted out, computed exactly in companion form
    (Morley 2002). The cycle ``y_t - tau_t`` is minus the sum of
    forecastable future changes.

    Args:
        result: A fitted :class:`~cultivars.univariate.ARIMA` with
            ``d = 1``, no seasonal part and no exogenous regressors.

    Raises:
        SpecificationError: If the result is not an ARIMA(p, 1, q) of
            that form.

    Example:
        >>> from cultivars.univariate import ARIMA
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(0.2 + rng.standard_normal(300))
        >>> res = BeveridgeNelsonDecomposition(ARIMA(y, order=(1, 1, 0)).fit()).compute()
        >>> res.trend.shape, res.offset
        ((299, 1), 1)
    """

    __slots__ = ("_result",)

    def __init__(self, result: ARMAResult) -> None:
        """Check the fitted model is one the decomposition is defined for."""
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
        """Decompose the levels.

        Returns:
            The :class:`DecompositionFilterResult`, one row shorter than the series
            (the first difference has no value at ``t = 0``).

        Raises:
            NumericalError: If the differenced series has a unit root.
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
