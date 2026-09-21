# filepath: /src/cultivars/forecast/comparison.py
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

"""Comparing forecasters: is the loss gap real, and does it persist?

Diebold-Mariano (1995) asks the plain question -- is the average loss
differential between two forecast series distinguishable from zero -- with
a variance that respects the serial correlation multi-step forecast errors
mechanically carry, and the Harvey-Leybourne-Newbold small-sample
correction always applied alongside, because evaluation windows in
macroeconomics are short. Giacomini-White (2006) sharpens the question to
a conditional one: not "who was better on average" but "was the gap
predictable from what we knew at the time," which is the version that
matters for choosing a forecaster going forward.

The inputs are aligned loss *series*, one value per evaluation origin, and
producing them is deliberately the caller's job: the rolling-origin loop
belongs in user driver code, because conflating model selection with model
evaluation inside a package is how look-ahead bias gets laundered. The
canonical loop, whose alignment this module cannot check and therefore
depends on:

    losses_a, losses_b = [], []
    for origin in range(start, len(y) - horizon):
        fit_a = ModelA(y[:origin], ...).fit(...)
        fit_b = ModelB(y[:origin], ...).fit(...)
        outcome = y[origin + horizon - 1]
        losses_a.append(score(fit_a, outcome))
        losses_b.append(score(fit_b, outcome))

Two standard abuses are warned against rather than silently permitted.
The test is about *forecasts*, not models: comparing nested models'
recursive forecasts breaks the asymptotics (Clark-McCracken territory,
named and not implemented). And the Giacomini-White framework wants
rolling-window estimation, whose finite memory is what makes its null
well-posed.

References:
    Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive
        accuracy. *Journal of Business & Economic Statistics*, 13(3),
        253-263.
    Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality
        of prediction mean squared errors. *International Journal of
        Forecasting*, 13(2), 281-291.
    Giacomini, R., & White, H. (2006). Tests of conditional predictive
        ability. *Econometrica*, 74(6), 1545-1578.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.stats as sst

from .._core import (
    _MIN_COMPARISON_ORIGINS,
    SummaryTable,
    _bartlett_long_run_variance,
    _clark_west,
    _forecast_encompassing,
    _mincer_zarnowitz,
    _validate_aligned_series,
)
from .._internals import _ForecastComparisonTest, _SummaryMixin
from ..exceptions import DimensionError, NumericalError, SpecificationError

__all__ = [
    "ClarkWestTest",
    "EncompassingTest",
    "ForecastComparison",
    "ForecastComparisonResult",
    "MincerZarnowitzTest",
    "clark_west",
    "encompassing",
    "mincer_zarnowitz",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ClarkWestTest(_ForecastComparisonTest):
    """Verdict of the Clark-West (2007) test that a nesting model forecasts better.

    The one-sided alternative is that the larger model improves on the
    smaller; under the null the smaller model is true and the larger one's
    extra parameters only add estimation noise, which the statistic
    removes before testing. A rejection says the added structure has
    predictive content; a non-rejection says it has not shown any on
    this window.

    Attributes:
        statistic: The adjusted-MSPE t-type statistic.
        pvalue: Its upper-tail standard normal p-value.
        adjusted_differential: Mean of the adjusted loss differential,
            ``e_r**2 - e_u**2 + (f_r - f_u)**2``; positive favors the
            larger model.
        mspe_restricted: Mean squared prediction error of the smaller model.
        mspe_unrestricted: Mean squared prediction error of the larger model.
        horizon: Forecast horizon behind the series.
        nobs: Evaluation origins.
    """

    adjusted_differential: float
    mspe_restricted: float
    mspe_unrestricted: float

    def _summary_table(self) -> SummaryTable:
        """Render as a table."""
        verdict = "larger model improves" if self.reject() else "no improvement shown"
        rows = (("Clark-West adjusted MSPE", f"{self.statistic:.4f}", f"{self.pvalue:.4f}"),)
        notes = (
            f"Adjusted loss differential {self.adjusted_differential:+.5f} (positive favors "
            "the larger model); raw MSPE "
            f"{self.mspe_restricted:.5f} restricted versus {self.mspe_unrestricted:.5f} "
            "unrestricted.",
            "One-sided standard normal reference, Bartlett long-run variance through "
            f"horizon - 1 = {self.horizon - 1} lags. The adjustment removes the estimation "
            "noise the larger model carries under the null, which is what makes Diebold-"
            "Mariano invalid for nested forecasts.",
        )
        return self._frame(
            title="Clark-West Nested Forecast Comparison",
            verdict=verdict,
            columns=("test", "statistic", "p-value"),
            rows=rows,
            notes=notes,
        )

    def __repr__(self) -> str:
        """Represent the test as a string."""
        return (
            f"ClarkWestTest(statistic={self.statistic:.4f}, pvalue={self.pvalue:.4g}, "
            f"horizon={self.horizon}, nobs={self.nobs})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MincerZarnowitzTest(_ForecastComparisonTest):
    """Verdict of the Mincer-Zarnowitz efficiency regression ``y = a + b f``.

    An unbiased, efficient point forecast has intercept zero and slope
    one. A slope below one says the forecast overreacts -- shrinking it
    toward its mean would help -- and above one that it underreacts; an
    intercept away from zero is bias. The joint restriction is tested
    with a HAC Wald statistic, since multi-step errors overlap.

    Attributes:
        intercept: Estimated ``a``.
        slope: Estimated ``b``.
        statistic: Wald statistic for ``(a, b) = (0, 1)``, chi-squared scale.
        pvalue: Upper-tail p-value of ``statistic / 2`` under ``F(2, T - 2)``.
        r_squared: Fit of the regression, the forecast's explanatory
            share of the outcome.
        horizon: Forecast horizon behind the series.
        nobs: Evaluation origins.
    """

    intercept: float
    slope: float
    r_squared: float

    def _summary_table(self) -> SummaryTable:
        """Render as a table."""
        verdict = "efficiency rejected" if self.reject() else "efficiency not rejected"
        rows = (
            ("intercept (0 under H0)", f"{self.intercept:.4f}", ""),
            ("slope (1 under H0)", f"{self.slope:.4f}", ""),
            ("Wald, F(2, T - 2) reference", f"{self.statistic:.4f}", f"{self.pvalue:.4f}"),
        )
        reading = (
            "a slope below one says the forecast overreacts and would gain from shrinkage "
            "toward its mean; above one, that it underreacts."
        )
        notes = (
            f"R-squared {self.r_squared:.3f}. Under the null the forecast is unbiased and "
            "efficient with respect to itself; " + reading,
            "HAC covariance with Bartlett weights through "
            f"horizon - 1 = {self.horizon - 1} lags and T / (T - 2) scaling, W / 2 referred "
            "to F(2, T - 2) (Mincer & Zarnowitz, 1969)."
            + (
                " At multi-step horizons the truncated kernel understates the long-run "
                "variance and the test over-rejects -- roughly 13% at nominal 5% for a "
                "four-step horizon on 100 origins -- so read a marginal rejection with care."
                if self.horizon > 1
                else ""
            ),
        )
        return self._frame(
            title="Mincer-Zarnowitz Efficiency Regression",
            verdict=verdict,
            columns=("quantity", "estimate", "p-value"),
            rows=rows,
            notes=notes,
        )

    def __repr__(self) -> str:
        """Represent the test as a string."""
        return (
            f"MincerZarnowitzTest(intercept={self.intercept:.4f}, slope={self.slope:.4f}, "
            f"statistic={self.statistic:.4f}, pvalue={self.pvalue:.4g}, nobs={self.nobs})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class EncompassingTest(_ForecastComparisonTest):
    """Verdict of the forecast encompassing test: does A already contain what B knows?

    Forecast A encompasses B when the optimal linear combination of the
    two puts zero weight on B. The one-sided alternative is that B carries
    information A lacks; a rejection says combining would help, and the
    reported weight is the least-squares share B would get.

    Attributes:
        statistic: The HLN-corrected statistic on ``e_A (e_A - e_B)``.
        pvalue: Its upper-tail ``t`` p-value.
        weight: Least-squares combination weight on B; zero under the null.
        horizon: Forecast horizon behind the series.
        nobs: Evaluation origins.
    """

    weight: float

    def _summary_table(self) -> SummaryTable:
        """Render as a table."""
        verdict = "B adds information" if self.reject() else "A encompasses B"
        rows = (("HLN encompassing", f"{self.statistic:.4f}", f"{self.pvalue:.4f}"),)
        notes = (
            f"Least-squares combination weight on B: {self.weight:+.3f} (zero under the null "
            "that A encompasses B; one would say B encompasses A).",
            "Diebold-Mariano machinery on e_A (e_A - e_B) with the Harvey-Leybourne-Newbold "
            f"correction, Bartlett long-run variance through horizon - 1 = {self.horizon - 1} "
            "lags, one-sided t reference (Harvey, Leybourne & Newbold, 1998).",
        )
        return self._frame(
            title="Forecast Encompassing",
            verdict=verdict,
            columns=("test", "statistic", "p-value"),
            rows=rows,
            notes=notes,
        )

    def __repr__(self) -> str:
        """Represent the test as a string."""
        return (
            f"EncompassingTest(statistic={self.statistic:.4f}, pvalue={self.pvalue:.4g}, "
            f"weight={self.weight:.3f}, nobs={self.nobs})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ForecastComparisonResult(_SummaryMixin):
    """The verdict on a loss differential, unconditional and conditional.

    Losses are negatively oriented, so a *negative* mean differential
    favors the first forecaster.

    Attributes:
        mean_differential: Average of ``losses_a - losses_b``.
        dm_statistic: Diebold-Mariano statistic under the standard normal
            null.
        dm_pvalue: Its two-sided p-value.
        hln_statistic: The Harvey-Leybourne-Newbold corrected statistic,
            referred to a ``t`` distribution -- the number to trust in the
            short evaluation windows macroeconomics actually has.
        hln_pvalue: Its two-sided p-value.
        gw_statistic: Giacomini-White conditional statistic (constant and
            lagged differential as instruments), chi-squared with two
            degrees of freedom under the null of no conditional
            predictability.
        gw_pvalue: Its upper-tail p-value.
        horizon: The forecast horizon the losses were produced at, which
            sets the variance's serial-correlation window.
        nobs: Evaluation origins.
    """

    mean_differential: float
    dm_statistic: float
    dm_pvalue: float
    hln_statistic: float
    hln_pvalue: float
    gw_statistic: float
    gw_pvalue: float
    horizon: int
    nobs: int

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        favored = "A" if self.mean_differential < 0.0 else "B"
        rows = (
            ("Diebold-Mariano", f"{self.dm_statistic:.4f}", f"{self.dm_pvalue:.4f}"),
            (
                "Harvey-Leybourne-Newbold",
                f"{self.hln_statistic:.4f}",
                f"{self.hln_pvalue:.4f}",
            ),
            (
                "Giacomini-White (conditional)",
                f"{self.gw_statistic:.4f}",
                f"{self.gw_pvalue:.4f}",
            ),
        )
        notes = [
            f"Mean loss differential {self.mean_differential:+.5f}: the "
            f"sign favors forecaster {favored} (losses are negatively "
            "oriented).",
            "The HLN row is the DM statistic with the small-sample "
            "correction, referred to a t distribution; in short evaluation "
            "windows it is the number to trust.",
            "Giacomini-White asks whether the gap was *predictable* from a "
            "constant and the lagged differential; its framework wants "
            "rolling-window estimation behind the losses.",
            "These are tests about forecasts, not models: nested-model "
            "comparisons need the Clark-McCracken corrections, which are "
            "deliberately not implemented here.",
        ]
        return SummaryTable(
            title="Forecast Comparison",
            metadata=(
                ("Origins", f"{self.nobs}"),
                ("Horizon", f"{self.horizon}"),
                ("Mean differential", f"{self.mean_differential:+.5f}"),
            ),
            columns=("test", "statistic", "p-value"),
            rows=rows,
            notes=tuple(notes),
        )


class ForecastComparison:
    """Test a loss differential between two forecasters.

    Args:
        losses_a: ``(T,)`` losses of the first forecaster, one per
            evaluation origin, negatively oriented.
        losses_b: ``(T,)`` losses of the second, aligned origin by origin
            -- an alignment this object cannot verify and wholly depends
            on.

    Raises:
        DimensionError: If the series' shapes disagree.
        SpecificationError: If the window is too short, or the series are
            numerically identical and there is no differential to test.
        NumericalError: If the losses are not finite.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> shared = rng.standard_normal(120) ** 2
        >>> a = shared + 0.05 * rng.standard_normal(120)
        >>> b = shared + 0.4 + 0.05 * rng.standard_normal(120)
        >>> verdict = ForecastComparison(a, b).compute()
        >>> bool(verdict.mean_differential < 0.0)
        True
        >>> bool(verdict.hln_pvalue < 0.01)
        True
    """

    __slots__ = ("_losses_a", "_losses_b")

    def __init__(self, losses_a: npt.ArrayLike, losses_b: npt.ArrayLike) -> None:
        """Validate and align the two loss series."""
        first = np.asarray(losses_a, dtype=np.float64).ravel()
        second = np.asarray(losses_b, dtype=np.float64).ravel()
        if first.shape != second.shape:
            raise DimensionError(
                f"the loss series must align origin by origin; got shapes "
                f"{first.shape} and {second.shape}."
            )
        if first.shape[0] < _MIN_COMPARISON_ORIGINS:
            raise SpecificationError(
                f"a comparison over {first.shape[0]} origins has no power "
                f"and unreliable size; provide at least {_MIN_COMPARISON_ORIGINS}."
            )
        if not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
            raise NumericalError("losses must be finite.")
        self._losses_a = first
        self._losses_b = second

    def compute(self, *, horizon: int = 1) -> ForecastComparisonResult:
        """Run the unconditional and conditional tests.

        Args:
            horizon: The forecast horizon behind the losses; the
                differential's variance sums autocovariances through
                ``horizon - 1`` lags with Bartlett weights, since an
                ``h``-step error is mechanically an MA(``h - 1``).

        Returns:
            The :class:`ForecastComparisonResult`.

        Raises:
            SpecificationError: If the horizon is not positive or exceeds
                what the window can support.
            NumericalError: If the conditional moment matrix degenerates.
        """
        if horizon < 1:
            raise SpecificationError(f"horizon must be at least 1; got {horizon}.")
        differential = self._losses_a - self._losses_b
        count = differential.shape[0]
        if horizon >= count // 2:
            raise SpecificationError(
                f"a horizon of {horizon} needs more than {2 * horizon} "
                f"evaluation origins; got {count}."
            )
        centered = differential - differential.mean()
        if float(np.abs(centered).max()) <= 1e-14 * max(float(np.abs(differential).max()), 1.0):
            raise SpecificationError(
                "the two loss series are numerically identical; there is no differential to test."
            )
        variance = _bartlett_long_run_variance(centered, horizon)
        mean = float(differential.mean())
        spread = float(np.sqrt(variance / count))
        for lag in range(1, horizon):
            weight = 1.0 - lag / horizon
            variance += 2.0 * weight * float(centered[lag:] @ centered[:-lag]) / count
        mean = float(differential.mean())
        spread = float(np.sqrt(max(variance, 1e-300) / count))
        dm = mean / spread
        dm_pvalue = 2.0 * float(sst.norm.sf(abs(dm)))
        adjust = float(np.sqrt((count + 1 - 2 * horizon + horizon * (horizon - 1) / count) / count))
        hln = dm * adjust
        hln_pvalue = 2.0 * float(sst.t.sf(abs(hln), count - 1))
        instruments = np.column_stack([np.ones(count - 1), differential[:-1]])
        moments = instruments * differential[1:, None]
        mean_moment = moments.mean(axis=0)
        outer = moments.T @ moments / (count - 1)
        try:
            solved = np.linalg.solve(outer, mean_moment)
        except np.linalg.LinAlgError as error:
            raise NumericalError(
                "the conditional moment matrix is singular; the losses "
                "carry no usable variation for the Giacomini-White test."
            ) from error
        gw = float((count - 1) * mean_moment @ solved)
        gw_pvalue = float(sst.chi2.sf(gw, 2))
        return ForecastComparisonResult(
            mean_differential=mean,
            dm_statistic=dm,
            dm_pvalue=dm_pvalue,
            hln_statistic=hln,
            hln_pvalue=hln_pvalue,
            gw_statistic=gw,
            gw_pvalue=gw_pvalue,
            horizon=int(horizon),
            nobs=count,
        )


def clark_west(
    realized: npt.ArrayLike,
    restricted: npt.ArrayLike,
    unrestricted: npt.ArrayLike,
    *,
    horizon: int = 1,
) -> ClarkWestTest:
    """Clark-West (2007) test that a nesting model forecasts better than the model it nests.

    Under the null that the smaller model is true, the larger one's extra
    parameters are estimated noise, so its mean squared prediction error
    is *expected* to exceed the smaller one's and Diebold-Mariano is biased
    against it. Clark and West subtract the noise term ``(f_r - f_u)**2``
    from the loss differential and test what remains with a one-sided
    normal reference, which is approximately correctly sized in the
    recursive and rolling schemes a backtest runs.

    Args:
        realized: ``(T,)`` outcomes.
        restricted: ``(T,)`` point forecasts of the nested model.
        unrestricted: ``(T,)`` point forecasts of the nesting model,
            aligned origin by origin -- ``BacktestResult.point[:, h - 1,
            j]`` from two backtests on the same schedule.
        horizon: The forecast horizon behind the series; sets the
            long-run variance's Bartlett window.

    Returns:
        The :class:`ClarkWestTest`.

    Raises:
        DimensionError: If the series do not align.
        SpecificationError: If there are too few origins or the horizon
            is not positive or too long for the window.
        NumericalError: If a series is not finite.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal(200)
        >>> y = 0.5 * x + rng.standard_normal(200)
        >>> verdict = clark_west(y, np.zeros(200), 0.5 * x)
        >>> bool(verdict.pvalue < 0.01)
        True
    """
    series, count = _validate_aligned_series(
        realized,
        restricted,
        unrestricted,
        horizon=horizon,
        minimum=_MIN_COMPARISON_ORIGINS,
        labels="realized, restricted, and unrestricted",
    )
    statistic, pvalue, adjusted = _clark_west(series[0], series[1], series[2], horizon=horizon)
    return ClarkWestTest(
        statistic=statistic,
        pvalue=pvalue,
        adjusted_differential=adjusted,
        mspe_restricted=float(((series[0] - series[1]) ** 2).mean()),
        mspe_unrestricted=float(((series[0] - series[2]) ** 2).mean()),
        horizon=int(horizon),
        nobs=int(count),
    )


def mincer_zarnowitz(
    realized: npt.ArrayLike, forecast: npt.ArrayLike, *, horizon: int = 1
) -> MincerZarnowitzTest:
    """Mincer-Zarnowitz efficiency regression of outcomes on point forecasts.

    ``y_t = a + b f_t + u_t`` with the joint test of ``(a, b) = (0, 1)``:
    the forecast is unbiased and cannot be improved by a linear
    recalibration of itself. Multi-step errors overlap, so the Wald
    covariance is HAC with Bartlett weights through ``horizon - 1`` lags;
    at those horizons the test over-rejects in short windows (about 13%
    at nominal 5% for four steps on 100 origins), and the record says so.

    Args:
        realized: ``(T,)`` outcomes.
        forecast: ``(T,)`` point forecasts, aligned origin by origin --
            ``BacktestResult.point[:, h - 1, j]`` against
            ``BacktestResult.realized[:, h - 1, j]``.
        horizon: The forecast horizon behind the series.

    Returns:
        The :class:`MincerZarnowitzTest`.

    Raises:
        DimensionError: If the series do not align.
        SpecificationError: If there are too few origins or the horizon
            is unusable.
        NumericalError: If a series is not finite or the forecasts are
            constant.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> f = rng.standard_normal(200)
        >>> verdict = mincer_zarnowitz(f + 0.5 * rng.standard_normal(200), f)
        >>> bool(verdict.pvalue > 0.05)
        True
        >>> bool(mincer_zarnowitz(0.5 * f + 0.5 * rng.standard_normal(200), f).pvalue < 0.01)
        True
    """
    (y, f), count = _validate_aligned_series(
        realized,
        forecast,
        horizon=horizon,
        minimum=_MIN_COMPARISON_ORIGINS,
        labels="realized and forecast",
    )
    intercept, slope, statistic, pvalue, r_squared = _mincer_zarnowitz(y, f, horizon=horizon)
    return MincerZarnowitzTest(
        intercept=intercept,
        slope=slope,
        statistic=statistic,
        pvalue=pvalue,
        r_squared=r_squared,
        horizon=int(horizon),
        nobs=int(count),
    )


def encompassing(
    realized: npt.ArrayLike,
    forecast_a: npt.ArrayLike,
    forecast_b: npt.ArrayLike,
    *,
    horizon: int = 1,
) -> EncompassingTest:
    """Test whether forecast A encompasses forecast B (Harvey, Leybourne & Newbold, 1998).

    A encompasses B when the optimal combination of the two puts no
    weight on B, so combining them would not help. The test asks whether
    ``e_A (e_A - e_B)`` has a positive mean, one-sided; a rejection says B
    carries information A lacks, and the reported weight is the share B
    would get in a least-squares combination. Run it both ways to learn
    whether either forecast is redundant.

    Args:
        realized: ``(T,)`` outcomes.
        forecast_a: ``(T,)`` point forecasts claimed to encompass.
        forecast_b: ``(T,)`` point forecasts claimed to be encompassed,
            aligned origin by origin.
        horizon: The forecast horizon behind the series.

    Returns:
        The :class:`EncompassingTest`.

    Raises:
        DimensionError: If the series do not align.
        SpecificationError: If there are too few origins, the horizon is
            unusable, or the two forecasts are identical.
        NumericalError: If a series is not finite.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> x, z = rng.standard_normal(200), rng.standard_normal(200)
        >>> y = x + z + 0.5 * rng.standard_normal(200)
        >>> bool(encompassing(y, x, x + z).pvalue < 0.01)  # x lacks what z knows
        True
        >>> bool(encompassing(y, x + z, x).pvalue > 0.05)  # x + z already has it
        True
    """
    (y, a, b), count = _validate_aligned_series(
        realized,
        forecast_a,
        forecast_b,
        horizon=horizon,
        minimum=_MIN_COMPARISON_ORIGINS,
        labels="realized and both forecasts",
    )
    statistic, pvalue, weight = _forecast_encompassing(y - a, y - b, horizon=horizon)
    return EncompassingTest(
        statistic=statistic, pvalue=pvalue, weight=weight, horizon=int(horizon), nobs=int(count)
    )
