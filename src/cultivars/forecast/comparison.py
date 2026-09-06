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

from .._core import SummaryTable
from .._internals import _SummaryMixin
from ..exceptions import DimensionError, NumericalError, SpecificationError


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
        if first.shape[0] < 8:
            raise SpecificationError(
                f"a comparison over {first.shape[0]} origins has no power "
                "and unreliable size; provide at least 8."
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
        variance = float(centered @ centered) / count
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
