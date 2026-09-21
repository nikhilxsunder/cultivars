# filepath: /src/cultivars/diagnostics/seasonality.py
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

"""Seasonal unit roots: is the seasonal pattern deterministic, or does it wander?

Seasonal differencing ``1 - L^s`` removes ``s`` unit roots at once, one
at every frequency ``2 pi k / s``, and a series rarely needs all of them:
a stable seasonal pattern is handled by dummies, a unit root at the zero
frequency by ``1 - L``, and only a pattern whose summers slowly turn
into winters by the seasonal difference. HEGY (1990) tests each root
separately in one regression, with the seasonal random walk as the null,
so the augmented Dickey-Fuller reading carries over frequency by
frequency: a ``t`` that cannot reject at the zero frequency and ``F``
statistics that reject at every seasonal one say "difference once, use
dummies". Canova and Hansen (1995) turn the null around, as KPSS does
for the zero frequency: stationary seasonality against a unit root at
the frequency tested, and a series that HEGY cannot reject a seasonal
root for and Canova-Hansen cannot reject stationarity for is one the
sample does not decide.

Both tests are read against simulated laws rather than the published
tables: HEGY's finite-sample null is drawn at the sample size and
deterministic set in hand for any even period, and the Canova-Hansen
limit is the generalized von Mises law, drawn as the integral of a
squared Brownian bridge.

References:
    Hylleberg, S., Engle, R. F., Granger, C. W. J., & Yoo, B. S. (1990).
        Seasonal integration and cointegration. *Journal of
        Econometrics*, 44(1-2), 215-238.
    Beaulieu, J. J., & Miron, J. A. (1993). Seasonal unit roots in
        aggregate U.S. data. *Journal of Econometrics*, 55(1-2), 305-328.
    Canova, F., & Hansen, B. E. (1995). Are seasonal patterns constant
        over time? A test for seasonal stability. *Journal of Business &
        Economic Statistics*, 13(3), 237-252.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from .._core import (
    _HEGY_REPLICATIONS,
    _canova_hansen,
    _frequency_labels,
    _hegy_null_draws,
    _hegy_statistics,
    _newey_west_bandwidth,
    _schwert_max_lags,
    _select_hegy_lags,
    _simulated_critical_values,
    _validate_seasonal,
    _von_mises_draws,
    validate_choice,
    validate_order,
)
from ..exceptions import SpecificationError
from .unit_roots import UnitRootTest

__all__ = ["SeasonalUnitRootTest", "canova_hansen", "hegy"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SeasonalUnitRootTest(UnitRootTest):
    """Verdict of a seasonal unit-root or seasonal-stability test, one frequency per row.

    A :class:`UnitRootTest` whose companions are the other frequencies
    of the same regression, and whose rows may reject in different
    tails: HEGY's ``t`` statistics at the real roots in the lower tail,
    its ``F`` statistics in the upper; every Canova-Hansen row in the
    upper. The record adds nothing to the fields, only the title and the
    reading notes a mixed table needs.

    Attributes:
        family: ``"HEGY"`` or ``"Canova-Hansen"``.
        period: Seasonal period ``s``.
    """

    family: str
    period: int

    def _title(self) -> str:
        return (
            "HEGY Seasonal Unit Root Test"
            if self.family == "HEGY"
            else "Canova-Hansen Seasonal Stability Test"
        )

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        return (("Period", str(self.period)), *UnitRootTest._metadata(self))

    def _notes(self) -> tuple[str, ...]:
        if self.family == "HEGY":
            return (
                "t statistics reject in the lower tail, F statistics in the upper; every row "
                "is read against its own simulated finite-sample law.",
                "Read: keep at frequency 0 and reject at every seasonal frequency means "
                "difference once and model the season with dummies; keep at a seasonal "
                "frequency means that root needs its factor of the seasonal difference.",
            )
        return (
            "Rejection lies in the upper tail; the joint row is the headline and the "
            "per-frequency rows locate the instability.",
            "The null is stationary seasonality, so this test and HEGY are read together "
            "as KPSS and ADF are.",
        )


def hegy(
    endog: npt.ArrayLike,
    *,
    period: int,
    trend: str = "c",
    seasonal: bool = True,
    lags: int | None = None,
    max_lags: int | None = None,
    method: str = "aic",
    replications: int = _HEGY_REPLICATIONS,
    seed: int | np.random.Generator | None = None,
) -> SeasonalUnitRootTest:
    """HEGY test of a unit root at each seasonal frequency, and at the zero frequency.

    ``(1 - L^s) y_t = d_t + pi_0 x_{0,t-1} + pi_{s/2} x_{s/2,t-1} + sum_k
    (pi_k^a x^a_{k,t-1} + pi_k^b x^b_{k,t-1}) + lags + e_t``, where each
    ``x`` is the series with every root of ``1 - L^s`` but one filtered
    out. The zero and Nyquist roots are tested by ``t`` in the lower
    tail, each harmonic pair by ``F``, and two joint ``F`` statistics --
    every seasonal root, and every root -- close the family. Under the
    null of a seasonal random walk none of these has a tabulated law
    that fits every period and deterministic set, so ``replications``
    seasonal random walks of the same length are run through the same
    regression and the p-values and critical values read off them; at
    ``s = 4``, ``T = 100`` with intercept and dummies the simulated 5%
    points reproduce Hylleberg et al.'s Table 1 to within Monte Carlo
    error. The augmentation absorbs serial correlation in the seasonal
    difference and is chosen by ``method`` on a common sample when
    ``lags`` is not given.

    Args:
        endog: The series.
        period: Seasonal period ``s``, even.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        seasonal: Whether seasonal dummies enter the regression, which
            is what makes the test robust to a deterministic seasonal
            pattern under the alternative.
        lags: Augmentation lags of the seasonal difference, or ``None``
            to select by ``method``.
        max_lags: Largest augmentation under selection; default
            Schwert's ``12 (T / 100)^(1/4)``.
        method: ``"aic"``, ``"bic"`` or ``"t-stat"`` for the selection.
            BIC under-augments against moving-average errors -- with an
            MA(1) coefficient of 0.5 it picks one or two lags where four
            are needed and the tests over-reject at 10-15% -- so AIC is
            the default, as for :func:`adf`.
        replications: Seasonal random walks behind the p-values.
        seed: Seed or generator for the simulation.

    Returns:
        The zero-frequency :class:`SeasonalUnitRootTest`, with the Nyquist ``t``,
        each harmonic ``F``, the joint seasonal ``F`` and the joint ``F``
        as ``companions``.

    Raises:
        SpecificationError: If the period, trend, lags, or replication
            count is unusable or the series is too short.
        NumericalError: If the regression is degenerate.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = rng.standard_normal(200)
        >>> for t in range(4, 200):
        ...     y[t] = y[t - 4] + rng.standard_normal()
        >>> verdict = hegy(y, period=4, replications=500, seed=0)
        >>> bool(verdict.reject()), all(not c.reject() for c in verdict.companions)
        (False, True)
    """
    y = _validate_seasonal(endog, period)
    trend = validate_choice(trend, ("n", "c", "ct"), "trend")
    if replications < 500:
        raise SpecificationError(f"replications must be at least 500; got {replications}.")
    if lags is None:
        ceiling = _schwert_max_lags(y.shape[0]) if max_lags is None else int(max_lags)
        ceiling = validate_order(ceiling, "max_lags", minimum=0)
        method = validate_choice(method, ("aic", "bic", "t-stat"), "method")
        chosen = _select_hegy_lags(y, period, trend, ceiling, method, seasonal=seasonal)
        label = f"{method} over 0..{ceiling}"
    else:
        chosen = validate_order(lags, "lags", minimum=0)
        label = "fixed"
    statistics, nobs = _hegy_statistics(y, period, trend, chosen, seasonal=seasonal)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    draws = _hegy_null_draws(
        period, y.shape[0], trend, seasonal=seasonal, replications=replications, rng=rng
    )
    spec = f"{trend}+s" if seasonal else trend
    labels = _frequency_labels(period)
    names = [
        "HEGY t, frequency 0",
        f"HEGY t, frequency {labels[0]}",
        *(f"HEGY F, frequency {label}" for label in labels[1:]),
        "HEGY F, all seasonal frequencies",
        "HEGY F, all frequencies",
    ]
    nulls = [
        "unit root at frequency 0",
        f"unit root at frequency {labels[0]}",
        *(f"unit root at frequency {label}" for label in labels[1:]),
        "unit roots at every seasonal frequency",
        "unit roots at every frequency",
    ]
    lower = [True, True] + [False] * (len(names) - 2)
    records = []
    for i, (name, null, tail) in enumerate(zip(names, nulls, lower, strict=True)):
        pvalue, cv = _simulated_critical_values(draws[:, i], float(statistics[i]), lower_tail=tail)
        records.append(
            SeasonalUnitRootTest(
                name=name,
                statistic=float(statistics[i]),
                pvalue=pvalue,
                critical_values=cv,
                null=null,
                lower_tail=tail,
                trend=spec,
                lags=chosen,
                nobs=nobs,
                method=label,
                family="HEGY",
                period=period,
            )
        )
    return replace(records[0], companions=tuple(records[1:]))


def canova_hansen(
    endog: npt.ArrayLike,
    *,
    period: int,
    trend: str = "c",
    lags: int = 0,
    bandwidth: int | None = None,
    n_draws: int = 5000,
    grid: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> SeasonalUnitRootTest:
    """Canova-Hansen test of stationary seasonality against a seasonal unit root.

    Regress ``y_t`` on the trend terms, ``s - 1`` trigonometric seasonal
    regressors and ``lags`` lags of ``y``; for the seasonal columns of
    the frequency under test, the LM statistic is ``T^{-2} sum_t F_t'
    Omega^{-1} F_t`` with ``F_t`` the partial sums of the scores and
    ``Omega`` their Bartlett long-run covariance. Its limit is the
    generalized von Mises law with ``q`` the number of columns tested,
    one at the Nyquist frequency, two at each harmonic and ``s - 1``
    jointly; the p-values come from ``n_draws`` simulated bridges, whose
    5% point at ``q = 1`` is KPSS's 0.46. Rejection lies in the upper
    tail. The joint test is the headline and the per-frequency tests
    say where the instability sits.

    Args:
        endog: The series.
        period: Seasonal period ``s``, even.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        lags: Lags of ``y`` among the regressors, which absorb short-run
            dynamics that would otherwise inflate the statistic.
        bandwidth: Bartlett lags of the long-run covariance; default
            Newey-West's ``4 (T / 100)^(2/9)``.
        n_draws: Simulated bridges behind each p-value.
        grid: Points on which each bridge is simulated.
        seed: Seed or generator for the simulation.

    Returns:
        The joint :class:`SeasonalUnitRootTest`, with the Nyquist and each
        harmonic frequency as ``companions``.

    Raises:
        SpecificationError: If the period, trend, lags, bandwidth, or
            simulation sizes are unusable or the series is too short.
        NumericalError: If a long-run covariance is singular.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> t = np.arange(300)
        >>> stable = np.cos(np.pi * t / 2) + rng.standard_normal(300)
        >>> bool(canova_hansen(stable, period=4, seed=0).reject())
        False
    """
    y = _validate_seasonal(endog, period)
    trend = validate_choice(trend, ("n", "c", "ct"), "trend")
    lags = validate_order(lags, "lags", minimum=0)
    if n_draws < 500 or grid < 100:
        raise SpecificationError(
            f"n_draws must be at least 500 and grid at least 100; got {n_draws}, {grid}."
        )
    width = _newey_west_bandwidth(y.shape[0]) if bandwidth is None else int(np.floor(bandwidth))
    width = validate_order(width, "bandwidth", minimum=0)
    statistics, degrees, nobs = _canova_hansen(y, period, trend, lags, bandwidth=width)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    draws = {
        int(q): _von_mises_draws(int(q), n_draws=n_draws, grid=grid, rng=rng) for q in set(degrees)
    }
    labels = _frequency_labels(period)
    names = [*(f"Canova-Hansen, frequency {label}" for label in labels), "Canova-Hansen, joint"]
    nulls = [
        *(f"stationary seasonality at frequency {label}" for label in labels),
        "stationary seasonality at every frequency",
    ]
    records = []
    for i, (name, null) in enumerate(zip(names, nulls, strict=True)):
        pvalue, cv = _simulated_critical_values(draws[int(degrees[i])], float(statistics[i]))
        records.append(
            SeasonalUnitRootTest(
                name=name,
                statistic=float(statistics[i]),
                pvalue=pvalue,
                critical_values=cv,
                null=null,
                lower_tail=False,
                trend=trend,
                lags=width,
                nobs=nobs,
                method=f"Bartlett kernel, {lags} lag(s) of y",
                family="Canova-Hansen",
                period=period,
            )
        )
    return replace(records[-1], companions=tuple(records[:-1]))
