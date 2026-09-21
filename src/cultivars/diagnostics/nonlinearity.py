# filepath: /src/cultivars/diagnostics/nonlinearity.py
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

"""Nonlinearity gateways: the tests that justify a regime model before it is fit.

A threshold or smooth-transition model is a claim that a linear
autoregression is wrong in a specific way, and the claim should be
tested on the linear model's own terms before the nonlinear one is
estimated. Four tests do that from different directions. Teräsvirta's
LM test nests the smooth transition in a cubic auxiliary regression and
carries a device for choosing between logistic and exponential
transitions. Tsay's arranged autoregression sorts the sample by the
threshold variable and asks whether the recursive predictive residuals
drift as the recursion crosses a threshold; it names no functional form.
Hansen's sup-F is the likelihood-ratio test of a SETAR against its
linear restriction, whose nuisance parameter -- the threshold -- is
unidentified under the null, so its p-value is simulated. Ramsey's RESET
is the general-purpose check that the linear fit has left a smooth
function of itself in the residual. The BDS statistic is the odd one
out: applied to residuals, it tests independence against any dependence
at all, linear or not, and rejects for neglected conditional
heteroskedasticity as readily as for a threshold, which is why it is
read after an ARCH test and not instead of one.

References:
    Teräsvirta, T. (1994). Specification, estimation, and evaluation of
        smooth transition autoregressive models. *Journal of the
        American Statistical Association*, 89(425), 208-218.
    Tsay, R. S. (1989). Testing and modeling threshold autoregressive
        processes. *Journal of the American Statistical Association*,
        84(405), 231-240.
    Hansen, B. E. (1996). Inference when a nuisance parameter is not
        identified under the null hypothesis. *Econometrica*, 64(2),
        413-430.
    Ramsey, J. B. (1969). Tests for specification errors in classical
        linear least-squares regression analysis. *Journal of the Royal
        Statistical Society B*, 31(2), 350-371.
    Brock, W. A., Dechert, W. D., Scheinkman, J. A., & LeBaron, B.
        (1996). A test for independence based on the correlation
        dimension. *Econometric Reviews*, 15(3), 197-235.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core import (
    _BDS_RADIUS,
    _CRITICAL_LEVELS,
    _DEFAULT_GRID,
    _DEFAULT_TRIM,
    _HANSEN_REPLICATIONS,
    _NULL,
    SummaryTable,
    _bds,
    _hansen_threshold,
    _reset_test,
    _simulated_critical_values,
    _terasvirta_lm,
    _tsay_arranged,
    validate_endog,
    validate_open_interval,
    validate_order,
)
from .._internals import _HypothesisTest
from ..exceptions import SpecificationError

__all__ = [
    "BDSTest",
    "LinearityTest",
    "bds",
    "hansen_threshold",
    "reset",
    "terasvirta",
    "tsay",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class LinearityTest(_HypothesisTest):
    """Verdict of a test of a linear autoregression against a nonlinear alternative.

    The four members share a null -- the AR(``order``) is correctly
    specified -- and differ in the alternative they aim at, which the
    record names: a rejection by Tsay says "threshold", a rejection by
    Teräsvirta says "smooth transition" and, through the escalation
    sequence carried as ``companions``, which kind. Hansen's test alone
    locates the threshold it found and reports simulated critical
    values, since its sup statistic has no tabulated law.

    Attributes:
        name: The test.
        statistic: The test statistic, ``F`` for the regression tests
            and sup-``F`` for Hansen.
        pvalue: Its p-value.
        df: ``(numerator, denominator)`` degrees of freedom of an
            ``F`` reference; ``None`` for a simulated law.
        null: The hypothesis under test.
        alternative: What a rejection points to.
        order: Autoregressive order of the linear null.
        delay: Delay of the transition variable, or ``None`` where the
            test has none.
        nobs: Observations in the auxiliary regression.
        threshold: Hansen's estimated threshold; ``None`` otherwise.
        critical_values: Simulated ``{"1%", "5%", "10%"}`` critical
            values for Hansen; ``None`` otherwise.
        suggestion: Teräsvirta's model-selection reading, ``"LSTAR"``
            or ``"ESTAR"``; ``None`` otherwise.
        companions: The steps of Teräsvirta's escalation sequence.
    """

    name: str
    pvalue: float
    df: tuple[int, int] | None
    null: str
    alternative: str
    order: int
    delay: int | None
    nobs: int
    threshold: float | None = None
    critical_values: dict[str, float] | None = None
    suggestion: str | None = None
    companions: tuple[LinearityTest, ...] = field(default=(), repr=False)

    def _row(self) -> tuple[str, ...]:
        """One table row."""
        df = "" if self.df is None else f"({self.df[0]}, {self.df[1]})"
        return (self.name, f"{self.statistic:.4f}", df, f"{self.pvalue:.4f}")

    def _summary_table(self) -> SummaryTable:
        """Render as a table, companions included."""
        verdict = "reject linearity" if self.reject() else "keep linearity"
        metadata = [
            ("Null", self.null),
            ("Alternative", self.alternative),
            ("Order", str(self.order)),
            ("Observations", str(self.nobs)),
            ("Verdict at 5%", verdict),
        ]
        if self.delay is not None:
            metadata.insert(3, ("Delay", str(self.delay)))
        if self.threshold is not None:
            metadata.append(("Threshold", f"{self.threshold:.4f}"))
        notes: list[str] = []
        if self.critical_values is not None:
            cv = ", ".join(f"{k}: {v:.3f}" for k, v in self.critical_values.items())
            notes.append(f"Simulated critical values {cv}.")
        if self.suggestion is not None:
            notes.append(
                f"Escalation sequence H04, H03, H02 read by Teräsvirta's rule: the "
                f"strongest rejection points to {self.suggestion}."
            )
        return SummaryTable(
            title=f"{self.name} Linearity Test",
            metadata=tuple(metadata),
            columns=("test", "statistic", "df", "p-value"),
            rows=(self._row(), *(c._row() for c in self.companions)),
            notes=tuple(notes),
        )

    def __repr__(self) -> str:
        """One-line verdict."""
        return (
            f"LinearityTest(name={self.name!r}, statistic={self.statistic:.4f}, "
            f"pvalue={self.pvalue:.4g}, order={self.order}, nobs={self.nobs})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class BDSTest(_HypothesisTest):
    """Verdict of the BDS test of independence at one embedding dimension.

    The record is the statistic at the largest dimension asked for, with
    the lower dimensions as ``companions``; a dependence that shows at
    every dimension is the usual signature of neglected conditional
    heteroskedasticity, one that appears only at high dimension is rarer
    and worth a look. The asymptotic normal reference is reliable from
    about 500 observations and over-rejects below that -- roughly 8% at
    nominal 5% on 250 Gaussian observations, 15% on 100 -- which the
    record says.

    Attributes:
        statistic: The standardized statistic ``W_m``.
        pvalue: Its two-sided standard normal p-value.
        dimension: The embedding dimension ``m``.
        epsilon: The radius, in the units of the series.
        nobs: Observations.
        companions: The lower dimensions ``2 .. m - 1``.
    """

    pvalue: float
    dimension: int
    epsilon: float
    nobs: int
    companions: tuple[BDSTest, ...] = field(default=(), repr=False)

    def _row(self) -> tuple[str, ...]:
        """One table row."""
        return (str(self.dimension), f"{self.statistic:.4f}", f"{self.pvalue:.4f}")

    def _summary_table(self) -> SummaryTable:
        """Render as a table over the embedding dimensions."""
        verdict = "reject independence" if self.reject() else "keep independence"
        rows = (*(c._row() for c in self.companions), self._row())
        notes = [
            "Two-sided standard normal reference. On residuals, a rejection at every "
            "dimension usually means neglected conditional heteroskedasticity; run an "
            "ARCH test before reading it as a threshold.",
        ]
        if self.nobs < 500:
            notes.append(
                f"With {self.nobs} observations the asymptotic reference over-rejects; "
                "treat p-values near the level as inconclusive."
            )
        return SummaryTable(
            title="BDS Independence Test",
            metadata=(
                ("Radius", f"{self.epsilon:.4f}"),
                ("Observations", str(self.nobs)),
                ("Verdict at 5%", verdict),
            ),
            columns=("dimension", "statistic", "p-value"),
            rows=rows,
            notes=tuple(notes),
        )

    def __repr__(self) -> str:
        """One-line verdict."""
        return (
            f"BDSTest(statistic={self.statistic:.4f}, pvalue={self.pvalue:.4g}, "
            f"dimension={self.dimension}, nobs={self.nobs})"
        )


def _check_specification(y: npt.NDArray[np.float64], order: int, delay: int) -> None:
    """Refuse an order or delay the series cannot carry.

    Raises:
        SpecificationError: If the sample is shorter than the auxiliary
            regression needs.
    """
    needed = max(order, delay) + 4 * order + 10
    if y.shape[0] < needed:
        raise SpecificationError(
            f"a nonlinearity test of order {order} and delay {delay} needs at least {needed} "
            f"observations; got {y.shape[0]}."
        )


def terasvirta(endog: npt.ArrayLike, *, order: int, delay: int = 1) -> LinearityTest:
    """Teräsvirta's (1994) LM-type test of linearity against smooth transition.

    The STAR alternative is replaced by its third-order Taylor expansion
    in the transition variable ``y_{t - delay}``, so the null of
    linearity becomes the exclusion of ``3 order`` cubic interaction
    terms from an auxiliary regression of the AR residual, referred to
    ``F`` because the chi-squared form over-rejects at the sample sizes
    the test meets. The escalation sequence ``H04, H03, H02`` in
    ``companions`` is the paper's rule for choosing the transition: when
    the quadratic step ``H03`` rejects most strongly the transition is
    exponential (ESTAR), otherwise logistic (LSTAR). At order 1 the test
    holds size at 4% on 100 observations and rejects a two-regime SETAR
    or LSTAR with 250 observations essentially always.

    Args:
        endog: The series.
        order: Autoregressive order of the linear null.
        delay: Delay of the transition variable.

    Returns:
        The :class:`LinearityTest` with ``suggestion`` set.

    Raises:
        SpecificationError: If the order or delay is not positive or the
            series is too short.
        NumericalError: If the auxiliary regressions are degenerate.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(300)
        >>> for t in range(1, 300):
        ...     g = 1.0 / (1.0 + np.exp(-5.0 * y[t - 1]))
        ...     y[t] = 0.8 * y[t - 1] - 1.4 * y[t - 1] * g + rng.standard_normal()
        >>> verdict = terasvirta(y, order=1)
        >>> bool(verdict.reject()), verdict.suggestion
        (True, 'LSTAR')
    """
    y = validate_endog(endog)
    order = validate_order(order, "order", minimum=1)
    delay = validate_order(delay, "delay", minimum=1)
    _check_specification(y, order, delay)
    (statistic, pvalue, df1, df2), steps = _terasvirta_lm(y, order, delay)
    nobs = y.shape[0] - max(order, delay)
    null = _NULL.format(order=order)
    companions = tuple(
        LinearityTest(
            name=label,
            statistic=s,
            pvalue=p,
            df=(a, b),
            null=hypothesis,
            alternative="smooth transition",
            order=order,
            delay=delay,
            nobs=nobs,
        )
        for label, hypothesis, (s, p, a, b) in zip(
            ("H04", "H03", "H02"),
            ("cubic terms vanish", "quadratic terms vanish given cubic", "linear terms vanish"),
            steps,
            strict=True,
        )
    )
    suggestion = "ESTAR" if steps[1][1] < min(steps[0][1], steps[2][1]) else "LSTAR"
    return LinearityTest(
        name="Teräsvirta",
        statistic=statistic,
        pvalue=pvalue,
        df=(df1, df2),
        null=null,
        alternative="smooth transition",
        order=order,
        delay=delay,
        nobs=nobs,
        suggestion=suggestion,
        companions=companions,
    )


def tsay(endog: npt.ArrayLike, *, order: int, delay: int = 1) -> LinearityTest:
    """Tsay's (1989) arranged-autoregression test of linearity against a threshold.

    The AR rows are sorted by ``y_{t - delay}`` and fitted recursively
    down the sorted sample from ``3 sqrt(T) + order`` rows on; under
    linearity the standardized one-step predictive residuals are white
    and orthogonal to the regressors, while a threshold shifts their
    mean once the recursion crosses it. The ``F`` test of the predictive
    residuals on the regressors is the statistic. It assumes no
    functional form for the transition and is the natural first look
    before a SETAR; at order 1 it holds 5% size on 100 observations.

    Args:
        endog: The series.
        order: Autoregressive order of the linear null.
        delay: Delay of the threshold variable.

    Returns:
        The :class:`LinearityTest`.

    Raises:
        SpecificationError: If the order or delay is not positive or the
            series is too short.
        NumericalError: If the recursion cannot start or the final
            regression is degenerate.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(300)
        >>> for t in range(1, 300):
        ...     slope = 0.8 if y[t - 1] <= 0.0 else -0.5
        ...     y[t] = slope * y[t - 1] + rng.standard_normal()
        >>> bool(tsay(y, order=1).reject())
        True
    """
    y = validate_endog(endog)
    order = validate_order(order, "order", minimum=1)
    delay = validate_order(delay, "delay", minimum=1)
    _check_specification(y, order, delay)
    statistic, pvalue, df1, df2 = _tsay_arranged(y, order, delay)
    return LinearityTest(
        name="Tsay",
        statistic=statistic,
        pvalue=pvalue,
        df=(df1, df2),
        null=_NULL.format(order=order),
        alternative="threshold",
        order=order,
        delay=delay,
        nobs=y.shape[0] - max(order, delay),
    )


def reset(endog: npt.ArrayLike, *, order: int, powers: int = 3) -> LinearityTest:
    """Ramsey's RESET on an autoregression: do powers of the fitted value explain the residual?

    Powers ``2 .. powers`` of the AR(``order``) fitted value are added to
    the regression and their joint exclusion tested by ``F``. The test
    names no alternative -- any smooth misspecification of the
    conditional mean raises the fitted value's powers -- so it is the
    general-purpose check to run when the regime tests disagree.

    Args:
        endog: The series.
        order: Autoregressive order of the linear null.
        powers: Highest power of the fitted value added; ``2`` tests
            the square alone.

    Returns:
        The :class:`LinearityTest`.

    Raises:
        SpecificationError: If the order is not positive, ``powers`` is
            below 2, or the series is too short.
        NumericalError: If the augmented regression is degenerate.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(300)
        >>> for t in range(1, 300):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal()
        >>> bool(reset(y, order=1).pvalue > 0.05)
        True
    """
    y = validate_endog(endog)
    order = validate_order(order, "order", minimum=1)
    powers = validate_order(powers, "powers", minimum=2)
    _check_specification(y, order, 1)
    statistic, pvalue, df1, df2 = _reset_test(y, order, powers)
    return LinearityTest(
        name="RESET",
        statistic=statistic,
        pvalue=pvalue,
        df=(df1, df2),
        null=_NULL.format(order=order),
        alternative="smooth misspecification",
        order=order,
        delay=None,
        nobs=y.shape[0] - order,
    )


def hansen_threshold(
    endog: npt.ArrayLike,
    *,
    order: int,
    delay: int | None = None,
    trim: float = _DEFAULT_TRIM,
    n_grid: int = _DEFAULT_GRID,
    replications: int = _HANSEN_REPLICATIONS,
    seed: int | np.random.Generator | None = None,
) -> LinearityTest:
    """Hansen's (1996) sup-F test of a SETAR against its linear restriction.

    At every candidate threshold on the trimmed quantile grid of
    ``y_{t - delay}`` the two-regime regression is fitted and the ``F``
    statistic for equal coefficients computed; the sup over the grid is
    the statistic, and its maximizer the threshold estimate. Because the
    threshold is unidentified under the null the sup has no tabulated
    law: the p-value comes from ``replications`` draws in which the
    response is replaced by standard normal noise with the regressors
    and splits held fixed, Hansen's fixed-regressor simulation. Leaving
    ``delay`` as ``None`` searches ``1 .. order`` jointly, and the
    simulated sup ranges over the same delays so the search is paid for.
    Homoskedastic errors are assumed; on 100 observations at order 1 the
    size is 5.5%, on 250 it is 3.5%.

    Args:
        endog: The series.
        order: Autoregressive order per regime.
        delay: Delay of the threshold variable; ``None`` searches
            ``1 .. order``.
        trim: Fraction of the sorted threshold variable excluded at
            each end of the grid.
        n_grid: Candidate thresholds per delay.
        replications: Simulated sup statistics behind the p-value.
        seed: Seed or generator for the simulation.

    Returns:
        The :class:`LinearityTest` with ``threshold`` and simulated
        ``critical_values`` set.

    Raises:
        SpecificationError: If the order, delay, trim, grid, or
            replication count is unusable or the series is too short.
        NumericalError: If no admissible split exists.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(300)
        >>> for t in range(1, 300):
        ...     slope = 0.8 if y[t - 1] <= 0.0 else -0.5
        ...     y[t] = slope * y[t - 1] + rng.standard_normal()
        >>> verdict = hansen_threshold(y, order=1, replications=500, seed=0)
        >>> bool(verdict.reject()), bool(abs(verdict.threshold) < 0.5)
        (True, True)
    """
    y = validate_endog(endog)
    order = validate_order(order, "order", minimum=1)
    trim = validate_open_interval(trim, "trim", low=0.0, high=0.5)
    n_grid = validate_order(n_grid, "n_grid", minimum=1)
    if replications < 200:
        raise SpecificationError(f"replications must be at least 200; got {replications}.")
    delays = (
        tuple(range(1, order + 1))
        if delay is None
        else (validate_order(delay, "delay", minimum=1),)
    )
    _check_specification(y, order, max(delays))
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    statistic, pvalue, threshold, chosen, simulated = _hansen_threshold(
        y, order, delays, trim=trim, n_grid=n_grid, replications=replications, rng=rng
    )
    _p, critical_values = _simulated_critical_values(simulated, statistic)
    return LinearityTest(
        name="Hansen sup-F",
        statistic=statistic,
        pvalue=pvalue,
        df=None,
        null=_NULL.format(order=order),
        alternative="threshold",
        order=order,
        delay=chosen,
        nobs=y.shape[0] - max(order, max(delays)),
        threshold=threshold,
        critical_values={k: critical_values[k] for k in _CRITICAL_LEVELS},
    )


def bds(endog: npt.ArrayLike, *, dimension: int = 3, epsilon: float | None = None) -> BDSTest:
    """The BDS test of independence, usually on residuals.

    The correlation integral of the ``m``-histories at radius
    ``epsilon`` is compared with the ``m``-th power of the one-point
    integral, which is what independence implies; the standardized gap
    is asymptotically standard normal. The radius defaults to ``1.5``
    standard deviations of the series, the choice with the best size in
    the Brock et al. tables, and the reference is trustworthy from about
    500 observations. On GARCH residuals with 500 observations the test
    rejects about 90% of the time, which is its most common use: what
    the ARCH test says with a form, BDS says without one.

    Args:
        endog: The series, typically standardized residuals.
        dimension: Largest embedding dimension; ``2`` and every dimension
            up to it are reported.
        epsilon: Radius in the units of the series; ``None`` takes
            ``1.5`` standard deviations.

    Returns:
        The :class:`BDSTest` at ``dimension``, lower dimensions as
        ``companions``.

    Raises:
        SpecificationError: If the dimension is below 2, the radius is
            not positive, or the series is too short.
        NumericalError: If the radius leaves no pairs close or every
            pair close.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> bool(bds(rng.standard_normal(600)).pvalue > 0.01)
        True
    """
    y = validate_endog(endog)
    dimension = validate_order(dimension, "dimension", minimum=2)
    if y.shape[0] < 10 * dimension + 50:
        raise SpecificationError(
            f"the BDS test at dimension {dimension} needs at least {10 * dimension + 50} "
            f"observations; got {y.shape[0]}."
        )
    radius = _BDS_RADIUS * float(y.std()) if epsilon is None else float(epsilon)
    if not radius > 0.0:
        raise SpecificationError(f"epsilon must be positive; got {radius}.")
    statistics, pvalues = _bds(y, dimension, radius)
    nobs = y.shape[0]
    companions = tuple(
        BDSTest(statistic=float(s), pvalue=float(p), dimension=m, epsilon=radius, nobs=nobs)
        for m, s, p in zip(range(2, dimension), statistics[:-1], pvalues[:-1], strict=True)
    )
    return BDSTest(
        statistic=float(statistics[-1]),
        pvalue=float(pvalues[-1]),
        dimension=dimension,
        epsilon=radius,
        nobs=nobs,
        companions=companions,
    )
