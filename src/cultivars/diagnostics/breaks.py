# filepath: /src/cultivars/diagnostics/breaks.py
#
# [MIT header, Copyright (c) 2026 Nikhil Sunder]

"""Structural-break tests: whether one regression held over the whole sample.

A break at a *known* date is a Chow test and needs nothing here. The
tests below are for the case that matters in practice, a break at a date
the data must locate. :func:`sup_wald` scans the trimmed window and
reports the largest Wald statistic (Andrews, 1993) together with the
exponential and average functionals of Andrews and Ploberger (1994),
whose limit is a Brownian-bridge functional this module simulates at the
trimming actually used, so the p-values are the asymptotic ones rather
than an interpolation of a printed table. :func:`bai_perron` extends the
search to several breaks with the dynamic-programming global minimizer
of Bai and Perron (2003), selects their number by an information
criterion or the sequential ``sup F(l + 1 | l)`` procedure, and reports
every partition considered. :func:`cusum` is the recursive-residual
diagnostic of Brown, Durbin and Evans (1975): a path that crosses its
boundary says the coefficients drifted, without a date to name.

Each takes a target and, optionally, regressors; with none, the
regression is on the deterministic terms alone, so the test is for a
break in the mean or trend. Serially correlated errors are not corrected
for: include the lags the dynamics call for in ``exog``, or read the
result as a diagnostic rather than a test.

References:
    Andrews, D. W. K. (1993). Tests for parameter instability and
        structural change with unknown change point. *Econometrica*,
        61(4), 821-856.
    Andrews, D. W. K., & Ploberger, W. (1994). Optimal tests when a
        nuisance parameter is present only under the alternative.
        *Econometrica*, 62(6), 1383-1414.
    Bai, J., & Perron, P. (1998). Estimating and testing linear models
        with multiple structural changes. *Econometrica*, 66(1), 47-78.
    Bai, J., & Perron, P. (2003). Computation and analysis of multiple
        structural change models. *Journal of Applied Econometrics*,
        18(1), 1-22.
    Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for
        testing the constancy of regression relationships over time.
        *Journal of the Royal Statistical Society B*, 37(2), 149-192.

Example:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = np.concatenate([rng.standard_normal(100), 2.0 + rng.standard_normal(100)])
    >>> test = sup_wald(y, seed=0)
    >>> test.reject(), 90 <= test.break_index <= 110
    (True, True)
    >>> bai_perron(y, max_breaks=3).n_breaks
    1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from .._core import (
    _CRITICAL_ALPHAS,
    _CRITICAL_LEVELS,
    _CUSUM_BOUNDARY,
    SummaryTable,
    _bai_perron_partition,
    _bridge_functionals,
    _cusum_squares_quantiles,
    _recursive_residuals,
    _segment_ssr,
    _simulated_critical_values,
    _validate_regression,
)
from .._internals import _TabulatedTest
from ..exceptions import SpecificationError

__all__ = ["BreakTest", "MultipleBreakTest", "bai_perron", "cusum", "sup_wald"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class BreakTest(_TabulatedTest):
    """Verdict of a test for one structural break at an unknown date.

    Andrews' (1993) sup-Wald and Andrews and Ploberger's (1994) exp- and
    ave-Wald share one limit process, a ``q``-dimensional Brownian bridge
    over the trimmed window, and the p-values here come from that limit
    simulated at the trimming actually used; the CUSUM tests of Brown,
    Durbin and Evans (1975) compare a recursive-residual path with a
    boundary, and the record carries the path so the crossing can be
    seen. Rejection always lies in the upper tail.

    Attributes:
        break_index: First observation of the new regime at the sup, or
            the first boundary crossing; ``None`` when nothing is located.
        n_restrictions: Coefficients allowed to change, ``q``.
        trimming: Fraction of the sample excluded at each end, or
            ``None`` for a boundary test.
        path: The statistic path over candidate dates, or the CUSUM path.
        bounds: The boundary at the tabulated levels, ``(3, n)``, for a
            CUSUM test; ``None`` otherwise.
    """

    break_index: int | None
    n_restrictions: int
    trimming: float | None
    path: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    bounds: npt.NDArray[np.float64] | None = field(default=None, repr=False)

    def _verdict(self) -> str:
        try:
            rejected = self.reject()
        except SpecificationError:
            return "see critical values"
        return "break" if rejected else "no break"

    def _title(self) -> str:
        return f"{self.name} Structural Break Test"

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        out = [
            ("Null", self.null),
            ("Restrictions", str(self.n_restrictions)),
            ("Observations", str(self.nobs)),
        ]
        if self.trimming is not None:
            out.append(("Trimming", f"{self.trimming:.2f}"))
        if self.break_index is not None:
            out.append(("Break at", str(self.break_index)))
        return tuple(out)

    def _notes(self) -> tuple[str, ...]:
        return ()

    def _repr_fields(self) -> tuple[str, ...]:
        return (f"break_index={self.break_index}",)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MultipleBreakTest:
    """Bai and Perron's (1998, 2003) multiple-break analysis of a linear regression.

    For every number of breaks up to ``max_breaks`` the record carries
    the global least-squares partition; the selected number comes from
    the information criterion asked for, or from the sequential
    ``sup F(l + 1 | l)`` procedure whose p-values follow from the
    single-break limit raised to the power ``l + 1``.

    Attributes:
        break_indices: First observation of each new regime under the
            selected number of breaks.
        n_breaks: The selected number.
        criterion: ``"bic"``, ``"lwz"`` or ``"sequential"``.
        ssr: Sum of squared residuals for ``0 .. max_breaks`` breaks.
        bic: Bayesian information criterion for each count.
        lwz: Liu-Wu-Zidek criterion for each count.
        partitions: The global partition for each count.
        sequential: The ``sup F(l + 1 | l)`` tests, one per step taken.
        n_restrictions: Regression coefficients, ``q``.
        trimming: Minimum segment length as a fraction of the sample.
        nobs: Observations.
    """

    break_indices: tuple[int, ...]
    n_breaks: int
    criterion: str
    ssr: npt.NDArray[np.float64] = field(repr=False)
    bic: npt.NDArray[np.float64] = field(repr=False)
    lwz: npt.NDArray[np.float64] = field(repr=False)
    partitions: tuple[tuple[int, ...], ...] = field(repr=False)
    sequential: tuple[BreakTest, ...] = field(repr=False)
    n_restrictions: int
    trimming: float
    nobs: int

    @property
    def max_breaks(self) -> int:
        """Largest number of breaks considered."""
        return int(self.ssr.shape[0] - 1)

    def summary(self) -> SummaryTable:
        """Render as a table over the number of breaks."""
        rows = []
        for m in range(self.max_breaks + 1):
            dates = ", ".join(str(i) for i in self.partitions[m]) or "-"
            rows.append(
                (str(m), f"{self.ssr[m]:.4f}", f"{self.bic[m]:.4f}", f"{self.lwz[m]:.4f}", dates)
            )
        notes = tuple(
            f"sup F({step + 1} | {step}) = {t.statistic:.3f}"
            + ("" if t.pvalue is None else f", p = {t.pvalue:.4f}")
            for step, t in enumerate(self.sequential)
        )
        return SummaryTable(
            title="Bai-Perron Multiple Breaks",
            metadata=(
                ("Selected breaks", str(self.n_breaks)),
                ("Criterion", self.criterion),
                ("Break dates", ", ".join(str(i) for i in self.break_indices) or "none"),
                ("Trimming", f"{self.trimming:.2f}"),
                ("Observations", str(self.nobs)),
            ),
            columns=("breaks", "SSR", "BIC", "LWZ", "dates"),
            rows=tuple(rows),
            notes=notes,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MultipleBreakTest(n_breaks={self.n_breaks}, break_indices={self.break_indices}, "
            f"criterion={self.criterion!r}, nobs={self.nobs})"
        )


def sup_wald(
    endog: npt.ArrayLike,
    exog: npt.ArrayLike | None = None,
    *,
    trend: str = "c",
    trimming: float = 0.15,
    n_draws: int = 5000,
    grid: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> BreakTest:
    """Andrews' sup-Wald test for one break at an unknown date, with exp- and ave-Wald.

    At every date in the trimmed window the regression is split, the
    Wald statistic for equality of the two coefficient vectors computed
    under a common error variance, and the sup, the log-average of
    ``exp(W / 2)``, and the average taken over the window. All three
    p-values come from ``n_draws`` simulated paths of the limit process,
    a ``q``-dimensional Brownian bridge on a grid of ``grid`` points, at
    the stated trimming; the sup's 5% value at ``q = 1``, ``trimming =
    0.15`` reproduces Andrews' corrected 8.68 to within Monte Carlo
    error.

    Args:
        endog: The target.
        exog: Regressors ``(T, k)`` or ``(T,)``, or ``None``.
        trend: Deterministic terms, ``"n"``, ``"c"`` or ``"ct"``.
        trimming: Fraction of the sample excluded at each end.
        n_draws: Limit-process paths for the p-values.
        grid: Points on which each path is simulated.
        seed: Seed or generator for the limit simulation.

    Returns:
        The sup-Wald :class:`BreakTest`, with exp- and ave-Wald as
        ``companions`` and the Wald path over candidate dates in
        ``path`` (``nan`` outside the window).

    Raises:
        SpecificationError: If the trend, trimming, or counts are
            unusable, or the window leaves no candidate.
        DimensionError: If the regressors do not align.
        NumericalError: If the design is rank deficient.
    """
    if not 0.0 < trimming < 0.5:
        raise SpecificationError(f"trimming must lie in (0, 0.5); got {trimming}.")
    if n_draws < 500 or grid < 100:
        raise SpecificationError(
            f"n_draws must be at least 500 and grid at least 100; got {n_draws}, {grid}."
        )
    y, design = _validate_regression(endog, exog, trend)
    nobs, q = design.shape
    first = max(int(np.floor(trimming * nobs)), q + 1)
    last = nobs - first
    if last <= first:
        raise SpecificationError("the trimmed window leaves no candidate break date.")
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    ssr_full = float(np.sum((y - design @ coef) ** 2))
    path = np.full(nobs, np.nan)
    for tb in range(first, last + 1):
        before, after = design[:tb], design[tb:]
        b0, _, _, _ = np.linalg.lstsq(before, y[:tb], rcond=None)
        b1, _, _, _ = np.linalg.lstsq(after, y[tb:], rcond=None)
        ssr_split = float(np.sum((y[:tb] - before @ b0) ** 2) + np.sum((y[tb:] - after @ b1) ** 2))
        path[tb] = (ssr_full - ssr_split) / (ssr_split / (nobs - 2 * q))
    window = path[first : last + 1]
    sup = float(window.max())
    break_index = first + int(np.argmax(window))
    exp_w = float(np.log(np.mean(np.exp(0.5 * window))))
    ave = float(window.mean())
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    sup_draws, exp_draws, ave_draws = _bridge_functionals(
        q, trimming, n_draws=n_draws, grid=grid, rng=rng
    )
    null = "no break in the coefficients"

    def record(name: str, statistic: float, draws: npt.NDArray[np.float64]) -> BreakTest:
        pvalue, cv = _simulated_critical_values(draws, statistic)
        return BreakTest(
            name=name,
            statistic=statistic,
            pvalue=pvalue,
            critical_values=cv,
            null=null,
            break_index=break_index if name == "sup-Wald" else None,
            n_restrictions=q,
            trimming=trimming,
            nobs=nobs,
        )

    companions = (record("exp-Wald", exp_w, exp_draws), record("ave-Wald", ave, ave_draws))
    pvalue, cv = _simulated_critical_values(sup_draws, sup)
    return BreakTest(
        name="sup-Wald",
        statistic=sup,
        pvalue=pvalue,
        critical_values=cv,
        null=null,
        break_index=break_index,
        n_restrictions=q,
        trimming=trimming,
        nobs=nobs,
        companions=companions,
        path=path,
    )


def bai_perron(
    endog: npt.ArrayLike,
    exog: npt.ArrayLike | None = None,
    *,
    trend: str = "c",
    max_breaks: int = 5,
    trimming: float = 0.15,
    criterion: str = "bic",
    alpha: float = 0.05,
    n_draws: int = 5000,
    grid: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> MultipleBreakTest:
    """Bai-Perron estimation of multiple breaks in a linear regression.

    The sum of squared residuals of every admissible segment is tabled
    once, and the global least-squares partition for each number of
    breaks up to ``max_breaks`` follows by dynamic programming. The
    number of breaks is chosen by BIC, by the Liu-Wu-Zidek criterion the
    authors found less prone to over-fitting, or sequentially: starting
    from no break, the largest ``sup F`` over the current segments tests
    ``l`` against ``l + 1`` breaks, and the procedure stops at the first
    non-rejection. Sequential p-values follow from the single-break
    limit, ``P(sup F(l + 1 | l) > x) = 1 - G(x)^(l + 1)``.

    Args:
        endog: The target.
        exog: Regressors, or ``None``.
        trend: Deterministic terms.
        max_breaks: Largest number of breaks considered.
        trimming: Minimum segment length as a fraction of the sample.
        criterion: ``"bic"``, ``"lwz"`` or ``"sequential"``.
        alpha: Level of the sequential tests.
        n_draws: Limit-process paths for the sequential p-values.
        grid: Points on which each path is simulated.
        seed: Seed or generator for the limit simulation.

    Returns:
        The :class:`MultipleBreakTest`.

    Raises:
        SpecificationError: If the criterion, trimming, level, or counts
            are unusable, or the segments do not fit the sample.
        DimensionError: If the regressors do not align.
        NumericalError: If the design is rank deficient.
    """
    if criterion not in ("bic", "lwz", "sequential"):
        raise SpecificationError(
            f"criterion must be 'bic', 'lwz' or 'sequential'; got {criterion!r}."
        )
    if not 0.0 < trimming < 0.5:
        raise SpecificationError(f"trimming must lie in (0, 0.5); got {trimming}.")
    if max_breaks < 1:
        raise SpecificationError(f"max_breaks must be at least 1; got {max_breaks}.")
    if not 0.0 < alpha < 1.0:
        raise SpecificationError(f"alpha must lie in (0, 1); got {alpha}.")
    y, design = _validate_regression(endog, exog, trend)
    nobs, q = design.shape
    min_size = max(int(np.floor(trimming * nobs)), q + 1)
    if (max_breaks + 1) * min_size > nobs:
        raise SpecificationError(
            f"{max_breaks} breaks with segments of at least {min_size} observations do not "
            f"fit {nobs}; lower max_breaks or trimming."
        )
    table = _segment_ssr(y, design, min_size)
    ssr = np.empty(max_breaks + 1)
    partitions: list[tuple[int, ...]] = []
    for m in range(max_breaks + 1):
        value, breaks = _bai_perron_partition(table, m, min_size)
        ssr[m] = value
        partitions.append(breaks)
    counts = np.arange(max_breaks + 1)
    n_params = (counts + 1) * q + counts
    bic = np.log(ssr / nobs) + n_params * np.log(nobs) / nobs
    lwz = np.log(ssr / (nobs - n_params)) + n_params * 0.299 * np.log(nobs) ** 2.1 / nobs

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    sup_draws, _, _ = _bridge_functionals(q, trimming, n_draws=n_draws, grid=grid, rng=rng)
    sequential: list[BreakTest] = []
    selected_sequential = 0
    for step in range(max_breaks):
        segments = (0, *partitions[step], nobs)
        best_stat, best_date = -np.inf, None
        for start, end in pairwise(segments):
            length = end - start
            if length < 2 * min_size:
                continue
            base = table[start, end]
            for tb in range(start + min_size, end - min_size + 1):
                split = table[start, tb] + table[tb, end]
                stat = (base - split) / (split / (length - 2 * q))
                if stat > best_stat:
                    best_stat, best_date = stat, tb
        if best_date is None:
            break
        survival = float(np.mean(sup_draws >= best_stat))
        pvalue = 1.0 - (1.0 - survival) ** (step + 1)
        cv = {
            label: float(np.quantile(sup_draws, (1.0 - level) ** (1.0 / (step + 1))))
            for label, level in zip(_CRITICAL_LEVELS, _CRITICAL_ALPHAS, strict=True)
        }
        test = BreakTest(
            name=f"sup F({step + 1} | {step})",
            statistic=float(best_stat),
            pvalue=pvalue,
            critical_values=cv,
            null=f"{step} breaks against {step + 1}",
            break_index=best_date,
            n_restrictions=q,
            trimming=trimming,
            nobs=nobs,
        )
        sequential.append(test)
        if pvalue >= alpha:
            break
        selected_sequential = step + 1
    if criterion == "bic":
        n_breaks = int(np.argmin(bic))
    elif criterion == "lwz":
        n_breaks = int(np.argmin(lwz))
    else:
        n_breaks = selected_sequential
    return MultipleBreakTest(
        break_indices=partitions[n_breaks],
        n_breaks=n_breaks,
        criterion=criterion,
        ssr=ssr,
        bic=np.asarray(bic, dtype=np.float64),
        lwz=np.asarray(lwz, dtype=np.float64),
        partitions=tuple(partitions),
        sequential=tuple(sequential),
        n_restrictions=q,
        trimming=trimming,
        nobs=nobs,
    )


def cusum(
    endog: npt.ArrayLike,
    exog: npt.ArrayLike | None = None,
    *,
    trend: str = "c",
    squares: bool = False,
    n_draws: int = 5000,
    seed: int | np.random.Generator | None = None,
) -> BreakTest:
    """CUSUM and CUSUM-of-squares tests of coefficient constancy.

    The recursive residuals are the one-step forecast errors of the
    regression re-estimated as each observation arrives, standardized so
    that under constant coefficients they are i.i.d. Their cumulative
    sum, scaled by its standard deviation, is compared with the linear
    boundary of Brown, Durbin and Evans; the statistic is the largest
    ratio of the path to the boundary shape, so it reads against the
    tabulated multipliers ``1.143``, ``0.948``, ``0.850``. The squared
    version cumulates the squared residuals against their expected share
    and is the sharper test for a change in variance; its finite-sample
    boundary is simulated exactly, since under the null the path is a
    function of i.i.d. Gaussian draws alone.

    Args:
        endog: The target.
        exog: Regressors, or ``None``.
        trend: Deterministic terms.
        squares: Whether to run the CUSUM-of-squares test.
        n_draws: Simulated paths for the CUSUM-of-squares boundary.
        seed: Seed or generator.

    Returns:
        The :class:`BreakTest` with the path and its boundaries; the
        CUSUM test carries no p-value, the squared one does.

    Raises:
        SpecificationError: If the trend or counts are unusable.
        DimensionError: If the regressors do not align.
        NumericalError: If the design is rank deficient.
    """
    if n_draws < 500:
        raise SpecificationError(f"n_draws must be at least 500; got {n_draws}.")
    y, design = _validate_regression(endog, exog, trend)
    nobs, q = design.shape
    w = _recursive_residuals(y, design)
    count = w.shape[0]
    steps = np.arange(1, count + 1, dtype=np.float64)
    if squares:
        path = np.cumsum(w**2) / float(w @ w)
        deviation = np.abs(path - steps / count)
        statistic = float(deviation.max())
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        quantiles, draws = _cusum_squares_quantiles(
            count, (0.01, 0.05, 0.10), n_draws=n_draws, rng=rng
        )
        cv = {
            label: quantiles[level]
            for label, level in zip(_CRITICAL_LEVELS, _CRITICAL_ALPHAS, strict=True)
        }
        pvalue = float(np.mean(draws >= statistic))
        bounds = np.vstack([steps / count + cv[label] for label in _CRITICAL_LEVELS])
        crossing = np.flatnonzero(deviation > cv["5%"])
        name, null = "CUSUM of squares", "constant coefficients and variance"
    else:
        sigma = float(np.sqrt(np.sum((w - w.mean()) ** 2) / (count - 1)))
        path = np.cumsum(w) / sigma
        shape = np.sqrt(count) * (1.0 + 2.0 * steps / count)
        ratio = np.abs(path) / shape
        statistic = float(ratio.max())
        cv = dict(_CUSUM_BOUNDARY)
        pvalue = None
        bounds = np.vstack([shape * cv[label] for label in _CRITICAL_LEVELS])
        crossing = np.flatnonzero(ratio > cv["5%"])
        name, null = "CUSUM", "constant coefficients"
    break_index = int(q + crossing[0]) if crossing.size else None
    return BreakTest(
        name=name,
        statistic=statistic,
        pvalue=pvalue,
        critical_values=cv,
        null=null,
        break_index=break_index,
        n_restrictions=q,
        trimming=None,
        nobs=nobs,
        path=np.asarray(path, dtype=np.float64),
        bounds=np.asarray(bounds, dtype=np.float64),
    )
