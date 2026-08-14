# filepath: /src/cultivars/univariate/threshold.py
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

"""Observed-regime autoregressions: SETAR, TAR, LSTAR, and ESTAR.

All four models here are two-regime autoregressions whose regime is a
*deterministic function of something you can see* -- a lag of the series, or an
external variable you supply. That is what separates them from the
Markov-switching family, where the regime is latent and only its filtered
probability is ever recovered. Here the regime weight is computed, not
inferred, which is why estimation is least squares rather than EM and why the
results carry the transition variable itself as an aligned series.

The group splits on how abruptly the regime changes, and the split runs all the
way down to the estimator:

1. **Hard threshold** (:class:`SETAR`, :class:`TAR`). The weight is an
   indicator, so the sum of squares is a step function of the threshold --
   piecewise constant, nowhere differentiable in it. No gradient method can
   find the minimum, so the estimator is an exhaustive grid search over
   trimmed quantiles of the transition variable, with regime-wise OLS at each
   candidate. The delay may be searched jointly.
2. **Smooth transition** (:class:`LSTAR`, :class:`ESTAR`). The weight is a
   logistic or exponential function taking values in ``[0, 1]``, so the surface
   is smooth in the two transition parameters. Conditional on those, the model
   is linear, so the regime coefficients are concentrated out and only
   ``(gamma, c)`` is searched -- derivative-free and multi-start, because the
   surface is flat in ``gamma`` far from the data and multimodal in ``c``.

Two properties of the group are reported carefully rather than conveniently.

Stationarity is reported *per regime*, never as a single verdict. A two-regime
nonlinear autoregression has no single companion polynomial, and regime-wise
stationarity is sufficient but not necessary for the process to be globally
stationary: a threshold model with an explosive inner regime and a contracting
outer one is perfectly well behaved, since excursions are pulled back. The
results therefore expose :attr:`ObservedRegimeResult.lower_stability` and
:attr:`ObservedRegimeResult.upper_stability` and refuse to collapse them.

Likelihood-ratio tests are blocked outright. The threshold is not identified
under the null of linearity -- when the model is linear, ``c`` and ``d`` simply
do not appear in the likelihood -- so the usual chi-squared limit does not hold
for any test that involves them (the Davies problem). Since the inherited
:meth:`compare` and :meth:`likelihood_ratio_test` would otherwise be perfectly
happy to hand back a well-formed and wrong p-value,
:meth:`ObservedRegimeResult.likelihood_ratio_test` raises instead.

References:
    Tong, H. (1990). *Non-linear Time Series: A Dynamical System Approach*.
    Tsay, R. S. (1989). Testing and modeling threshold autoregressive
    processes. *Journal of the American Statistical Association*, 84(405).
    Terasvirta, T. (1994). Specification, estimation, and evaluation of smooth
    transition autoregressive models. *Journal of the American Statistical
    Association*, 89(425).
    van Dijk, D., Terasvirta, T. & Franses, P. H. (2002). Smooth transition
    autoregressive models -- a survey of recent developments. *Econometric
    Reviews*, 21(1).
    Davies, R. B. (1987). Hypothesis testing when a nuisance parameter is
    present only under the alternative. *Biometrika*, 74(1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import SummaryTable, trailing_lag
from .._internals import (
    _ObservedRegimeResult,
    _ThresholdFit,
    _ThresholdModel,
)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SETARResult(_ObservedRegimeResult):
    """A fitted hard-threshold autoregression.

    Attributes:
        self_exciting: Whether the transition variable is a lag of the series
            itself (:class:`SETAR`) or an external variable (:class:`TAR`).
        n_lower: Observations assigned to the lower regime.
        n_upper: Observations assigned to the upper regime.
        searched_delay: Whether the delay was chosen by the grid search rather
            than fixed by the caller.
    """

    self_exciting: bool
    n_lower: int
    n_upper: int
    searched_delay: bool

    @classmethod
    def _from_fit(cls, fit: _ThresholdFit, model: _ThresholdModel[SETARResult]) -> SETARResult:
        """Assemble the public result from a raw fit and its specification."""
        external = model._threshold_variable
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            order=model.order,
            delay=fit.delay,
            threshold=fit.threshold,
            threshold_values=trailing_lag(
                model.endog if external is None else external,
                delay=fit.delay,
                length=fit.nobs,
            ),
            lower_params=fit.lower_params,
            upper_params=fit.upper_params,
            sigma2=fit.sigma2,
            ssr=fit.ssr,
            self_exciting=model.self_exciting,
            n_lower=fit.n_lower,
            n_upper=fit.n_upper,
            searched_delay=model.delay is None,
        )

    @property
    def regime_weight(self) -> npt.NDArray[np.float64]:
        """Indicator of the upper regime: ``1.0`` above the threshold, else ``0.0``.

        The comparison is strict on the upper side, matching the estimator's
        ``z <= r`` assignment to the lower regime, so a transition value
        landing exactly on the threshold is counted the same way here as it was
        when the coefficients were solved for.
        """
        return (self.threshold_values > self.threshold).astype(np.float64)

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        family = "SETAR" if self.self_exciting else "TAR"
        return f"{family}({self.order}, d={self.delay})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        split = (
            f"Regime split: {self.n_lower} below, {self.n_upper} above "
            f"({100.0 * self.upper_fraction:.1f}% upper)."
        )
        notes = [self._stationarity_note(), split]
        if self.searched_delay:
            notes.append(
                f"The delay was selected by the same grid search as the threshold; "
                f"d = {self.delay} minimized the total sum of squares."
            )
        if not self.self_exciting:
            notes.append("The transition variable is external, not a lag of the series.")
        notes.append(
            "The threshold is not identified under linearity, so information criteria "
            "rank specifications but do not test for a threshold."
        )
        notes.append("Standard errors are not yet available for this estimator.")
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=self._summary_metadata(),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=tuple(notes),
        )


class SETAR(_ThresholdModel[SETARResult]):
    """Self-exciting threshold autoregression with two regimes.

    The regime is decided by ``y_{t-d}`` against a threshold, both estimated by
    grid search. Leaving ``delay`` as ``None`` searches ``1..order`` jointly
    with the threshold, at the cost of ``order`` times the grid.

    Args:
        endog: The series.
        order: Autoregressive order within each regime.
        delay: Threshold delay; ``None`` searches ``1..order``.
        trim: Fraction trimmed from each tail of the quantile grid, so that
            neither regime can be estimated from a handful of extreme points.
        n_grid: Candidate thresholds per delay.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(400)
        >>> for t in range(1, 400):
        ...     phi = 0.6 if y[t - 1] <= 0.0 else -0.4
        ...     y[t] = phi * y[t - 1] + rng.standard_normal()
        >>> res = SETAR(y, order=1, delay=1).fit()
        >>> res.n_lower + res.n_upper == res.nobs
        True
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        delay: int | None = None,
        trim: float = 0.15,
        n_grid: int = 300,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(
            endog,
            order=order,
            delay=delay,
            trim=trim,
            n_grid=n_grid,
            threshold_variable=None,
        )

    def fit(self) -> SETARResult:
        """Estimate the threshold, the delay, and both regimes by grid search."""
        return SETARResult._from_fit(self._fit_family(), self)


class TAR(_ThresholdModel[SETARResult]):
    """Threshold autoregression driven by an external transition variable.

    Identical machinery to :class:`SETAR`, but the regime is decided by a
    variable you supply rather than by the series' own past. The delay is fixed
    rather than searched: with an exogenous driver the lag is usually a
    modelling choice with economic content, not a nuisance parameter.

    Args:
        endog: The series.
        order: Autoregressive order within each regime.
        threshold_variable: The transition variable, aligned with ``endog``.
        delay: Delay applied to the transition variable.
        trim: Fraction trimmed from each tail of the quantile grid.
        n_grid: Candidate thresholds.
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        threshold_variable: npt.ArrayLike,
        delay: int = 1,
        trim: float = 0.15,
        n_grid: int = 300,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(
            endog,
            order=order,
            delay=delay,
            trim=trim,
            n_grid=n_grid,
            threshold_variable=threshold_variable,
        )

    def fit(self) -> SETARResult:
        """Estimate the threshold and both regimes by grid search."""
        return SETARResult._from_fit(self._fit_family(), self)
