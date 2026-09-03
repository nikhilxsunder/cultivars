# filepath: /src/cultivars/multivariate/nonlinear/threshold.py
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

"""The threshold vector autoregression: two systems, one observable switch.

A TVAR is two complete VARs -- coefficients, deterministics, *and* innovation
covariance -- glued along a threshold on an observable variable: below the
split one system generates the data, above it the other. The regime is
computed, never inferred, which is what separates this family from the
Markov-switching one and what makes estimation least squares rather than EM.
The canonical macro-finance use is exactly this shape: credit conditions
tighten past some level and the whole transmission mechanism changes, not just
one coefficient (Balke 2000).

The estimator is honest about the geometry of the problem. The likelihood is a
step function of the threshold -- piecewise constant, nowhere differentiable in
it -- so no gradient method applies, and the search is an exhaustive grid over
trimmed quantiles of the transition variable with regime-wise multivariate
least squares at each candidate. The criterion is the total Gaussian
log-likelihood with a separate covariance per regime; a sum of squares would
weight the equations by their innovation variances and let the noisiest series
choose the split, and holding the covariance fixed across regimes would assume
away the volatility shift that is usually half the finding.

Everything downstream must name a regime, and the result refuses to pretend
otherwise: there is no single companion matrix, no single moving-average
representation, and no chi-squared linearity test -- the threshold is not
identified under the null (Davies), so :meth:`TVARResult.likelihood_ratio_test`
raises rather than returning a well-formed and wrong p-value. Impulse
responses are regime-conditional linearizations, stated as such.

References:
    Tsay, R. S. (1998). Testing and modeling multivariate threshold models.
        *Journal of the American Statistical Association*, 93(443), 1188-1202.
    Balke, N. S. (2000). Credit and economic activity: Credit regimes and
        nonlinear propagation of shocks. *Review of Economics and Statistics*,
        82(2), 344-349.
    Hansen, B. E. (1996). Inference when a nuisance parameter is not
        identified under the null hypothesis. *Econometrica*, 64(2), 413-430.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import Regime, SummaryTable, validate_choice
from ..._internals import (
    _ThresholdVectorAutoRegressionModel,
    _VectorObservedRegimeResult,
    _VectorThresholdFit,
)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class TVARResult(_VectorObservedRegimeResult):
    """A fitted two-regime threshold vector autoregression.

    Attributes:
        self_exciting: Whether the transition variable is a column of the
            system itself.
        searched_delay: Whether the delay was chosen by the grid search
            rather than fixed by the caller.
        n_lower: Observations assigned to the lower regime.
        n_upper: Observations assigned to the upper regime.
        lower_sigma_u: Lower-regime innovation covariance, dof-corrected.
        upper_sigma_u: Upper-regime innovation covariance, dof-corrected.
    """

    self_exciting: bool
    searched_delay: bool
    n_lower: int
    n_upper: int
    lower_sigma_u: npt.NDArray[np.float64] = field(repr=False)
    upper_sigma_u: npt.NDArray[np.float64] = field(repr=False)

    @classmethod
    def _from_fit(
        cls, fit: _VectorThresholdFit, model: _ThresholdVectorAutoRegressionModel[TVARResult]
    ) -> TVARResult:
        """Assemble the public result from a raw fit and its specification."""
        return cls(
            endog=model.endog,
            names=model.names,
            order=model.order,
            trend=model.trend,
            delay=fit.delay,
            threshold=fit.threshold,
            threshold_name=model.transition_name,
            threshold_values=fit.threshold_values,
            transition_series=model.transition_series,
            lower_coefficients=fit.lower_coefficients,
            upper_coefficients=fit.upper_coefficients,
            lower_deterministic=fit.lower_deterministic,
            upper_deterministic=fit.upper_deterministic,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            self_exciting=model.self_exciting,
            searched_delay=model.delay is None,
            n_lower=fit.n_lower,
            n_upper=fit.n_upper,
            lower_sigma_u=fit.lower_sigma_u,
            upper_sigma_u=fit.upper_sigma_u,
        )

    @property
    def regime_weight(self) -> npt.NDArray[np.float64]:
        """Indicator of the upper regime: ``1.0`` above the threshold, else ``0.0``.

        Strict on the upper side, matching the estimator's ``z <= r``
        assignment to the lower regime, so a transition value landing exactly
        on the threshold is counted here as it was when the coefficients were
        solved for.
        """
        return (self.threshold_values > self.threshold).astype(np.float64)

    def _weight_at(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The upper-regime indicator at arbitrary transition values."""
        return (values > self.threshold).astype(np.float64)

    def _regime_sigma(self, regime: Regime) -> npt.NDArray[np.float64]:
        """The named regime's own innovation covariance."""
        choice = validate_choice(regime, Regime, "regime")
        return self.lower_sigma_u if choice == "lower" else self.upper_sigma_u

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table."""
        return f"TVAR({self.order}, d={self.delay})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        split = (
            f"Regime split: {self.n_lower} below, {self.n_upper} above "
            f"({100.0 * self.upper_fraction:.1f}% upper). The innovation "
            "covariance switches with the regime; lower_sigma_u and "
            "upper_sigma_u carry the two estimates."
        )
        notes = [self._stationarity_note(), split]
        if self.searched_delay:
            notes.append(
                f"The delay was selected by the same grid search as the "
                f"threshold; d = {self.delay} maximized the total likelihood."
            )
        if not self.self_exciting:
            notes.append("The transition variable is external, not a column of the system.")
        notes.append(
            "The threshold is not identified under linearity, so information "
            "criteria rank specifications but do not test for a threshold."
        )
        notes.append(self._regime_conditional_note())
        notes.append("Standard errors are not yet available for this estimator.")
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=self._summary_metadata(),
            columns=("equation", "lower own L1", "upper own L1"),
            rows=self._own_lag_rows(),
            notes=tuple(notes),
        )


class TVAR(_ThresholdVectorAutoRegressionModel[TVARResult]):
    """Two-regime threshold vector autoregression, Tsay (1998).

    The regime is decided by an observable -- a named column's own lag, or an
    external series you supply -- against a threshold estimated by grid
    search. Each regime gets its own coefficient stack, deterministic terms,
    and innovation covariance.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order within each regime.
        transition_variable: A variable name from ``names`` (self-exciting:
            the regime is driven by that variable's lag) or an aligned
            external series.
        delay: Threshold delay. ``None`` -- allowed only when self-exciting
            -- searches ``1..order`` jointly with the threshold; with an
            external driver the lag is a modelling choice with economic
            content, so it must be stated.
        trim: Fraction trimmed from each tail of the quantile grid, so that
            neither regime is estimated from a handful of extreme points.
        n_grid: Candidate thresholds per delay.
        trend: Deterministic terms per regime.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((300, 2))
        >>> for t in range(1, 300):
        ...     a = 0.6 if y[t - 1, 0] <= 0.0 else -0.3
        ...     y[t] = a * y[t - 1] + rng.standard_normal(2)
        >>> res = TVAR(y, order=1, transition_variable="y1", delay=1).fit()
        >>> res.n_lower + res.n_upper == res.nobs
        True
        >>> res.irf(4, regime="lower").shape
        (5, 2, 2)
    """

    __slots__ = ()

    def fit(self) -> TVARResult:
        """Estimate the threshold, the delay, and both regimes by grid search."""
        return TVARResult._from_fit(self._fit_regimes(), self)
