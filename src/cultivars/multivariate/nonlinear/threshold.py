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
r"""The threshold vector autoregression: two systems, one observable switch.

A TVAR is two complete VARs -- coefficients, deterministics, *and*
innovation covariance -- glued along a threshold on an observable
variable,

.. math::

   y_t = \begin{cases}
   c_L + \sum_{i=1}^{p} A_{L,i}\, y_{t-i} + u_t, & u_t \sim N(0, \Sigma_L),
   & z_t \le r, \\
   c_U + \sum_{i=1}^{p} A_{U,i}\, y_{t-i} + u_t, & u_t \sim N(0, \Sigma_U),
   & z_t > r,
   \end{cases}
   \qquad z_t = s_{t-d}:

below the split one system generates the data, above it the other. The
regime is computed, never inferred, which is what separates this family
from the Markov-switching one and what makes estimation least squares
rather than EM. The canonical macro-finance use is exactly this shape:
credit conditions tighten past some level and the whole transmission
mechanism changes, not just one coefficient (Balke, 2000).

Two commitments shape the surface. First, the estimator is honest about
the geometry of the problem. The likelihood is a step function of the
threshold -- piecewise constant, nowhere differentiable in it -- so no
gradient method applies, and the search is an exhaustive grid over trimmed
quantiles of the transition variable (and over the delays ``1..order``
when the delay is left to be chosen, which is allowed only for a
self-exciting model) with regime-wise multivariate least squares at each
candidate. The criterion is the total Gaussian log-likelihood with a
separate covariance per regime,
:math:`\ell_L(\hat\Sigma_L) + \ell_U(\hat\Sigma_U)`; a sum of squares
would weight the equations by their innovation variances and let the
noisiest series choose the split, and holding the covariance fixed across
regimes would assume away the volatility shift that is usually half the
finding. Second, everything downstream must name a regime, and the result
refuses to pretend otherwise: there is no single companion matrix, no
single moving-average representation, and no chi-squared linearity test
-- the threshold is not identified under the null (Davies; Hansen, 1996),
so :meth:`TVARResult.likelihood_ratio_test` raises rather than returning a
well-formed and wrong p-value, and information criteria rank threshold
specifications against one another without testing for a threshold.
Impulse responses, variance decompositions and moving-average matrices
are regime-conditional linearizations, stated as such; the forecast is
the skeleton iteration, exact at one step and an approximation beyond,
because the multi-step conditional mean of a nonlinear model averages the
switch over future shocks. Standard errors are not yet offered.

Layout. :class:`TVAR` validates on
``_ThresholdVectorAutoRegressionModel`` in ``_internals``, which takes the
named-or-external transition variable, its delay or delay range and the
aligned design from ``_ObservedRegimeVectorModel``, adds the grid's
``trim`` and ``n_grid``, and in ``_fit_regimes`` scans every delay and
candidate threshold through ``_split_moments`` -- one regime's least
squares, residuals, covariance and log-likelihood -- skipping candidates
that leave a regime fewer than ``w + 1`` rows or a singular covariance,
and packs a ``_VectorThresholdFit`` at the winner. :class:`TVARResult`
extends ``_VectorObservedRegimeResult`` -- the two coefficient stacks, the
regime-conditional propagation surface, the forecast, regime-wise
stability and comparison by information criteria -- supplying the
indicator weight and a covariance per regime. The smooth relaxation is
:mod:`~cultivars.multivariate.nonlinear.smooth_transition`; the
nonparametric check on the switch's shape is
:mod:`~cultivars.multivariate.nonlinear.functional_coefficient`; regimes
inferred from a latent chain are in
:mod:`~cultivars.multivariate.regime_switching.markov_switching`; the
univariate model is :mod:`~cultivars.univariate.threshold`.

References:
    Tsay, R. S. (1998). Testing and modeling multivariate threshold models.
    *Journal of the American Statistical Association*, 93(443), 1188-1202.

    Balke, N. S. (2000). Credit and economic activity: Credit regimes and
    nonlinear propagation of shocks. *Review of Economics and Statistics*,
    82(2), 344-349.

    Hansen, B. E. (1996). Inference when a nuisance parameter is not
    identified under the null hypothesis. *Econometrica*, 64(2), 413-430.

    Davies, R. B. (1987). Hypothesis testing when a nuisance parameter is
    present only under the alternative. *Biometrika*, 74(1), 33-43.

Example:
    A credit-regime shape in miniature: when the first variable's lag is
    at or below zero the system is persistent and quiet, above it the
    system mean-reverts and its shocks are twice as volatile. The fit
    finds the split, both coefficient stacks, and the volatility shift:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((400, 2))
    >>> for t in range(1, 400):
    ...     tight = y[t - 1, 0] > 0.0
    ...     a, scale = (-0.3, 2.0) if tight else (0.6, 1.0)
    ...     y[t] = a * y[t - 1] + scale * rng.standard_normal(2)
    >>> res = TVAR(y, order=1, transition_variable="y1", delay=1).fit()
    >>> bool(abs(res.threshold) < 0.4), res.n_lower > res.n_upper
    (True, True)
    >>> lower, upper = np.diag(res.lower_coefficients[0]), np.diag(res.upper_coefficients[0])
    >>> bool(np.all(lower > 0.4)), bool(np.all(upper < 0.0))
    (True, True)
    >>> ratio = np.diag(res.upper_sigma_u) / np.diag(res.lower_sigma_u)
    >>> bool(np.all(ratio > 2.5))
    True
    >>> res.likelihood_ratio_test(res)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    cultivars.exceptions.SpecificationError: a chi-squared likelihood-ratio test is not valid ...
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ...engine._core import Regime, SummaryTable, validate_choice
from ...engine._internals import (
    _ThresholdVectorAutoRegressionModel,
    _VectorObservedRegimeResult,
    _VectorThresholdFit,
)

__all__ = ["TVAR", "TVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class TVARResult(_VectorObservedRegimeResult):
    r"""A fitted two-regime threshold vector autoregression.

    The estimate of

    .. math::

       y_t = \begin{cases}
       c_L + \sum_{i=1}^{p} A_{L,i}\, y_{t-i} + u_t, & u_t \sim N(0, \Sigma_L),
       & z_t \le r, \\
       c_U + \sum_{i=1}^{p} A_{U,i}\, y_{t-i} + u_t, & u_t \sim N(0, \Sigma_U),
       & z_t > r,
       \end{cases}
       \qquad z_t = s_{t-d},

    two complete systems -- coefficients, deterministics and innovation
    covariance -- glued along a threshold :math:`r` on the delayed
    transition variable, which is a column of the system itself
    (self-exciting) or an external series. The split is found by exhaustive
    grid search over trimmed quantiles of :math:`z`, with regime-wise
    multivariate least squares at each candidate and the total Gaussian
    log-likelihood under separate covariances as the criterion; when the
    delay is searched, the grid runs over ``1..order`` as well. The record
    extends the observed-regime surface -- the two coefficient stacks,
    regime-conditional ``irf``, ``fevd``, ``ma_representation`` and
    ``forecast``, the stability of each regime, and comparison by
    information criteria -- with the regime sizes and the two covariances.

    Note:
        Observations with :math:`z_t \le r` belong to the lower regime, so
        a transition value exactly at the threshold is lower; the threshold
        itself is a grid quantile of the sample, so it is always an
        observed value of :math:`z`. ``n_params`` counts both regimes'
        coefficients, both covariances and the threshold, but not a
        searched delay. The threshold is not identified under linearity,
        so information criteria rank specifications but do not test for a
        threshold, and :meth:`likelihood_ratio_test` refuses rather than
        return a chi-squared p-value that is not one. Standard errors are
        not yet available.

    Attributes:
        self_exciting: Whether the transition variable is a column of the
            system itself.
        searched_delay: Whether the delay was chosen by the grid search
            rather than fixed by the caller.
        n_lower: Observations assigned to the lower regime.
        n_upper: Observations assigned to the upper regime.
        lower_sigma_u: Lower-regime innovation covariance, dof-corrected.
        upper_sigma_u: Upper-regime innovation covariance, dof-corrected.

    See Also:
        * :class:`TVAR` -- the model that produces this record.
        * :class:`~cultivars.multivariate.nonlinear.smooth_transition.STVARResult`
          -- the smooth blend of the same two regimes, with one covariance.
        * :class:`~cultivars.univariate.threshold.SETARResult` -- the
          univariate counterpart.

    References:
        Tsay, R. S. (1998). Testing and modeling multivariate threshold
        models. *Journal of the American Statistical Association*, 93(443),
        1188-1202.

        Hansen, B. E. (1996). Inference when a nuisance parameter is not
        identified under the null hypothesis. *Econometrica*, 64(2),
        413-430.

    Example:
        A bivariate system that is persistent when ``y1``'s lag is at or
        below zero and mean-reverting above it; the fit places the
        threshold near zero, recovers both regimes, and reports the
        regime sizes and covariances:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((300, 2))
        >>> for t in range(1, 300):
        ...     a = 0.6 if y[t - 1, 0] <= 0.0 else -0.3
        ...     y[t] = a * y[t - 1] + rng.standard_normal(2)
        >>> res = TVAR(y, order=1, transition_variable="y1", delay=1).fit()
        >>> res.self_exciting, res.searched_delay, res.delay, res.threshold_name
        (True, False, 1, 'y1')
        >>> bool(abs(res.threshold) < 0.3), res.n_lower + res.n_upper == res.nobs, res.n_params
        (True, True, 19.0)
        >>> lower, upper = np.diag(res.lower_coefficients[0]), np.diag(res.upper_coefficients[0])
        >>> bool(np.all(np.abs(lower - 0.6) < 0.15)), bool(np.all(np.abs(upper + 0.3) < 0.2))
        (True, True)
        >>> res.lower_sigma_u.shape, bool(np.all(np.diag(res.lower_sigma_u) > 0.0))
        ((2, 2), True)
        >>> bool(np.isclose(res.regime_weight.mean(), res.n_upper / res.nobs))
        True
    """

    self_exciting: bool
    """``True`` when the transition variable is one of the system's own columns."""
    searched_delay: bool
    """``True`` when the delay was chosen by the grid search over ``1..order``."""
    n_lower: int
    """Effective-sample observations with ``z <= threshold``."""
    n_upper: int
    """Effective-sample observations with ``z > threshold``."""
    lower_sigma_u: npt.NDArray[np.float64] = field(repr=False)
    """``(k, k)`` lower-regime covariance, divided by ``n_lower - w``. Kept out of the repr."""
    upper_sigma_u: npt.NDArray[np.float64] = field(repr=False)
    """``(k, k)`` upper-regime covariance, divided by ``n_upper - w``. Kept out of the repr."""

    @classmethod
    def _from_fit(
        cls, fit: _VectorThresholdFit, model: _ThresholdVectorAutoRegressionModel[TVARResult]
    ) -> TVARResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The packed grid-search fit.
            model: The validated specification it was estimated on.

        Returns:
            The frozen :class:`TVARResult`.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> model = TVAR(y, order=2, transition_variable="y2")
            >>> res = TVARResult._from_fit(model._fit_regimes(), model)
            >>> res.searched_delay, res.delay in (1, 2), res.threshold_name
            (True, True, 'y2')
        """
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

        Returns:
            An array of length ``nobs`` with entries in ``{0.0, 1.0}``,
            aligned with ``resid``; its sum is ``n_upper``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> res = TVAR(y, order=1, transition_variable="y1", delay=1).fit()
            >>> weight = res.regime_weight
            >>> weight.shape, sorted(set(weight.tolist())), int(weight.sum()) == res.n_upper
            ((199,), [0.0, 1.0], True)
            >>> bool(np.all(weight == (res.threshold_values > res.threshold)))
            True
        """
        return (self.threshold_values > self.threshold).astype(np.float64)

    def _weight_at(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The upper-regime indicator at arbitrary transition values.

        Args:
            values: Transition-variable values, in the original units.

        Returns:
            ``1.0`` where a value exceeds the threshold, ``0.0`` otherwise.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> res = TVAR(y, order=1, transition_variable="y1", delay=1).fit()
            >>> r = res.threshold
            >>> res._weight_at(np.array([r - 1.0, r, r + 1e-9]))
            array([0., 0., 1.])
        """
        return (values > self.threshold).astype(np.float64)

    def _regime_sigma(self, regime: Regime) -> npt.NDArray[np.float64]:
        """The named regime's own innovation covariance.

        Args:
            regime: ``"lower"`` or ``"upper"``.

        Returns:
            ``lower_sigma_u`` or ``upper_sigma_u``.

        Raises:
            SpecificationError: If ``regime`` is neither name.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> res = TVAR(y, order=1, transition_variable="y1", delay=1).fit()
            >>> res._regime_sigma("upper") is res.upper_sigma_u
            True
            >>> res._regime_sigma("middle")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: regime must be one of ('lower', 'upper'); ...
        """
        choice = validate_choice(regime, Regime, "regime")
        return self.lower_sigma_u if choice == "lower" else self.upper_sigma_u

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> TVAR(y, order=2, transition_variable="y1", delay=2).fit()._comparison_label()
            'TVAR(2, d=2)'
        """
        return f"TVAR({self.order}, d={self.delay})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Returns:
            The observed-regime rows (own first lag in each regime) and
            metadata; the notes state regime-wise stationarity, the regime
            split and the switching covariance, the delay search when it
            ran, the external driver when there is one, the
            non-identification of the threshold under linearity, the
            regime-conditional reading of the propagation methods, and the
            absence of standard errors.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> fixed = TVAR(y, order=1, transition_variable="y1", delay=1).fit()._summary_table()
            >>> fixed.title, fixed.columns, len(fixed.rows), len(fixed.notes)
            ('TVAR(1, d=1) Results', ('equation', 'lower own L1', 'upper own L1'), 2, 5)
            >>> fixed.notes[1].startswith("Regime split:")
            True
            >>> searched = TVAR(y, order=2, transition_variable="y1").fit()._summary_table()
            >>> len(searched.notes), "selected by the same grid search" in searched.notes[2]
            (6, True)
        """
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
    r"""Two-regime threshold vector autoregression, after Tsay (1998).

    The regime is decided by an observable -- a named column's own lag, or
    an external series you supply -- against a threshold estimated by grid
    search. Each regime gets its own coefficient stack, deterministic terms,
    and innovation covariance. The likelihood is a step function of the
    threshold, piecewise constant and nowhere differentiable in it, so the
    search is exhaustive: candidate thresholds are the quantiles of the
    delayed transition variable between ``trim`` and ``1 - trim``, each
    regime at each candidate is one multivariate least-squares solve, and
    the candidate maximizing the total Gaussian log-likelihood
    :math:`\ell_L(\hat\Sigma_L) + \ell_U(\hat\Sigma_U)` wins. A separate
    covariance per regime is the multivariate replacement for total sum of
    squares: it weights each equation by its own noise rather than letting
    the noisiest series choose the split, and lets the volatility shift that
    is usually half the finding be part of the fit.

    Attributes:
        _endog: The observed panel, from the base model.
        _order: The autoregressive order ``p``, from the base model.
        _trend: The deterministic specification, from the base model.
        _names: The variable labels, from the base model.
        _delays: The candidate delays -- one when fixed, ``1..order`` when
            searched -- from the observed-regime base.
        _threshold_name: The transition variable's label, from the
            observed-regime base.
        _transition_series: The aligned transition series, from the
            observed-regime base.
        _trim: Fraction trimmed from each tail of the quantile grid.
        _n_grid: Candidate thresholds per delay.

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

    Raises:
        SpecificationError: If the specification is malformed, ``trim`` is
            outside ``(0, 0.5)``, ``n_grid`` is below one, or the delay is
            left to be searched with an external transition variable.
        DimensionError: If the sample cannot support the lags and delay.

    See Also:
        * :class:`TVARResult` -- the record ``fit`` returns.
        * :class:`~cultivars.multivariate.nonlinear.smooth_transition.STVAR`
          -- the smooth blend of the same two regimes.
        * :class:`~cultivars.multivariate.regime_switching.markov_switching.MarkovSwitchingVAR`
          -- regimes inferred from a latent chain rather than read off an
          observable.
        * :class:`~cultivars.univariate.threshold.SETAR` -- the univariate
          self-exciting model.

    References:
        Tsay, R. S. (1998). Testing and modeling multivariate threshold
        models. *Journal of the American Statistical Association*, 93(443),
        1188-1202.

        Balke, N. S. (2000). Credit and economic activity: Credit regimes
        and nonlinear propagation of shocks. *Review of Economics and
        Statistics*, 82(2), 344-349.

        Hansen, B. E. (1996). Inference when a nuisance parameter is not
        identified under the null hypothesis. *Econometrica*, 64(2),
        413-430.

    Example:
        >>> import numpy as np
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
        >>> searched = TVAR(y, order=2, transition_variable="y1").fit()
        >>> searched.searched_delay, searched.delay
        (True, 1)
        >>> external = TVAR(y, order=1, transition_variable=np.sin(np.arange(300) / 10), delay=1)
        >>> external.fit().self_exciting
        False
        >>> TVAR(y, order=1, transition_variable=np.sin(np.arange(300) / 10))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: the delay is searched only for a self-exciting ...
    """

    __slots__ = ()

    def fit(self) -> TVARResult:
        """Estimate the threshold, the delay, and both regimes by grid search.

        For each candidate delay the trimmed quantile grid of the delayed
        transition variable is scanned; candidates leaving either regime
        fewer than ``w + 1`` observations, or with a singular regime
        covariance, are skipped; the surviving candidate with the largest
        total log-likelihood fixes the delay, the threshold and both
        regimes. The covariances are divided by each regime's own degrees
        of freedom.

        Returns:
            The fitted :class:`TVARResult`.

        Raises:
            NumericalError: If no candidate split is admissible.

        Example:
            The hard split and its smooth relaxation rank through the
            shared comparison surface, and the hard split's two covariances
            can differ where the smooth model is forced to share one:

            >>> import numpy as np
            >>> from cultivars.multivariate.nonlinear.smooth_transition import STVAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((300, 2))
            >>> for t in range(1, 300):
            ...     a = 0.6 if y[t - 1, 0] <= 0.0 else -0.3
            ...     y[t] = a * y[t - 1] + rng.standard_normal(2)
            >>> hard = TVAR(y, order=1, transition_variable="y1", delay=1).fit()
            >>> smooth = STVAR(y, order=1, transition_variable="y1").fit()
            >>> hard.n_params, smooth.n_params
            (19.0, 17.0)
            >>> table = hard.compare(smooth, criterion="aic")
            >>> sorted(row[0] for row in table.rows)
            ['STVAR(1, logistic, d=1)', 'TVAR(1, d=1)']
            >>> hard.lower_sigma_u.shape == hard.upper_sigma_u.shape == (2, 2)
            True
        """
        return TVARResult._from_fit(self._fit_regimes(), self)
