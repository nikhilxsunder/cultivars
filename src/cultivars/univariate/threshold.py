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
r"""Threshold autoregressions: SETAR and TAR.

Two linear autoregressions selected by an indicator on an observed
transition variable,

.. math::

   y_t = \bigl(1 - I[z_{t-d} > r]\bigr)\,\phi_L' x_t
   + I[z_{t-d} > r]\,\phi_U' x_t + \varepsilon_t,
   \qquad x_t = (1, y_{t-1}, \ldots, y_{t-p})',

with :math:`z_t = y_t` for the self-exciting :class:`SETAR` and an
external series for :class:`TAR`. The regime is a deterministic
function of something you can see -- a lag of the series, or a variable
you supply -- which is what separates this family from
:mod:`~cultivars.univariate.regime_switching`, where the regime is
latent and only its filtered probability is ever recovered. Here the
regime weight is computed, not inferred, so estimation is least squares
rather than EM and the results carry the transition variable itself as
an aligned series. Because the weight is an indicator, the sum of
squares is a step function of the threshold -- piecewise constant,
nowhere differentiable in :math:`r` -- so no gradient method can find
its minimum, and the estimator is an exhaustive search,

.. math::

   (\hat r, \hat d) = \arg\min_{r \in \mathcal{G}_d,\; d}
   \Bigl[\min_{\phi_L} \sum_{z_{t-d} \le r} (y_t - \phi_L' x_t)^2
   + \min_{\phi_U} \sum_{z_{t-d} > r} (y_t - \phi_U' x_t)^2\Bigr],

over a grid :math:`\mathcal{G}_d` of ``n_grid`` quantiles of the
transition variable with ``trim`` of each tail removed, regime-wise
least squares at every candidate, and -- when the delay is left open --
every :math:`d` in ``1..order`` on one common effective sample so the
sums of squares are comparable across delays. The threshold estimate is
super-consistent (Chan, 1993), which is why the coefficients can be
read as ordinary least squares conditional on it.

Two commitments shape the surface. First, stationarity is reported per
regime, never as a single verdict. A two-regime autoregression has no
single companion polynomial, and regime-wise stationarity is sufficient
but not necessary for the process to be globally stationary: a model
with an explosive inner regime and a contracting outer one is well
behaved, since excursions are pulled back. The result exposes
``lower_stability`` and ``upper_stability`` and refuses to collapse
them. Second, the likelihood-ratio test is blocked outright. Under
linearity the threshold and the delay do not appear in the likelihood,
so the chi-squared limit does not hold for any test that involves them
(the Davies problem); ``likelihood_ratio_test`` raises rather than hand
back a well-formed and wrong p-value, the criteria rank specifications
without testing them, and the tests that do respect the problem are
:func:`~cultivars.diagnostics.nonlinearity.tsay` and the bootstrap
sup-F :func:`~cultivars.diagnostics.nonlinearity.hansen_threshold`.

Layout. :class:`SETAR` and :class:`TAR` differ only in the transition
variable and delegate to ``_ThresholdModel`` in ``_internals``, which
validates ``order``, ``delay`` and ``n_grid`` through ``validate_order``,
``trim`` through ``validate_open_interval`` and an external variable
through ``validate_aligned``; its ``_fit_family`` lays out the
regressors with ``deterministic_columns`` and ``lag_matrix``, runs the
grid with ``ols`` from ``_core`` on each split, and packs a
``_ThresholdFit``. :class:`SETARResult` extends
``_ObservedRegimeResult``, which supplies the per-regime stability
verdicts, the refusal of the likelihood-ratio test, the aligned series
and ``simulate`` through ``_simulate_two_regime``; the result aligns the
transition variable with ``trailing_lag`` from ``_core`` and emits its
mean as a :class:`~cultivars.state_space.nonlinear.NonlinearSSM` through
``NonlinearSSM._lag_stack_state_space``. The smooth-weight counterpart
is :mod:`~cultivars.univariate.smooth_transition`, of which the hard
switch is the infinite-speed limit; the neural counterpart is
:class:`~cultivars.univariate.mean_function.TARNN`; the vector form is
:mod:`cultivars.multivariate.regime_switching`.

References:
    Tong, H. (1990). *Non-linear Time Series: A Dynamical System
    Approach*. Oxford University Press.

    Tsay, R. S. (1989). Testing and modeling threshold autoregressive
    processes. *Journal of the American Statistical Association*,
    84(405), 231-240.

    Chan, K. S. (1993). Consistency and limiting distribution of the
    least squares estimator of a threshold autoregressive model. *Annals
    of Statistics*, 21(1), 520-533.

    Hansen, B. E. (1997). Inference in TAR models. *Studies in Nonlinear
    Dynamics and Econometrics*, 2(1), 1-14.

    Davies, R. B. (1987). Hypothesis testing when a nuisance parameter
    is present only under the alternative. *Biometrika*, 74(1), 33-43.

Example:
    A hard switch recovered by the grid, ranked on BIC against the
    linear fit, with the likelihood-ratio test refused as it should be:

    >>> import numpy as np
    >>> from cultivars.univariate.autoregression import AR
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros(400)
    >>> for t in range(1, 400):
    ...     phi = 0.6 if y[t - 1] <= 0.0 else -0.4
    ...     y[t] = phi * y[t - 1] + rng.standard_normal()
    >>> hard = SETAR(y, order=1, delay=1).fit()
    >>> linear = AR(y, order=1).fit()
    >>> bool(hard.information_criteria.bic < linear.information_criteria.bic - 20)
    True
    >>> hard.is_regimewise_stationary, bool(abs(hard.threshold) < 0.3)
    (True, True)
    >>> hard.likelihood_ratio_test(linear)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    cultivars.exceptions.SpecificationError: ...
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..engine._core import _SIMULATION_BURN, SummaryTable, trailing_lag
from ..engine._internals import (
    _ObservedRegimeResult,
    _ThresholdFit,
    _ThresholdModel,
)
from ..exceptions import SpecificationError
from ..state_space.nonlinear import NonlinearSSM

__all__ = ["SETAR", "TAR", "SETARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SETARResult(_ObservedRegimeResult):
    r"""A fitted hard-threshold autoregression.

    Two linear autoregressions selected by an indicator on a lagged
    transition variable,

    .. math::

       y_t = \begin{cases}
       \phi_L' x_t + \varepsilon_t & z_{t-d} \le r \\
       \phi_U' x_t + \varepsilon_t & z_{t-d} > r
       \end{cases},
       \qquad x_t = (1, y_{t-1}, \ldots, y_{t-p})',
       \quad \varepsilon_t \sim (0, \sigma^2),

    with :math:`z_t = y_t` for the self-exciting :class:`SETAR` and an
    external series for :class:`TAR`. Conditional on :math:`(r, d)` the
    model is two least-squares problems, so the estimator is a grid
    search: every candidate threshold on a trimmed quantile grid of the
    transition variable (and every candidate delay, when the caller left
    it open) is scored by the total sum of squares and the minimizer is
    kept. The record stores the split point, the regime assignment and
    the two coefficient blocks; the inherited fields, the per-regime
    stability verdicts, the information criteria and the refusal of the
    likelihood-ratio test come from ``_ObservedRegimeResult``.

    Note:
        The assignment is ``z <= r`` to the lower regime, and every
        reader of the fit -- ``regime_weight``, ``_weight_at``,
        ``simulate`` and ``nonlinear_state_space`` -- uses the same
        strict comparison on the upper side, so a transition value
        landing exactly on the threshold is counted as it was when the
        coefficients were solved for. ``n_params`` counts the two
        coefficient blocks and the threshold, not the variance, so the
        criteria are comparable with :class:`~cultivars.univariate.smooth_transition.STARResult`
        but sit one parameter below the linear families' convention.
        ``searched_delay`` is ``True`` only when more than one delay was
        on the grid: ``SETAR(y, order=1)`` has nothing to search.

    Attributes:
        self_exciting: Whether the transition variable is a lag of the
            series itself (:class:`SETAR`) or an external variable
            (:class:`TAR`).
        n_lower: Observations assigned to the lower regime.
        n_upper: Observations assigned to the upper regime.
        searched_delay: Whether the delay was chosen by the grid search
            rather than fixed by the caller.

    See Also:
        * :class:`SETAR` -- the self-exciting model that produces this
          record.
        * :class:`TAR` -- the externally driven model that produces this
          record.
        * :class:`~cultivars.univariate.smooth_transition.STARResult` --
          the same two regimes blended by a smooth weight.
        * :class:`~cultivars.univariate.regime_switching.MarkovSwitchingARResult`
          -- regimes driven by a latent chain rather than an observed
          variable.

    References:
        Tong, H. (1990). *Non-linear Time Series: A Dynamical System
        Approach*. Oxford University Press, ch. 3.

        Hansen, B. E. (1997). Inference in TAR models. *Studies in
        Nonlinear Dynamics and Econometrics*, 2(1), 1-14.

    Example:
        A two-regime AR(1) with a switch at zero, recovered with its
        split:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(400)
        >>> for t in range(1, 400):
        ...     phi = 0.6 if y[t - 1] <= 0.0 else -0.4
        ...     y[t] = phi * y[t - 1] + rng.standard_normal()
        >>> res = SETAR(y, order=1, delay=1).fit()
        >>> res.self_exciting, res.searched_delay, res.n_lower, res.n_upper, res.nobs
        (True, False, 288, 111, 399)
        >>> bool(abs(res.threshold) < 0.3), bool(abs(res.lower_params[1] - 0.6) < 0.1)
        (True, True)
        >>> bool(abs(res.upper_params[1] + 0.4) < 0.25), res.is_regimewise_stationary
        (True, True)
        >>> res.n_params, round(res.sigma2, 2)
        (5, 0.99)
    """

    self_exciting: bool
    """Whether the transition variable is the series' own lag (``SETAR``) or external."""
    n_lower: int
    """Observations assigned to the lower regime (``z <= r``)."""
    n_upper: int
    """Observations assigned to the upper regime (``z > r``)."""
    searched_delay: bool
    """Whether the delay was selected by the grid rather than fixed by the caller."""

    @classmethod
    def _from_fit(cls, fit: _ThresholdFit, model: _ThresholdModel[SETARResult]) -> SETARResult:
        """Assemble the public result from a raw fit and its specification.

        Rebuilds the aligned transition variable with ``trailing_lag`` --
        the series itself or the model's external variable, lagged by the
        selected delay and cut to the effective sample -- so the record
        can reproduce the regime assignment without the model.

        Args:
            fit: The packed estimate from ``_fit_family``.
            model: The specification the fit was produced for.

        Returns:
            The public result record.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros(300)
            >>> for t in range(1, 300):
            ...     y[t] = (0.6 if y[t - 1] <= 0.0 else -0.4) * y[t - 1] + rng.standard_normal()
            >>> model = SETAR(y, order=1, delay=1)
            >>> res = SETARResult._from_fit(model._fit_family(), model)
            >>> res.threshold_values.shape, res.nobs, res.searched_delay
            ((299,), 299, False)
        """
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
        when the coefficients were solved for. Its sum is ``n_upper`` and
        its mean is ``upper_fraction``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros(300)
            >>> for t in range(1, 300):
            ...     y[t] = (0.6 if y[t - 1] <= 0.0 else -0.4) * y[t - 1] + rng.standard_normal()
            >>> res = SETAR(y, order=1, delay=1).fit()
            >>> weight = res.regime_weight
            >>> weight.shape, int(weight.sum()) == res.n_upper, set(np.unique(weight)) == {0.0, 1.0}
            ((299,), True, True)
        """
        return (self.threshold_values > self.threshold).astype(np.float64)

    def nonlinear_state_space(self, measurement_variance: float) -> NonlinearSSM:
        """The fitted threshold mean as a noisily observed state space.

        The errors-in-variables reading of the fit: the latent state is
        the lag stack of the *true* series, the transition applies the
        estimated regime-wise autoregression (hard switch at the fitted
        threshold, ``z <= r`` to the lower regime as in estimation), and
        the observation reads the current level through Gaussian
        measurement error of the stated variance. That variance is
        required and strictly positive because with none the model is
        observation-driven and filtering it is vacuous. The particle
        filter is the natural reader -- the hard switch makes the
        transition non-smooth, so the extended filter's Jacobians are
        untrustworthy near the threshold; the initial state is diffuse at
        the sample mean and variance.

        Args:
            measurement_variance: Observation noise variance the emitted
                system is read through, strictly positive.

        Returns:
            The :class:`NonlinearSSM` with ``max(order, delay)`` states.

        Raises:
            SpecificationError: If the transition variable is external
                (a TAR fit) -- the emitted transition can depend only on
                the state, so only self-exciting fits have a closed
                state-space form -- or if the variance is not strictly
                positive.

        Example:
            The fitted mean read through measurement noise by the
            particle filter, which recovers the latent series better than
            the noisy observations do:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros(400)
            >>> for t in range(1, 400):
            ...     y[t] = (0.6 if y[t - 1] <= 0.0 else -0.4) * y[t - 1] + rng.standard_normal()
            >>> res = SETAR(y, order=1, delay=1).fit()
            >>> system = res.nonlinear_state_space(0.5)
            >>> system.k_states, system.k_endog
            (1, 1)
            >>> noisy = y + np.sqrt(0.5) * rng.standard_normal(400)
            >>> filtered = system.particle_filter(noisy, n_particles=500, seed=0)
            >>> latent = filtered.filtered_state[:, 0]
            >>> bool(np.mean((latent - y) ** 2) < np.mean((noisy - y) ** 2))
            True
            >>> res.nonlinear_state_space(0.0)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: measurement_variance must be strictly ...
        """
        if not self.self_exciting:
            raise SpecificationError(
                "the transition variable is an external series, so the "
                "regime at each step depends on something outside the "
                "state and the transition map has no closed form; only "
                "self-exciting fits emit a state space."
            )
        lower = np.asarray(self.lower_params, dtype=np.float64)
        upper = np.asarray(self.upper_params, dtype=np.float64)
        order, delay, threshold = self.order, self.delay, self.threshold
        depth = max(order, delay)

        def mean_map(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            base = np.column_stack([np.ones(states.shape[0]), states[:, :order]])
            return np.asarray(
                np.where(states[:, delay - 1] > threshold, base @ upper, base @ lower),
                dtype=np.float64,
            )

        return NonlinearSSM._lag_stack_state_space(
            mean_map,
            order=depth,
            sigma2=self.sigma2,
            measurement_variance=float(measurement_variance),
            center=float(np.mean(self.endog)),
            spread=float(np.var(self.endog)),
        )

    def _weight_at(self, z: float) -> float:
        """Indicator of the upper regime at one transition value.

        The scalar form of ``regime_weight`` that ``simulate`` consults
        at each step; strict on the upper side like the estimator.

        Args:
            z: The transition variable.

        Returns:
            ``1.0`` if ``z`` exceeds the threshold, else ``0.0``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros(300)
            >>> for t in range(1, 300):
            ...     y[t] = (0.6 if y[t - 1] <= 0.0 else -0.4) * y[t - 1] + rng.standard_normal()
            >>> res = SETAR(y, order=1, delay=1).fit()
            >>> res._weight_at(res.threshold), res._weight_at(res.threshold + 1e-9)
            (0.0, 1.0)
        """
        return 1.0 if z > self.threshold else 0.0

    def simulate(
        self,
        n: int = 200,
        *,
        seed: int | np.random.Generator | None = None,
        burn: int = _SIMULATION_BURN,
    ) -> npt.NDArray[np.float64]:
        """A fresh sample path at the fitted parameters.

        The regime at each step is read from the simulated series' own
        ``delay``-th lag with the estimator's ``z <= r`` assignment, so
        only a self-exciting fit can simulate; a TAR fit's transition
        variable is an external series whose future is not part of the
        model. Innovations are Gaussian at the fitted ``sigma2``.

        Args:
            n: Observations kept.
            seed: Seed or generator.
            burn: Periods discarded from the start.

        Returns:
            An array of shape ``(n,)``.

        Raises:
            SpecificationError: If the counts are not usable or the fit
                is not self-exciting.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros(400)
            >>> for t in range(1, 400):
            ...     y[t] = (0.6 if y[t - 1] <= 0.0 else -0.4) * y[t - 1] + rng.standard_normal()
            >>> res = SETAR(y, order=1, delay=1).fit()
            >>> path = res.simulate(300, seed=3)
            >>> path.shape, bool(np.array_equal(path, res.simulate(300, seed=3)))
            ((300,), True)
            >>> share_lower = np.mean(path <= res.threshold)
            >>> bool(abs(share_lower - res.n_lower / res.nobs) < 0.1)
            True
            >>> z = rng.standard_normal(400)
            >>> x = np.zeros(400)
            >>> for t in range(1, 400):
            ...     x[t] = (0.5 if z[t - 1] <= 0.0 else -0.5) * x[t - 1] + rng.standard_normal()
            >>> tar = TAR(x, order=1, delay=1, threshold_variable=z).fit()
            >>> tar.simulate(10)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: a TAR fit cannot simulate its own sample: ...
        """
        if not self.self_exciting:
            raise SpecificationError(
                "a TAR fit cannot simulate its own sample: the transition variable is an "
                "external series whose future is not part of the model."
            )
        return _ObservedRegimeResult.simulate(self, n, seed=seed, burn=burn)

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Returns:
            ``"SETAR(p, d=k)"`` or ``"TAR(p, d=k)"``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros(300)
            >>> for t in range(1, 300):
            ...     y[t] = (0.6 if y[t - 1] <= 0.0 else -0.4) * y[t - 1] + rng.standard_normal()
            >>> SETAR(y, order=2).fit()._comparison_label()
            'SETAR(2, d=1)'
        """
        family = "SETAR" if self.self_exciting else "TAR"
        return f"{family}({self.order}, d={self.delay})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        Returns:
            One row per coefficient in ``params`` order; the notes carry
            the regime-wise stability verdict, the regime split, whether
            the delay was searched, whether the transition variable is
            external, the identification caveat and the absence of
            standard errors.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros(300)
            >>> for t in range(1, 300):
            ...     y[t] = (0.6 if y[t - 1] <= 0.0 else -0.4) * y[t - 1] + rng.standard_normal()
            >>> table = SETAR(y, order=1, delay=1).fit()._summary_table()
            >>> table.title, len(table.notes)
            ('SETAR(1, d=1) Results', 4)
            >>> [row[0] for row in table.rows]
            ['lower.const', 'lower.ar.L1', 'upper.const', 'upper.ar.L1', 'threshold', 'sigma2']
            >>> len(SETAR(y, order=2).fit()._summary_table().notes)
            5
        """
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
    r"""Self-exciting threshold autoregression with two regimes.

    The regime is decided by the series' own lag :math:`y_{t-d}` against
    a threshold :math:`r`,

    .. math::

       y_t = \begin{cases}
       \phi_L' x_t + \varepsilon_t & y_{t-d} \le r \\
       \phi_U' x_t + \varepsilon_t & y_{t-d} > r
       \end{cases},
       \qquad x_t = (1, y_{t-1}, \ldots, y_{t-p})',

    with :math:`(r, d)` estimated by grid search and the two coefficient
    blocks by least squares at each candidate. The candidate thresholds
    are ``n_grid`` quantiles of the transition variable with ``trim`` of
    each tail removed, so neither regime can be estimated from a handful
    of extreme points; a split leaving either regime fewer than
    ``order + 2`` observations is skipped. Leaving ``delay`` as ``None``
    searches ``1..order`` jointly with the threshold, at the cost of
    ``order`` times the grid; each candidate is two closed-form
    regressions, so even the joint search is fast.

    Attributes:
        _endog: The observed series as a float array, shape ``(n,)``.
        _order: The autoregressive order within each regime.
        _delays: The candidate delays, one entry when fixed.
        _trim: The fraction trimmed from each tail of the grid.
        _n_grid: The number of candidate thresholds per delay.
        _threshold_variable: ``None``; the series is its own driver.

    Args:
        endog: The series.
        order: Autoregressive order within each regime, at least 1.
        delay: Threshold delay, at least 1; ``None`` searches ``1..order``.
        trim: Fraction trimmed from each tail of the quantile grid, in
            ``(0, 0.5)``, so that neither regime can be estimated from a
            handful of extreme points.
        n_grid: Candidate thresholds per delay, at least 1.

    Raises:
        SpecificationError: If ``order``, ``delay`` or ``n_grid`` is below
            1 or ``trim`` is outside ``(0, 0.5)``.
        DimensionError: If the series is not one-dimensional or shorter
            than ``2 (order + 2) + max(delay)``.

    See Also:
        * :class:`SETARResult` -- the record ``fit`` returns.
        * :class:`TAR` -- the same estimator with an external transition
          variable.
        * :class:`~cultivars.univariate.smooth_transition.LSTAR` -- the
          smooth-weight counterpart; a SETAR is its limit as the speed
          goes to infinity.
        * :class:`~cultivars.univariate.mean_function.TARNN` -- the same
          hard switch with a neural mean per regime.

    References:
        Tong, H., & Lim, K. S. (1980). Threshold autoregression, limit
        cycles and cyclical data. *Journal of the Royal Statistical
        Society B*, 42(3), 245-268.

        Hansen, B. E. (1997). Inference in TAR models. *Studies in
        Nonlinear Dynamics and Econometrics*, 2(1), 1-14.

    Example:
        A switch at zero with a fixed delay, then a delay-2 process whose
        delay the grid recovers:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(400)
        >>> for t in range(1, 400):
        ...     phi = 0.6 if y[t - 1] <= 0.0 else -0.4
        ...     y[t] = phi * y[t - 1] + rng.standard_normal()
        >>> res = SETAR(y, order=1, delay=1).fit()
        >>> res.n_lower + res.n_upper == res.nobs, bool(abs(res.threshold) < 0.3)
        (True, True)
        >>> rng = np.random.default_rng(1)
        >>> w = np.zeros(600)
        >>> for t in range(2, 600):
        ...     phi = 0.5 if w[t - 2] <= 0.0 else -0.5
        ...     w[t] = phi * w[t - 1] + rng.standard_normal()
        >>> searched = SETAR(w, order=2).fit()
        >>> searched.delay, searched.searched_delay, searched._comparison_label()
        (2, True, 'SETAR(2, d=2)')
        >>> SETAR(y, order=1, trim=0.5)
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: trim must lie in (0.0, 0.5); got 0.5.
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
        """Validate the specification and the data.

        Delegates to ``_ThresholdModel`` with no external transition
        variable, which makes the fit self-exciting.

        Args:
            endog: The series.
            order: Autoregressive order within each regime.
            delay: Threshold delay, or ``None`` to search ``1..order``.
            trim: Fraction trimmed from each tail of the quantile grid.
            n_grid: Candidate thresholds per delay.

        Example:
            >>> import numpy as np
            >>> model = SETAR(np.random.default_rng(0).standard_normal(100), order=2)
            >>> model.order, model.delay, model.self_exciting
            (2, None, True)
        """
        super().__init__(
            endog,
            order=order,
            delay=delay,
            trim=trim,
            n_grid=n_grid,
            threshold_variable=None,
        )

    def fit(self) -> SETARResult:
        """Estimate the threshold, the delay, and both regimes by grid search.

        Returns:
            The fitted two-regime autoregression.

        Raises:
            NumericalError: If no candidate split leaves both regimes
                with at least ``order + 2`` observations.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros(300)
            >>> for t in range(1, 300):
            ...     y[t] = (0.6 if y[t - 1] <= 0.0 else -0.4) * y[t - 1] + rng.standard_normal()
            >>> res = SETAR(y, order=1, delay=1).fit()
            >>> res.self_exciting, bool(res.lower_params[1] > res.upper_params[1])
            (True, True)
        """
        return SETARResult._from_fit(self._fit_family(), self)


class TAR(_ThresholdModel[SETARResult]):
    r"""Threshold autoregression driven by an external transition variable.

    Identical machinery to :class:`SETAR`, but the regime is decided by a
    variable you supply rather than by the series' own past,

    .. math::

       y_t = \begin{cases}
       \phi_L' x_t + \varepsilon_t & z_{t-d} \le r \\
       \phi_U' x_t + \varepsilon_t & z_{t-d} > r
       \end{cases},

    so the threshold is in the units of :math:`z` and the grid is built
    on its quantiles. The delay is fixed rather than searched: with an
    exogenous driver the lag is usually a modelling choice with economic
    content, not a nuisance parameter. Because the driver's future is
    not part of the model, the result refuses ``simulate`` and
    ``nonlinear_state_space``; everything else on
    :class:`SETARResult` applies.

    Attributes:
        _endog: The observed series as a float array, shape ``(n,)``.
        _order: The autoregressive order within each regime.
        _delays: The single fixed delay.
        _trim: The fraction trimmed from each tail of the grid.
        _n_grid: The number of candidate thresholds.
        _threshold_variable: The transition variable, shape ``(n,)``.

    Args:
        endog: The series.
        order: Autoregressive order within each regime, at least 1.
        threshold_variable: The transition variable, one-dimensional and
            aligned with ``endog``.
        delay: Delay applied to the transition variable, at least 1.
        trim: Fraction trimmed from each tail of the quantile grid, in
            ``(0, 0.5)``.
        n_grid: Candidate thresholds, at least 1.

    Raises:
        SpecificationError: If ``order``, ``delay`` or ``n_grid`` is below
            1 or ``trim`` is outside ``(0, 0.5)``.
        DimensionError: If either series is not one-dimensional, the
            transition variable is not the length of ``endog``, or the
            sample is shorter than ``2 (order + 2) + delay``.

    See Also:
        * :class:`SETARResult` -- the record ``fit`` returns.
        * :class:`SETAR` -- the self-exciting form, which can simulate
          and emit a state space.

    References:
        Tong, H. (1990). *Non-linear Time Series: A Dynamical System
        Approach*. Oxford University Press, ch. 3.

    Example:
        An autoregression whose persistence flips with the sign of an
        external driver:

        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> z = rng.standard_normal(400)
        >>> y = np.zeros(400)
        >>> for t in range(1, 400):
        ...     phi = 0.5 if z[t - 1] <= 0.0 else -0.5
        ...     y[t] = phi * y[t - 1] + rng.standard_normal()
        >>> res = TAR(y, order=1, threshold_variable=z).fit()
        >>> res.self_exciting, res.searched_delay, res._comparison_label()
        (False, False, 'TAR(1, d=1)')
        >>> bool(abs(res.threshold) < 0.3), bool(np.allclose(res.threshold_values, z[:-1]))
        (True, True)
        >>> bool(abs(res.lower_params[1] - 0.5) < 0.1), bool(abs(res.upper_params[1] + 0.5) < 0.1)
        (True, True)
        >>> TAR(y, order=1, threshold_variable=z[:300])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.DimensionError: threshold_variable must be one-dimensional with ...
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
        """Validate the specification and the data.

        Delegates to ``_ThresholdModel`` with the external transition
        variable, which fixes the delay and marks the fit as not
        self-exciting.

        Args:
            endog: The series.
            order: Autoregressive order within each regime.
            threshold_variable: The transition variable, aligned with
                ``endog``.
            delay: Delay applied to the transition variable.
            trim: Fraction trimmed from each tail of the quantile grid.
            n_grid: Candidate thresholds.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> driver = rng.standard_normal(100)
            >>> model = TAR(rng.standard_normal(100), order=1, threshold_variable=driver)
            >>> model.delay, model.self_exciting
            (1, False)
        """
        super().__init__(
            endog,
            order=order,
            delay=delay,
            trim=trim,
            n_grid=n_grid,
            threshold_variable=threshold_variable,
        )

    def fit(self) -> SETARResult:
        """Estimate the threshold and both regimes by grid search.

        Returns:
            The fitted two-regime autoregression, with
            ``self_exciting`` false.

        Raises:
            NumericalError: If no candidate split leaves both regimes
                with at least ``order + 2`` observations.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(1)
            >>> z = rng.standard_normal(300)
            >>> y = np.zeros(300)
            >>> for t in range(1, 300):
            ...     y[t] = (0.5 if z[t - 1] <= 0.0 else -0.5) * y[t - 1] + rng.standard_normal()
            >>> res = TAR(y, order=1, threshold_variable=z).fit()
            >>> res.self_exciting, res.n_lower + res.n_upper == res.nobs
            (False, True)
        """
        return SETARResult._from_fit(self._fit_family(), self)
