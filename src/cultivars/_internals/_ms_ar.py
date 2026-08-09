# filepath: /src/cultivars/_internals/_ms_ar.py
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

"""Markov-switching autoregression estimated by expectation-maximization.

The E-step runs the Hamilton filter and Kim smoother to obtain regime
responsibilities; the M-step is closed-form given those responsibilities --
a responsibility-weighted GLS solve for the coefficients, expected transition
counts for the transition matrix, and a weighted residual variance.

The likelihood is badly multimodal, so estimation screens many random starts
for a few iterations each and refines only the winner. The starting values
matter more than the optimizer:

- Regime intercepts are seeded from actual observations (k-means++ style)
  rather than jittered around the grand mean, which would place every regime
  in the same place.
- Variances come from hard-assigning observations to the nearest intercept
  and taking within-cluster variances. Seeding on the order of the *total*
  variance lets one wide regime absorb the sample and strands EM there.
- AR blocks start at zero. The marginal autocorrelation of a switching series
  reflects regime persistence, not within-regime dynamics, so seeding from it
  is actively misleading.

Regimes are permuted into ascending intercept order before the fit is
returned, which fixes the label-switching non-identification so that repeated
fits are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core._defaults import (
    _DEFAULT_MAX_ITER,
    _DEFAULT_STARTS,
    _DEFAULT_TOL,
    _LOG_2PI,
)
from .._core._linalg import ergodic_distribution
from .._core._validators import validate_order
from ..exceptions import NumericalError, SpecificationError
from ._models import _UnivariateModel
from ._regime_switching import hamilton_filter, kim_smoother


@dataclass(frozen=True, slots=True)
class _Layout:
    """Column bookkeeping for the responsibility-weighted regression.

    Switching and non-switching blocks share one design matrix; this object
    is the single source of truth for which column belongs to which regime,
    so the M-step never has to recompute offsets inline.
    """

    n_regimes: int
    order: int
    switching_mean: bool
    switching_ar: bool

    @property
    def n_intercept(self) -> int:
        """Number of intercept columns."""
        return self.n_regimes if self.switching_mean else 1

    @property
    def n_ar(self) -> int:
        """Number of autoregressive columns."""
        return self.n_regimes * self.order if self.switching_ar else self.order

    @property
    def width(self) -> int:
        """Total design width."""
        return self.n_intercept + self.n_ar

    def intercept_col(self, regime: int) -> int:
        """Index of the intercept column serving ``regime``."""
        return regime if self.switching_mean else 0

    def ar_slice(self, regime: int) -> slice:
        """Column slice of the AR block serving ``regime``."""
        base = self.n_intercept
        if self.switching_ar:
            return slice(base + regime * self.order, base + (regime + 1) * self.order)
        return slice(base, base + self.order)


@dataclass(frozen=True, slots=True)
class _MSARFit:
    """Raw outputs of a Markov-switching autoregression fit."""

    transition: npt.NDArray[np.float64]
    intercepts: npt.NDArray[np.float64]
    ar_params: npt.NDArray[np.float64]
    variances: npt.NDArray[np.float64]
    filtered_prob: npt.NDArray[np.float64]
    predicted_prob: npt.NDArray[np.float64]
    smoothed_prob: npt.NDArray[np.float64]
    ergodic_prob: npt.NDArray[np.float64]
    expected_durations: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int
    n_iter: int
    converged: bool


@dataclass(frozen=True, slots=True)
class _EMFit:
    """Internal state carried between EM restarts."""

    transition: npt.NDArray[np.float64]
    intercepts: npt.NDArray[np.float64]
    ar_params: npt.NDArray[np.float64]
    sigma2: npt.NDArray[np.float64]
    filtered_prob: npt.NDArray[np.float64]
    predicted_prob: npt.NDArray[np.float64]
    smoothed_prob: npt.NDArray[np.float64]
    llf: float
    n_iter: int
    converged: bool


def _ms_lag_matrix(y: npt.NDArray[np.float64], order: int) -> npt.NDArray[np.float64]:
    """Lagged levels aligned to the effective sample ``t = p .. n - 1``.

    Distinct from :func:`cultivars._core._design.lag_matrix`: this variant
    returns a zero-width matrix of full length when ``order == 0``, which the
    EM loop relies on for the switching-intercept-only specification.

    Args:
        y: The series.
        order: Autoregressive order.

    Returns:
        An ``(n - order, order)`` array, or ``(n, 0)`` when ``order == 0``.
    """
    n = y.shape[0]
    if order == 0:
        return np.zeros((n, 0), dtype=np.float64)
    return np.column_stack([y[order - 1 - i : n - 1 - i] for i in range(order)])


def _regime_means(
    intercepts: npt.NDArray[np.float64],
    ar_params: npt.NDArray[np.float64],
    lags: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Per-regime conditional means of shape ``(T_eff, K)``."""
    if lags.shape[1] == 0:
        return np.broadcast_to(intercepts, (lags.shape[0], intercepts.shape[0])).copy()
    return intercepts[None, :] + lags @ ar_params.T


def _log_densities(
    target: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    sigma2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Gaussian log conditional densities of shape ``(T_eff, K)``."""
    resid = target[:, None] - means
    return -0.5 * (_LOG_2PI + np.log(sigma2)[None, :] + resid**2 / sigma2[None, :])


def _update_transition(
    smoothed: npt.NDArray[np.float64],
    joint: npt.NDArray[np.float64],
    floor: float,
) -> npt.NDArray[np.float64]:
    """M-step transition update from expected transition counts.

    Probabilities are floored before renormalizing, so a regime that becomes
    momentarily unvisited can still be re-entered instead of being absorbed.

    Args:
        smoothed: Smoothed regime probabilities.
        joint: Smoothed joint probabilities ``Pr(S_t = i, S_{t+1} = j | y)``.
        floor: Minimum admissible probability.

    Returns:
        The updated row-stochastic transition matrix.
    """
    k = smoothed.shape[1]
    if joint.shape[0] == 0:
        return np.full((k, k), 1.0 / k)
    numer = joint.sum(axis=0)
    denom = smoothed[:-1].sum(axis=0)
    p = numer / np.clip(denom[:, None], floor, None)
    p = np.clip(p, floor, None)
    return p / p.sum(axis=1, keepdims=True)


def _update_coefficients(
    target: npt.NDArray[np.float64],
    lags: npt.NDArray[np.float64],
    smoothed: npt.NDArray[np.float64],
    sigma2: npt.NDArray[np.float64],
    layout: _Layout,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """M-step coefficient update by responsibility-weighted GLS.

    All regimes are solved in one stacked system so that non-switching blocks
    are estimated jointly across regimes rather than per regime.

    Args:
        target: The effective sample.
        lags: Lagged levels.
        smoothed: Smoothed regime probabilities.
        sigma2: Current per-regime variances.
        layout: Column bookkeeping.

    Returns:
        A tuple ``(intercepts, ar_params)`` of shapes ``(K,)`` and ``(K, p)``.
    """
    n_eff = target.shape[0]
    k, p = layout.n_regimes, layout.order
    d = layout.width
    a = np.zeros((d, d), dtype=np.float64)
    b = np.zeros(d, dtype=np.float64)
    for j in range(k):
        design = np.zeros((n_eff, d), dtype=np.float64)
        design[:, layout.intercept_col(j)] = 1.0
        if p:
            design[:, layout.ar_slice(j)] = lags
        weight = smoothed[:, j] / sigma2[j]
        a += design.T @ (design * weight[:, None])
        b += design.T @ (weight * target)
    beta, *_ = np.linalg.lstsq(a, b, rcond=None)

    intercepts = np.empty(k, dtype=np.float64)
    ar_params = np.zeros((k, p), dtype=np.float64)
    for j in range(k):
        intercepts[j] = beta[layout.intercept_col(j)]
        if p:
            ar_params[j] = beta[layout.ar_slice(j)]
    return intercepts, ar_params


def _update_variance(
    target: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    smoothed: npt.NDArray[np.float64],
    switching_variance: bool,
    floor: float,
) -> npt.NDArray[np.float64]:
    """M-step variance update, per regime or pooled.

    Args:
        target: The effective sample.
        means: Per-regime conditional means.
        smoothed: Smoothed regime probabilities.
        switching_variance: Whether the variance switches across regimes.
        floor: Minimum admissible variance.

    Returns:
        Per-regime variances of shape ``(K,)``.
    """
    k = smoothed.shape[1]
    sq = (target[:, None] - means) ** 2
    if switching_variance:
        sigma2 = (smoothed * sq).sum(axis=0) / np.clip(smoothed.sum(axis=0), 1e-12, None)
    else:
        sigma2 = np.full(k, float((smoothed * sq).sum() / target.shape[0]))
    return np.clip(sigma2, floor, None)


def _initial_transition(
    k: int, rng: np.random.Generator, diagonal: float
) -> npt.NDArray[np.float64]:
    """Randomized persistent transition matrix for a restart.

    Args:
        k: Number of regimes.
        rng: Random generator.
        diagonal: Target self-transition probability.

    Returns:
        A row-stochastic ``(K, K)`` matrix.
    """
    off = (1.0 - diagonal) / (k - 1)
    p = np.full((k, k), off) + (diagonal - off) * np.eye(k)
    p = p * rng.uniform(0.9, 1.1, size=(k, k))
    return p / p.sum(axis=1, keepdims=True)


def _run_em(
    target: npt.NDArray[np.float64],
    lags: npt.NDArray[np.float64],
    layout: _Layout,
    transition0: npt.NDArray[np.float64],
    intercepts0: npt.NDArray[np.float64],
    ar0: npt.NDArray[np.float64],
    sigma20: npt.NDArray[np.float64],
    var_floor: float,
    prob_floor: float,
    max_iter: int,
    tol: float,
    switching_variance: bool,
) -> _EMFit:
    """Run EM to convergence or ``max_iter`` from one set of starting values.

    Returns:
        The :class:`_EMFit` reached.

    Raises:
        NumericalError: If the log-likelihood becomes non-finite.
    """
    transition = transition0.copy()
    intercepts = intercepts0.copy()
    ar_params = ar0.copy()
    sigma2 = sigma20.copy()
    prev_llf = -np.inf
    filtered = predicted = smoothed = np.empty((0, layout.n_regimes))
    n_iter = 0
    converged = False

    for n_iter in range(1, max_iter + 1):
        means = _regime_means(intercepts, ar_params, lags)
        logd = _log_densities(target, means, sigma2)
        filt = hamilton_filter(logd, transition, initial_prob=ergodic_distribution(transition))
        smooth = kim_smoother(filt, transition)
        filtered = filt.filtered_prob
        predicted = filt.predicted_prob
        smoothed = smooth.smoothed_prob
        llf = filt.loglikelihood
        if not np.isfinite(llf):
            raise NumericalError("MS-AR log-likelihood became non-finite during EM.")
        if llf - prev_llf < tol and n_iter > 1:
            converged = True
            prev_llf = llf
            break
        prev_llf = llf
        transition = _update_transition(smoothed, smooth.smoothed_joint_prob, prob_floor)
        intercepts, ar_params = _update_coefficients(target, lags, smoothed, sigma2, layout)
        means = _regime_means(intercepts, ar_params, lags)
        sigma2 = _update_variance(target, means, smoothed, switching_variance, var_floor)

    return _EMFit(
        transition=transition,
        intercepts=intercepts,
        ar_params=ar_params,
        sigma2=sigma2,
        filtered_prob=filtered,
        predicted_prob=predicted,
        smoothed_prob=smoothed,
        llf=float(prev_llf),
        n_iter=n_iter,
        converged=converged,
    )


class _MSARModel[R](_UnivariateModel[R]):
    """Specification surface for Markov-switching autoregressions.

    Args:
        endog: The series.
        order: Autoregressive order ``p``.
        n_regimes: Number of regimes ``K``.
        switching_mean: Whether the intercept switches.
        switching_variance: Whether the innovation variance switches.
        switching_ar: Whether the AR coefficients switch.

    Raises:
        SpecificationError: If no component switches, so no regime is
            identified, or an order is invalid.
        DimensionError: If the series is too short for ``K`` regimes.
    """

    __slots__ = (
        "_k",
        "_order",
        "_sw_ar",
        "_sw_mean",
        "_sw_var",
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        n_regimes: int = 2,
        switching_mean: bool = True,
        switching_variance: bool = True,
        switching_ar: bool = False,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog)
        self._order = validate_order(order, "order")
        self._k = validate_order(n_regimes, "n_regimes", minimum=2)
        self._sw_mean = bool(switching_mean)
        self._sw_var = bool(switching_variance)
        self._sw_ar = bool(switching_ar)
        if not (self._sw_mean or self._sw_var or self._sw_ar):
            raise SpecificationError(
                "at least one of switching_mean, switching_variance, switching_ar "
                "must be True; otherwise no regime is identified."
            )
        self._ensure_length(self._k * (self._order + 2), f"MSAR({self._order}, K={self._k})")

    @property
    def order(self) -> int:
        """The autoregressive order."""
        return self._order

    @property
    def n_regimes(self) -> int:
        """The number of regimes."""
        return self._k

    def _fit_family(
        self,
        *,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
        n_init: int = _DEFAULT_STARTS,
        screen_iter: int = 15,
        seed: int | np.random.Generator | None = None,
    ) -> _MSARFit:
        """Estimate by EM with multi-start screening.

        Args:
            max_iter: Maximum EM iterations for the refined winning start.
            tol: Convergence tolerance on the log-likelihood increment.
            n_init: Number of random starts to screen.
            screen_iter: Iterations used to score each screening start.
            seed: Seed or generator for the random starts.

        Returns:
            The packed :class:`_MSARFit`, regimes ordered by intercept.

        Raises:
            NumericalError: If every start fails to produce a finite likelihood.
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        y = self.endog
        order, k = self._order, self._k
        layout = _Layout(k, order, self._sw_mean, self._sw_ar)
        target = y[order:]
        lags = _ms_lag_matrix(y, order)
        total_var = float(np.var(y))
        var_floor = 1e-8 * total_var + 1e-12
        prob_floor = 1e-8

        def cluster_sigma2(
            intercepts: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            assign = np.argmin(np.abs(target[:, None] - intercepts[None, :]), axis=1)
            sig = np.empty(k, dtype=np.float64)
            for j in range(k):
                group = target[assign == j]
                sig[j] = float(np.var(group)) if group.size > 1 else total_var
            if not self._sw_var:
                sig[:] = sig.mean()
            return np.clip(sig, var_floor, None)

        def make_start(
            index: int,
        ) -> tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ]:
            if index == 0:
                intercepts = np.quantile(y, np.linspace(0.5 / k, 1.0 - 0.5 / k, k))
                diagonal = 0.9
            else:
                intercepts = np.sort(rng.choice(target, size=k, replace=False))
                diagonal = float(rng.uniform(0.8, 0.95))
            return (
                _initial_transition(k, rng, diagonal),
                intercepts,
                np.zeros((k, order), dtype=np.float64),
                cluster_sigma2(intercepts),
            )

        best: _EMFit | None = None
        for index in range(max(n_init, 1)):
            transition0, intercepts0, ar_start, sigma20 = make_start(index)
            try:
                fit = _run_em(
                    target,
                    lags,
                    layout,
                    transition0,
                    intercepts0,
                    ar_start,
                    sigma20,
                    var_floor,
                    prob_floor,
                    screen_iter,
                    tol,
                    self._sw_var,
                )
            except NumericalError:
                continue
            if best is None or fit.llf > best.llf:
                best = fit
        if best is None:
            raise NumericalError("MS-AR estimation failed for every start.")

        refined = _run_em(
            target,
            lags,
            layout,
            best.transition,
            best.intercepts,
            best.ar_params,
            best.sigma2,
            var_floor,
            prob_floor,
            max_iter,
            tol,
            self._sw_var,
        )
        fit = refined if refined.llf >= best.llf else best

        perm = np.argsort(fit.intercepts)
        transition = fit.transition[np.ix_(perm, perm)]
        intercepts = fit.intercepts[perm]
        ar_params = fit.ar_params[perm]
        variances = fit.sigma2[perm]
        smoothed = fit.smoothed_prob[:, perm]
        means = _regime_means(intercepts, ar_params, lags)
        fitted = (smoothed * means).sum(axis=1)
        return _MSARFit(
            transition=transition,
            intercepts=intercepts,
            ar_params=ar_params,
            variances=variances,
            filtered_prob=fit.filtered_prob[:, perm],
            predicted_prob=fit.predicted_prob[:, perm],
            smoothed_prob=smoothed,
            ergodic_prob=ergodic_distribution(transition),
            expected_durations=1.0 / np.clip(1.0 - np.diag(transition), 1e-12, None),
            resid=target - fitted,
            fittedvalues=fitted,
            llf=fit.llf,
            nobs=target.shape[0],
            n_params=k * (k - 1) + layout.n_intercept + layout.n_ar + (k if self._sw_var else 1),
            n_iter=fit.n_iter,
            converged=fit.converged,
        )
