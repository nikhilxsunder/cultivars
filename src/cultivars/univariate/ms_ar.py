# filepath: /src/cultivars/univariate/ms_ar.py
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

"""Markov-switching autoregression (MS-AR).

A ``K``-regime autoregression whose intercept, autoregressive coefficients, and
innovation variance may each switch with a latent first-order Markov chain
``S_t in {0, ..., K-1}``:

    y_t = c_{S_t} + phi_{1, S_t} y_{t-1} + ... + phi_{p, S_t} y_{t-p} + eps_t,
    eps_t ~ N(0, sigma2_{S_t}),

with regime transitions governed by a row-stochastic matrix ``P`` where
``P[i, j] = Pr(S_t = j | S_{t-1} = i)``. Which blocks actually switch is chosen
at construction (``switching_mean`` / ``switching_ar`` / ``switching_variance``),
so this one class spans the Krolzig (1997) MSI / MSA / MSH taxonomy — the
recession-dating MSIH(2)-AR(4) of Hamilton's tradition is
``MSAR(y, order=4, n_regimes=2)`` with switching intercept and variance.

This is the **intercept-switching** parameterization (MSI): the regime enters
contemporaneously through ``c_{S_t}``, so conditional on the observed lags the
model is linear in each regime and the inference problem is a plain ``K``-state
chain. That is what lets estimation ride on the discrete Hamilton filter / Kim
smoother in :mod:`cultivars.state_space.regime_switching` with ``K`` states
rather than the ``K**(p+1)`` states that Hamilton's original *mean*-switching
(MSM) form requires. The two forms coincide at ``p = 0`` and differ in transient
dynamics otherwise; MSM is a candidate for a later ``switching="mean"`` option.

Estimation is by the EM algorithm (Hamilton 1990): the E-step runs the filter
and smoother to obtain regime responsibilities, and the M-step updates the
transition matrix from expected transition counts and the regime coefficients /
variances by responsibility-weighted least squares. Because the likelihood is
multimodal, :meth:`MSAR.fit` screens several random starts and refines the best.

References:
    Hamilton, J. D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. *Econometrica*, 57(2).
    Hamilton, J. D. (1990). Analysis of time series subject to changes in
    regime. *Journal of Econometrics*, 45(1-2).
    Krolzig, H.-M. (1997). *Markov-Switching Vector Autoregressions*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError, NumericalError, SpecificationError
from ..state_space.regime_switching import (
    ergodic_distribution,
    hamilton_filter,
    kim_smoother,
)
from ._base import InformationCriteria, information_criteria

# --------------------------------------------------------------------------
# Internal EM machinery
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Layout:
    """Column bookkeeping for the responsibility-weighted regression."""

    n_regimes: int
    order: int
    switching_mean: bool
    switching_ar: bool

    @property
    def n_intercept(self) -> int:
        return self.n_regimes if self.switching_mean else 1

    @property
    def n_ar(self) -> int:
        return self.n_regimes * self.order if self.switching_ar else self.order

    @property
    def width(self) -> int:
        return self.n_intercept + self.n_ar

    def intercept_col(self, regime: int) -> int:
        return regime if self.switching_mean else 0

    def ar_slice(self, regime: int) -> slice:
        base = self.n_intercept
        if self.switching_ar:
            return slice(base + regime * self.order, base + (regime + 1) * self.order)
        return slice(base, base + self.order)


@dataclass(frozen=True)
class _EMFit:
    transition: npt.NDArray[np.float64]
    intercepts: npt.NDArray[np.float64]      # (K,)
    ar_params: npt.NDArray[np.float64]       # (K, p)
    sigma2: npt.NDArray[np.float64]          # (K,)
    filtered_prob: npt.NDArray[np.float64]
    predicted_prob: npt.NDArray[np.float64]
    smoothed_prob: npt.NDArray[np.float64]
    llf: float
    n_iter: int
    converged: bool


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
    transition = transition0.copy()
    intercepts = intercepts0.copy()
    ar_params = ar0.copy()
    sigma2 = sigma20.copy()
    prev_llf = -np.inf
    filtered = predicted = smoothed = np.empty((0, layout.n_regimes))
    n_iter = 0
    converged = False

    for n_iter in range(1, max_iter + 1):
        # E-step -----------------------------------------------------------
        means = _regime_means(intercepts, ar_params, lags)
        logd = _log_densities(target, means, sigma2)
        init = ergodic_distribution(transition)
        filt = hamilton_filter(logd, transition, initial_prob=init)
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

        # M-step -----------------------------------------------------------
        transition = _update_transition(smoothed, smooth.smoothed_joint_prob, prob_floor)
        intercepts, ar_params = _update_coefficients(
            target, lags, smoothed, sigma2, layout
        )
        means = _regime_means(intercepts, ar_params, lags)
        sigma2 = _update_variance(
            target, means, smoothed, switching_variance, var_floor
        )

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


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class MSARResult(_MeanResult):
    n_regimes: int
    order: int
    switching_mean: bool
    switching_ar: bool
    switching_variance: bool
    transition: npt.NDArray[np.float64]
    intercepts: npt.NDArray[np.float64]
    ar_params: npt.NDArray[np.float64]
    filtered_prob: npt.NDArray[np.float64]
    predicted_prob: npt.NDArray[np.float64]
    smoothed_prob: npt.NDArray[np.float64]
    ergodic_prob: npt.NDArray[np.float64]
    expected_durations: npt.NDArray[np.float64]
    n_iter: int
    converged: bool


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------

class MSAR:
    """Markov-switching autoregression MS-AR(p) with ``K`` regimes.

    Args:
        endog: Univariate series (1-D array-like).
        order: Autoregressive order ``p`` (non-negative integer).
        n_regimes: Number of regimes ``K`` (>= 2). Defaults to 2.
        switching_mean: Whether the intercept switches across regimes (default True).
        switching_variance: Whether the innovation variance switches (default True).
        switching_ar: Whether the AR coefficients switch (default False).

    Raises:
        SpecificationError: If the orders/flags are invalid or the series is too
            short for the requested specification.
        DimensionError: If ``endog`` is not one-dimensional.
        NumericalError: If ``endog`` contains non-finite values.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> P = np.array([[0.97, 0.03], [0.05, 0.95]])
        >>> mu = np.array([-1.0, 1.5])
        >>> s = np.zeros(1500, dtype=int)
        >>> for t in range(1, 1500):
        ...     s[t] = np.searchsorted(np.cumsum(P[s[t - 1]]), rng.random())
        >>> y = mu[s] + 0.5 * rng.standard_normal(1500)
        >>> res = MSAR(y, order=1, n_regimes=2).fit(seed=0)
        >>> bool(res.intercepts[0] < res.intercepts[1])
        True
    """

    __slots__ = (
        "_y", "_order", "_k", "_sw_mean", "_sw_var", "_sw_ar",
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        order: int,
        n_regimes: int = 2,
        *,
        switching_mean: bool = True,
        switching_variance: bool = True,
        switching_ar: bool = False,
    ) -> None:
        """Initialize the MS-AR specification."""
        y = np.asarray(endog, dtype=np.float64)
        if y.ndim != 1:
            raise DimensionError(f"endog must be one-dimensional; got shape {y.shape}.")
        if not np.all(np.isfinite(y)):
            raise NumericalError("endog contains non-finite values.")
        if not isinstance(order, (int, np.integer)) or order < 0:
            raise SpecificationError(f"order must be a non-negative integer; got {order!r}.")
        if not isinstance(n_regimes, (int, np.integer)) or n_regimes < 2:
            raise SpecificationError(f"n_regimes must be an integer >= 2; got {n_regimes!r}.")
        if not (switching_mean or switching_variance or switching_ar):
            raise SpecificationError(
                "at least one of switching_mean, switching_variance, switching_ar "
                "must be True; otherwise the model has no regime dependence."
            )
        n_eff = y.shape[0] - order
        min_eff = 2 * n_regimes * (order + 2)
        if n_eff < min_eff:
            raise SpecificationError(
                f"series of length {y.shape[0]} is too short for MS-AR({order}) with "
                f"{n_regimes} regimes (need at least {min_eff + order} observations)."
            )
        self._y = y
        self._order = int(order)
        self._k = int(n_regimes)
        self._sw_mean = bool(switching_mean)
        self._sw_var = bool(switching_variance)
        self._sw_ar = bool(switching_ar)

    @property
    def order(self) -> int:
        """The autoregressive order ``p``."""
        return self._order

    @property
    def n_regimes(self) -> int:
        """The number of regimes ``K``."""
        return self._k

    def fit(
        self,
        *,
        max_iter: int = 500,
        tol: float = 1e-6,
        n_init: int = 10,
        screen_iter: int = 15,
        seed: int | np.random.Generator | None = None,
    ) -> MSARResult:
        """Estimate the model by EM with multi-start screening.

        Args:
            max_iter: Maximum EM iterations for the refined (winning) start.
            tol: Convergence tolerance on the log-likelihood increment.
            n_init: Number of random starts to screen.
            screen_iter: EM iterations used to score each screening start before
                the best is refined to ``max_iter``.
            seed: Seed or generator for the random starts (reproducibility).

        Returns:
            The fitted :class:`MSARResult`.

        Raises:
            NumericalError: If every start fails to produce a finite likelihood.
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        y = self._y
        order, k = self._order, self._k
        layout = _Layout(k, order, self._sw_mean, self._sw_ar)
        target = y[order:]
        lags = _lag_matrix(y, order)
        total_var = float(np.var(y))
        var_floor = 1e-8 * total_var + 1e-12
        prob_floor = 1e-8

        def cluster_sigma2(intercepts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            # Hard-assign each observation to its nearest intercept and use the
            # within-cluster variance as a correctly-scaled variance start. A
            # variance seed on the order of the *total* variance would let one
            # wide regime absorb the sample and strand EM at a spurious mode.
            assign = np.argmin(np.abs(target[:, None] - intercepts[None, :]), axis=1)
            sig = np.empty(k, dtype=np.float64)
            for j in range(k):
                group = target[assign == j]
                sig[j] = float(np.var(group)) if group.size > 1 else total_var
            if not self._sw_var:
                sig[:] = sig.mean()
            return np.clip(sig, var_floor, None)

        def make_start(index: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
                                             npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            if index == 0:
                intercepts = np.quantile(y, np.linspace(0.5 / k, 1.0 - 0.5 / k, k))
                diagonal = 0.9
            else:
                # k-means++-style seeding from actual observations spreads the
                # regime means realistically and is far more reliable than a
                # symmetric jitter around the grand mean.
                intercepts = np.sort(rng.choice(target, size=k, replace=False))
                diagonal = float(rng.uniform(0.8, 0.95))
            transition = _initial_transition(k, rng, diagonal)
            sigma2 = cluster_sigma2(intercepts)
            # AR blocks start at zero: the marginal autocorrelation of a
            # switching series reflects regime persistence, not within-regime
            # dynamics, so seeding it there strands EM at a spurious mode.
            ar_params = np.zeros((k, order), dtype=np.float64)
            return transition, intercepts, ar_params, sigma2

        best: _EMFit | None = None
        for index in range(max(n_init, 1)):
            transition0, intercepts0, ar_start, sigma20 = make_start(index)
            try:
                fit = _run_em(
                    target, lags, layout, transition0, intercepts0, ar_start, sigma20,
                    var_floor, prob_floor, screen_iter, tol, self._sw_var,
                )
            except NumericalError:
                continue
            if best is None or fit.llf > best.llf:
                best = fit

        if best is None:
            raise NumericalError("MS-AR estimation failed for every start.")

        # Refine the winning start to full convergence.
        refined = _run_em(
            target, lags, layout, best.transition, best.intercepts, best.ar_params,
            best.sigma2, var_floor, prob_floor, max_iter, tol, self._sw_var,
        )
        fit = refined if refined.llf >= best.llf else best
        return self._finalize(fit, layout, target, lags)

    def _finalize(
        self,
        fit: _EMFit,
        layout: _Layout,
        target: npt.NDArray[np.float64],
        lags: npt.NDArray[np.float64],
    ) -> MSARResult:
        order, k = self._order, self._k
        perm = np.argsort(fit.intercepts)             # order regimes by intercept
        transition = fit.transition[np.ix_(perm, perm)]
        intercepts = fit.intercepts[perm]
        ar_params = fit.ar_params[perm]
        sigma2 = fit.sigma2[perm]
        filtered = fit.filtered_prob[:, perm]
        predicted = fit.predicted_prob[:, perm]
        smoothed = fit.smoothed_prob[:, perm]

        means = _regime_means(intercepts, ar_params, lags)
        fitted = (smoothed * means).sum(axis=1)
        resid = target - fitted

        n_params = (
            k * (k - 1)
            + layout.n_intercept
            + layout.n_ar
            + (k if self._sw_var else 1)
        )
        return MSARResult(
            n_regimes=k,
            order=order,
            switching_mean=self._sw_mean,
            switching_ar=self._sw_ar,
            switching_variance=self._sw_var,
            transition=transition,
            intercepts=intercepts,
            ar_params=ar_params,
            sigma2=sigma2,
            filtered_prob=filtered,
            predicted_prob=predicted,
            smoothed_prob=smoothed,
            ergodic_prob=ergodic_distribution(transition),
            expected_durations=1.0 / np.clip(1.0 - np.diag(transition), 1e-12, None),
            llf=fit.llf,
            nobs=target.shape[0],
            n_iter=fit.n_iter,
            converged=fit.converged,
            resid=resid,
            fittedvalues=fitted,
            endog=self._y,
            _n_params=n_params,
        )