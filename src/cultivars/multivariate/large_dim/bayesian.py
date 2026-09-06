# filepath: /src/cultivars/multivariate/large_dim/bayesian.py
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

"""The conjugate Bayesian VAR: an exact posterior, and the number that ranks priors.

A Bayesian VAR earns its place in the large-dimensional family because
shrinkage is what makes a large system estimable at all: with the
Normal-inverse-Wishart form of the Minnesota prior, a hundred-variable VAR is
routine (Banbura, Giannone & Reichlin 2010), and the prior's tightness -- not
the sample size -- controls the effective dimensionality. This model is the
conjugate core of that program. The posterior over coefficients *and*
innovation covariance is exact, drawn rather than approximated, and every
reported band is a posterior statement, not an asymptotic one.

Two design commitments distinguish the surface. First, conjugacy is verified,
not assumed: the prior's variances must factor equation-by-column, which is
the Minnesota structure with the cross-equation weight pinned at one. A prior
that keeps Litterman's extra cross-variable shrinkage is refused with
directions to the per-equation point path -- the two are different estimators
with different claims, not the same object with different numbers. Second,
the scalar this result reports is the log *marginal likelihood* of the
sample, computed in closed form with any dummy-observation content divided
out (Giannone, Lenza & Primiceri 2015). Differences of that number across
priors are log Bayes factors, which is what hyperparameter choice should
maximize -- the hierarchical layer built on top of this model does exactly
that. There is deliberately no ``llf`` and no information criteria: a
posterior has neither, and the marginal likelihood is the honest replacement.

References:
    Litterman, R. B. (1986). Forecasting with Bayesian vector
        autoregressions: Five years of experience. *Journal of Business &
        Economic Statistics*, 4(1), 25-38.
    Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and conditional
        projection using realistic prior distributions. *Econometric
        Reviews*, 3(1), 1-100.
    Banbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector
        auto regressions. *Journal of Applied Econometrics*, 25(1), 71-92.
    Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for
        vector autoregressions. *Review of Economics and Statistics*, 97(2),
        436-451.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable, Trend, companion_matrix
from ..._internals import (
    _BayesianVectorAutoRegressionModel,
    _Prior,
    _SummaryMixin,
    _VectorConjugateFit,
)
from ...bayes.priors import NormalInverseWishartPrior
from ...exceptions import SpecificationError

__all__ = ["BVAR", "BVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class BVARResult(_SummaryMixin):
    """A fitted conjugate Bayesian VAR: an exact posterior over ``(B, Sigma)``.

    Point summaries are posterior means; the retained draws carry the full
    posterior, and every method that propagates -- impulse responses, the
    predictive -- propagates draw by draw, so its bands are the posterior of
    the propagated object rather than the propagation of a point.

    Deliberately absent: ``llf``, ``n_params``, information criteria. The
    comparison number is :attr:`log_marginal_likelihood`, whose differences
    across priors are log Bayes factors.

    Attributes:
        endog: The observed panel.
        names: Variable labels, in column order.
        order: Autoregressive order.
        trend: Deterministic specification.
        prior_label: Short description of the prior estimated under.
        coefficients: ``(p, k, k)`` lag stack at the posterior mean.
        deterministic: ``(n_det, k)`` deterministic block at the posterior
            mean.
        beta_mean: ``(w, k)`` full posterior mean coefficient matrix.
        sigma_u: ``(k, k)`` posterior mean innovation covariance.
        beta_draws: ``(S, w, k)`` coefficient draws.
        sigma_draws: ``(S, k, k)`` covariance draws.
        log_marginal_likelihood: Log marginal likelihood of the sample, the
            dummy-observation contribution divided out.
        posterior_df: Inverse-Wishart posterior degrees of freedom.
        resid: Residuals at the posterior mean.
        fittedvalues: One-step means at the posterior mean.
        nobs: Effective sample size, dummy rows excluded.
        n_dummy: Artificial rows the prior contributed.
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    names: tuple[str, ...]
    order: int
    trend: str
    prior_label: str
    coefficients: npt.NDArray[np.float64] = field(repr=False)
    deterministic: npt.NDArray[np.float64] = field(repr=False)
    beta_mean: npt.NDArray[np.float64] = field(repr=False)
    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    beta_draws: npt.NDArray[np.float64] = field(repr=False)
    sigma_draws: npt.NDArray[np.float64] = field(repr=False)
    log_marginal_likelihood: float
    posterior_df: float
    resid: npt.NDArray[np.float64] = field(repr=False)
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    nobs: int
    n_dummy: int

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorConjugateFit,
        model: _BayesianVectorAutoRegressionModel[BVARResult],
    ) -> BVARResult:
        """Assemble the public result from a raw fit and its specification."""
        return cls(
            endog=model.endog,
            names=model.names,
            order=model.order,
            trend=model.trend,
            prior_label=model.prior._label(),
            coefficients=fit.coefficient_stack,
            deterministic=fit.deterministic,
            beta_mean=fit.beta_mean,
            sigma_u=fit.sigma_u,
            beta_draws=fit.beta_draws,
            sigma_draws=fit.sigma_draws,
            log_marginal_likelihood=fit.log_marginal_likelihood,
            posterior_df=fit.posterior_df,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            nobs=fit.nobs,
            n_dummy=fit.n_dummy,
        )

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables."""
        return len(self.names)

    @property
    def n_kept(self) -> int:
        """Posterior draws retained."""
        return int(self.beta_draws.shape[0])

    @property
    def _n_deterministic(self) -> int:
        """Deterministic columns per equation."""
        return {"n": 0, "c": 1, "ct": 2}[self.trend]

    def _regressor_labels(self) -> tuple[str, ...]:
        """Per-equation regressor labels, in design order."""
        det = ("const", "trend")[: self._n_deterministic]
        lags = tuple(f"{source}.L{lag + 1}" for lag in range(self.order) for source in self.names)
        return (*det, *lags)

    def credible_interval(self, equation: str, regressor: str) -> npt.NDArray[np.float64]:
        """One coefficient's posterior summary: 16th percentile, mean, 84th.

        Args:
            equation: An endogenous variable.
            regressor: A per-equation regressor label -- ``"const"``,
                ``"trend"``, or ``"{name}.L{lag}"``.

        Returns:
            A ``(3,)`` array ``(low, mean, high)``.

        Raises:
            SpecificationError: If the equation or regressor is unknown.
        """
        if equation not in self.names:
            raise SpecificationError(
                f"unknown variable {equation!r}; expected one of {self.names}."
            )
        labels = self._regressor_labels()
        if regressor not in labels:
            raise SpecificationError(f"unknown regressor {regressor!r}; expected one of {labels}.")
        draws = self.beta_draws[:, labels.index(regressor), self.names.index(equation)]
        return np.array(
            [float(np.quantile(draws, 0.16)), float(draws.mean()), float(np.quantile(draws, 0.84))]
        )

    def _stack_of(self, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Slice one ``(w, k)`` draw into its ``(p, k, k)`` lag stack."""
        k, offset = self.k_endog, self._n_deterministic
        if not self.order:
            return np.zeros((0, k, k))
        return np.stack(
            [beta[offset + lag * k : offset + (lag + 1) * k].T for lag in range(self.order)]
        )

    def _irf_draws(self, horizon: int, *, orthogonalized: bool) -> npt.NDArray[np.float64]:
        """Impulse responses of every retained draw, ``(S, horizon + 1, k, k)``."""
        if horizon < 0:
            raise SpecificationError(f"horizon must be non-negative; got {horizon}.")
        k, p, n_kept = self.k_endog, self.order, self.n_kept
        out = np.empty((n_kept, horizon + 1, k, k))
        for s in range(n_kept):
            psi = np.empty((horizon + 1, k, k))
            if p == 0:
                psi[:] = 0.0
                psi[0] = np.eye(k)
            else:
                selector = np.zeros((k, k * p))
                selector[:, :k] = np.eye(k)
                power = np.eye(k * p)
                companion = companion_matrix(self._stack_of(self.beta_draws[s]))
                for h in range(horizon + 1):
                    psi[h] = selector @ power @ selector.T
                    power = power @ companion
            out[s] = psi @ np.linalg.cholesky(self.sigma_draws[s]) if orthogonalized else psi
        return out

    def irf(
        self,
        horizon: int = 20,
        *,
        orthogonalized: bool = True,
        cumulative: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Posterior mean impulse responses.

        The mean of each draw's response, not the response of the mean draw
        -- the distinction matters because propagation is nonlinear in the
        coefficients. Orthogonalization applies each draw's own Cholesky
        factor, so the ordering of ``names`` is an identifying assumption,
        as for any recursive scheme.

        Args:
            horizon: Largest lead to return.
            orthogonalized: Rotate each draw by its own Cholesky factor.
            cumulative: Return running sums.

        Returns:
            An array of shape ``(horizon + 1, k, k)``; entry ``[h, i, j]``
            is the response of variable ``i`` at lead ``h`` to shock ``j``.

        Raises:
            SpecificationError: If ``horizon`` is negative.
        """
        out = self._irf_draws(horizon, orthogonalized=orthogonalized).mean(axis=0)
        return np.cumsum(out, axis=0) if cumulative else out

    def irf_bands(
        self,
        horizon: int = 20,
        *,
        orthogonalized: bool = True,
        cumulative: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Posterior bands of the impulse responses.

        Args:
            horizon: Largest lead to return.
            orthogonalized: Rotate each draw by its own Cholesky factor.
            cumulative: Band the running sums instead.

        Returns:
            An array of shape ``(horizon + 1, k, k, 3)`` whose last axis is
            ``(16th percentile, mean, 84th percentile)`` across draws.

        Raises:
            SpecificationError: If ``horizon`` is negative.
        """
        draws = self._irf_draws(horizon, orthogonalized=orthogonalized)
        if cumulative:
            draws = np.cumsum(draws, axis=1)
        return np.stack(
            [
                np.quantile(draws, 0.16, axis=0),
                draws.mean(axis=0),
                np.quantile(draws, 0.84, axis=0),
            ],
            axis=-1,
        )

    def forecast(
        self,
        steps: int = 8,
        *,
        seed: int | np.random.Generator | None = None,
    ) -> npt.NDArray[np.float64]:
        """The posterior predictive: parameter *and* shock uncertainty.

        Each retained draw simulates its own future -- its coefficients, its
        covariance, its innovations -- so the bands are bands of the
        predictive distribution, which is the object a forecast evaluation
        actually scores. This is where the Bayesian model pays for itself:
        the point path's forecast intervals condition on the estimated
        parameters, and for a large shrunk system that conditioning is the
        larger half of the uncertainty.

        Args:
            steps: Horizons ahead.
            seed: Seed or generator for the predictive shocks.

        Returns:
            An array of shape ``(steps, k, 3)`` whose last axis is
            ``(16th percentile, mean, 84th percentile)``.

        Raises:
            SpecificationError: If ``steps`` is not positive.
        """
        if steps < 1:
            raise SpecificationError(f"steps must be positive; got {steps}.")
        rng = np.random.default_rng(seed)
        k, p, n = self.k_endog, self.order, self.endog.shape[0]
        offset = self._n_deterministic
        paths = np.empty((self.n_kept, steps, k))
        for s in range(self.n_kept):
            beta = self.beta_draws[s]
            stack = self._stack_of(beta)
            chol = np.linalg.cholesky(self.sigma_draws[s])
            history = list(self.endog[n - p :][::-1]) if p else []
            for h in range(steps):
                det = {"n": [], "c": [1.0], "ct": [1.0, float(n + h + 1)]}[self.trend]
                value = np.asarray(det, dtype=np.float64) @ beta[:offset]
                for lag in range(p):
                    value = value + stack[lag] @ history[lag]
                value = value + chol @ rng.standard_normal(k)
                paths[s, h] = value
                if p:
                    history = [value, *history[:-1]]
        return np.stack(
            [
                np.quantile(paths, 0.16, axis=0),
                paths.mean(axis=0),
                np.quantile(paths, 0.84, axis=0),
            ],
            axis=-1,
        )

    @property
    def stable_share(self) -> float:
        """Posterior probability that the system is stable.

        The share of retained draws whose companion matrix has every root
        strictly inside the unit circle. Under a random-walk-centred prior
        this is routinely well below one, and that is information about the
        posterior, not a defect: near-unit-root mass is what the prior
        asserts.
        """
        if self.order == 0:
            return 1.0
        stable = 0
        for s in range(self.n_kept):
            eigs = np.linalg.eigvals(companion_matrix(self._stack_of(self.beta_draws[s])))
            stable += int(float(np.abs(eigs).max(initial=0.0)) < 1.0)
        return stable / self.n_kept

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        rows = []
        for name in self.names:
            if self.order:
                low, mid, high = self.credible_interval(name, f"{name}.L1")
                rows.append((name, f"{mid:.4f}", f"[{low:.4f}, {high:.4f}]"))
            else:
                rows.append((name, "-", "-"))
        notes = [
            "The posterior is exact Normal-inverse-Wishart; draws are "
            "independent, not a Markov chain, so no convergence diagnostics "
            "apply.",
            "log_marginal_likelihood is the sample's, with the "
            "dummy-observation contribution divided out; differences across "
            "priors are log Bayes factors, and hyperparameter choice should "
            "maximize it (Giannone-Lenza-Primiceri).",
            f"Posterior probability of stability: {self.stable_share:.2f}. "
            "Mass near the unit circle is the random-walk prior speaking, "
            "not a defect.",
            "No llf, parameter count, or information criteria are reported: "
            "a posterior has none, and the marginal likelihood is the "
            "honest replacement.",
            "forecast() is the full posterior predictive -- parameter and "
            "shock uncertainty jointly.",
        ]
        return SummaryTable(
            title=f"BVAR({self.order}) Results",
            metadata=(
                ("Model", f"BVAR({self.order})"),
                ("Prior", self.prior_label),
                ("Draws", f"{self.n_kept}"),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.nobs}"),
                ("Dummy rows", f"{self.n_dummy}"),
                ("Trend", self.trend),
                ("log ML", f"{self.log_marginal_likelihood:.3f}"),
            ),
            columns=("equation", "own L1 mean", "68% interval"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class BVAR(_BayesianVectorAutoRegressionModel[BVARResult]):
    """Conjugate Normal-inverse-Wishart Bayesian VAR, Banbura-Giannone-Reichlin.

    The default prior is the conjugate Minnesota
    (:class:`~cultivars.bayes.priors.NormalInverseWishartPrior` at its
    defaults); compose the Giannone-Lenza-Primiceri dummies onto it with
    ``+`` -- ``NormalInverseWishartPrior() + SumOfCoefficientsPrior() +
    DummyInitialObservationPrior()`` -- and rank the candidates by
    :attr:`BVARResult.log_marginal_likelihood`.

    Args:
        endog: The observed panel, shape ``(nobs, k)``. Shrinkage is what
            makes ``k`` large admissible: with a proper prior the sample
            length constraint is one observation, not one per regressor.
        order: Autoregressive order.
        prior: A conjugacy-compatible prior, possibly a composition.
            Defaults to ``NormalInverseWishartPrior()``. A prior whose
            variances break the Kronecker factorization -- Litterman's
            ``cross_equation != 1`` -- is refused at fit with directions to
            the per-equation point path.
        trend: Deterministic terms.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((120, 2))
        >>> for t in range(1, 120):
        ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
        >>> res = BVAR(y, order=1).fit(n_draws=200, seed=0)
        >>> res.beta_draws.shape
        (200, 3, 2)
        >>> res.irf_bands(4).shape
        (5, 2, 2, 3)
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        prior: _Prior | None = None,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
    ) -> None:
        """Default the prior to the conjugate Minnesota, then validate."""
        super().__init__(
            endog,
            order=order,
            trend=trend,
            names=names,
            prior=prior if prior is not None else NormalInverseWishartPrior(),
        )

    def fit(
        self,
        *,
        n_draws: int = 1000,
        seed: int | np.random.Generator | None = None,
    ) -> BVARResult:
        """Update the posterior exactly and draw from it.

        Args:
            n_draws: Independent posterior draws of ``(B, Sigma)`` to
                retain.
            seed: Seed or generator, for reproducibility.

        Returns:
            The fitted :class:`BVARResult`.

        Raises:
            SpecificationError: If the prior is improper or not
                conjugacy-compatible.
            NumericalError: If the posterior degenerates.
        """
        return BVARResult._from_fit(self._fit_conjugate(n_draws=n_draws, seed=seed), self)
