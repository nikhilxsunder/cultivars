# filepath: /src/cultivars/multivariate/large_dim/hierarchical.py
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

"""Hierarchical shrinkage: the tightness is estimated, not asserted.

Every Bayesian VAR result depends on hyperparameters someone chose -- the
Minnesota tightness above all -- and the standard practice of fixing them at
folklore values means the reported bands condition on a number nobody
defends. Giannone, Lenza & Primiceri (2015) close that gap: treat the
hyperparameters as unknowns with their own (hyper)priors, and let the
closed-form marginal likelihood -- exactly the number the conjugate model
already reports -- carry the data's opinion about them. The posterior over
hyperparameters is one three-dimensional density, and everything downstream
follows from standard machinery.

Both of the paper's readings are offered, behind one surface. Empirical
Bayes (``method="empirical"``) maximizes the hyperparameter posterior and
conditions on the mode: fast, and what most applied work does. The full
hierarchy (``method="full"``, the default) runs an adaptive random-walk
Metropolis chain over the log hyperparameters and draws the VAR's
coefficients at every kept state, so the retained draws *marginalize* over
the tightness and the bands stop pretending it was known. On stubbornly
informative samples the two agree; when they disagree, the disagreement is
the finding.

The hierarchy spans the Minnesota tightness ``lambda`` and, optionally, the
sum-of-coefficients and dummy-initial-observation dummies, parameterized as
in the paper (``mu`` and ``delta``, larger = looser) with the paper's Gamma
hyperpriors: mode 0.2 and standard deviation 0.4 for ``lambda``, mode 1 and
standard deviation 1 for ``mu`` and ``delta``.

References:
    Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for
        vector autoregressions. *Review of Economics and Statistics*, 97(2),
        436-451.
    Banbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector
        auto regressions. *Journal of Applied Econometrics*, 25(1), 71-92.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable, Trend, _gamma_from_mode
from ..._internals import (
    _BayesianVectorAutoRegressionModel,
    _Prior,
)
from ...bayes.priors import (
    DummyInitialObservationPrior,
    NormalInverseWishartPrior,
    SumOfCoefficientsPrior,
)
from ...exceptions import SpecificationError
from .bayesian import BVARResult

__all__ = ["HierarchicalBVAR", "HierarchicalBVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class HierarchicalBVARResult(BVARResult):
    """A fitted hierarchical BVAR: the tightness carries a posterior too.

    Everything the conjugate result reports, with one change of meaning and
    one addition. The change: under the full method the retained
    ``(B, Sigma)`` draws marginalize over the hyperparameters, so every band
    downstream -- coefficients, impulse responses, the predictive --
    includes hyperparameter uncertainty; :attr:`log_marginal_likelihood` is
    the value *at the hyperparameter mode*, conditional on it. The addition
    is the hyperparameter layer itself: the posterior mode, the kept draws,
    and the chain's acceptance rate.

    Attributes:
        hyper_names: One label per hyperparameter, in draw-column order --
            ``"lambda"``, then ``"mu"`` and ``"delta"`` when active,
            parameterized as in Giannone-Lenza-Primiceri.
        hyper_mode: Posterior-mode hyperparameter vector.
        hyper_draws: ``(S, d)`` kept hyperparameter draws; empty under
            empirical Bayes.
        acceptance_rate: Metropolis acceptance over the kept span; ``nan``
            under empirical Bayes.
        method: ``"full"`` or ``"empirical"``.
    """

    hyper_names: tuple[str, ...]
    hyper_mode: npt.NDArray[np.float64] = field(repr=False)
    hyper_draws: npt.NDArray[np.float64] = field(repr=False)
    acceptance_rate: float
    method: str

    def hyperparameter_interval(self, name: str) -> npt.NDArray[np.float64]:
        """One hyperparameter's posterior summary: 16th percentile, mean, 84th.

        Args:
            name: A label from :attr:`hyper_names`.

        Returns:
            A ``(3,)`` array ``(low, mean, high)``.

        Raises:
            SpecificationError: If the name is unknown, or the fit was
                empirical Bayes and holds no hyperparameter draws.
        """
        if name not in self.hyper_names:
            raise SpecificationError(
                f"unknown hyperparameter {name!r}; expected one of {self.hyper_names}."
            )
        if not self.hyper_draws.shape[0]:
            raise SpecificationError(
                "the empirical-Bayes fit conditions on the hyperparameter "
                "mode and holds no hyperparameter draws; refit with "
                "method='full' for a hyperparameter posterior."
            )
        draws = self.hyper_draws[:, self.hyper_names.index(name)]
        return np.array(
            [float(np.quantile(draws, 0.16)), float(draws.mean()), float(np.quantile(draws, 0.84))]
        )

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        rows = []
        for i, name in enumerate(self.hyper_names):
            if self.hyper_draws.shape[0]:
                low, mid, high = self.hyperparameter_interval(name)
                rows.append(
                    (name, f"{self.hyper_mode[i]:.4f}", f"{mid:.4f}", f"[{low:.4f}, {high:.4f}]")
                )
            else:
                rows.append((name, f"{self.hyper_mode[i]:.4f}", "-", "-"))
        notes = [
            "Hyperparameters follow Giannone-Lenza-Primiceri: Gamma "
            "hyperpriors (lambda: mode 0.2, sd 0.4; mu, delta: mode 1, sd "
            "1), posterior explored "
            + (
                "by adaptive random-walk Metropolis over log "
                "hyperparameters; the (B, Sigma) draws marginalize over "
                "them."
                if self.method == "full"
                else "to its mode only (empirical Bayes); the (B, Sigma) "
                "draws condition on the mode."
            ),
            "log_marginal_likelihood is evaluated at the hyperparameter "
            "mode, conditional on it; the full model evidence would "
            "integrate over the hyperprior and is not reported.",
            f"Posterior probability of stability: {self.stable_share:.2f}.",
        ]
        if self.method == "full":
            notes.insert(
                1,
                f"Metropolis acceptance rate {self.acceptance_rate:.2f}; "
                "rates far outside 0.1-0.6 warrant a longer burn-in.",
            )
        return SummaryTable(
            title=f"Hierarchical BVAR({self.order}) Results",
            metadata=(
                ("Model", f"Hierarchical BVAR({self.order})"),
                ("Method", self.method),
                ("Draws", f"{self.n_kept}"),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.nobs}"),
                ("Trend", self.trend),
                ("log ML at mode", f"{self.log_marginal_likelihood:.3f}"),
            ),
            columns=("hyperparameter", "mode", "posterior mean", "68% interval"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class HierarchicalBVAR(_BayesianVectorAutoRegressionModel[HierarchicalBVARResult]):
    """Giannone-Lenza-Primiceri hierarchical shrinkage over the conjugate BVAR.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order.
        sum_of_coefficients: Include the sum-of-coefficients dummies, with
            their looseness ``mu`` as a hyperparameter.
        dummy_initial_observation: Include Sims' co-persistence dummy, with
            its looseness ``delta`` as a hyperparameter.
        decay: Minnesota lag decay, held fixed as in the paper.
        exogenous: Looseness of the deterministic block, held fixed.
        persistence: Prior mean of each variable's own first lag.
        trend: Deterministic terms.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((150, 2))
        >>> for t in range(1, 150):
        ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(2)
        >>> res = HierarchicalBVAR(y, order=1).fit(
        ...     method="empirical", n_draws=100, seed=0
        ... )
        >>> res.hyper_names
        ('lambda', 'mu', 'delta')
        >>> res.beta_draws.shape
        (100, 3, 2)
    """

    __slots__ = (
        "_decay",
        "_dio",
        "_exogenous",
        "_persistence",
        "_soc",
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        sum_of_coefficients: bool = True,
        dummy_initial_observation: bool = True,
        decay: float = 1.0,
        exogenous: float = 100.0,
        persistence: float | Sequence[float] = 1.0,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
    ) -> None:
        """Validate the specification; the prior itself is hyperparameterized."""
        self._soc = bool(sum_of_coefficients)
        self._dio = bool(dummy_initial_observation)
        self._decay = float(decay)
        self._exogenous = float(exogenous)
        self._persistence = persistence
        super().__init__(
            endog,
            order=order,
            trend=trend,
            names=names,
            prior=self._prior_at(np.array([0.2, 1.0, 1.0])),
        )

    def _prior_at(self, theta: npt.NDArray[np.float64]) -> _Prior:
        """The prior a hyperparameter vector names.

        The vector is always three long -- ``(lambda, mu, delta)`` in the
        paper's parameterization -- with inactive dummies simply not
        composed in.
        """
        prior: _Prior = NormalInverseWishartPrior(
            tightness=float(theta[0]),
            decay=self._decay,
            exogenous=self._exogenous,
            persistence=self._persistence,
        )
        if self._soc:
            prior = prior + SumOfCoefficientsPrior(tightness=1.0 / float(theta[1]))
        if self._dio:
            prior = prior + DummyInitialObservationPrior(tightness=1.0 / float(theta[2]))
        return prior

    def fit(
        self,
        *,
        method: str = "full",
        n_draws: int = 1000,
        n_burn: int = 500,
        seed: int | np.random.Generator | None = None,
    ) -> HierarchicalBVARResult:
        """Estimate the hyperparameter posterior, then the VAR under it.

        Args:
            method: ``"full"`` marginalizes the ``(B, Sigma)`` draws over
                the hyperparameters by Metropolis; ``"empirical"``
                conditions on the hyperparameter posterior mode.
            n_draws: Kept ``(B, Sigma)`` draws.
            n_burn: Burn-in Metropolis iterations (full method).
            seed: Seed or generator, for reproducibility.

        Returns:
            The fitted :class:`HierarchicalBVARResult`.

        Raises:
            SpecificationError: If the method or counts are malformed.
            NumericalError: If the mode search fails.
        """
        names = ["lambda"]
        modes = [0.2]
        sds = [0.4]
        if self._soc:
            names.append("mu")
            modes.append(1.0)
            sds.append(1.0)
        if self._dio:
            names.append("delta")
            modes.append(1.0)
            sds.append(1.0)
        shapes_scales = [_gamma_from_mode(m, s) for m, s in zip(modes, sds, strict=True)]
        active = tuple(names)

        def factory(theta: npt.NDArray[np.float64]) -> _Prior:
            full = np.ones(3)
            full[0] = theta[0]
            position = 1
            if self._soc:
                full[1] = theta[position]
                position += 1
            if self._dio:
                full[2] = theta[position]
            return self._prior_at(full)

        fit = self._fit_hierarchical(
            factory=factory,
            hyper_names=active,
            hyper_shapes=np.array([pair[0] for pair in shapes_scales]),
            hyper_scales=np.array([pair[1] for pair in shapes_scales]),
            start=np.asarray(modes, dtype=np.float64),
            method=method,
            n_draws=n_draws,
            n_burn=n_burn,
            seed=seed,
        )
        return HierarchicalBVARResult(
            endog=self.endog,
            names=self.names,
            order=self.order,
            trend=self.trend,
            prior_label=f"glp({', '.join(active)})",
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
            hyper_names=fit.hyper_names,
            hyper_mode=fit.hyper_mode,
            hyper_draws=fit.hyper_draws,
            acceptance_rate=fit.acceptance,
            method=fit.method,
        )
