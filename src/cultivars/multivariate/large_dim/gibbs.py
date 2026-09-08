# filepath: /src/cultivars/multivariate/large_dim/gibbs.py
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

"""The Gibbs-sampled BVAR: every prior the conjugate model must refuse.

The conjugate model buys its exact posterior by tying the coefficient prior
to the innovation covariance through a Kronecker product, and two whole
families of priors cannot pay that price. Litterman's original prior --
cross-variable coefficients shrunk harder than own lags -- breaks the
factorization by design. And the adaptive shrinkage hierarchies -- the
horseshoe, spike-and-slab selection, Dirichlet-Laplace, Normal-Gamma --
have no fixed variance to factor at all: each coefficient's variance is a
latent state with its own full conditional. This model is where both
families estimate. The prior is stated *independently* of the covariance
(the independent Normal-Wishart pairing, Koop & Korobilis 2010), and the
posterior is reached by Gibbs sampling: one joint generalized-least-squares
draw of all coefficients given the covariance -- exact and ordering-free --
an inverse-Wishart draw of the covariance given the coefficients, and, for
an adaptive prior, one sweep of its scale hierarchy's own exact
conditionals.

What is given up is stated rather than hidden. There is no marginal
likelihood -- with the prior independent of the covariance the evidence has
no closed form, and a simulated stand-in would not deserve the name -- so
prior comparison by Bayes factor belongs to the conjugate path. Draws are a
Markov chain, not independent, so burn-in and thinning are real choices
here. And the joint coefficient draw factorizes nothing, so each sweep
costs a Cholesky of a ``(k^2 p + k) x (k^2 p + k)``-ish precision: exact at
any moderate dimension, and deliberately not pretending to Banbura-scale
systems.

For a selection prior the distinctive output is
:meth:`GibbsBVARResult.inclusion_probabilities` -- the posterior probability
that each lag coefficient is in the slab, which is the George-Sun-Ni
restriction search read as a result rather than a procedure.

References:
    Koop, G., & Korobilis, D. (2010). Bayesian multivariate time series
        methods for empirical macroeconomics. *Foundations and Trends in
        Econometrics*, 3(4), 267-358.
    George, E. I., Sun, D., & Ni, S. (2008). Bayesian stochastic search
        for VAR model restrictions. *Journal of Econometrics*, 142(1),
        553-580.
    Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe
        estimator for sparse signals. *Biometrika*, 97(2), 465-480.
    Griffin, J. E., & Brown, P. J. (2010). Inference with normal-gamma
        prior distributions in regression problems. *Bayesian Analysis*,
        5(1), 171-188.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable, Trend
from ..._internals import (
    _GibbsBayesianVectorAutoRegressionModel,
    _Prior,
    _VectorGibbsFit,
    _VectorPosteriorDrawsResult,
)
from ...bayes import IndependentNormalWishartPrior
from ...exceptions import SpecificationError

__all__ = ["GibbsBVAR", "GibbsBVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class GibbsBVARResult(_VectorPosteriorDrawsResult):
    """A fitted Gibbs BVAR: a sampled posterior, and the prior's own diagnostics.

    The propagation surface -- credible intervals, impulse responses with
    posterior bands, the posterior predictive -- is the family's shared
    one. What this result adds is the adaptive prior's averaged latent
    diagnostic: inclusion probabilities under a selection prior, local
    shrinkage scales under a global-local one.

    Deliberately absent, beyond the family-wide refusals: a marginal
    likelihood. With the coefficient prior independent of the covariance
    the evidence has no closed form, so Bayes-factor prior comparison
    belongs to the conjugate model.

    Attributes:
        prior_label: Short description of the prior estimated under.
        shrinkage: ``(w, k)`` posterior mean per-coefficient diagnostic from
            an adaptive prior, in design-row order; empty under a static
            prior. Read it through :meth:`inclusion_probabilities` or
            :meth:`shrinkage_scales`.
        shrinkage_label: What ``shrinkage`` is; empty under a static prior.
        n_dummy: Artificial rows the prior contributed.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.
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
    shrinkage: npt.NDArray[np.float64] = field(repr=False)
    shrinkage_label: str
    resid: npt.NDArray[np.float64] = field(repr=False)
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    nobs: int
    n_dummy: int
    n_draws: int
    n_burn: int
    thin: int

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorGibbsFit,
        model: _GibbsBayesianVectorAutoRegressionModel[GibbsBVARResult],
    ) -> GibbsBVARResult:
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
            shrinkage=fit.shrinkage,
            shrinkage_label=fit.shrinkage_label,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            nobs=fit.nobs,
            n_dummy=fit.n_dummy,
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
        )

    def _lag_diagnostic(self) -> npt.NDArray[np.float64]:
        """The lag-block rows of ``shrinkage`` as a ``(p, k, k)`` stack."""
        offset, k = self._n_deterministic, self.k_endog
        return np.stack(
            [
                self.shrinkage[offset + lag * k : offset + (lag + 1) * k, :].T
                for lag in range(self.order)
            ]
        )

    def inclusion_probabilities(self) -> npt.NDArray[np.float64]:
        """Posterior inclusion probability of each lag coefficient.

        Entry ``[lag, i, j]`` is the posterior probability that variable
        ``j``'s coefficient at that lag in equation ``i`` sits in the slab
        -- is included -- averaged over kept sweeps
        (Rao-Blackwellized: the averaged quantity is the exact conditional
        probability, not the binary indicator).

        Returns:
            A ``(p, k, k)`` stack aligned with :attr:`coefficients`.

        Raises:
            SpecificationError: If the prior was not a selection prior --
                only a spike-and-slab posterior defines inclusion.
        """
        if "inclusion" not in self.shrinkage_label:
            raise SpecificationError(
                "inclusion probabilities exist only under a selection prior; "
                f"this model was fit under {self.prior_label}. Global-local "
                "shrinkage reports shrinkage_scales() instead."
            )
        return self._lag_diagnostic()

    def shrinkage_scales(self) -> npt.NDArray[np.float64]:
        """Posterior mean local-global scale of each standardized lag coefficient.

        Entry ``[lag, i, j]`` is the averaged prior standard deviation the
        hierarchy assigned to that coefficient on the standardized
        (unit-free) scale: near zero means shrunk away, order one means
        left free. Comparable within a fit, not across priors.

        Returns:
            A ``(p, k, k)`` stack aligned with :attr:`coefficients`.

        Raises:
            SpecificationError: If the prior was not a global-local
                shrinkage prior.
        """
        if "scale" not in self.shrinkage_label:
            raise SpecificationError(
                "shrinkage scales exist only under a global-local prior; "
                f"this model was fit under {self.prior_label}."
            )
        return self._lag_diagnostic()

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
            "The posterior is sampled by Gibbs (joint GLS coefficient draw, "
            "inverse-Wishart covariance), so draws are a Markov chain: "
            "burn-in and thinning matter, and independent-draw intuition "
            "does not apply.",
            "No marginal likelihood is reported: with the prior independent "
            "of the covariance the evidence has no closed form, and prior "
            "comparison by Bayes factor belongs to the conjugate BVAR.",
            self._stability_note(),
            "No llf, parameter count, or information criteria are reported: a posterior has none.",
            "forecast() is the full posterior predictive -- parameter and "
            "shock uncertainty jointly.",
        ]
        if self.shrinkage_label:
            if "inclusion" in self.shrinkage_label:
                included = int(np.sum(self._lag_diagnostic() > 0.5))
                total = self.order * self.k_endog**2
                notes.insert(
                    0,
                    f"Selection prior: {included} of {total} lag coefficients "
                    "carry posterior inclusion probability above one half; "
                    "inclusion_probabilities() has the full map.",
                )
            else:
                notes.insert(
                    0,
                    "Global-local prior: shrinkage_scales() maps how much "
                    "freedom each standardized lag coefficient retained.",
                )
        return SummaryTable(
            title=f"Gibbs BVAR({self.order}) Results",
            metadata=(
                ("Model", f"GibbsBVAR({self.order})"),
                ("Prior", self.prior_label),
                ("Draws", f"{self.n_kept} kept of {self.n_draws}"),
                ("Burn-in", f"{self.n_burn}"),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.nobs}"),
                ("Dummy rows", f"{self.n_dummy}"),
                ("Trend", self.trend),
            ),
            columns=("equation", "own L1 mean", "68% interval"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class GibbsBVAR(_GibbsBayesianVectorAutoRegressionModel[GibbsBVARResult]):
    """Non-conjugate Bayesian VAR: independent Normal-Wishart, sampled by Gibbs.

    One model, two families of priors the conjugate path refuses. Static
    moment priors with any diagonal variance -- the default
    :class:`~cultivars.bayes.priors.IndependentNormalWishartPrior`, which
    is Litterman's full prior with the cross-equation weight kept, or a
    :class:`~cultivars.bayes.priors.MinnesotaPrior` passed directly --
    and the adaptive shrinkage hierarchies:
    :class:`~cultivars.bayes.priors.HorseshoePrior`,
    :class:`~cultivars.bayes.priors.SpikeAndSlabPrior`,
    :class:`~cultivars.bayes.priors.DirichletLaplacePrior`,
    :class:`~cultivars.bayes.priors.NormalGammaPrior`.

    Args:
        endog: The observed panel, shape ``(nobs, k)``. A proper prior
            makes wide systems admissible, though each Gibbs sweep costs a
            Cholesky of the joint coefficient precision, so very large
            ``k`` belongs to the conjugate model.
        order: Autoregressive order.
        prior: Any finite-variance prior, static or adaptive. Defaults to
            ``IndependentNormalWishartPrior()``. Adaptive priors do not
            compose with ``+``.
        trend: Deterministic terms.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Example:
        >>> from cultivars.bayes.priors import SpikeAndSlabPrior
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((200, 3))
        >>> for t in range(1, 200):
        ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(3)
        >>> model = GibbsBVAR(y, order=1, prior=SpikeAndSlabPrior())
        >>> res = model.fit(n_draws=300, n_burn=100, seed=0)
        >>> res.inclusion_probabilities().shape
        (1, 3, 3)
        >>> bool(np.all(np.diag(res.inclusion_probabilities()[0]) > 0.5))
        True
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
        """Default the prior to the independent Normal-Wishart, then validate."""
        super().__init__(
            endog,
            order=order,
            trend=trend,
            names=names,
            prior=prior if prior is not None else IndependentNormalWishartPrior(),
        )

    def fit(
        self,
        *,
        n_draws: int = 2000,
        n_burn: int = 500,
        thin: int = 1,
        seed: int | np.random.Generator | None = None,
    ) -> GibbsBVARResult:
        """Run the Gibbs sampler and keep the post-burn draws.

        Args:
            n_draws: Total sampler iterations.
            n_burn: Burn-in iterations discarded.
            thin: Keep every ``thin``-th post-burn draw.
            seed: Seed or generator, for reproducibility.

        Returns:
            The fitted :class:`GibbsBVARResult`.

        Raises:
            SpecificationError: If the prior is improper, mixes adaptive
                components into a composition, or the draw bookkeeping is
                inconsistent.
            NumericalError: If a conditional draw collapses.
        """
        return GibbsBVARResult._from_fit(
            self._fit_gibbs(n_draws=n_draws, n_burn=n_burn, thin=thin, seed=seed),
            self,
        )
