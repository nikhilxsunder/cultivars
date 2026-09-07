# filepath: /src/cultivars/_internals/_chains.py
#
# (MIT header)
"""Markov chains over the parameters of a nonlinear state-space model.

The particle filter returns an unbiased *estimate* of the likelihood, and
Andrieu, Doucet, and Holenstein (2010) showed that plugging that estimate
into a Metropolis-Hastings acceptance ratio -- one fresh estimate per
proposal, the current one carried rather than recomputed -- leaves the
exact posterior invariant. That is the particle marginal Metropolis-Hastings
sampler implemented here, and it is the one estimator in the package that
is honest for a model outside the additive-Gaussian form: no
linearization, no mixture approximation, and the Monte Carlo error of the
likelihood estimate enters as extra stickiness of the chain rather than as
bias.

The chain moves in an unconstrained parameterization supplied by the
caller (the same reparameterizations the deterministic objectives use),
with a Gaussian random walk whose covariance adapts during burn-in
(Haario, Saksman, and Tamminen, 2001) and whose global scale is tuned
toward a target acceptance rate. Adaptation stops at the end of burn-in,
so the kept draws come from a fixed-kernel chain.

References:
    Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle Markov
        chain Monte Carlo methods. *Journal of the Royal Statistical
        Society B*, 72(3), 269-342.
    Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis
        algorithm. *Bernoulli*, 7(2), 223-242.
    Pitt, M. K., dos Santos Silva, R., Giordani, P., & Kohn, R. (2012). On
        some properties of Markov chain Monte Carlo simulation methods
        based on the particle filter. *Journal of Econometrics*, 171(2),
        134-151.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..exceptions import NumericalError, SpecificationError
from .._core import _TARGET_ACCEPTANCE, _ADAPT_FLOOR
from ._fits import _ParticleChainFit
from ._substrates import _NonlinearStateSpace


@dataclass(frozen=True, kw_only=True, slots=True)
class _ParticleMarginalChain:
    """Particle marginal Metropolis-Hastings over a nonlinear state space.

    Attributes:
        data: The observations, ``(n,)`` or ``(n, p)``, ``numpy.nan`` rows
            missing.
        build: Map from an unconstrained parameter vector to the
            :class:`_NonlinearStateSpace` it implies. It may raise
            :class:`SpecificationError` or :class:`NumericalError` for an
            inadmissible point, which the chain treats as a rejected
            proposal.
        log_prior: Log prior density *in the unconstrained space*, i.e.
            including the Jacobian of whatever reparameterization
            ``build`` inverts. ``-inf`` marks an excluded region.
        n_particles: Particles per likelihood estimate.
        method: Particle filter flavor, ``"bootstrap"`` or ``"auxiliary"``.
        ess_threshold: Resampling threshold handed to the particle filter.
    """

    data: npt.NDArray[np.float64]
    build: Callable[[npt.NDArray[np.float64]], _NonlinearStateSpace]
    log_prior: Callable[[npt.NDArray[np.float64]], float]
    n_particles: int = 1000
    method: str = "bootstrap"
    ess_threshold: float = 0.5

    def _estimate(self, theta: npt.NDArray[np.float64], rng: np.random.Generator) -> float:
        """One particle log-likelihood estimate, or ``-inf`` if inadmissible."""
        prior = float(self.log_prior(theta))
        if not np.isfinite(prior):
            return -np.inf
        try:
            model = self.build(theta)
            value = model.particle_filter(
                self.data,
                n_particles=self.n_particles,
                method=self.method,
                ess_threshold=self.ess_threshold,
                seed=rng,
            ).loglikelihood
        except (SpecificationError, NumericalError, np.linalg.LinAlgError, FloatingPointError):
            return -np.inf
        return float(value) if np.isfinite(value) else -np.inf

    def run(
        self,
        theta0: npt.NDArray[np.float64],
        *,
        n_draws: int,
        n_burn: int,
        thin: int,
        proposal_scale: npt.ArrayLike | None,
        adapt: bool,
        seed: int | np.random.Generator | None,
    ) -> _ParticleChainFit:
        """Run the chain from a starting point.

        Args:
            theta0: Starting point in the unconstrained space. Must be
                admissible: the chain refuses to start from a point whose
                likelihood estimate is not finite.
            n_draws: Total iterations.
            n_burn: Burn-in discarded, during which the proposal adapts.
            thin: Keep every ``thin``-th post-burn draw.
            proposal_scale: Initial proposal covariance ``(d, d)``, a
                per-coordinate standard deviation vector ``(d,)``, or
                ``None`` for ``0.1**2`` times the identity.
            adapt: Whether to adapt the covariance and scale during
                burn-in.
            seed: Seed or generator.

        Returns:
            The packed :class:`_ParticleChainFit`.

        Raises:
            SpecificationError: If the draw bookkeeping is inconsistent or
                the start is inadmissible.
        """
        if n_draws <= n_burn:
            raise SpecificationError(f"n_draws ({n_draws}) must exceed n_burn ({n_burn}).")
        if thin < 1:
            raise SpecificationError(f"thin must be at least 1; got {thin}.")
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        theta = np.asarray(theta0, dtype=np.float64).ravel()
        d = theta.shape[0]
        if proposal_scale is None:
            cov = np.eye(d) * 0.01
        else:
            scale = np.asarray(proposal_scale, dtype=np.float64)
            cov = np.diag(scale.ravel() ** 2) if scale.ndim == 1 else scale.copy()
        if cov.shape != (d, d):
            raise SpecificationError(
                f"proposal_scale must describe a ({d}, {d}) covariance; got shape {cov.shape}."
            )
        loglik = self._estimate(theta, rng)
        if not np.isfinite(loglik):
            raise SpecificationError(
                "the chain cannot start from a point with a non-finite likelihood "
                "estimate; supply an admissible theta0."
            )
        log_post = loglik + float(self.log_prior(theta))
        keep = (n_draws - n_burn + thin - 1) // thin
        theta_kept = np.empty((keep, d))
        loglik_kept = np.empty(keep)
        post_kept = np.empty(keep)
        kept = 0
        accepted = 0
        log_scale = 0.0
        running_mean = theta.copy()
        running_cov = np.zeros((d, d))
        chol = np.linalg.cholesky(cov)
        for iteration in range(n_draws):
            proposal = theta + np.exp(log_scale) * (chol @ rng.standard_normal(d))
            prop_loglik = self._estimate(proposal, rng)
            if np.isfinite(prop_loglik):
                prop_post = prop_loglik + float(self.log_prior(proposal))
                alpha = min(1.0, float(np.exp(min(prop_post - log_post, 0.0))))
                if rng.random() < alpha:
                    theta, loglik, log_post = proposal, prop_loglik, prop_post
                    accepted += 1
            else:
                alpha = 0.0
            if adapt and iteration < n_burn:
                count = iteration + 1
                delta = theta - running_mean
                running_mean = running_mean + delta / count
                update = np.outer(delta, theta - running_mean) - running_cov
                running_cov = running_cov + update / count
                log_scale += (alpha - _TARGET_ACCEPTANCE) / np.sqrt(count)
                if count >= max(20, 2 * d):
                    adapted = (2.38**2 / d) * running_cov + _ADAPT_FLOOR * np.eye(d)
                    with contextlib.suppress(np.linalg.LinAlgError):
                        chol = np.linalg.cholesky(adapted)
            if iteration >= n_burn and (iteration - n_burn) % thin == 0:
                theta_kept[kept] = theta
                loglik_kept[kept] = loglik
                post_kept[kept] = log_post
                kept += 1
        final_cov = np.exp(2.0 * log_scale) * (chol @ chol.T)
        return _ParticleChainFit(
            theta_draws=theta_kept,
            loglik_draws=loglik_kept,
            log_posterior_draws=post_kept,
            acceptance_rate=accepted / n_draws,
            proposal_cov=final_cov,
            n_particles=self.n_particles,
            n_draws=n_draws,
            n_burn=n_burn,
            thin=thin,
        )
