# filepath: /src/cultivars/engine/_internals/_chains.py
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

Admissibility is enforced by rejection rather than by exception. A proposal
whose prior is not finite, whose system cannot be built, or whose particle
cloud degenerates is given a log likelihood of ``-inf`` and rejected, so a
long run is not lost to one bad draw and the chain never leaves the region
where the model is defined. The cost is that a misconfigured filter is
indistinguishable from an inadmissible point: it fails at the starting
point, and the run refuses to begin.

Two tuning facts follow from the likelihood being an estimate. The number
of particles does not change what the chain converges to, only how well it
mixes, and Pitt et al. (2012) put the efficient choice where the standard
deviation of the log-likelihood estimate is about one. And the acceptance
rate has a ceiling below one that no step size can lift, set by that same
noise. The tuning target of ``0.234`` (Roberts, Gelman, and Gilks, 1997) is
the optimum for an exactly evaluated likelihood, so it is reachable only
when the estimate is reasonably tight; Sherlock et al. (2015) find the
efficient rate for a pseudo-marginal random walk to be far lower, near 7%.

Layout. :class:`_ParticleMarginalChain` is the sampler: a frozen record of
the data, the two callables, and the filter settings, with ``run`` as its
one entry point and ``_estimate`` as the likelihood kernel. It returns the
raw draws as a :class:`~cultivars.engine._internals._fits._ParticleChainFit`;
the posterior records that summarize them live in ``_posteriors``. The
filter is :meth:`_NonlinearStateSpace.particle_filter` in ``_substrates``.
The stochastic-volatility and perturbation-DSGE engines in ``_models`` are
the two callers.

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

    Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence
    and optimal scaling of random walk Metropolis algorithms. *The
    Annals of Applied Probability*, 7(1), 110-120.

    Sherlock, C., Thiery, A. H., Roberts, G. O., & Rosenthal, J. S. (2015).
    On the efficiency of pseudo-marginal random walk Metropolis
    algorithms. *The Annals of Statistics*, 43(1), 238-275.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...exceptions import NumericalError, SpecificationError
from .._core._defaults import _ADAPT_FLOOR, _TARGET_ACCEPTANCE
from ._fits import _ParticleChainFit
from ._substrates import _NonlinearStateSpace

__all__ = [
    "_ParticleMarginalChain",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class _ParticleMarginalChain:
    """Particle marginal Metropolis-Hastings over a nonlinear state space.

    A random-walk Metropolis chain on an unconstrained parameter vector
    ``theta`` whose target is the posterior ``p(theta | y)``, with the
    intractable likelihood replaced by a particle-filter estimate. The
    estimate at the current point is carried from the iteration that
    accepted it and never refreshed; each proposal gets exactly one fresh
    estimate. That bookkeeping is what makes the chain exact for any number
    of particles (Andrieu, Doucet, and Holenstein, 2010): the particle count
    governs how well the chain mixes, not what it converges to.

    The record holds the problem only: the data, the map from ``theta`` to a
    system, the log prior, and the filter settings. Everything about a
    particular run (length, burn-in, thinning, proposal, seed) is an argument
    of :meth:`run`, so one chain can be run repeatedly from different starts
    or seeds. Construction is keyword-only and validates nothing; a
    malformed filter setting is first seen by the particle filter inside
    :meth:`run`.

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

    Note:
        The chain works in whatever unconstrained space ``build`` and
        ``log_prior`` agree on, and returns its draws in that space. Mapping
        them back to the model's natural parameters is the caller's job.

    See Also:
        * :class:`_ParticleChainFit` -- the raw draws :meth:`run` returns.
        * :meth:`_NonlinearStateSpace.particle_filter` -- the likelihood
          estimator, and the meaning of ``method`` and ``ess_threshold``.

    Example:
        An AR(1) signal observed in noise. The one parameter is the
        persistence, mapped to ``(-1, 1)`` through ``tanh``, with a standard
        normal prior on the unconstrained coordinate:

        >>> import numpy as np
        >>> from cultivars.engine._internals._substrates import _NonlinearStateSpace
        >>> rng = np.random.default_rng(0)
        >>> signal = np.zeros(60)
        >>> for t in range(1, 60):
        ...     signal[t] = 0.8 * signal[t - 1] + 0.5 * rng.standard_normal()
        >>> y = signal + 0.5 * rng.standard_normal(60)
        >>> def build(theta):
        ...     phi = float(np.tanh(theta[0]))
        ...     return _NonlinearStateSpace(
        ...         lambda a: phi * a,
        ...         lambda a: a,
        ...         state_cov=[[0.25]],
        ...         obs_cov=[[0.25]],
        ...         initial_state=[0.0],
        ...         initial_state_cov=[[1.0]],
        ...     )
        >>> chain = _ParticleMarginalChain(
        ...     data=y,
        ...     build=build,
        ...     log_prior=lambda theta: float(-0.5 * theta @ theta),
        ...     n_particles=100,
        ... )
        >>> settings = dict(n_draws=60, n_burn=20, thin=3, proposal_scale=None, adapt=True)
        >>> fit = chain.run(np.array([0.5]), seed=0, **settings)
        >>> fit.theta_draws.shape, fit.n_kept, fit.proposal_cov.shape
        ((14, 1), 14, (1, 1))
        >>> 0.0 <= fit.acceptance_rate <= 1.0
        True

        The same integer seed reproduces the run exactly:

        >>> again = chain.run(np.array([0.5]), seed=0, **settings)
        >>> bool(np.array_equal(fit.theta_draws, again.theta_draws))
        True
    """

    data: npt.NDArray[np.float64]
    """The observations, ``(n,)`` or ``(n, p)``; a ``numpy.nan`` row is a missing period.

    Handed unchanged to the particle filter on every likelihood estimate,
    which coerces and validates it each time. A 1-D series is accepted only
    when the system has a single observable. Missingness is whole-row: the
    filter refuses a row with some but not all entries missing.
    """
    build: Callable[[npt.NDArray[np.float64]], _NonlinearStateSpace]
    """Map from an unconstrained ``(d,)`` parameter vector to the system it implies.

    Called once per likelihood estimate, so its cost is paid on every
    proposal. A :class:`SpecificationError`, :class:`NumericalError`,
    ``numpy.linalg.LinAlgError``, or ``FloatingPointError`` raised here
    marks the point inadmissible and rejects the proposal; any other
    exception, :class:`DimensionError` included, stops the chain.
    """
    log_prior: Callable[[npt.NDArray[np.float64]], float]
    """Log prior density in the unconstrained space, Jacobian included.

    The chain targets this density times the likelihood exactly as given, so
    the log-determinant of the reparameterization that ``build`` inverts
    belongs here; leaving it out samples a different posterior without any
    error. It need not be normalized. Any non-finite value excludes the
    point, ``nan`` and ``+inf`` as well as ``-inf``. Evaluated twice at
    every admissible point.
    """
    n_particles: int = 1000
    """Particles per likelihood estimate; at least 2.

    Sets how noisy each estimate is and therefore how sticky the chain is,
    not what it converges to; the cost of an iteration is linear in it. Half
    the particle filter's own default of 2000.
    """
    method: str = "bootstrap"
    """Particle filter flavor, ``"bootstrap"`` or ``"auxiliary"``.

    Not checked at construction. An unrecognized value makes every estimate
    fail, and :meth:`run` then refuses to start.
    """
    ess_threshold: float = 0.5
    """Resample when the effective sample size falls below this fraction of the cloud.

    Must lie in ``(0, 1]``. Read by the bootstrap filter only; the auxiliary
    filter resamples every period by construction.
    """

    def _estimate(self, theta: npt.NDArray[np.float64], rng: np.random.Generator) -> float:
        """One particle log-likelihood estimate, or ``-inf`` if inadmissible.

        The likelihood kernel of the chain. The prior is evaluated first and
        the filter is skipped when it is not finite; otherwise the system is
        built at ``theta`` and one particle filter is run over the data with
        a fresh cloud drawn from ``rng``. The result is the logarithm of an
        unbiased estimate of ``p(y | theta)``. It is random, so two calls at
        the same ``theta`` differ, and :meth:`run` depends on calling it
        exactly once per proposal.

        The prior decides admissibility here but is not added to the return
        value; :meth:`run` adds it when forming the log posterior.

        A point is inadmissible, and ``-inf`` is returned, when any of the
        following holds:

        * the log prior is not finite (``-inf``, ``nan``, or ``+inf``);
        * ``build`` or the filter raises :class:`SpecificationError`,
          :class:`NumericalError`, ``numpy.linalg.LinAlgError``, or
          ``FloatingPointError``;
        * the estimate itself is not finite.

        The filter validates its own arguments with
        :class:`SpecificationError`, so an unrecognized ``method``, an
        ``n_particles`` below 2, an ``ess_threshold`` outside ``(0, 1]``, or
        a partially observed data row falls under the second rule and makes
        every point inadmissible. Exceptions outside that list,
        :class:`DimensionError` among them, propagate.

        Args:
            theta: Point in the unconstrained space, shape ``(d,)``.
            rng: Generator the particle filter draws from. Advanced in
                place, except when the point is excluded before the filter
                runs.

        Returns:
            The particle estimate of the log likelihood at ``theta``, or
            ``-inf`` if the point is inadmissible.
        """
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

        Each iteration proposes ``theta + s * L @ z`` with ``z`` standard
        normal, ``L`` the Cholesky factor of the current proposal covariance,
        and ``s`` the global scale. It estimates the likelihood there with a
        fresh particle cloud and accepts with probability
        ``min(1, exp(log_post_new - log_post))``, where ``log_post`` is the
        value *carried* with the current point. An inadmissible proposal is
        rejected outright.

        With ``adapt=True`` two things move during burn-in and are frozen
        after it:

        * the scale: after iteration ``i`` (counting from one), ``log s``
          changes by ``(alpha - 0.234) / sqrt(i)``, with ``alpha`` the
          acceptance probability of that proposal (zero if inadmissible);
        * the shape: once ``max(20, 2 * d)`` iterations have run, the
          covariance becomes ``2.38**2 / d`` times the running covariance of
          the chain's states so far, plus ``1e-8`` on the diagonal. If that
          matrix cannot be factorized, the previous factor is kept.

        The scale is not reset when the shape first switches from the
        initial covariance to the adapted one. A burn-in shorter than
        ``max(20, 2 * d)`` therefore adapts the scale alone, and
        ``n_burn=0`` adapts nothing.

        ``n_draws`` counts every iteration, burn-in included. The kept draws
        are the states after iterations ``n_burn``, ``n_burn + thin``, and so
        on (counting from zero), ``ceil((n_draws - n_burn) / thin)`` of them;
        the starting point itself is never among them.

        All randomness (proposals, acceptance draws, and every particle
        cloud) comes from one generator, so an integer ``seed`` reproduces
        the run exactly. A generator passed as ``seed`` is advanced in place.

        Args:
            theta0: Starting point in the unconstrained space; flattened to
                ``(d,)``. Must be admissible: the chain refuses to start
                from a point whose prior or likelihood estimate is not
                finite.
            n_draws: Total iterations, burn-in included. Must exceed
                ``n_burn``.
            n_burn: Burn-in discarded, during which the proposal adapts.
                Must be non-negative. This is not validated: a negative
                value returns draw arrays whose trailing rows were never
                written.
            thin: Keep every ``thin``-th post-burn draw; at least 1.
            proposal_scale: Initial proposal covariance ``(d, d)``, a
                per-coordinate standard deviation vector ``(d,)``, or
                ``None`` for ``0.1**2`` times the identity. Must be positive
                definite.
            adapt: Whether to adapt the covariance and scale during
                burn-in. If ``False``, the initial proposal is used
                unchanged throughout.
            seed: Seed or generator.

        Returns:
            The packed :class:`_ParticleChainFit`: the kept draws with the
            log-likelihood estimate and log posterior carried at each, the
            acceptance rate over all ``n_draws`` iterations (burn-in
            included), and the proposal covariance the chain ended with,
            global scale folded in.

        Raises:
            SpecificationError: If ``n_draws <= n_burn``, if ``thin < 1``,
                if ``proposal_scale`` does not describe a ``(d, d)``
                covariance, or if the start is inadmissible. The last is
                also how a malformed filter setting on the chain surfaces,
                because :meth:`_estimate` absorbs the filter's own error.
            numpy.linalg.LinAlgError: If the initial proposal covariance is
                not positive definite, a zero standard deviation included.
            DimensionError: If ``data`` does not match the observation
                dimension of the system built at ``theta0``.

        Note:
            The acceptance rate of a chain driven by a likelihood estimate
            has a ceiling below one, however small the step. If the estimate
            is noisy enough that the ceiling lies under the ``0.234`` target,
            the scale shrinks throughout burn-in and the chain barely moves
            afterwards. The symptoms are a very small ``proposal_cov`` and
            long runs of repeated draws; the remedy is more particles, not a
            longer burn-in.

        Example:
            See the class example, which builds a chain and runs it twice.
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
