# filepath: /src/cultivars/engine/_internals/_posteriors.py
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
"""Posterior records: what a sampler returns once its draws are packed.

A sampler in this package produces a *fit* record of raw draws (``_fits``).
A posterior record is that fit made usable: the draws under the names a
user reads, the bookkeeping of the run, and every derived quantity as a
property or method of the draws. Posterior means, pointwise bands, the
parameter record at the posterior mean, the state-space model that mean
implies, replicated and simulated data, and the summary table all live
here. Nothing in this module runs a sampler.

There are four records:

* :class:`_ConjugatePosterior`: the Normal-inverse-Wishart posterior of a
  conjugate Bayesian VAR, in closed form. It holds no draws, only the four
  quantities that define the distribution and its exact log marginal
  likelihood.
* :class:`_SVPosterior`: the log-AR(1) stochastic-volatility model, from
  the Kim-Shephard-Chib Gibbs sampler or from the particle chain.
* :class:`_UCSVPosterior`: the unobserved-components stochastic-volatility
  model of Stock and Watson, from its Gibbs sampler.
* :class:`_PerturbationDSGEPosterior`: a perturbation-solved DSGE model,
  from the particle chain.

Three conventions hold across the draw-based records.

Naming. Every field whose name ends in ``_draws`` carries the kept draws on
its leading axis. The suffix is load-bearing: the convergence mixin finds
its arrays by it, and drops any whose leading axis is shorter than the
longest, which is how the single smoothed path of the particle-chain
stochastic-volatility posterior stays out of the diagnostics.

Absence. A posterior has no ``llf``, no ``n_params``, and no information
criteria, and these records do not pretend otherwise. Where a likelihood
appears it is a sequence of particle estimates carried with the draws.

Plug-in summaries. A point summary is the mean of a draw array. The model
"at the posterior mean" (``params``, ``state_space``, ``solution``, and
``simulate`` without a ``draw``) plugs those means into the model, which
is not the posterior expectation of anything nonlinear in the parameters.
The methods that take a draw index, and the replication and
impulse-response methods that loop over draws, are the ones that carry
parameter uncertainty.

Layout. The three draw-based records are re-exported under public names:
``SVPosterior`` and ``UCSVPosterior`` from
:mod:`cultivars.univariate.stochastic_volatility`, and
``PerturbationDSGEPosterior`` from
:mod:`cultivars.multivariate.structural.perturbation`. The conjugate record
is built by the Normal-inverse-Wishart update in ``_solvers``. The summary,
series, and convergence behavior comes from the mixins in ``_mixins``; the
state-space emitters are in ``_emitters`` and the simulators in
``_simulators``.

References:
    Geweke, J. (1999). Using simulation methods for Bayesian econometric
    models: Inference, development, and communication. *Econometric
    Reviews*, 18(1), 1-73.

    Herbst, E. P., & Schorfheide, F. (2015). *Bayesian Estimation of DSGE
    Models*. Princeton University Press.

    Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
    Likelihood inference and comparison with ARCH models. *The Review
    of Economic Studies*, 65(3), 361-393.

    Stock, J. H., & Watson, M. W. (2007). Why has U.S. inflation become
    harder to forecast? *Journal of Money, Credit and Banking*, 39(s1),
    3-33.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ...exceptions import NumericalError, SpecificationError
from .._core._containers import _SummaryTable
from .._core._defaults import _MHM_TAU, _SIMULATION_BURN
from .._core._estimators import _modified_harmonic_mean
from .._core._matrices import _quantiles
from ._emitters import (
    _trend_volatility_state_space,
    _volatility_state_space,
)
from ._fits import _ParticleChainFit, _TrendVolatilityFit, _VolatilityDrawsFit
from ._mixins import _ConvergenceMixin, _SeriesMixin, _SummaryMixin
from ._parameters import _StochasticVolatilityParameters, _TrendVolatilityParameters
from ._selections import _MarginalLikelihoodSelection
from ._simulators import (
    _impulse_responses,
    _simulate_perturbation,
    _simulate_stochastic_volatility,
)
from ._solutions import _PerturbationSolution
from ._substrates import _LinearGaussianStateSpace, _NonlinearStateSpace

if TYPE_CHECKING:
    from ._models import (
        _PerturbationModel,
        _StochasticVolatilityModel,
        _TrendVolatilityModel,
    )

__all__ = [
    "_ConjugatePosterior",
    "_PerturbationDSGEPosterior",
    "_SVPosterior",
    "_UCSVPosterior",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class _ConjugatePosterior:
    """Joint Normal-inverse-Wishart posterior over ``(B, Sigma)``.

    The conjugate counterpart of :class:`_PosteriorCovariance`, and it holds
    Kronecker factors again because conjugacy is exactly the condition under
    which they exist: with the prior variance factoring as ``Sigma x Omega``,
    the posterior of the coefficient matrix given ``Sigma`` is matrix normal
    with row precision ``K = Omega^-1 + X'X`` shared by every equation, and
    ``Sigma`` itself is inverse-Wishart. Cross-equation posterior dependence
    -- what the per-equation record cannot carry -- lives in that product.

    Attributes:
        coefficients: The ``(width, k)`` posterior mean, in design-column
            order.
        row_precision: The ``(width, width)`` shared row precision ``K``.
        scale: The ``(k, k)`` inverse-Wishart posterior scale.
        df: Inverse-Wishart posterior degrees of freedom.
        log_ml: Log marginal likelihood of the rows this posterior was
            updated on, given the prior -- exact, from the matrix-variate-t
            form.

    In symbols, with ``B`` the ``(width, k)`` coefficient matrix::

        B | Sigma ~ MN(coefficients, row_precision^-1, Sigma)
        Sigma ~ IW(scale, df)

    so the covariance of ``vec(B)`` given ``Sigma`` is the Kronecker product
    of ``Sigma`` with the inverse of ``row_precision``.

    Note:
        The record is a plain container. It is built by the conjugate
        update in ``_solvers`` and validates nothing itself. As with every
        frozen record that holds arrays, ``==`` raises ``ValueError`` and
        ``hash`` raises ``TypeError``.

    Example:
        >>> import numpy as np
        >>> posterior = _ConjugatePosterior(
        ...     coefficients=np.zeros((3, 2)),
        ...     row_precision=np.eye(3),
        ...     scale=6.0 * np.eye(2),
        ...     df=5.0,
        ...     log_ml=-1.0,
        ... )
        >>> posterior.sigma_mean.tolist()
        [[3.0, 0.0], [0.0, 3.0]]
    """

    coefficients: npt.NDArray[np.float64]
    """Posterior mean of the coefficient matrix, ``(width, k)``, in design-column order.

    Rows are design columns and columns are equations, the layout every
    prior uses. It is the mean of ``B`` given any ``Sigma``, and so also the
    marginal posterior mean.
    """
    row_precision: npt.NDArray[np.float64]
    """Row precision ``K = Omega^-1 + X'X`` shared by every equation, ``(width, width)``.

    Symmetric. Its inverse, scaled by ``Sigma``, is the covariance of the
    coefficients given ``Sigma``.
    """
    scale: npt.NDArray[np.float64]
    """Inverse-Wishart posterior scale matrix, ``(k, k)``.

    Divided by ``df - k - 1`` it is the posterior mean of ``Sigma``; see
    :attr:`sigma_mean`.
    """
    df: float
    """Inverse-Wishart posterior degrees of freedom: the prior's plus the rows updated on."""
    log_ml: float
    """Log marginal likelihood of the rows this posterior was updated on, given the prior.

    Exact, from the matrix-variate-t form; no simulation is involved.
    """

    @property
    def sigma_mean(self) -> npt.NDArray[np.float64]:
        """Posterior mean of the innovation covariance.

        Returns:
            ``scale / (df - k - 1)``, shape ``(k, k)``.

        Raises:
            NumericalError: If the degrees of freedom do not support a mean.
        """
        k = int(self.scale.shape[0])
        if self.df <= k + 1:
            raise NumericalError(
                f"the inverse-Wishart mean needs df > k + 1; got df={self.df} with k={k}."
            )
        return self.scale / (self.df - k - 1.0)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _SVPosterior(_SummaryMixin, _SeriesMixin, _ConvergenceMixin):
    """A posterior over the stochastic-volatility model, from either sampler.

    Deliberately absent: ``llf``, ``n_params``, information criteria. A
    posterior has none of them; :attr:`loglik_draws` (particle chain only)
    is the sequence of likelihood *estimates* the chain accepted.

    Attributes:
        endog: The observed series.
        mean_spec: ``"constant"`` or ``"zero"``.
        method: ``"gibbs"`` or ``"particle"``.
        mu_draws: ``(S,)`` draws of the log-variance mean.
        phi_draws: ``(S,)`` draws of the persistence.
        sigma2_draws: ``(S,)`` draws of the innovation variance.
        mean_draws: ``(S,)`` draws of the observation mean (constant under
            the particle chain, which concentrates it at the sample mean).
        h_draws: ``(S, n)`` log-variance path draws (Gibbs) or, for the
            particle chain, the particle-smoothed path at the posterior
            mean repeated once, ``(1, n)``.
        nu_draws: ``(S,)`` draws of the Student-t degrees of freedom, or
            ``None`` under Gaussian noise.
        rho_draws: ``(S,)`` draws of the leverage correlation, or ``None``
            without leverage.
        loglik_draws: ``(S,)`` particle likelihood estimates, or ``None``.
        acceptance_rate: The chain's acceptance rate, or ``None``.
        nobs: Observations.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.

    The model is ``y_t = c + exp(h_t / 2) eps_t`` with
    ``h_{t+1} = mu + phi (h_t - mu) + sigma eta_t``. The two samplers fill
    the record differently, and these fields mean different things under
    each:

    * ``h_draws``: under Gibbs, one sampled log-variance path per kept
      draw. Under the particle chain, a single path, the particle-smoothed
      state at the posterior-mean parameters, so everything derived from it
      (``volatility_quantiles``, the bands in ``to_pandas``) carries no
      path uncertainty.
    * ``mean_draws``: sampled under Gibbs when the mean is estimated.
      Constant under the particle chain, which holds the mean fixed at the
      sample mean or at zero.
    * ``acceptance_rate``: under the particle chain, the chain's acceptance
      rate. Under Gibbs, the acceptance rate of the Metropolis step for the
      degrees of freedom, and ``None`` when the noise is Gaussian.
    * ``loglik_draws`` and ``rho_draws``: particle chain only.

    Point summaries are means over the kept draws. ``params``,
    ``state_space``, and ``simulate`` without a ``draw`` plug those means
    into the model.

    See Also:
        * :class:`~cultivars.engine._internals._fits._VolatilityDrawsFit`
          and :class:`~cultivars.engine._internals._fits._ParticleChainFit`
          -- the raw sampler outputs this record is assembled from.
        * :class:`_UCSVPosterior` -- the random-walk-volatility counterpart.

    Example:
        >>> import numpy as np
        >>> from cultivars.engine._internals._simulators import _simulate_stochastic_volatility
        >>> from cultivars.univariate.stochastic_volatility import SV
        >>> rng = np.random.default_rng(0)
        >>> y, _ = _simulate_stochastic_volatility(
        ...     150, mu=-1.0, phi=0.95, sigma2=0.05, mean=0.0, rng=rng
        ... )
        >>> post = SV(y, mean="zero").sample(n_draws=120, n_burn=40, seed=0)
        >>> isinstance(post, _SVPosterior), post.method, post.n_kept
        (True, 'gibbs', 80)
        >>> post.h_draws.shape, post.volatility.shape
        ((80, 150), (150,))
        >>> post.posterior_replications(5, seed=0).shape
        (5, 150)
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    """The observed series, ``(n,)``; kept out of the repr."""
    mean_spec: str
    """Observation-mean specification, ``"constant"`` or ``"zero"``."""
    method: str
    """Sampler that produced the draws, ``"gibbs"`` or ``"particle"``.

    Decides how ``h_draws``, ``mean_draws``, and ``acceptance_rate`` are to
    be read; see the class docstring.
    """
    mu_draws: npt.NDArray[np.float64] = field(repr=False)
    """Kept draws of the log-variance mean ``mu``, ``(S,)``."""
    phi_draws: npt.NDArray[np.float64] = field(repr=False)
    """Kept draws of the persistence ``phi``, ``(S,)``, each in ``(-1, 1)``."""
    sigma2_draws: npt.NDArray[np.float64] = field(repr=False)
    """Kept draws of the log-variance innovation variance ``sigma**2``, ``(S,)``."""
    mean_draws: npt.NDArray[np.float64] = field(repr=False)
    """Draws of the observation mean ``c``, ``(S,)``.

    Sampled under Gibbs with a constant mean. All zero when the mean is
    fixed at zero, and constant at the sample mean under the particle
    chain.
    """
    h_draws: npt.NDArray[np.float64] = field(repr=False)
    """Log-variance paths: ``(S, n)`` under Gibbs, ``(1, n)`` under the particle chain.

    The particle-chain row is the smoothed path at the posterior-mean
    parameters, not a draw. The convergence diagnostics skip it because its
    leading axis is not the kept-draw count.
    """
    nu_draws: npt.NDArray[np.float64] | None = field(repr=False)
    """Kept draws of the Student-t degrees of freedom, ``(S,)``; ``None`` for Gaussian noise."""
    rho_draws: npt.NDArray[np.float64] | None = field(repr=False)
    """Kept draws of the leverage correlation, ``(S,)``; ``None`` without leverage.

    Only the particle chain estimates leverage, so this is always ``None``
    under Gibbs.
    """
    loglik_draws: npt.NDArray[np.float64] | None = field(repr=False)
    """Particle log-likelihood estimates, one per kept draw, ``(S,)``; ``None`` under Gibbs.

    The estimates the chain accepted, each from one particle filter. They
    are not exact likelihoods.
    """
    acceptance_rate: float | None
    """Acceptance rate of whichever Metropolis step the sampler has, or ``None``.

    Particle chain: the chain's rate over the whole run, burn-in included.
    Gibbs: the rate of the step for the degrees of freedom, and ``None``
    when the noise is Gaussian and there is no such step.
    """
    nobs: int
    """Number of observations ``n``."""
    n_draws: int
    """Total sampler iterations, burn-in included."""
    n_burn: int
    """Iterations discarded as burn-in."""
    thin: int
    """Thinning interval applied after burn-in."""

    @classmethod
    def _from_gibbs(
        cls, fit: _VolatilityDrawsFit, model: _StochasticVolatilityModel[Any]
    ) -> _SVPosterior:
        """Assemble from the Gibbs sampler's output.

        The draw arrays are taken from the fit as they are, without copying.
        ``rho_draws`` and ``loglik_draws`` are ``None``, and
        ``acceptance_rate`` is the fit's acceptance rate for the
        degrees-of-freedom step.

        Args:
            fit: The packed Gibbs output.
            model: The engine that ran the sampler; supplies the series and
                the mean specification.

        Returns:
            The posterior record with ``method="gibbs"``.
        """
        return cls(
            endog=model.endog,
            mean_spec=model.mean_spec,
            method="gibbs",
            mu_draws=fit.mu_draws,
            phi_draws=fit.phi_draws,
            sigma2_draws=fit.sigma2_draws,
            mean_draws=fit.mean_draws,
            h_draws=fit.h_draws,
            nu_draws=fit.nu_draws,
            rho_draws=None,
            loglik_draws=None,
            acceptance_rate=fit.nu_acceptance,
            nobs=fit.nobs,
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
        )

    @classmethod
    def _from_chain(
        cls,
        fit: _ParticleChainFit,
        mean: float,
        model: _StochasticVolatilityModel[Any],
        *,
        n_particles: int,
        seed: int | None,
    ) -> _SVPosterior:
        """Assemble from the particle chain's output.

        Maps the chain's unconstrained coordinates back to the model's
        parameters: ``mu`` as it is, ``phi = tanh(.)``, ``sigma2 = exp(.)``,
        then ``nu = 2 + exp(.)`` if the model is heavy-tailed and
        ``rho = tanh(.)`` if it carries leverage, in that order. The path
        block is then filled by one particle-smoother run at the
        posterior-mean parameters.

        Args:
            fit: The packed chain output, draws in the unconstrained space.
            mean: The observation mean the chain held fixed.
            model: The engine that ran the chain.
            n_particles: Particle count of the chain. The smoother uses
                ``min(n_particles, 500)``.
            seed: Seed for the smoother; the caller passes the chain's.

        Returns:
            The posterior record with ``method="particle"``, a constant
            ``mean_draws``, and a ``(1, n)`` ``h_draws``.
        """
        theta = fit.theta_draws
        mu = theta[:, 0]
        phi = np.tanh(theta[:, 1])
        sigma2 = np.exp(theta[:, 2])
        index = 3
        nu = None
        rho = None
        if model.heavy_tailed:
            nu = 2.0 + np.exp(theta[:, index])
            index += 1
        if model.leveraged:
            rho = np.tanh(theta[:, index])
        params = _StochasticVolatilityParameters(
            mu=float(mu.mean()),
            phi=float(phi.mean()),
            sigma2=float(sigma2.mean()),
            mean=mean,
            nu=float(nu.mean()) if nu is not None else None,
            rho=float(rho.mean()) if rho is not None else 0.0,
        )
        smoothed = _volatility_state_space(params).particle_smoother(
            model.endog, n_particles=min(n_particles, 500), seed=seed
        )
        return cls(
            endog=model.endog,
            mean_spec=model.mean_spec,
            method="particle",
            mu_draws=np.asarray(mu, dtype=np.float64),
            phi_draws=np.asarray(phi, dtype=np.float64),
            sigma2_draws=np.asarray(sigma2, dtype=np.float64),
            mean_draws=np.full(theta.shape[0], mean),
            h_draws=np.asarray(smoothed.smoothed_state[:, 0], dtype=np.float64).reshape(1, -1),
            nu_draws=np.asarray(nu, dtype=np.float64) if nu is not None else None,
            rho_draws=np.asarray(rho, dtype=np.float64) if rho is not None else None,
            loglik_draws=fit.loglik_draws,
            acceptance_rate=fit.acceptance_rate,
            nobs=int(model.endog.shape[0]),
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
        )

    @property
    def n_kept(self) -> int:
        """Posterior draws retained.

        Returns:
            The length ``S`` of the parameter draw arrays.
        """
        return int(self.mu_draws.shape[0])

    def _convergence_label(self) -> str:
        """Title of the convergence report, naming the specification."""
        return f"SV[{self._spec_label()}] posterior"

    def _spec_label(self) -> str:
        """``mean[, t][, leverage]`` for titles.

        Returns:
            The mean specification, followed by ``"t"`` when the degrees of
            freedom were sampled and ``"leverage"`` when the correlation was.
        """
        parts = [self.mean_spec]
        if self.nu_draws is not None:
            parts.append("t")
        if self.rho_draws is not None:
            parts.append("leverage")
        return ", ".join(parts)

    @property
    def params(self) -> _StochasticVolatilityParameters:
        """The posterior-mean parameter record.

        Each entry is the mean of its draw array. ``nu`` is ``None`` and
        ``rho`` is ``0.0`` when the posterior does not carry them.

        Returns:
            A :class:`_StochasticVolatilityParameters` at the posterior
            means.
        """
        return _StochasticVolatilityParameters(
            mu=float(self.mu_draws.mean()),
            phi=float(self.phi_draws.mean()),
            sigma2=float(self.sigma2_draws.mean()),
            mean=float(self.mean_draws.mean()),
            nu=float(self.nu_draws.mean()) if self.nu_draws is not None else None,
            rho=float(self.rho_draws.mean()) if self.rho_draws is not None else 0.0,
        )

    @property
    def log_variance(self) -> npt.NDArray[np.float64]:
        """Posterior-mean log-variance path, ``(n,)``.

        Returns:
            The mean of ``h_draws`` over its leading axis. Under the
            particle chain that is the single smoothed path.
        """
        return np.asarray(self.h_draws.mean(axis=0), dtype=np.float64)

    @property
    def volatility(self) -> npt.NDArray[np.float64]:
        """Posterior-mean volatility ``E[exp(h_t / 2) | y]``, ``(n,)``.

        Returns:
            The mean over draws of ``exp(h / 2)``. By Jensen's inequality it
            is not ``exp(log_variance / 2)`` unless there is a single path.
        """
        return np.asarray(np.exp(0.5 * self.h_draws).mean(axis=0), dtype=np.float64)

    def volatility_quantiles(
        self, levels: tuple[float, ...] = (0.05, 0.5, 0.95)
    ) -> npt.NDArray[np.float64]:
        """Pointwise posterior quantiles of the volatility, ``(len(levels), n)``.

        Under the particle chain the path block holds one smoothed path,
        so the quantiles collapse to it; the parameter draws carry the
        uncertainty there.

        Args:
            levels: Probabilities in ``[0, 1]``, one row of the result each.

        Returns:
            The pointwise quantiles of ``exp(h / 2)`` across draws.
        """
        return _quantiles(np.exp(0.5 * self.h_draws), levels)

    @property
    def state_space(self) -> _NonlinearStateSpace:
        """The model at the posterior mean, on the nonlinear substrate.

        Returns:
            The state-space model built from :attr:`params`. It is rebuilt
            on every access.
        """
        return _volatility_state_space(self.params)

    @property
    def observed(self) -> npt.NDArray[np.float64]:
        """The series the replications are shaped like, ``(n,)``.

        Returns:
            The observed series itself, not a copy.
        """
        return np.asarray(self.endog, dtype=np.float64)

    def _replication_notes(self) -> tuple[str, ...]:
        """Remarks a predictive check carries about how this posterior replicates.

        Returns:
            One note, stating that each replication simulates a fresh
            volatility path and does not condition on the smoothed one.
        """
        return (
            "Replications simulate a fresh log-variance path from each draw's (mu, phi, "
            "sigma2), started at its stationary distribution: the check reads the whole "
            "model, volatility law included, not the observation equation given the "
            "smoothed path.",
        )

    def posterior_replications(
        self,
        n_replications: int = 200,
        *,
        seed: int | np.random.Generator | None = None,
    ) -> npt.NDArray[np.float64]:
        """Replicated series, one per posterior draw used.

        Each replication takes one kept draw of ``(mu, phi, sigma2, mean)``
        -- and of ``nu`` or ``rho`` where the posterior carries them -- simulates a fresh
        log-variance path from its stationary start, and the series from
        it. Draws are taken evenly across the kept sequence when fewer
        replications than draws are asked for.

        When more replications than kept draws are asked for, draws are
        reused. The persistence of each draw is clipped to ``0.9999`` in
        absolute value before simulating.

        Args:
            n_replications: Replicated series.
            seed: Seed or generator.

        Returns:
            An array of shape ``(n_replications, n)`` aligned with
            :attr:`observed`.

        Raises:
            SpecificationError: If the count is not positive.
        """
        if n_replications < 1:
            raise SpecificationError(f"n_replications must be positive; got {n_replications}.")
        rng = np.random.default_rng(seed)
        n = int(self.endog.shape[0])
        indices = np.linspace(0, self.n_kept - 1, n_replications).round().astype(int)
        out = np.empty((n_replications, n))
        for r, draw in enumerate(indices):
            phi = float(np.clip(self.phi_draws[draw], -0.9999, 0.9999))
            nu = float(self.nu_draws[draw]) if self.nu_draws is not None else None
            rho = float(self.rho_draws[draw]) if self.rho_draws is not None else 0.0
            out[r], _ = _simulate_stochastic_volatility(
                n,
                mu=float(self.mu_draws[draw]),
                phi=phi,
                sigma2=float(self.sigma2_draws[draw]),
                mean=float(self.mean_draws[draw]),
                rng=rng,
                nu=nu,
                rho=rho,
            )
        return out

    def simulate(
        self,
        n: int = 200,
        *,
        seed: int | np.random.Generator | None = None,
        burn: int = 0,
        draw: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """A fresh sample path at the posterior mean or at one kept draw.

        The log variance starts from its stationary law, so no burn-in is
        needed; ``burn`` is honoured for uniformity with the other results.

        Args:
            n: Observations kept.
            seed: Seed or generator.
            burn: Periods discarded from the start.
            draw: Index of a kept draw, or ``None`` for the posterior mean.

        Returns:
            An array of shape ``(n,)``.

        Raises:
            SpecificationError: If the counts are not usable, the draw
                index is out of range, or the parameters leave the
                stationary region.
        """
        if n < 1 or burn < 0:
            raise SpecificationError(f"n must be positive and burn non-negative; got {n}, {burn}.")
        if draw is not None and not 0 <= draw < self.n_kept:
            raise SpecificationError(f"draw must lie in 0 .. {self.n_kept - 1}; got {draw}.")
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        if draw is None:
            params = self.params
            mu, phi, sigma2, mean = params.mu, params.phi, params.sigma2, params.mean
            nu, rho = params.nu, params.rho
        else:
            mu, phi = float(self.mu_draws[draw]), float(self.phi_draws[draw])
            sigma2, mean = float(self.sigma2_draws[draw]), float(self.mean_draws[draw])
            nu = float(self.nu_draws[draw]) if self.nu_draws is not None else None
            rho = float(self.rho_draws[draw]) if self.rho_draws is not None else 0.0
        y, _ = _simulate_stochastic_volatility(
            burn + n, mu=mu, phi=phi, sigma2=sigma2, mean=mean, rng=rng, nu=nu, rho=rho
        )
        return np.asarray(y[burn:], dtype=np.float64)

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """Aligned per-observation output.

        Returns:
            The observed series, the posterior-mean log variance and
            volatility, and the 5%, 50%, and 95% pointwise volatility
            quantiles, each ``(n,)``. Under the particle chain the three
            quantile columns are identical.
        """
        bands = self.volatility_quantiles()
        return {
            "observed": self.endog,
            "log_variance": self.log_variance,
            "volatility": self.volatility,
            "volatility_q05": bands[0],
            "volatility_q50": bands[1],
            "volatility_q95": bands[2],
        }

    def _summary_table(self) -> _SummaryTable:
        """Structured summary: posterior mean, sd, and a central interval.

        One row per parameter with its mean, standard deviation, and 5% and
        95% quantiles over the kept draws: the observation mean first when
        the specification has one, then ``mu``, ``phi``, ``sigma2``, and
        ``nu`` and ``rho`` where present. Under the particle chain the mean
        still gets a row, with zero spread. The metadata lists the sampler
        and the run's bookkeeping; the thinning interval is listed only
        when an acceptance rate is.

        Returns:
            The :class:`_SummaryTable` that ``summary()``, ``str()``, and
            the notebook renderer display.
        """

        def row(name: str, draws: npt.NDArray[np.float64]) -> tuple[str, ...]:
            """Format one quantity's draws as a table row: mean, sd, 5% and 95% quantiles."""
            lo, hi = np.quantile(draws, [0.05, 0.95])
            return (name, f"{draws.mean():.4f}", f"{draws.std():.4f}", f"{lo:.4f}", f"{hi:.4f}")

        rows = [
            row("mu", self.mu_draws),
            row("phi", self.phi_draws),
            row("sigma2", self.sigma2_draws),
        ]
        if self.nu_draws is not None:
            rows.append(row("nu", self.nu_draws))
        if self.rho_draws is not None:
            rows.append(row("rho", self.rho_draws))
        if self.mean_spec == "constant":
            rows.insert(0, row("mean", self.mean_draws))
        notes = [
            "Kim-Shephard-Chib mixture Gibbs sampler: exact conditional on the "
            "seven-component approximation to log chi-squared(1)."
            if self.method == "gibbs"
            else "Particle marginal Metropolis-Hastings: exact posterior, likelihood "
            "estimated by the bootstrap particle filter; the observation mean is "
            "concentrated at the sample mean.",
        ]
        if self.nu_draws is not None:
            notes.append(
                "Student-t observation noise as a scale mixture; under Gibbs the "
                "degrees of freedom move by a random-walk Metropolis step on "
                "log(nu - 2) whose acceptance rate is reported."
                if self.method == "gibbs"
                else "Student-t observation noise; the degrees of freedom are a "
                "coordinate of the chain."
            )
        if self.rho_draws is not None:
            notes.append(
                "Leverage: the log-variance transition conditions on the previous "
                "return, and the particle smoother reads that conditional density "
                "exactly."
            )
        metadata: list[tuple[str, str]] = [
            ("Sampler", self.method),
            ("Draws kept", f"{self.n_kept}"),
            ("Observations", f"{self.nobs}"),
            ("Burn-in", f"{self.n_burn}"),
        ]
        if self.acceptance_rate is not None:
            label = "Acceptance" if self.method == "particle" else "nu acceptance"
            metadata.append((label, f"{self.acceptance_rate:.3f}"))
            metadata.append(("Thin", f"{self.thin}"))
        return _SummaryTable(
            title=f"SV[{self._spec_label()}] Posterior",
            metadata=tuple(metadata),
            columns=("", "mean", "sd", "5%", "95%"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _UCSVPosterior(_SummaryMixin, _SeriesMixin, _ConvergenceMixin):
    """A posterior over the unobserved-components stochastic-volatility model.

    Attributes:
        endog: The observed series.
        gamma_fixed: Whether the vol-of-vol was held fixed.
        trend_draws: ``(S, n)`` trend paths.
        h_draws: ``(S, n)`` log variances of the irregular.
        q_draws: ``(S, n)`` log variances of the trend innovation.
        gamma2_irregular_draws: ``(S,)`` vol-of-vol variance draws of ``h``.
        gamma2_trend_draws: ``(S,)`` vol-of-vol variance draws of ``q``.
        nobs: Observations.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.

    Stock and Watson's (2007) model, in the timing the sampler uses::

        y_t = tau_t + exp(h_t / 2) eps_t
        tau_{t+1} = tau_t + exp(q_t / 2) eta_t
        h_{t+1} = h_t + gamma_h nu_t
        q_{t+1} = q_t + gamma_q xi_t

    Every path is sampled, so unlike :class:`_SVPosterior` under the
    particle chain the bands here carry path uncertainty. When the
    vol-of-vol was held fixed, the two ``gamma2`` arrays are constant.

    Example:
        >>> import numpy as np
        >>> from cultivars.univariate.stochastic_volatility import UCSV
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(0.2 * rng.standard_normal(120)) + 0.7 * rng.standard_normal(120)
        >>> post = UCSV(y).fit(n_draws=120, n_burn=60, seed=0)
        >>> isinstance(post, _UCSVPosterior), post.gamma_fixed, post.n_kept
        (True, True, 60)
        >>> post.trend.shape, post.trend_quantiles().shape
        ((120,), (3, 120))
        >>> post.forecast(4, n_paths=50, seed=0).shape
        (50, 4)
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    """The observed series, ``(n,)``; kept out of the repr."""
    gamma_fixed: bool
    """Whether the vol-of-vol was held fixed and not sampled.

    When ``True`` the two ``gamma2`` arrays repeat the fixed value and their
    summary rows are degenerate.
    """
    trend_draws: npt.NDArray[np.float64] = field(repr=False)
    """Kept trend paths ``tau_t``, ``(S, n)``."""
    h_draws: npt.NDArray[np.float64] = field(repr=False)
    """Kept log-variance paths of the irregular, ``(S, n)``."""
    q_draws: npt.NDArray[np.float64] = field(repr=False)
    """Kept log-variance paths of the trend innovation, ``(S, n)``."""
    gamma2_irregular_draws: npt.NDArray[np.float64] = field(repr=False)
    """Kept draws of the step variance of ``h``, ``(S,)``; a variance, not a standard deviation."""
    gamma2_trend_draws: npt.NDArray[np.float64] = field(repr=False)
    """Kept draws of the step variance of ``q``, ``(S,)``; a variance, not a standard deviation."""
    nobs: int
    """Number of observations ``n``."""
    n_draws: int
    """Total sampler iterations, burn-in included."""
    n_burn: int
    """Iterations discarded as burn-in."""
    thin: int
    """Thinning interval applied after burn-in."""

    @classmethod
    def _from_fit(
        cls, fit: _TrendVolatilityFit, model: _TrendVolatilityModel[Any]
    ) -> _UCSVPosterior:
        """Assemble the public result from the sampler's output.

        The draw arrays are taken from the fit as they are, without copying.

        Args:
            fit: The packed Gibbs output.
            model: The engine that ran the sampler; supplies the series.

        Returns:
            The posterior record.
        """
        return cls(
            endog=model.endog,
            gamma_fixed=fit.gamma_fixed,
            trend_draws=fit.trend_draws,
            h_draws=fit.h_draws,
            q_draws=fit.q_draws,
            gamma2_irregular_draws=fit.gamma2_irregular_draws,
            gamma2_trend_draws=fit.gamma2_trend_draws,
            nobs=fit.nobs,
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
        )

    @property
    def n_kept(self) -> int:
        """Posterior draws retained.

        Returns:
            The number of kept trend paths, ``S``.
        """
        return int(self.trend_draws.shape[0])

    @property
    def params(self) -> _TrendVolatilityParameters:
        """The posterior-mean parameter record.

        Returns:
            A :class:`_TrendVolatilityParameters` holding the means of the
            two step variances and the posterior means of the trend and of
            both log variances at the first date.
        """
        return _TrendVolatilityParameters(
            gamma2_irregular=float(self.gamma2_irregular_draws.mean()),
            gamma2_trend=float(self.gamma2_trend_draws.mean()),
            h0=float(self.h_draws[:, 0].mean()),
            q0=float(self.q_draws[:, 0].mean()),
            tau0=float(self.trend_draws[:, 0].mean()),
        )

    @property
    def trend(self) -> npt.NDArray[np.float64]:
        """Posterior-mean trend, ``(n,)``.

        Returns:
            The mean of ``trend_draws`` over the kept draws.
        """
        return np.asarray(self.trend_draws.mean(axis=0), dtype=np.float64)

    def trend_quantiles(
        self, levels: tuple[float, ...] = (0.05, 0.5, 0.95)
    ) -> npt.NDArray[np.float64]:
        """Pointwise posterior quantiles of the trend, ``(len(levels), n)``.

        Args:
            levels: Probabilities in ``[0, 1]``, one row of the result each.

        Returns:
            The pointwise quantiles of the trend paths across draws.
        """
        return _quantiles(self.trend_draws, levels)

    @property
    def irregular_volatility(self) -> npt.NDArray[np.float64]:
        """Posterior-mean volatility of the irregular ``E[exp(h_t / 2) | y]``.

        Returns:
            The mean over draws of ``exp(h / 2)``, shape ``(n,)``.
        """
        return np.asarray(np.exp(0.5 * self.h_draws).mean(axis=0), dtype=np.float64)

    @property
    def trend_volatility(self) -> npt.NDArray[np.float64]:
        """Posterior-mean volatility of the trend innovation ``E[exp(q_t / 2) | y]``.

        Returns:
            The mean over draws of ``exp(q / 2)``, shape ``(n,)``.
        """
        return np.asarray(np.exp(0.5 * self.q_draws).mean(axis=0), dtype=np.float64)

    @property
    def gap(self) -> npt.NDArray[np.float64]:
        """The observed series minus the posterior-mean trend.

        Returns:
            The gap, shape ``(n,)``. It is a point summary: the difference
            from the mean trend, not the mean of a sampled irregular.
        """
        return np.asarray(self.endog - self.trend, dtype=np.float64)

    @property
    def state_space(self) -> _NonlinearStateSpace:
        """The model at the posterior mean, on the nonlinear substrate.

        Carries both non-Gaussian hooks (state-dependent trend noise, a
        multiplicative measurement), so only the particle filter reads it.
        Its ``particle_filter`` on new data reads that data with these
        vol-of-vol values; the initial state is the posterior mean at the
        first date.

        Returns:
            The state-space model built from :attr:`params`. It is rebuilt
            on every access.
        """
        return _trend_volatility_state_space(self.params)

    def forecast(
        self, steps: int, *, n_paths: int = 2000, seed: int | None = None
    ) -> npt.NDArray[np.float64]:
        """Predictive draws of the series, ``(n_paths, steps)``.

        Each path picks a kept draw, then propagates both random-walk
        log variances and the trend forward and adds the irregular.

        Within a step both log variances are advanced first, and the trend
        then moves with the *advanced* variance of its innovation. That is
        one period ahead of the timing the sampler and :attr:`state_space`
        use, where the step from ``t`` to ``t + 1`` is scaled by ``q_t``.

        Kept draws are picked with replacement. ``n_paths`` is not
        validated: zero returns an empty array, and a negative value raises
        NumPy's ``ValueError``.

        Args:
            steps: Horizons ahead, at least 1.
            n_paths: Predictive paths.
            seed: Seed.

        Returns:
            Simulated future values, one row per path and one column per
            horizon.

        Raises:
            SpecificationError: If ``steps`` is not positive.
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, self.n_kept, size=n_paths)
        tau = self.trend_draws[picks, -1]
        h = self.h_draws[picks, -1]
        q = self.q_draws[picks, -1]
        g_h = np.sqrt(self.gamma2_irregular_draws[picks])
        g_q = np.sqrt(self.gamma2_trend_draws[picks])
        out = np.empty((n_paths, steps))
        for k in range(steps):
            q = q + g_q * rng.standard_normal(n_paths)
            h = h + g_h * rng.standard_normal(n_paths)
            tau = tau + np.exp(0.5 * q) * rng.standard_normal(n_paths)
            out[:, k] = tau + np.exp(0.5 * h) * rng.standard_normal(n_paths)
        return out

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """Aligned per-observation output: the decomposition and its volatilities.

        Returns:
            The observed series, the posterior-mean trend with its 5% and
            95% pointwise quantiles, the gap, and the two posterior-mean
            volatilities, each ``(n,)``. There is no median column.
        """
        bands = self.trend_quantiles()
        return {
            "observed": self.endog,
            "trend": self.trend,
            "trend_q05": bands[0],
            "trend_q95": bands[2],
            "gap": self.gap,
            "irregular_volatility": self.irregular_volatility,
            "trend_volatility": self.trend_volatility,
        }

    def _summary_table(self) -> _SummaryTable:
        """Structured summary: vol-of-vol and end-of-sample volatilities.

        Five rows, each with the mean, standard deviation, and 5% and 95%
        quantiles over the kept draws: the two vol-of-vol parameters as
        standard deviations, and the trend and both volatilities at the last
        date. The thinning interval is not listed.

        Returns:
            The :class:`_SummaryTable` that ``summary()``, ``str()``, and
            the notebook renderer display.
        """

        def row(name: str, draws: npt.NDArray[np.float64]) -> tuple[str, ...]:
            """Format one quantity's draws as a table row: mean, sd, 5% and 95% quantiles."""
            lo, hi = np.quantile(draws, [0.05, 0.95])
            return (name, f"{draws.mean():.4f}", f"{draws.std():.4f}", f"{lo:.4f}", f"{hi:.4f}")

        rows = [
            row("gamma.irregular", np.sqrt(self.gamma2_irregular_draws)),
            row("gamma.trend", np.sqrt(self.gamma2_trend_draws)),
            row("trend[T]", self.trend_draws[:, -1]),
            row("vol.irregular[T]", np.exp(0.5 * self.h_draws[:, -1])),
            row("vol.trend[T]", np.exp(0.5 * self.q_draws[:, -1])),
        ]
        notes = [
            "Vol-of-vol held fixed at the Stock-Watson value; the first two rows are degenerate."
            if self.gamma_fixed
            else "Vol-of-vol estimated under inverse-gamma priors.",
            "Trend drawn exactly by the simulation smoother under the current "
            "variance paths; variances drawn by the Kim-Shephard-Chib mixture step.",
        ]
        return _SummaryTable(
            title="UC-SV Posterior",
            metadata=(
                ("Sampler", "gibbs"),
                ("Draws kept", f"{self.n_kept}"),
                ("Observations", f"{self.nobs}"),
                ("Burn-in", f"{self.n_burn}"),
            ),
            columns=("", "mean", "sd", "5%", "95%"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _PerturbationDSGEPosterior(_SummaryMixin, _ConvergenceMixin):
    """A particle-chain posterior over a perturbation model.

    Attributes:
        data: The observables.
        parameter_names: Structural parameter labels.
        theta_draws: ``(S, d)`` kept draws in the model's own space.
        loglik_draws: ``(S,)`` particle likelihood estimates carried with
            each kept draw.
        log_posterior_draws: ``(S,)`` log prior plus log-likelihood estimate
            at each kept draw, in the chain's unconstrained space.
        acceptance_rate: Acceptance rate over the whole run.
        order: Perturbation order.
        n_particles: Particles per likelihood estimate.
        nobs: Observations.
        n_draws: Total iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.

    The draws are stored twice. ``theta_draws`` holds them in the model's
    own, bounded space and is what every summary reads. ``_unconstrained``
    holds the same draws in the space the chain moved in, which is the
    space ``log_posterior_draws`` is a density on and therefore the space
    the marginal likelihood is computed in.

    The record keeps a reference to the engine that produced it, because
    everything beyond the draws (``solution``, ``state_space``,
    ``impulse_responses``, ``simulate``) solves the model again at a
    parameter value. None of it is cached.

    Example:
        A latent AR(1) written as a one-state model, with a chain far too
        short for inference:

        >>> import numpy as np
        >>> from cultivars.multivariate.structural.perturbation import PerturbationDSGE
        >>> class Latent:
        ...     parameter_names, bounds = ("rho", "sigma"), ((0.0, 0.99), (0.01, 1.0))
        ...     n_states, n_controls = 1, 1
        ...
        ...     def equations(self, th, y_next, y, x_next, x):
        ...         return np.array([y[0] - x[0], x_next[0] - th[0] * x[0]])
        ...
        ...     def steady_state(self, th):
        ...         return np.zeros(1), np.zeros(1)
        ...
        ...     def shock_loading(self, th):
        ...         return np.array([[th[1]]])
        ...
        ...     def observation(self, th):
        ...         return np.array([[0.0, 1.0]]), np.zeros(1), np.array([[1e-4]])
        ...
        ...     def log_prior(self, th):
        ...         return 0.0
        >>> rng = np.random.default_rng(0)
        >>> x = np.zeros(80)
        >>> for t in range(1, 80):
        ...     x[t] = 0.8 * x[t - 1] + 0.5 * rng.standard_normal()
        >>> model = PerturbationDSGE(Latent(), x, order=2)
        >>> start = model.fit([0.5, 0.3]).theta
        >>> post = model.sample(start, n_particles=50, n_draws=30, n_burn=10, seed=0)
        >>> isinstance(post, _PerturbationDSGEPosterior), post.n_kept, list(post.params)
        (True, 20, ['rho', 'sigma'])
        >>> states, controls = post.impulse_responses(horizon=5, n_draws=4, seed=0)
        >>> states.shape, controls.shape
        ((4, 1, 5, 1), (4, 1, 5, 1))
    """

    data: npt.NDArray[np.float64] = field(repr=False)
    """The observables, ``(n, p)``; kept out of the repr."""
    parameter_names: tuple[str, ...]
    """Structural parameter labels, in the column order of ``theta_draws``."""
    theta_draws: npt.NDArray[np.float64] = field(repr=False)
    """Kept draws in the model's own, constrained space, ``(S, d)``."""
    loglik_draws: npt.NDArray[np.float64] = field(repr=False)
    """Particle log-likelihood estimate carried with each kept draw, ``(S,)``."""
    log_posterior_draws: npt.NDArray[np.float64] = field(repr=False)
    """Log prior plus log-likelihood estimate at each kept draw, ``(S,)``.

    Evaluated in the unconstrained space, so it includes the log-Jacobian
    of the reparameterization. It is unnormalized.
    """
    acceptance_rate: float
    """Acceptance rate of the chain over the whole run, burn-in included."""
    order: int
    """Perturbation order of the solution the likelihood was evaluated on."""
    n_particles: int
    """Particles per likelihood estimate."""
    nobs: int
    """Number of observations ``n``."""
    n_draws: int
    """Total chain iterations, burn-in included."""
    n_burn: int
    """Iterations discarded as burn-in."""
    thin: int
    """Thinning interval applied after burn-in."""
    _engine: _PerturbationModel[Any] = field(repr=False)
    """The engine that produced the draws, used to solve the model at a parameter value."""
    _unconstrained: npt.NDArray[np.float64] = field(repr=False)
    """The kept draws in the space the chain moved in, ``(S, d)``.

    Its name does not end in ``_draws``, so the convergence diagnostics
    report ``theta_draws`` and not this array.
    """

    def marginal_likelihood(self, *, tau: float = _MHM_TAU) -> _MarginalLikelihoodSelection:
        """Geweke's modified harmonic mean of the log marginal likelihood.

        Computed in the chain's unconstrained space, where the Gaussian
        envelope is least wrong; the evidence is invariant to the
        reparameterization. The likelihood entering the kernel is the
        particle-filter *estimate* carried by each draw, so the number is
        the Herbst-Schorfheide practice rather than an exact ordinate: the
        estimate noise biases it, and the record says so.

        Args:
            tau: Probability mass of the envelope retained.

        Returns:
            The record, with the Monte Carlo error and a coverage note.

        Raises:
            SpecificationError: If ``tau`` is not in ``(0, 1)``, or no kept
                draw falls inside the envelope.
        """
        log_ml, mcse, coverage = _modified_harmonic_mean(
            self._unconstrained, self.log_posterior_draws, tau=tau
        )
        notes = [
            "Modified harmonic mean (Geweke, 1999) on the unconstrained chain; the kernel "
            "carries particle-filter likelihood estimates, whose noise biases the value.",
        ]
        if abs(coverage - tau) > 0.1:
            notes.append(
                f"Only {coverage:.0%} of the draws fall inside the {tau:.0%} envelope, so the "
                "posterior is far from the Gaussian the envelope assumes; read the value "
                "with that in mind."
            )
        return _MarginalLikelihoodSelection(
            log_value=log_ml,
            mcse=mcse,
            method="modified harmonic mean",
            n_draws=self.n_kept,
            source=f"Perturbation DSGE (order {self.order})",
            nobs=int(self.nobs),
            notes=tuple(notes),
        )

    def _draw_labels(self) -> dict[str, tuple[str, ...]]:
        """Name the columns of ``theta_draws`` after the structural parameters."""
        return {"theta": self.parameter_names}

    def _convergence_label(self) -> str:
        """Title of the convergence report, naming the perturbation order."""
        return f"Perturbation DSGE (order {self.order}) posterior"

    @classmethod
    def _from_fit(
        cls, fit: _ParticleChainFit, model: _PerturbationModel[Any]
    ) -> _PerturbationDSGEPosterior:
        """Assemble the public result, mapping draws back to the model's space.

        Each kept draw is passed through the engine's inverse
        reparameterization to give ``theta_draws``; the unconstrained draws
        are kept beside them for the marginal likelihood.

        Args:
            fit: The packed chain output, draws in the unconstrained space.
            model: The engine that ran the chain.

        Returns:
            The posterior record.
        """
        theta = np.stack([model._to_constrained(z) for z in fit.theta_draws])
        return cls(
            data=model.data,
            parameter_names=tuple(model.parameter_names),
            theta_draws=np.asarray(theta, dtype=np.float64),
            loglik_draws=fit.loglik_draws,
            log_posterior_draws=fit.log_posterior_draws,
            acceptance_rate=fit.acceptance_rate,
            order=model.order,
            n_particles=fit.n_particles,
            nobs=int(model.data.shape[0]),
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
            _engine=model,
            _unconstrained=np.asarray(fit.theta_draws, dtype=np.float64),
        )

    @property
    def n_kept(self) -> int:
        """Posterior draws retained.

        Returns:
            The number of rows ``S`` of ``theta_draws``.
        """
        return int(self.theta_draws.shape[0])

    @property
    def theta_mean(self) -> npt.NDArray[np.float64]:
        """The posterior mean.

        Returns:
            The mean of ``theta_draws`` over the kept draws, shape ``(d,)``,
            in the model's own space.
        """
        return np.asarray(self.theta_draws.mean(axis=0), dtype=np.float64)

    @property
    def params(self) -> dict[str, float]:
        """The posterior mean by name.

        Returns:
            A mapping from each structural parameter name to its posterior
            mean, in the order of ``parameter_names``.
        """
        return {
            name: float(value)
            for name, value in zip(self.parameter_names, self.theta_mean, strict=True)
        }

    @property
    def solution(self) -> _PerturbationSolution:
        """The perturbation solution at the posterior mean.

        Returns:
            The solution the engine computes at :attr:`theta_mean`. The
            model is solved again on every access.
        """
        return self._engine.solve(self.theta_mean)

    @property
    def state_space(self) -> _LinearGaussianStateSpace | _NonlinearStateSpace:
        """The solution at the posterior mean on its substrate.

        Returns:
            A linear-Gaussian state space for a first-order solution and a
            nonlinear one for a second-order solution. It is rebuilt on
            every access.
        """
        return self._engine.state_space(self.theta_mean)

    def impulse_responses(
        self, *, horizon: int = 40, size: float = 1.0, n_draws: int = 200, seed: int | None = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Posterior draws of the responses.

        Returns ``(states, controls)`` of shapes ``(n_draws, n_eps, horizon, n_x)``
        and ``(n_draws, n_eps, horizon, n_y)``.

        Each draw resolves the model at one kept parameter draw, so the
        band is the posterior of the response and not the response at the
        posterior mean.

        At most ``n_kept`` draws are used, chosen without replacement, so
        the leading dimension is ``min(n_draws, n_kept)``. ``n_draws`` is
        not otherwise validated: zero or a negative value raises NumPy's
        ``ValueError``.

        Args:
            horizon: Periods after the impulse.
            size: Impulse size in standard deviations.
            n_draws: Parameter draws to propagate.
            seed: Seed for the draw selection.

        Returns:
            The state responses and the control responses, indexed by draw,
            shock, period, and variable.

        Raises:
            SpecificationError: If the horizon is not positive.
        """
        if horizon < 1:
            raise SpecificationError(f"horizon must be at least 1; got {horizon}.")
        rng = np.random.default_rng(seed)
        picks = rng.choice(self.n_kept, size=min(n_draws, self.n_kept), replace=False)
        states = []
        controls = []
        for index in picks:
            solution = self._engine.solve(self.theta_draws[index])
            s_path, c_path = _impulse_responses(solution, horizon=horizon, size=size)
            states.append(s_path)
            controls.append(c_path)
        return np.stack(states), np.stack(controls)

    def simulate(
        self,
        n: int = 200,
        *,
        seed: int | np.random.Generator | None = None,
        burn: int = _SIMULATION_BURN,
        draw: int | None = None,
    ) -> npt.NDArray[np.float64]:
        """A fresh sample of the observables at the posterior mean or at one kept draw.

        Args:
            n: Observations kept.
            seed: Seed or generator.
            burn: Periods discarded from the start.
            draw: Index of a kept parameter draw, or ``None`` for the
                posterior mean.

        Returns:
            An array of shape ``(n, p)`` in the order of the data columns.

        Raises:
            SpecificationError: If the counts are not usable or the draw
                index is out of range.
        """
        if draw is not None and not 0 <= draw < self.n_kept:
            raise SpecificationError(f"draw must lie in 0 .. {self.n_kept - 1}; got {draw}.")
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        theta = self.theta_mean if draw is None else self.theta_draws[draw]
        solution = self._engine.solve(theta)
        design, intercept, obs_cov = self._engine._measurement(theta, solution)
        return _simulate_perturbation(
            solution, n, design=design, intercept=intercept, obs_cov=obs_cov, rng=rng, burn=burn
        )

    def _summary_table(self) -> _SummaryTable:
        """Structured summary: posterior mean, sd, and a central interval.

        One row per structural parameter with its mean, standard deviation,
        and 5% and 95% quantiles over the kept draws in the model's own
        space. The thinning interval is not listed.

        Returns:
            The :class:`_SummaryTable` that ``summary()``, ``str()``, and
            the notebook renderer display.
        """
        rows = []
        for j, name in enumerate(self.parameter_names):
            draws = self.theta_draws[:, j]
            lo, hi = np.quantile(draws, [0.05, 0.95])
            rows.append(
                (name, f"{draws.mean():.5g}", f"{draws.std():.4g}", f"{lo:.5g}", f"{hi:.5g}")
            )
        return _SummaryTable(
            title=f"Perturbation DSGE (order {self.order}) Posterior",
            metadata=(
                ("Sampler", "particle MH"),
                ("Draws kept", f"{self.n_kept}"),
                ("Observations", f"{self.nobs}"),
                ("Particles", f"{self.n_particles}"),
                ("Acceptance", f"{self.acceptance_rate:.3f}"),
                ("Burn-in", f"{self.n_burn}"),
            ),
            columns=("", "mean", "sd", "5%", "95%"),
            rows=tuple(rows),
            notes=(
                "Particle marginal Metropolis-Hastings on the pruned solution; the "
                "proposal adapted during burn-in and was fixed afterwards.",
            ),
        )
