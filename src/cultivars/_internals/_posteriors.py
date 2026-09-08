from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .._core import SummaryTable, _quantiles
from ..exceptions import NumericalError, SpecificationError
from ._emitters import (
    _trend_volatility_state_space,
    _volatility_state_space,
)
from ._fits import _ParticleChainFit, _TrendVolatilityFit, _VolatilityDrawsFit
from ._mixins import _SeriesMixin, _SummaryMixin
from ._parameters import _StochasticVolatilityParameters, _TrendVolatilityParameters
from ._simulators import _impulse_responses
from ._solutions import _PerturbationSolution
from ._substrates import _LinearGaussianStateSpace, _NonlinearStateSpace

if TYPE_CHECKING:
    from ._models import (
        _PerturbationModel,
        _StochasticVolatilityModel,
        _TrendVolatilityModel,
    )


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
    """

    coefficients: npt.NDArray[np.float64]
    row_precision: npt.NDArray[np.float64]
    scale: npt.NDArray[np.float64]
    df: float
    log_ml: float

    @property
    def sigma_mean(self) -> npt.NDArray[np.float64]:
        """Posterior mean of the innovation covariance.

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
class _SVPosterior(_SummaryMixin, _SeriesMixin):
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
        loglik_draws: ``(S,)`` particle likelihood estimates, or ``None``.
        acceptance_rate: The chain's acceptance rate, or ``None``.
        nobs: Observations.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    mean_spec: str
    method: str
    mu_draws: npt.NDArray[np.float64] = field(repr=False)
    phi_draws: npt.NDArray[np.float64] = field(repr=False)
    sigma2_draws: npt.NDArray[np.float64] = field(repr=False)
    mean_draws: npt.NDArray[np.float64] = field(repr=False)
    h_draws: npt.NDArray[np.float64] = field(repr=False)
    nu_draws: npt.NDArray[np.float64] | None = field(repr=False)
    rho_draws: npt.NDArray[np.float64] | None = field(repr=False)
    loglik_draws: npt.NDArray[np.float64] | None = field(repr=False)
    acceptance_rate: float | None
    nobs: int
    n_draws: int
    n_burn: int
    thin: int

    @classmethod
    def _from_gibbs(
        cls, fit: _VolatilityDrawsFit, model: _StochasticVolatilityModel[Any]
    ) -> _SVPosterior:
        """Assemble from the Gibbs sampler's output."""
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
        """Assemble from the particle chain's output."""
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
        """Posterior draws retained."""
        return int(self.mu_draws.shape[0])

    def _spec_label(self) -> str:
        """``mean[, t][, leverage]`` for titles."""
        parts = [self.mean_spec]
        if self.nu_draws is not None:
            parts.append("t")
        if self.rho_draws is not None:
            parts.append("leverage")
        return ", ".join(parts)

    @property
    def params(self) -> _StochasticVolatilityParameters:
        """The posterior-mean parameter record."""
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
        """Posterior-mean log-variance path, ``(n,)``."""
        return np.asarray(self.h_draws.mean(axis=0), dtype=np.float64)

    @property
    def volatility(self) -> npt.NDArray[np.float64]:
        """Posterior-mean volatility ``E[exp(h_t / 2) | y]``, ``(n,)``."""
        return np.asarray(np.exp(0.5 * self.h_draws).mean(axis=0), dtype=np.float64)

    def volatility_quantiles(
        self, levels: tuple[float, ...] = (0.05, 0.5, 0.95)
    ) -> npt.NDArray[np.float64]:
        """Pointwise posterior quantiles of the volatility, ``(len(levels), n)``.

        Under the particle chain the path block holds one smoothed path,
        so the quantiles collapse to it; the parameter draws carry the
        uncertainty there.
        """
        return _quantiles(np.exp(0.5 * self.h_draws), levels)

    @property
    def state_space(self) -> _NonlinearStateSpace:
        """The model at the posterior mean, on the nonlinear substrate."""
        return _volatility_state_space(self.params)

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """Aligned per-observation output."""
        bands = self.volatility_quantiles()
        return {
            "observed": self.endog,
            "log_variance": self.log_variance,
            "volatility": self.volatility,
            "volatility_q05": bands[0],
            "volatility_q50": bands[1],
            "volatility_q95": bands[2],
        }

    def _summary_table(self) -> SummaryTable:
        """Structured summary: posterior mean, sd, and a central interval."""

        def row(name: str, draws: npt.NDArray[np.float64]) -> tuple[str, ...]:
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
        return SummaryTable(
            title=f"SV[{self._spec_label()}] Posterior",
            metadata=tuple(metadata),
            columns=("", "mean", "sd", "5%", "95%"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _UCSVPosterior(_SummaryMixin, _SeriesMixin):
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
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    gamma_fixed: bool
    trend_draws: npt.NDArray[np.float64] = field(repr=False)
    h_draws: npt.NDArray[np.float64] = field(repr=False)
    q_draws: npt.NDArray[np.float64] = field(repr=False)
    gamma2_irregular_draws: npt.NDArray[np.float64] = field(repr=False)
    gamma2_trend_draws: npt.NDArray[np.float64] = field(repr=False)
    nobs: int
    n_draws: int
    n_burn: int
    thin: int

    @classmethod
    def _from_fit(
        cls, fit: _TrendVolatilityFit, model: _TrendVolatilityModel[Any]
    ) -> _UCSVPosterior:
        """Assemble the public result from the sampler's output."""
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
        """Posterior draws retained."""
        return int(self.trend_draws.shape[0])

    @property
    def params(self) -> _TrendVolatilityParameters:
        """The posterior-mean parameter record."""
        return _TrendVolatilityParameters(
            gamma2_irregular=float(self.gamma2_irregular_draws.mean()),
            gamma2_trend=float(self.gamma2_trend_draws.mean()),
            h0=float(self.h_draws[:, 0].mean()),
            q0=float(self.q_draws[:, 0].mean()),
            tau0=float(self.trend_draws[:, 0].mean()),
        )

    @property
    def trend(self) -> npt.NDArray[np.float64]:
        """Posterior-mean trend, ``(n,)``."""
        return np.asarray(self.trend_draws.mean(axis=0), dtype=np.float64)

    def trend_quantiles(
        self, levels: tuple[float, ...] = (0.05, 0.5, 0.95)
    ) -> npt.NDArray[np.float64]:
        """Pointwise posterior quantiles of the trend, ``(len(levels), n)``."""
        return _quantiles(self.trend_draws, levels)

    @property
    def irregular_volatility(self) -> npt.NDArray[np.float64]:
        """Posterior-mean volatility of the irregular ``E[exp(h_t / 2) | y]``."""
        return np.asarray(np.exp(0.5 * self.h_draws).mean(axis=0), dtype=np.float64)

    @property
    def trend_volatility(self) -> npt.NDArray[np.float64]:
        """Posterior-mean volatility of the trend innovation ``E[exp(q_t / 2) | y]``."""
        return np.asarray(np.exp(0.5 * self.q_draws).mean(axis=0), dtype=np.float64)

    @property
    def gap(self) -> npt.NDArray[np.float64]:
        """The observed series minus the posterior-mean trend."""
        return np.asarray(self.endog - self.trend, dtype=np.float64)

    @property
    def state_space(self) -> _NonlinearStateSpace:
        """The model at the posterior mean, on the nonlinear substrate.

        Carries both non-Gaussian hooks (state-dependent trend noise, a
        multiplicative measurement), so only the particle filter reads it.
        Its ``particle_filter`` on new data reads that data with these
        vol-of-vol values; the initial state is the posterior mean at the
        first date.
        """
        return _trend_volatility_state_space(self.params)

    def forecast(
        self, steps: int, *, n_paths: int = 2000, seed: int | None = None
    ) -> npt.NDArray[np.float64]:
        """Predictive draws of the series, ``(n_paths, steps)``.

        Each path picks a kept draw, then propagates both random-walk
        log variances and the trend forward and adds the irregular.

        Args:
            steps: Horizons ahead, at least 1.
            n_paths: Predictive paths.
            seed: Seed.

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
        """Aligned per-observation output: the decomposition and its volatilities."""
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

    def _summary_table(self) -> SummaryTable:
        """Structured summary: vol-of-vol and end-of-sample volatilities."""

        def row(name: str, draws: npt.NDArray[np.float64]) -> tuple[str, ...]:
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
        return SummaryTable(
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
class _PerturbationDSGEPosterior(_SummaryMixin):
    """A particle-chain posterior over a perturbation model.

    Attributes:
        data: The observables.
        parameter_names: Structural parameter labels.
        theta_draws: ``(S, d)`` kept draws in the model's own space.
        loglik_draws: ``(S,)`` particle likelihood estimates carried with
            each kept draw.
        acceptance_rate: Acceptance rate over the whole run.
        order: Perturbation order.
        n_particles: Particles per likelihood estimate.
        nobs: Observations.
        n_draws: Total iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.
    """

    data: npt.NDArray[np.float64] = field(repr=False)
    parameter_names: tuple[str, ...]
    theta_draws: npt.NDArray[np.float64] = field(repr=False)
    loglik_draws: npt.NDArray[np.float64] = field(repr=False)
    acceptance_rate: float
    order: int
    n_particles: int
    nobs: int
    n_draws: int
    n_burn: int
    thin: int
    _engine: _PerturbationModel[Any] = field(repr=False)

    @classmethod
    def _from_fit(
        cls, fit: _ParticleChainFit, model: _PerturbationModel[Any]
    ) -> _PerturbationDSGEPosterior:
        """Assemble the public result, mapping draws back to the model's space."""
        theta = np.stack([model._to_constrained(z) for z in fit.theta_draws])
        return cls(
            data=model.data,
            parameter_names=tuple(model.parameter_names),
            theta_draws=np.asarray(theta, dtype=np.float64),
            loglik_draws=fit.loglik_draws,
            acceptance_rate=fit.acceptance_rate,
            order=model.order,
            n_particles=fit.n_particles,
            nobs=int(model.data.shape[0]),
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
            _engine=model,
        )

    @property
    def n_kept(self) -> int:
        """Posterior draws retained."""
        return int(self.theta_draws.shape[0])

    @property
    def theta_mean(self) -> npt.NDArray[np.float64]:
        """The posterior mean."""
        return np.asarray(self.theta_draws.mean(axis=0), dtype=np.float64)

    @property
    def params(self) -> dict[str, float]:
        """The posterior mean by name."""
        return {
            name: float(value)
            for name, value in zip(self.parameter_names, self.theta_mean, strict=True)
        }

    @property
    def solution(self) -> _PerturbationSolution:
        """The perturbation solution at the posterior mean."""
        return self._engine.solve(self.theta_mean)

    @property
    def state_space(self) -> _LinearGaussianStateSpace | _NonlinearStateSpace:
        """The solution at the posterior mean on its substrate."""
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

        Args:
            horizon: Periods after the impulse.
            size: Impulse size in standard deviations.
            n_draws: Parameter draws to propagate.
            seed: Seed for the draw selection.

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

    def _summary_table(self) -> SummaryTable:
        """Structured summary: posterior mean, sd, and a central interval."""
        rows = []
        for j, name in enumerate(self.parameter_names):
            draws = self.theta_draws[:, j]
            lo, hi = np.quantile(draws, [0.05, 0.95])
            rows.append(
                (name, f"{draws.mean():.5g}", f"{draws.std():.4g}", f"{lo:.5g}", f"{hi:.5g}")
            )
        return SummaryTable(
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
