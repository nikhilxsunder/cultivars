# filepath: /src/cultivars/univariate/stochastic_volatility.py
#
# (MIT header)
"""Stochastic volatility: variance as a latent state, not a function of the past.

A GARCH variance is a deterministic function of past observations; a
stochastic-volatility variance is a latent process with its own
innovation, observed only through the returns it scales. That makes the
model a nonlinear state space -- the measurement is multiplicative in the
state -- and puts its likelihood out of closed-form reach. Two honest
routes exist and both are offered:

- The Kim-Shephard-Chib (1998) route linearizes by squaring and taking
  logs, then replaces the ``log chi-squared(1)`` error with a
  seven-component Gaussian mixture; conditional on the mixture
  indicators the model is linear-Gaussian and a Gibbs sampler is exact.
  The mixture is a very good approximation, not the truth.
- The particle route (Andrieu, Doucet, and Holenstein, 2010) makes no
  approximation: a particle filter estimates the likelihood without bias
  and a Metropolis-Hastings chain on that estimate targets the exact
  posterior. It costs a particle filter per draw.

The two samplers are given the same prior, so they should agree, and a
disagreement is the diagnostic that the mixture is binding. Point
estimation is offered for the same model: the Harvey-Ruiz-Shephard
quasi-maximum likelihood on the linearized model (fast, consistent, the
warm start for everything else) and simulated maximum likelihood on the
particle estimate under common random numbers.

The second model here is Stock and Watson's (2007) unobserved-components
stochastic-volatility model: a random-walk trend observed through an
irregular, each innovation with a random-walk log variance. It is the
workhorse of trend-inflation measurement and the model that joins the
package's structural decomposition to its volatility machinery; its
estimator is the Gibbs sampler, with the trend drawn exactly by the
simulation smoother under the current variance paths.

References:
    Taylor, S. J. (1986). *Modelling Financial Time Series*. Wiley.
    Harvey, A. C., Ruiz, E., & Shephard, N. (1994). Multivariate
        stochastic variance models. *Review of Economic Studies*, 61(2),
        247-264.
    Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
        Likelihood inference and comparison with ARCH models. *Review of
        Economic Studies*, 65(3), 361-393.
    Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle Markov
        chain Monte Carlo methods. *Journal of the Royal Statistical
        Society B*, 72(3), 269-342.
    Stock, J. H., & Watson, M. W. (2007). Why has U.S. inflation become
        harder to forecast? *Journal of Money, Credit and Banking*, 39(s1),
        3-33.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core import SummaryTable
from .._internals import (
    _ComparisonMixin,
    _quasi_volatility_state_space,
    _SeriesMixin,
    _StochasticVolatilityFit,
    _StochasticVolatilityModel,
    _StochasticVolatilityParameters,
    _SummaryMixin,
    _SVPosterior,
    _TrendVolatilityModel,
    _UCSVPosterior,
    _volatility_state_space,
)
from ..exceptions import SpecificationError
from ..state_space import LinearGaussianSSM, NonlinearSSM

__all__ = [
    "SV",
    "UCSV",
    "SVResult",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SVResult(_SummaryMixin, _SeriesMixin, _ComparisonMixin):
    """A point-estimated stochastic-volatility model.

    Attributes:
        endog: The observed series.
        mean_spec: ``"constant"`` or ``"zero"``.
        mean: The observation mean ``c``.
        mu: Unconditional mean of the log variance.
        phi: Persistence of the log variance.
        sigma2: Innovation variance of the log variance.
        llf: The criterion at the optimum. Under ``method="qml"`` this is
            the exact Gaussian likelihood of the *linearized* model, a
            quasi-likelihood for the true one; under ``method="particle"``
            it is a particle estimate of the exact likelihood, carrying
            Monte Carlo error. Information criteria are comparable only
            within one method.
        method: ``"qml"`` or ``"particle"``.
        nobs: Observations.
        n_params: Free parameters.
        log_variance: Smoothed log-variance path, ``(n,)``.
        log_variance_std: Smoothed log-variance standard deviations.
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    mean_spec: str
    mean: float
    mu: float
    phi: float
    sigma2: float
    nu: float | None
    rho: float
    llf: float
    method: str
    nobs: int
    n_params: float
    log_variance: npt.NDArray[np.float64] = field(repr=False)
    log_variance_std: npt.NDArray[np.float64] = field(repr=False)

    @classmethod
    def _from_fit(cls, fit: _StochasticVolatilityFit, model: SV) -> SVResult:
        """Assemble the public result from a raw fit and its specification."""
        return cls(
            endog=model.endog,
            mean_spec=model.mean_spec,
            mean=fit.params.mean,
            mu=fit.params.mu,
            phi=fit.params.phi,
            sigma2=fit.params.sigma2,
            nu=fit.params.nu,
            rho=fit.params.rho,
            llf=fit.llf,
            method=fit.method,
            nobs=fit.nobs,
            n_params=float(fit.n_params),
            log_variance=fit.log_variance,
            log_variance_std=fit.log_variance_std,
        )

    @property
    def _params(self) -> _StochasticVolatilityParameters:
        """The parameter record, rebuilt for the emitters."""
        return _StochasticVolatilityParameters(
            mu=self.mu, phi=self.phi, sigma2=self.sigma2, mean=self.mean, nu=self.nu, rho=self.rho
        )

    @property
    def volatility(self) -> npt.NDArray[np.float64]:
        """Smoothed volatility ``exp(h_t / 2)``, ``(n,)``."""
        return np.asarray(np.exp(0.5 * self.log_variance), dtype=np.float64)

    def volatility_bands(
        self, *, alpha: float = 0.05
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Pointwise ``1 - alpha`` bands on the volatility from the smoothed log variance.

        Args:
            alpha: Two-sided tail mass.

        Returns:
            ``(lower, upper)`` arrays of length ``n``.
        """
        from scipy.stats import norm

        z = float(norm.ppf(1.0 - alpha / 2.0))
        lower = np.exp(0.5 * (self.log_variance - z * self.log_variance_std))
        upper = np.exp(0.5 * (self.log_variance + z * self.log_variance_std))
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    @property
    def half_life(self) -> float:
        """Periods for a log-variance shock to halve, ``log 0.5 / log phi``."""
        if not 0.0 < self.phi < 1.0:
            return float("inf")
        return float(np.log(0.5) / np.log(self.phi))

    @property
    def unconditional_variance(self) -> float:
        """``E[exp(h_t)]`` under the stationary log-normal law."""
        return float(np.exp(self.mu + 0.5 * self.sigma2 / (1.0 - self.phi**2)))

    @property
    def state_space(self) -> NonlinearSSM:
        """The fitted model on the nonlinear substrate.

        Carries the multiplicative measurement through the
        ``observation_loglik`` hook, so only the particle filter reads it;
        ``particle_filter`` on the estimation sample estimates the exact
        likelihood, and on new data reads that data with this fit.
        """
        return _volatility_state_space(self._params)

    @property
    def quasi_state_space(self) -> LinearGaussianSSM:
        """The Harvey-Ruiz-Shephard linearization on the linear substrate.

        Filters ``log((y - c)**2)``, not ``y``; its ``loglikelihood`` on the
        linearized estimation sample reproduces ``llf`` when the method
        was ``"qml"``.
        """
        return _quasi_volatility_state_space(self._params)

    def forecast_variance(self, steps: int) -> npt.NDArray[np.float64]:
        """Expected variance ``E[exp(h_{T+k}) | y]`` for ``k = 1..steps``.

        Propagates the last smoothed log-variance mean and variance
        through the AR(1) and applies the log-normal correction.

        Args:
            steps: Horizons ahead, at least 1.

        Raises:
            SpecificationError: If ``steps`` is not positive.
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        mean = float(self.log_variance[-1])
        var = float(self.log_variance_std[-1] ** 2)
        out = np.empty(steps)
        for k in range(steps):
            mean = self.mu + self.phi * (mean - self.mu)
            var = self.phi**2 * var + self.sigma2
            out[k] = np.exp(mean + 0.5 * var)
        return out

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """Aligned per-observation output."""
        lower, upper = self.volatility_bands()
        return {
            "observed": self.endog,
            "log_variance": self.log_variance,
            "volatility": self.volatility,
            "volatility_lower": lower,
            "volatility_upper": upper,
            "standardized": (self.endog - self.mean) / self.volatility,
        }

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        parts = [self.mean_spec]
        if self.nu is not None:
            parts.append("t")
        if self.rho != 0.0:
            parts.append("leverage")
        parts.append(self.method)
        return f"SV[{', '.join(parts)}]"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        ic = self.information_criteria
        rows: list[tuple[str, str]] = [
            ("mu", f"{self.mu:.4f}"),
            ("phi", f"{self.phi:.4f}"),
            ("sigma2", f"{self.sigma2:.6g}"),
        ]
        if self.nu is not None:
            rows.append(("nu", f"{self.nu:.3f}"))
        if self.rho != 0.0:
            rows.append(("rho", f"{self.rho:.4f}"))
        if self.mean_spec == "constant":
            rows.insert(0, ("mean", f"{self.mean:.6g}"))
        notes: list[str] = []
        if self.nu is not None and self.method == "qml":
            notes.append(
                "Under the linearization the degrees of freedom enter only through "
                "the mean and variance of log(lambda), so nu is weakly identified by "
                "QML; the particle estimators and the samplers see the tail directly."
            )
        criterion = (
            "The likelihood is the quasi-likelihood of the Harvey-Ruiz-Shephard "
            "linearization, exact for the linearized model and consistent for "
            "this one; compare only against other QML fits."
            if self.method == "qml"
            else "The likelihood is a particle estimate of the exact likelihood "
            "under common random numbers and carries Monte Carlo error; the "
            "posterior sampler is the research-grade estimator."
        )
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=(
                ("Method", self.method),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("AIC", f"{ic.aic:.3f}"),
                ("Half-life", f"{self.half_life:.1f}"),
                ("BIC", f"{ic.bic:.3f}"),
            ),
            columns=("", "estimate"),
            rows=tuple(rows),
            notes=(criterion, *notes),
        )


class SV(_StochasticVolatilityModel[SVResult]):
    """The log-AR(1) stochastic-volatility model.

    ``y_t = c + exp(h_t / 2) eps_t``, ``h_{t+1} = mu + phi (h_t - mu) +
    sigma eta_t``, with ``eps`` and ``eta`` independent standard normals.

    Args:
        endog: The observed series (returns, typically).
        mean: ``"constant"`` to estimate ``c``, ``"zero"`` to fix it.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 400
        >>> h = np.empty(n)
        >>> h[0] = -1.0
        >>> for t in range(1, n):
        ...     h[t] = -1.0 + 0.95 * (h[t - 1] + 1.0) + 0.3 * rng.standard_normal()
        >>> y = np.exp(h / 2) * rng.standard_normal(n)
        >>> res = SV(y, mean="zero").fit()
        >>> bool(0.8 < res.phi < 1.0)
        True
    """

    def fit(
        self, *, method: str = "qml", n_particles: int = 500, seed: int | None = None
    ) -> SVResult:
        """Point-estimate the model.

        Args:
            method: ``"qml"`` for Harvey-Ruiz-Shephard quasi-maximum
                likelihood on the linearized model, ``"particle"`` for
                simulated maximum likelihood on the particle filter's
                estimate under common random numbers (Nelder-Mead from the
                QML start).
            n_particles: Particles per evaluation under ``"particle"``.
            seed: The common seed under ``"particle"``.

        Raises:
            SpecificationError: If the method is unknown.
        """
        if method == "qml":
            return SVResult._from_fit(self._fit_quasi(), self)
        if method == "particle":
            return SVResult._from_fit(
                self._fit_particle(n_particles=n_particles, seed=seed, filter_method="bootstrap"),
                self,
            )
        raise SpecificationError(f"method must be 'qml' or 'particle'; got {method!r}.")

    def sample(
        self,
        *,
        method: str = "gibbs",
        n_draws: int = 5000,
        n_burn: int = 1000,
        thin: int = 1,
        n_particles: int = 500,
        prior_mu: tuple[float, float] = (0.0, 10.0),
        prior_phi: tuple[float, float] = (20.0, 1.5),
        prior_sigma2: tuple[float, float] = (2.5, 0.025),
        seed: int | None = None,
    ) -> _SVPosterior:
        """Sample the posterior.

        Both samplers use the same prior: Gaussian ``mu ~ N(m, v)``, Beta
        ``(phi + 1) / 2 ~ Beta(a, b)`` (the Kim-Shephard-Chib default
        ``(20, 1.5)`` puts the persistence near ``0.86`` a priori),
        inverse-gamma ``sigma2 ~ IG(shape, rate)``.

        Args:
            method: ``"gibbs"`` for the Kim-Shephard-Chib mixture sampler,
                ``"particle"`` for particle marginal Metropolis-Hastings.
            n_draws: Total iterations.
            n_burn: Burn-in discarded (the particle chain adapts its
                proposal during burn-in).
            thin: Keep every ``thin``-th post-burn draw.
            n_particles: Particles per likelihood estimate under
                ``"particle"``.
            prior_mu: ``(mean, variance)``.
            prior_phi: ``(a, b)`` of the Beta prior on ``(phi + 1) / 2``.
            prior_sigma2: ``(shape, rate)`` of the inverse-gamma prior.
            seed: Seed.

        Raises:
            SpecificationError: If the method is unknown or the draw
                bookkeeping is inconsistent.
        """
        priors = {"prior_mu": prior_mu, "prior_phi": prior_phi, "prior_sigma2": prior_sigma2}
        if method == "gibbs":
            fit = self._sample_gibbs(n_draws=n_draws, n_burn=n_burn, thin=thin, seed=seed, **priors)
            return _SVPosterior._from_gibbs(fit, self)
        if method == "particle":
            chain, mean = self._sample_chain(
                n_particles=n_particles,
                n_draws=n_draws,
                n_burn=n_burn,
                thin=thin,
                filter_method="bootstrap",
                seed=seed,
                **priors,
            )
            return _SVPosterior._from_chain(chain, mean, self, n_particles=n_particles, seed=seed)
        raise SpecificationError(f"method must be 'gibbs' or 'particle'; got {method!r}.")


class UCSV(_TrendVolatilityModel[_UCSVPosterior]):
    """Stock and Watson's unobserved-components stochastic-volatility model.

    ``y_t = tau_t + exp(h_t / 2) eps_t``, ``tau_{t+1} = tau_t + exp(q_t / 2)
    eta_t``, with ``h`` and ``q`` random walks of innovation standard
    deviation ``gamma``. Its estimator is a Gibbs sampler, so ``fit``
    returns a posterior.

    Args:
        endog: The observed series (inflation, typically).
        gamma: The vol-of-vol standard deviation held fixed for both log
            variances (Stock and Watson use ``0.2``), or ``None`` to
            estimate both under inverse-gamma priors.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 200
        >>> trend = np.cumsum(0.2 * rng.standard_normal(n))
        >>> y = trend + 0.7 * rng.standard_normal(n)
        >>> post = UCSV(y).fit(n_draws=300, n_burn=100, seed=0)
        >>> bool(np.corrcoef(post.trend, trend)[0, 1] > 0.9)
        True
    """

    def fit(
        self,
        *,
        n_draws: int = 3000,
        n_burn: int = 1000,
        thin: int = 1,
        prior_gamma2: tuple[float, float] = (3.0, 0.04),
        seed: int | None = None,
    ) -> _UCSVPosterior:
        """Run the Gibbs sampler and return the posterior.

        Args:
            n_draws: Total iterations.
            n_burn: Burn-in discarded.
            thin: Keep every ``thin``-th post-burn draw.
            prior_gamma2: ``(shape, rate)`` of the inverse-gamma prior on
                each vol-of-vol variance, used only when ``gamma`` is
                ``None``.
            seed: Seed.
        """
        return _UCSVPosterior._from_fit(
            self._sample_gibbs(
                n_draws=n_draws, n_burn=n_burn, thin=thin, prior_gamma2=prior_gamma2, seed=seed
            ),
            self,
        )
