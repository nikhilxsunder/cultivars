# filepath: /src/cultivars/univariate/stochastic_volatility.py
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
r"""Stochastic volatility: variance as a latent state, not a function of the past.

A GARCH variance is a deterministic function of past observations; a
stochastic-volatility variance is a latent process with its own
innovation, observed only through the returns it scales,

.. math::

   y_t = c + \exp(h_t / 2)\,\varepsilon_t,
   \qquad
   h_{t+1} = \mu + \phi\,(h_t - \mu) + \sigma\,\eta_t,
   \qquad
   \varepsilon_t, \eta_t \sim N(0, 1).

That makes the model a nonlinear state space -- the measurement is
multiplicative in the state -- and puts its likelihood
:math:`p(y) = \int p(y \mid h)\,p(h)\,dh` out of closed-form reach. Two
honest routes exist and both are offered. The Kim-Shephard-Chib (1998)
route linearizes by squaring and taking logs,

.. math::

   \log (y_t - c)^2 = h_t + \log \varepsilon_t^2,

then replaces the :math:`\log \chi^2_1` error with a seven-component
Gaussian mixture; conditional on the mixture indicators the model is
linear-Gaussian and a Gibbs sampler is exact. The mixture is a very good
approximation, not the truth. The particle route (Andrieu, Doucet, and
Holenstein, 2010) makes no approximation: a bootstrap particle filter
estimates :math:`p(y)` without bias and a Metropolis-Hastings chain on
that estimate targets the exact posterior, at the cost of a particle
filter per draw. Point estimation is offered for the same model: the
Harvey-Ruiz-Shephard quasi-maximum likelihood, which runs the Kalman
filter on the linearized series with :math:`\log \varepsilon_t^2` treated
as Gaussian with mean :math:`-1.2704` and variance :math:`\pi^2/2`
(fast, consistent, the warm start for everything else), and simulated
maximum likelihood on the particle estimate under common random numbers.

Two commitments shape the surface. First, the two samplers are given the
same prior -- Gaussian on :math:`\mu`, Beta on :math:`(\phi + 1)/2`,
inverse-gamma on :math:`\sigma^2` -- so they should agree, and a
disagreement is the diagnostic that the mixture is binding; the point
estimates say in their summaries which criterion they maximized and that
criteria are comparable only within one method. Second, the latent path
is reported with its uncertainty: every record carries the smoothed log
variance and its standard deviation, the volatility bands are
exponentiated normal bands on that path, and the variance forecast
carries the log-normal correction, so nothing volatility-related is read
off a point path as if it were observed.

The second model here is Stock and Watson's (2007) unobserved-components
stochastic-volatility model,

.. math::

   y_t = \tau_t + \exp(h_t / 2)\,\varepsilon_t,
   \qquad
   \tau_{t+1} = \tau_t + \exp(q_t / 2)\,\eta_t,

a random-walk trend observed through an irregular, each innovation with
a random-walk log variance of innovation standard deviation
:math:`\gamma`. It is the workhorse of trend-inflation measurement and
the model that joins the package's structural decomposition to its
volatility machinery; its estimator is the Gibbs sampler, with the trend
drawn exactly by the simulation smoother under the current variance
paths.

Layout. :class:`SV` validates ``mean``, ``dist`` and ``leverage`` on
``_StochasticVolatilityModel`` in ``_internals``; ``_fit_quasi`` runs a
``_QuasiVolatilityObjective`` through ``_maximize_likelihood`` and
smooths on the linear substrate, ``_fit_particle`` runs a
``_ParticleLikelihoodObjective`` from the QML start, ``_sample_gibbs``
and ``_sample_chain`` are the two samplers, and ``_SVPosterior`` in
``_internals._posteriors`` packs the draws. :class:`UCSV` validates
``gamma`` on ``_TrendVolatilityModel`` and its ``_sample_gibbs`` packs a
``_UCSVPosterior``. The emitters ``_volatility_state_space`` and
``_quasi_volatility_state_space`` in ``_internals._emitters`` build the
nonlinear and linear substrates the results expose as ``state_space``
and ``quasi_state_space``; the priors are
:class:`~cultivars.bayes.priors.VolatilityPrior` and
:class:`~cultivars.bayes.priors.RandomWalkVolatilityPrior`, with the
Kim-Shephard-Chib defaults in ``_core``. The observation-driven
alternative is :mod:`~cultivars.univariate.conditional_variance`; the
constant-variance trend models are
:mod:`~cultivars.univariate.unobserved_components`.

References:
    Taylor, S. J. (1986). *Modelling Financial Time Series*. Wiley.

    Harvey, A. C., Ruiz, E., & Shephard, N. (1994). Multivariate
    stochastic variance models. *Review of Economic Studies*, 61(2),
    247-264.

    Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
    Likelihood inference and comparison with ARCH models. *Review of
    Economic Studies*, 65(3), 361-393.

    Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle Markov
    chain Monte Carlo methods. *Journal of the Royal Statistical Society
    B*, 72(3), 269-342.

    Stock, J. H., & Watson, M. W. (2007). Why has U.S. inflation become
    harder to forecast? *Journal of Money, Credit and Banking*, 39(s1),
    3-33.

Example:
    QML as the warm start and the Gibbs sampler as the estimator, both
    reading the same latent path:

    >>> import numpy as np
    >>> from cultivars._internals import _simulate_stochastic_volatility
    >>> rng = np.random.default_rng(0)
    >>> y, h = _simulate_stochastic_volatility(
    ...     500, mu=-1.0, phi=0.95, sigma2=0.05, mean=0.0, rng=rng
    ... )
    >>> model = SV(y, mean="zero")
    >>> point = model.fit()
    >>> posterior = model.sample(n_draws=400, n_burn=100, seed=0)
    >>> point.method, posterior.method, posterior.n_kept
    ('qml', 'gibbs', 300)
    >>> bool(abs(point.phi - 0.95) < 0.05), bool(abs(posterior.phi_draws.mean() - 0.95) < 0.05)
    (True, True)
    >>> corr = np.corrcoef(np.vstack([h, point.log_variance, posterior.log_variance]))
    >>> bool(corr[0, 1] > 0.5), bool(corr[0, 2] > 0.7)
    (True, True)
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
    _simulate_stochastic_volatility,
    _StochasticVolatilityFit,
    _StochasticVolatilityModel,
    _StochasticVolatilityParameters,
    _SummaryMixin,
    _SVPosterior,
    _TrendVolatilityModel,
    _UCSVPosterior,
    _volatility_state_space,
)
from ..bayes.priors import RandomWalkVolatilityPrior, VolatilityPrior
from ..exceptions import SpecificationError
from ..state_space.linear_gaussian import LinearGaussianSSM
from ..state_space.nonlinear import NonlinearSSM

__all__ = [
    "SV",
    "UCSV",
    "SVResult",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SVResult(_SummaryMixin, _SeriesMixin, _ComparisonMixin):
    r"""A point-estimated stochastic-volatility model.

    The model

    .. math::

        y_t = c + \exp(h_t / 2)\,\varepsilon_t,
        \qquad
        h_{t+1} = \mu + \phi\,(h_t - \mu) + \sigma\,\eta_t,

    with :math:`\varepsilon_t` standard normal or Student-t with
    :math:`\nu` degrees of freedom, :math:`\eta_t` standard normal, and
    :math:`\operatorname{corr}(\varepsilon_t, \eta_t) = \rho` when
    leverage is specified. The log variance is latent, so what the record
    carries beside the parameters is its smoothed path and standard
    deviation, and everything volatility-related is read off those:
    :attr:`volatility` is :math:`\exp(\hat h_t / 2)`, the bands exponentiate
    normal bands on :math:`\hat h_t`, and the variance forecast propagates
    the last smoothed moment through the AR(1) with the log-normal
    correction :math:`E[\exp(h)] = \exp(E[h] + \tfrac12 \operatorname{Var}[h])`.
    Two estimators produce it: the Harvey-Ruiz-Shephard quasi-maximum
    likelihood on the linearized model
    :math:`\log (y_t - c)^2 = h_t + \log \varepsilon_t^2`, where the
    Kalman filter treats :math:`\log \varepsilon_t^2` as Gaussian with
    mean :math:`-1.2704` and variance :math:`\pi^2 / 2`, and simulated
    maximum likelihood on a particle estimate of the exact likelihood.

    Attributes:
        endog: The observed series.
        mean_spec: ``"constant"`` or ``"zero"``.
        mean: The observation mean ``c``.
        mu: Unconditional mean of the log variance.
        phi: Persistence of the log variance.
        sigma2: Innovation variance of the log variance.
        nu: Degrees of freedom of the observation noise, or ``None`` under
            Gaussian tails.
        rho: Correlation between the observation and log-variance
            innovations; ``0.0`` without leverage.
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

    Note:
        QML reads the data only through :math:`\log (y_t - c)^2`, which
        discards the sign of :math:`y_t - c`; a leverage correlation is
        therefore not identified by it and comes back as ``rho = 0.0``
        with the parameter count unchanged, and :math:`\nu` is weakly
        identified because the degrees of freedom enter only through the
        two moments of :math:`\log \varepsilon_t^2`. The samplers on
        :class:`SV` are the estimators for either. The linearized sample
        is :math:`\log\bigl((y_t - c)^2 + 10^{-6}\bigr)`, the offset
        guarding an exact zero. Standard errors are not yet reported by
        this estimator.

    See Also:
        * :class:`SV` -- the specification that produces this, and whose
          ``sample`` returns the posterior record instead.
        * :class:`~cultivars.univariate.conditional_variance.GARCHResult`
          -- observation-driven volatility, where the variance is a
          function of past data rather than a latent state.
        * :class:`~cultivars.state_space.nonlinear.NonlinearSSM` -- what
          :attr:`state_space` returns.

    References:
        Taylor, S. J. (1986). *Modelling Financial Time Series*. Wiley.

        Harvey, A. C., Ruiz, E., & Shephard, N. (1994). Multivariate
        stochastic variance models. *Review of Economic Studies*, 61(2),
        247-264.

        Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
        Likelihood inference and comparison with ARCH models. *Review of
        Economic Studies*, 65(3), 361-393.

    Example:
        A persistent log variance recovered by QML, with the smoothed
        path tracking the true one:

        >>> import numpy as np
        >>> from cultivars._internals import _simulate_stochastic_volatility
        >>> rng = np.random.default_rng(0)
        >>> y, h = _simulate_stochastic_volatility(
        ...     500, mu=-1.0, phi=0.95, sigma2=0.05, mean=0.0, rng=rng
        ... )
        >>> res = SV(y, mean="zero").fit()
        >>> res.method, res.nobs, res.n_params, res.nu, res.rho
        ('qml', 500, 3.0, None, 0.0)
        >>> bool(abs(res.phi - 0.95) < 0.05), bool(np.corrcoef(res.log_variance, h)[0, 1] > 0.5)
        (True, True)
        >>> bool(abs(res.unconditional_variance - y.var()) < 0.05)
        True
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    """``(n,)`` observed series. Kept out of the repr."""

    mean_spec: str
    """``"constant"`` when ``c`` was estimated, ``"zero"`` when fixed."""

    mean: float
    """The observation mean ``c``; ``0.0`` under ``mean_spec="zero"``."""

    mu: float
    """Unconditional mean of the log variance."""

    phi: float
    """AR(1) coefficient of the log variance."""

    sigma2: float
    """Innovation variance of the log variance."""

    nu: float | None
    """Student-t degrees of freedom, or ``None`` under Gaussian tails."""

    rho: float
    """Leverage correlation; ``0.0`` without leverage or under QML."""

    llf: float
    """Quasi-likelihood (QML) or particle likelihood estimate at the optimum."""

    method: str
    """``"qml"`` or ``"particle"``."""

    nobs: int
    """Observations the criterion was evaluated on."""

    n_params: float
    """Free parameters: three for the log variance plus the mean if estimated."""

    log_variance: npt.NDArray[np.float64] = field(repr=False)
    """``(n,)`` smoothed log-variance path. Kept out of the repr."""

    log_variance_std: npt.NDArray[np.float64] = field(repr=False)
    """``(n,)`` smoothed log-variance standard deviations. Kept out of the repr."""

    @classmethod
    def _from_fit(cls, fit: _StochasticVolatilityFit, model: SV) -> SVResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The raw estimator output.
            model: The specification, read for the series and the mean
                choice.

        Returns:
            The assembled result.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> model = SV(rng.standard_normal(200))
            >>> res = SVResult._from_fit(model._fit_quasi(), model)
            >>> res.mean_spec, res.method, res.log_variance.shape
            ('constant', 'qml', (200,))
        """
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
        """The parameter record, rebuilt for the emitters.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = SV(rng.standard_normal(200), mean="zero").fit()
            >>> p = res._params
            >>> (p.mu, p.phi, p.sigma2) == (res.mu, res.phi, res.sigma2)
            True
        """
        return _StochasticVolatilityParameters(
            mu=self.mu, phi=self.phi, sigma2=self.sigma2, mean=self.mean, nu=self.nu, rho=self.rho
        )

    @property
    def volatility(self) -> npt.NDArray[np.float64]:
        """Smoothed volatility ``exp(h_t / 2)``, ``(n,)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = SV(rng.standard_normal(200), mean="zero").fit()
            >>> bool(np.allclose(res.volatility**2, np.exp(res.log_variance)))
            True
        """
        return np.asarray(np.exp(0.5 * self.log_variance), dtype=np.float64)

    def volatility_bands(
        self, *, alpha: float = 0.05
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Pointwise ``1 - alpha`` bands on the volatility from the smoothed log variance.

        Normal bands on the smoothed log variance, exponentiated; they are
        asymmetric about :attr:`volatility` and never cross zero.

        Args:
            alpha: Two-sided tail mass.

        Returns:
            ``(lower, upper)`` arrays of length ``n``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = SV(rng.standard_normal(200), mean="zero").fit()
            >>> lower, upper = res.volatility_bands()
            >>> bool(np.all(lower < res.volatility)), bool(np.all(res.volatility < upper))
            (True, True)
            >>> narrow_lo, narrow_hi = res.volatility_bands(alpha=0.5)
            >>> bool(np.all(narrow_hi - narrow_lo < upper - lower))
            True
        """
        from scipy.stats import norm

        z = float(norm.ppf(1.0 - alpha / 2.0))
        lower = np.exp(0.5 * (self.log_variance - z * self.log_variance_std))
        upper = np.exp(0.5 * (self.log_variance + z * self.log_variance_std))
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    @property
    def half_life(self) -> float:
        """Periods for a log-variance shock to halve, ``log 0.5 / log phi``.

        Returns ``inf`` when ``phi`` is outside ``(0, 1)``.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _simulate_stochastic_volatility
            >>> rng = np.random.default_rng(0)
            >>> y, _ = _simulate_stochastic_volatility(
            ...     300, mu=-1.0, phi=0.95, sigma2=0.05, mean=0.0, rng=rng
            ... )
            >>> res = SV(y, mean="zero").fit()
            >>> bool(5.0 < res.half_life < 60.0)
            True
            >>> bool(np.isclose(res.half_life, np.log(0.5) / np.log(res.phi)))
            True
        """
        if not 0.0 < self.phi < 1.0:
            return float("inf")
        return float(np.log(0.5) / np.log(self.phi))

    @property
    def unconditional_variance(self) -> float:
        r"""``E[exp(h_t)]`` under the stationary log-normal law.

        :math:`\exp\bigl(\mu + \tfrac12 \sigma^2 / (1 - \phi^2)\bigr)`, the
        variance of :math:`y_t - c` implied by the fit.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = SV(rng.standard_normal(200), mean="zero").fit()
            >>> expected = np.exp(res.mu + 0.5 * res.sigma2 / (1 - res.phi**2))
            >>> bool(np.isclose(res.unconditional_variance, expected))
            True
        """
        return float(np.exp(self.mu + 0.5 * self.sigma2 / (1.0 - self.phi**2)))

    @property
    def state_space(self) -> NonlinearSSM:
        """The fitted model on the nonlinear substrate.

        Carries the multiplicative measurement through the
        ``observation_loglik`` hook, so only the particle filter reads it;
        ``particle_filter`` on the estimation sample estimates the exact
        likelihood, and on new data reads that data with this fit.

        Example:
            A particle estimate of the exact likelihood at the QML point:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(200)
            >>> res = SV(y, mean="zero").fit()
            >>> ssm = res.state_space
            >>> ssm.k_states, ssm.k_endog
            (1, 1)
            >>> bool(np.isfinite(ssm.particle_filter(y, n_particles=200, seed=0).loglikelihood))
            True
        """
        return _volatility_state_space(self._params)

    @property
    def quasi_state_space(self) -> LinearGaussianSSM:
        r"""The Harvey-Ruiz-Shephard linearization on the linear substrate.

        Filters ``log((y - c)**2)``, not ``y``; its ``loglikelihood`` on the
        linearized estimation sample reproduces ``llf`` when the method
        was ``"qml"``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(200)
            >>> res = SV(y, mean="zero").fit()
            >>> linearized = np.log((y - res.mean) ** 2 + 1e-6)
            >>> bool(abs(res.quasi_state_space.loglikelihood(linearized) - res.llf) < 1e-8)
            True
        """
        return _quasi_volatility_state_space(self._params)

    def forecast_variance(self, steps: int) -> npt.NDArray[np.float64]:
        r"""Expected variance ``E[exp(h_{T+k}) | y]`` for ``k = 1..steps``.

        Propagates the last smoothed log-variance mean and variance
        through the AR(1) and applies the log-normal correction,

        .. math::

            m_k = \mu + \phi (m_{k-1} - \mu), \qquad
            v_k = \phi^2 v_{k-1} + \sigma^2, \qquad
            E[\exp(h_{T+k})] = \exp(m_k + \tfrac12 v_k),

        so the path converges to :attr:`unconditional_variance`.

        Args:
            steps: Horizons ahead, at least 1.

        Returns:
            An array of shape ``(steps,)``.

        Raises:
            SpecificationError: If ``steps`` is not positive.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = SV(rng.standard_normal(200), mean="zero").fit()
            >>> path = res.forecast_variance(400)
            >>> path.shape, bool(abs(path[-1] - res.unconditional_variance) < 0.02)
            ((400,), True)
            >>> res.forecast_variance(0)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: steps must be at least 1; got 0.
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
        """Aligned per-observation output.

        Returns:
            The series, the smoothed log variance and volatility with its
            95% bands, and the series standardized by the smoothed
            volatility.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = SV(rng.standard_normal(200), mean="zero").fit()
            >>> list(res._series())[:3], list(res._series())[-1]
            (['observed', 'log_variance', 'volatility'], 'standardized')
        """
        lower, upper = self.volatility_bands()
        return {
            "observed": self.endog,
            "log_variance": self.log_variance,
            "volatility": self.volatility,
            "volatility_lower": lower,
            "volatility_upper": upper,
            "standardized": (self.endog - self.mean) / self.volatility,
        }

    def simulate(
        self,
        n: int = 200,
        *,
        seed: int | np.random.Generator | None = None,
        burn: int = 0,
    ) -> npt.NDArray[np.float64]:
        """A fresh sample path at the fitted parameters.

        The log variance starts from its stationary law and the series is
        read from it with the fitted tails and leverage, so no burn-in is
        needed; ``burn`` is honoured for uniformity with the other results.

        Args:
            n: Observations kept.
            seed: Seed or generator.
            burn: Periods discarded from the start.

        Returns:
            An array of shape ``(n,)``.

        Raises:
            SpecificationError: If the counts are not usable or the
                parameters leave the stationary region.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = SV(rng.standard_normal(200), mean="zero").fit()
            >>> path = res.simulate(20000, seed=1)
            >>> path.shape, bool(abs(path.var() - res.unconditional_variance) < 0.1)
            ((20000,), True)
            >>> res.simulate(0)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: n must be positive and burn non-negative; ...
        """
        if n < 1 or burn < 0:
            raise SpecificationError(f"n must be positive and burn non-negative; got {n}, {burn}.")
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        y, _ = _simulate_stochastic_volatility(
            burn + n,
            mu=self.mu,
            phi=self.phi,
            sigma2=self.sigma2,
            mean=self.mean,
            rng=rng,
            nu=self.nu,
            rho=self.rho,
        )
        return np.asarray(y[burn:], dtype=np.float64)

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        The mean choice, then ``t`` and ``leverage`` when present, then
        the method, so fits from different estimators never share a
        label.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(200)
            >>> SV(y, mean="zero").fit()._comparison_label(), SV(y).fit()._comparison_label()
            ('SV[zero, qml]', 'SV[constant, qml]')
        """
        parts = [self.mean_spec]
        if self.nu is not None:
            parts.append("t")
        if self.rho != 0.0:
            parts.append("leverage")
        parts.append(self.method)
        return f"SV[{', '.join(parts)}]"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        Metadata carries the method, criterion, sample size, AIC/BIC and
        the half-life; the estimate table lists the mean when estimated,
        then ``mu``, ``phi``, ``sigma2``, and ``nu``/``rho`` when present;
        the notes say what the criterion is under this method and warn
        about ``nu`` under QML.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> table = SV(rng.standard_normal(200)).fit()._summary_table()
            >>> table.title, [row[0] for row in table.rows], table.metadata[0]
            ('SV[constant, qml] Results', ['mean', 'mu', 'phi', 'sigma2'], ('Method', 'qml'))
            >>> table.notes[0][:45]
            'The likelihood is the quasi-likelihood of the'
        """
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
    r"""The log-AR(1) stochastic-volatility model.

    .. math::

        y_t = c + \exp(h_t / 2)\,\varepsilon_t,
        \qquad
        h_{t+1} = \mu + \phi\,(h_t - \mu) + \sigma\,\eta_t,

    with :math:`\varepsilon_t` and :math:`\eta_t` standard normal and
    independent in the base model; ``dist="t"`` makes
    :math:`\varepsilon_t` Student-t with estimated degrees of freedom and
    ``leverage=True`` correlates the two innovations, but not both at
    once. The variance is a latent state rather than a function of past
    data, which is what separates this from the
    :mod:`~cultivars.univariate.conditional_variance` family: the
    likelihood is an integral over the log-variance path, and each
    estimator here is a way of doing that integral -- the
    Harvey-Ruiz-Shephard linearization (``fit``, QML), simulated maximum
    likelihood on a particle estimate (``fit(method="particle")``), the
    Kim-Shephard-Chib mixture Gibbs sampler and particle marginal
    Metropolis-Hastings (``sample``). ``fit`` returns a point-estimate
    record; ``sample`` returns a posterior.

    Attributes:
        _endog: The validated series.
        _mean_spec: ``"constant"`` or ``"zero"``.
        _tails: Whether the observation noise is Student-t.
        _leverage: Whether the innovations are correlated.

    Args:
        endog: The observed series (returns, typically).
        mean: ``"constant"`` to estimate ``c``, ``"zero"`` to fix it.
        dist: ``"normal"`` or ``"t"`` observation noise.
        leverage: Whether to correlate the observation and log-variance
            innovations; estimated by the particle methods only.

    Raises:
        SpecificationError: If ``mean`` or ``dist`` is unrecognized, or
            ``dist="t"`` is combined with ``leverage=True``.
        DimensionError: If ``endog`` is not one-dimensional or has fewer
            than 50 observations.
        NumericalError: If ``endog`` contains non-finite values.

    See Also:
        * :class:`SVResult` -- what :meth:`fit` returns.
        * :class:`UCSV` -- stochastic volatility on a random-walk trend
          and its irregular.
        * :class:`~cultivars.univariate.conditional_variance.GARCH` -- the
          observation-driven alternative.
        * :class:`~cultivars.bayes.priors.VolatilityPrior` -- the prior
          both samplers share.

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

        The specification is validated at construction:

        >>> SV(y, dist="t", leverage=True)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: heavy tails and leverage are not offered ...
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
            n_particles: Particles per evaluation under ``"particle"``, at
                least 50.
            seed: The common seed under ``"particle"``.

        Returns:
            The fitted :class:`SVResult`.

        Raises:
            SpecificationError: If the method is unknown, or
                ``n_particles`` is below 50 under ``"particle"``.

        Example:
            QML is the fast estimator; the particle estimator starts from
            it and is far slower, so it is not run here:

            >>> import numpy as np
            >>> from cultivars._internals import _simulate_stochastic_volatility
            >>> rng = np.random.default_rng(0)
            >>> y, _ = _simulate_stochastic_volatility(
            ...     300, mu=-1.0, phi=0.95, sigma2=0.05, mean=0.0, rng=rng
            ... )
            >>> res = SV(y, mean="zero").fit()
            >>> res.method, bool(0.85 < res.phi < 1.0)
            ('qml', True)
            >>> SV(y).fit(method="mle")
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: method must be 'qml' or 'particle'; got 'mle'.
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
        prior: VolatilityPrior | None = None,
        seed: int | None = None,
    ) -> _SVPosterior:
        r"""Sample the posterior.

        Both samplers use the same prior: Gaussian ``mu ~ N(m, v)``, Beta
        ``(phi + 1) / 2 ~ Beta(a, b)`` (the Kim-Shephard-Chib default
        ``(20, 1.5)`` puts the persistence near ``0.86`` a priori),
        inverse-gamma ``sigma2 ~ IG(shape, rate)``. The Gibbs sampler
        replaces :math:`\log \varepsilon_t^2` by a seven-component normal
        mixture and draws the log-variance path exactly by simulation
        smoothing; the particle sampler targets the exact posterior
        through a particle estimate of the likelihood, so the two should
        agree, and a disagreement is the diagnostic that the mixture is
        binding.

        Args:
            method: ``"gibbs"`` for the Kim-Shephard-Chib mixture sampler,
                ``"particle"`` for particle marginal Metropolis-Hastings.
            n_draws: Total iterations.
            n_burn: Burn-in discarded (the particle chain adapts its
                proposal during burn-in).
            thin: Keep every ``thin``-th post-burn draw.
            n_particles: Particles per likelihood estimate under
                ``"particle"``.
            prior: :class:`~cultivars.bayes.priors.VolatilityPrior`
                instance; the Kim-Shephard-Chib defaults when ``None``.
            seed: Seed.

        Returns:
            The posterior record, with per-parameter draws, the posterior
            mean log-variance path and the convergence diagnostics.

        Raises:
            SpecificationError: If the method is unknown or the draw
                bookkeeping is inconsistent.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _simulate_stochastic_volatility
            >>> rng = np.random.default_rng(0)
            >>> y, h = _simulate_stochastic_volatility(
            ...     500, mu=-1.0, phi=0.95, sigma2=0.05, mean=0.0, rng=rng
            ... )
            >>> post = SV(y, mean="zero").sample(n_draws=400, n_burn=100, seed=0)
            >>> post.method, post.n_kept, post.phi_draws.shape
            ('gibbs', 300, (300,))
            >>> bool(abs(post.phi_draws.mean() - 0.95) < 0.05)
            True
            >>> bool(np.corrcoef(post.log_variance, h)[0, 1] > 0.7)
            True
            >>> SV(y).sample(n_draws=100, n_burn=200)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: n_draws (100) must exceed n_burn (200).
        """
        if prior is None:
            prior = VolatilityPrior()
        if method == "gibbs":
            fit = self._sample_gibbs(
                n_draws=n_draws, n_burn=n_burn, thin=thin, seed=seed, prior=prior
            )
            return _SVPosterior._from_gibbs(fit, self)
        if method == "particle":
            chain, mean = self._sample_chain(
                n_particles=n_particles,
                n_draws=n_draws,
                n_burn=n_burn,
                thin=thin,
                filter_method="bootstrap",
                seed=seed,
                prior=prior,
            )
            return _SVPosterior._from_chain(chain, mean, self, n_particles=n_particles, seed=seed)
        raise SpecificationError(f"method must be 'gibbs' or 'particle'; got {method!r}.")


class UCSV(_TrendVolatilityModel[_UCSVPosterior]):
    r"""Stock and Watson's unobserved-components stochastic-volatility model.

    .. math::

        y_t = \tau_t + \exp(h_t / 2)\,\varepsilon_t,
        \qquad
        \tau_{t+1} = \tau_t + \exp(q_t / 2)\,\eta_t,

    with :math:`h_t` and :math:`q_t` random walks of innovation standard
    deviation :math:`\gamma`, so the irregular's variance and the trend's
    variance each drift freely. It is the workhorse of trend-inflation
    measurement: the trend is a local level whose signal-to-noise ratio
    is itself time-varying, and the fitted :math:`\tau_t` is the
    permanent component under that reading. Its estimator is a Gibbs
    sampler -- the trend drawn exactly by the simulation smoother under
    the current variance paths, the two log variances by the
    Kim-Shephard-Chib mixture -- so ``fit`` returns a posterior.

    Attributes:
        _endog: The validated series.
        _gamma: The fixed vol-of-vol, or ``None`` when estimated.

    Args:
        endog: The observed series (inflation, typically).
        gamma: The vol-of-vol standard deviation held fixed for both log
            variances (Stock and Watson use ``0.2``), or ``None`` to
            estimate both under inverse-gamma priors.

    Raises:
        SpecificationError: If a fixed ``gamma`` is not strictly positive.
        DimensionError: If ``endog`` is not one-dimensional or has fewer
            than 40 observations.
        NumericalError: If ``endog`` contains non-finite values.

    See Also:
        * :class:`SV` -- stochastic volatility on a series with a fixed
          mean.
        * :class:`~cultivars.univariate.unobserved_components.UnobservedComponents`
          -- the local level with constant variances.
        * :class:`~cultivars.bayes.priors.RandomWalkVolatilityPrior` --
          the prior on the vol-of-vol when it is estimated.

    References:
        Stock, J. H., & Watson, M. W. (2007). Why has U.S. inflation
        become harder to forecast? *Journal of Money, Credit and
        Banking*, 39(s1), 3-33.

        Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
        Likelihood inference and comparison with ARCH models. *Review of
        Economic Studies*, 65(3), 361-393.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 200
        >>> trend = np.cumsum(0.2 * rng.standard_normal(n))
        >>> y = trend + 0.7 * rng.standard_normal(n)
        >>> post = UCSV(y).fit(n_draws=300, n_burn=100, seed=0)
        >>> bool(np.corrcoef(post.trend, trend)[0, 1] > 0.9)
        True
        >>> post.n_kept, post.gamma_fixed, post.trend.shape
        (200, True, (200,))
        >>> UCSV(y, gamma=0.0)
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: gamma must be strictly positive; got 0.0.
    """

    def fit(
        self,
        *,
        n_draws: int = 3000,
        n_burn: int = 1000,
        thin: int = 1,
        prior: RandomWalkVolatilityPrior | None = None,
        seed: int | None = None,
    ) -> _UCSVPosterior:
        """Run the Gibbs sampler and return the posterior.

        Args:
            n_draws: Total iterations.
            n_burn: Burn-in discarded.
            thin: Keep every ``thin``-th post-burn draw.
            prior: :class:`~cultivars.bayes.priors.RandomWalkVolatilityPrior`
                instance, read only when ``gamma`` is estimated; the
                defaults when ``None``.
            seed: Seed.

        Returns:
            The posterior record, with the trend and irregular-volatility
            paths, the per-draw arrays and the convergence diagnostics.

        Raises:
            SpecificationError: If the draw bookkeeping is inconsistent.

        Example:
            Estimating the vol-of-vol rather than fixing it:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + 0.7 * rng.standard_normal(200)
            >>> post = UCSV(y, gamma=None).fit(n_draws=300, n_burn=100, seed=0)
            >>> post.gamma_fixed, post.gamma2_trend_draws.shape, post.n_kept
            (False, (200,), 200)
        """
        return _UCSVPosterior._from_fit(
            self._sample_gibbs(
                n_draws=n_draws,
                n_burn=n_burn,
                thin=thin,
                prior=RandomWalkVolatilityPrior() if prior is None else prior,
                seed=seed,
            ),
            self,
        )
