# filepath: /src/cultivars/multivariate/large_dim/factor_volatility.py
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

"""Factor stochastic volatility: a moving covariance matrix built from a few paths.

A multivariate GARCH of even moderate dimension drowns in parameters; a
factor model with stochastic volatility does not. Let ``r`` common factors
drive the panel's comovement, give each factor and each idiosyncratic
error its own latent log-AR(1) variance, and the ``(k, k)`` covariance at
every date is ``Lambda H_t Lambda' + D_t`` -- ``k + r`` volatility paths
and a loading matrix, whatever ``k`` is (Pitt and Shephard 1999; Chib,
Nardari, and Shephard 2006). The time variation in correlations comes from
the factor volatilities moving against the idiosyncratic ones, which is
the mechanism, and it is readable rather than implicit.

Estimation is Bayesian, by an exact Gibbs sampler: the factors are one
Gaussian per period given everything else, the loadings one weighted
regression per row, and every log-variance path and its parameters follow
the Kim-Shephard-Chib blocks the univariate model uses. Identification is
the triangular convention -- ``Lambda`` lower triangular with a unit
diagonal in its leading ``r`` rows, so factor ``j`` is scaled and signed by
series ``j`` and the order of the series is a modelling choice the user
makes. What comes back is a posterior, and the object holds it: loadings,
factors and volatility paths as draws, and the conditional covariance and
correlation as posterior means.

References:
    Pitt, M. K., & Shephard, N. (1999). Time-varying covariances: A factor
        stochastic volatility approach. In *Bayesian Statistics 6* (pp.
        547-570). Oxford University Press.
    Chib, S., Nardari, F., & Shephard, N. (2006). Analysis of high
        dimensional multivariate stochastic volatility models. *Journal of
        Econometrics*, 134(2), 341-371.
    Kastner, G., Fruehwirth-Schnatter, S., & Lopes, H. F. (2017). Efficient
        Bayesian inference for multivariate factor stochastic volatility
        models. *Journal of Computational and Graphical Statistics*, 26(4),
        905-917.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable, _validate_quantiles
from ..._internals import (
    _ConvergenceMixin,
    _FactorVolatilityFit,
    _FactorVolatilityModel,
    _SummaryMixin,
)
from ...exceptions import SpecificationError

__all__ = ["FactorSV", "FactorSVResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class FactorSVResult(_SummaryMixin, _ConvergenceMixin):
    """The posterior of a factor stochastic-volatility model.

    Attributes:
        panel: The observed ``(nobs, k)`` panel.
        series_names: One label per series.
        loading_draws: ``(S, k, r)`` posterior draws of the loading matrix.
        factor_draws: ``(S, T, r)`` posterior draws of the factor paths.
        h_factor_draws: ``(S, T, r)`` posterior draws of the factor log
            variances.
        h_idio_draws: ``(S, T, k)`` posterior draws of the idiosyncratic
            log variances.
        mu_factor_draws: ``(S, r)`` factor log-variance means.
        phi_factor_draws: ``(S, r)`` factor log-variance persistences.
        sigma2_factor_draws: ``(S, r)`` factor log-variance innovation
            variances.
        mu_idio_draws: ``(S, k)`` idiosyncratic log-variance means.
        phi_idio_draws: ``(S, k)`` idiosyncratic persistences.
        sigma2_idio_draws: ``(S, k)`` idiosyncratic innovation variances.
        means: ``(k,)`` series means removed before sampling.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.
    """

    panel: npt.NDArray[np.float64] = field(repr=False)
    series_names: tuple[str, ...]
    loading_draws: npt.NDArray[np.float64] = field(repr=False)
    factor_draws: npt.NDArray[np.float64] = field(repr=False)
    h_factor_draws: npt.NDArray[np.float64] = field(repr=False)
    h_idio_draws: npt.NDArray[np.float64] = field(repr=False)
    mu_factor_draws: npt.NDArray[np.float64] = field(repr=False)
    phi_factor_draws: npt.NDArray[np.float64] = field(repr=False)
    sigma2_factor_draws: npt.NDArray[np.float64] = field(repr=False)
    mu_idio_draws: npt.NDArray[np.float64] = field(repr=False)
    phi_idio_draws: npt.NDArray[np.float64] = field(repr=False)
    sigma2_idio_draws: npt.NDArray[np.float64] = field(repr=False)
    means: npt.NDArray[np.float64] = field(repr=False)
    n_draws: int
    n_burn: int
    thin: int

    @classmethod
    def _from_fit(cls, fit: _FactorVolatilityFit, model: FactorSV) -> FactorSVResult:
        """Assemble the public result from a raw fit and its specification."""
        return cls(
            panel=model.panel,
            series_names=model.series_names,
            loading_draws=fit.loading_draws,
            factor_draws=fit.factor_draws,
            h_factor_draws=fit.h_factor_draws,
            h_idio_draws=fit.h_idio_draws,
            mu_factor_draws=fit.mu_factor_draws,
            phi_factor_draws=fit.phi_factor_draws,
            sigma2_factor_draws=fit.sigma2_factor_draws,
            mu_idio_draws=fit.mu_idio_draws,
            phi_idio_draws=fit.phi_idio_draws,
            sigma2_idio_draws=fit.sigma2_idio_draws,
            means=fit.means,
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
        )

    @property
    def n_series(self) -> int:
        """Number of series."""
        return len(self.series_names)

    @property
    def n_factors(self) -> int:
        """Number of factors."""
        return int(self.loading_draws.shape[2])

    @property
    def nobs(self) -> int:
        """Panel length."""
        return int(self.panel.shape[0])

    @property
    def n_kept(self) -> int:
        """Posterior draws retained."""
        return int(self.loading_draws.shape[0])

    @property
    def loadings(self) -> npt.NDArray[np.float64]:
        """Posterior-mean loading matrix, ``(k, r)``."""
        return np.asarray(self.loading_draws.mean(axis=0), dtype=np.float64)

    @property
    def factors(self) -> npt.NDArray[np.float64]:
        """Posterior-mean factor paths, ``(T, r)``."""
        return np.asarray(self.factor_draws.mean(axis=0), dtype=np.float64)

    def factor_volatility(
        self, factor: int, *, quantiles: Sequence[float] = (0.16, 0.5, 0.84)
    ) -> npt.NDArray[np.float64]:
        """Posterior quantiles of one factor's standard deviation, ``(len(quantiles), T)``.

        Args:
            factor: Factor index.
            quantiles: Probability levels, each in ``(0, 1)``.

        Raises:
            SpecificationError: If the index is out of range or a quantile
                is outside the open unit interval.
        """
        levels = _validate_quantiles(quantiles)
        if not 0 <= int(factor) < self.n_factors:
            raise SpecificationError(f"factor index {factor} is out of range for {self.n_factors}.")
        return np.quantile(np.exp(0.5 * self.h_factor_draws[:, :, int(factor)]), levels, axis=0)

    def idiosyncratic_volatility(
        self, series: str | int, *, quantiles: Sequence[float] = (0.16, 0.5, 0.84)
    ) -> npt.NDArray[np.float64]:
        """Posterior quantiles of one series' idiosyncratic standard deviation.

        Args:
            series: A series label or column index.
            quantiles: Probability levels, each in ``(0, 1)``.

        Returns:
            An array of shape ``(len(quantiles), T)``.

        Raises:
            SpecificationError: If the series is unknown or a quantile is
                outside the open unit interval.
        """
        levels = _validate_quantiles(quantiles)
        column = self._column(series)
        return np.quantile(np.exp(0.5 * self.h_idio_draws[:, :, column]), levels, axis=0)

    def conditional_covariance(self) -> npt.NDArray[np.float64]:
        """Posterior-mean conditional covariance ``Lambda H_t Lambda' + D_t``, ``(T, k, k)``.

        Averaged over draws, so it is the posterior mean of the covariance
        rather than the covariance at the posterior mean; the two differ
        by the posterior uncertainty in loadings and paths.
        """
        n_kept = self.n_kept
        out = np.zeros((self.nobs, self.n_series, self.n_series))
        for index in range(n_kept):
            lam = self.loading_draws[index]
            scaled = np.exp(self.h_factor_draws[index])[:, None, :] * lam[None, :, :]
            out += np.einsum("tij,kj->tik", scaled, lam)
            idx = np.arange(self.n_series)
            out[:, idx, idx] += np.exp(self.h_idio_draws[index])
        return np.asarray(out / n_kept, dtype=np.float64)

    def conditional_correlation(self) -> npt.NDArray[np.float64]:
        """Conditional correlations implied by :meth:`conditional_covariance`, ``(T, k, k)``."""
        cov = self.conditional_covariance()
        sd = np.sqrt(np.einsum("tii->ti", cov))
        return np.asarray(cov / (sd[:, :, None] * sd[:, None, :]), dtype=np.float64)

    def communality(self) -> npt.NDArray[np.float64]:
        """Time-averaged share of each series' variance carried by the factors, ``(k,)``.

        Computed draw by draw from the model's own decomposition
        ``diag(Lambda H_t Lambda') / (diag(Lambda H_t Lambda') + D_t)`` and
        averaged over time and draws.
        """
        shares = np.zeros(self.n_series)
        for index in range(self.n_kept):
            lam = self.loading_draws[index]
            common = np.exp(self.h_factor_draws[index]) @ (lam**2).T
            private = np.exp(self.h_idio_draws[index])
            shares += (common / (common + private)).mean(axis=0)
        return np.asarray(shares / self.n_kept, dtype=np.float64)

    def _column(self, series: str | int) -> int:
        """Resolve a series label or index to a column."""
        if isinstance(series, str):
            if series not in self.series_names:
                raise SpecificationError(f"unknown series {series!r}; have {self.series_names}.")
            return self.series_names.index(series)
        if not 0 <= int(series) < self.n_series:
            raise SpecificationError(f"series index {series} is out of range for {self.n_series}.")
        return int(series)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        lam = self.loadings
        lo, hi = np.quantile(self.loading_draws, [0.16, 0.84], axis=0)
        shares = self.communality()
        rows = []
        for i, name in enumerate(self.series_names):
            cells = []
            for j in range(self.n_factors):
                if i == j:
                    cells.append("1 (fixed)")
                elif i < j:
                    cells.append("0 (fixed)")
                else:
                    cells.append(f"{lam[i, j]:.3f} [{lo[i, j]:.3f}, {hi[i, j]:.3f}]")
            rows.append((name, *cells, f"{shares[i]:.3f}"))
        persistence_f = ", ".join(f"{v:.3f}" for v in self.phi_factor_draws.mean(axis=0))
        notes = (
            "Gibbs sampler with exact blocks: factors one Gaussian per period, loadings "
            "one weighted regression per row, and Kim-Shephard-Chib mixture steps for "
            "every log-variance path. Draws are a Markov chain; loadings and factor "
            "scales mix slowly without the Kastner et al. interweaving, which is not "
            "done here -- thin accordingly.",
            "Identification is the triangular convention: loadings are lower triangular "
            "with a unit diagonal in the leading r rows, so factor j is scaled and signed "
            "by series j. The order of the series is a modelling choice, not a fact.",
            f"Factor log-variance persistence: {persistence_f}. The conditional "
            "covariance moves through r factor volatilities and k idiosyncratic ones; "
            "conditional_correlation() shows what that does to the correlations.",
            "No llf, parameter count, or information criteria are reported: a posterior has none.",
        )
        return SummaryTable(
            title="Factor Stochastic Volatility (posterior)",
            metadata=(
                ("Series", f"{self.n_series}"),
                ("Factors", f"{self.n_factors}"),
                ("Observations", f"{self.nobs}"),
                ("Draws kept", f"{self.n_kept}"),
                ("Burn-in", f"{self.n_burn}"),
                ("Thin", f"{self.thin}"),
            ),
            columns=(
                "series",
                *[f"loading f{j + 1} (68%)" for j in range(self.n_factors)],
                "communality",
            ),
            rows=tuple(rows),
            notes=notes,
        )


class FactorSV(_FactorVolatilityModel[FactorSVResult]):
    """Factor stochastic volatility, Chib-Nardari-Shephard.

    ``y_t = Lambda f_t + eps_t`` with a log-AR(1) variance on every factor
    and every idiosyncratic error: a time-varying covariance matrix for a
    wide panel from ``k + r`` volatility paths. Estimation is by Gibbs
    sampling and the result is a posterior.

    Args:
        panel: The observed ``(nobs, k)`` panel (returns, typically); the
            sample mean of each series is removed.
        n_factors: Number of latent factors, at least one and below ``k``.
        series_names: One label per series. Defaults to ``x1 ... xk``.
            The first ``n_factors`` series scale and sign the factors, so
            put the series that should anchor each factor first.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> n, k = 300, 5
        >>> h = np.zeros(n)
        >>> for t in range(1, n):
        ...     h[t] = 0.95 * h[t - 1] + 0.3 * rng.standard_normal()
        >>> f = np.exp(h / 2) * rng.standard_normal(n)
        >>> lam = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        >>> y = np.outer(f, lam) + 0.5 * rng.standard_normal((n, k))
        >>> res = FactorSV(y, n_factors=1).fit(n_draws=300, n_burn=100, seed=0)
        >>> res.loadings.shape
        (5, 1)
        >>> bool(res.communality()[0] > res.communality()[-1])
        True
    """

    __slots__ = ()

    def fit(
        self,
        *,
        n_draws: int = 3000,
        n_burn: int = 1000,
        thin: int = 1,
        prior_mu: tuple[float, float] = (0.0, 10.0),
        prior_phi: tuple[float, float] = (20.0, 1.5),
        prior_sigma2: tuple[float, float] = (2.5, 0.025),
        loading_prior_precision: float = 1.0,
        seed: int | np.random.Generator | None = None,
    ) -> FactorSVResult:
        """Sample the posterior.

        Every log-variance path carries the Kim-Shephard-Chib prior --
        Gaussian on the mean, Beta on ``(phi + 1) / 2``, inverse-gamma on
        the innovation variance -- and each free loading an independent
        Gaussian prior of the given precision.

        Args:
            n_draws: Total sampler iterations.
            n_burn: Burn-in discarded.
            thin: Keep every ``thin``-th post-burn draw.
            prior_mu: ``(mean, variance)`` of the prior on each
                log-variance mean.
            prior_phi: ``(a, b)`` of the Beta prior on ``(phi + 1) / 2``.
            prior_sigma2: ``(shape, rate)`` of the inverse-gamma prior.
            loading_prior_precision: Prior precision on each free loading.
            seed: Seed or generator.

        Returns:
            The :class:`FactorSVResult`.

        Raises:
            SpecificationError: If the draw bookkeeping or the loading
                prior is inconsistent.
        """
        fit = self._sample(
            n_draws=n_draws,
            n_burn=n_burn,
            thin=thin,
            prior_mu=prior_mu,
            prior_phi=prior_phi,
            prior_sigma2=prior_sigma2,
            loading_prior_precision=loading_prior_precision,
            seed=seed,
        )
        return FactorSVResult._from_fit(fit, self)
