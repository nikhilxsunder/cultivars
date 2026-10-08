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
r"""Factor stochastic volatility: a moving covariance matrix built from a few paths.

A multivariate GARCH of even moderate dimension drowns in parameters; a
factor model with stochastic volatility does not. Let :math:`r` common
factors drive the panel's comovement and give each factor and each
idiosyncratic error its own latent log-AR(1) variance,

.. math::

   y_t = \Lambda f_t + \varepsilon_t,
   \qquad
   f_t \sim N(0, H_t), \quad \varepsilon_t \sim N(0, D_t),

.. math::

   h_{j,t+1} = \mu_j + \phi_j (h_{jt} - \mu_j) + \sigma_j \xi_{jt},
   \qquad
   H_t = \operatorname{diag}(e^{h_{1t}}, \ldots, e^{h_{rt}}),

and likewise for the :math:`k` idiosyncratic log variances in
:math:`D_t`. The :math:`(k, k)` covariance at every date is then
:math:`\Sigma_t = \Lambda H_t \Lambda' + D_t` -- :math:`k + r` volatility
paths and one loading matrix, whatever :math:`k` is (Pitt and Shephard,
1999; Chib, Nardari and Shephard, 2006). The time variation in
correlations comes from the factor volatilities moving against the
idiosyncratic ones, which is the mechanism, and it is readable rather
than implicit: when a factor's variance rises relative to the noise,
every pair that loads on it correlates more.

Estimation is Bayesian, by an exact Gibbs sampler. Given everything
else the factors are one Gaussian per period, the free loadings one
weighted regression per row, and every log-variance path with its three
parameters follows the Kim-Shephard-Chib mixture blocks the univariate
model uses, so no step is approximate beyond the seven-component mixture
for :math:`\log \chi^2_1`. What comes back is a posterior and the record
holds it as draws; the conditional covariance, the correlations and the
communalities are posterior means computed from those draws on demand.

Two commitments shape the surface. First, identification is stated, not
hidden: :math:`\Lambda` is lower triangular with a unit diagonal in its
leading :math:`r` rows, so factor :math:`j` is scaled and signed by
series :math:`j`, the order of the columns is a modelling choice the
user makes and the summary says so. Second, mixing is reported rather
than assumed. The sampler is the plain Gibbs scheme without the ancillary
interweaving of Kastner, Fruehwirth-Schnatter and Lopes (2017), so
loadings and factor scales mix slowly; every ``*_draws`` field shares the
kept-draw axis and ``convergence`` assesses all of them, the fixed
loadings are reported as never having moved rather than flagged, and the
summary carries no likelihood or information criteria because a
posterior has none.

Layout. :class:`FactorSV` validates the panel through
``_validate_wide_panel`` and the factor count and labels on
``_FactorVolatilityModel`` in ``_internals``; its ``_sample`` runs the
sweep with ``_draw_factors``, ``_draw_loading_rows``,
``_draw_stationary_volatility_path`` and ``_draw_volatility_parameters``
from ``_core._samplers`` under a
:class:`~cultivars.bayes.priors.VolatilityPrior`, and packs a
``_FactorVolatilityFit``. :class:`FactorSVResult` takes the summary from
``_SummaryMixin`` and the chain diagnostics from ``_ConvergenceMixin``.
The constant-variance factor model is
:mod:`~cultivars.multivariate.large_dim.dynamic_factor`; the vector
autoregression with stochastic volatility on its innovations is
:mod:`~cultivars.multivariate.large_dim.volatility`; the univariate law
each path follows is :mod:`~cultivars.univariate.stochastic_volatility`.

References:
    Pitt, M. K., & Shephard, N. (1999). Time-varying covariances: A
    factor stochastic volatility approach. In *Bayesian Statistics 6*
    (pp. 547-570). Oxford University Press.

    Chib, S., Nardari, F., & Shephard, N. (2006). Analysis of high
    dimensional multivariate stochastic volatility models. *Journal of
    Econometrics*, 134(2), 341-371.

    Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
    Likelihood inference and comparison with ARCH models. *Review of
    Economic Studies*, 65(3), 361-393.

    Kastner, G., Fruehwirth-Schnatter, S., & Lopes, H. F. (2017).
    Efficient Bayesian inference for multivariate factor stochastic
    volatility models. *Journal of Computational and Graphical
    Statistics*, 26(4), 905-917.

Example:
    One factor whose variance swells in the middle of the sample; the
    posterior conditional correlation between two series that load on it
    rises there and falls back:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n, k = 300, 4
    >>> h = np.where((n // 3 <= np.arange(n)) & (np.arange(n) < 2 * n // 3), 2.0, 0.0)
    >>> f = np.exp(h / 2) * rng.standard_normal(n)
    >>> y = np.outer(f, [1.0, 0.8, 0.6, 0.4]) + 0.6 * rng.standard_normal((n, k))
    >>> res = FactorSV(y, n_factors=1).fit(n_draws=300, n_burn=100, seed=0)
    >>> corr = res.conditional_correlation()[:, 0, 1]
    >>> middle = corr[n // 3 : 2 * n // 3].mean()
    >>> edges = np.r_[corr[: n // 3], corr[2 * n // 3 :]].mean()
    >>> bool(middle > edges + 0.1)
    True
    >>> vol = res.factor_volatility(0)[1]
    >>> bool(vol[n // 3 : 2 * n // 3].mean() > 1.5 * vol[: n // 3].mean())
    True
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
from ...bayes.priors import VolatilityPrior
from ...exceptions import SpecificationError

__all__ = ["FactorSV", "FactorSVResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class FactorSVResult(_SummaryMixin, _ConvergenceMixin):
    r"""The posterior of a factor stochastic-volatility model.

    The model is

    .. math::

       y_t = \Lambda f_t + \varepsilon_t,
       \qquad
       f_t \sim N\bigl(0, H_t\bigr),
       \quad
       \varepsilon_t \sim N\bigl(0, D_t\bigr),

    with :math:`H_t = \operatorname{diag}(e^{h_{1t}}, \ldots, e^{h_{rt}})`
    and :math:`D_t = \operatorname{diag}(e^{\eta_{1t}}, \ldots,
    e^{\eta_{kt}})`, every log variance a latent AR(1),

    .. math::

       h_{j,t+1} = \mu_j + \phi_j (h_{jt} - \mu_j) + \sigma_j \xi_{jt},

    so the conditional covariance :math:`\Sigma_t = \Lambda H_t \Lambda'
    + D_t` moves through :math:`k + r` volatility paths and one loading
    matrix. Identification is the triangular convention: :math:`\Lambda`
    is lower triangular with a unit diagonal in its leading :math:`r`
    rows, so factor :math:`j` is scaled and signed by series :math:`j`.
    The sampler is an exact Gibbs scheme -- factors one Gaussian per
    period, loadings one weighted regression per row, every log-variance
    path and its three parameters by the Kim-Shephard-Chib mixture
    blocks -- and what comes back is the posterior: the record stores
    the retained draws of every unknown, and the point summaries
    (``loadings``, ``factors``, the conditional covariance and
    correlation, the communalities) are posterior means computed from
    them on demand.

    Note:
        Every ``*_draws`` field shares the kept-draw axis ``S``, which is
        what ``convergence`` from ``_ConvergenceMixin`` assesses. The
        fixed loadings (the unit diagonal and the zeros above it) never
        move and are reported as such rather than flagged. The chain is
        a plain Gibbs sampler without the Kastner, Fruehwirth-Schnatter
        and Lopes (2017) interweaving, so loadings and factor scales mix
        slowly and the volatility paths have low effective sample sizes
        at short runs; thin and lengthen accordingly, and read
        ``convergence`` before the point summaries. The series means are
        removed before sampling and kept in ``means``; ``panel`` is the
        raw input.

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

    See Also:
        * :class:`FactorSV` -- the model that produces this record.
        * :class:`~cultivars.univariate.stochastic_volatility.SV` -- the
          univariate model each volatility path follows.
        * :class:`~cultivars.multivariate.large_dim.dynamic_factor.DFM` -- the
          constant-variance factor model.

    References:
        Pitt, M. K., & Shephard, N. (1999). Time-varying covariances: A
        factor stochastic volatility approach. In *Bayesian Statistics 6*
        (pp. 547-570). Oxford University Press.

        Chib, S., Nardari, F., & Shephard, N. (2006). Analysis of high
        dimensional multivariate stochastic volatility models. *Journal
        of Econometrics*, 134(2), 341-371.

        Kastner, G., Fruehwirth-Schnatter, S., & Lopes, H. F. (2017).
        Efficient Bayesian inference for multivariate factor stochastic
        volatility models. *Journal of Computational and Graphical
        Statistics*, 26(4), 905-917.

    Example:
        One factor with a persistent log variance, loaded onto five
        series with decreasing weights; the posterior recovers the
        loadings, the factor path and its volatility:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n, k = 300, 5
        >>> h = np.zeros(n)
        >>> for t in range(1, n):
        ...     h[t] = 0.95 * h[t - 1] + 0.3 * rng.standard_normal()
        >>> f = np.exp(h / 2) * rng.standard_normal(n)
        >>> lam = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        >>> y = np.outer(f, lam) + 0.5 * rng.standard_normal((n, k))
        >>> res = FactorSV(y, n_factors=1).fit(n_draws=300, n_burn=100, seed=0)
        >>> res.n_series, res.n_factors, res.nobs, res.n_kept
        (5, 1, 300, 200)
        >>> np.round(res.loadings[:, 0], 1)
        array([1. , 0.8, 0.6, 0.4, 0.2])
        >>> bool(np.corrcoef(res.factors[:, 0], f)[0, 1] > 0.9)
        True
        >>> bool(np.corrcoef(res.h_factor_draws.mean(axis=0)[:, 0], h)[0, 1] > 0.7)
        True
    """

    panel: npt.NDArray[np.float64] = field(repr=False)
    """The observed ``(nobs, k)`` panel, means not removed. Kept out of the repr."""
    series_names: tuple[str, ...]
    """One label per series, in column order."""
    loading_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, k, r)`` draws of the loading matrix. Kept out of the repr."""
    factor_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, T, r)`` draws of the factor paths. Kept out of the repr."""
    h_factor_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, T, r)`` draws of the factor log variances. Kept out of the repr."""
    h_idio_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, T, k)`` draws of the idiosyncratic log variances. Kept out of the repr."""
    mu_factor_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, r)`` draws of the factor log-variance means. Kept out of the repr."""
    phi_factor_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, r)`` draws of the factor log-variance persistences. Kept out of the repr."""
    sigma2_factor_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, r)`` draws of the factor log-variance innovation variances. Kept out of the repr."""
    mu_idio_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, k)`` draws of the idiosyncratic log-variance means. Kept out of the repr."""
    phi_idio_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, k)`` draws of the idiosyncratic persistences. Kept out of the repr."""
    sigma2_idio_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, k)`` draws of the idiosyncratic innovation variances. Kept out of the repr."""
    means: npt.NDArray[np.float64] = field(repr=False)
    """``(k,)`` series means removed before sampling. Kept out of the repr."""
    n_draws: int
    """Total sampler iterations, burn-in included."""
    n_burn: int
    """Iterations discarded from the start."""
    thin: int
    """Every ``thin``-th post-burn iteration is kept."""

    @classmethod
    def _from_fit(cls, fit: _FactorVolatilityFit, model: FactorSV) -> FactorSVResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The packed draws from ``_sample``.
            model: The specification the draws were produced for.

        Returns:
            The public posterior record.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal((120, 3))
            >>> model = FactorSV(y, n_factors=1)
            >>> fit = model._sample(
            ...     n_draws=40, n_burn=10, thin=1, prior=VolatilityPrior(),
            ...     loading_prior_precision=1.0, seed=0,
            ... )
            >>> res = FactorSVResult._from_fit(fit, model)
            >>> res.series_names, res.n_kept, bool(np.allclose(res.panel, y))
            (('x1', 'x2', 'x3'), 30, True)
        """
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
        """Number of series, ``k``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 3))
            >>> FactorSV(y, n_factors=1).fit(n_draws=40, n_burn=10, seed=0).n_series
            3
        """
        return len(self.series_names)

    @property
    def n_factors(self) -> int:
        """Number of factors, ``r``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 3))
            >>> FactorSV(y, n_factors=2).fit(n_draws=40, n_burn=10, seed=0).n_factors
            2
        """
        return int(self.loading_draws.shape[2])

    @property
    def nobs(self) -> int:
        """Panel length, ``T``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 3))
            >>> FactorSV(y, n_factors=1).fit(n_draws=40, n_burn=10, seed=0).nobs
            120
        """
        return int(self.panel.shape[0])

    @property
    def n_kept(self) -> int:
        """Posterior draws retained, ``S = (n_draws - n_burn) // thin``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 3))
            >>> res = FactorSV(y, n_factors=1).fit(n_draws=100, n_burn=20, thin=4, seed=0)
            >>> res.n_kept, res.loading_draws.shape[0]
            (20, 20)
        """
        return int(self.loading_draws.shape[0])

    @property
    def loadings(self) -> npt.NDArray[np.float64]:
        """Posterior-mean loading matrix, ``(k, r)``.

        Lower triangular with a unit diagonal in the leading ``r`` rows by
        construction; those cells have no posterior spread.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(1)
            >>> factors = rng.standard_normal((250, 2))
            >>> lam = np.array([[1.0, 0.0], [0.5, 1.0], [0.7, -0.3], [0.2, 0.6]])
            >>> y = factors @ lam.T + 0.4 * rng.standard_normal((250, 4))
            >>> res = FactorSV(y, n_factors=2).fit(n_draws=200, n_burn=50, seed=0)
            >>> res.loadings[0], res.loadings[1, 1]
            (array([1., 0.]), np.float64(1.0))
            >>> bool(np.max(np.abs(res.loadings - lam)) < 0.15)
            True
        """
        return np.asarray(self.loading_draws.mean(axis=0), dtype=np.float64)

    @property
    def factors(self) -> npt.NDArray[np.float64]:
        """Posterior-mean factor paths, ``(T, r)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> f = rng.standard_normal(200)
            >>> y = np.outer(f, [1.0, 0.8, 0.6]) + 0.3 * rng.standard_normal((200, 3))
            >>> res = FactorSV(y, n_factors=1).fit(n_draws=200, n_burn=50, seed=0)
            >>> res.factors.shape, bool(np.corrcoef(res.factors[:, 0], f)[0, 1] > 0.95)
            ((200, 1), True)
        """
        return np.asarray(self.factor_draws.mean(axis=0), dtype=np.float64)

    def factor_volatility(
        self, factor: int, *, quantiles: Sequence[float] = (0.16, 0.5, 0.84)
    ) -> npt.NDArray[np.float64]:
        r"""Posterior quantiles of one factor's standard deviation, ``(len(quantiles), T)``.

        The quantiles of :math:`\exp(h_{jt} / 2)` across draws at each
        date, so the default triple is the posterior median with a 68%
        band.

        Args:
            factor: Factor index.
            quantiles: Probability levels, each in ``(0, 1)``.

        Returns:
            An array of shape ``(len(quantiles), T)``.

        Raises:
            SpecificationError: If the index is out of range or a quantile
                is outside the open unit interval.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> n = 300
            >>> h = np.zeros(n)
            >>> for t in range(1, n):
            ...     h[t] = 0.95 * h[t - 1] + 0.3 * rng.standard_normal()
            >>> f = np.exp(h / 2) * rng.standard_normal(n)
            >>> y = np.outer(f, [1.0, 0.8, 0.6]) + 0.3 * rng.standard_normal((n, 3))
            >>> res = FactorSV(y, n_factors=1).fit(n_draws=300, n_burn=100, seed=0)
            >>> band = res.factor_volatility(0)
            >>> band.shape, bool(np.all(band[0] <= band[1])), bool(np.all(band[1] <= band[2]))
            ((3, 300), True, True)
            >>> bool(np.corrcoef(np.log(band[1]), h / 2)[0, 1] > 0.7)
            True
            >>> res.factor_volatility(1)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: factor index 1 is out of range for 1.
        """
        levels = _validate_quantiles(quantiles)
        if not 0 <= int(factor) < self.n_factors:
            raise SpecificationError(f"factor index {factor} is out of range for {self.n_factors}.")
        return np.quantile(np.exp(0.5 * self.h_factor_draws[:, :, int(factor)]), levels, axis=0)

    def idiosyncratic_volatility(
        self, series: str | int, *, quantiles: Sequence[float] = (0.16, 0.5, 0.84)
    ) -> npt.NDArray[np.float64]:
        r"""Posterior quantiles of one series' idiosyncratic standard deviation.

        The quantiles of :math:`\exp(\eta_{it} / 2)` across draws at each
        date: the part of the series' volatility the factors do not
        carry.

        Args:
            series: A series label or column index.
            quantiles: Probability levels, each in ``(0, 1)``.

        Returns:
            An array of shape ``(len(quantiles), T)``.

        Raises:
            SpecificationError: If the series is unknown or a quantile is
                outside the open unit interval.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> f = rng.standard_normal(200)
            >>> y = np.outer(f, [1.0, 0.8, 0.6]) + 0.3 * rng.standard_normal((200, 3))
            >>> res = FactorSV(y, n_factors=1, series_names=("a", "b", "c")).fit(
            ...     n_draws=200, n_burn=50, seed=0
            ... )
            >>> band = res.idiosyncratic_volatility("b")
            >>> band.shape, bool(np.array_equal(band, res.idiosyncratic_volatility(1)))
            ((3, 200), True)
            >>> bool(0.2 < band[1].mean() < 0.4)
            True
            >>> res.idiosyncratic_volatility("d")
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: unknown series 'd'; have ('a', 'b', 'c').
        """
        levels = _validate_quantiles(quantiles)
        column = self._column(series)
        return np.quantile(np.exp(0.5 * self.h_idio_draws[:, :, column]), levels, axis=0)

    def conditional_covariance(self) -> npt.NDArray[np.float64]:
        r"""Posterior-mean conditional covariance ``Lambda H_t Lambda' + D_t``, ``(T, k, k)``.

        Averaged over draws, so it is the posterior mean of the covariance
        rather than the covariance at the posterior mean; the two differ
        by the posterior uncertainty in loadings and paths. Symmetric and
        positive definite at every date, since each draw's
        :math:`\Lambda H_t \Lambda'` is positive semidefinite and
        :math:`D_t` is positive.

        Returns:
            An array of shape ``(T, k, k)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> f = rng.standard_normal(200)
            >>> y = np.outer(f, [1.0, 0.8, 0.6]) + 0.3 * rng.standard_normal((200, 3))
            >>> res = FactorSV(y, n_factors=1).fit(n_draws=200, n_burn=50, seed=0)
            >>> cov = res.conditional_covariance()
            >>> cov.shape, bool(np.allclose(cov, np.swapaxes(cov, 1, 2)))
            ((200, 3, 3), True)
            >>> bool(min(np.linalg.eigvalsh(c).min() for c in cov) > 0)
            True
            >>> bool(abs(cov[:, 0, 1].mean() - 0.8) < 0.2)
            True
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
        """Conditional correlations implied by :meth:`conditional_covariance`, ``(T, k, k)``.

        The posterior-mean covariance scaled by its own diagonal, so this
        is the correlation of the posterior-mean covariance, not the
        posterior mean of the correlation. Correlations move only
        because factor volatilities move against idiosyncratic ones.

        Returns:
            An array of shape ``(T, k, k)`` with unit diagonals.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> f = rng.standard_normal(200)
            >>> y = np.outer(f, [1.0, 0.8, 0.6]) + 0.3 * rng.standard_normal((200, 3))
            >>> res = FactorSV(y, n_factors=1).fit(n_draws=200, n_burn=50, seed=0)
            >>> corr = res.conditional_correlation()
            >>> bool(np.allclose(np.einsum("tii->ti", corr), 1.0))
            True
            >>> bool(corr[:, 0, 1].mean() > corr[:, 0, 2].mean() > 0.5)
            True
        """
        cov = self.conditional_covariance()
        sd = np.sqrt(np.einsum("tii->ti", cov))
        return np.asarray(cov / (sd[:, :, None] * sd[:, None, :]), dtype=np.float64)

    def communality(self) -> npt.NDArray[np.float64]:
        """Time-averaged share of each series' variance carried by the factors, ``(k,)``.

        Computed draw by draw from the model's own decomposition
        ``diag(Lambda H_t Lambda') / (diag(Lambda H_t Lambda') + D_t)`` and
        averaged over time and draws, so it is in ``(0, 1)`` and ordered
        by how strongly each series loads relative to its own noise.

        Returns:
            An array of shape ``(k,)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> f = rng.standard_normal(200)
            >>> y = np.outer(f, [1.0, 0.8, 0.3]) + 0.3 * rng.standard_normal((200, 3))
            >>> res = FactorSV(y, n_factors=1).fit(n_draws=200, n_burn=50, seed=0)
            >>> share = res.communality()
            >>> share.shape, bool(np.all((share > 0) & (share < 1)))
            ((3,), True)
            >>> bool(share[0] > share[1] > share[2])
            True
        """
        shares = np.zeros(self.n_series)
        for index in range(self.n_kept):
            lam = self.loading_draws[index]
            common = np.exp(self.h_factor_draws[index]) @ (lam**2).T
            private = np.exp(self.h_idio_draws[index])
            shares += (common / (common + private)).mean(axis=0)
        return np.asarray(shares / self.n_kept, dtype=np.float64)

    def _column(self, series: str | int) -> int:
        """Resolve a series label or index to a column.

        Args:
            series: A label from ``series_names`` or a column index.

        Returns:
            The column index.

        Raises:
            SpecificationError: If the label is unknown or the index is
                out of range.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 3))
            >>> res = FactorSV(y, n_factors=1, series_names=("a", "b", "c")).fit(
            ...     n_draws=40, n_burn=10, seed=0
            ... )
            >>> res._column("c"), res._column(0)
            (2, 0)
            >>> res._column(3)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: series index 3 is out of range for 3.
        """
        if isinstance(series, str):
            if series not in self.series_names:
                raise SpecificationError(f"unknown series {series!r}; have {self.series_names}.")
            return self.series_names.index(series)
        if not 0 <= int(series) < self.n_series:
            raise SpecificationError(f"series index {series} is out of range for {self.n_series}.")
        return int(series)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Returns:
            One row per series with each loading as a posterior mean and
            68% band -- the fixed cells of the triangular convention
            printed as such -- and the communality; the notes state the
            sampler, the identification, the factor persistences and
            that a posterior carries no likelihood or criteria.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(1)
            >>> factors = rng.standard_normal((250, 2))
            >>> lam = np.array([[1.0, 0.0], [0.5, 1.0], [0.7, -0.3], [0.2, 0.6]])
            >>> y = factors @ lam.T + 0.4 * rng.standard_normal((250, 4))
            >>> res = FactorSV(y, n_factors=2, series_names=("a", "b", "c", "d")).fit(
            ...     n_draws=200, n_burn=50, seed=0
            ... )
            >>> table = res._summary_table()
            >>> table.columns
            ('series', 'loading f1 (68%)', 'loading f2 (68%)', 'communality')
            >>> table.rows[0][:3], table.rows[1][2]
            (('a', '1 (fixed)', '0 (fixed)'), '1 (fixed)')
            >>> len(table.notes), table.metadata[3]
            (4, ('Draws kept', '150'))
        """
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
    r"""Factor stochastic volatility, Chib-Nardari-Shephard.

    .. math::

       y_t = \Lambda f_t + \varepsilon_t,
       \qquad
       f_{jt} = e^{h_{jt}/2}\,u_{jt},
       \quad
       \varepsilon_{it} = e^{\eta_{it}/2}\,v_{it},

    with a log-AR(1) variance on every factor and every idiosyncratic
    error, so a wide panel's time-varying covariance
    :math:`\Lambda H_t \Lambda' + D_t` is built from :math:`k + r`
    volatility paths and one loading matrix rather than the
    :math:`O(k^2)` parameters a multivariate GARCH needs. Estimation is
    by an exact Gibbs sampler and the result is a posterior, not a point
    estimate: :class:`FactorSVResult` holds the retained draws of the
    loadings, the factors and every volatility path.

    Identification is the triangular convention: :math:`\Lambda` is
    lower triangular with a unit diagonal in its leading :math:`r` rows,
    so factor :math:`j` is measured in the units of series :math:`j` and
    signed by it. The ordering of the panel's columns is therefore a
    modelling decision -- the first ``n_factors`` series anchor the
    factors -- and the posterior is not invariant to it. The sample mean
    of each series is removed before sampling and kept on the result as
    ``means``.

    Attributes:
        _panel: The validated ``(nobs, k)`` panel as a float array.
        _n_factors: The number of latent factors.
        _series_names: One label per column.

    Args:
        panel: The observed ``(nobs, k)`` panel (returns, typically); the
            sample mean of each series is removed.
        n_factors: Number of latent factors, at least one and below ``k``.
        series_names: One label per series. Defaults to ``x1 ... xk``.
            The first ``n_factors`` series scale and sign the factors, so
            put the series that should anchor each factor first.

    Raises:
        SpecificationError: If ``n_factors`` is not a positive integer or
            ``series_names`` does not have one entry per column.
        DimensionError: If the panel is not two-dimensional, has fewer
            than 50 rows, or has no more columns than ``n_factors``.

    See Also:
        * :class:`FactorSVResult` -- the posterior record ``fit``
          returns.
        * :class:`~cultivars.multivariate.large_dim.dynamic_factor.DFM` --
          the constant-variance factor model.
        * :class:`~cultivars.univariate.stochastic_volatility.SV` -- the
          univariate volatility law every path follows.
        * :class:`~cultivars.bayes.priors.VolatilityPrior` -- the prior
          on that law.

    References:
        Chib, S., Nardari, F., & Shephard, N. (2006). Analysis of high
        dimensional multivariate stochastic volatility models. *Journal
        of Econometrics*, 134(2), 341-371.

        Pitt, M. K., & Shephard, N. (1999). Time-varying covariances: A
        factor stochastic volatility approach. In *Bayesian Statistics 6*
        (pp. 547-570). Oxford University Press.

    Example:
        >>> import numpy as np
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
        >>> FactorSV(y, n_factors=5)
        Traceback (most recent call last):
        cultivars.exceptions.DimensionError: n_factors (5) must be below the number of series (5).
    """

    __slots__ = ()

    def fit(
        self,
        *,
        n_draws: int = 3000,
        n_burn: int = 1000,
        thin: int = 1,
        prior: VolatilityPrior | None = None,
        loading_prior_precision: float = 1.0,
        seed: int | np.random.Generator | None = None,
    ) -> FactorSVResult:
        r"""Sample the posterior.

        Every log-variance path carries the Kim-Shephard-Chib prior --
        Gaussian on the mean, Beta on :math:`(\phi + 1) / 2`,
        inverse-gamma on the innovation variance -- and each free loading
        an independent :math:`N(0, 1 / \text{precision})` prior. One
        Gibbs sweep draws the factors (one Gaussian per period), the free
        loadings (one weighted regression per row), and every log-variance
        path with its three parameters through the mixture blocks; the
        first ``n_burn`` sweeps are discarded and every ``thin``-th of the
        rest is kept, so ``(n_draws - n_burn) // thin`` draws come back.
        The same ``seed`` reproduces the chain exactly.

        Args:
            n_draws: Total sampler iterations.
            n_burn: Burn-in discarded, below ``n_draws``.
            thin: Keep every ``thin``-th post-burn draw, at least 1.
            prior: The prior on each factor's volatility law, an instance of
                :class:`~cultivars.bayes.priors.VolatilityPrior` or
                ``None`` to use the default Kim-Shephard-Chib prior.
            loading_prior_precision: Prior precision on each free loading,
                positive; larger shrinks the free loadings toward zero.
            seed: Seed or generator.

        Returns:
            The :class:`FactorSVResult`.

        Raises:
            SpecificationError: If ``n_burn`` is not below ``n_draws``,
                ``thin`` is below 1, or ``loading_prior_precision`` is not
                positive.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> f = rng.standard_normal(200)
            >>> y = np.outer(f, [1.0, 0.8, 0.6]) + 0.3 * rng.standard_normal((200, 3))
            >>> model = FactorSV(y, n_factors=1)
            >>> res = model.fit(n_draws=120, n_burn=40, thin=2, seed=0)
            >>> res.n_kept, res.n_draws, res.n_burn, res.thin
            (40, 120, 40, 2)
            >>> again = model.fit(n_draws=120, n_burn=40, thin=2, seed=0)
            >>> bool(np.array_equal(res.loading_draws, again.loading_draws))
            True
            >>> tight = model.fit(
            ...     n_draws=120, n_burn=40, thin=2, seed=0, loading_prior_precision=100.0
            ... )
            >>> bool(abs(tight.loadings[1, 0]) < abs(res.loadings[1, 0]))
            True
            >>> model.fit(n_draws=100, n_burn=100)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: n_draws (100) must exceed n_burn (100).
        """
        fit = self._sample(
            n_draws=n_draws,
            n_burn=n_burn,
            thin=thin,
            prior=VolatilityPrior() if prior is None else prior,
            loading_prior_precision=loading_prior_precision,
            seed=seed,
        )
        return FactorSVResult._from_fit(fit, self)
