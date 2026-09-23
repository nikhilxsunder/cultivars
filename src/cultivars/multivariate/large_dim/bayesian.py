# filepath: /src/cultivars/multivariate/large_dim/bayesian.py
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

r"""The conjugate Bayesian VAR: an exact posterior, and the number that ranks priors.

This module is the conjugate core of the large-dimensional family. It
defines the public pair :class:`BVAR` and :class:`BVARResult` and nothing
else; the estimator itself lives one layer down. A Bayesian VAR earns its
place here because shrinkage is what makes a large system estimable at all:
under the Normal-inverse-Wishart form of the Minnesota prior a
hundred-variable VAR is routine (Banbura, Giannone & Reichlin 2010), and the
prior's tightness, not the sample size, sets the effective dimensionality.
The posterior over coefficients *and* innovation covariance is exact, drawn
rather than approximated, so every band the result reports is a posterior
statement, not an asymptotic one, and every propagated object -- impulse
responses, the predictive -- is propagated draw by draw.

Two design commitments distinguish the surface. First, conjugacy is verified,
not assumed. The prior's coefficient variances must factor as an
equation-scale times a column-profile, :math:`\sigma_i^2 \cdot \omega_j`,
which is the Minnesota structure with the cross-equation weight pinned at
one. A prior that keeps Litterman's extra cross-variable shrinkage is refused
at fit with directions to the per-equation point path, because the two are
different estimators with different claims, not the same object with
different numbers; the priors this model refuses are what ``gibbs.py``
exists to accept. Second, the scalar the result reports is the log
*marginal likelihood* of the sample, computed in closed form with any
dummy-observation content divided out (Giannone, Lenza & Primiceri 2015).
Differences of that number across priors are log Bayes factors, which is
what hyperparameter choice should maximize, and ``hierarchical.py`` is
built on this model to do exactly that. There is deliberately no ``llf`` and
no information criterion on the result: a posterior has neither, and the
marginal likelihood is the honest replacement.

Layout. :class:`BVAR` is a thin specification class: it substitutes the
default prior, guards the order, and delegates to
:meth:`_BayesianVectorAutoRegressionModel._fit_conjugate`, which assembles
the design and the prior's dummy rows, runs the closed-form update through
``_solvers._conjugate_posterior``, and packs the numerics into a
:class:`_VectorConjugateFit`. :meth:`BVARResult._from_fit` joins that packed
fit with the specification into the frozen public record. The result's own
code is the propagation -- lag-stack slicing, companion powers for the
responses, draw-by-draw simulation for the predictive -- and the summary;
:meth:`~BVARResult.summary`, :meth:`~BVARResult.convergence`,
:meth:`~BVARResult.simulate`, and :meth:`~BVARResult.posterior_replications`
come from the shared mixins. The other Bayesian VARs in this package share
the layout and differ in the innovation model or the sampler:
``student.py`` for Student-t innovations, ``volatility.py`` for stochastic
volatility, ``gibbs.py`` for non-conjugate priors, ``hierarchical.py`` for
estimated tightness.

References:
    Litterman, R. B. (1986). Forecasting with Bayesian vector
    autoregressions: Five years of experience. *Journal of Business &
    Economic Statistics*, 4(1), 25-38.

    Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and conditional
    projection using realistic prior distributions. *Econometric Reviews*,
    3(1), 1-100.

    Kadiyala, K. R., & Karlsson, S. (1997). Numerical methods for estimation
    and inference in Bayesian VAR-models. *Journal of Applied Econometrics*,
    12(2), 99-132.

    Banbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector
    auto regressions. *Journal of Applied Econometrics*, 25(1), 71-92.

    Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for
    vector autoregressions. *Review of Economics and Statistics*, 97(2),
    436-451.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable, Trend, companion_matrix
from ..._internals import (
    _BayesianVectorAutoRegressionModel,
    _ConvergenceMixin,
    _MarginalLikelihoodSelection,
    _Prior,
    _ReplicationMixin,
    _SummaryMixin,
    _VectorConjugateFit,
)
from ...bayes import NormalInverseWishartPrior
from ...exceptions import SpecificationError

__all__ = ["BVAR", "BVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class BVARResult(_SummaryMixin, _ConvergenceMixin, _ReplicationMixin):
    r"""A fitted conjugate Bayesian VAR: an exact posterior over :math:`(B, \Sigma)`.

    The object :meth:`BVAR.fit` returns. The posterior over the coefficient
    matrix and the innovation covariance is Normal-inverse-Wishart in closed
    form, so nothing here is approximated by a Markov chain: the ``S``
    retained draws are independent samples from that posterior, point
    summaries are posterior means, and every method that propagates the
    system -- :meth:`irf`, :meth:`irf_bands`, :meth:`forecast`,
    :meth:`forecast_paths`, :meth:`posterior_replications`, and
    :meth:`simulate` at a draw -- propagates each draw separately. Their
    bands are therefore the posterior of the propagated object, not the
    propagation of a point.

    The system is

    .. math::

       y_t = c + \sum_{l=1}^{p} A_l\, y_{t-l} + u_t,
       \qquad u_t \sim \mathcal{N}(0, \Sigma),

    stacked over the effective sample as :math:`Y = X B + U` with
    :math:`Y` of shape :math:`(T - p) \times k`, :math:`X` of shape
    :math:`(T - p) \times w`, and :math:`B` of shape :math:`w \times k`. The
    prior is the conjugate Minnesota form,

    .. math::

       \Sigma \sim \mathcal{IW}(S_0, \nu_0),
       \qquad
       B \mid \Sigma \sim \mathcal{MN}(B_0,\, \Omega,\, \Sigma),

    that is, :math:`\operatorname{vec}(B) \mid \Sigma \sim
    \mathcal{N}(\operatorname{vec}(B_0), \Sigma \otimes \Omega)`, with
    :math:`\nu_0 = k + 2`, :math:`S_0 = \operatorname{diag}(s_1^2, \ldots,
    s_k^2)` built from the per-variable univariate residual scales, and
    :math:`\Omega` the diagonal Minnesota variance profile with the
    cross-equation weight pinned at one. That Kronecker factorization is what
    makes the update exact:

    .. math::

       K = \Omega^{-1} + X^\top X,
       \qquad
       \bar B = K^{-1}\left(\Omega^{-1} B_0 + X^\top Y\right),

    .. math::

       \bar S = S_0 + Y^\top Y + B_0^\top \Omega^{-1} B_0
       - \bar B^\top K \bar B,
       \qquad
       \bar\nu = \nu_0 + n,

    where :math:`n` counts every row updated on, sample and dummy rows
    together. Then :math:`\Sigma \mid Y \sim \mathcal{IW}(\bar S, \bar\nu)`
    and :math:`B \mid \Sigma, Y \sim \mathcal{MN}(\bar B, K^{-1}, \Sigma)`.
    :attr:`beta_mean` is :math:`\bar B`, :attr:`sigma_u` is the exact
    posterior mean :math:`\bar S / (\bar\nu - k - 1)`, and
    :attr:`posterior_df` is :math:`\bar\nu`.

    The scalar this result reports for model comparison is the log marginal
    likelihood, the matrix-variate-*t* normalizing-constant ratio in closed
    form,

    .. math::

       \log p(Y) = -\tfrac{nk}{2}\log\pi
       - \tfrac{k}{2}\bigl(\log|K| + \log|\Omega|\bigr)
       + \tfrac{\nu_0}{2}\log|S_0| - \tfrac{\bar\nu}{2}\log|\bar S|
       + \log\Gamma_k\!\bigl(\tfrac{\bar\nu}{2}\bigr)
       - \log\Gamma_k\!\bigl(\tfrac{\nu_0}{2}\bigr).

    When the prior contributes dummy observations :math:`Y_d` (a
    sum-of-coefficients or dummy-initial-observation component), they are
    part of the prior, not of the sample, so :attr:`log_marginal_likelihood`
    is :math:`\log p(Y \mid Y_d) = \log p(Y, Y_d) - \log p(Y_d)`. That makes
    the number comparable across priors with different dummy content:
    differences across priors on the same sample are log Bayes factors, and
    the hierarchical layer, :class:`~cultivars.multivariate.HierarchicalBVAR`,
    chooses hyperparameters by maximizing it.

    Deliberately absent: ``llf``, ``n_params``, and information criteria. A
    posterior has none of them, and the marginal likelihood is the honest
    replacement.

    Shapes:
        T: Rows of :attr:`endog`. The first ``p`` rows are the presample the
            fit conditions on.
        p: Autoregressive order, :attr:`order`.
        k: Endogenous variables, :attr:`k_endog`.
        n_det: Deterministic regressors per equation: 0, 1, or 2 for
            :attr:`trend` ``"n"``, ``"c"``, or ``"ct"``.
        w: Regressors per equation, ``n_det + k * p``.
        S: Retained posterior draws, :attr:`n_kept`.

    Attributes:
        endog: The observed panel, ``(T, k)``, including the presample.
        names: Variable labels in column order. Their order is also the
            recursive ordering behind every orthogonalized impulse response.
        order: Autoregressive order ``p``.
        trend: Deterministic specification, one of ``"n"``, ``"c"``,
            ``"ct"``.
        prior_label: Short description of the prior the posterior was
            updated under, as it appears in the summary and in comparison
            tables.
        coefficients: ``(p, k, k)`` lag stack at the posterior mean:
            ``coefficients[l - 1]`` is :math:`A_l`, and its ``[i, j]`` entry
            is the response of equation ``i`` to variable ``j`` at lag
            ``l``. Each slice is the transpose of the corresponding lag
            block of :attr:`beta_mean`.
        deterministic: ``(n_det, k)`` deterministic block at the posterior
            mean: the constant row, then the trend row when present.
        beta_mean: ``(w, k)`` posterior mean :math:`\bar B` in design-row
            order: the deterministic rows first, then the lag blocks
            lag-major, so row ``n_det + (l - 1) * k + j`` carries variable
            ``j`` at lag ``l``; columns index equations. The row labels
            :meth:`credible_interval` accepts are ``"const"``, ``"trend"``,
            and ``"{name}.L{lag}"``.
        sigma_u: ``(k, k)`` exact posterior mean of the innovation
            covariance, :math:`\bar S / (\bar\nu - k - 1)`; not the average
            of :attr:`sigma_draws`, which converges to it.
        beta_draws: ``(S, w, k)`` independent coefficient draws, laid out
            as :attr:`beta_mean`.
        sigma_draws: ``(S, k, k)`` independent covariance draws, paired
            with :attr:`beta_draws` draw by draw.
        log_marginal_likelihood: Log marginal likelihood of the sample given
            the prior, dummy-observation content divided out.
        posterior_df: Inverse-Wishart posterior degrees of freedom
            :math:`\bar\nu = k + 2 + \text{nobs} + \text{n\_dummy}`.
        resid: ``(nobs, k)`` residuals at the posterior mean, aligned with
            ``endog[p:]``.
        fittedvalues: ``(nobs, k)`` one-step-ahead means at the posterior
            mean, aligned with ``endog[p:]``.
        nobs: Effective sample size ``T - p``; dummy rows excluded.
        n_dummy: Artificial rows the prior contributed to the update.

    Note:
        Every three-number summary on this result -- :meth:`credible_interval`,
        :meth:`irf_bands`, :meth:`forecast` -- is ``(16th percentile, mean,
        84th percentile)`` taken pointwise across the draws: a 68% central
        credible band, one standard deviation wide under normality. Take
        other quantiles from :attr:`beta_draws`, :attr:`sigma_draws`, or
        :meth:`forecast_paths` directly.

        The draws are independent, so :meth:`convergence` exists for
        interface parity with the Gibbs-sampled members of the family rather
        than as a diagnostic: R-hat is one by construction, and an
        effective-sample flag means only that ``n_draws`` was small.

    Warning:
        Orthogonalized responses rotate each draw by its own Cholesky factor,
        so the order of :attr:`names` is an identifying assumption exactly as
        in a recursive SVAR. Reorder the columns of ``endog`` before fitting
        to change it.

    Example:
        Simulate a stable bivariate system, fit it, and read the posterior:

        >>> import numpy as np
        >>> from cultivars.multivariate import BVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((120, 2))
        >>> for t in range(1, 120):
        ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
        >>> res = BVAR(y, order=1, names=("growth", "inflation")).fit(n_draws=200, seed=0)
        >>> res.beta_draws.shape, res.sigma_draws.shape
        ((200, 3, 2), (200, 2, 2))
        >>> res.nobs, res.n_dummy, res.posterior_df
        (119, 0, 123.0)
        >>> low, mean, high = res.credible_interval("growth", "growth.L1")
        >>> bool(low < mean < high)
        True
        >>> res.stable_share
        1.0

        Propagation is draw by draw, so the bands are posterior bands of the
        propagated object:

        >>> res.irf_bands(8).shape
        (9, 2, 2, 3)
        >>> res.forecast(4).shape
        (4, 2, 3)
        >>> res.forecast_paths(4, seed=0).shape
        (200, 4, 2)

        Rank priors on the same sample by log marginal likelihood; the
        difference is a log Bayes factor:

        >>> from cultivars.bayes import NormalInverseWishartPrior, compare
        >>> tight = NormalInverseWishartPrior(tightness=0.02)
        >>> alt = BVAR(y, order=1, names=("growth", "inflation"), prior=tight)
        >>> alt = alt.fit(n_draws=200, seed=0)
        >>> round(res.log_marginal_likelihood - alt.log_marginal_likelihood, 1)
        6.1
        >>> compare(res.marginal_likelihood(), alt.marginal_likelihood()).rows[0][0]
        'BVAR(1), niw(l1=0.2, l3=1, l4=100)'

    See Also:
        * :class:`BVAR` -- the model whose :meth:`~BVAR.fit` produces this
          result.
        * :class:`~cultivars.multivariate.HierarchicalBVAR` -- treats the
          prior tightnesses as unknowns and integrates over them.
        * :class:`~cultivars.multivariate.GibbsBVAR` -- the same system under
          an independent Normal-Wishart prior, sampled rather than exact.
        * :class:`~cultivars.multivariate.StudentBVAR` and
          :class:`~cultivars.multivariate.BVARSV` -- heavy-tailed and
          stochastic-volatility members of the family.
        * :class:`~cultivars.multivariate.VAR` -- the point-estimate path,
          which also accepts a :class:`~cultivars.bayes.MinnesotaPrior` with
          cross-equation shrinkage.
        * :func:`~cultivars.bayes.compare` -- ranks :meth:`marginal_likelihood`
          records across models.
        * :func:`~cultivars.bayes.posterior_predictive_check` -- scores
          :meth:`posterior_replications` against :attr:`observed`.
        * :func:`~cultivars.forecast.fan_chart` -- draws
          :meth:`forecast_paths` as a fan.

    References:
        Litterman, R. B. (1986). Forecasting with Bayesian vector
        autoregressions: Five years of experience. *Journal of Business &
        Economic Statistics*, 4(1), 25-38.

        Kadiyala, K. R., & Karlsson, S. (1997). Numerical methods for
        estimation and inference in Bayesian VAR-models. *Journal of Applied
        Econometrics*, 12(2), 99-132.

        Banbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian
        vector auto regressions. *Journal of Applied Econometrics*, 25(1),
        71-92.

        Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection
        for vector autoregressions. *Review of Economics and Statistics*,
        97(2), 436-451.

        Karlsson, S. (2013). Forecasting with Bayesian vector autoregression.
        In G. Elliott & A. Timmermann (Eds.), *Handbook of Economic
        Forecasting* (Vol. 2A, pp. 791-897). Elsevier.
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    """The observed panel, ``(T, k)``, including the presample.

    The first ``p`` rows are the presample the fit conditions on; every
    ``(nobs, k)`` array on this result -- :attr:`resid`,
    :attr:`fittedvalues`, :attr:`observed` -- aligns with ``endog[p:]``.
    """
    names: tuple[str, ...]
    """Variable labels, one per column of :attr:`endog`, in column order.

    They name the equations of :attr:`beta_mean` and the lag blocks of
    :attr:`coefficients`, and they are the order in which each draw's
    Cholesky factor is applied, so for :meth:`irf` and :meth:`irf_bands`
    with ``orthogonalized=True`` this order is the recursive identifying
    assumption.
    """
    order: int
    """Autoregressive order ``p``: the number of lag blocks in :attr:`coefficients`."""
    trend: str
    """Deterministic specification: ``"n"`` (none), ``"c"`` (constant), or ``"ct"``.

    Sets ``n_det``, the number of rows in :attr:`deterministic` and of
    leading rows in :attr:`beta_mean`: 0, 1, or 2 respectively.
    """
    prior_label: str
    """Short description of the prior the posterior was updated under.

    A composition renders as its components joined with ``+``, for example
    ``"niw(l1=0.2, l3=1, l4=100) + soc(1) + dio(1)"``. It is the ``Prior``
    line of :meth:`summary` and the ``source`` of the record
    :meth:`marginal_likelihood` returns, so two fits differing only in
    prior are told apart in a :func:`~cultivars.bayes.compare` table.
    """
    coefficients: npt.NDArray[np.float64] = field(repr=False)
    r"""``(p, k, k)`` lag stack at the posterior mean.

    ``coefficients[l - 1]`` is :math:`A_l`, and its ``[i, j]`` entry is the
    response of equation ``i`` to variable ``j`` at lag ``l``, so that
    ``coefficients[l - 1] @ y[t - l]`` is the lag-``l`` contribution to
    the one-step mean. Each slice is the transpose of the corresponding lag
    block of :attr:`beta_mean`. Empty, ``(0, k, k)``, when ``order`` is
    zero.
    """
    deterministic: npt.NDArray[np.float64] = field(repr=False)
    """``(n_det, k)`` deterministic block at the posterior mean.

    The constant row, then the trend row when :attr:`trend` is ``"ct"``;
    the first ``n_det`` rows of :attr:`beta_mean`. The trend regressor is
    the observation index ``t = p + 1, ..., T`` on the effective sample and
    continues as ``T + h`` in :meth:`forecast_paths`.
    """
    beta_mean: npt.NDArray[np.float64] = field(repr=False)
    r"""``(w, k)`` posterior mean coefficient matrix :math:`\bar B`.

    Rows follow the design order -- the ``n_det`` deterministic rows first,
    then the lag blocks lag-major, so row ``n_det + (l - 1) * k + j``
    carries variable ``j`` at lag ``l`` -- and columns index equations, in
    the order of :attr:`names`. The row labels :meth:`credible_interval`
    accepts are ``"const"``, ``"trend"``, and ``"{name}.L{lag}"``, in that
    same order. It is the mean of :attr:`beta_draws` in the limit, but it
    is computed exactly, :math:`\bar B = K^{-1}(\Omega^{-1} B_0 + X^\top Y)`,
    not averaged.
    """
    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    r"""``(k, k)`` exact posterior mean of the innovation covariance.

    :math:`\bar S / (\bar\nu - k - 1)`, the mean of the inverse-Wishart
    posterior, not the average of :attr:`sigma_draws`, which converges to it
    as the draw count grows. It is the covariance :meth:`simulate` uses at
    the posterior mean (``draw=None``).
    """
    beta_draws: npt.NDArray[np.float64] = field(repr=False)
    r"""``(S, w, k)`` independent posterior draws of the coefficient matrix.

    Each draw is laid out exactly as :attr:`beta_mean`, and draw ``s`` was
    generated conditional on ``sigma_draws[s]`` from the matrix-normal
    conditional :math:`B \mid \Sigma, Y \sim \mathcal{MN}(\bar B, K^{-1},
    \Sigma)`. The draws are i.i.d., not a chain: no burn-in or thinning was
    applied and none is needed. Every propagating method -- :meth:`irf`,
    :meth:`irf_bands`, :meth:`forecast_paths`, :meth:`posterior_replications`
    -- iterates over this axis, so its length ``S`` is :attr:`n_kept`.
    """
    sigma_draws: npt.NDArray[np.float64] = field(repr=False)
    r"""``(S, k, k)`` independent posterior draws of the innovation covariance.

    Draw ``s`` is :math:`\Sigma^{(s)} \sim \mathcal{IW}(\bar S, \bar\nu)`
    and pairs with ``beta_draws[s]``; propagating methods use the two
    together, so a draw's impulse responses or forecast paths are computed
    under its own covariance. Each slice is symmetric positive definite.
    """
    log_marginal_likelihood: float
    r"""Log marginal likelihood of the sample given the prior, in closed form.

    :math:`\log p(Y \mid Y_d) = \log p(Y, Y_d) - \log p(Y_d)` when the
    prior contributes dummy observations :math:`Y_d`, and :math:`\log p(Y)`
    otherwise; the dummy rows are part of the prior, so their contribution
    is divided out and the number stays comparable across priors with
    different dummy content. Differences across priors fitted to the same
    sample are log Bayes factors. It is exact -- the matrix-variate-*t*
    normalizing-constant ratio -- so :meth:`marginal_likelihood` reports it
    with zero Monte Carlo error. It is not a log likelihood and has no
    parameter count attached; there is no ``llf`` on this result.
    """
    posterior_df: float
    r"""Inverse-Wishart posterior degrees of freedom :math:`\bar\nu`.

    ``k + 2 + nobs + n_dummy``: the prior's ``k + 2``, the smallest value at
    which the prior covariance mean exists, plus one for every row updated
    on, sample and dummy rows alike. :attr:`sigma_u` is the posterior scale
    divided by ``posterior_df - k - 1``.
    """
    resid: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, k)`` residuals at the posterior mean.

    ``endog[p:] - fittedvalues``: the innovations implied by :attr:`beta_mean`,
    not a posterior over residuals. Their sample covariance is close to, but
    not equal to, :attr:`sigma_u`, which also carries the prior scale and
    the dummy rows.
    """
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, k)`` one-step-ahead conditional means at the posterior mean.

    Row ``t`` is the design row for observation ``p + t`` multiplied by
    :attr:`beta_mean`, aligned with ``endog[p:]`` and with :attr:`resid`.
    """
    nobs: int
    """Effective sample size ``T - p``: the rows the posterior was updated on, dummy rows excluded.

    The leading axis of :attr:`resid`, :attr:`fittedvalues`, :attr:`observed`,
    and of each replication from :meth:`posterior_replications`.
    """
    n_dummy: int
    """Artificial rows the prior contributed to the update.

    Zero for a plain :class:`~cultivars.bayes.NormalInverseWishartPrior`;
    ``k`` for a sum-of-coefficients component and one for a
    dummy-initial-observation component. They enter :attr:`posterior_df`
    and the posterior itself but are divided out of
    :attr:`log_marginal_likelihood`, and they are never part of
    :attr:`nobs`.
    """

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorConjugateFit,
        model: _BayesianVectorAutoRegressionModel[BVARResult],
    ) -> BVARResult:
        """Assemble the public result from a raw fit and the model that produced it.

        The single construction point for :class:`BVARResult`. :meth:`BVAR.fit`
        runs the conjugate update through
        :meth:`_BayesianVectorAutoRegressionModel._fit_conjugate`, which returns
        a :class:`_VectorConjugateFit` holding only what the numerics produced,
        and this method joins that with the specification the model holds --
        the panel, the labels, the order, the trend, and the prior's label --
        to make the frozen public record. Keeping assembly here rather than in
        ``__init__`` keeps the result a plain dataclass with no logic of its
        own, and keeps the raw fit free of anything the sampler does not need.

        The mapping is one-to-one, with a single rename:
        ``fit.coefficient_stack`` becomes :attr:`coefficients`. Arrays are
        passed through by reference, not copied; both containers are frozen,
        so the aliasing is safe.

        Nothing is validated. The fit is trusted to be consistent with the
        model that produced it -- same ``k``, same ``p``, same trend, draws
        already laid out in the model's design order -- because
        ``_fit_conjugate`` is the only producer and it guarantees exactly that.
        Passing a fit from a different model or specification produces a
        result whose arrays disagree with its labels, silently.

        Args:
            fit: The raw conjugate posterior output: posterior-mean blocks,
                the ``(S, w, k)`` and ``(S, k, k)`` draws, the closed-form log
                marginal likelihood, and the sample accounting.
            model: The specification the fit was computed under. Supplies
                :attr:`endog`, :attr:`names`, :attr:`order`, :attr:`trend`,
                and, through ``model.prior._label()``, :attr:`prior_label`.

        Returns:
            The assembled :class:`BVARResult`, sharing the fit's arrays.

        Example:
            Round-tripping a fit through the assembler is a pure field mapping:

            >>> import numpy as np
            >>> from cultivars.multivariate import BVAR
            >>> from cultivars.multivariate.large_dim.bayesian import BVARResult
            >>> model = BVAR(np.random.default_rng(0).standard_normal((60, 2)), order=1)
            >>> fit = model._fit_conjugate(n_draws=20, seed=0)
            >>> res = BVARResult._from_fit(fit, model)
            >>> res.nobs == fit.nobs and res.n_kept == fit.beta_draws.shape[0]
            True
            >>> res.beta_draws is fit.beta_draws and res.endog is model.endog
            True
            >>> res.prior_label
            'niw(l1=0.2, l3=1, l4=100)'
        """
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
            log_marginal_likelihood=fit.log_marginal_likelihood,
            posterior_df=fit.posterior_df,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            nobs=fit.nobs,
            n_dummy=fit.n_dummy,
        )

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables, ``k``.

        The width of :attr:`endog` and the length of :attr:`names`; the
        trailing axis of :attr:`beta_mean`, :attr:`resid`, and
        :attr:`fittedvalues`, both axes of :attr:`sigma_u`, and the
        dimension of the inverse-Wishart posterior, whose prior degrees of
        freedom are ``k + 2``. A large ``k`` is what this model exists for:
        with a proper prior the sample-length constraint is one observation,
        not one per regressor.
        """
        return len(self.names)

    @property
    def n_kept(self) -> int:
        """Posterior draws retained, ``S``.

        The leading axis of :attr:`beta_draws` and :attr:`sigma_draws`, and
        of the paths :meth:`forecast_paths` returns. Equal to the
        ``n_draws`` passed to :meth:`BVAR.fit`: the draws are independent
        samples from an exact posterior, so none are discarded to burn-in
        or thinning, unlike the Gibbs-sampled members of the family where
        the kept count is smaller than the iteration count. It is the
        divisor in :attr:`stable_share` and the pool
        :meth:`posterior_replications` spreads its replications across.
        """
        return int(self.beta_draws.shape[0])

    @property
    def _n_deterministic(self) -> int:
        """Deterministic regressors per equation, ``n_det``: 0, 1, or 2.

        Derived from :attr:`trend` -- ``"n"``, ``"c"``, ``"ct"`` -- and used
        wherever the design layout is sliced: the row count of
        :attr:`deterministic`, the offset before the first lag block in
        :attr:`beta_mean` and each draw of :attr:`beta_draws`, the leading
        labels in :meth:`_regressor_labels`, and the deterministic prefix
        :meth:`forecast_paths` extends. No ``KeyError`` guard: ``trend`` is
        validated at model construction and the result is frozen.
        """
        return {"n": 0, "c": 1, "ct": 2}[self.trend]

    def _regressor_labels(self) -> tuple[str, ...]:
        """Per-equation regressor labels, in design-row order.

        The deterministic labels first -- ``"const"``, then ``"trend"`` when
        :attr:`trend` is ``"ct"`` -- followed by the lag blocks lag-major,
        ``"{name}.L1"`` for every name, then ``"{name}.L2"``, and so on. The
        tuple has length ``w`` and indexes the rows of :attr:`beta_mean` and
        of each draw in :attr:`beta_draws`, so ``labels.index(label)`` is
        the row a user-facing label refers to. It is the lookup table
        :meth:`credible_interval` validates against and the table
        :meth:`_summary_table` reads the own-first-lag row from.

        Returns:
            Labels of length ``n_det + k * p``, in design-row order.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate import BVAR
            >>> y = np.random.default_rng(0).standard_normal((60, 2))
            >>> res = BVAR(y, order=2, trend="ct", names=("a", "b")).fit(n_draws=10, seed=0)
            >>> res._regressor_labels()
            ('const', 'trend', 'a.L1', 'b.L1', 'a.L2', 'b.L2')
            >>> len(res._regressor_labels()) == res.beta_mean.shape[0]
            True
        """
        det = ("const", "trend")[: self._n_deterministic]
        lags = tuple(f"{source}.L{lag + 1}" for lag in range(self.order) for source in self.names)
        return (*det, *lags)

    def credible_interval(self, equation: str, regressor: str) -> npt.NDArray[np.float64]:
        r"""Posterior summary of one coefficient: 16th percentile, mean, 84th percentile.

        Selects the scalar :math:`B_{r,e}` -- row ``regressor``, column
        ``equation`` -- from every retained draw and summarizes the ``S``
        values by their 16th percentile, arithmetic mean, and 84th
        percentile. The interval is the 68% central credible interval of
        that coefficient's marginal posterior, one standard deviation wide
        under normality, which is the convention every band on this result
        uses. The middle entry is the Monte Carlo mean of the draws; it
        approaches the exact posterior mean :attr:`beta_mean`\ ``[r, e]``
        as ``S`` grows but is not identical to it. For a different quantile,
        a joint region, or a functional of several coefficients, work from
        :attr:`beta_draws` directly with the same indices.

        Args:
            equation: The endogenous variable whose equation the coefficient
                belongs to; one of :attr:`names`.
            regressor: The per-equation regressor label: ``"const"``,
                ``"trend"`` (only when :attr:`trend` is ``"ct"``), or
                ``"{name}.L{lag}"`` with ``name`` in :attr:`names` and
                ``lag`` in ``1 .. p``.

        Returns:
            A ``(3,)`` array ``(low, mean, high)``: the 16th percentile,
            the mean, and the 84th percentile of the coefficient across
            :attr:`beta_draws`.

        Raises:
            SpecificationError: If ``equation`` is not one of :attr:`names`,
                or ``regressor`` is not a label the fitted design has. The
                message lists the admissible labels.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate import BVAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((120, 2))
            >>> for t in range(1, 120):
            ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
            >>> res = BVAR(y, order=1, names=("growth", "inflation")).fit(n_draws=200, seed=0)
            >>> low, mean, high = res.credible_interval("growth", "growth.L1")
            >>> bool(low < mean < high)
            True
            >>> bool(abs(mean - res.beta_mean[1, 0]) < 0.05)
            True
            >>> res.credible_interval("growth", "growth.L2")  # doctest: +NORMALIZE_WHITESPACE
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: unknown regressor 'growth.L2';
            expected one of ('const', 'growth.L1', 'inflation.L1').
        """
        if equation not in self.names:
            raise SpecificationError(
                f"unknown variable {equation!r}; expected one of {self.names}."
            )
        labels = self._regressor_labels()
        if regressor not in labels:
            raise SpecificationError(f"unknown regressor {regressor!r}; expected one of {labels}.")
        draws = self.beta_draws[:, labels.index(regressor), self.names.index(equation)]
        return np.array(
            [float(np.quantile(draws, 0.16)), float(draws.mean()), float(np.quantile(draws, 0.84))]
        )

    def _stack_of(self, beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r"""Slice one ``(w, k)`` coefficient matrix into its ``(p, k, k)`` lag stack.

        The inverse of the design layout: rows ``n_det + (l - 1) * k`` through
        ``n_det + l * k - 1`` of ``beta`` are the lag-``l`` block, and each is
        transposed so that slice ``l - 1`` is :math:`A_l` with ``[i, j]`` the
        response of equation ``i`` to variable ``j``. Applied to
        :attr:`beta_mean` it reproduces :attr:`coefficients`; applied to a
        draw it gives that draw's lag stack, which is what
        :meth:`_irf_draws`, :meth:`forecast_paths`, and :attr:`stable_share`
        hand to :func:`companion_matrix` or iterate directly. The
        deterministic rows are dropped, not returned; callers that need
        them slice ``beta[:n_det]`` themselves.

        Args:
            beta: A ``(w, k)`` coefficient matrix in design-row order --
                :attr:`beta_mean` or one slice of :attr:`beta_draws`.

        Returns:
            A ``(p, k, k)`` array; ``(0, k, k)`` when ``order`` is zero.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate import BVAR
            >>> y = np.random.default_rng(0).standard_normal((60, 3))
            >>> res = BVAR(y, order=2, trend="ct").fit(n_draws=10, seed=0)
            >>> np.allclose(res._stack_of(res.beta_mean), res.coefficients)
            True
            >>> res._stack_of(res.beta_draws[0]).shape
            (2, 3, 3)
        """
        k, offset = self.k_endog, self._n_deterministic
        if not self.order:
            return np.zeros((0, k, k))
        return np.stack(
            [beta[offset + lag * k : offset + (lag + 1) * k].T for lag in range(self.order)]
        )

    def _irf_draws(self, horizon: int, *, orthogonalized: bool) -> npt.NDArray[np.float64]:
        r"""Impulse responses of every retained draw, ``(S, horizon + 1, k, k)``.

        The shared engine behind :meth:`irf` and :meth:`irf_bands`. For each
        draw it builds the companion matrix :math:`F^{(s)}` from
        :meth:`_stack_of` and reads the moving-average coefficients off its
        powers, :math:`\Psi_h^{(s)} = J (F^{(s)})^h J^\top` with
        :math:`J = [I_k\ 0\ \cdots\ 0]`, accumulating the power one
        multiplication per lead rather than recomputing it. When
        ``orthogonalized`` it post-multiplies every lead by the draw's own
        lower Cholesky factor, :math:`\Theta_h^{(s)} = \Psi_h^{(s)}
        P^{(s)}` with :math:`\Sigma^{(s)} = P^{(s)} P^{(s)\top}`. With
        ``order`` zero the responses are the identity at lead zero and zero
        after, before any rotation.

        Cost is :math:`O(S \cdot H \cdot (kp)^3)` from the companion powers
        and :math:`O(S \cdot k^3)` from the Cholesky factors, with the whole
        ``(S, H + 1, k, k)`` array held in memory. Callers summarize across
        the draw axis; nothing is cached, so :meth:`irf` and
        :meth:`irf_bands` with the same arguments each recompute it.

        Args:
            horizon: Largest lead ``H``; the array has ``H + 1`` leads.
            orthogonalized: Rotate each draw by its own Cholesky factor.

        Returns:
            A ``(S, horizon + 1, k, k)`` array; entry ``[s, h, i, j]`` is
            draw ``s``'s response of variable ``i`` at lead ``h`` to shock
            ``j``.

        Raises:
            SpecificationError: If ``horizon`` is negative.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate import BVAR
            >>> y = np.random.default_rng(0).standard_normal((60, 2))
            >>> res = BVAR(y, order=1).fit(n_draws=10, seed=0)
            >>> draws = res._irf_draws(3, orthogonalized=False)
            >>> draws.shape
            (10, 4, 2, 2)
            >>> np.allclose(draws[:, 0], np.eye(2))
            True
        """
        if horizon < 0:
            raise SpecificationError(f"horizon must be non-negative; got {horizon}.")
        k, p, n_kept = self.k_endog, self.order, self.n_kept
        out = np.empty((n_kept, horizon + 1, k, k))
        for s in range(n_kept):
            psi = np.empty((horizon + 1, k, k))
            if p == 0:
                psi[:] = 0.0
                psi[0] = np.eye(k)
            else:
                selector = np.zeros((k, k * p))
                selector[:, :k] = np.eye(k)
                power = np.eye(k * p)
                companion = companion_matrix(self._stack_of(self.beta_draws[s]))
                for h in range(horizon + 1):
                    psi[h] = selector @ power @ selector.T
                    power = power @ companion
            out[s] = psi @ np.linalg.cholesky(self.sigma_draws[s]) if orthogonalized else psi
        return out

    def irf(
        self,
        horizon: int = 20,
        *,
        orthogonalized: bool = True,
        cumulative: bool = False,
    ) -> npt.NDArray[np.float64]:
        r"""Posterior mean impulse responses.

        The moving-average representation of the system writes each
        observation as a sum of current and past innovations,

        .. math::

           y_t = \text{deterministic} + \sum_{h=0}^{\infty} \Psi_h\, u_{t-h},
           \qquad
           \Psi_0 = I_k,
           \quad
           \Psi_h = \sum_{l=1}^{\min(h, p)} \Psi_{h-l} A_l,

        so :math:`\Psi_h` is the response of the system at lead ``h`` to a
        unit innovation. Innovations are correlated across equations, so a
        unit innovation in one is not an interpretable experiment; the
        orthogonalized responses :math:`\Theta_h = \Psi_h P`, with
        :math:`P` the lower Cholesky factor of :math:`\Sigma`, are the
        responses to one-standard-deviation *structural* shocks under the
        recursive identification in which each variable responds
        contemporaneously only to shocks ordered before it.

        This method returns the posterior mean of that object,

        .. math::

           \mathbb{E}[\Theta_h \mid Y]
           \approx \frac{1}{S} \sum_{s=1}^{S} \Psi_h(B^{(s)})\, P(\Sigma^{(s)}),

        computed draw by draw: each draw's own coefficients and its own
        Cholesky factor, then the average. This is not the response of the
        posterior mean coefficients, :math:`\Psi_h(\bar B)`, because
        :math:`\Psi_h` is a degree-``h`` polynomial in the lag matrices and
        the mean of a polynomial is not the polynomial of the mean; the two
        coincide only at lead zero and lead one. The difference grows with
        the lead and with posterior dispersion, so it is largest exactly
        where a loosely shrunk system is least informative.

        Args:
            horizon: Largest lead ``H`` to return; leads ``0 .. H``.
            orthogonalized: Rotate each draw by its own lower Cholesky
                factor so that shocks are orthogonal, one-standard-deviation
                structural shocks in the order of :attr:`names`. ``False``
                returns responses to unit reduced-form innovations,
                :math:`\Psi_h`, which do not depend on the ordering.
            cumulative: Return running sums over leads,
                :math:`\sum_{g \le h} \Theta_g`, the response of the level
                when the modelled series are differences.

        Returns:
            An array of shape ``(horizon + 1, k, k)``; entry ``[h, i, j]``
            is the posterior mean response of variable ``i`` at lead ``h``
            to shock ``j``. Lead zero is :math:`I_k` when not
            orthogonalized and the posterior mean Cholesky factor,
            lower-triangular, when it is.

        Raises:
            SpecificationError: If ``horizon`` is negative.

        Warning:
            With ``orthogonalized=True`` the column order of ``endog`` is
            an identifying assumption, exactly as in
            :class:`~cultivars.multivariate.RecursiveSVAR`: shock ``j``
            moves variables ``j, j + 1, ...`` on impact and variables before
            ``j`` only from lead one. Reorder the columns before fitting to
            assert a different causal ordering; there is no reordering
            argument here because the posterior draws are already
            conditioned on the fitted layout.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate import BVAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((120, 2))
            >>> for t in range(1, 120):
            ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
            >>> res = BVAR(y, order=1, names=("growth", "inflation")).fit(n_draws=200, seed=0)
            >>> res.irf(8).shape
            (9, 2, 2)

            Lead zero is the identity for reduced-form responses and a
            lower-triangular factor for orthogonalized ones, so the first
            variable does not respond on impact to the second shock:

            >>> np.allclose(res.irf(8, orthogonalized=False)[0], np.eye(2))
            True
            >>> float(res.irf(8)[0, 0, 1])
            0.0

            Cumulative responses are running sums:

            >>> np.allclose(res.irf(8, cumulative=True)[-1], res.irf(8).sum(axis=0))
            True

            The posterior mean response is not the response at the
            posterior mean beyond lead one:

            >>> at_mean = np.linalg.matrix_power(res.coefficients[0], 2)
            >>> np.allclose(res.irf(2, orthogonalized=False)[2], at_mean)
            False

        See Also:
            * :meth:`irf_bands` -- the same responses with their 68%
              posterior bands.
            * :class:`~cultivars.multivariate.RecursiveSVAR` -- the same
              recursive identification on a point-estimated
              :class:`~cultivars.multivariate.VAR`.
            * :class:`~cultivars.multivariate.SignRestrictedSVAR` -- set
              identification when no recursive ordering is credible.

        References:
            Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*,
            48(1), 1-48.

            Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
            Analysis*. Springer. Chapters 2 and 3.

            Koop, G., & Korobilis, D. (2010). Bayesian multivariate time
            series methods for empirical macroeconomics. *Foundations and
            Trends in Econometrics*, 3(4), 267-358.
        """
        out = self._irf_draws(horizon, orthogonalized=orthogonalized).mean(axis=0)
        return np.cumsum(out, axis=0) if cumulative else out

    def irf_bands(
        self,
        horizon: int = 20,
        *,
        orthogonalized: bool = True,
        cumulative: bool = False,
    ) -> npt.NDArray[np.float64]:
        r"""Posterior bands of the impulse responses.

        The same draw-by-draw responses as :meth:`irf`, summarized at every
        ``[h, i, j]`` by the 16th percentile, mean, and 84th percentile
        across the ``S`` draws: a pointwise 68% central credible band, one
        standard deviation wide under normality. Pointwise means each
        ``(h, i, j)`` cell is banded on its own; the band is not a joint
        region over leads, so a path that stays inside it at every lead is
        not thereby a posterior-probable path. The middle slice equals
        :meth:`irf` with the same arguments.

        With ``cumulative``, the running sums are formed draw by draw first
        and the quantiles taken of those, which is the posterior of the
        cumulative response; banding :meth:`irf` and then summing would
        instead sum quantiles, which is not a quantile of anything.

        Args:
            horizon: Largest lead ``H`` to return; leads ``0 .. H``.
            orthogonalized: Rotate each draw by its own lower Cholesky
                factor; see :meth:`irf` for the identifying assumption this
                carries.
            cumulative: Band the running sums over leads instead of the
                responses themselves.

        Returns:
            An array of shape ``(horizon + 1, k, k, 3)`` whose last axis is
            ``(16th percentile, mean, 84th percentile)`` across draws;
            entry ``[h, i, j, :]`` summarizes the response of variable
            ``i`` at lead ``h`` to shock ``j``.

        Raises:
            SpecificationError: If ``horizon`` is negative.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate import BVAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((120, 2))
            >>> for t in range(1, 120):
            ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
            >>> res = BVAR(y, order=1, names=("growth", "inflation")).fit(n_draws=200, seed=0)
            >>> bands = res.irf_bands(8)
            >>> bands.shape
            (9, 2, 2, 3)
            >>> bool((bands[..., 0] <= bands[..., 1]).all())
            True
            >>> bool((bands[..., 1] <= bands[..., 2]).all())
            True
            >>> np.allclose(bands[..., 1], res.irf(8))
            True

            The own response of ``growth`` to its own shock, lead by lead,
            as ``(low, mean, high)`` rows:

            >>> bands[:3, 0, 0, :].shape
            (3, 3)

        See Also:
            * :meth:`irf` -- the posterior mean alone.
            * :attr:`beta_draws` and :attr:`sigma_draws` -- for other
              quantiles, joint bands, or responses to a non-recursive
              rotation, propagate the draws yourself.
        """
        draws = self._irf_draws(horizon, orthogonalized=orthogonalized)
        if cumulative:
            draws = np.cumsum(draws, axis=1)
        return np.stack(
            [
                np.quantile(draws, 0.16, axis=0),
                draws.mean(axis=0),
                np.quantile(draws, 0.84, axis=0),
            ],
            axis=-1,
        )

    def forecast_paths(
        self,
        steps: int = 8,
        *,
        seed: int | np.random.Generator | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""The posterior predictive's raw simulated paths, one per retained draw.

        Draw ``s`` simulates its own future from the end of the sample,

        .. math::

           y^{(s)}_{T+h} = d_{T+h}^\top \beta^{(s)}_{\text{det}}
           + \sum_{l=1}^{p} A_l^{(s)}\, y^{(s)}_{T+h-l}
           + P^{(s)} \varepsilon^{(s)}_h,
           \qquad
           \varepsilon^{(s)}_h \sim \mathcal{N}(0, I_k),

        for ``h = 1 .. steps``, where :math:`y^{(s)}_{T+h-l}` is the
        observed value when ``T + h - l <= T`` and the path's own earlier
        simulated value otherwise, :math:`d_{T+h}` is the deterministic
        row (``1`` and, for ``"ct"``, the continued time index ``T + h``),
        and :math:`P^{(s)}` is the lower Cholesky factor of
        :math:`\Sigma^{(s)}`. Across draws the paths are samples from the
        joint posterior predictive

        .. math::

           p(y_{T+1:T+H} \mid Y)
           = \int p(y_{T+1:T+H} \mid B, \Sigma, Y)\, p(B, \Sigma \mid Y)
           \, dB\, d\Sigma,

        parameter uncertainty and shock uncertainty integrated together.

        The paths are returned unsummarized because the summary loses what
        most downstream uses need. A density score needs the sample, not
        three quantiles; a fan chart takes its own quantiles at its own
        levels; a model average mixes paths across models; and any
        path-dependent quantity -- cumulative growth over the horizon, the
        probability that a series stays below a threshold for every lead --
        is a function of whole paths, which the per-lead quantiles of
        :meth:`forecast` cannot recover. Each path is one draw's future, so
        the joint distribution across leads is preserved.

        Args:
            steps: Horizons ahead ``H``; the paths cover ``T + 1 .. T + H``.
            seed: Seed or generator for the predictive shocks. An integer
                gives a fresh generator and a reproducible array; a
                :class:`numpy.random.Generator` is used in place and
                advanced, so two calls with the same generator object give
                different paths.

        Returns:
            An array of shape ``(n_kept, steps, k)``; entry ``[s, h, i]``
            is draw ``s``'s simulated value of variable ``i`` at lead
            ``h + 1``.

        Raises:
            SpecificationError: If ``steps`` is not positive.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate import BVAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((120, 2))
            >>> for t in range(1, 120):
            ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
            >>> res = BVAR(y, order=1, names=("growth", "inflation")).fit(n_draws=200, seed=0)
            >>> paths = res.forecast_paths(4, seed=1)
            >>> paths.shape
            (200, 4, 2)
            >>> np.allclose(paths, res.forecast_paths(4, seed=1))
            True

            A path-dependent probability -- that cumulative ``growth`` over
            the four leads is negative -- is a mean over paths:

            >>> prob = (paths[:, :, 0].sum(axis=1) < 0).mean()
            >>> bool(0.0 <= prob <= 1.0)
            True

        See Also:
            * :meth:`forecast` -- the ``(16th, mean, 84th)`` summary of
              these paths.
            * :func:`~cultivars.forecast.fan_chart` -- quantile fans from
              the paths.
            * :class:`~cultivars.forecast.DensityScore` -- scores the
              predictive sample against realized values.
            * :func:`~cultivars.bayes.bayesian_model_average` -- mixes paths
              across models by posterior model probability.
        """
        if steps < 1:
            raise SpecificationError(f"steps must be positive; got {steps}.")
        rng = np.random.default_rng(seed)
        k, p, n = self.k_endog, self.order, self.endog.shape[0]
        offset = self._n_deterministic
        paths = np.empty((self.n_kept, steps, k))
        for s in range(self.n_kept):
            beta = self.beta_draws[s]
            stack = self._stack_of(beta)
            chol = np.linalg.cholesky(self.sigma_draws[s])
            history = list(self.endog[n - p :][::-1]) if p else []
            for h in range(steps):
                det = {"n": [], "c": [1.0], "ct": [1.0, float(n + h + 1)]}[self.trend]
                value = np.asarray(det, dtype=np.float64) @ beta[:offset]
                for lag in range(p):
                    value = value + stack[lag] @ history[lag]
                value = value + chol @ rng.standard_normal(k)
                paths[s, h] = value
                if p:
                    history = [value, *history[:-1]]
        return paths

    def forecast(
        self,
        steps: int = 8,
        *,
        seed: int | np.random.Generator | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""The posterior predictive: parameter *and* shock uncertainty.

        The three-number summary of :meth:`forecast_paths`: at every lead
        and variable, the 16th percentile, mean, and 84th percentile across
        the simulated paths, a pointwise 68% central band of the predictive
        distribution :math:`p(y_{T+h} \mid Y)`. Because each path was
        simulated under its own draw of :math:`(B, \Sigma)`, the band
        integrates over parameter uncertainty as well as over the shocks.
        The point path's intervals condition on the estimated parameters
        instead, and for a large shrunk system that conditioning discards
        the larger half of the uncertainty; the width of this band relative
        to that one is a direct reading of how much the posterior is still
        spread.

        The middle entry is the predictive mean estimated by Monte Carlo,
        not the point forecast from iterating :attr:`beta_mean`. The two
        differ from lead two onward because a multi-step forecast is
        nonlinear in the coefficients, and they differ at every lead by the
        Monte Carlo error of ``S`` paths. For the mean forecast of a
        specific draw, or of the posterior mean, use :meth:`simulate` or
        iterate :attr:`coefficients` and :attr:`deterministic` directly.

        Args:
            steps: Horizons ahead ``H``; the summary covers ``T + 1 .. T + H``.
            seed: Seed or generator for the predictive shocks, passed to
                :meth:`forecast_paths`; the same seed reproduces the same
                summary.

        Returns:
            An array of shape ``(steps, k, 3)`` whose last axis is
            ``(16th percentile, mean, 84th percentile)``; entry
            ``[h, i, :]`` summarizes variable ``i`` at lead ``h + 1``.

        Raises:
            SpecificationError: If ``steps`` is not positive.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate import BVAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((120, 2))
            >>> for t in range(1, 120):
            ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
            >>> res = BVAR(y, order=1, names=("growth", "inflation")).fit(n_draws=200, seed=0)
            >>> fc = res.forecast(4, seed=0)
            >>> fc.shape
            (4, 2, 3)

            The summary is exactly that of the paths under the same seed:

            >>> paths = res.forecast_paths(4, seed=0)
            >>> np.allclose(fc[..., 1], paths.mean(axis=0))
            True

            For a stationary system the predictive band widens with the
            lead:

            >>> width = fc[:, :, 2] - fc[:, :, 0]
            >>> bool((width[-1] > width[0]).all())
            True

        See Also:
            * :meth:`forecast_paths` -- the paths this summarizes, for
              scoring, fan charts, and path-dependent quantities.
            * :class:`~cultivars.forecast.Backtest` -- rolls the predictive
              through the sample and scores it out of sample.
            * :class:`~cultivars.multivariate.VARResult` -- the
              point-estimated counterpart, whose forecast intervals condition
              on the estimated coefficients.

        References:
            Kadiyala, K. R., & Karlsson, S. (1997). Numerical methods for
            estimation and inference in Bayesian VAR-models. *Journal of
            Applied Econometrics*, 12(2), 99-132.

            Karlsson, S. (2013). Forecasting with Bayesian vector
            autoregression. In G. Elliott & A. Timmermann (Eds.), *Handbook
            of Economic Forecasting* (Vol. 2A, pp. 791-897). Elsevier.
        """
        paths = self.forecast_paths(steps, seed=seed)
        return np.stack(
            [
                np.quantile(paths, 0.16, axis=0),
                paths.mean(axis=0),
                np.quantile(paths, 0.84, axis=0),
            ],
            axis=-1,
        )

    @property
    def stable_share(self) -> float:
        r"""Posterior probability that the system is stable.

        The share of retained draws whose companion matrix
        :math:`F^{(s)}`, built from that draw's lag stack, has spectral
        radius strictly below one,

        .. math::

           \Pr(\text{stable} \mid Y)
           \approx \frac{1}{S} \sum_{s=1}^{S}
           \mathbf{1}\bigl\{\rho(F^{(s)}) < 1\bigr\},

        a Monte Carlo estimate with standard error
        :math:`\sqrt{q(1 - q) / S}`. Stability is the condition under which
        the moving-average representation converges, so it is the condition
        for :meth:`irf` to die out and for the predictive of
        :meth:`forecast_paths` to have finite variance at long leads.

        Under a random-walk-centred prior this is routinely well below one
        on persistent data, and that is information about the posterior,
        not a defect: the prior puts mass at the unit root, the sample
        rarely rules it out, and draws just past it are what the posterior
        looks like there. A low value says long-horizon responses and
        forecasts carry explosive draws and should be read with that in
        mind, or that the series should be differenced or the prior
        centred at zero. It is reported on the summary. Costs one
        eigendecomposition of a ``(kp, kp)`` matrix per draw and is
        recomputed on every access; with ``order`` zero it is one.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate import BVAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((120, 2))
            >>> for t in range(1, 120):
            ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
            >>> BVAR(y, order=1).fit(n_draws=200, seed=0).stable_share
            1.0

            On random walks the random-walk prior keeps posterior mass past
            the unit circle:

            >>> walks = rng.standard_normal((120, 2)).cumsum(axis=0)
            >>> bool(BVAR(walks, order=1).fit(n_draws=200, seed=0).stable_share < 1.0)
            True

        See Also:
            * :attr:`~cultivars.bayes.NormalInverseWishartPrior.persistence`
              -- the prior mean of each own first lag; zero centres the
              prior on stationarity.
        """
        if self.order == 0:
            return 1.0
        stable = 0
        for s in range(self.n_kept):
            eigs = np.linalg.eigvals(companion_matrix(self._stack_of(self.beta_draws[s])))
            stable += int(float(np.abs(eigs).max(initial=0.0)) < 1.0)
        return stable / self.n_kept

    def marginal_likelihood(self) -> _MarginalLikelihoodSelection:
        r"""The closed-form log marginal likelihood as a comparable record.

        Wraps :attr:`log_marginal_likelihood` in a
        :class:`~cultivars.bayes.MarginalLikelihoodSelection` so it can sit
        in one :func:`~cultivars.bayes.compare` table beside simulated
        estimates -- Chib, modified harmonic mean, bridge sampling -- from
        the Gibbs and particle-chain members of the family. The record
        carries the value with ``mcse=0.0`` and ``method="analytic"``,
        because the number is exact: it uses no draws (``n_draws=0``) and
        has no Monte Carlo error. Its ``source`` is
        ``"BVAR({order}), {prior_label}"``, so two fits of the same order
        under different priors remain distinguishable in a table, and its
        ``nobs`` is :attr:`nobs`, which :func:`~cultivars.bayes.compare`
        checks across records: a fit of a different order scores a
        different number of observations and is not comparable on this
        number without adjustment.

        Differences of the log value across records fitted to the same
        sample are log Bayes factors; :func:`~cultivars.bayes.compare`
        converts them to posterior model probabilities under the prior
        model probabilities it is given.

        Returns:
            The record: ``log_value``, ``mcse``, ``method``, ``n_draws``,
            ``source``, and ``nobs``.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate import BVAR
            >>> from cultivars.bayes import NormalInverseWishartPrior, compare
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((120, 2))
            >>> for t in range(1, 120):
            ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
            >>> res = BVAR(y, order=1).fit(n_draws=200, seed=0)
            >>> record = res.marginal_likelihood()
            >>> record.method, record.mcse, record.n_draws
            ('analytic', 0.0, 0)
            >>> record.log_value == res.log_marginal_likelihood
            True
            >>> record.source
            'BVAR(1), niw(l1=0.2, l3=1, l4=100)'

            Rank two priors on the same sample:

            >>> alt = BVAR(y, order=1, prior=NormalInverseWishartPrior(tightness=0.02))
            >>> table = compare(record, alt.fit(n_draws=200, seed=0).marginal_likelihood())
            >>> table.rows[0][0]
            'BVAR(1), niw(l1=0.2, l3=1, l4=100)'

        See Also:
            * :func:`~cultivars.bayes.compare` -- the table, with Kass and
              Raftery's reading of the Bayes factors.
            * :class:`~cultivars.multivariate.HierarchicalBVAR` -- maximizes
              this number over the prior's hyperparameters rather than
              comparing a hand-picked few.
            * :func:`~cultivars.bayes.marginal_likelihood`,
              :func:`~cultivars.bayes.bridge_sampling`,
              :func:`~cultivars.bayes.modified_harmonic_mean` -- simulated
              estimates for results without a closed form.

        References:
            Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *Journal
            of the American Statistical Association*, 90(430), 773-795.

            Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior
            selection for vector autoregressions. *Review of Economics and
            Statistics*, 97(2), 436-451.
        """
        return _MarginalLikelihoodSelection(
            log_value=float(self.log_marginal_likelihood),
            mcse=0.0,
            method="analytic",
            n_draws=0,
            source=f"BVAR({self.order}), {self.prior_label}",
            nobs=int(self.nobs),
        )

    def _summary_table(self) -> SummaryTable:
        r"""Build the structured summary every renderer draws from.

        The single description of this result that :meth:`summary`,
        ``print(res)``, a bare ``res`` in a notebook, and
        ``res.summary().to_pandas()`` all render, per the
        :class:`_SummaryMixin` contract. The metadata block carries the
        specification and the posterior's accounting -- model, prior label,
        draws, variables, observations, dummy rows, trend, and the log
        marginal likelihood to three decimals. The table has one row per
        equation with that equation's own first-lag coefficient: its
        posterior mean and 68% interval from :meth:`credible_interval` at
        label ``"{name}.L1"``, four decimals, or dashes when ``order`` is
        zero. Own first lags are the one coefficient every equation has and
        the one the Minnesota prior shrinks toward :math:`\delta`, so the
        row reads as how far the data moved each equation off the prior
        mean; every other coefficient is available through
        :meth:`credible_interval`.

        The notes state what a reader coming from a likelihood-based result
        would otherwise look for and misread: that the draws are
        independent so no convergence diagnostics apply, what the log
        marginal likelihood is and is not, the posterior probability of
        stability from :attr:`stable_share`, the absence of ``llf`` and
        information criteria, and that :meth:`forecast` is the full
        predictive. Building the table evaluates :attr:`stable_share`, so
        it costs one eigendecomposition per draw.

        Returns:
            The :class:`SummaryTable` with title ``"BVAR({order}) Results"``,
            columns ``("equation", "own L1 mean", "68% interval")``, one row
            per name, and the notes above.
        """
        rows = []
        for name in self.names:
            if self.order:
                low, mid, high = self.credible_interval(name, f"{name}.L1")
                rows.append((name, f"{mid:.4f}", f"[{low:.4f}, {high:.4f}]"))
            else:
                rows.append((name, "-", "-"))
        notes = [
            "The posterior is exact Normal-inverse-Wishart; draws are "
            "independent, not a Markov chain, so no convergence diagnostics "
            "apply.",
            "log_marginal_likelihood is the sample's, with the "
            "dummy-observation contribution divided out; differences across "
            "priors are log Bayes factors, and hyperparameter choice should "
            "maximize it (Giannone-Lenza-Primiceri).",
            f"Posterior probability of stability: {self.stable_share:.2f}. "
            "Mass near the unit circle is the random-walk prior speaking, "
            "not a defect.",
            "No llf, parameter count, or information criteria are reported: "
            "a posterior has none, and the marginal likelihood is the "
            "honest replacement.",
            "forecast() is the full posterior predictive -- parameter and "
            "shock uncertainty jointly.",
        ]
        return SummaryTable(
            title=f"BVAR({self.order}) Results",
            metadata=(
                ("Model", f"BVAR({self.order})"),
                ("Prior", self.prior_label),
                ("Draws", f"{self.n_kept}"),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.nobs}"),
                ("Dummy rows", f"{self.n_dummy}"),
                ("Trend", self.trend),
                ("log ML", f"{self.log_marginal_likelihood:.3f}"),
            ),
            columns=("equation", "own L1 mean", "68% interval"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class BVAR(_BayesianVectorAutoRegressionModel[BVARResult]):
    r"""Conjugate Normal-inverse-Wishart Bayesian VAR, after Banbura, Giannone and Reichlin.

    The reduced-form vector autoregression

    .. math::

       y_t = c + \sum_{l=1}^{p} A_l\, y_{t-l} + u_t,
       \qquad u_t \sim \mathcal{N}(0, \Sigma),

    estimated under a prior that makes the joint posterior over the
    coefficient matrix and the innovation covariance exact rather than
    sampled. Stacking the effective sample as :math:`Y = X B + U`, the
    prior is

    .. math::

       \Sigma \sim \mathcal{IW}(S_0, \nu_0),
       \qquad
       \operatorname{vec}(B) \mid \Sigma
       \sim \mathcal{N}\bigl(\operatorname{vec}(B_0),\ \Sigma \otimes \Omega\bigr),

    with :math:`B_0` centring each variable's own first lag at
    ``persistence`` (one, the random walk, by default) and every other
    coefficient at zero, :math:`\Omega` diagonal with the Minnesota
    variance profile -- tighter on longer lags, loose on the deterministic
    block -- and :math:`S_0 = \operatorname{diag}(s_1^2, \ldots, s_k^2)`
    with :math:`\nu_0 = k + 2` built from univariate residual scales. The
    Kronecker form :math:`\Sigma \otimes \Omega` is the whole point: it is
    exactly Litterman's prior with the cross-equation weight pinned at one,
    and it is the condition for Normal-inverse-Wishart conjugacy, so the
    posterior is a closed-form update, its draws are independent, and its
    marginal likelihood has a closed form. What it gives up is Litterman's
    claim that cross-variable coefficients deserve extra shrinkage. A prior
    that keeps that claim is refused rather than approximated, with
    directions to the point-estimate path that honours it.

    Shrinkage is what makes a large system estimable at all. With a proper
    prior the sample-length constraint is one observation per lag plus one,
    not one per regressor, so a system with more regressors than rows --
    routine at twenty or a hundred variables -- fits here where
    :class:`~cultivars.multivariate.VAR` refuses. The prior's tightness,
    not the sample size, sets the effective number of parameters, which is
    why choosing it well matters and why the number this model reports for
    that choice is the marginal likelihood :math:`p(Y)` rather than a
    likelihood or an information criterion: differences of its log across
    priors on the same sample are log Bayes factors, and
    :class:`~cultivars.multivariate.HierarchicalBVAR` maximizes it over the
    hyperparameters directly.

    Prior components compose with ``+``. The Doan-Litterman-Sims
    sum-of-coefficients and the Sims dummy-initial-observation priors are
    artificial rows :math:`(Y_d, X_d)` appended to the sample before the
    update, expressing beliefs about unit roots and cointegration on the
    scale of the presample means. They are part of the prior, so the fit
    reports the marginal likelihood of the sample given them,
    :math:`\log p(Y \mid Y_d)`, with the dummy rows' own contribution
    divided out, and the number stays comparable across priors with and
    without dummies.

    Args:
        endog: The observed panel, shape ``(nobs, k)``, time down the rows;
            anything :func:`numpy.asarray` accepts. A one-dimensional series
            is promoted to a single column. Must have more rows than
            columns and at least ``order + 2`` rows; beyond that, the
            regressor count imposes no constraint because the prior is
            proper.
        order: Autoregressive order ``p``, a positive integer. The Minnesota
            prior centres each variable's own first lag, so ``p = 0`` has
            no prior to state and is refused. For choosing it,
            :meth:`lag_order_selection` scores every order on one common
            sample.
        prior: The prior to update under. ``None`` selects
            :class:`~cultivars.bayes.NormalInverseWishartPrior` at its
            defaults. A composition such as
            ``NormalInverseWishartPrior() + SumOfCoefficientsPrior() +
            DummyInitialObservationPrior()`` is accepted; the admissible
            components are listed below. Conjugacy is checked when
            :meth:`fit` runs, not here: a prior whose coefficient
            variances do not factor as :math:`\sigma_i^2 \cdot \omega_j` is
            refused with a :class:`~cultivars.exceptions.SpecificationError`
            at that point.
        trend: Deterministic terms: ``"n"`` none, ``"c"`` a constant
            (default), ``"ct"`` a constant and a linear trend.
        names: One label per column of ``endog``, unique. Defaults to
            ``y1 ... yk``. The order of the labels is the column order and,
            for orthogonalized impulse responses on the result, the
            recursive identifying ordering.

    Priors:
        NormalInverseWishartPrior: The conjugate Minnesota prior and the
            default: ``tightness``, ``decay``, ``exogenous``, and
            ``persistence`` set :math:`\Omega` and :math:`B_0`. Proper by
            construction, so it always carries a marginal likelihood.
        MinnesotaPrior: Accepted only with ``cross_equation=1``, where it is
            the same prior as above under Litterman's parameterization. Any
            other cross-equation weight breaks the Kronecker factorization
            and is refused at :meth:`fit`; pass it to
            :class:`~cultivars.multivariate.VAR` instead.
        SumOfCoefficientsPrior: A dummy-observation component, ``k``
            artificial rows, expressing that each variable's own lags sum
            to one. Add it to a proper base prior; alone it is improper.
        DummyInitialObservationPrior: A dummy-observation component, one
            artificial row, expressing that the system starts at its
            presample mean. Add it to a proper base prior; alone it is
            improper.

    Raises:
        SpecificationError: If ``order`` is not a positive integer, if
            ``trend`` is not one of ``"n"``, ``"c"``, ``"ct"``, or if
            ``names`` has the wrong length or repeats a label.
        DimensionError: If ``endog`` is not two-dimensional after promotion,
            has no columns, has no more rows than columns, or has fewer
            than ``order + 1`` rows. The fit needs one row more, for the
            prior's scales, and raises the same error when it is missing.
        NumericalError: If ``endog`` contains non-finite values.

    Note:
        Construction validates the sample and the specification; the prior
        is validated when :meth:`fit` runs, because conjugacy is a property
        of the prior's variances evaluated on this sample's scales. A model
        can therefore be constructed with a prior it will later refuse.
        :meth:`prior_replications` and
        :func:`~cultivars.bayes.prior_predictive_check` also evaluate the
        prior and raise the same error for an unusable one.

    Example:
        Fit under the default prior and read the posterior:

        >>> import numpy as np
        >>> from cultivars.multivariate import BVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((120, 2))
        >>> for t in range(1, 120):
        ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
        >>> res = BVAR(y, order=1, names=("growth", "inflation")).fit(n_draws=200, seed=0)
        >>> res.beta_draws.shape, res.n_dummy, res.posterior_df
        ((200, 3, 2), 0, 123.0)

        A proper prior admits a system with more regressors than rows;
        twenty variables at two lags is 41 regressors per equation on 30
        observations:

        >>> from cultivars.multivariate import VAR
        >>> wide = rng.standard_normal((30, 20)).cumsum(axis=0)
        >>> VAR(wide, order=2)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.DimensionError: a sample of 30 rows is too short for VAR(2); ...
        >>> BVAR(wide, order=2).fit(n_draws=100, seed=0).beta_draws.shape
        (100, 41, 20)

        Compose dummy observations onto the base prior and compare on the
        marginal likelihood of the same sample:

        >>> from cultivars.bayes import (
        ...     DummyInitialObservationPrior,
        ...     NormalInverseWishartPrior,
        ...     SumOfCoefficientsPrior,
        ... )
        >>> prior = (
        ...     NormalInverseWishartPrior()
        ...     + SumOfCoefficientsPrior()
        ...     + DummyInitialObservationPrior()
        ... )
        >>> dummies = BVAR(y, order=1, names=("growth", "inflation"), prior=prior)
        >>> res_d = dummies.fit(n_draws=200, seed=0)
        >>> res_d.n_dummy, res_d.prior_label
        (3, 'niw(l1=0.2, l3=1, l4=100) + soc(1) + dio(1)')
        >>> bool(res_d.log_marginal_likelihood > res.log_marginal_likelihood)
        True

        A prior with Litterman's cross-equation shrinkage is not conjugate
        and is refused when the fit runs:

        >>> from cultivars.bayes import MinnesotaPrior
        >>> model = BVAR(y, order=1, prior=MinnesotaPrior(cross_equation=0.5))
        >>> model.fit(n_draws=10, seed=0)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: the prior variance does not factor as ...

        The order is chosen on a common sample with the inherited
        information criteria:

        >>> BVAR(y, order=1).lag_order_selection(4).selected["bic"]
        1

    See Also:
        * :class:`BVARResult` -- the fitted posterior, with the update
          formulas, the draws, and the propagating methods.
        * :class:`~cultivars.multivariate.HierarchicalBVAR` -- treats the
          tightnesses as unknowns and maximizes or integrates the marginal
          likelihood over them.
        * :class:`~cultivars.multivariate.GibbsBVAR` -- the independent
          Normal-Wishart prior, which drops the Kronecker restriction at the
          cost of a sampler.
        * :class:`~cultivars.multivariate.StudentBVAR` and
          :class:`~cultivars.multivariate.BVARSV` -- heavy-tailed and
          stochastic-volatility innovations.
        * :class:`~cultivars.multivariate.VAR` -- the point-estimate path,
          which accepts a :class:`~cultivars.bayes.MinnesotaPrior` with
          cross-equation shrinkage as a per-equation ridge.
        * :func:`~cultivars.bayes.prior_predictive_check` -- what the prior
          deems plausible on the scale of the data, before any fit.

    References:
        Litterman, R. B. (1986). Forecasting with Bayesian vector
        autoregressions: Five years of experience. *Journal of Business &
        Economic Statistics*, 4(1), 25-38.

        Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and
        conditional projection using realistic prior distributions.
        *Econometric Reviews*, 3(1), 1-100.

        Sims, C. A. (1993). A nine-variable probabilistic macroeconomic
        forecasting model. In J. H. Stock & M. W. Watson (Eds.), *Business
        Cycles, Indicators, and Forecasting* (pp. 179-212). University of
        Chicago Press.

        Kadiyala, K. R., & Karlsson, S. (1997). Numerical methods for
        estimation and inference in Bayesian VAR-models. *Journal of Applied
        Econometrics*, 12(2), 99-132.

        Banbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian
        vector auto regressions. *Journal of Applied Econometrics*, 25(1),
        71-92.

        Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection
        for vector autoregressions. *Review of Economics and Statistics*,
        97(2), 436-451.
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
        """Substitute the conjugate Minnesota default, guard the order, then validate.

        The base constructor reads ``prior=None`` as unrestricted least
        squares and sizes its sample-length guard from :attr:`is_shrunk`,
        so the default has to be substituted before ``super().__init__``
        runs, not after: a Bayesian VAR is never unshrunk, and its guard
        must be the one-observation-per-lag rule a proper prior earns, not
        the one-per-regressor rule of the point path. The order guard sits
        here for the same reason. The base admits ``order == 0``, which is
        a valid least-squares specification, but the Minnesota prior
        centres each variable's own first lag and has nothing to say about
        a system without one, so the refusal belongs to this class and is
        raised before any state exists. ``__slots__ = ()`` keeps the base's
        slots layout; the class adds no state of its own.

        Args:
            endog: Forwarded to the base after no processing.
            order: Checked to be a positive integer before forwarding.
            prior: Forwarded, with ``None`` replaced by
                :class:`~cultivars.bayes.NormalInverseWishartPrior` at its
                defaults. Conjugacy is not checked here; see :meth:`fit`.
            trend: Forwarded.
            names: Forwarded.

        Raises:
            SpecificationError: If ``order`` is not a positive integer, from
                this guard; or if ``trend`` or ``names`` is malformed, from
                the base.
            DimensionError: If ``endog`` cannot support the specification,
                from the base.
            NumericalError: If ``endog`` is not finite, from the base.
        """
        if int(order) != order or order < 1:
            raise SpecificationError(
                f"order must be an integer >= 1 for {type(self).__name__}; got {order!r}."
            )
        super().__init__(
            endog,
            order=order,
            trend=trend,
            names=names,
            prior=prior if prior is not None else NormalInverseWishartPrior(),
        )

    def fit(
        self,
        *,
        n_draws: int = 1000,
        seed: int | np.random.Generator | None = None,
    ) -> BVARResult:
        r"""Update the posterior exactly and draw from it.

        Runs the closed-form Normal-inverse-Wishart update -- the design and
        the prior's dummy rows stacked, :math:`K = \Omega^{-1} + X^\top X`,
        :math:`\bar B`, :math:`\bar S`, :math:`\bar\nu` -- and the closed-form
        log marginal likelihood, then draws ``n_draws`` independent pairs
        :math:`(B^{(s)}, \Sigma^{(s)})` from the posterior:
        :math:`\Sigma^{(s)} \sim \mathcal{IW}(\bar S, \bar\nu)` by Bartlett
        decomposition, and :math:`B^{(s)} = \bar B + L^{-\top} Z
        (\Sigma^{(s)})^{1/2\top}` with :math:`L L^\top = K` factored once,
        :math:`Z` a standard normal ``(w, k)`` matrix, and
        :math:`(\Sigma^{(s)})^{1/2}` that draw's lower Cholesky factor.
        The update formulas are given in full on :class:`BVARResult`.

        Because the posterior is exact, ``n_draws`` sets Monte Carlo
        precision only. The posterior mean :attr:`~BVARResult.beta_mean`,
        the covariance mean :attr:`~BVARResult.sigma_u`, the
        :attr:`~BVARResult.log_marginal_likelihood`, and every
        summary-table number are identical across draw counts and seeds;
        what tightens with more draws is the bands of
        :meth:`~BVARResult.credible_interval`,
        :meth:`~BVARResult.irf_bands`, and :meth:`~BVARResult.forecast`,
        and the resolution of :attr:`~BVARResult.stable_share`. A thousand
        draws puts a 68% band's endpoints within about a twentieth of a
        posterior standard deviation; the cost is one ``(w, w)`` Cholesky
        factorization plus, per draw, an inverse-Wishart draw and two
        ``(w, k)`` matrix products, so a hundred-variable system at four
        lags takes seconds, not minutes.

        Args:
            n_draws: Independent posterior draws of :math:`(B, \Sigma)` to
                retain, all of them; there is no burn-in or thinning to
                subtract. Becomes :attr:`~BVARResult.n_kept`.
            seed: Seed or generator for the draws. An integer gives a fresh
                generator and a reproducible result; a
                :class:`numpy.random.Generator` is used in place and
                advanced; ``None`` draws from operating-system entropy and
                is not reproducible. The posterior itself does not depend
                on it.

        Returns:
            The fitted :class:`BVARResult`: exact posterior-mean blocks, the
            ``(n_draws, w, k)`` and ``(n_draws, k, k)`` draws, the log
            marginal likelihood, and the sample accounting.

        Raises:
            SpecificationError: If ``n_draws`` is not positive; if the prior
                is improper, leaving some coefficient variance infinite or
                non-positive; or if the prior's variances do not factor as
                :math:`\sigma_i^2 \cdot \omega_j`, which is the case for a
                :class:`~cultivars.bayes.MinnesotaPrior` with
                ``cross_equation != 1`` and for
                :class:`~cultivars.bayes.IndependentNormalWishartPrior`.
            DimensionError: If the sample has fewer than ``order + 2`` rows,
                which the prior's univariate residual scales need.
            NumericalError: If a posterior scale matrix is not positive
                definite, which a proper prior rules out except under
                numerically degenerate data.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate import BVAR
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((120, 2))
            >>> for t in range(1, 120):
            ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
            >>> model = BVAR(y, order=1, names=("growth", "inflation"))
            >>> res = model.fit(n_draws=200, seed=0)
            >>> res.n_kept, res.beta_draws.shape
            (200, (200, 3, 2))

            The posterior is exact, so the draw count changes only the
            Monte Carlo resolution of the bands, never the posterior mean
            or the marginal likelihood:

            >>> few = model.fit(n_draws=10, seed=1)
            >>> np.allclose(few.beta_mean, res.beta_mean)
            True
            >>> few.log_marginal_likelihood == res.log_marginal_likelihood
            True

            An integer seed reproduces the draws; a generator is advanced:

            >>> np.allclose(model.fit(n_draws=50, seed=3).beta_draws,
            ...             model.fit(n_draws=50, seed=3).beta_draws)
            True
            >>> gen = np.random.default_rng(3)
            >>> a = model.fit(n_draws=50, seed=gen).beta_draws
            >>> b = model.fit(n_draws=50, seed=gen).beta_draws
            >>> np.allclose(a, b)
            False
            >>> model.fit(n_draws=0)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: n_draws must be positive; got 0.

        See Also:
            * :class:`BVARResult` -- what the fit returns, with the update
              in closed form.
            * :meth:`prior_replications` and
              :func:`~cultivars.bayes.prior_predictive_check` -- inspect
              the prior on the scale of the data before fitting.
            * :class:`~cultivars.multivariate.HierarchicalBVAR` -- when
              the tightness should be estimated rather than set.
        """
        return BVARResult._from_fit(self._fit_conjugate(n_draws=n_draws, seed=seed), self)
