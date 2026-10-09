# filepath: /src/cultivars/univariate/garch.py
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
r"""Conditional-variance models: GARCH, GJR, EGARCH, and FIGARCH.

Every model here writes the series as a conditional mean plus a shock
whose variance moves,

.. math::

   y_t = \mu_t + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t,
   \qquad z_t \sim N(0, 1),

with :math:`\mu_t` a constant, zero, or an ARMA recursion, and
:math:`\sigma_t^2` one of four recursions on the past shocks:

.. math::

   \text{GARCH:}\quad
   \sigma_t^2 = \omega + \sum_i \alpha_i \varepsilon_{t-i}^2
   + \sum_j \beta_j \sigma_{t-j}^2,

   \text{GJR:}\quad
   \sigma_t^2 = \omega + \sum_i \alpha_i \varepsilon_{t-i}^2
   + \sum_k \gamma_k \varepsilon_{t-k}^2\,\mathbb 1\{\varepsilon_{t-k} < 0\}
   + \sum_j \beta_j \sigma_{t-j}^2,

   \text{EGARCH:}\quad
   \log\sigma_t^2 = \omega + \sum_i \alpha_i\bigl(|z_{t-i}| - \sqrt{2/\pi}\bigr)
   + \sum_k \gamma_k z_{t-k} + \sum_j \beta_j \log\sigma_{t-j}^2,

   \text{FIGARCH:}\quad
   \sigma_t^2 = \frac{\omega}{1 - \beta}
   + \Bigl[1 - \frac{(1 - \phi L)(1 - L)^d}{1 - \beta L}\Bigr]\varepsilon_t^2.

All four estimate the mean and the variance *jointly* by maximizing

.. math::

   \log L = -\tfrac12 \sum_t \bigl(\log 2\pi + \log\sigma_t^2
   + \varepsilon_t^2 / \sigma_t^2\bigr),

so the reported likelihood is the true joint one rather than the
two-step approximation you get from fitting a mean model and then a
variance model to its residuals. The mean uses conditional sum of
squares rather than the exact filter because the exact filter assumes a
constant innovation variance, which is the one thing these models deny.

Two commitments shape the surface. First, the group splits in two, and
the split is not cosmetic. GARCH, GJR and EGARCH are finite-order
recursions whose shocks decay geometrically, so persistence is a single
number, a half-life exists, and -- for the level families -- so does an
unconditional variance. FIGARCH applies a fractional filter whose
ARCH(:math:`\infty`) weights decay as :math:`k^{d-1}` and sum to one:
no geometric rate describes them, no half-life is defined, and the
process is not covariance stationary for any :math:`d > 0`. Sharing one
``persistence`` implementation across both would produce a plausible
number for FIGARCH that means nothing, which is why the base result
declares it and :class:`GARCHResult` and :class:`FIGARCHResult` answer
it separately. Second, a number that would mislead is withheld rather
than approximated. The log-variance family has no closed-form
unconditional variance in the reported parameters -- its stationary
level is the mean of a lognormal -- so
:attr:`GARCHResult.unconditional_variance` returns ``None`` for EGARCH
rather than ``omega / (1 - beta)``; FIGARCH refuses to ``simulate``
because no finite burn-in makes a long-memory sample forget its start;
and stationarity on the variance side is ``persistence < 1`` alone, so
the constant-mean results carry no mean-stationarity mixin, while the
ARMA-mean results carry it and keep the two verdicts apart.

Layout. The constant-mean fronts :class:`GARCH`, :class:`GJR`,
:class:`EGARCH` and :class:`FIGARCH` and the ARMA-mean fronts
:class:`ARGARCH`, :class:`ARMAGARCH`, :class:`ARMAGJR`,
:class:`ARMAEGARCH` and :class:`ARMAFIGARCH` are thin: each fixes a
family and a mean shape and delegates to ``_ShortMemoryVarianceModel``
or ``_FractionalVarianceModel`` in ``_internals``, whose ``__init__``
validates the orders through ``validate_order`` and the family and mean
through ``validate_choice`` against :data:`~cultivars.typing.Vol`. The
mean is a ``_MeanLayer`` -- ``_LinearMean`` for a lag regression,
``_ARMAMean`` when a moving-average block forces the constrained
recursion -- and the variance a ``_VarianceObjective`` whose
``is_admissible`` rejects draws with persistence at or above ``0.999``
(short memory) or a negative retained weight (long memory) before
``_linear_variance_recursion``, ``_log_variance_recursion`` or
``_arch_infinity_variance`` in ``_core`` runs the path from an
``ewma_mean_square`` backcast and ``_gaussian_negloglik`` scores it.
``_maximize_likelihood`` searches the unconstrained space, ``omega`` in
logs and the coefficients through ``softplus`` or ``sigmoid``. The
packed ``_ShortMemoryVarianceFit`` or ``_FractionalVarianceFit`` is
assembled by each result's ``_from_fit``, and ``simulate`` on the
short-memory results runs ``_simulate_conditional_variance``. Latent
rather than observation-driven volatility is
:mod:`~cultivars.univariate.stochastic_volatility`; long memory in the
level rather than the variance is
:mod:`~cultivars.univariate.fractional_integration`.

References:
    Engle, R. F. (1982). Autoregressive conditional heteroscedasticity
    with estimates of the variance of United Kingdom inflation.
    *Econometrica*, 50(4), 987-1007.

    Bollerslev, T. (1986). Generalized autoregressive conditional
    heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

    Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the
    relation between the expected value and the volatility of the
    nominal excess return on stocks. *Journal of Finance*, 48(5),
    1779-1801.

    Nelson, D. B. (1991). Conditional heteroskedasticity in asset
    returns: A new approach. *Econometrica*, 59(2), 347-370.

    Baillie, R. T., Bollerslev, T., & Mikkelsen, H. O. (1996).
    Fractionally integrated generalized autoregressive conditional
    heteroskedasticity. *Journal of Econometrics*, 74(1), 3-30.

Example:
    Four families on one series with a leverage effect, ranked by BIC:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 2500
    >>> h, e = np.ones(n), np.zeros(n)
    >>> for t in range(1, n):
    ...     h[t] = 0.02 + 0.05 * e[t - 1] ** 2 + 0.85 * h[t - 1]
    ...     h[t] += 0.10 * e[t - 1] ** 2 * (e[t - 1] < 0)
    ...     e[t] = np.sqrt(h[t]) * rng.standard_normal()
    >>> fits = [GARCH(e).fit(), GJR(e).fit(), EGARCH(e).fit(), FIGARCH(e).fit()]
    >>> table = fits[0].compare(*fits[1:], criterion="bic")
    >>> table.rows[0][0] in {"GJR(1, 1, 1)", "EGARCH(1, 1, 1)"}, table.rows[-1][0]
    (True, 'FIGARCH(1, d, 1)')
    >>> [round(fit.persistence, 2) < 1.0 for fit in fits]
    [True, True, True, False]
    >>> fits[2].unconditional_variance, fits[3].half_life
    (None, inf)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..engine._core import (
    _DEFAULT_TRUNCATION,
    _SIMULATION_BURN,
    InformationCriteria,
    SummaryTable,
    _mean_label,
)
from ..engine._internals import (
    _ConditionalVarianceResult,
    _FractionalVarianceFit,
    _FractionalVarianceModel,
    _InvertibilityMixin,
    _ShortMemoryVarianceFit,
    _ShortMemoryVarianceModel,
    _simulate_conditional_variance,
    _StationarityMixin,
)
from ..exceptions import SpecificationError

__all__ = [
    "ARGARCH",
    "ARMAEGARCH",
    "ARMAFIGARCH",
    "ARMAGARCH",
    "ARMAGJR",
    "EGARCH",
    "FIGARCH",
    "GARCH",
    "GJR",
    "ARMAFIGARCHResult",
    "ARMAGARCHResult",
    "FIGARCHResult",
    "GARCHResult",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class GARCHResult(_ConditionalVarianceResult):
    r"""A fitted finite-order conditional-variance model.

    One record for the three geometric-decay families, whose variance
    recursions on the mean residual :math:`\varepsilon_t = \sigma_t z_t` are

    .. math::

        \text{GARCH:}\quad
        \sigma_t^2 = \omega + \sum_{i=1}^{p} \alpha_i \varepsilon_{t-i}^2
        + \sum_{j=1}^{q} \beta_j \sigma_{t-j}^2,

        \text{GJR:}\quad
        \sigma_t^2 = \omega + \sum_{i} \alpha_i \varepsilon_{t-i}^2
        + \sum_{k=1}^{o} \gamma_k \varepsilon_{t-k}^2\,\mathbb 1\{\varepsilon_{t-k} < 0\}
        + \sum_{j} \beta_j \sigma_{t-j}^2,

        \text{EGARCH:}\quad
        \log\sigma_t^2 = \omega + \sum_{i} \alpha_i\bigl(|z_{t-i}| - E|z|\bigr)
        + \sum_{k} \gamma_k z_{t-k} + \sum_{j} \beta_j \log\sigma_{t-j}^2,

    with the conditional mean a constant, zero, or a short autoregression
    fitted jointly. The record carries the coefficient blocks and the
    fitted variance path; everything derived -- persistence, the
    unconditional variance, the half-life, the standardized residuals --
    is computed from them by the family's own rule, because the same
    coefficients mean different things across the three. ``vol`` is
    what makes that dispatch possible.

    Attributes:
        vol: The family that produced the fit, needed because persistence and
            the unconditional variance are different functionals across the
            three.
        ar_params: Conditional-mean AR coefficients; empty when ``ar_lags == 0``.
        alpha: Coefficients on the shock magnitude.
        gamma: Asymmetry coefficients; empty for the symmetric family.
        beta: Persistence coefficients.

    Note:
        The inherited fields are ``endog``, ``fittedvalues`` and ``resid``
        (the mean side), ``llf``, ``nobs``, ``n_params``, ``mean``,
        ``const``, ``omega`` and ``conditional_variance``; the inherited
        properties ``conditional_volatility``, ``standardized_resid``,
        ``is_covariance_stationary`` (``persistence < 1``) and
        ``half_life`` (:math:`\log 0.5 / \log \text{persistence}`) read
        them. Stationarity here is variance stationarity; the
        conditional-mean autoregression plays no part in it, and this
        record carries no mean-stationarity mixin. Standard errors are
        not yet reported by this estimator.

    See Also:
        * :class:`GARCH`, :class:`GJR`, :class:`EGARCH` -- the
          specifications that produce this.
        * :class:`ARMAGARCHResult` -- the same with a full ARMA mean and
          the mean-stationarity verdicts.
        * :class:`FIGARCHResult` -- the hyperbolic-decay family, whose
          persistence is not a single number.

    References:
        Bollerslev, T. (1986). Generalized autoregressive conditional
        heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

        Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the
        relation between the expected value and the volatility of the
        nominal excess return on stocks. *Journal of Finance*, 48(5),
        1779-1801.

        Nelson, D. B. (1991). Conditional heteroskedasticity in asset
        returns: A new approach. *Econometrica*, 59(2), 347-370.

    Example:
        A GARCH(1, 1) with :math:`\omega = 0.05`, :math:`\alpha = 0.1`,
        :math:`\beta = 0.85` and unconditional variance 1:

        >>> rng = np.random.default_rng(0)
        >>> n = 3000
        >>> h, y = np.ones(n), np.zeros(n)
        >>> for t in range(1, n):
        ...     h[t] = 0.05 + 0.1 * y[t - 1] ** 2 + 0.85 * h[t - 1]
        ...     y[t] = np.sqrt(h[t]) * rng.standard_normal()
        >>> res = GARCH(y, p=1, q=1).fit()
        >>> res.vol, res.order, list(res.params)
        ('GARCH', (1, 0, 1), ['const', 'omega', 'alpha[1]', 'beta[1]'])
        >>> bool(abs(res.persistence - 0.95) < 0.03)
        True
        >>> bool(abs(res.unconditional_variance - 1.0) < 0.1)
        True
        >>> bool(np.allclose(res.standardized_resid, res.resid / np.sqrt(res.conditional_variance)))
        True

        The log-variance family withholds the unconditional variance:

        >>> EGARCH(y, p=1, o=1, q=1).fit().unconditional_variance is None
        True
    """

    vol: str
    """``"GARCH"``, ``"GJR"`` or ``"EGARCH"``."""

    ar_params: npt.NDArray[np.float64]
    """``(ar_lags,)`` conditional-mean AR coefficients."""

    alpha: npt.NDArray[np.float64]
    """``(p,)`` coefficients on the shock magnitude."""

    gamma: npt.NDArray[np.float64]
    """``(o,)`` asymmetry coefficients; empty for GARCH."""

    beta: npt.NDArray[np.float64]
    """``(q,)`` persistence coefficients."""

    @classmethod
    def _from_fit(
        cls, fit: _ShortMemoryVarianceFit, model: _ShortMemoryVarianceModel[GARCHResult]
    ) -> GARCHResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The raw estimator output.
            model: The specification, read for the series and the mean
                choice.

        Returns:
            The assembled result.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> model = GARCH(rng.standard_normal(300), p=1, q=1)
            >>> res = GARCHResult._from_fit(model._fit_family(), model)
            >>> res.vol, res.mean, res.endog.shape
            ('GARCH', 'constant', (300,))
        """
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            mean="constant" if model.has_constant_mean else "zero",
            const=fit.const,
            omega=fit.omega,
            conditional_variance=fit.conditional_variance,
            vol=fit.vol,
            ar_params=fit.ar_params,
            alpha=fit.alpha,
            gamma=fit.gamma,
            beta=fit.beta,
        )

    @property
    def order(self) -> tuple[int, int, int]:
        """The variance order ``(p, o, q)`` recovered from the coefficient blocks.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(400)
            >>> GARCH(y, p=2, q=1).fit().order, GJR(y, p=1, o=1, q=1).fit().order
            ((2, 0, 1), (1, 1, 1))
        """
        return (self.alpha.size, self.gamma.size, self.beta.size)

    @property
    def persistence(self) -> float:
        r"""The decay rate of a shock to the conditional variance.

        For the level families this sums the coefficients with the asymmetry
        block at half weight,

        .. math::

            \sum_i \alpha_i + \tfrac12 \sum_k \gamma_k + \sum_j \beta_j,

        its unconditional frequency under a symmetric innovation
        distribution. For the log-variance family it is the
        autoregressive root of the log variance alone, :math:`\sum_j
        \beta_j`: the magnitude and sign terms are mean-zero innovations,
        not persistence.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(500)
            >>> res = GARCH(y, p=1, q=1).fit()
            >>> bool(np.isclose(res.persistence, res.alpha.sum() + res.beta.sum()))
            True
            >>> res.is_covariance_stationary
            True
        """
        if self.vol == "EGARCH":
            return float(self.beta.sum())
        return float(self.alpha.sum() + 0.5 * self.gamma.sum() + self.beta.sum())

    @property
    def unconditional_variance(self) -> float | None:
        """The long-run variance ``omega / (1 - persistence)``.

        Returns:
            ``None`` for the log-variance family, whose stationary variance is
            the mean of a lognormal and has no closed form in the reported
            parameters, and ``None`` when the process is not covariance
            stationary. Returning a number in either case would invite it
            straight into a risk calculation.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(500)
            >>> res = GARCH(y, p=1, q=1).fit()
            >>> bool(np.isclose(res.unconditional_variance, res.omega / (1 - res.persistence)))
            True
        """
        if self.vol == "EGARCH" or not self.is_covariance_stationary:
            return None
        return self.omega / (1.0 - self.persistence)

    @property
    def has_leverage(self) -> bool:
        """Whether an asymmetry block was estimated.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(400)
            >>> GARCH(y, p=1, q=1).fit().has_leverage, GJR(y, p=1, o=1, q=1).fit().has_leverage
            (False, True)
        """
        return self.gamma.size > 0

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order.

        ``const`` when the mean is constant, then ``ar.L1`` .., then
        ``omega``, ``alpha[1]`` .., ``gamma[1]`` .. and ``beta[1]`` ..;
        empty blocks contribute nothing.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(400)
            >>> list(GJR(y, p=1, o=1, q=1, ar_lags=1).fit().params)
            ['const', 'ar.L1', 'omega', 'alpha[1]', 'gamma[1]', 'beta[1]']
            >>> list(GARCH(y, p=1, q=1, mean="zero").fit().params)
            ['omega', 'alpha[1]', 'beta[1]']
        """
        out: dict[str, float] = {}
        if self.const is not None:
            out["const"] = self.const
        for i, value in enumerate(self.ar_params, start=1):
            out[f"ar.L{i}"] = float(value)
        out["omega"] = self.omega
        for i, value in enumerate(self.alpha, start=1):
            out[f"alpha[{i}]"] = float(value)
        for i, value in enumerate(self.gamma, start=1):
            out[f"gamma[{i}]"] = float(value)
        for i, value in enumerate(self.beta, start=1):
            out[f"beta[{i}]"] = float(value)
        return out

    def _mean_ma_params(self) -> npt.NDArray[np.float64]:
        """Moving-average coefficients of the conditional mean; none here.

        The hook :class:`ARMAGARCHResult` overrides so that ``simulate``
        can drive the same recursion with a full ARMA mean.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> GARCH(rng.standard_normal(300), p=1, q=1).fit()._mean_ma_params().shape
            (0,)
        """
        return np.zeros(0)

    def simulate(
        self,
        n: int = 200,
        *,
        seed: int | np.random.Generator | None = None,
        burn: int = _SIMULATION_BURN,
    ) -> npt.NDArray[np.float64]:
        """A fresh sample path at the fitted parameters.

        The variance recursion is the family's own -- linear for GARCH
        and GJR, in logs for EGARCH -- started at the unconditional
        variance and driven by Gaussian standardized shocks; the
        conditional mean is applied on top. ``burn`` initial periods are
        discarded so the kept sample does not remember the start.

        Args:
            n: Observations kept.
            seed: Seed or generator.
            burn: Periods discarded from the start.

        Returns:
            An array of shape ``(n,)``.

        Raises:
            SpecificationError: If the counts are not usable.

        Example:
            The simulated sample variance is the fitted unconditional
            variance:

            >>> rng = np.random.default_rng(0)
            >>> n = 3000
            >>> h, y = np.ones(n), np.zeros(n)
            >>> for t in range(1, n):
            ...     h[t] = 0.05 + 0.1 * y[t - 1] ** 2 + 0.85 * h[t - 1]
            ...     y[t] = np.sqrt(h[t]) * rng.standard_normal()
            >>> res = GARCH(y, p=1, q=1).fit()
            >>> path = res.simulate(20000, seed=0)
            >>> bool(abs(path.var() - res.unconditional_variance) < 0.1)
            True
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        y, _ = _simulate_conditional_variance(
            n,
            vol=self.vol,
            omega=self.omega,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            const=self.const or 0.0,
            ar=self.ar_params,
            ma=self._mean_ma_params(),
            rng=rng,
            burn=burn,
        )
        return y

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(400)
            >>> GARCH(y, p=1, q=1).fit()._comparison_label()
            'GARCH(1, 1)'
            >>> GJR(y, p=1, o=1, q=1).fit()._comparison_label()
            'GJR(1, 1, 1)'
        """
        p, o, q = self.order
        return f"{self.vol}({p}, {o}, {q})" if o else f"{self.vol}({p}, {q})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        Metadata carries the label, mean choice, AR lag count, sample
        size, and the likelihood with its criteria; the coefficient table
        is :attr:`params` in order; the notes give persistence, the
        stationarity verdict and half-life, the unconditional variance
        where it is defined, and the reason it is withheld for EGARCH.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> table = GARCH(rng.standard_normal(400), p=1, q=1).fit()._summary_table()
            >>> table.title, table.columns, table.notes[0][:12]
            ('GARCH(1, 1) Results', ('', 'coef'), 'Persistence:')
        """
        ic: InformationCriteria = self.information_criteria
        uncond = self.unconditional_variance
        notes = [
            f"Persistence: {self.persistence:.4f}   "
            f"Covariance stationary: {self.is_covariance_stationary}   "
            f"Half-life: {self.half_life:.1f}",
        ]
        if uncond is not None:
            notes.append(f"Unconditional variance: {uncond:.4f}")
        elif self.vol == "EGARCH":
            notes.append(
                "Unconditional variance is not reported for a log-variance model; "
                "its stationary level is the mean of a lognormal."
            )
        notes.append("Standard errors are not yet available for this estimator.")
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=(
                ("Model", self._comparison_label()),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Mean", self.mean),
                ("AIC", f"{ic.aic:.3f}"),
                ("AR lags", f"{self.ar_params.size}"),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=tuple(notes),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class FIGARCHResult(_ConditionalVarianceResult):
    r"""A fitted fractionally integrated conditional-variance model.

    The FIGARCH(1, d, 1) variance of Baillie, Bollerslev and Mikkelsen,

    .. math::

        \sigma_t^2 = \omega + \beta\,\sigma_{t-1}^2
        + \bigl[1 - \beta L - (1 - \phi L)(1 - L)^d\bigr]\,\varepsilon_t^2,

    is estimated in its ARCH(:math:`\infty`) form

    .. math::

        \sigma_t^2 = \frac{\omega}{1 - \beta} + \sum_{k=1}^{\infty} \lambda_k\,
        \varepsilon_{t-k}^2,
        \qquad
        \lambda(L) = 1 - \frac{(1 - \phi L)(1 - L)^d}{1 - \beta L},

    with the sum cut at ``truncation`` lags and the tail applied to a
    backcast. The weights decay as :math:`k^{d-1}` rather than
    geometrically, and :math:`\lambda(1) = 1` for every :math:`d > 0`:
    the shocks' total weight is unity, which is why :attr:`persistence`
    is one by construction, no half-life exists, and the process is
    not covariance stationary. Everything the geometric families report
    from a single decay rate is therefore withheld here rather than
    approximated; the one number this family adds is ``d``.

    Attributes:
        truncation: Number of infinite-order weights retained.
        phi: Short-memory numerator weight of the fractional polynomial.
        d: Fractional integration order.
        beta: Denominator weight.

    Note:
        ``phi``, ``d`` and ``beta`` are each searched on :math:`(0, 1)`
        and ``omega`` on :math:`(0, \infty)`, with any draw whose
        retained weights turn negative rejected; the estimate cannot land
        on ``d = 0`` exactly, so :attr:`has_long_memory` is a threshold
        on the point estimate, not a test, and a short-memory series will
        still come back with a positive ``d``. The inherited fields and
        properties are those of :class:`GARCHResult`; ``half_life``
        returns ``inf`` here and there is no ``unconditional_variance``.
        Standard errors are not yet reported by this estimator.

    See Also:
        * :class:`FIGARCH` -- the specification that produces this.
        * :class:`ARMAFIGARCHResult` -- the same with a full ARMA mean.
        * :class:`GARCHResult` -- the geometric-decay families, whose
          persistence is a single number.
        * :mod:`~cultivars.univariate.fractional_integration` -- long
          memory in the mean rather than the variance.

    References:
        Baillie, R. T., Bollerslev, T., & Mikkelsen, H. O. (1996).
        Fractionally integrated generalized autoregressive conditional
        heteroskedasticity. *Journal of Econometrics*, 74(1), 3-30.

        Chung, C.-F. (1999). Estimating the fractionally integrated GARCH
        model. Working paper, National Taiwan University.

    Example:
        A series with a persistent GARCH variance, read as long memory:

        >>> rng = np.random.default_rng(0)
        >>> n = 1500
        >>> h, y = np.ones(n), np.zeros(n)
        >>> for t in range(1, n):
        ...     h[t] = 0.05 + 0.1 * y[t - 1] ** 2 + 0.85 * h[t - 1]
        ...     y[t] = np.sqrt(h[t]) * rng.standard_normal()
        >>> res = FIGARCH(y).fit()
        >>> list(res.params), res.truncation
        (['const', 'omega', 'phi', 'd', 'beta'], 1000)
        >>> bool(0.0 < res.d < 1.0), res.has_long_memory
        (True, True)
        >>> res.persistence, res.is_covariance_stationary, res.half_life
        (1.0, False, inf)
    """

    truncation: int
    """Number of ARCH(:math:`\\infty`) weights retained before the backcast tail."""

    phi: float
    """Short-memory numerator weight, in :math:`(0, 1)`."""

    d: float
    """Fractional integration order, in :math:`(0, 1)`."""

    beta: float
    """Denominator weight, in :math:`(0, 1)`."""

    @classmethod
    def _from_fit(
        cls, fit: _FractionalVarianceFit, model: _FractionalVarianceModel[FIGARCHResult]
    ) -> FIGARCHResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The raw estimator output.
            model: The specification, read for the series, the mean
                choice and the truncation lag.

        Returns:
            The assembled result.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> model = FIGARCH(rng.standard_normal(600), truncation=300)
            >>> res = FIGARCHResult._from_fit(model._fit_family(), model)
            >>> res.truncation, res.mean, res.endog.shape
            (300, 'constant', (600,))
        """
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            mean="constant" if model.has_constant_mean else "zero",
            const=fit.const,
            omega=fit.omega,
            conditional_variance=fit.conditional_variance,
            truncation=model.truncation,
            phi=fit.phi,
            d=fit.d,
            beta=fit.beta,
        )

    @property
    def persistence(self) -> float:
        r"""Unity, by construction.

        A fractionally integrated variance is not covariance stationary for any
        ``d > 0``: shocks decay hyperbolically rather than geometrically, so no
        geometric rate describes them, and the ARCH(:math:`\infty`) weights
        sum to :math:`\lambda(1) = 1` exactly. Reporting the finite-order
        formula ``phi + beta`` here would produce a plausible number that
        means nothing, which is why :class:`GARCHResult` and this class do
        not share one.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> FIGARCH(rng.standard_normal(600)).fit().persistence
            1.0
        """
        return 1.0

    @property
    def is_covariance_stationary(self) -> bool:
        """Always ``False``; see :attr:`persistence`.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> FIGARCH(rng.standard_normal(600)).fit().is_covariance_stationary
            False
        """
        return False

    @property
    def has_long_memory(self) -> bool:
        """Whether ``d`` is far enough from zero to imply hyperbolic decay.

        A threshold on the point estimate at ``1e-3``, not a test; the
        parameterization keeps ``d`` strictly inside :math:`(0, 1)`, so
        this is ``True`` for nearly every fit.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> res = FIGARCH(rng.standard_normal(600)).fit()
            >>> res.has_long_memory == (res.d > 1e-3)
            True
        """
        return self.d > 1e-3

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order.

        ``const`` when the mean is constant, then ``omega``, ``phi``,
        ``d`` and ``beta``.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(600)
            >>> list(FIGARCH(y, mean="zero").fit().params)
            ['omega', 'phi', 'd', 'beta']
        """
        out: dict[str, float] = {}
        if self.const is not None:
            out["const"] = self.const
        out["omega"] = self.omega
        out["phi"] = self.phi
        out["d"] = self.d
        out["beta"] = self.beta
        return out

    def simulate(
        self,
        n: int = 200,
        *,
        seed: int | np.random.Generator | None = None,
        burn: int = _SIMULATION_BURN,
    ) -> npt.NDArray[np.float64]:
        """Refused: a long-memory variance has no finite-state recursion to start from.

        The FIGARCH variance is an infinite sum over past squared shocks
        whose weights decay hyperbolically, so no burn-in of fixed length
        makes a sample forget its start; a truncated simulation would
        carry a bias the fitted model does not describe.

        Args:
            n: Ignored.
            seed: Ignored.
            burn: Ignored.

        Raises:
            SpecificationError: Always.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> FIGARCH(rng.standard_normal(600)).fit().simulate(10)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: FIGARCH results do not simulate: ...
        """
        raise SpecificationError(
            "FIGARCH results do not simulate: the long-memory variance recursion has no "
            "finite-length burn-in after which the sample forgets its start."
        )

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> FIGARCH(rng.standard_normal(600)).fit()._comparison_label()
            'FIGARCH(1, d, 1)'
        """
        return "FIGARCH(1, d, 1)"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        Metadata carries the label, mean choice, truncation lag, sample
        size, and the likelihood with its criteria; the coefficient table
        is :attr:`params` in order; the notes give the long-memory
        verdict with ``d`` and state why no half-life is reported.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> table = FIGARCH(rng.standard_normal(600)).fit()._summary_table()
            >>> table.title, table.metadata[4], table.notes[0][:17]
            ('FIGARCH(1, d, 1) Results', ('Truncation', '1000'), 'Long memory: True')
        """
        ic: InformationCriteria = self.information_criteria
        return SummaryTable(
            title="FIGARCH(1, d, 1) Results",
            metadata=(
                ("Model", "FIGARCH(1, d, 1)"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Mean", self.mean),
                ("AIC", f"{ic.aic:.3f}"),
                ("Truncation", f"{self.truncation}"),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=(
                f"Long memory: {self.has_long_memory}   d = {self.d:.4f}",
                "Not covariance stationary for any d > 0: shocks to the variance "
                "decay hyperbolically, so no half-life is defined.",
                "Standard errors are not yet available for this estimator.",
            ),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ARMAGARCHResult(GARCHResult, _StationarityMixin, _InvertibilityMixin):
    r"""A finite-order variance model under an ARMA conditional mean.

    The mean is

    .. math::

        y_t = c + \sum_{i=1}^{p} \phi_i\, y_{t-i} + \varepsilon_t
        + \sum_{j=1}^{q} \theta_j\, \varepsilon_{t-j},
        \qquad \varepsilon_t = \sigma_t z_t,

    with :math:`\sigma_t^2` one of the :class:`GARCHResult` recursions and
    the two blocks estimated jointly, so ``llf`` is the joint likelihood
    rather than the two-step approximation. ``const`` is the recursion
    intercept :math:`c`, not the unconditional mean
    :math:`c / (1 - \sum_i \phi_i)`; that is the
    :class:`~cultivars.univariate.autoregression.ARResult` convention,
    not the :class:`~cultivars.univariate.box_jenkins.ARMAResult` one.

    Subclasses :class:`GARCHResult` rather than reimplementing it: every
    variance property it exposes -- persistence, the unconditional
    variance, the leverage flag -- is a functional of the variance block
    alone and stays exactly as true when the mean grows a moving-average
    term. What is added is the mean side of the story: the
    moving-average block, and the stationarity and invertibility verdicts
    from the two mixins, whose meaning depends on how the mean was
    searched (see :attr:`is_structurally_constrained`).

    Attributes:
        mean_order: The conditional-mean order ``(ar_lags, ma_lags)``.
        ma_params: Moving-average coefficients of ``1 + theta_1 L + ...``;
            empty when ``ma_lags == 0``.

    Note:
        ``fittedvalues``, ``resid`` and ``conditional_variance`` are
        aligned with the last ``nobs`` observations of ``endog``, the
        mean's presample having been dropped; ``resid`` is the mean
        residual :math:`\varepsilon_t`, so ``standardized_resid`` divides
        it by :math:`\sigma_t`. ``stability`` and ``invertibility`` hold
        the companion eigenvalues (the inverse roots), so their
        ``max_modulus`` is below one for a stationary or invertible block.

    See Also:
        * :class:`ARMAGARCH`, :class:`ARMAGJR`, :class:`ARMAEGARCH`,
          :class:`ARGARCH` -- the specifications that produce this.
        * :class:`GARCHResult` -- the same variance under a constant or
          zero mean, and the source of every variance property here.
        * :class:`~cultivars.univariate.box_jenkins.ARMAResult` -- the
          same mean with a constant variance and the exact likelihood.

    Example:
        An ARMA(1, 1) mean over a GARCH(1, 1) variance, recovered jointly:

        >>> rng = np.random.default_rng(0)
        >>> n = 2000
        >>> h, e, y = np.ones(n), np.zeros(n), np.zeros(n)
        >>> for t in range(1, n):
        ...     h[t] = 0.05 + 0.1 * e[t - 1] ** 2 + 0.85 * h[t - 1]
        ...     e[t] = np.sqrt(h[t]) * rng.standard_normal()
        ...     y[t] = 0.2 + 0.5 * y[t - 1] + e[t] + 0.3 * e[t - 1]
        >>> res = ARMAGARCH(y, ar_lags=1, ma_lags=1).fit()
        >>> res.mean_order, res.order, res.nobs
        ((1, 1), (1, 0, 1), 1999)
        >>> list(res.params)
        ['const', 'ar.L1', 'ma.L1', 'omega', 'alpha[1]', 'beta[1]']
        >>> bool(abs(res.ar_params[0] - 0.5) < 0.1), bool(abs(res.ma_params[0] - 0.3) < 0.1)
        (True, True)
        >>> res.is_stationary, res.is_invertible, res.is_structurally_constrained
        (True, True, True)
        >>> bool(abs(res.persistence - 0.95) < 0.05)
        True
    """

    mean_order: tuple[int, int]
    """``(ar_lags, ma_lags)`` of the conditional mean."""

    ma_params: npt.NDArray[np.float64]
    """``(ma_lags,)`` moving-average coefficients, plus-sign convention."""

    @classmethod
    def _from_fit(
        cls,
        fit: _ShortMemoryVarianceFit,
        model: _ShortMemoryVarianceModel[GARCHResult],
    ) -> ARMAGARCHResult:
        """Assemble the public result from a raw fit and its specification.

        The model is annotated at the parent's result binding rather than this
        class's. Nothing read here depends on which result the model produces
        -- ``mean_order`` and ``has_constant_mean`` live on the shared
        specification base -- so narrowing the binding would break the override
        contract to express a dependency that does not exist.

        Args:
            fit: The raw estimator output.
            model: The specification, read for the series, the mean
                choice and the mean order.

        Returns:
            The assembled result.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> model = ARMAGARCH(rng.standard_normal(400), ar_lags=1, ma_lags=1)
            >>> res = ARMAGARCHResult._from_fit(model._fit_family(), model)
            >>> res.mean_order, res.mean, res.endog.shape
            ((1, 1), 'constant', (400,))
        """
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            mean="constant" if model.has_constant_mean else "zero",
            const=fit.const,
            omega=fit.omega,
            conditional_variance=fit.conditional_variance,
            vol=fit.vol,
            ar_params=fit.ar_params,
            alpha=fit.alpha,
            gamma=fit.gamma,
            beta=fit.beta,
            mean_order=model.mean_order,
            ma_params=fit.ma_params,
        )

    @property
    def is_structurally_constrained(self) -> bool:
        """Whether the mean was searched inside the stationary-invertible region.

        ``True`` exactly when a moving-average block is present, because only
        then does the residual recursion feed on its own output and need the
        constraint to stay finite. When ``False``, the autoregressive lag
        weights were estimated unconstrained and :attr:`is_stationary` carries
        information about the data instead of about the transform.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(500)
            >>> ARGARCH(y, ar_lags=1).fit().is_structurally_constrained
            False
            >>> ARMAGARCH(y, ar_lags=1, ma_lags=1).fit().is_structurally_constrained
            True
        """
        return self.mean_order[1] > 0

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order.

        ``const`` when the mean is constant, then ``ar.L1`` .., ``ma.L1``
        .., ``omega``, ``alpha[1]`` .., ``gamma[1]`` .. and ``beta[1]`` ..;
        empty blocks contribute nothing.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(500)
            >>> list(ARMAGJR(y, ar_lags=1, ma_lags=1, mean="zero").fit().params)
            ['ar.L1', 'ma.L1', 'omega', 'alpha[1]', 'gamma[1]', 'beta[1]']
        """
        out: dict[str, float] = {}
        if self.const is not None:
            out["const"] = self.const
        for i, value in enumerate(self.ar_params, start=1):
            out[f"ar.L{i}"] = float(value)
        for i, value in enumerate(self.ma_params, start=1):
            out[f"ma.L{i}"] = float(value)
        out["omega"] = self.omega
        for i, value in enumerate(self.alpha, start=1):
            out[f"alpha[{i}]"] = float(value)
        for i, value in enumerate(self.gamma, start=1):
            out[f"gamma[{i}]"] = float(value)
        for i, value in enumerate(self.beta, start=1):
            out[f"beta[{i}]"] = float(value)
        return out

    def _mean_ma_params(self) -> npt.NDArray[np.float64]:
        """Moving-average coefficients of the conditional mean.

        Overrides the parent's empty block so that the inherited
        ``simulate`` drives the ARMA recursion.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> res = ARMAGARCH(rng.standard_normal(500), ar_lags=1, ma_lags=2).fit()
            >>> res._mean_ma_params().shape
            (2,)
        """
        return self.ma_params

    def _variance_label(self) -> str:
        """Name the variance family and its order.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(500)
            >>> ARMAGJR(y, ar_lags=1, ma_lags=1).fit()._variance_label()
            'GJR(1, 1, 1)'
        """
        p, o, q = self.order
        return f"{self.vol}({p}, {o}, {q})" if o else f"{self.vol}({p}, {q})"

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        The mean is named by ``_mean_label`` so a pure autoregression
        reads ``AR(p)`` rather than ``ARMA(p, 0)``.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(500)
            >>> ARMAGARCH(y, ar_lags=1, ma_lags=1).fit()._comparison_label()
            'ARMA(1, 1)-GARCH(1, 1)'
            >>> ARGARCH(y, ar_lags=2).fit()._comparison_label()
            'AR(2)-GARCH(1, 1)'
        """
        mean = _mean_label(self.mean_order, has_const=self.const is not None)
        return f"{mean}-{self._variance_label()}"

    def _mean_notes(self) -> list[str]:
        """Diagnostics for the conditional-mean block.

        One line with the stationarity verdict and largest companion
        modulus, extended with the invertibility verdict when a
        moving-average block exists, then a line saying whether those
        verdicts describe the data or verify the transform.

        Returns:
            One or two note strings.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> notes = ARGARCH(rng.standard_normal(500), ar_lags=1).fit()._mean_notes()
            >>> len(notes), notes[0][:22], notes[1][:22]
            (2, 'Mean: stationary True ', 'The mean is a regressi')
        """
        notes = [
            f"Mean: stationary {self.is_stationary} "
            f"(max |AR root| = {self.stability.max_modulus:.4f})"
            + (
                f", invertible {self.is_invertible} "
                f"(max |MA root| = {self.invertibility.max_modulus:.4f})"
                if self.ma_params.size
                else ""
            )
            + "."
        ]
        if self.is_structurally_constrained:
            notes.append(
                "Both mean blocks were searched through the partial autocorrelations, "
                "so stationarity and invertibility hold by construction; the line above "
                "verifies the transform rather than describing the data."
            )
        elif self.ar_params.size:
            notes.append(
                "The mean is a regression on lagged levels with unconstrained weights, "
                "so the stationarity verdict above is a statement about the fit."
            )
        return notes

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        Metadata carries the joint label, the mean and variance labels
        separately, the sample size, the likelihood with its criteria,
        persistence and the parameter count; the coefficient table is
        :attr:`params` in order; the notes are :meth:`_mean_notes`
        followed by the variance diagnostics, the unconditional variance
        or the reason it is withheld, and how the joint likelihood was
        formed.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> res = ARMAGARCH(rng.standard_normal(500), ar_lags=1, ma_lags=1).fit()
            >>> table = res._summary_table()
            >>> table.title, len(table.notes)
            ('ARMA(1, 1)-GARCH(1, 1) Results', 6)
            >>> table.metadata[2], table.metadata[4]
            (('Mean', 'ARMA(1, 1)'), ('Variance', 'GARCH(1, 1)'))
        """
        ic: InformationCriteria = self.information_criteria
        notes = self._mean_notes()
        notes.append(
            f"Variance: persistence {self.persistence:.4f}, covariance stationary "
            f"{self.is_covariance_stationary}"
            + (f", half-life {self.half_life:.2f} periods." if self.half_life else ".")
        )
        if self.unconditional_variance is not None:
            notes.append(f"Unconditional variance: {self.unconditional_variance:.6f}.")
        elif self.vol == "EGARCH":
            notes.append(
                "No unconditional variance is reported: the log-variance family's "
                "stationary level is the mean of a lognormal."
            )
        notes.append(
            "Mean and variance are estimated jointly, so the log-likelihood is the "
            "true joint one; the mean uses conditional sum of squares because the "
            "exact-ML filter assumes a constant innovation variance."
        )
        notes.append("Standard errors are not yet available for this estimator.")
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=(
                ("Model", self._comparison_label()),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Mean", _mean_label(self.mean_order, has_const=self.const is not None)),
                ("AIC", f"{ic.aic:.3f}"),
                ("Variance", self._variance_label()),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
                ("Persistence", f"{self.persistence:.4f}"),
                ("Parameters", f"{self.n_params:g}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=tuple(notes),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ARMAFIGARCHResult(FIGARCHResult, _StationarityMixin, _InvertibilityMixin):
    r"""A fractionally integrated variance model under an ARMA conditional mean.

    The mean is

    .. math::

        y_t = c + \sum_{i=1}^{p} \phi_i\, y_{t-i} + \varepsilon_t
        + \sum_{j=1}^{q} \theta_j\, \varepsilon_{t-j},
        \qquad \varepsilon_t = \sigma_t z_t,

    with :math:`\sigma_t^2` the FIGARCH(1, d, 1) recursion of
    :class:`FIGARCHResult`, estimated jointly. Two long-memory-adjacent
    objects live in one model here and they are not the same object. The
    *mean* has an ARMA structure whose shocks decay geometrically; the
    *variance* is fractionally integrated and its shocks decay
    hyperbolically. :attr:`is_stationary` speaks only to the first,
    :attr:`is_covariance_stationary` only to the second, and the latter
    is ``False`` by construction for any ``d > 0``. ``const`` is the
    recursion intercept :math:`c`, not the unconditional mean.

    Everything on the variance side -- ``persistence``, ``half_life``,
    ``has_long_memory``, the refusal to ``simulate`` -- is inherited
    unchanged from :class:`FIGARCHResult`; what is added is the mean
    block and its stationarity and invertibility verdicts, whose meaning
    depends on how the mean was searched (see
    :attr:`is_structurally_constrained`).

    Attributes:
        mean_order: The conditional-mean order ``(ar_lags, ma_lags)``.
        ar_params: Autoregressive coefficients of the mean; empty when
            ``ar_lags == 0``.
        ma_params: Moving-average coefficients of ``1 + theta_1 L + ...``;
            empty when ``ma_lags == 0``.

    Note:
        ``fittedvalues``, ``resid`` and ``conditional_variance`` are
        aligned with the last ``nobs = len(endog) - max(ar_lags,
        ma_lags)`` observations. ``stability`` and ``invertibility`` hold
        the companion eigenvalues (the inverse roots), so their
        ``max_modulus`` is below one for a stationary or invertible block.
        A fit with ``mean_order == (0, 0)`` is a :class:`FIGARCH` under
        another label.

    See Also:
        * :class:`ARMAFIGARCH` -- the specification that produces this.
        * :class:`FIGARCHResult` -- the same variance under a constant or
          zero mean, and the source of every variance property here.
        * :class:`ARMAGARCHResult` -- the same mean over a geometric-decay
          variance, where a half-life exists.

    Example:
        An ARMA(1, 1) mean over a persistent variance, with the two
        verdicts kept apart:

        >>> rng = np.random.default_rng(0)
        >>> n = 1500
        >>> h, e, y = np.ones(n), np.zeros(n), np.zeros(n)
        >>> for t in range(1, n):
        ...     h[t] = 0.05 + 0.1 * e[t - 1] ** 2 + 0.85 * h[t - 1]
        ...     e[t] = np.sqrt(h[t]) * rng.standard_normal()
        ...     y[t] = 0.2 + 0.5 * y[t - 1] + e[t] + 0.3 * e[t - 1]
        >>> res = ARMAFIGARCH(y, ar_lags=1, ma_lags=1).fit()
        >>> list(res.params), res.nobs
        (['const', 'ar.L1', 'ma.L1', 'omega', 'phi', 'd', 'beta'], 1499)
        >>> bool(abs(res.ar_params[0] - 0.5) < 0.1), bool(abs(res.ma_params[0] - 0.3) < 0.1)
        (True, True)
        >>> res.is_stationary, res.is_covariance_stationary, res.half_life
        (True, False, inf)
    """

    mean_order: tuple[int, int]
    """``(ar_lags, ma_lags)`` of the conditional mean."""

    ar_params: npt.NDArray[np.float64]
    """``(ar_lags,)`` autoregressive coefficients of the mean."""

    ma_params: npt.NDArray[np.float64]
    """``(ma_lags,)`` moving-average coefficients, plus-sign convention."""

    @classmethod
    def _from_fit(
        cls,
        fit: _FractionalVarianceFit,
        model: _FractionalVarianceModel[FIGARCHResult],
    ) -> ARMAFIGARCHResult:
        """Assemble the public result from a raw fit and its specification.

        The model is annotated at the parent's result binding for the
        reason given on :meth:`ARMAGARCHResult._from_fit`: nothing read
        here depends on which result the model produces.

        Args:
            fit: The raw estimator output.
            model: The specification, read for the series, the mean
                choice, the mean order and the truncation lag.

        Returns:
            The assembled result.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> model = ARMAFIGARCH(rng.standard_normal(600), ar_lags=1, ma_lags=1, truncation=200)
            >>> res = ARMAFIGARCHResult._from_fit(model._fit_family(), model)
            >>> res.truncation, res.mean_order, res.mean
            (200, (1, 1), 'constant')
        """
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            mean="constant" if model.has_constant_mean else "zero",
            const=fit.const,
            omega=fit.omega,
            conditional_variance=fit.conditional_variance,
            truncation=model.truncation,
            phi=fit.phi,
            d=fit.d,
            beta=fit.beta,
            mean_order=model.mean_order,
            ar_params=fit.ar_params,
            ma_params=fit.ma_params,
        )

    @property
    def is_structurally_constrained(self) -> bool:
        """Whether the mean was searched inside the stationary-invertible region.

        ``True`` exactly when a moving-average block is present; see
        :attr:`ARMAGARCHResult.is_structurally_constrained` for why. When
        ``False`` the autoregressive weights were estimated unconstrained
        and :attr:`is_stationary` describes the fit rather than the
        transform.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(600)
            >>> ARMAFIGARCH(y, ar_lags=2, ma_lags=0).fit().is_structurally_constrained
            False
            >>> ARMAFIGARCH(y, ar_lags=1, ma_lags=1).fit().is_structurally_constrained
            True
        """
        return self.mean_order[1] > 0

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order.

        ``const`` when the mean is constant, then ``ar.L1`` .., ``ma.L1``
        .., ``omega``, ``phi``, ``d`` and ``beta``.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(600)
            >>> list(ARMAFIGARCH(y, ar_lags=1, ma_lags=1, mean="zero").fit().params)
            ['ar.L1', 'ma.L1', 'omega', 'phi', 'd', 'beta']
        """
        out: dict[str, float] = {}
        if self.const is not None:
            out["const"] = self.const
        for i, value in enumerate(self.ar_params, start=1):
            out[f"ar.L{i}"] = float(value)
        for i, value in enumerate(self.ma_params, start=1):
            out[f"ma.L{i}"] = float(value)
        out["omega"] = self.omega
        out["phi"] = self.phi
        out["d"] = self.d
        out["beta"] = self.beta
        return out

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(600)
            >>> ARMAFIGARCH(y, ar_lags=1, ma_lags=1).fit()._comparison_label()
            'ARMA(1, 1)-FIGARCH(1, d, 1)'
            >>> ARMAFIGARCH(y, ar_lags=2, ma_lags=0).fit()._comparison_label()
            'AR(2)-FIGARCH(1, d, 1)'
        """
        mean = _mean_label(self.mean_order, has_const=self.const is not None)
        return f"{mean}-FIGARCH(1, d, 1)"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        Metadata carries the joint label, the mean label, truncation lag,
        sample size, the likelihood with its criteria, ``d`` and the
        parameter count; the coefficient table is :attr:`params` in
        order; the notes give the mean verdicts and whether they describe
        the data or the transform, the long-memory verdict with the
        reason no half-life exists, and how the joint likelihood was
        formed.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> res = ARMAFIGARCH(rng.standard_normal(600), ar_lags=1, ma_lags=1).fit()
            >>> table = res._summary_table()
            >>> table.title, table.metadata[8][0], len(table.notes)
            ('ARMA(1, 1)-FIGARCH(1, d, 1) Results', 'd', 5)
            >>> len(ARMAFIGARCH(res.endog, ar_lags=1, ma_lags=0).fit()._summary_table().notes)
            4
        """
        ic: InformationCriteria = self.information_criteria
        notes = [
            f"Mean: stationary {self.is_stationary} "
            f"(max |AR root| = {self.stability.max_modulus:.4f})"
            + (
                f", invertible {self.is_invertible} "
                f"(max |MA root| = {self.invertibility.max_modulus:.4f})"
                if self.ma_params.size
                else ""
            )
            + ". The mean's shocks decay geometrically; the variance's do not.",
        ]
        if self.is_structurally_constrained:
            notes.append(
                "Both mean blocks were searched through the partial autocorrelations, "
                "so the verdict above verifies the transform rather than the data."
            )
        notes.extend(
            (
                f"Variance: long memory {self.has_long_memory}, d = {self.d:.4f}. Not "
                f"covariance stationary for any d > 0, so no half-life is defined and "
                f"persistence is reported as 1 by construction.",
                "Mean and variance are estimated jointly, so the log-likelihood is the "
                "true joint one; the mean uses conditional sum of squares because the "
                "exact-ML filter assumes a constant innovation variance.",
                "Standard errors are not yet available for this estimator.",
            )
        )
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=(
                ("Model", self._comparison_label()),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Mean", _mean_label(self.mean_order, has_const=self.const is not None)),
                ("AIC", f"{ic.aic:.3f}"),
                ("Truncation", f"{self.truncation}"),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
                ("d", f"{self.d:.4f}"),
                ("Parameters", f"{self.n_params:g}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=tuple(notes),
        )


class GARCH(_ShortMemoryVarianceModel[GARCHResult]):
    r"""Symmetric GARCH(p, q) with an optional AR mean.

    The variance recursion

    .. math::

        \sigma_t^2 = \omega + \sum_{i=1}^{p} \alpha_i \varepsilon_{t-i}^2
        + \sum_{j=1}^{q} \beta_j \sigma_{t-j}^2

    under a constant, zero, or autoregressive mean, the two estimated
    jointly by Gaussian maximum likelihood. Positivity is enforced by
    searching :math:`\omega` in logs and the coefficients through a
    transform that keeps them non-negative; stationarity is not imposed,
    so a persistence at or above one is reported as such rather than
    clipped. Both :math:`p = 0` and :math:`q = 0` are allowed -- the
    first is a pure variance autoregression, the second an ARCH(p) -- and
    the asymmetry order is fixed at zero; :class:`GJR` is the family with
    a sign term.

    Attributes:
        _vol: The family name, ``"GARCH"``.
        _p: ARCH order.
        _o: Asymmetry order, always zero here.
        _q: GARCH order.
        _endog: The validated series.
        _ar_lags: Conditional-mean AR order.
        _ma_lags: Conditional-mean MA order, always zero here.
        _const: Whether the mean carries an intercept.

    Args:
        endog: The series, typically returns or residuals.
        p: ARCH order, the number of lagged squared residuals.
        q: GARCH order, the number of lagged variances.
        ar_lags: Conditional-mean AR order.
        mean: ``"constant"`` or ``"zero"``.

    Raises:
        SpecificationError: If an order is not a non-negative integer, or
            ``mean`` is not one of the two choices.
        DimensionError: If ``endog`` is not one-dimensional or is shorter
            than the orders require.
        NumericalError: If ``endog`` contains non-finite values.

    See Also:
        * :class:`GARCHResult` -- what :meth:`fit` returns.
        * :class:`GJR` -- the same with a leverage term.
        * :class:`EGARCH` -- the log-variance family.
        * :class:`ARGARCH`, :class:`ARMAGARCH` -- the same variance under
          a mean that is the point of the specification.
        * :mod:`~cultivars.univariate.stochastic_volatility` -- latent
          rather than observation-driven volatility.

    References:
        Bollerslev, T. (1986). Generalized autoregressive conditional
        heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

        Engle, R. F. (1982). Autoregressive conditional heteroscedasticity
        with estimates of the variance of United Kingdom inflation.
        *Econometrica*, 50(4), 987-1007.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> e = np.zeros(2000)
        >>> s2 = np.full(2000, 2.5)
        >>> z = rng.standard_normal(2000)
        >>> for t in range(1, 2000):
        ...     s2[t] = 0.05 + 0.08 * e[t - 1] ** 2 + 0.90 * s2[t - 1]
        ...     e[t] = np.sqrt(s2[t]) * z[t]
        >>> res = GARCH(e).fit()
        >>> res.is_covariance_stationary
        True
        >>> bool(abs(res.persistence - 0.98) < 0.03)
        True

        An ARCH(2) and a zero-mean fit are the same front with the orders
        changed:

        >>> GARCH(e, p=2, q=0).fit().order, list(GARCH(e, mean="zero").fit().params)
        ((2, 0, 0), ['omega', 'alpha[1]', 'beta[1]'])
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        p: int = 1,
        q: int = 1,
        ar_lags: int = 0,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data.

        Args:
            endog: The series.
            p: ARCH order.
            q: GARCH order.
            ar_lags: Conditional-mean AR order.
            mean: ``"constant"`` or ``"zero"``.

        Raises:
            SpecificationError: If an order or the mean choice is invalid.
            DimensionError: If the series has the wrong shape or is too
                short.
            NumericalError: If the series is not finite.

        Example:
            >>> import numpy as np
            >>> model = GARCH(np.arange(50.0), p=2, q=1, ar_lags=1)
            >>> model.vol, model.order, model.mean_order, model.has_constant_mean
            ('GARCH', (2, 0, 1), (1, 0), True)
            >>> GARCH(np.arange(50.0), p=1.5)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: p must be an integer; got 1.5.
        """
        super().__init__(endog, vol="GARCH", p=p, o=0, q=q, ar_lags=ar_lags, mean=mean)

    def fit(self) -> GARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood.

        Returns:
            The fitted :class:`GARCHResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = GARCH(rng.standard_normal(400)).fit()
            >>> res.order, res.nobs, res.n_params
            ((1, 0, 1), 400, 4)
        """
        return GARCHResult._from_fit(self._fit_family(), self)


class GJR(_ShortMemoryVarianceModel[GARCHResult]):
    r"""GJR-GARCH(p, o, q): a level model with a sign-asymmetric ARCH term.

    The variance recursion

    .. math::

        \sigma_t^2 = \omega + \sum_{i=1}^{p} \alpha_i \varepsilon_{t-i}^2
        + \sum_{k=1}^{o} \gamma_k\, \varepsilon_{t-k}^2\,
        \mathbb 1\{\varepsilon_{t-k} < 0\}
        + \sum_{j=1}^{q} \beta_j \sigma_{t-j}^2.

    The asymmetry block loads only on negative shocks, so ``gamma > 0`` is the
    leverage effect -- bad news raising volatility more than good news of the
    same size. Under a symmetric innovation the indicator is on half the
    time, which is why the result's persistence counts :math:`\gamma` at
    half weight. :math:`\omega`, :math:`\alpha` and :math:`\beta` are
    kept positive by their transforms; :math:`\gamma` is searched
    unconstrained, so a negative estimate is reported as the reverse
    asymmetry rather than clipped to zero. The asymmetry order must be at
    least one: a GJR with ``o = 0`` is a :class:`GARCH`, and the name
    should not be able to claim a leverage term it does not carry.

    Attributes:
        _vol: The family name, ``"GJR"``.
        _p: ARCH order.
        _o: Asymmetry order.
        _q: GARCH order.
        _endog: The validated series.
        _ar_lags: Conditional-mean AR order.
        _ma_lags: Conditional-mean MA order, always zero here.
        _const: Whether the mean carries an intercept.

    Args:
        endog: The series, typically returns or residuals.
        p: ARCH order.
        o: Asymmetry order; must be at least 1.
        q: GARCH order.
        ar_lags: Conditional-mean AR order.
        mean: ``"constant"`` or ``"zero"``.

    Raises:
        SpecificationError: If an order is not a non-negative integer,
            ``o`` is zero, or ``mean`` is not one of the two choices.
        DimensionError: If ``endog`` is not one-dimensional or is shorter
            than the orders require.
        NumericalError: If ``endog`` contains non-finite values.

    See Also:
        * :class:`GARCHResult` -- what :meth:`fit` returns, with
          ``has_leverage`` set.
        * :class:`GARCH` -- the symmetric special case.
        * :class:`EGARCH` -- asymmetry through the sign of the
          standardized shock in a log-variance recursion.
        * :class:`ARMAGJR` -- the same variance under an ARMA mean.

    References:
        Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the
        relation between the expected value and the volatility of the
        nominal excess return on stocks. *Journal of Finance*, 48(5),
        1779-1801.

    Example:
        A leverage effect recovered, and preferred by BIC over the
        symmetric fit:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 3000
        >>> h, e = np.ones(n), np.zeros(n)
        >>> for t in range(1, n):
        ...     h[t] = 0.05 + 0.03 * e[t - 1] ** 2 + 0.85 * h[t - 1]
        ...     h[t] += 0.10 * e[t - 1] ** 2 * (e[t - 1] < 0)
        ...     e[t] = np.sqrt(h[t]) * rng.standard_normal()
        >>> res = GJR(e).fit()
        >>> res.has_leverage, list(res.params)
        (True, ['const', 'omega', 'alpha[1]', 'gamma[1]', 'beta[1]'])
        >>> bool(abs(res.gamma[0] - 0.10) < 0.05)
        True
        >>> symmetric = GARCH(e).fit()
        >>> bool(res.information_criteria.bic < symmetric.information_criteria.bic)
        True
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        ar_lags: int = 0,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data.

        Args:
            endog: The series.
            p: ARCH order.
            o: Asymmetry order, at least 1.
            q: GARCH order.
            ar_lags: Conditional-mean AR order.
            mean: ``"constant"`` or ``"zero"``.

        Raises:
            SpecificationError: If an order or the mean choice is invalid,
                or ``o`` is zero.
            DimensionError: If the series has the wrong shape or is too
                short.
            NumericalError: If the series is not finite.

        Example:
            >>> import numpy as np
            >>> GJR(np.arange(50.0), o=2).order
            (1, 2, 1)
            >>> GJR(np.arange(50.0), o=0)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: GJR requires an asymmetry order o >= 1.
        """
        super().__init__(endog, vol="GJR", p=p, o=o, q=q, ar_lags=ar_lags, mean=mean)

    def fit(self) -> GARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood.

        Returns:
            The fitted :class:`GARCHResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = GJR(rng.standard_normal(400)).fit()
            >>> res.vol, res.order, res.n_params
            ('GJR', (1, 1, 1), 5)
        """
        return GARCHResult._from_fit(self._fit_family(), self)


class EGARCH(_ShortMemoryVarianceModel[GARCHResult]):
    r"""EGARCH(p, o, q): the recursion runs in logs, so no positivity constraint binds.

    The variance recursion

    .. math::

        \log\sigma_t^2 = \omega
        + \sum_{i=1}^{p} \alpha_i\bigl(|z_{t-i}| - \sqrt{2/\pi}\bigr)
        + \sum_{k=1}^{o} \gamma_k\, z_{t-k}
        + \sum_{j=1}^{q} \beta_j \log\sigma_{t-j}^2,
        \qquad z_t = \varepsilon_t / \sigma_t,

    driven by the standardized shock rather than the squared residual.
    Because the variance is exponentiated, :math:`\omega`, :math:`\alpha`
    and :math:`\gamma` are searched freely and may be negative; only the
    persistence block is bounded, :math:`|\sum_j \beta_j| < 0.999`, which
    keeps the log variance stationary. The magnitude term is centred by
    :math:`E|z| = \sqrt{2/\pi}` so a Gaussian shock contributes no drift;
    the sign term is the leverage effect, with :math:`\gamma < 0` meaning
    a negative shock raises volatility more than a positive one of the
    same size -- the opposite sign convention from :class:`GJR`. The
    reported parameters are on the log scale, which is why
    :attr:`GARCHResult.unconditional_variance` declines to translate them
    back: the stationary level is the mean of a lognormal and depends on
    the whole innovation distribution. The asymmetry order must be at
    least one.

    Attributes:
        _vol: The family name, ``"EGARCH"``.
        _p: Magnitude order.
        _o: Sign order.
        _q: Log-variance persistence order.
        _endog: The validated series.
        _ar_lags: Conditional-mean AR order.
        _ma_lags: Conditional-mean MA order, always zero here.
        _const: Whether the mean carries an intercept.

    Args:
        endog: The series, typically returns or residuals.
        p: Magnitude order.
        o: Sign (leverage) order; must be at least 1.
        q: Log-variance persistence order.
        ar_lags: Conditional-mean AR order.
        mean: ``"constant"`` or ``"zero"``.

    Raises:
        SpecificationError: If an order is not a non-negative integer,
            ``o`` is zero, or ``mean`` is not one of the two choices.
        DimensionError: If ``endog`` is not one-dimensional or is shorter
            than the orders require.
        NumericalError: If ``endog`` contains non-finite values.

    See Also:
        * :class:`GARCHResult` -- what :meth:`fit` returns; its
          ``persistence`` is :math:`\sum_j \beta_j` for this family.
        * :class:`GJR` -- leverage in a level recursion, with a closed-form
          unconditional variance.
        * :class:`ARMAEGARCH` -- the same variance under an ARMA mean.

    References:
        Nelson, D. B. (1991). Conditional heteroskedasticity in asset
        returns: A new approach. *Econometrica*, 59(2), 347-370.

    Example:
        A leverage effect in logs recovered with its sign:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 3000
        >>> lh, e, z = np.zeros(n), np.zeros(n), rng.standard_normal(n)
        >>> for t in range(1, n):
        ...     lh[t] = -0.1 + 0.15 * (abs(z[t - 1]) - np.sqrt(2 / np.pi))
        ...     lh[t] += -0.08 * z[t - 1] + 0.95 * lh[t - 1]
        ...     e[t] = np.exp(lh[t] / 2) * z[t]
        >>> res = EGARCH(e).fit()
        >>> res.vol, res.order, res.has_leverage
        ('EGARCH', (1, 1, 1), True)
        >>> bool(res.gamma[0] < 0), bool(abs(res.beta[0] - 0.95) < 0.05)
        (True, True)
        >>> res.unconditional_variance is None, bool(res.half_life > 0)
        (True, True)
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        ar_lags: int = 0,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data.

        Args:
            endog: The series.
            p: Magnitude order.
            o: Sign order, at least 1.
            q: Log-variance persistence order.
            ar_lags: Conditional-mean AR order.
            mean: ``"constant"`` or ``"zero"``.

        Raises:
            SpecificationError: If an order or the mean choice is invalid,
                or ``o`` is zero.
            DimensionError: If the series has the wrong shape or is too
                short.
            NumericalError: If the series is not finite.

        Example:
            >>> import numpy as np
            >>> EGARCH(np.arange(50.0), p=2, o=1, q=1).order
            (2, 1, 1)
            >>> EGARCH(np.arange(50.0), o=0)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: EGARCH requires an asymmetry order o >= 1.
        """
        super().__init__(endog, vol="EGARCH", p=p, o=o, q=q, ar_lags=ar_lags, mean=mean)

    def fit(self) -> GARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood.

        Returns:
            The fitted :class:`GARCHResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = EGARCH(rng.standard_normal(400)).fit()
            >>> res.vol, res.n_params, res.unconditional_variance
            ('EGARCH', 5, None)
        """
        return GARCHResult._from_fit(self._fit_family(), self)


class ARGARCH(_ShortMemoryVarianceModel[ARMAGARCHResult]):
    r"""Autoregressive mean with a symmetric GARCH variance, estimated jointly.

    The same estimator as ``GARCH(y, ar_lags=...)``, under the name a reader
    expects when the mean is the point of the specification rather than an
    afterthought, and returning :class:`ARMAGARCHResult` so the mean's
    stationarity verdict is on the record. The mean is a regression on
    lagged levels, so its weights are unconstrained and
    :attr:`ARMAGARCHResult.is_stationary` is informative; the two fronts
    reach the same likelihood to the last digit.

    Attributes:
        _vol: The family name, ``"GARCH"``.
        _p: ARCH order.
        _o: Asymmetry order, always zero here.
        _q: GARCH order.
        _endog: The validated series.
        _ar_lags: Conditional-mean AR order.
        _ma_lags: Conditional-mean MA order, always zero here.
        _const: Whether the mean carries an intercept.

    Args:
        endog: The series.
        ar_lags: Conditional-mean autoregressive order.
        p: Order of the shock-magnitude block.
        q: Order of the variance persistence block.
        mean: ``"constant"`` or ``"zero"``.

    Raises:
        SpecificationError: If an order is not a non-negative integer or
            ``mean`` is not one of the two choices.
        DimensionError: If ``endog`` is not one-dimensional or is shorter
            than the orders require.
        NumericalError: If ``endog`` contains non-finite values.

    See Also:
        * :class:`ARMAGARCHResult` -- what :meth:`fit` returns.
        * :class:`GARCH` -- the same fit returned as a
          :class:`GARCHResult`, without the mean verdicts.
        * :class:`ARMAGARCH` -- with a moving-average block, which
          switches the mean onto the constrained recursion.
        * :class:`~cultivars.univariate.autoregression.AR` -- the same
          mean with a constant variance.

    Example:
        The two fronts agree on the likelihood and differ in what they
        report:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = rng.standard_normal(500)
        >>> named, plain = ARGARCH(y, ar_lags=2).fit(), GARCH(y, ar_lags=2).fit()
        >>> round(named.llf - plain.llf, 8)
        0.0
        >>> named._comparison_label(), plain._comparison_label()
        ('AR(2)-GARCH(1, 1)', 'GARCH(1, 1)')
        >>> named.is_stationary, named.is_structurally_constrained
        (True, False)
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        ar_lags: int = 1,
        p: int = 1,
        q: int = 1,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data.

        Args:
            endog: The series.
            ar_lags: Conditional-mean AR order.
            p: ARCH order.
            q: GARCH order.
            mean: ``"constant"`` or ``"zero"``.

        Raises:
            SpecificationError: If an order or the mean choice is invalid.
            DimensionError: If the series has the wrong shape or is too
                short.
            NumericalError: If the series is not finite.

        Example:
            >>> import numpy as np
            >>> model = ARGARCH(np.arange(50.0), ar_lags=3)
            >>> model.mean_order, model.order
            ((3, 0), (1, 0, 1))
        """
        super().__init__(endog, vol="GARCH", p=p, o=0, q=q, ar_lags=ar_lags, ma_lags=0, mean=mean)

    def fit(self) -> ARMAGARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood.

        Returns:
            The fitted :class:`ARMAGARCHResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = ARGARCH(rng.standard_normal(400), ar_lags=1).fit()
            >>> list(res.params), res.ma_params.shape
            (['const', 'ar.L1', 'omega', 'alpha[1]', 'beta[1]'], (0,))
        """
        return ARMAGARCHResult._from_fit(self._fit_family(), self)


class ARMAGARCH(_ShortMemoryVarianceModel[ARMAGARCHResult]):
    r"""ARMA mean with a symmetric GARCH variance, estimated jointly.

    The mean

    .. math::

        y_t = c + \sum_{i=1}^{p} \phi_i\, y_{t-i} + \varepsilon_t
        + \sum_{j=1}^{q} \theta_j\, \varepsilon_{t-j},
        \qquad \varepsilon_t = \sigma_t z_t,

    over the :class:`GARCH` variance, with one likelihood for both. When
    ``ma_lags > 0`` the mean residual feeds on its own past and the mean
    is searched through the partial autocorrelations, so the fit is
    stationary and invertible by construction; when ``ma_lags == 0`` the
    specification collapses to :class:`ARGARCH` and the lag weights are
    unconstrained. Both defaults are one, so ``ARMAGARCH(y, ar_lags=2)``
    keeps a moving-average term.

    Attributes:
        _vol: The family name, ``"GARCH"``.
        _p: ARCH order.
        _o: Asymmetry order, always zero here.
        _q: GARCH order.
        _endog: The validated series.
        _ar_lags: Conditional-mean AR order.
        _ma_lags: Conditional-mean MA order.
        _const: Whether the mean carries an intercept.

    Args:
        endog: The series.
        ar_lags: Conditional-mean autoregressive order.
        ma_lags: Conditional-mean moving-average order.
        p: Order of the shock-magnitude block.
        q: Order of the variance persistence block.
        mean: ``"constant"`` or ``"zero"``.

    Raises:
        SpecificationError: If an order is not a non-negative integer or
            ``mean`` is not one of the two choices.
        DimensionError: If ``endog`` is not one-dimensional or is shorter
            than the orders require.
        NumericalError: If ``endog`` contains non-finite values.

    See Also:
        * :class:`ARMAGARCHResult` -- what :meth:`fit` returns.
        * :class:`ARMAGJR`, :class:`ARMAEGARCH` -- the same mean over the
          asymmetric variances.
        * :class:`ARMAFIGARCH` -- the same mean over a long-memory
          variance.
        * :class:`~cultivars.univariate.box_jenkins.ARMA` -- the same mean
          with a constant variance and the exact likelihood.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 2000
        >>> s2 = np.ones(n)
        >>> e = np.zeros(n)
        >>> y = np.zeros(n)
        >>> for t in range(1, n):
        ...     s2[t] = 0.05 + 0.10 * e[t - 1] ** 2 + 0.85 * s2[t - 1]
        ...     e[t] = np.sqrt(s2[t]) * rng.standard_normal()
        ...     y[t] = 0.5 * y[t - 1] + e[t] + 0.4 * e[t - 1]
        >>> res = ARMAGARCH(y, ar_lags=1, ma_lags=1).fit()
        >>> bool(res.is_stationary and res.is_invertible)
        True
        >>> bool(abs(res.ar_params[0] - 0.5) < 0.1), bool(abs(res.ma_params[0] - 0.4) < 0.1)
        (True, True)
        >>> res._comparison_label()
        'ARMA(1, 1)-GARCH(1, 1)'
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        ar_lags: int = 1,
        ma_lags: int = 1,
        p: int = 1,
        q: int = 1,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data.

        Args:
            endog: The series.
            ar_lags: Conditional-mean AR order.
            ma_lags: Conditional-mean MA order.
            p: ARCH order.
            q: GARCH order.
            mean: ``"constant"`` or ``"zero"``.

        Raises:
            SpecificationError: If an order or the mean choice is invalid.
            DimensionError: If the series has the wrong shape or is too
                short.
            NumericalError: If the series is not finite.

        Example:
            >>> import numpy as np
            >>> model = ARMAGARCH(np.arange(50.0), ar_lags=2, ma_lags=1, p=1, q=2)
            >>> model.mean_order, model.order
            ((2, 1), (1, 0, 2))
        """
        super().__init__(
            endog, vol="GARCH", p=p, o=0, q=q, ar_lags=ar_lags, ma_lags=ma_lags, mean=mean
        )

    def fit(self) -> ARMAGARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood.

        Returns:
            The fitted :class:`ARMAGARCHResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = ARMAGARCH(rng.standard_normal(400)).fit()
            >>> res.mean_order, res.n_params, res.nobs
            ((1, 1), 6, 399)
        """
        return ARMAGARCHResult._from_fit(self._fit_family(), self)


class ARMAGJR(_ShortMemoryVarianceModel[ARMAGARCHResult]):
    r"""ARMA mean with a sign-asymmetric GJR variance, estimated jointly.

    The :class:`ARMAGARCH` mean over the :class:`GJR` variance, so a
    negative mean residual loads an extra :math:`\gamma` on next period's
    variance. The result's ``has_leverage`` is ``True`` by construction
    here, since the asymmetry order must be at least one; the sign and
    size of ``gamma[1]`` are what say whether leverage was found.

    Attributes:
        _vol: The family name, ``"GJR"``.
        _p: ARCH order.
        _o: Asymmetry order.
        _q: GARCH order.
        _endog: The validated series.
        _ar_lags: Conditional-mean AR order.
        _ma_lags: Conditional-mean MA order.
        _const: Whether the mean carries an intercept.

    Args:
        endog: The series.
        ar_lags: Conditional-mean autoregressive order.
        ma_lags: Conditional-mean moving-average order.
        p: Order of the shock-magnitude block.
        o: Order of the asymmetry block, at least one.
        q: Order of the variance persistence block.
        mean: ``"constant"`` or ``"zero"``.

    Raises:
        SpecificationError: If an order is not a non-negative integer,
            ``o`` is zero, or ``mean`` is not one of the two choices.
        DimensionError: If ``endog`` is not one-dimensional or is shorter
            than the orders require.
        NumericalError: If ``endog`` contains non-finite values.

    See Also:
        * :class:`ARMAGARCHResult` -- what :meth:`fit` returns.
        * :class:`GJR` -- the same variance under a constant or AR mean.
        * :class:`ARMAGARCH` -- the symmetric special case.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> res = ARMAGJR(rng.standard_normal(500)).fit()
        >>> res._comparison_label(), res.has_leverage
        ('ARMA(1, 1)-GJR(1, 1, 1)', True)
        >>> list(res.params)
        ['const', 'ar.L1', 'ma.L1', 'omega', 'alpha[1]', 'gamma[1]', 'beta[1]']
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        ar_lags: int = 1,
        ma_lags: int = 1,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data.

        Args:
            endog: The series.
            ar_lags: Conditional-mean AR order.
            ma_lags: Conditional-mean MA order.
            p: ARCH order.
            o: Asymmetry order, at least 1.
            q: GARCH order.
            mean: ``"constant"`` or ``"zero"``.

        Raises:
            SpecificationError: If an order or the mean choice is invalid,
                or ``o`` is zero.
            DimensionError: If the series has the wrong shape or is too
                short.
            NumericalError: If the series is not finite.

        Example:
            >>> import numpy as np
            >>> ARMAGJR(np.arange(50.0), ar_lags=2, ma_lags=0, o=2).order
            (1, 2, 1)
            >>> ARMAGJR(np.arange(50.0), o=0)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: GJR requires an asymmetry order o >= 1.
        """
        super().__init__(
            endog, vol="GJR", p=p, o=o, q=q, ar_lags=ar_lags, ma_lags=ma_lags, mean=mean
        )

    def fit(self) -> ARMAGARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood.

        Returns:
            The fitted :class:`ARMAGARCHResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = ARMAGJR(rng.standard_normal(400)).fit()
            >>> res.vol, res.order, res.mean_order, res.n_params
            ('GJR', (1, 1, 1), (1, 1), 7)
        """
        return ARMAGARCHResult._from_fit(self._fit_family(), self)


class ARMAEGARCH(_ShortMemoryVarianceModel[ARMAGARCHResult]):
    r"""ARMA mean with a log-variance EGARCH process, estimated jointly.

    The :class:`ARMAGARCH` mean over the :class:`EGARCH` variance. The
    standardized mean residual :math:`z_t = \varepsilon_t / \sigma_t`
    drives the log variance through its magnitude and its sign, so the
    mean and variance blocks are coupled more tightly than in the level
    families: a misspecified mean changes the shocks the variance sees
    in both size and sign. The result's ``unconditional_variance`` is
    ``None`` for this family.

    Attributes:
        _vol: The family name, ``"EGARCH"``.
        _p: Magnitude order.
        _o: Sign order.
        _q: Log-variance persistence order.
        _endog: The validated series.
        _ar_lags: Conditional-mean AR order.
        _ma_lags: Conditional-mean MA order.
        _const: Whether the mean carries an intercept.

    Args:
        endog: The series.
        ar_lags: Conditional-mean autoregressive order.
        ma_lags: Conditional-mean moving-average order.
        p: Order of the shock-magnitude block.
        o: Order of the asymmetry block, at least one.
        q: Order of the variance persistence block.
        mean: ``"constant"`` or ``"zero"``.

    Raises:
        SpecificationError: If an order is not a non-negative integer,
            ``o`` is zero, or ``mean`` is not one of the two choices.
        DimensionError: If ``endog`` is not one-dimensional or is shorter
            than the orders require.
        NumericalError: If ``endog`` contains non-finite values.

    See Also:
        * :class:`ARMAGARCHResult` -- what :meth:`fit` returns.
        * :class:`EGARCH` -- the same variance under a constant or AR
          mean, with the recursion written out.
        * :class:`ARMAGJR` -- asymmetry in a level recursion.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> res = ARMAEGARCH(rng.standard_normal(500)).fit()
        >>> res._comparison_label(), res.unconditional_variance
        ('ARMA(1, 1)-EGARCH(1, 1, 1)', None)
        >>> res.is_stationary, res.is_invertible, res.is_covariance_stationary
        (True, True, True)
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        ar_lags: int = 1,
        ma_lags: int = 1,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data.

        Args:
            endog: The series.
            ar_lags: Conditional-mean AR order.
            ma_lags: Conditional-mean MA order.
            p: Magnitude order.
            o: Sign order, at least 1.
            q: Log-variance persistence order.
            mean: ``"constant"`` or ``"zero"``.

        Raises:
            SpecificationError: If an order or the mean choice is invalid,
                or ``o`` is zero.
            DimensionError: If the series has the wrong shape or is too
                short.
            NumericalError: If the series is not finite.

        Example:
            >>> import numpy as np
            >>> model = ARMAEGARCH(np.arange(50.0), ar_lags=1, ma_lags=0)
            >>> model.vol, model.mean_order, model.order
            ('EGARCH', (1, 0), (1, 1, 1))
        """
        super().__init__(
            endog, vol="EGARCH", p=p, o=o, q=q, ar_lags=ar_lags, ma_lags=ma_lags, mean=mean
        )

    def fit(self) -> ARMAGARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood.

        Returns:
            The fitted :class:`ARMAGARCHResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = ARMAEGARCH(rng.standard_normal(400)).fit()
            >>> res.vol, res.n_params, res.nobs
            ('EGARCH', 7, 399)
        """
        return ARMAGARCHResult._from_fit(self._fit_family(), self)


class FIGARCH(_FractionalVarianceModel[FIGARCHResult]):
    r"""FIGARCH(1, d, 1): long-memory volatility through a fractional filter.

    The variance

    .. math::

        \sigma_t^2 = \frac{\omega}{1 - \beta}
        + \Bigl[1 - \frac{(1 - \phi L)(1 - L)^d}{1 - \beta L}\Bigr]\varepsilon_t^2,

    under a constant or zero mean. The order is fixed, so the only
    structural choice beyond the mean is how far the infinite-order
    representation is truncated. Weights beyond the available history are
    applied to the pre-sample variance rather than dropped, so the
    truncation tail contributes a constant instead of vanishing silently;
    at the default of 1000 lags the likelihood is insensitive to the
    choice for samples of a few hundred observations. Draws whose
    retained weights turn negative are rejected, which is the positivity
    condition on the ARCH(:math:`\infty`) representation. A mean with
    lags is :class:`ARMAFIGARCH`; this front does not accept one.

    Attributes:
        _truncation: The retained number of weights.
        _endog: The validated series.
        _ar_lags: Conditional-mean AR order, always zero here.
        _ma_lags: Conditional-mean MA order, always zero here.
        _const: Whether the mean carries an intercept.

    Args:
        endog: The series, typically returns or residuals.
        mean: ``"constant"`` or ``"zero"``.
        truncation: Infinite-order truncation lag, at least 1.

    Raises:
        SpecificationError: If ``truncation`` is not a positive integer or
            ``mean`` is not one of the two choices.
        DimensionError: If ``endog`` is not one-dimensional or has fewer
            than five observations.
        NumericalError: If ``endog`` contains non-finite values.

    See Also:
        * :class:`FIGARCHResult` -- what :meth:`fit` returns.
        * :class:`ARMAFIGARCH` -- the same variance under an ARMA mean.
        * :class:`GARCH` -- the geometric-decay alternative, whose
          persistence pinned near one is the usual sign this model is
          wanted.
        * :class:`~cultivars.univariate.fractional_integration.ARFIMA` --
          long memory in the level rather than the variance.

    References:
        Baillie, R. T., Bollerslev, T., & Mikkelsen, H. O. (1996).
        Fractionally integrated generalized autoregressive conditional
        heteroskedasticity. *Journal of Econometrics*, 74(1), 3-30.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = rng.standard_normal(600)
        >>> res = FIGARCH(y, truncation=200).fit()
        >>> res.truncation, list(res.params)
        (200, ['const', 'omega', 'phi', 'd', 'beta'])
        >>> bool(abs(res.llf - FIGARCH(y).fit().llf) < 2.0)
        True
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        mean: str = "constant",
        truncation: int = _DEFAULT_TRUNCATION,
    ) -> None:
        """Validate the specification and the data.

        Args:
            endog: The series.
            mean: ``"constant"`` or ``"zero"``.
            truncation: Infinite-order truncation lag.

        Raises:
            SpecificationError: If ``truncation`` or the mean choice is
                invalid.
            DimensionError: If the series has the wrong shape or is too
                short.
            NumericalError: If the series is not finite.

        Example:
            >>> import numpy as np
            >>> model = FIGARCH(np.arange(50.0), truncation=20)
            >>> model.truncation, model.mean_order, model.has_constant_mean
            (20, (0, 0), True)
            >>> FIGARCH(np.arange(50.0), ar_lags=1)
            Traceback (most recent call last):
                ...
            TypeError: FIGARCH.__init__() got an unexpected keyword argument 'ar_lags'
        """
        super().__init__(endog, mean=mean, ar_lags=0, ma_lags=0, truncation=truncation)

    def fit(self) -> FIGARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood.

        Returns:
            The fitted :class:`FIGARCHResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = FIGARCH(rng.standard_normal(600)).fit()
            >>> res.persistence, res.is_covariance_stationary, bool(0 < res.d < 1)
            (1.0, False, True)
        """
        return FIGARCHResult._from_fit(self._fit_family(), self)


class ARMAFIGARCH(_FractionalVarianceModel[ARMAFIGARCHResult]):
    r"""ARMA mean with a fractionally integrated variance, estimated jointly.

    The combination the constant-mean :class:`FIGARCH` cannot express:
    short-memory dynamics in the level alongside long-memory dynamics in
    the volatility, which is the usual empirical picture for a return
    series sampled finely enough to show mean reversion. The mean is the
    :class:`ARMAGARCH` one, searched through the partial autocorrelations
    when ``ma_lags > 0`` and as an unconstrained lag regression
    otherwise; the variance is the :class:`FIGARCH` filter. Both mean
    defaults are one, so ``ARMAFIGARCH(y, ar_lags=2)`` keeps a
    moving-average term.

    Attributes:
        _truncation: The retained number of weights.
        _endog: The validated series.
        _ar_lags: Conditional-mean AR order.
        _ma_lags: Conditional-mean MA order.
        _const: Whether the mean carries an intercept.

    Args:
        endog: The series.
        ar_lags: Conditional-mean autoregressive order.
        ma_lags: Conditional-mean moving-average order.
        mean: ``"constant"`` or ``"zero"``.
        truncation: Infinite-order truncation lag for the fractional filter.

    Raises:
        SpecificationError: If an order or ``truncation`` is not a
            non-negative integer (``truncation`` at least 1), or ``mean``
            is not one of the two choices.
        DimensionError: If ``endog`` is not one-dimensional or is shorter
            than the mean orders plus four.
        NumericalError: If ``endog`` contains non-finite values.

    See Also:
        * :class:`ARMAFIGARCHResult` -- what :meth:`fit` returns.
        * :class:`FIGARCH` -- the same variance under a constant or zero
          mean, with the filter written out.
        * :class:`ARMAGARCH` -- the same mean over a geometric-decay
          variance.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(600)
        >>> e = rng.standard_normal(600)
        >>> for t in range(1, 600):
        ...     y[t] = 0.7 * y[t - 1] + e[t]
        >>> res = ARMAFIGARCH(y, ar_lags=1, ma_lags=0).fit()
        >>> res._comparison_label(), bool(abs(res.ar_params[0] - 0.7) < 0.1)
        ('AR(1)-FIGARCH(1, d, 1)', True)
        >>> res.is_stationary, res.is_covariance_stationary
        (True, False)
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        ar_lags: int = 1,
        ma_lags: int = 1,
        mean: str = "constant",
        truncation: int = 1000,
    ) -> None:
        """Validate the specification and the data.

        Args:
            endog: The series.
            ar_lags: Conditional-mean AR order.
            ma_lags: Conditional-mean MA order.
            mean: ``"constant"`` or ``"zero"``.
            truncation: Infinite-order truncation lag.

        Raises:
            SpecificationError: If an order, ``truncation`` or the mean
                choice is invalid.
            DimensionError: If the series has the wrong shape or is too
                short.
            NumericalError: If the series is not finite.

        Example:
            >>> import numpy as np
            >>> model = ARMAFIGARCH(np.arange(50.0), ar_lags=2, ma_lags=1, truncation=30)
            >>> model.mean_order, model.truncation
            ((2, 1), 30)
        """
        super().__init__(endog, mean=mean, ar_lags=ar_lags, ma_lags=ma_lags, truncation=truncation)

    def fit(self) -> ARMAFIGARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood.

        Returns:
            The fitted :class:`ARMAFIGARCHResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = ARMAFIGARCH(rng.standard_normal(600)).fit()
            >>> res.mean_order, res.nobs, list(res.params)[:3]
            ((1, 1), 599, ['const', 'ar.L1', 'ma.L1'])
        """
        return ARMAFIGARCHResult._from_fit(self._fit_family(), self)
