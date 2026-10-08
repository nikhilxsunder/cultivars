# filepath: /src/cultivars/univariate/ar.py
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
r"""Autoregressive model AR(p) -- the public specification and result.

The model is

.. math::

   y_t = c + \delta t + \sum_{i=1}^{p} \phi_i\, y_{t-i} + \varepsilon_t,
   \qquad \varepsilon_t \sim N(0, \sigma^2),

held under the package's three-object discipline: :class:`AR` is the
immutable specification, estimation belongs to the family base it
inherits, and :class:`ARResult` is the frozen result the user actually
handles. Two estimators are reachable through ``method``. ``"css"``
conditions on the first :math:`p` observations and solves the lagged
regression by ordinary least squares, so the likelihood is the
conditional one on :math:`n - p` observations. ``"exact"`` maximizes
the exact Gaussian likelihood,

.. math::

   \log L = \log p(y_1, \dots, y_p) + \sum_{t=p+1}^{n} \log p(y_t \mid y_{t-1}, \dots, y_{t-p}),

through the companion state-space embedding, with the search kept
inside the stationary region by the partial-autocorrelation
reparameterization of Monahan (1984): the optimizer moves in
unconstrained :math:`\mathbb R^p`, each coordinate maps to a partial
autocorrelation in :math:`(-1, 1)`, and the Durbin-Levinson recursion
turns those into coefficients whose companion roots lie inside the unit
circle by construction. The two estimators differ in finite samples by
the stationary density of the first :math:`p` observations and converge
as the sample grows.

Two commitments shape the surface. First, the effective sample is
reported, not hidden: ``nobs`` is :math:`n - p` under CSS and :math:`n`
under exact ML, so the information criteria are on different scales
across the two, and :meth:`ARResult.compare` refuses to rank results
with differing ``nobs`` rather than silently order incomparable
numbers -- to select an order, fit every candidate with ``"exact"``.
Second, the result carries what a downstream consumer needs and nothing
about how it was found: the series, the fitted and residual paths
aligned to it, the coefficients by display name, and the companion
matrix's verdict on stationarity; ``simulate`` draws from the fitted
law with the deterministic terms handled as regressors inside the
recursion, and refuses a fit whose coefficients sum to one.

Layout. :class:`AR` validates ``order``, ``trend`` and ``method`` at
construction through ``validate_order`` and ``validate_choice`` against
the :data:`~cultivars.typing.Trend` and :data:`~cultivars.typing.Method`
aliases, and ``fit`` dispatches through ``_fit_family`` to ``_fit_css``
or ``_fit_exact`` on the ``_AutoRegressionModel`` base in
``_internals``; the exact path builds an ``_AutoRegressionObjective``
and hands it to ``_maximize_likelihood``, with ``pacf_to_coeffs`` and
``coeffs_to_pacf`` in ``_core`` doing the reparameterization. The
packed ``_AutoRegressionFit`` record is assembled into
:class:`ARResult` by ``_from_fit``, and ``_simulate_arma`` with the
``_SIMULATION_BURN`` default backs ``simulate``. The four mixins the
result composes -- ``_SummaryMixin``, ``_SeriesMixin``,
``_ComparisonMixin``, ``_StationarityMixin`` -- live in ``_internals``
and are shared with every other univariate result. The
moving-average and integrated generalization is
:mod:`~cultivars.univariate.box_jenkins`; the long-memory one
:mod:`~cultivars.univariate.fractional_integration`.

References:
    Hamilton, J. D. (1994). *Time Series Analysis*, chapters 5 and 13.
    Princeton University Press.

    Monahan, J. F. (1984). A note on enforcing stationarity in ARMA
    models. *Biometrika*, 71(2), 403-404.

    Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M.
    (2015). *Time Series Analysis: Forecasting and Control*, 5th ed.
    Wiley.

Example:
    Order selection the way the module intends it -- every candidate
    fitted by exact ML so the criteria share a sample -- picks the true
    order of a simulated AR(2):

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> e = rng.standard_normal(600)
    >>> z = np.zeros(600)
    >>> for t in range(2, 600):
    ...     z[t] = 0.5 * z[t - 1] - 0.3 * z[t - 2] + e[t]
    >>> fits = {p: AR(z, order=p, method="exact").fit() for p in range(1, 5)}
    >>> table = fits[1].compare(fits[2], fits[3], fits[4], criterion="bic")
    >>> table.columns
    ('model', 'llf', 'k', 'BIC', 'dBIC')
    >>> [row[0] for row in table.rows]
    ['AR(2)', 'AR(3)', 'AR(4)', 'AR(1)']
    >>> bool(np.allclose(fits[2].ar_params, [0.5, -0.3], atol=0.05)), fits[2].is_stationary
    (True, True)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import _SIMULATION_BURN, InformationCriteria, SummaryTable
from .._internals import (
    _AutoRegressionFit,
    _AutoRegressionModel,
    _ComparisonMixin,
    _SeriesMixin,
    _simulate_arma,
    _StationarityMixin,
    _SummaryMixin,
)
from ..exceptions import SpecificationError

__all__ = ["AR", "ARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ARResult(_SummaryMixin, _SeriesMixin, _ComparisonMixin, _StationarityMixin):
    r"""A fitted autoregression.

    The estimates of

    .. math::

        y_t = c + \delta t + \sum_{i=1}^{p} \phi_i\, y_{t-i} + \varepsilon_t,
        \qquad \varepsilon_t \sim N(0, \sigma^2),

    together with the one-step fitted values and residuals on the
    effective sample, the maximized log-likelihood, and the counts that
    turn the likelihood into information criteria. Composes four
    capabilities: rendering (``summary``, and the summary is what a bare
    result prints), frame interop (``to_pandas``, ``to_polars``,
    ``to_dict``), criterion-named comparison (``information_criteria``,
    ``compare``, ``likelihood_ratio_test``), and stationarity assessment
    (``stability``, ``is_stationary`` from the companion matrix of the
    autoregressive coefficients). ``repr=False`` is required -- see
    :class:`~cultivars._internals._SummaryMixin` -- so the summary is
    what a bare result prints.

    Attributes:
        endog: The full observed series, retained so residual and fitted paths
            can be aligned against it.
        fittedvalues: One-step fitted values over the effective sample.
        resid: One-step residuals over the effective sample.
        llf: Maximized log-likelihood, conditional for CSS and exact otherwise.
        nobs: Observations the likelihood was evaluated on.
        n_params: Free parameter count, including the innovation variance.
        order: Autoregressive order ``p``.
        trend: Deterministic specification.
        method: Estimator that produced the fit.
        const: Intercept, or ``None`` when ``trend == "n"``.
        trend_coeff: Linear-trend slope, or ``None`` unless ``trend == "ct"``.
        ar_params: Autoregressive coefficients.
        sigma2: Innovation variance.

    Note:
        Under ``"css"`` the effective sample is the last :math:`n - p`
        observations and ``llf`` is the conditional likelihood; under
        ``"exact"`` it is all :math:`n` and ``llf`` includes the density
        of the first :math:`p` observations under stationarity. The two
        are not on the same scale, and neither are their information
        criteria, so ``compare`` refuses results with differing
        ``nobs`` rather than rank them: to compare orders, fit every
        candidate with ``method="exact"``. The unconditional mean and
        variance the estimates imply are :math:`c / (1 - \sum \phi_i)`
        and, for :math:`p = 1`, :math:`\sigma^2 / (1 - \phi_1^2)`; a fit
        whose coefficients sum to one has neither, and
        :meth:`simulate` says so rather than returning a divergent path.
        Standard errors are not yet reported by this estimator.

    See Also:
        * :class:`AR` -- the specification that produces this.
        * :class:`~cultivars.univariate.box_jenkins.ARIMA` -- the
          moving-average and integrated generalization, with the same
          result surface.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
          -- the vector counterpart.

    References:
        Hamilton, J. D. (1994). *Time Series Analysis*, chapters 5 and
        13. Princeton University Press.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(500)
        >>> for t in range(1, 500):
        ...     y[t] = 1.0 + 0.5 * y[t - 1] + rng.standard_normal()
        >>> res = AR(y, order=1, trend="c").fit()
        >>> res.nobs, res.n_params, list(res.params)
        (499, 3, ['const', 'ar.L1', 'sigma2'])
        >>> round(float(res.const), 2), res.ar_params.round(2).tolist(), round(res.sigma2, 2)
        (1.02, [0.48], 1.03)

        CSS is ordinary least squares on the lagged regression, so the
        estimates match ``lstsq`` exactly and the residuals are the
        series minus its fitted values:

        >>> design = np.column_stack([np.ones(499), y[:-1]])
        >>> beta = np.linalg.lstsq(design, y[1:], rcond=None)[0]
        >>> bool(np.allclose(beta, [res.const, res.ar_params[0]]))
        True
        >>> bool(np.allclose(res.endog[1:] - res.fittedvalues, res.resid))
        True
        >>> res.is_stationary, round(res.stability.max_modulus, 2)
        (True, 0.48)
    """

    endog: npt.NDArray[np.float64]
    """The full observed series."""

    fittedvalues: npt.NDArray[np.float64]
    """One-step fitted values over the effective sample."""

    resid: npt.NDArray[np.float64]
    """One-step residuals over the effective sample."""

    llf: float
    """Maximized log-likelihood; conditional under CSS, exact otherwise."""

    nobs: int
    """Observations the likelihood was evaluated on."""

    n_params: float
    """Free parameter count, including the innovation variance."""

    order: int
    """Autoregressive order ``p``."""

    trend: str
    """``"n"``, ``"c"`` or ``"ct"``."""

    method: str
    """``"css"`` or ``"exact"``."""

    const: float | None
    """Intercept, or ``None`` when ``trend == "n"``."""

    trend_coeff: float | None
    """Linear-trend slope, or ``None`` unless ``trend == "ct"``."""

    ar_params: npt.NDArray[np.float64]
    """``(p,)`` autoregressive coefficients."""

    sigma2: float
    """Innovation variance."""

    @classmethod
    def _from_fit(cls, fit: _AutoRegressionFit, model: _AutoRegressionModel[ARResult]) -> ARResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The raw estimator output.
            model: The specification that produced it, read for the fields the
                fit record deliberately does not carry.

        Returns:
            The assembled result.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> model = AR(rng.standard_normal(100), order=2)
            >>> res = ARResult._from_fit(model._fit_family(), model)
            >>> res.order, res.trend, res.method, res.endog.shape
            (2, 'c', 'css', (100,))
        """
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            order=model.order,
            trend=model.trend,
            method=model.method,
            const=fit.const,
            trend_coeff=fit.trend_coeff,
            ar_params=fit.ar_params,
            sigma2=fit.sigma2,
        )

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order.

        ``const`` and ``trend`` appear only when the specification
        includes them; the autoregressive coefficients are ``ar.L1`` ..
        ``ar.Lp``; ``sigma2`` is last.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(200)
            >>> list(AR(y, order=2, trend="ct").fit().params)
            ['const', 'trend', 'ar.L1', 'ar.L2', 'sigma2']
            >>> list(AR(y, order=1, trend="n").fit().params)
            ['ar.L1', 'sigma2']
        """
        out: dict[str, float] = {}
        if self.const is not None:
            out["const"] = self.const
        if self.trend_coeff is not None:
            out["trend"] = self.trend_coeff
        for i, value in enumerate(self.ar_params, start=1):
            out[f"ar.L{i}"] = float(value)
        out["sigma2"] = self.sigma2
        return out

    def simulate(
        self,
        n: int = 200,
        *,
        seed: int | np.random.Generator | None = None,
        burn: int = _SIMULATION_BURN,
    ) -> npt.NDArray[np.float64]:
        r"""A fresh sample path at the fitted parameters.

        ``burn + n`` periods of the zero-mean autoregression are generated
        from a zero start with Gaussian innovations of variance ``sigma2``
        and the first ``burn`` discarded. The constant and trend are
        regressors *inside* the recursion,
        :math:`y_t = c + b t + \sum_i \phi_i y_{t-i} + \varepsilon_t`,
        so the mean path they imply is what is added,

        .. math::

            A + B t,\qquad
            B = \frac{b}{1 - \sum_i \phi_i},\qquad
            A = \frac{c - B \sum_i i\,\phi_i}{1 - \sum_i \phi_i},

        with the time index running ``1 .. n``.

        Args:
            n: Observations kept, at least 1.
            seed: Seed or generator.
            burn: Periods discarded from the start, non-negative.

        Returns:
            An array of shape ``(n,)``.

        Raises:
            SpecificationError: If the counts are not usable, or the
                autoregressive coefficients sum to one so that no mean
                path exists.

        Example:
            The simulated mean and variance are the fitted model's
            unconditional moments:

            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros(500)
            >>> for t in range(1, 500):
            ...     y[t] = 1.0 + 0.5 * y[t - 1] + rng.standard_normal()
            >>> res = AR(y, order=1, trend="c").fit()
            >>> path = res.simulate(20000, seed=0)
            >>> mean = res.const / (1 - res.ar_params[0])
            >>> variance = res.sigma2 / (1 - res.ar_params[0] ** 2)
            >>> bool(abs(path.mean() - mean) < 0.05), bool(abs(path.var() - variance) < 0.05)
            (True, True)
            >>> res.simulate(0)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: n must be at least 1; got 0.
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        time = np.arange(1, n + 1, dtype=np.float64)
        lags = np.arange(1, self.ar_params.shape[0] + 1, dtype=np.float64)
        persistence = 1.0 - float(np.sum(self.ar_params))
        if abs(persistence) < 1e-12:
            raise SpecificationError(
                "the autoregressive coefficients sum to one, so the fitted mean path is not "
                "finite and the result has no stationary law to sample from."
            )
        slope = (self.trend_coeff or 0.0) / persistence
        level = ((self.const or 0.0) - slope * float(lags @ self.ar_params)) / persistence
        intercept = level + slope * time
        return _simulate_arma(
            n,
            ar=self.ar_params,
            ma=np.zeros(0),
            sigma=float(np.sqrt(self.sigma2)),
            rng=rng,
            intercept=intercept,
            burn=burn,
        )

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(100)
            >>> AR(y, order=2).fit()._comparison_label()
            'AR(2)'
            >>> AR(y, order=1, trend="ct").fit()._comparison_label()
            'AR(1) trend=ct'
        """
        return f"AR({self.order}){'' if self.trend == 'c' else f' trend={self.trend}'}"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        Metadata carries the specification, the log-likelihood and the
        three information criteria; the coefficient table is
        :attr:`params` in order; the notes give the stationarity verdict
        with the largest companion root.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> table = AR(rng.standard_normal(100), order=1).fit()._summary_table()
            >>> table.title, table.columns, [row[0] for row in table.rows]
            ('AR(1) Results', ('', 'coef'), ['const', 'ar.L1', 'sigma2'])
        """
        ic: InformationCriteria = self.information_criteria
        stability = self.stability
        return SummaryTable(
            title=f"AR({self.order}) Results",
            metadata=(
                ("Model", f"AR({self.order})"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Method", self.method),
                ("AIC", f"{ic.aic:.3f}"),
                ("Trend", self.trend),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=(
                f"Stationary: {self.is_stationary}   "
                f"max |companion root| = {stability.max_modulus:.4f}",
                "Standard errors are not yet available for this estimator.",
            ),
        )


class AR(_AutoRegressionModel[ARResult]):
    r"""Autoregressive AR(p) specification.

    The immutable statement of

    .. math::

        y_t = c + \delta t + \sum_{i=1}^{p} \phi_i\, y_{t-i} + \varepsilon_t,
        \qquad \varepsilon_t \sim N(0, \sigma^2),

    validated at construction and estimated by :meth:`fit`. Two
    estimators are reachable through ``method``: ``"css"`` conditions on
    the first :math:`p` observations and solves the lagged regression by
    ordinary least squares, so the effective sample is :math:`n - p`
    and the likelihood is the conditional one; ``"exact"`` maximizes the
    exact Gaussian likelihood through the companion state-space
    embedding, the search kept inside the stationary region by a
    partial-autocorrelation reparameterization, so the first :math:`p`
    observations contribute their stationary density and the effective
    sample is :math:`n`. The two agree as the sample grows and differ in
    finite samples by the treatment of those :math:`p` observations.
    ``order=0`` is admitted and fits the deterministic terms alone.

    Args:
        endog: The endogenous series, one-dimensional, of length at
            least ``order + 2``.
        order: Autoregressive order ``p``, a non-negative integer.
        trend: Deterministic specification (``"n"``, ``"c"``, ``"ct"``).
        method: ``"css"`` or ``"exact"``.

    Raises:
        SpecificationError: If the order is not a non-negative integer,
            the trend or method is unrecognized, or exact ML is requested
            with a linear trend.
        DimensionError: If the series is not one-dimensional or is
            shorter than ``order + 2``.

    Note:
        Exact ML with ``trend="ct"`` is refused in this release; a
        linear trend is available under CSS only. Because the two
        estimators report different effective sample sizes, information
        criteria are comparable across orders only when every candidate
        uses ``"exact"``, and :meth:`ARResult.compare` enforces that.

    See Also:
        * :class:`ARResult` -- what :meth:`fit` returns.
        * :class:`~cultivars.univariate.box_jenkins.ARIMA` -- the
          moving-average and integrated generalization.
        * :func:`~cultivars.diagnostics.unit_roots.adf` -- whether the
          series belongs in levels before an autoregression is fitted
          to it.

    References:
        Hamilton, J. D. (1994). *Time Series Analysis*, chapters 5 and
        13. Princeton University Press.

        Monahan, J. F. (1984). A note on enforcing stationarity in ARMA
        models. *Biometrika*, 71(2), 403-404.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> e = rng.standard_normal(500)
        >>> y = np.zeros(500)
        >>> for t in range(1, 500):
        ...     y[t] = 0.5 * y[t - 1] + e[t]
        >>> res = AR(y, order=1, trend="c").fit()
        >>> bool(0.3 < res.ar_params[0] < 0.7)
        True

        CSS and exact ML agree closely on a stationary AR(2) and differ
        in the sample they report:

        >>> z = np.zeros(400)
        >>> for t in range(2, 400):
        ...     z[t] = 0.5 * z[t - 1] - 0.3 * z[t - 2] + e[t]
        >>> css, exact = AR(z, order=2).fit(), AR(z, order=2, method="exact").fit()
        >>> css.nobs, exact.nobs, bool(np.allclose(css.ar_params, exact.ar_params, atol=0.01))
        (398, 400, True)
        >>> AR(z, order=1, trend="ct", method="exact")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: exact ML with trend='ct' is not supported ...
    """

    __slots__ = ()

    def fit(self) -> ARResult:
        """Estimate the model by the selected method.

        Returns:
            The fitted :class:`ARResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = AR(rng.standard_normal(100), order=0).fit()
            >>> list(res.params), res.nobs
            (['const', 'sigma2'], 100)
        """
        return ARResult._from_fit(self._fit_family(), self)
