# filepath: /src/cultivars/univariate/arma.py
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
r"""ARMA / ARIMA / SARIMA / SARIMAX -- one engine behind four public fronts.

All four are the same estimator. The conditional mean is the
multiplicative seasonal ARMA

.. math::

   \phi(L)\,\Phi(L^s)\,\bigl(w_t - x_t'\beta\bigr) = \theta(L)\,\Theta(L^s)\,\varepsilon_t,
   \qquad
   w_t = (1 - L)^d (1 - L^s)^D y_t,
   \qquad \varepsilon_t \sim N(0, \sigma^2),

with :math:`\theta(L) = 1 + \theta_1 L + \cdots` (the plus sign
convention), cast in the Harvey state-space form so the likelihood is
exact and runs through the Kalman filter,

.. math::

   \log L = -\tfrac12 \sum_{t} \bigl(\log 2\pi + \log F_t + v_t^2 / F_t\bigr),

with :math:`v_t` and :math:`F_t` the filter's innovations and their
variances. Integration is applied by differencing before estimation;
deterministic terms and exogenous regressors enter the observation
intercept, which makes ``ARIMAX`` and ``SARIMAX`` nothing more than
``ARIMA`` and ``SARIMA`` with ``exog`` supplied. Stationarity and
invertibility are enforced structurally by the partial-autocorrelation
reparameterization, so the optimizer searches an unconstrained space
and the reported diagnostics verify the transform rather than describe
the data -- a unit root fitted this way lands at a root of 0.999 with
``Stationary: True``, and ``d`` is the right answer, not a different
optimizer.

Two commitments shape the surface. First, :class:`SARIMAX` is the real
specification and the other four names are narrowing constructors over
it, not separate models: a user who reaches for ``ARMA`` cannot pass a
differencing order, one who reaches for ``SARIMA`` need not spell out
``(0, 0, 0, 0)``, and one who reaches for ``ARIMAX`` must supply
regressors, so an import path reads as the specification it is and the
four fits of one model return identical likelihoods. Second, both
stability verdicts are assessed on the *multiplied* polynomials
:math:`\phi(L)\Phi(L^s)` and :math:`\theta(L)\Theta(L^s)`; checking the
non-seasonal block alone would pass a specification with an explosive
seasonal root, and the result exposes the expanded polynomials so the
verdict can be audited.

Layout. :class:`SARIMAX` validates ``order`` and ``seasonal_order``
through ``validate_order_tuple``, the trend through ``validate_choice``
against :data:`~cultivars.typing.Trend`, and the regressors through
``validate_exog``, all on the ``_BoxJenkinsModel`` base in
``_internals``; ``fit`` builds a ``_BoxJenkinsObjective`` whose
``state_space`` maps a parameter vector to a
``_LinearGaussianStateSpace`` through ``_from_arma``, and hands it to
``_maximize_likelihood``. :class:`ARIMAX`, :class:`SARIMA`,
:class:`ARIMA` and :class:`ARMA` fix or forbid arguments and delegate.
The packed ``_BoxJenkinsFit`` is assembled into :class:`ARMAResult` by
``_from_fit``; ``expand_ar`` and ``expand_ma`` in ``_core`` multiply
the polynomials for the stability and invertibility checks and for
``simulate``, which runs ``_simulate_arma`` and undoes the differencing
with ``_integrate``; ``combined_difference`` applies it, and
``deterministic_columns`` and ``n_deterministic`` lay out the trend
block. The pure autoregression with conditional least squares is
:mod:`~cultivars.univariate.autoregression`; the long-memory
generalization :mod:`~cultivars.univariate.fractional_integration`; the
permanent-transitory reading of an ARIMA(p, 1, q) result
:class:`~cultivars.spectral.filters.BeveridgeNelsonDecomposition`.

References:
    Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M.
    (2015). *Time Series Analysis: Forecasting and Control*, 5th ed.
    Wiley.

    Harvey, A. C. (1989). *Forecasting, Structural Time Series Models
    and the Kalman Filter*. Cambridge University Press.

    Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State
    Space Methods*, 2nd ed., chapter 3. Oxford University Press.

    Monahan, J. F. (1984). A note on enforcing stationarity in ARMA
    models. *Biometrika*, 71(2), 403-404.

Example:
    One model, four fronts, one likelihood; then order selection on a
    common sample:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> e = rng.standard_normal(500)
    >>> y = np.zeros(500)
    >>> for t in range(1, 500):
    ...     y[t] = 0.6 * y[t - 1] + e[t] + 0.3 * e[t - 1]
    >>> fits = (
    ...     ARMA(y, order=(1, 1)).fit(),
    ...     ARIMA(y, order=(1, 0, 1)).fit(),
    ...     SARIMA(y, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0)).fit(),
    ...     SARIMAX(y, order=(1, 0, 1)).fit(),
    ... )
    >>> len({round(fit.llf, 6) for fit in fits})
    1
    >>> candidates = [ARMA(y, order=(p, q)).fit() for p in (1, 2) for q in (0, 1)]
    >>> table = candidates[0].compare(*candidates[1:], criterion="bic")
    >>> table.rows[0][0] in {"ARIMA(1, 0, 1)", "ARIMA(2, 0, 0)"}
    True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..engine._core import (
    _SIMULATION_BURN,
    InformationCriteria,
    SummaryTable,
    deterministic_columns,
    expand_ar,
    expand_ma,
    n_deterministic,
)
from ..engine._internals import (
    _BoxJenkinsFit,
    _BoxJenkinsModel,
    _ComparisonMixin,
    _integrate,
    _InvertibilityMixin,
    _SeriesMixin,
    _simulate_arma,
    _StationarityMixin,
    _SummaryMixin,
)
from ..exceptions import SpecificationError

__all__ = ["ARIMA", "ARIMAX", "ARMA", "SARIMA", "SARIMAX", "ARMAResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ARMAResult(
    _SummaryMixin, _SeriesMixin, _ComparisonMixin, _StationarityMixin, _InvertibilityMixin
):
    r"""A fitted (seasonal) ARIMA, optionally with exogenous regressors.

    The estimates of the multiplicative seasonal model on the differenced
    series :math:`w_t = \Delta^d \Delta_s^D y_t`,

    .. math::

        \phi(L)\,\Phi(L^s)\,\bigl(w_t - x_t'\beta\bigr)
        = \theta(L)\,\Theta(L^s)\,\varepsilon_t,
        \qquad \varepsilon_t \sim N(0, \sigma^2),

    where :math:`x_t` stacks the deterministic terms and the exogenous
    regressors, so that :math:`\beta` shifts the observation and the
    ARMA structure describes what is left. One record serves ARMA, ARIMA,
    SARIMA and SARIMAX: the orders and the seasonal block are fields, and
    the polynomials a diagnostic reads are always the multiplied ones,
    :math:`\phi(L)\Phi(L^s)` for stationarity and
    :math:`\theta(L)\Theta(L^s)` for invertibility, because either factor
    alone can pass while their product has an explosive root. Composes
    five capabilities: rendering, frame interop, criterion-named
    comparison, and the stationarity and invertibility assessments.

    Attributes:
        endog: The full observed series.
        fittedvalues: One-step fitted values on the differenced modeling series.
        resid: One-step residuals on the differenced modeling series.
        llf: Maximized exact log-likelihood.
        nobs: Observations after differencing.
        n_params: Free parameter count, including the innovation variance.
        order: Non-seasonal ``(p, d, q)``.
        seasonal_order: Seasonal ``(P, D, Q, s)``.
        trend: Deterministic specification.
        k_exog: Number of exogenous regressors.
        ar_params: Non-seasonal AR coefficients.
        ma_params: Non-seasonal MA coefficients.
        seasonal_ar_params: Seasonal AR coefficients.
        seasonal_ma_params: Seasonal MA coefficients.
        beta: Coefficients on the deterministic and exogenous block.
        sigma2: Innovation variance.

    Note:
        ``beta`` is an *observation* intercept, not a term inside the
        autoregression: under ``trend="c"`` its first entry is the mean
        of the differenced series, :math:`E[w_t] = \beta_0`, where the
        ``const`` of :class:`~cultivars.univariate.autoregression.ARResult`
        is the recursion's intercept with mean
        :math:`c / (1 - \sum\phi_i)`. The two agree only at
        :math:`\phi = 0`; compare them through the implied mean, not the
        coefficient. The fitted and residual paths live on the
        differenced scale and have ``nobs`` rows, ``d + sD`` fewer than
        ``endog``. Standard errors are not yet reported by this
        estimator.

    See Also:
        * :class:`SARIMAX`, :class:`ARIMAX`, :class:`SARIMA`,
          :class:`ARIMA`, :class:`ARMA` -- the specifications that
          produce this.
        * :class:`~cultivars.univariate.autoregression.ARResult` -- the
          pure autoregression, with the other intercept convention.
        * :class:`~cultivars.spectral.filters.BeveridgeNelsonDecomposition`
          -- reads an ARIMA(p, 1, q) result into a permanent-transitory
          decomposition.

    References:
        Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M.
        (2015). *Time Series Analysis: Forecasting and Control*, 5th ed.
        Wiley.

        Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by
        State Space Methods*, 2nd ed., chapter 3. Oxford University
        Press.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> e = rng.standard_normal(600)
        >>> y = np.zeros(600)
        >>> for t in range(1, 600):
        ...     y[t] = 0.6 * y[t - 1] + e[t] + 0.4 * e[t - 1]
        >>> res = ARMA(y, order=(1, 1)).fit()
        >>> res.order, res.seasonal_order, res.nobs, res.n_params
        ((1, 0, 1), (0, 0, 0, 0), 600, 4)
        >>> list(res.params)
        ['const', 'ar.L1', 'ma.L1', 'sigma2']
        >>> bool(abs(res.ar_params[0] - 0.6) < 0.1), bool(abs(res.ma_params[0] - 0.4) < 0.1)
        (True, True)
        >>> res.is_stationary, res.is_invertible
        (True, True)

        A seasonal fit is judged on the multiplied polynomial, whose
        length is :math:`p + sP`:

        >>> s = np.zeros(600)
        >>> for t in range(13, 600):
        ...     s[t] = 0.5 * s[t - 1] + 0.4 * s[t - 12] - 0.2 * s[t - 13] + e[t]
        >>> seasonal = SARIMA(s, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)).fit()
        >>> list(seasonal.params), seasonal._stationarity_ar().shape
        (['const', 'ar.L1', 'ar.S.L12', 'sigma2'], (13,))
    """

    endog: npt.NDArray[np.float64]
    """The full observed series, on the level scale."""

    fittedvalues: npt.NDArray[np.float64]
    """``(nobs,)`` one-step fitted values on the differenced scale."""

    resid: npt.NDArray[np.float64]
    """``(nobs,)`` one-step residuals on the differenced scale."""

    llf: float
    """Maximized exact log-likelihood."""

    nobs: int
    """Observations after differencing."""

    n_params: float
    """Free parameter count, including the innovation variance."""

    order: tuple[int, int, int]
    """Non-seasonal ``(p, d, q)``."""

    seasonal_order: tuple[int, int, int, int]
    """Seasonal ``(P, D, Q, s)``; all zero without a seasonal block."""

    trend: str
    """``"n"``, ``"c"`` or ``"ct"``."""

    k_exog: int
    """Number of exogenous regressors."""

    ar_params: npt.NDArray[np.float64]
    """``(p,)`` non-seasonal AR coefficients."""

    ma_params: npt.NDArray[np.float64]
    """``(q,)`` non-seasonal MA coefficients."""

    seasonal_ar_params: npt.NDArray[np.float64]
    """``(P,)`` seasonal AR coefficients."""

    seasonal_ma_params: npt.NDArray[np.float64]
    """``(Q,)`` seasonal MA coefficients."""

    beta: npt.NDArray[np.float64]
    """Coefficients on the deterministic terms then the exogenous regressors."""

    sigma2: float
    """Innovation variance."""

    @classmethod
    def _from_fit(cls, fit: _BoxJenkinsFit, model: _BoxJenkinsModel[ARMAResult]) -> ARMAResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The raw estimator output.
            model: The specification that produced it, read for the
                orders, trend, exogenous count and series.

        Returns:
            The assembled result.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> model = ARMA(rng.standard_normal(100), order=(1, 0))
            >>> res = ARMAResult._from_fit(model._fit_family(), model)
            >>> res.order, res.k_exog, res.endog.shape
            ((1, 0, 0), 0, (100,))
        """
        exog = model.exog
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            order=model.order,
            seasonal_order=model.seasonal_order,
            trend=model.trend,
            k_exog=0 if exog is None else int(exog.shape[1]),
            ar_params=fit.ar_params,
            ma_params=fit.ma_params,
            seasonal_ar_params=fit.seasonal_ar_params,
            seasonal_ma_params=fit.seasonal_ma_params,
            beta=fit.beta,
            sigma2=fit.sigma2,
        )

    @property
    def seasonal_period(self) -> int:
        """The seasonal period ``s``; ``0`` when no seasonal block is present.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(120)
            >>> ARMA(y, order=(1, 0)).fit().seasonal_period
            0
            >>> SARIMA(y, order=(1, 0, 0), seasonal_order=(0, 0, 1, 4)).fit().seasonal_period
            4
        """
        return self.seasonal_order[3]

    def _stationarity_ar(self) -> npt.NDArray[np.float64]:
        r"""The multiplied polynomial :math:`\phi(L)\,\Phi(L^s)`.

        Assessing the non-seasonal block alone would pass a specification whose
        seasonal root is explosive.

        Returns:
            ``(p + sP,)`` coefficients of the expanded autoregressive
            polynomial, which :attr:`stability` reads.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(200)
            >>> res = SARIMA(y, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4)).fit()
            >>> expanded = res._stationarity_ar()
            >>> expanded.shape, bool(np.isclose(expanded[3], res.seasonal_ar_params[0]))
            ((5,), True)
            >>> bool(np.isclose(expanded[4], -res.ar_params[0] * res.seasonal_ar_params[0]))
            True
        """
        return expand_ar(self.ar_params, self.seasonal_ar_params, self.seasonal_period)

    def _invertibility_ma(self) -> npt.NDArray[np.float64]:
        r"""The multiplied polynomial :math:`\theta(L)\,\Theta(L^s)`, sign-corrected.

        The invertibility assessment expects the polynomial in the same
        sign convention as the autoregressive one, so the expanded MA
        coefficients are negated before they are handed over.

        Returns:
            ``(q + sQ,)`` coefficients.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> res = ARMA(rng.standard_normal(200), order=(0, 1)).fit()
            >>> bool(np.isclose(res._invertibility_ma()[0], -res.ma_params[0]))
            True
        """
        return -expand_ma(self.ma_params, self.seasonal_ma_params, self.seasonal_period)

    def simulate(
        self,
        n: int = 200,
        *,
        seed: int | np.random.Generator | None = None,
        burn: int = _SIMULATION_BURN,
    ) -> npt.NDArray[np.float64]:
        """A fresh sample path at the fitted parameters.

        The stationary ARMA part -- seasonal and non-seasonal polynomials
        multiplied out -- is generated from a zero start over ``burn + n``
        periods with the first ``burn`` discarded; the deterministic block
        is added on the differenced scale with the time index running
        ``1 .. n``, exactly as the estimator laid it out; and the result
        is integrated ``d`` and ``D`` times from zero initial levels, so
        the first ``d + sD`` entries of the returned path are zero.

        Args:
            n: Observations kept, on the level scale; must exceed
                ``d + sD``.
            seed: Seed or generator.
            burn: Periods discarded from the start.

        Returns:
            An array of shape ``(n,)``.

        Raises:
            SpecificationError: If the counts are not usable or the fit
                carries exogenous regressors, whose future the result
                cannot know.

        Example:
            The simulated variance and mean are the ARMA(1, 1) closed
            forms at the estimates:

            >>> rng = np.random.default_rng(0)
            >>> e = rng.standard_normal(600)
            >>> y = np.zeros(600)
            >>> for t in range(1, 600):
            ...     y[t] = 0.6 * y[t - 1] + e[t] + 0.4 * e[t - 1]
            >>> res = ARMA(y, order=(1, 1)).fit()
            >>> path = res.simulate(50000, seed=0)
            >>> phi, theta = float(res.ar_params[0]), float(res.ma_params[0])
            >>> variance = res.sigma2 * (1 + 2 * phi * theta + theta**2) / (1 - phi**2)
            >>> bool(abs(path.var() - variance) < 0.05), bool(abs(path.mean() - res.beta[0]) < 0.05)
            (True, True)
            >>> ARIMA(np.cumsum(y), order=(1, 1, 1)).fit().simulate(1)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: n must exceed the integration orders (1); ...
        """
        if self.k_exog:
            raise SpecificationError(
                "a fit with exogenous regressors cannot simulate its own sample: the "
                "regressors' future is not part of the model."
            )
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        d, cap_d, s = self.order[1], self.seasonal_order[1], self.seasonal_order[3]
        det = n_deterministic(self.trend)
        differenced = n - d - s * cap_d
        if differenced < 1:
            raise SpecificationError(
                f"n must exceed the integration orders ({d + s * cap_d}); got {n}."
            )
        intercept = (
            deterministic_columns(self.trend, differenced) @ self.beta[:det]
            if det
            else np.zeros(differenced)
        )
        w = _simulate_arma(
            differenced,
            ar=expand_ar(self.ar_params, self.seasonal_ar_params, s),
            ma=expand_ma(self.ma_params, self.seasonal_ma_params, s),
            sigma=float(np.sqrt(self.sigma2)),
            rng=rng,
            intercept=intercept,
            burn=burn,
        )
        levels = _integrate(w, d, cap_d, s)
        return np.concatenate([np.zeros(n - differenced), levels])

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order.

        The deterministic terms (``const``, ``trend``) and exogenous
        regressors (``x1`` .. ``xk``) come first in the order of
        ``beta``, then ``ar.L1`` .. ``ar.Lp``, ``ma.L1`` .. ``ma.Lq``,
        the seasonal terms at their lag multiples ``ar.S.Ls`` .. and
        ``ma.S.Ls`` .., and ``sigma2`` last.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(200)
            >>> list(SARIMA(y, order=(1, 0, 1), seasonal_order=(0, 0, 1, 4)).fit().params)
            ['const', 'ar.L1', 'ma.L1', 'ma.S.L4', 'sigma2']
            >>> X = rng.standard_normal((200, 2))
            >>> list(ARIMAX(y, order=(1, 0, 0), exog=X, trend="n").fit().params)
            ['x1', 'x2', 'ar.L1', 'sigma2']
        """
        out: dict[str, float] = {}
        det = n_deterministic(self.trend)
        names = (["const", "trend"][:det]) + [f"x{i}" for i in range(1, self.k_exog + 1)]
        for name, value in zip(names, self.beta, strict=True):
            out[name] = float(value)
        s = self.seasonal_period
        for i, value in enumerate(self.ar_params, start=1):
            out[f"ar.L{i}"] = float(value)
        for i, value in enumerate(self.ma_params, start=1):
            out[f"ma.L{i}"] = float(value)
        for i, value in enumerate(self.seasonal_ar_params, start=1):
            out[f"ar.S.L{i * s}"] = float(value)
        for i, value in enumerate(self.seasonal_ma_params, start=1):
            out[f"ma.S.L{i * s}"] = float(value)
        out["sigma2"] = self.sigma2
        return out

    def _specification(self) -> str:
        """Compact specification label, widening only as the model does.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(200)
            >>> ARMA(y, order=(1, 1)).fit()._specification()
            'ARIMA(1, 0, 1)'
            >>> SARIMA(y, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4)).fit()._specification()
            'SARIMA(1, 0, 0)(1, 0, 0, 4)'
            >>> X = rng.standard_normal((200, 1))
            >>> ARIMAX(y, order=(1, 0, 0), exog=X).fit()._specification()
            'ARIMA(1, 0, 0)+X1'
        """
        base = f"ARIMA{self.order}"
        if any(self.seasonal_order[:3]):
            base = f"SARIMA{self.order}{self.seasonal_order}"
        return f"{base}+X{self.k_exog}" if self.k_exog else base

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> ARMA(rng.standard_normal(100), order=(2, 0)).fit()._comparison_label()
            'ARIMA(2, 0, 0)'
        """
        return self._specification()

    def _notes(self) -> tuple[str, ...]:
        """Closing diagnostics, omitting any that the specification makes vacuous.

        The stationarity line always appears; the invertibility line
        only when there is a moving-average block to be invertible.

        Returns:
            Two or three note strings.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(200)
            >>> len(ARMA(y, order=(1, 0)).fit()._notes()), len(ARMA(y, order=(1, 1)).fit()._notes())
            (2, 3)
        """
        notes = [
            f"Stationary: {self.is_stationary}   max |AR root| = {self.stability.max_modulus:.4f}"
        ]
        if self.ma_params.size or self.seasonal_ma_params.size:
            notes.append(
                f"Invertible: {self.is_invertible}   "
                f"max |MA root| = {self.invertibility.max_modulus:.4f}"
            )
        notes.append("Standard errors are not yet available for this estimator.")
        return tuple(notes)

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        Metadata carries the specification label, trend, exogenous
        count, sample size and the likelihood with its three criteria;
        the coefficient table is :attr:`params` in order; the notes are
        :meth:`_notes`.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> table = ARMA(rng.standard_normal(200), order=(1, 1)).fit()._summary_table()
            >>> table.title, table.metadata[0], table.columns
            ('ARIMA(1, 0, 1) Results', ('Model', 'ARIMA(1, 0, 1)'), ('', 'coef'))
        """
        ic: InformationCriteria = self.information_criteria
        return SummaryTable(
            title=f"{self._specification()} Results",
            metadata=(
                ("Model", self._specification()),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Trend", self.trend),
                ("AIC", f"{ic.aic:.3f}"),
                ("Exog regressors", f"{self.k_exog}"),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=self._notes(),
        )


class SARIMAX(_BoxJenkinsModel[ARMAResult]):
    r"""Seasonal ARIMA with exogenous regressors -- the general specification.

    The model on the differenced series
    :math:`w_t = (1 - L)^d (1 - L^s)^D y_t` is

    .. math::

        \phi(L)\,\Phi(L^s)\,(w_t - x_t'\beta)
        = \theta(L)\,\Theta(L^s)\,\varepsilon_t,
        \qquad \varepsilon_t \sim N(0, \sigma^2),

    with :math:`\phi(L) = 1 - \phi_1 L - \cdots - \phi_p L^p`,
    :math:`\theta(L) = 1 + \theta_1 L + \cdots + \theta_q L^q`, their
    seasonal counterparts in :math:`L^s`, and :math:`x_t` the
    deterministic terms of ``trend`` followed by the columns of ``exog``,
    differenced alongside the series. The moving-average sign convention
    is the plus one: a simulated :math:`e_t - 0.4 e_{t-1}` is estimated
    as :math:`\theta_1 \approx -0.4`. Estimation is exact maximum
    likelihood through the Harvey state-space form and the Kalman
    filter, with the autoregressive and moving-average polynomials kept
    stationary and invertible by the partial-autocorrelation
    reparameterization; starting values come from a regression of the
    differenced series on :math:`x_t` and a short autoregression on its
    residuals.

    Every other class in this module narrows this one. Here alone
    ``exog`` is optional, because this is the form a user reaches for
    when they want the full surface rather than a named special case;
    :class:`ARIMAX` is the non-seasonal variant that makes regressors
    mandatory.

    Args:
        endog: The endogenous series, one-dimensional, of length at
            least :math:`p + d + q + s(P + D + Q) + 2`.
        order: Non-seasonal ``(p, d, q)``, non-negative integers.
        seasonal_order: Seasonal ``(P, D, Q, s)``; ``s`` must be at
            least 2 whenever any of ``P``, ``D``, ``Q`` is positive.
        trend: Deterministic specification (``"n"``, ``"c"``, ``"ct"``).
        exog: Optional exogenous regressors, ``(nobs, k)`` or ``(nobs,)``,
            differenced alongside ``endog``.

    Raises:
        SpecificationError: If an order tuple has the wrong length or a
            negative entry, seasonal terms are requested with a period
            below 2, or the trend is unrecognized.
        DimensionError: If the series is not one-dimensional, is too
            short for the specification, or ``exog`` does not have one
            row per observation.

    Note:
        Regressors enter as an observation intercept, so this is a
        regression with SARIMA errors, not a transfer-function model:
        ``exog`` shifts the level of :math:`w_t` and the ARMA structure
        describes what remains. The differencing that ``d`` and ``D``
        apply to ``endog`` is applied to ``exog`` too, so a regressor in
        levels against a series in first differences must be supplied in
        levels, and its coefficient reads on the differenced scale. A
        result with regressors cannot ``simulate`` -- their future is
        not part of the model.

    See Also:
        * :class:`ARMAResult` -- what :meth:`fit` returns.
        * :class:`ARIMAX`, :class:`SARIMA`, :class:`ARIMA`, :class:`ARMA`
          -- the narrowing constructors.
        * :class:`~cultivars.univariate.autoregression.AR` -- the
          conditional-least-squares alternative for a pure
          autoregression.

    References:
        Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M.
        (2015). *Time Series Analysis: Forecasting and Control*, 5th ed.
        Wiley.

        Harvey, A. C. (1989). *Forecasting, Structural Time Series Models
        and the Kalman Filter*. Cambridge University Press.

    Example:
        The airline model on a series simulated from it:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> e = rng.standard_normal(400)
        >>> y = np.zeros(400)
        >>> for t in range(13, 400):
        ...     y[t] = (
        ...         y[t - 1] + y[t - 12] - y[t - 13]
        ...         + e[t] - 0.4 * e[t - 1] - 0.6 * e[t - 12] + 0.24 * e[t - 13]
        ...     )
        >>> res = SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), trend="n").fit()
        >>> res.nobs, res.ma_params.round(1).tolist(), res.seasonal_ma_params.round(1).tolist()
        (387, [-0.4], [-0.6])
        >>> res.is_invertible
        True

        A regression with AR(1) errors recovers its slopes:

        >>> X = rng.standard_normal((400, 2))
        >>> noise = rng.standard_normal(400)
        >>> reg = SARIMAX(X @ [1.0, -2.0] + noise, order=(1, 0, 0), exog=X).fit()
        >>> reg.k_exog, reg.beta[1:].round(1).tolist()
        (2, [1.0, -2.0])
    """

    __slots__ = ()

    def fit(self) -> ARMAResult:
        """Estimate by exact maximum likelihood.

        Returns:
            The fitted :class:`ARMAResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = SARIMAX(rng.standard_normal(120), order=(1, 0, 1)).fit()
            >>> res.order, list(res.params)
            ((1, 0, 1), ['const', 'ar.L1', 'ma.L1', 'sigma2'])
        """
        return ARMAResult._from_fit(self._fit_family(), self)


class ARIMAX(SARIMAX):
    r"""ARIMA(p, d, q) with exogenous regressors, no seasonal block.

    A regression with ARIMA errors,

    .. math::

        (1 - L)^d\,(y_t - x_t'\beta) = \frac{\theta(L)}{\phi(L)}\,\varepsilon_t,

    so the regressors enter the observation intercept and the ARIMA
    structure describes what they leave behind. When ``d`` is non-zero
    the regressors are differenced alongside ``endog``, which is what
    makes the specification a *levels* relationship with an integrated
    error: :math:`y_t = x_t'\beta + u_t` with :math:`u_t \sim` ARIMA(p, d, q)
    is estimated on :math:`\Delta^d y_t = \Delta^d x_t'\beta + \Delta^d u_t`,
    and a cointegrating relation comes back correctly -- ``y = 2 x1 - x2 +
    I(1) error`` recovers about ``2.00`` and ``-1.00`` at ``d = 1``. The
    converse is the trap: a relationship between :math:`\Delta y_t` and
    the *level* of a stationary driver, :math:`\Delta y_t = 2 x_t +
    e_t`, is not this model, and fitting it at ``d = 1`` regresses
    :math:`\Delta y_t` on :math:`\Delta x_t` and returns about ``1.0``.
    That is a property of the specification rather than of the
    estimator; supply the cumulated driver, or model the differences at
    ``d = 0``.

    ``exog`` is required and an empty block is rejected. That is the whole
    point of the name: an ``ARIMAX`` with no regressors is an :class:`ARIMA`,
    and silently accepting one would let a call site claim a covariate model it
    does not have.

    Args:
        endog: The endogenous series.
        order: ``(p, d, q)``.
        exog: Exogenous regressors, shape ``(nobs,)`` or ``(nobs, k)``,
            with at least one column.
        trend: Deterministic specification.

    Raises:
        SpecificationError: If ``exog`` is ``None`` or has zero columns.
        DimensionError: If ``exog`` is not aligned with ``endog``.

    See Also:
        * :class:`SARIMAX` -- the same model with a seasonal block.
        * :class:`ARIMA` -- without regressors.
        * :mod:`~cultivars.diagnostics.cointegration` -- whether a
          levels relationship with an integrated error is the right
          reading of the data before it is imposed.

    Example:
        A cointegrating relation in levels with an I(1) error, and the
        misspecified reading of a differenced series against a
        stationary driver:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((400, 2))
        >>> levels = X @ [2.0, -1.0] + np.cumsum(rng.standard_normal(400))
        >>> beta = ARIMAX(levels, order=(0, 1, 0), exog=X, trend="n").fit().beta
        >>> bool(np.allclose(beta, [2.0, -1.0], atol=0.15))
        True
        >>> x = rng.standard_normal(400)
        >>> y = np.cumsum(2.0 * x + rng.standard_normal(400))
        >>> round(float(ARIMAX(y, order=(1, 1, 0), exog=x).fit().beta[1]), 1)
        1.0
        >>> round(float(ARIMAX(y, order=(1, 1, 0), exog=np.cumsum(x)).fit().beta[1]), 1)
        2.0
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: tuple[int, int, int],
        exog: npt.ArrayLike,
        trend: str = "c",
    ) -> None:
        """Validate the specification and the data.

        Args:
            endog: The endogenous series.
            order: ``(p, d, q)``.
            exog: Exogenous regressors; required.
            trend: Deterministic specification.

        Raises:
            SpecificationError: If ``exog`` is ``None`` or empty, or the
                order or trend is invalid.
            DimensionError: If ``exog`` is misaligned or the series is
                too short.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(100)
            >>> ARIMAX(y, order=(1, 0, 0), exog=None)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: ARIMAX requires exogenous regressors; ...
            >>> ARIMAX(y, order=(1, 0, 0), exog=np.zeros((100, 0)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: ARIMAX requires at least one exogenous ...
        """
        if exog is None:
            raise SpecificationError(
                "ARIMAX requires exogenous regressors; use ARIMA for a model without them."
            )
        super().__init__(endog, order=order, trend=trend, exog=exog)
        if self.exog is None or self.exog.shape[1] == 0:
            raise SpecificationError(
                "ARIMAX requires at least one exogenous regressor; "
                "use ARIMA for a model without them."
            )


class SARIMA(SARIMAX):
    r"""Multiplicative seasonal ARIMA, without exogenous regressors.

    The model

    .. math::

        \phi(L)\,\Phi(L^s)\,(1 - L)^d (1 - L^s)^D\,(y_t - \mu_t)
        = \theta(L)\,\Theta(L^s)\,\varepsilon_t,

    in which the seasonal polynomials act at multiples of the period and
    multiply the non-seasonal ones, so a first-order seasonal
    autoregression at :math:`s = 4` beside a non-seasonal AR(1) puts
    coefficients at lags 1, 4 and 5 with the lag-5 term fixed at
    :math:`-\phi_1\Phi_1`. The seasonal block is mandatory here -- a
    :class:`SARIMA` without one is an :class:`ARIMA`, and the name should
    not be able to claim a seasonal structure it does not carry -- and
    ``exog`` is not accepted; :class:`SARIMAX` is the form with both.

    Args:
        endog: The endogenous series.
        order: Non-seasonal ``(p, d, q)``.
        seasonal_order: Seasonal ``(P, D, Q, s)`` with ``s`` at least 2
            when any of ``P``, ``D``, ``Q`` is positive.
        trend: Deterministic specification.

    Raises:
        SpecificationError: If an order has the wrong length or a
            negative entry, or seasonal terms are given with a period
            below 2.
        DimensionError: If the series is too short for the
            specification.

    See Also:
        * :class:`SARIMAX` -- the same with exogenous regressors.
        * :class:`ARIMA` -- without the seasonal block.
        * :mod:`~cultivars.diagnostics.seasonality` -- whether the
          series has the seasonal structure this imposes.

    Example:
        A seasonal AR(1) over a non-seasonal AR(1) at quarterly period,
        recovered on a series simulated from the multiplied polynomial:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> e = rng.standard_normal(400)
        >>> y = np.zeros(400)
        >>> for t in range(5, 400):
        ...     y[t] = 0.5 * y[t - 1] + 0.4 * y[t - 4] - 0.2 * y[t - 5] + e[t]
        >>> res = SARIMA(y, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4)).fit()
        >>> list(res.params)
        ['const', 'ar.L1', 'ar.S.L4', 'sigma2']
        >>> phi, seasonal_phi = res.ar_params[0], res.seasonal_ar_params[0]
        >>> bool(abs(phi - 0.5) < 0.1), bool(abs(seasonal_phi - 0.4) < 0.1)
        (True, True)
        >>> res._stationarity_ar().shape, res.is_stationary
        ((5,), True)
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
        trend: str = "c",
    ) -> None:
        """Validate the specification and the data.

        Args:
            endog: The endogenous series.
            order: Non-seasonal ``(p, d, q)``.
            seasonal_order: Seasonal ``(P, D, Q, s)``; required.
            trend: Deterministic specification.

        Raises:
            SpecificationError: If an order or the trend is invalid.
            DimensionError: If the series is too short.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(120)
            >>> model = SARIMA(y, order=(0, 0, 1), seasonal_order=(0, 0, 1, 12))
            >>> model.order, model.seasonal_order, model.exog.shape
            ((0, 0, 1), (0, 0, 1, 12), (120, 0))
        """
        super().__init__(endog, order=order, seasonal_order=seasonal_order, trend=trend)


class ARIMA(SARIMAX):
    r"""ARIMA(p, d, q): non-seasonal differencing, no seasonal block, no regressors.

    The model :math:`\phi(L)\,(1 - L)^d\,(y_t - \mu_t) = \theta(L)\,\varepsilon_t`,
    with :math:`\mu_t` the deterministic terms of ``trend`` on the
    differenced scale, so that under ``trend="c"`` and ``d = 1`` the
    constant is the drift of the levels. Reach for :class:`ARIMAX` when
    there are exogenous regressors. Splitting the two is what makes
    either name informative: a reader of a call site knows from
    ``ARIMA`` alone that the fit has no external drivers.

    Args:
        endog: The endogenous series.
        order: ``(p, d, q)``.
        trend: Deterministic specification.

    Raises:
        SpecificationError: If the order has the wrong length or a
            negative entry, or the trend is unrecognized.
        DimensionError: If the series is too short for the
            specification.

    See Also:
        * :class:`ARIMAX` -- with exogenous regressors.
        * :class:`SARIMA` -- with a seasonal block.
        * :class:`ARMA` -- the ``d = 0`` case stated as such.
        * :func:`~cultivars.diagnostics.unit_roots.adf` -- how to choose
          ``d``.

    Example:
        A random walk with drift is an ARIMA(0, 1, 0) whose constant is
        the drift:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> levels = np.cumsum(0.3 + rng.standard_normal(500))
        >>> res = ARIMA(levels, order=(0, 1, 0)).fit()
        >>> res.nobs, round(float(res.beta[0]), 1)
        (499, 0.3)
    """

    __slots__ = ()

    def __init__(
        self, endog: npt.ArrayLike, *, order: tuple[int, int, int], trend: str = "c"
    ) -> None:
        """Validate the specification and the data.

        Args:
            endog: The endogenous series.
            order: ``(p, d, q)``.
            trend: Deterministic specification.

        Raises:
            SpecificationError: If the order or trend is invalid.
            DimensionError: If the series is too short.

        Example:
            >>> import numpy as np
            >>> model = ARIMA(np.arange(50.0), order=(1, 1, 1))
            >>> model.order, model.seasonal_order, model.trend
            ((1, 1, 1), (0, 0, 0, 0), 'c')
        """
        super().__init__(endog, order=order, trend=trend)


class ARMA(SARIMAX):
    r"""Stationary ARMA(p, q): no differencing, no seasonal block, no regressors.

    The model :math:`\phi(L)\,(y_t - \mu_t) = \theta(L)\,\varepsilon_t`
    on the series as given. Stationarity and invertibility are enforced
    by the reparameterization, so a series with a unit root is fitted
    with an autoregressive root pushed against the boundary rather than
    beyond it; the ``stability`` record's largest modulus is the tell,
    and :class:`ARIMA` with ``d = 1`` the remedy.

    Args:
        endog: The endogenous series.
        order: ``(p, q)``.
        trend: Deterministic specification.

    Raises:
        SpecificationError: If either order is negative or the trend is
            unrecognized.
        DimensionError: If the series is too short for the
            specification.

    See Also:
        * :class:`ARIMA` -- with differencing.
        * :class:`~cultivars.univariate.autoregression.AR` -- the pure
          autoregression, with conditional least squares available and
          the recursion-intercept convention for its constant.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> e = rng.standard_normal(600)
        >>> y = np.zeros(600)
        >>> for t in range(1, 600):
        ...     y[t] = 0.7 * y[t - 1] + e[t] + 0.4 * e[t - 1]
        >>> res = ARMA(y, order=(1, 1)).fit()
        >>> res.is_stationary and res.is_invertible
        True
        >>> bool(abs(res.ar_params[0] - 0.7) < 0.1), bool(abs(res.ma_params[0] - 0.4) < 0.1)
        (True, True)

        On a random walk the enforced stationarity shows as a root at
        the boundary:

        >>> walk = ARMA(np.cumsum(y), order=(1, 0)).fit()
        >>> bool(walk.ar_params[0] > 0.99), walk.is_stationary
        (True, True)
    """

    __slots__ = ()

    def __init__(self, endog: npt.ArrayLike, *, order: tuple[int, int], trend: str = "c") -> None:
        """Validate the specification and the data.

        Args:
            endog: The endogenous series.
            order: ``(p, q)``, expanded to ``(p, 0, q)``.
            trend: Deterministic specification.

        Raises:
            SpecificationError: If either order is negative or the trend
                is invalid.
            DimensionError: If the series is too short.

        Example:
            >>> import numpy as np
            >>> ARMA(np.arange(50.0), order=(2, 1)).order
            (2, 0, 1)
            >>> ARMA(np.arange(50.0), order=(1, -1))
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: q must be >= 0; got -1.
        """
        p, q = order
        super().__init__(endog, order=(p, 0, q), trend=trend)
