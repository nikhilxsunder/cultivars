# filepath: /src/cultivars/multivariate/reduced_form/vector_autoregression.py
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
r"""The reduced-form vector autoregression: the root of the multivariate surface.

A VAR(p) with deterministic terms,

.. math::

   y_t = D d_t + A_1 y_{t-1} + \cdots + A_p y_{t-p} + u_t,
   \qquad u_t \sim (0, \Sigma_u),

is estimated by multivariate least squares, which is equation-by-equation
OLS and, under Gaussian innovations, maximum likelihood at the same time.
The estimate itself is three arrays -- the coefficient stack, the innovation
covariance, the residuals -- and the point of the module is what those three
arrays are then made to answer: forecasts, impulse responses, the
forecast-error variance decomposition, the historical decomposition, Granger
causality, stability, lag selection, residual diagnostics and coefficient
inference. The VARX adds a distributed lag of weakly exogenous regressors to
the design and inherits every one of those answers unchanged, because
:math:`x` is conditioned on rather than shocked and so never enters the
moving-average representation; its forecast alone is new, and it is
conditional on a supplied path for :math:`x`.

Two commitments shape the surface. First, the inference lives on the arrays,
not on the estimator. Everything past the fit is written once on two mixins
in ``_internals`` -- propagation (companion form, stability, the
moving-average representation, impulse responses, decompositions, forecasts,
simulation) and inference (Granger and residual tests, coefficient
covariance and its standard errors) -- and neither asks how the arrays were
produced. That is what lets a VARX, a panel VAR, a VECM and a VARMA present
the same surface by supplying a different regressor block, a different
restriction or a different solve. Second, the one structural object on a
reduced-form result is labelled as such. An *orthogonalized* impulse
response requires a Cholesky factor, the factor is lower-triangular, and
triangularity is a recursive identifying restriction on the order of
``names``: permute the variables and the answer changes, while the forecast,
the Granger tests and the reduced-form responses do not. The result reports
it because everyone wants it, says on every summary that it is a recursive
SVAR whose assumption has not been declared, and points at
:mod:`~cultivars.multivariate.structural`, where the declaration goes.

Layout. :class:`VAR` is a thin ``fit()`` over
``_VectorAutoRegressionModel`` in ``_internals``, which validates the sample
through ``validate_endog_matrix`` in ``_core``, lays out
``[deterministic | lags]`` with ``deterministic_columns`` and ``lag_matrix``,
solves by ``_least_squares`` -- or by ``_shrunk_moments`` when a prior from
:mod:`~cultivars.bayes.priors` is attached -- and offers
``lag_order_selection`` before any fit. :class:`VARX` extends it with
``_ExogenousVectorAutoRegressionModel``, which appends the exogenous block
*after* the lags so that every offset the base computes stays correct
without an override. :class:`VARResult` and :class:`VARXResult` are
``_VectorResult`` records carrying the summary, comparison, inference and
propagation mixins; the VARX result adds the distributed-lag accessors, the
conditional forecast and ``to_varx``, the levels form that
:mod:`~cultivars.multivariate.reduced_form.closed_global` consumes. The
Cholesky caveat the summaries print is ``_CHOLESKY_NOTE`` in ``_core``.

References:
    Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1),
    1-48.

    Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
    Analysis*. Springer. Chapters 2-4 and 10.

    Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University
    Press. Chapter 11.

Example:
    A bivariate VAR(1) whose innovations are contemporaneously correlated.
    The same data fitted in the two variable orders give the same forecasts
    and the same Granger verdicts, and orthogonalized impulse responses that
    differ by more than a relabelling: the ordering is the identification.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> A = np.array([[0.5, 0.2], [0.0, 0.4]])
    >>> y = np.zeros((202, 2))
    >>> for t in range(1, 202):
    ...     u = rng.standard_normal(2)
    ...     u[1] += 0.8 * u[0]
    ...     y[t] = A @ y[t - 1] + u
    >>> y = y[2:]
    >>> ab = VAR(y, order=1, names=("a", "b")).fit()
    >>> ba = VAR(y[:, ::-1], order=1, names=("b", "a")).fit()
    >>> bool(np.allclose(ab.forecast(3), ba.forecast(3)[:, ::-1]))
    True
    >>> ab.granger_causality("b", "a").reject() == ba.granger_causality("b", "a").reject()
    True
    >>> swapped = ba.irf(2)[:, ::-1, ::-1]
    >>> bool(np.allclose(ab.irf(2), swapped)), round(float(np.abs(ab.irf(2) - swapped).max()), 2)
    (False, 0.84)
    >>> plain_ab, plain_ba = ab.irf(2, orthogonalized=False), ba.irf(2, orthogonalized=False)
    >>> bool(np.allclose(plain_ab, plain_ba[:, ::-1, ::-1]))
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from ..._core import (
    _CHOLESKY_NOTE,
    _UNSTABLE_NOTE,
    SummaryTable,
    deterministic_columns,
    validate_exog_matrix,
)
from ..._internals import (
    _ComparisonMixin,
    _ConditionalLevels,
    _ExogenousVectorAutoRegressionFit,
    _ExogenousVectorAutoRegressionModel,
    _SummaryMixin,
    _VectorAutoRegressionFit,
    _VectorAutoRegressionModel,
    _VectorInferenceMixin,
    _VectorPropagationMixin,
    _VectorResult,
)
from ...exceptions import DimensionError, SpecificationError

__all__ = ["VAR", "VARX", "VARResult", "VARXResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class VARResult(
    _VectorResult,
    _SummaryMixin,
    _ComparisonMixin,
    _VectorInferenceMixin,
    _VectorPropagationMixin,
):
    r"""A fitted reduced-form vector autoregression.

    The estimated system is

    .. math::

       y_t = D d_t + A_1 y_{t-1} + \cdots + A_p y_{t-p} + u_t,
       \qquad u_t \sim (0, \Sigma_u),

    solved by multivariate least squares, which under Gaussian innovations is
    the maximum-likelihood estimator of :math:`(D, A_1, \dots, A_p)` and
    equation-by-equation OLS at the same time. The record carries the
    coefficient stack, both residual covariances, the residuals and the
    regressor matrix; everything else on the surface -- forecasts, impulse
    responses, variance and historical decompositions, Granger causality,
    stability, residual diagnostics, coefficient inference -- is inherited
    from the mixins and computed from those arrays on demand.

    Note:
        Two residual covariances are stored and they are not
        interchangeable. :attr:`sigma_u` divides by :math:`T - kp - d` and
        feeds every inferential quantity: standard errors, Wald tests, the
        Cholesky impact matrix. :attr:`sigma_ml` divides by :math:`T` and
        feeds the likelihood and the information criteria, where the
        correction does not belong. Substituting one for the other biases
        either the criteria or the standard errors and nothing raises when
        it happens.

    Warning:
        Orthogonalized impulse responses and the variance decomposition use
        the Cholesky factor of :attr:`sigma_u`, which imposes the recursive
        ordering of :attr:`names` as an identifying restriction. Permute the
        variables and the answer changes. The summary says so on every fit;
        a declared identification scheme lives in
        :mod:`~cultivars.multivariate.structural`.

    Attributes:
        endog: The ``(n, k)`` sample as supplied.
        names: Variable labels, in Cholesky order.
        order: Autoregressive order ``p``.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        coefficients: ``(p, k, k)`` stack of ``A_1, ..., A_p``.
        deterministic: ``(d, k)`` deterministic coefficients, one row per
            term; zero rows when ``trend == "n"``.
        sigma_u: Residual covariance with the degrees-of-freedom correction.
        sigma_ml: Residual covariance divided by the effective sample.
        resid: ``(nobs, k)`` residuals over the effective sample.
        fittedvalues: ``(nobs, k)`` one-step conditional means.
        design: ``(nobs, d + kp)`` regressor matrix as estimated.
        llf: Gaussian log-likelihood at :attr:`sigma_ml`.
        nobs: Effective sample size ``n - p``.
        n_params: Free parameters, covariance included.
        posterior: Posterior coefficient inference when the model carried a
            prior; ``None`` for least squares.
        prior_label: What the prior was, for the summary; ``"none"`` for
            least squares.

    See Also:
        * :class:`VAR` -- the model whose ``fit()`` returns this record.
        * :class:`VARXResult` -- the same surface with a distributed lag of
          exogenous regressors appended to the design.
        * :class:`~cultivars.multivariate.reduced_form.error_correction.VECMResult`
          -- the same surface under a reduced-rank restriction on the levels.
        * :mod:`~cultivars.multivariate.structural` -- where the Cholesky
          ordering becomes a declared identification.

    References:
        Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*,
        48(1), 1-48.

        Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
        Analysis*. Springer. Chapters 2-4.

    Example:
        A bivariate VAR(1) with a one-way spillover. Least squares recovers
        the matrices, the Granger test finds the channel that is there and
        not the one that is not, and the one-step forecast is the fitted
        equation applied to the last observation:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> A = np.array([[0.5, 0.2], [0.0, 0.4]])
        >>> y = np.zeros((202, 2))
        >>> for t in range(1, 202):
        ...     y[t] = np.array([1.0, -0.5]) + A @ y[t - 1] + rng.standard_normal(2)
        >>> res = VAR(y[2:], order=1, names=("a", "b")).fit()
        >>> res.coefficients.shape, res.deterministic.shape, res.nobs, res.n_params
        ((1, 2, 2), (1, 2), 199, 9.0)
        >>> bool(np.abs(res.coefficients[0] - A).max() < 0.15)
        True
        >>> res.granger_causality("b", "a").reject(), res.granger_causality("a", "b").reject()
        (True, False)
        >>> hand = res.deterministic[0] + res.coefficients[0] @ y[-1]
        >>> bool(np.allclose(res.forecast(1)[0], hand))
        True
        >>> res.is_stable, res.irf(4).shape, res.fevd(4).shape
        (True, (5, 2, 2), (5, 2, 2))
    """

    coefficients: npt.NDArray[np.float64]
    """The ``(p, k, k)`` autoregressive stack ``A_1, ..., A_p``; ``coefficients[i][r, c]`` is the
    effect of variable ``c`` at lag ``i + 1`` on equation ``r``."""

    deterministic: npt.NDArray[np.float64]
    """The ``(d, k)`` deterministic coefficients, one row per term in the order constant, trend."""

    @classmethod
    def _from_fit(
        cls, fit: _VectorAutoRegressionFit, model: _VectorAutoRegressionModel[Self]
    ) -> Self:
        """Assemble the public result from the internal fit and its model.

        Args:
            fit: The arrays ``_fit_family`` produced, including the posterior
                inference when the model carried a prior.
            model: The model that produced it, read for the sample and the
                specification.

        Returns:
            A populated result.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> model = VAR(rng.standard_normal((120, 2)), order=1)
            >>> built = VARResult._from_fit(model._fit_family(), model)
            >>> bool(np.allclose(built.coefficients, model.fit().coefficients))
            True
        """
        return cls(
            endog=model.endog,
            names=model.names,
            order=model.order,
            trend=model.trend,
            coefficients=fit.coefficients,
            deterministic=fit.deterministic,
            sigma_u=fit.sigma_u,
            sigma_ml=fit.sigma_ml,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            design=fit.design,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            posterior=fit.posterior,
            prior_label=fit.prior_label,
        )

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table.

        The trend is named only when it is not the default constant.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal((120, 2))
            >>> VAR(y, order=2).fit()._comparison_label()
            'VAR(2)'
            >>> VAR(y, order=1, trend="ct").fit()._comparison_label()
            'VAR(1, trend=ct)'
        """
        tail = "" if self.trend == "c" else f", trend={self.trend}"
        return f"VAR({self.order}{tail})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        One row per coefficient with its standard error, statistic, p-value
        and interval; metadata with the criteria; notes with the stability
        verdict and the largest companion root, the Cholesky caveat, and an
        instability warning first when the system is not stable.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((120, 2)), order=1, names=("a", "b")).fit()
            >>> table = res._summary_table()
            >>> table.title, len(table.rows), table.rows[0][0]
            ('VAR(1) Results', 6, 'a: const')
            >>> dict(table.metadata)["Variables"], table.notes[0][:12]
            ('2', 'Stable: True')
        """
        criteria = self.information_criteria
        stability = self.stability_check()
        notes = [
            f"Stable: {self.is_stable}   max |companion root| = {stability.max_modulus:.4f}",
            _CHOLESKY_NOTE,
        ]
        if not self.is_stable:
            notes.insert(0, _UNSTABLE_NOTE)
        return SummaryTable(
            title=f"VAR({self.order}) Results",
            metadata=(
                ("Model", f"VAR({self.order})"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Variables", f"{self.k_endog}"),
                ("AIC", f"{criteria.aic:.3f}"),
                ("Trend", self.trend),
                ("BIC", f"{criteria.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{criteria.hqic:.3f}"),
            ),
            columns=self._coefficient_columns(),
            rows=self._coefficient_rows(),
            notes=tuple(notes),
        )


class VAR(_VectorAutoRegressionModel[VARResult]):
    """Reduced-form vector autoregression, estimated by multivariate least squares.

    The root of the multivariate surface: a ``(nobs, k)`` panel, a lag order
    and a deterministic specification, fitted in one least-squares solve. The
    constructor is inherited from ``_VectorAutoRegressionModel`` and takes an
    optional prior, which shrinks the solve toward a stated belief and lets a
    system with more regressors than observations estimate; the result then
    carries the posterior inference and its standard errors come from it.

    Args:
        endog: The ``(nobs, k)`` panel, time down the rows.
        order: Autoregressive order ``p``, at least zero.
        trend: ``"n"``, ``"c"`` (default) or ``"ct"``.
        names: One label per variable. Defaults to ``y1 ... yk``.
        prior: A prior from :mod:`~cultivars.bayes.priors`, or ``None`` for
            unrestricted least squares.

    Raises:
        SpecificationError: If the order, trend or names are malformed.
        DimensionError: If the sample cannot support the specification.

    Attributes:
        _endog: The validated ``(nobs, k)`` panel.
        _names: The resolved variable labels.
        _order: The autoregressive order.
        _trend: The validated trend code.
        _prior: The prior, or ``_NoPrior`` for least squares.

    See Also:
        * :class:`VARResult` -- the record ``fit()`` returns.
        * :class:`VARX` -- the same model with exogenous regressors.
        * :class:`~cultivars.multivariate.large_dim.bayesian.BVAR` -- full
          posterior simulation rather than a shrunk point estimate.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = rng.standard_normal((200, 2))
        >>> res = VAR(y, order=1).fit()
        >>> res.forecast(3).shape
        (3, 2)

        Lag selection runs on the model, before any fit, and the criteria
        agree on white noise:

        >>> selection = VAR(y, order=1).lag_order_selection(max_lags=4)
        >>> selection.selected
        {'aic': 0, 'bic': 0, 'hqic': 0, 'fpe': 0}

        A prior changes the solve and the result records it:

        >>> from cultivars.bayes.priors import MinnesotaPrior
        >>> shrunk = VAR(y, order=1, prior=MinnesotaPrior(tightness=0.1)).fit()
        >>> shrunk.prior_label, shrunk.posterior is not None, bool(shrunk.n_params < 9.0)
        ('minnesota(l1=0.1, l2=0.5, l3=1, l4=100)', True, True)
    """

    __slots__ = ()

    def fit(self) -> VARResult:
        """Estimate the system and return the fitted result.

        Returns:
            A :class:`VARResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((120, 3)), order=2, trend="n").fit()
            >>> res.coefficients.shape, res.deterministic.shape, res.nobs
            ((2, 3, 3), (0, 3), 118)
        """
        return VARResult._from_fit(self._fit_family(), self)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class VARXResult(
    _VectorResult,
    _SummaryMixin,
    _ComparisonMixin,
    _VectorInferenceMixin,
    _VectorPropagationMixin,
):
    r"""A fitted vector autoregression with exogenous regressors.

    The estimated system is

    .. math::

       y_t = D d_t + \sum_{i=1}^{p} A_i y_{t-i} + \sum_{j=0}^{s} B_j x_{t-j}
       + u_t,
       \qquad u_t \sim (0, \Sigma_u),

    with the distributed lag of :math:`x` appended to the design after the
    endogenous lags. Everything the reduced-form surface offers is inherited
    unchanged, because the exogenous block does not enter the moving-average
    representation: ``irf``, ``fevd``, ``historical_decomposition``,
    ``granger_causality``, ``stability_check`` and ``residual_diagnostics``
    read the endogenous dynamics and are therefore the same objects they are
    for a plain VAR on :math:`(A_1, \dots, A_p, \Sigma_u)`.

    :meth:`forecast` is the exception, and it is the only new behaviour in
    the model. A VARX forecast is conditional on a path for :math:`x`, and
    this class has no model of :math:`x` to supply one, so it asks for the
    path and refuses without it.

    Note:
        Granger causality is a test between endogenous variables. An
        exogenous regressor cannot be named as a cause, because its block is
        conditioned on rather than modelled; the coefficient on it and its
        lags, with standard errors, is in :attr:`params`, :attr:`bse` and
        :meth:`equation`.

    Attributes:
        endog: The ``(n, k)`` endogenous sample.
        exog: The ``(n, m)`` exogenous sample, aligned on the same index.
        names: Endogenous labels, in Cholesky order.
        exog_names: Exogenous labels.
        order: Endogenous autoregressive order ``p``.
        exog_order: Exogenous lags ``s`` beyond the contemporaneous term.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        coefficients: ``(p, k, k)`` stack of ``A_1, ..., A_p``.
        exog_coefficients: ``(s + 1, k, m)`` stack of ``B_0, ..., B_s``.
        deterministic: ``(d, k)`` deterministic coefficients.
        sigma_u: Residual covariance with the degrees-of-freedom correction.
        sigma_ml: Residual covariance divided by the effective sample.
        resid: ``(nobs, k)`` residuals over the effective sample.
        fittedvalues: ``(nobs, k)`` one-step conditional means.
        design: ``(nobs, d + kp + m(s + 1))`` regressor matrix as estimated.
        llf: Gaussian log-likelihood.
        nobs: Effective sample size ``n - max(p, s)``.
        n_params: Free parameters, covariance included.

    See Also:
        * :class:`VARX` -- the model whose ``fit()`` returns this record.
        * :class:`VARResult` -- the surface this record inherits.
        * :class:`~cultivars.multivariate.reduced_form.closed_global.GVAR`
          -- consumes :meth:`to_varx` to close conditional units into one
          system.

    References:
        Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
        Analysis*. Springer. Chapter 10.

    Example:
        A bivariate system driven by one exogenous series that hits the
        first equation on impact and the second with a lag. The fit
        recovers both impact matrices, the fitted values are the design
        applied row by row, and the forecast needs the future path:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> A = np.array([[0.5, 0.2], [0.0, 0.4]])
        >>> B0, B1 = np.array([[1.0], [0.0]]), np.array([[0.0], [0.5]])
        >>> x = rng.standard_normal((200, 1))
        >>> y = np.zeros((200, 2))
        >>> for t in range(1, 200):
        ...     y[t] = A @ y[t - 1] + B0 @ x[t] + B1 @ x[t - 1] + 0.3 * rng.standard_normal(2)
        >>> res = VARX(y, x, order=1, exog_order=1, names=("a", "b"), exog_names=("z",)).fit()
        >>> res.exog_coefficients.shape, res.k_exog, res.nobs, res.n_params
        ((2, 2, 1), 1, 199, 13)
        >>> bool(np.abs(res.exog_coefficients[0] - B0).max() < 0.1)
        True
        >>> bool(np.abs(res.exog_coefficients[1] - B1).max() < 0.1)
        True
        >>> fitted = (
        ...     res.deterministic[0]
        ...     + y[:-1] @ res.coefficients[0].T
        ...     + x[1:] @ res.exog_coefficients[0].T
        ...     + x[:-1] @ res.exog_coefficients[1].T
        ... )
        >>> bool(np.allclose(fitted, res.fittedvalues))
        True
        >>> res.forecast(2, exog_future=np.zeros((2, 1))).shape
        (2, 2)
        >>> res.forecast(2)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: a VARX forecast is conditional on the ...
    """

    coefficients: npt.NDArray[np.float64]
    """The ``(p, k, k)`` autoregressive stack ``A_1, ..., A_p`` on the endogenous lags."""

    deterministic: npt.NDArray[np.float64]
    """The ``(d, k)`` deterministic coefficients, one row per term in the order constant, trend."""

    exog: npt.NDArray[np.float64]
    """The ``(n, m)`` exogenous sample as supplied, aligned with :attr:`endog` row for row."""

    exog_names: tuple[str, ...]
    """The exogenous labels, in the column order of :attr:`exog`."""

    exog_order: int
    """Exogenous lags ``s`` beyond the contemporaneous term."""

    exog_coefficients: npt.NDArray[np.float64]
    """The ``(s + 1, k, m)`` distributed-lag stack ``B_0, ..., B_s``; ``B_0`` is the impact."""

    @classmethod
    def _from_fit(
        cls, fit: _ExogenousVectorAutoRegressionFit, model: _ExogenousVectorAutoRegressionModel
    ) -> Self:
        """Assemble the public result from the internal fit and its model.

        Args:
            fit: The arrays ``_fit_family`` produced, with the distributed-lag
                stack split off the trailing design block.
            model: The model that produced it, read for both samples and the
                specification.

        Returns:
            A populated result.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y, x = rng.standard_normal((120, 2)), rng.standard_normal((120, 1))
            >>> model = VARX(y, x, order=1)
            >>> built = VARXResult._from_fit(model._fit_family(), model)
            >>> bool(np.allclose(built.exog_coefficients, model.fit().exog_coefficients))
            True
        """
        return cls(
            endog=model.endog,
            exog=model.exog,
            names=model.names,
            exog_names=model.exog_names,
            order=model.order,
            exog_order=model.exog_order,
            trend=model.trend,
            coefficients=fit.coefficients,
            exog_coefficients=fit.exog_coefficients,
            deterministic=fit.deterministic,
            sigma_u=fit.sigma_u,
            sigma_ml=fit.sigma_ml,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            design=fit.design,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
        )

    def to_varx(self) -> _ConditionalLevels:
        """This unit's equations in levels, as a global system consumes them.

        A repackaging rather than a conversion: a VARX is already written in
        levels, so the only work is splitting the exogenous stack into its
        contemporaneous head and its lagged tail.

        Returns:
            A ``_ConditionalLevels`` with ``phi`` the endogenous stack,
            ``impact`` equal to ``B_0`` and ``exog_lags`` equal to
            ``B_1, ..., B_s``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y, x = rng.standard_normal((120, 2)), rng.standard_normal((120, 1))
            >>> res = VARX(y, x, order=1, exog_order=2).fit()
            >>> levels = res.to_varx()
            >>> levels.impact.shape, levels.exog_lags.shape, levels.exog_names
            ((2, 1), (2, 2, 1), ('x1',))
            >>> bool(np.allclose(levels.impact, res.exog_coefficients[0]))
            True
        """
        return _ConditionalLevels(
            phi=self.coefficients,
            impact=self.exog_coefficients[0],
            exog_lags=self.exog_coefficients[1:],
            deterministic=self.deterministic,
            names=self.names,
            exog_names=self.exog_names,
        )

    @property
    def k_exog(self) -> int:
        """Number of exogenous variables.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y, x = rng.standard_normal((120, 2)), rng.standard_normal((120, 3))
            >>> VARX(y, x, order=1).fit().k_exog
            3
        """
        return len(self.exog_names)

    def _trailing_blocks(self) -> tuple[npt.NDArray[np.float64], ...]:
        """The distributed-lag coefficients, in design order.

        One ``(m, k)`` block per exogenous lag, transposed to the row layout
        of :attr:`deterministic` so the mixin can stack them under the
        endogenous lags.

        Returns:
            ``s + 1`` blocks, contemporaneous term first.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y, x = rng.standard_normal((120, 2)), rng.standard_normal((120, 3))
            >>> res = VARX(y, x, order=1, exog_order=1).fit()
            >>> [block.shape for block in res._trailing_blocks()]
            [(3, 2), (3, 2)]
        """
        return tuple(self.exog_coefficients[j].T for j in range(self.exog_order + 1))

    def _trailing_labels(self) -> tuple[str, ...]:
        """Distributed-lag column names, contemporaneous term first.

        Returns:
            ``m (s + 1)`` labels, the contemporaneous term under the bare
            exogenous name and lag ``j`` under ``"{name}.L{j}"``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y, x = rng.standard_normal((120, 2)), rng.standard_normal((120, 2))
            >>> VARX(y, x, order=1, exog_order=1, exog_names=("r", "q")).fit()._trailing_labels()
            ('r', 'q', 'r.L1', 'q.L1')
        """
        return tuple(
            source if j == 0 else f"{source}.L{j}"
            for j in range(self.exog_order + 1)
            for source in self.exog_names
        )

    def forecast(
        self, steps: int = 1, *, exog_future: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Point forecasts conditional on a future path for the exogenous block.

        Iterates the fitted system forward with future innovations at their
        zero mean, the deterministic terms extrapolated, and the exogenous
        block read from ``exog_future`` for the horizon and from the last
        ``s`` sample rows for the lags that reach back into the sample.

        Args:
            steps: Horizon, at least one.
            exog_future: A ``(steps, m)`` path for the exogenous variables, in
                the column order of :attr:`exog_names`. Required.

        Returns:
            A ``(steps, k)`` array of conditional means.

        Raises:
            SpecificationError: If ``steps`` is not positive, or ``exog_future``
                is omitted.
            DimensionError: If ``exog_future`` has the wrong shape.

        Example:
            The two-step forecast is the fitted equation iterated by hand:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y, x = rng.standard_normal((120, 2)), rng.standard_normal((120, 1))
            >>> res = VARX(y, x, order=1, exog_order=1).fit()
            >>> future = np.array([[0.5], [-0.2]])
            >>> A, B0, B1, c = res.coefficients[0], *res.exog_coefficients, res.deterministic[0]
            >>> h0 = c + A @ y[-1] + B0 @ future[0] + B1 @ x[-1]
            >>> h1 = c + A @ h0 + B0 @ future[1] + B1 @ future[0]
            >>> bool(np.allclose(res.forecast(2, exog_future=future), np.vstack([h0, h1])))
            True
            >>> res.forecast(2, exog_future=np.zeros((3, 1)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.DimensionError: exog_future has 3 rows but the endogenous ...
            >>> res.forecast(2, exog_future=np.zeros((2, 2)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.DimensionError: exog_future has 2 columns but the model was ...
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        if exog_future is None:
            raise SpecificationError(
                "a VARX forecast is conditional on the exogenous path, so it cannot be "
                f"produced from the fitted model alone: pass exog_future with {steps} rows "
                f"and {self.k_exog} columns for {self.exog_names}. This model holds no "
                "process for x and will not invent one, because doing so would turn a "
                "conditional forecast into an unconditional forecast without saying so."
            )
        future = validate_exog_matrix(exog_future, nobs=steps, label="exog_future")
        if future.shape[1] != self.k_exog:
            raise DimensionError(
                f"exog_future has {future.shape[1]} columns but the model was fitted with "
                f"{self.k_exog}."
            )
        k, p, s = self.k_endog, self.order, self.exog_order
        nobs = self.endog.shape[0]
        det = deterministic_columns(self.trend, steps, start=nobs + 1)
        history = [self.endog[nobs - i - 1] for i in range(p)]
        path = np.vstack([self.exog[nobs - s :], future]) if s else future
        out = np.empty((steps, k), dtype=np.float64)
        for h in range(steps):
            point = det[h] @ self.deterministic if self.deterministic.shape[0] else np.zeros(k)
            for i in range(p):
                point = point + self.coefficients[i] @ history[i]
            for j in range(s + 1):
                point = point + self.exog_coefficients[j] @ path[s + h - j]
            out[h] = point
            history = [point, *history[: p - 1]] if p else []
        return out

    def equation(self, name: str) -> dict[str, float]:
        """The coefficients of one equation, keyed by regressor.

        Args:
            name: An endogenous variable.

        Returns:
            Deterministic terms, then endogenous lags, then the distributed lag,
            in the order they enter the design.

        Raises:
            SpecificationError: If the variable is unknown.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y, x = rng.standard_normal((120, 2)), rng.standard_normal((120, 1))
            >>> res = VARX(y, x, order=1, exog_order=1, names=("a", "b"), exog_names=("z",)).fit()
            >>> list(res.equation("b"))
            ['const', 'a.L1', 'b.L1', 'z', 'z.L1']
            >>> res.equation("z")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: unknown variable 'z'; expected one of ...
        """
        if name not in self.names:
            raise SpecificationError(f"unknown variable {name!r}; expected one of {self.names}.")
        row = self.names.index(name)
        out: dict[str, float] = {}
        labels = (["const"] if self.trend in ("c", "ct") else []) + (
            ["trend"] if self.trend == "ct" else []
        )
        for j, label in enumerate(labels):
            out[label] = float(self.deterministic[j, row])
        for lag in range(self.order):
            for j, source in enumerate(self.names):
                out[f"{source}.L{lag + 1}"] = float(self.coefficients[lag][row, j])
        for j in range(self.exog_order + 1):
            for q, source in enumerate(self.exog_names):
                key = source if j == 0 else f"{source}.L{j}"
                out[key] = float(self.exog_coefficients[j][row, q])
        return out

    @property
    def params(self) -> dict[str, float]:
        """Every coefficient, keyed ``"{equation}: {regressor}"``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y, x = rng.standard_normal((120, 2)), rng.standard_normal((120, 1))
            >>> res = VARX(y, x, order=1, names=("a", "b"), exog_names=("z",)).fit()
            >>> list(res.params)[:4], len(res.params)
            (['a: const', 'a: a.L1', 'a: b.L1', 'a: z'], 8)
            >>> res.params.keys() == res.bse.keys()
            True
        """
        return {
            f"{equation}: {name}": value
            for equation in self.names
            for name, value in self.equation(equation).items()
        }

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y, x = rng.standard_normal((120, 2)), rng.standard_normal((120, 1))
            >>> VARX(y, x, order=2, exog_order=1).fit()._comparison_label()
            'VARX(2, 1)'
            >>> VARX(y, x, order=1, trend="n").fit()._comparison_label()
            'VARX(1, 0, trend=n)'
        """
        tail = "" if self.trend == "c" else f", trend={self.trend}"
        return f"VARX({self.order}, {self.exog_order}{tail})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        The VAR table with the exogenous count in the metadata and two more
        notes: that stability, impulse responses and the variance
        decomposition read the endogenous block only, and that forecasts are
        conditional on a supplied exogenous path.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y, x = rng.standard_normal((120, 2)), rng.standard_normal((120, 1))
            >>> table = VARX(y, x, order=1, exog_order=1).fit()._summary_table()
            >>> table.title, len(table.rows), dict(table.metadata)["Exogenous"]
            ('VARX(1, 1) Results', 10, '1')
            >>> len(table.notes), table.notes[2][:23]
            (4, 'Forecasts are condition')
        """
        criteria = self.information_criteria
        stability = self.stability_check()
        notes = [
            f"Stable: {self.is_stable}   max |companion root| = {stability.max_modulus:.4f}",
            "Stability, impulse responses, and the variance decomposition read the "
            "endogenous block only. The exogenous regressors are conditioned on rather "
            "than shocked, so they do not enter the moving-average representation.",
            "Forecasts are conditional: forecast() requires the future exogenous path and "
            "will not extrapolate it.",
            _CHOLESKY_NOTE,
        ]
        if not self.is_stable:
            notes.insert(0, _UNSTABLE_NOTE)
        return SummaryTable(
            title=f"VARX({self.order}, {self.exog_order}) Results",
            metadata=(
                ("Model", f"VARX({self.order}, {self.exog_order})"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Endogenous", f"{self.k_endog}"),
                ("AIC", f"{criteria.aic:.3f}"),
                ("Exogenous", f"{self.k_exog}"),
                ("BIC", f"{criteria.bic:.3f}"),
                ("Trend", self.trend),
                ("HQIC", f"{criteria.hqic:.3f}"),
                ("Observations", f"{self.nobs}"),
            ),
            columns=self._coefficient_columns(),
            rows=self._coefficient_rows(),
            notes=tuple(notes),
        )


class VARX(_ExogenousVectorAutoRegressionModel[VARXResult]):
    """Vector autoregression with exogenous regressors, estimated by least squares.

    The design gains one ``(m,)`` block per exogenous lag, placed after the
    endogenous lags so that every offset the base computes stays correct
    without an override. The constructor is inherited from
    ``_ExogenousVectorAutoRegressionModel``; the exogenous sample must be
    aligned with the endogenous one row for row and the two label sets may
    not overlap. No prior is accepted here.

    Args:
        endog: The ``(nobs, k)`` endogenous panel.
        exog: The ``(nobs, m)`` exogenous block, aligned on the same index.
        order: Endogenous autoregressive order ``p``.
        exog_order: Exogenous lags ``s`` beyond the contemporaneous term.
            Defaults to zero.
        trend: ``"n"``, ``"c"`` (default) or ``"ct"``.
        names: Endogenous labels. Defaults to ``y1 ... yk``.
        exog_names: Exogenous labels. Defaults to ``x1 ... xm``.

    Raises:
        SpecificationError: If ``exog_order`` is malformed, the labels are
            malformed, or an endogenous and an exogenous label collide.
        DimensionError: If the two samples are not aligned or the sample is
            too short for the specification.

    Attributes:
        _endog: The validated ``(nobs, k)`` endogenous panel.
        _exog: The validated ``(nobs, m)`` exogenous block.
        _names: The resolved endogenous labels.
        _exog_names: The resolved exogenous labels.
        _order: The endogenous order.
        _exog_order: The exogenous order.
        _trend: The validated trend code.
        _prior: Always ``_NoPrior``.

    See Also:
        * :class:`VARXResult` -- the record ``fit()`` returns.
        * :class:`VAR` -- the model without the exogenous block.
        * :class:`~cultivars.multivariate.reduced_form.error_correction.VECMX`
          -- the exogenous block under a cointegration restriction.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal((200, 1))
        >>> y = np.column_stack([x[:, 0] + rng.standard_normal(200), rng.standard_normal(200)])
        >>> res = VARX(y, x, order=1, exog_order=1).fit()
        >>> res.forecast(3, exog_future=np.zeros((3, 1))).shape
        (3, 2)
        >>> VARX(y, x, order=1, names=("a", "b"), exog_names=("a",))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: names and exog_names must not overlap, ...
    """

    __slots__ = ()

    def fit(self) -> VARXResult:
        """Estimate the system and return the fitted result.

        Returns:
            A :class:`VARXResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y, x = rng.standard_normal((120, 2)), rng.standard_normal((120, 2))
            >>> res = VARX(y, x, order=2, exog_order=1).fit()
            >>> res.coefficients.shape, res.exog_coefficients.shape, res.nobs
            ((2, 2, 2), (2, 2, 2), 118)
        """
        return VARXResult._from_fit(self._fit_family(), self)
