# filepath: /src/cultivars/multivariate/reduced_form/moving_average.py
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
r"""The vector autoregressive moving average, estimated without a likelihood search.

A VARMA(p, q),

.. math::

   y_t = D d_t + \sum_{i=1}^{p} A_i\, y_{t-i} + u_t + \sum_{j=1}^{q} M_j\, u_{t-j},

buys parsimony with a harder estimation problem: the moving-average innovations
are unobserved, so least squares cannot be applied directly, and the Gaussian
likelihood is a surface over :math:`k^2 (p + q)` parameters that is neither
cheap nor -- for an unrestricted model -- uniquely maximized. This module takes
the Hannan-Rissanen route instead: recover the innovations from a long
autoregression of order :math:`h \sim \sqrt{T}`, then estimate the VARMA by
regressions in which those estimated innovations stand in for the true ones,
and once more after re-deriving them through the model. Three passes, every
one of them a least-squares solve, no numerical optimizer anywhere.

Two commitments shape the surface. First, the estimator's conditioning is
stated rather than hidden. The standard errors treat the lagged innovations
in the design as observed regressors, so they are conditional in the
Hannan-Rissanen sense and modestly understate uncertainty; the summary says
so on every result. The third pass runs only when the second-stage
moving-average block is invertible -- re-deriving innovations through a
non-invertible block would amplify the approximation error instead of
reducing it -- and the result records whether it ran. Second, the
identification caveat is structural rather than numerical and is repeated on
the result: distinct :math:`(A, M)` pairs can generate identical second
moments, and the standard resolution -- echelon-form restrictions indexed by
Kronecker indices -- is deliberately not imposed here. What survives the
ambiguity is exactly what most users come for: for a stable, invertible
representation the moving-average matrices
:math:`\Psi_h = M_h + \sum_i A_i \Psi_{h-i}`, and with them the forecasts,
impulse responses, and variance decompositions, are invariant across
observationally equivalent parameterizations. Overriding the moving-average
recursion is the single change that makes the inherited propagation surface
correct for a VARMA, because every member of that surface reads
:math:`\Psi_h` rather than the autoregressive companion.

Layout. :class:`VARMA` extends ``_VectorAutoRegressionModel`` in
``_internals`` with the moving-average order, a long-order rule and a
three-pass ``fit()`` built from the base's ``_least_squares`` and
``_gaussian_moments``: ``_stage_regression`` lays out
``[deterministic | own lags | innovation lags]`` with ``lag_matrix`` and
``deterministic_columns`` from ``_core``, ``_unpack`` splits a pass's
coefficients into the three blocks, and ``_recursive_innovations`` rebuilds
the innovations for the refinement; ``lag_order_selection`` is refused
because ``(p, q)`` candidates are not nested designs. :class:`VARMAResult`
is a full ``_VectorResult`` with the summary, comparison, inference and
propagation mixins, overriding ``ma_representation`` and ``forecast`` and
adding the invertibility check; the identification and conditioning notes
it prints are ``_VARMA_IDENTIFICATION_NOTE`` and ``_HR_CONDITIONAL_NOTE`` in
``_core``. The pure autoregression is
:mod:`~cultivars.multivariate.reduced_form.vector_autoregression`; the
scalar model with exact maximum likelihood is
:mod:`~cultivars.univariate.box_jenkins`.

References:
    Hannan, E. J., & Rissanen, J. (1982). Recursive estimation of mixed
    autoregressive-moving average order. *Biometrika*, 69(1), 81-94.

    Hannan, E. J., & Kavalieris, L. (1984). Multivariate linear time series
    models. *Advances in Applied Probability*, 16(3), 492-561.

    Dufour, J.-M., & Jouini, T. (2005). Asymptotic distribution of a simple
    linear estimator for VARMA models in echelon form. In *Statistical
    Modeling and Analysis for Complex Data Problems* (pp. 209-240). Springer.

    Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
    Analysis*. Springer. Chapters 11-12.

Example:
    A bivariate VARMA(1, 1) whose moving-average block carries a spillover
    from the first innovation into the second equation. The three passes
    recover both stacks, a VAR(1) on the same data cannot represent the
    moving-average part and leaves autocorrelated residuals, and the
    one-step forecast carries the last innovation across the sample boundary:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> A = np.array([[0.5, 0.1], [0.0, 0.4]])
    >>> M = np.array([[0.4, 0.0], [0.2, 0.3]])
    >>> u = rng.standard_normal((402, 2))
    >>> y = np.zeros((402, 2))
    >>> for t in range(1, 402):
    ...     y[t] = A @ y[t - 1] + u[t] + M @ u[t - 1]
    >>> res = VARMA(y[2:], order=(1, 1)).fit()
    >>> bool(np.abs(res.coefficients[0] - A).max() < 0.15)
    True
    >>> bool(np.abs(res.ma_coefficients[0] - M).max() < 0.15)
    True
    >>> res.refined, res.is_stable, res.is_invertible
    (True, True, True)
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> plain = VAR(y[2:], order=1).fit()
    >>> lag1 = lambda e: np.corrcoef(e[1:, 0], e[:-1, 0])[0, 1]
    >>> bool(abs(lag1(plain.resid)) > 2 * abs(lag1(res.resid)))
    True
    >>> res.forecast(2).shape
    (2, 2)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
import numpy.typing as npt

from ...engine._core import (
    _CHOLESKY_NOTE,
    _HR_CONDITIONAL_NOTE,
    _UNSTABLE_NOTE,
    _VARMA_IDENTIFICATION_NOTE,
    SummaryTable,
    Trend,
    deterministic_columns,
    lag_matrix,
    validate_order_tuple,
)
from ...engine._internals import (
    _ComparisonMixin,
    _StabilityAssessment,
    _SummaryMixin,
    _VectorAutoRegressionModel,
    _VectorInferenceMixin,
    _VectorMoments,
    _VectorPropagationMixin,
    _VectorResult,
)
from ...exceptions import SpecificationError

__all__ = ["VARMA", "VARMAResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class VARMAResult(
    _VectorResult,
    _SummaryMixin,
    _ComparisonMixin,
    _VectorInferenceMixin,
    _VectorPropagationMixin,
):
    r"""A fitted vector autoregressive moving average.

    The system

    .. math::

       y_t = D d_t + \sum_{i=1}^{p} A_i\, y_{t-i} + u_t + \sum_{j=1}^{q} M_j\, u_{t-j},
       \qquad u_t \sim (0, \Sigma_u),

    estimated by the three Hannan-Rissanen passes: a long autoregression whose
    residuals stand in for the unobserved :math:`u_t`, a regression of
    :math:`y_t` on its own lags and on lags of those residuals, and -- when the
    second-stage moving-average block is invertible -- a re-derivation of the
    innovations through the model and one more regression. The record holds
    the final pass, with the regressor matrix carrying lagged innovations in
    its trailing columns.

    The propagation surface is inherited with one override that changes
    everything downstream: :meth:`ma_representation` runs the VARMA recursion
    :math:`\Psi_h = M_h + \sum_i A_i \Psi_{h-i}` rather than powering the
    autoregressive companion, and since the impulse responses, both variance
    decompositions, and the historical decomposition all read the
    moving-average matrices rather than the companion, correcting the one
    method corrects them all. :meth:`forecast` is the other override, because
    a VARMA forecast carries the last ``q`` innovations across the sample
    boundary.

    Note:
        Two things the inherited surface says that need reading with care.
        The standard errors treat the lagged innovations in the design as
        observed regressors; they are conditional in the Hannan-Rissanen
        sense and modestly understate uncertainty, and the summary says so.
        And an unrestricted VARMA is not globally identified -- distinct
        :math:`(A, M)` pairs generate the same second moments, and the
        echelon-form restrictions that resolve this are not imposed -- so the
        coefficient tables describe one representative of an equivalence
        class, while the moving-average matrices, forecasts, impulse responses
        and variance decompositions of a stable, invertible representative are
        invariant across it. Fitted to a non-invertible truth, the estimator
        returns the invertible representative with the same autocovariances,
        which is the right answer to the question least squares can ask.

    Attributes:
        endog: The sample.
        names: Variable labels, in Cholesky order.
        order: Autoregressive order ``p``.
        ma_order: Moving-average order ``q``.
        long_order: Order of the stage-one autoregression that recovered the
            innovations.
        refined: Whether the third pass ran. ``False`` means the second-stage
            moving-average block was not invertible, so re-deriving residuals
            through it would have amplified rather than reduced the
            approximation error, and the second-stage estimate was kept.
        trend: Deterministic specification.
        coefficients: ``(p, k, k)`` stack of ``A_1, ..., A_p``.
        ma_coefficients: ``(q, k, k)`` stack of ``M_1, ..., M_q``.
        deterministic: Deterministic coefficients, one row per term.
        sigma_u: Residual covariance with the degrees-of-freedom correction.
        sigma_ml: Residual covariance divided by the effective sample.
        resid: One-step innovations over the effective sample.
        fittedvalues: One-step conditional means over the effective sample.
        design: The regressor matrix of the final pass.
        llf: Gaussian log-likelihood of the final pass.
        nobs: Effective sample size.
        n_params: Free parameters, covariance included.

    See Also:
        * :class:`VARMA` -- the model whose ``fit()`` returns this record.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
          -- the ``q = 0`` case, estimated in one pass with exact least
          squares.
        * :class:`~cultivars.univariate.box_jenkins.ARMAResult` -- the scalar
          counterpart, estimated by exact maximum likelihood.

    References:
        Hannan, E. J., & Rissanen, J. (1982). Recursive estimation of mixed
        autoregressive-moving average order. *Biometrika*, 69(1), 81-94.

        Dufour, J.-M., & Jouini, T. (2005). Asymptotic distribution of a
        simple linear estimator for VARMA models in echelon form. In
        *Statistical Modeling and Analysis for Complex Data Problems*
        (pp. 209-240). Springer.

        Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
        Analysis*. Springer. Chapters 11-12.

    Example:
        A bivariate VARMA(1, 1) with a diagonal-dominant autoregression and a
        moving-average block that spills from the first innovation into the
        second equation. The three passes recover both stacks, the long
        autoregression sits at the square root of the sample, the refinement
        ran, and the moving-average matrices start from the identity with
        :math:`\Psi_1 = A_1 + M_1`:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> A = np.array([[0.5, 0.1], [0.0, 0.4]])
        >>> M = np.array([[0.4, 0.0], [0.2, 0.3]])
        >>> u = rng.standard_normal((402, 2))
        >>> y = np.zeros((402, 2))
        >>> for t in range(1, 402):
        ...     y[t] = A @ y[t - 1] + u[t] + M @ u[t - 1]
        >>> res = VARMA(y[2:], order=(1, 1)).fit()
        >>> bool(np.abs(res.coefficients[0] - A).max() < 0.15)
        True
        >>> bool(np.abs(res.ma_coefficients[0] - M).max() < 0.15)
        True
        >>> res.long_order, res.refined, res.is_stable, res.is_invertible
        (20, True, True, True)
        >>> psi = res.ma_representation(2)
        >>> bool(np.allclose(psi[1], res.coefficients[0] + res.ma_coefficients[0]))
        True
        >>> res.forecast(3).shape, res.irf(4).shape
        ((3, 2), (5, 2, 2))
    """

    coefficients: npt.NDArray[np.float64]
    """The ``(p, k, k)`` autoregressive matrices of the final pass."""

    deterministic: npt.NDArray[np.float64]
    """The ``(d, k)`` deterministic coefficients, one row per term."""

    ma_coefficients: npt.NDArray[np.float64]
    """The ``(q, k, k)`` moving-average matrices of the final pass."""

    long_order: int
    """Order of the stage-one autoregression whose residuals proxied the innovations."""

    refined: bool
    """Whether the third pass ran; ``False`` keeps the second-stage estimate."""

    @property
    def ma_order(self) -> int:
        """Moving-average order ``q``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> VARMA(y, order=(1, 2)).fit().ma_order
            2
        """
        return int(self.ma_coefficients.shape[0])

    def _trailing_blocks(self) -> tuple[npt.NDArray[np.float64], ...]:
        """The moving-average coefficients, in design order.

        Returns:
            One ``(k, k)`` block per innovation lag, each transposed so its
            rows match the lagged-innovation columns at the end of
            :attr:`design`.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> res = VARMA(y, order=(1, 2)).fit()
            >>> [block.shape for block in res._trailing_blocks()]
            [(2, 2), (2, 2)]
            >>> bool(np.allclose(res._trailing_blocks()[1], res.ma_coefficients[1].T))
            True
        """
        return tuple(self.ma_coefficients[j].T for j in range(self.ma_order))

    def _trailing_labels(self) -> tuple[str, ...]:
        """Moving-average column names, innovation lag by innovation lag.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> VARMA(y, order=(1, 2), names=["a", "b"]).fit()._trailing_labels()
            ('u[a].L1', 'u[b].L1', 'u[a].L2', 'u[b].L2')
        """
        return tuple(f"u[{source}].L{j + 1}" for j in range(self.ma_order) for source in self.names)

    def ma_representation(self, horizon: int = 20) -> npt.NDArray[np.float64]:
        r"""Moving-average matrices ``Psi_0, ..., Psi_horizon`` of the VARMA.

        Computed by the recursion

        .. math::

           \Psi_h = M_h + \sum_{i=1}^{\min(h, p)} A_i\, \Psi_{h-i},
           \qquad M_h = 0 \;\text{ for } h > q,

        which is the representation the impulse responses, variance
        decompositions, and historical decomposition inherited from the
        propagation surface all consume -- overriding this one method is what
        makes every one of them correct for a VARMA.

        Args:
            horizon: Largest lead to return.

        Returns:
            An array of shape ``(horizon + 1, k, k)``; ``Psi_0`` is the
            identity.

        Raises:
            SpecificationError: If ``horizon`` is negative.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> u = rng.standard_normal((302, 2))
            >>> y = np.zeros((302, 2))
            >>> for t in range(1, 302):
            ...     y[t] = 0.5 * y[t - 1] + u[t] + 0.4 * u[t - 1]
            >>> res = VARMA(y[2:], order=(1, 1)).fit()
            >>> psi = res.ma_representation(3)
            >>> psi.shape, bool(np.allclose(psi[0], np.eye(2)))
            ((4, 2, 2), True)
            >>> bool(np.allclose(psi[2], res.coefficients[0] @ psi[1]))
            True
            >>> res.ma_representation(-1)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: horizon must be non-negative; got -1.
        """
        if horizon < 0:
            raise SpecificationError(f"horizon must be non-negative; got {horizon}.")
        k, p, q = self.k_endog, self.order, self.ma_order
        psi = np.zeros((horizon + 1, k, k), dtype=np.float64)
        psi[0] = np.eye(k)
        for h in range(1, horizon + 1):
            acc = self.ma_coefficients[h - 1].copy() if h <= q else np.zeros((k, k))
            for i in range(1, min(h, p) + 1):
                acc += self.coefficients[i - 1] @ psi[h - i]
            psi[h] = acc
        return psi

    def invertibility_check(self) -> _StabilityAssessment:
        """Eigenvalue verdict for the moving-average companion matrix.

        Returns:
            The :class:`_StabilityAssessment`; the moving-average polynomial is
            invertible exactly when every root lies inside the unit circle.
            Invertibility is what makes the innovations recoverable from the
            observed history, so a non-invertible estimate means the residuals
            -- and everything computed from them -- identify a different,
            observationally equivalent representation.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> u = rng.standard_normal((302, 2))
            >>> y = np.zeros((302, 2))
            >>> for t in range(1, 302):
            ...     y[t] = 0.5 * y[t - 1] + u[t] + 0.4 * u[t - 1]
            >>> check = VARMA(y[2:], order=(1, 1)).fit().invertibility_check()
            >>> check.is_stable, bool(check.max_modulus < 0.6)
            (True, True)
        """
        return _StabilityAssessment.assess_stability(self.ma_coefficients)

    @property
    def is_invertible(self) -> bool:
        """Whether every moving-average root lies inside the unit circle.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> u = rng.standard_normal((302, 2))
            >>> y = np.zeros((302, 2))
            >>> for t in range(1, 302):
            ...     y[t] = 0.5 * y[t - 1] + u[t] + 0.4 * u[t - 1]
            >>> VARMA(y[2:], order=(1, 1)).fit().is_invertible
            True
        """
        return self.invertibility_check().is_stable

    def forecast(self, steps: int = 1) -> npt.NDArray[np.float64]:
        r"""Deterministic multi-step forecasts from the end of the sample.

        The moving-average block contributes only while a forecast date is
        within ``q`` periods of the sample: the innovations after the sample
        end are set to their zero mean, and the last ``q`` estimated
        innovations decay out of the forecast one step at a time,

        .. math::

           \hat y_{T+h} = D d_{T+h} + \sum_{i=1}^{p} A_i\, \hat y_{T+h-i}
           + \sum_{j=h+1}^{q} M_j\, \hat u_{T+h-j}.

        Beyond ``q`` steps the iteration is exactly the autoregressive one.

        Args:
            steps: Forecast horizon, at least one.

        Returns:
            An array of shape ``(steps, k)``.

        Raises:
            SpecificationError: If ``steps`` is less than one.

        Example:
            The first step carries the last innovation through ``M_1``; the
            second is the pure autoregressive iteration of the first:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> u = rng.standard_normal((302, 2))
            >>> y = np.zeros((302, 2))
            >>> for t in range(1, 302):
            ...     y[t] = 0.5 * y[t - 1] + u[t] + 0.4 * u[t - 1]
            >>> res = VARMA(y[2:], order=(1, 1)).fit()
            >>> path = res.forecast(2)
            >>> first = res.deterministic[0] + res.coefficients[0] @ y[-1]
            >>> first = first + res.ma_coefficients[0] @ res.resid[-1]
            >>> bool(np.allclose(path[0], first))
            True
            >>> bool(np.allclose(path[1], res.deterministic[0] + res.coefficients[0] @ path[0]))
            True
            >>> res.forecast(0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: steps must be at least 1; got 0.
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        k, p, q = self.k_endog, self.order, self.ma_order
        n = self.endog.shape[0]
        det = deterministic_columns(self.trend, steps, start=n + 1)
        history = [self.endog[n - i - 1] for i in range(p)]
        tail = [
            self.resid[-m - 1] if m < self.resid.shape[0] else np.zeros(k, dtype=np.float64)
            for m in range(q)
        ]
        out = np.empty((steps, k), dtype=np.float64)
        for h in range(steps):
            point = det[h] @ self.deterministic if self.deterministic.shape[0] else np.zeros(k)
            for i in range(p):
                point = point + self.coefficients[i] @ history[i]
            for j in range(h + 1, q + 1):
                point = point + self.ma_coefficients[j - 1] @ tail[j - h - 1]
            out[h] = point
            history = [point, *history[: p - 1]] if p else []
        return out

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table.

        Returns:
            ``"VARMA(p, q)"``, with ``, trend=<trend>`` appended when the
            trend is not the default constant.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> VARMA(y, order=(1, 1)).fit()._comparison_label()
            'VARMA(1, 1)'
            >>> VARMA(y, order=(2, 1), trend="n").fit()._comparison_label()
            'VARMA(2, 1, trend=n)'
        """
        tail = "" if self.trend == "c" else f", trend={self.trend}"
        return f"VARMA({self.order}, {self.ma_order}{tail})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        The coefficient table with the autoregressive block first and the
        lagged-innovation block last; metadata pairing the fit statistics with
        the long autoregressive order and the refinement flag; notes on
        stability, invertibility, the conditional standard errors, the
        identification caveat, and the Cholesky ordering, with a skipped
        refinement explained and an unstable system flagged first.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> table = VARMA(y, order=(1, 1)).fit()._summary_table()
            >>> table.title, dict(table.metadata)["Long AR order"], dict(table.metadata)["Refined"]
            ('VARMA(1, 1) Results', '15', 'True')
            >>> [row[0] for row in table.rows][:5]
            ['y1: const', 'y1: y1.L1', 'y1: y2.L1', 'y1: u[y1].L1', 'y1: u[y2].L1']
            >>> table.notes[1][:11]
            'Invertible:'
        """
        criteria = self.information_criteria
        stability = self.stability_check()
        invertibility = self.invertibility_check()
        notes = [
            f"Stable: {self.is_stable}   max |AR companion root| = {stability.max_modulus:.4f}",
            f"Invertible: {self.is_invertible}   max |MA companion root| = "
            f"{invertibility.max_modulus:.4f}",
            _HR_CONDITIONAL_NOTE,
            _VARMA_IDENTIFICATION_NOTE,
            _CHOLESKY_NOTE,
        ]
        if not self.refined:
            notes.insert(
                2,
                "The refinement pass was skipped: the second-stage moving-average "
                "block was not invertible, so residuals re-derived through it "
                "would diverge. The second-stage estimate is reported.",
            )
        if not self.is_stable:
            notes.insert(0, _UNSTABLE_NOTE)
        return SummaryTable(
            title=f"VARMA({self.order}, {self.ma_order}) Results",
            metadata=(
                ("Model", f"VARMA({self.order}, {self.ma_order})"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Variables", f"{self.k_endog}"),
                ("AIC", f"{criteria.aic:.3f}"),
                ("Long AR order", f"{self.long_order}"),
                ("BIC", f"{criteria.bic:.3f}"),
                ("Refined", f"{self.refined}"),
                ("HQIC", f"{criteria.hqic:.3f}"),
                ("Trend", self.trend),
                ("Observations", f"{self.nobs}"),
            ),
            columns=self._coefficient_columns(),
            rows=self._coefficient_rows(),
            notes=tuple(notes),
        )


class VARMA(_VectorAutoRegressionModel[VARMAResult]):
    r"""Vector autoregressive moving average, estimated by Hannan-Rissanen.

    Three passes, each a least-squares solve. The first fits a long
    autoregression -- order growing with the square root of the sample -- whose
    residuals estimate the unobservable innovations. The second regresses the
    sample on its own lags and on lags of those estimated innovations, which is
    the VARMA equation with a generated regressor standing in for the truth.
    The third re-derives the innovations recursively from the second-stage
    parameters, so they are consistent with the model rather than with the long
    autoregression, and estimates once more. When the second-stage
    moving-average block is not invertible the recursion would diverge instead
    of converge, so the third pass is skipped and the result says so.

    No numerical optimizer, no likelihood surface, no starting values: the
    estimator is deterministic and fast, at the cost of the (small, and
    documented) efficiency loss relative to exact maximum likelihood, and of
    standard errors that condition on the estimated innovations. Lag-order
    selection is refused, because :math:`(p, q)` pairs are not nested
    least-squares designs; fit the candidates and rank them through the
    comparison surface.

    Attributes:
        _endog: The validated ``(nobs, k)`` panel.
        _names: Variable labels.
        _order: Autoregressive order ``p``.
        _trend: Deterministic terms.
        _prior: The base slot for a shrinkage prior, unused by this family.
        _ma_order: Moving-average order ``q``.

    See Also:
        * :class:`VARMAResult` -- the record ``fit()`` returns.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
          -- the ``q = 0`` model, one pass instead of three.
        * :class:`~cultivars.univariate.box_jenkins.ARMA` -- the scalar counterpart,
          estimated by exact maximum likelihood.

    References:
        Hannan, E. J., & Rissanen, J. (1982). Recursive estimation of mixed
        autoregressive-moving average order. *Biometrika*, 69(1), 81-94.

        Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
        Analysis*. Springer. Chapters 11-12.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> u = rng.standard_normal((202, 2))
        >>> y = np.zeros((202, 2))
        >>> for t in range(1, 202):
        ...     y[t] = 0.5 * y[t - 1] + u[t] + 0.4 * u[t - 1]
        >>> res = VARMA(y[2:], order=(1, 1)).fit()
        >>> res.forecast(3).shape
        (3, 2)
    """

    __slots__ = ("_ma_order",)

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: tuple[int, int],
        trend: Trend = "c",
        names: Sequence[str] | None = None,
    ) -> None:
        """Validate the sample and the specification.

        Args:
            endog: The ``(nobs, k)`` panel, time down the rows.
            order: The pair ``(p, q)``. ``q`` must be at least one -- with no
                moving-average block a plain
                :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
                is the right model, and it estimates in one pass instead of
                three.
            trend: Deterministic terms.
            names: Variable labels. Defaults to ``y1 ... yk``.

        Raises:
            SpecificationError: If the order pair is malformed or ``q`` is
                zero.
            DimensionError: If the sample cannot support the specification,
                including the long autoregression the first stage needs.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> model = VARMA(y, order=(2, 1), names=["a", "b"])
            >>> model.order, model.ma_order, model.n_regressors
            (2, 1, 7)
            >>> VARMA(y, order=(1, 0))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: q must be at least 1; a VARMA with no ...
            >>> VARMA(y, order=(1,))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: order must have 2 elements ('p', 'q'); ...
        """
        p, q = validate_order_tuple(order, ("p", "q"))
        if q < 1:
            raise SpecificationError(
                "q must be at least 1; a VARMA with no moving-average block is a "
                "VAR, which estimates in one pass instead of three."
            )
        self._ma_order = int(q)
        super().__init__(endog, order=p, trend=trend, names=names)

    @property
    def ma_order(self) -> int:
        """Moving-average order ``q``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> VARMA(y, order=(1, 3)).ma_order
            3
        """
        return self._ma_order

    @property
    def n_regressors(self) -> int:
        """Regressors per equation, including the innovation lags.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 3))
            >>> VARMA(y, order=(1, 2)).n_regressors
            10
        """
        return super().n_regressors + self.k_endog * self._ma_order

    def _long_order(self) -> int:
        """Order of the stage-one autoregression.

        Grows like the square root of the sample, which dominates any fixed
        ``(p, q)`` asymptotically and keeps the residuals consistent for the
        innovations, capped at what the sample can identify so that the first
        stage never fails on a design its own target cannot support.

        Returns:
            ``max(p + q, ceil(sqrt(nobs)))``, capped by the largest order the
            sample's rows and width allow, and at least one.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> VARMA(rng.standard_normal((400, 2)), order=(1, 1))._long_order()
            20
            >>> VARMA(rng.standard_normal((30, 2)), order=(1, 1))._long_order()
            6
        """
        nobs = int(self._endog.shape[0])
        wanted = max(self._order + self._ma_order, int(np.ceil(np.sqrt(nobs))))
        cap = (nobs - self._n_deterministic_columns - 1) // (self.k_endog + 1)
        return max(min(wanted, cap), 1)

    def _burn_for(self, order: int) -> int:
        """Leading observations lost across all three passes.

        The second pass needs the long autoregression's residuals plus ``q``
        lags of them, so the burn is ``long_order + q`` whenever that exceeds
        the autoregressive order; it is what the base class's minimum-sample
        check reads.

        Args:
            order: Autoregressive order under consideration.

        Returns:
            ``max(order, long_order + q)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> VARMA(rng.standard_normal((400, 2)), order=(1, 1))._burn_for(1)
            21
        """
        return max(order, self._long_order() + self._ma_order)

    def _max_supported_lags(self) -> int:
        """Largest autoregressive order the sample can identify.

        Returns:
            The order at which the design would still have more rows than
            columns once the deterministic terms and the ``k * q`` innovation
            columns are counted.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> VARMA(rng.standard_normal((400, 2)), order=(1, 1))._max_supported_lags()
            132
        """
        free = (
            int(self._endog.shape[0])
            - self._n_deterministic_columns
            - self.k_endog * self._ma_order
            - 1
        )
        return max(free // (self.k_endog + 1), 0)

    def lag_order_selection(self, max_lags: int | None = None) -> NoReturn:
        """Unavailable: VARMA orders are not nested least-squares candidates.

        The autoregressive scan the base class runs compares pure lag
        structures on one design family; a VARMA's ``(p, q)`` trades the two
        polynomials off against each other, and scoring that honestly means
        fitting each candidate in full. Fit the specifications under
        consideration and rank them with ``information_criteria`` -- the
        comparison surface exists precisely for this.

        Args:
            max_lags: Ignored.

        Raises:
            SpecificationError: Always.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> VARMA(y, order=(1, 1)).lag_order_selection()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: lag-order selection over a single ...
        """
        raise SpecificationError(
            "lag-order selection over a single autoregressive scan is not "
            "meaningful for a VARMA: (p, q) candidates are not nested least-"
            "squares designs. Fit the candidate specifications and compare "
            "their information criteria."
        )

    def _innovation_block(
        self, innovations: npt.NDArray[np.float64], start: int
    ) -> npt.NDArray[np.float64]:
        """Stack ``u_{t-1}, ..., u_{t-q}`` for rows ``start`` onward.

        Args:
            innovations: An ``(nobs, k)`` innovation series aligned with the
                sample.
            start: First time index to build a row for; must be at least
                ``q``.

        Returns:
            An ``(nobs - start, k * q)`` block, lag-major.

        Example:
            >>> import numpy as np
            >>> model = VARMA(np.random.default_rng(0).standard_normal((200, 2)), order=(1, 2))
            >>> innovations = np.arange(400.0).reshape(200, 2)
            >>> block = model._innovation_block(innovations, 10)
            >>> block.shape, block[0].tolist()
            ((190, 4), [18.0, 19.0, 16.0, 17.0])
        """
        nobs = innovations.shape[0]
        return np.column_stack(
            [innovations[start - j : nobs - j] for j in range(1, self._ma_order + 1)]
        )

    def _stage_regression(
        self, innovations: npt.NDArray[np.float64], start: int
    ) -> tuple[_VectorMoments, npt.NDArray[np.float64]]:
        """Regress the sample on its lags and on lagged innovations.

        Args:
            innovations: Estimated innovations aligned with the sample; rows
                before their first valid index must already be zero-filled.
            start: First time index with a complete regressor history.

        Returns:
            The Gaussian moments of the regression and the design behind them.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal((200, 2))
            >>> model = VARMA(y, order=(1, 1))
            >>> moments, design = model._stage_regression(rng.standard_normal((200, 2)), 5)
            >>> design.shape, moments.resid.shape, moments.nobs
            ((195, 5), (195, 2), 195)
        """
        nobs = self._endog.shape[0]
        effective = nobs - start
        det = deterministic_columns(self._trend, effective, start=start + 1)
        design = np.column_stack(
            [
                det,
                lag_matrix(self._endog, self._order, start=start),
                self._innovation_block(innovations, start),
            ]
        )
        return self._gaussian_moments(self._endog[start:], design), design

    def _unpack(
        self, moments: _VectorMoments
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Split a stage's coefficients into deterministic, AR, and MA blocks.

        Args:
            moments: The Gaussian moments of a stage regression whose design
                was laid out by :meth:`_stage_regression`.

        Returns:
            The ``(d, k)`` deterministic rows, the ``(p, k, k)``
            autoregressive stack, and the ``(q, k, k)`` moving-average stack.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal((200, 2))
            >>> model = VARMA(y, order=(2, 1))
            >>> moments, _ = model._stage_regression(rng.standard_normal((200, 2)), 5)
            >>> [block.shape for block in model._unpack(moments)]
            [(1, 2), (2, 2, 2), (1, 2, 2)]
        """
        k, p, q = self.k_endog, self._order, self._ma_order
        head = self._n_deterministic_columns
        deterministic = moments.coef[:head]
        ar = self._lag_blocks(moments.coef)
        ma = np.stack(
            [moments.coef[head + k * p + j * k : head + k * p + (j + 1) * k, :].T for j in range(q)]
        )
        return deterministic, ar, ma

    def _recursive_innovations(
        self,
        deterministic: npt.NDArray[np.float64],
        ar: npt.NDArray[np.float64],
        ma: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Innovations implied by a parameter draw, computed forward in time.

        Pre-sample innovations are set to zero; under an invertible
        moving-average block their influence decays geometrically, which is
        exactly why the caller checks invertibility before trusting this.

        Args:
            deterministic: Deterministic coefficients, one row per term.
            ar: ``(p, k, k)`` autoregressive stack.
            ma: ``(q, k, k)`` moving-average stack.

        Returns:
            An ``(nobs, k)`` array, zero for the first ``p`` rows.

        Example:
            At the true parameters the recursion recovers the innovations
            that generated the sample almost exactly:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> u = rng.standard_normal((302, 2))
            >>> y = np.zeros((302, 2))
            >>> for t in range(1, 302):
            ...     y[t] = 0.5 * y[t - 1] + u[t] + 0.4 * u[t - 1]
            >>> model = VARMA(y[2:], order=(1, 1))
            >>> ar, ma = 0.5 * np.eye(2)[None], 0.4 * np.eye(2)[None]
            >>> recovered = model._recursive_innovations(np.zeros((1, 2)), ar, ma)
            >>> recovered.shape, bool(np.allclose(recovered[0], 0.0))
            ((300, 2), True)
            >>> bool(np.corrcoef(recovered[50:, 0], u[52:, 0])[0, 1] > 0.99)
            True
        """
        nobs, k = self._endog.shape
        det = deterministic_columns(self._trend, nobs, start=1)
        base = det @ deterministic if deterministic.shape[0] else np.zeros((nobs, k))
        u = np.zeros((nobs, k), dtype=np.float64)
        for t in range(self._order, nobs):
            point = base[t].copy()
            for i in range(self._order):
                point += ar[i] @ self._endog[t - 1 - i]
            for j in range(self._ma_order):
                if t - 1 - j >= 0:
                    point += ma[j] @ u[t - 1 - j]
            u[t] = self._endog[t] - point
        return u

    def fit(self) -> VARMAResult:
        """Estimate the system by the three Hannan-Rissanen passes.

        Pass one regresses the sample on ``long_order`` of its own lags and
        keeps the residuals as innovation proxies, zero before they exist.
        Pass two regresses the sample on ``p`` own lags and ``q`` lags of the
        proxies from the first row where both are complete. Pass three, when
        the second-stage moving-average block is invertible, rebuilds the
        innovations recursively from the second-stage parameters and
        re-estimates from row ``max(p, q) + p``; otherwise the second-stage
        estimate is reported with ``refined=False``.

        Returns:
            The fitted result. ``refined`` on the result records whether the
            third pass ran; it is skipped only when the second-stage
            moving-average block is not invertible.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> u = rng.standard_normal((302, 2))
            >>> y = np.zeros((302, 2))
            >>> for t in range(1, 302):
            ...     y[t] = 0.5 * y[t - 1] + u[t] + 0.4 * u[t - 1]
            >>> res = VARMA(y[2:], order=(1, 1)).fit()
            >>> res.long_order, res.refined, res.nobs, res.design.shape
            (18, True, 298, (298, 5))
            >>> ar, ma = res.coefficients[0].diagonal(), res.ma_coefficients[0].diagonal()
            >>> bool(np.abs(ar - 0.5).max() < 0.1), bool(np.abs(ma - 0.4).max() < 0.1)
            (True, True)
        """
        k, p, q = self.k_endog, self._order, self._ma_order
        nobs = self._endog.shape[0]
        h = self._long_order()

        det1 = deterministic_columns(self._trend, nobs - h, start=h + 1)
        design1 = np.column_stack([det1, lag_matrix(self._endog, h, start=h)])
        _, resid1 = self._least_squares(self._endog[h:], design1)
        innovations = np.zeros((nobs, k), dtype=np.float64)
        innovations[h:] = resid1

        moments, design = self._stage_regression(innovations, max(p, h) + q)
        deterministic, ar, ma = self._unpack(moments)

        refined = bool(_StabilityAssessment.assess_stability(ma).is_stable)
        if refined:
            recursive = self._recursive_innovations(deterministic, ar, ma)
            moments, design = self._stage_regression(recursive, max(p, q) + p)
            deterministic, ar, ma = self._unpack(moments)

        return VARMAResult(
            endog=self._endog,
            names=self._names,
            order=p,
            trend=self._trend,
            coefficients=ar,
            ma_coefficients=ma,
            deterministic=deterministic,
            long_order=h,
            refined=refined,
            sigma_u=moments.sigma_u,
            sigma_ml=moments.sigma_ml,
            resid=moments.resid,
            fittedvalues=moments.fittedvalues,
            design=design,
            llf=moments.llf,
            nobs=moments.nobs,
            n_params=k * moments.width + k * (k + 1) / 2,
        )
