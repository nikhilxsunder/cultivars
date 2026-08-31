# filepath: /src/cultivars/var/reduced_form/moving_average.py
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

"""The vector autoregressive moving average, estimated without a likelihood search.

A VARMA(p, q) buys parsimony with a harder estimation problem: the moving-average
innovations are unobserved, so least squares cannot be applied directly, and the
Gaussian likelihood is a surface over ``k^2 (p + q)`` parameters that is neither
cheap nor -- for an unrestricted model -- uniquely maximized. This module takes
the Hannan-Rissanen route instead: recover the innovations from a long
autoregression, then estimate the VARMA by regressions in which those estimated
innovations stand in for the true ones. Three passes, every one of them a
least-squares solve, no numerical optimizer anywhere.

The identification caveat is structural rather than numerical and is repeated on
the result: distinct ``(A, M)`` pairs can generate identical second moments, and
the standard resolution -- echelon-form restrictions indexed by Kronecker
indices -- is deliberately not imposed here. What survives the ambiguity is
exactly what most users come for: for a stable, invertible representation the
moving-average coefficients, and with them the forecasts, impulse responses,
and variance decompositions, are invariant across observationally equivalent
parameterizations.

References:
    Hannan, E. J., & Rissanen, J. (1982). Recursive estimation of mixed
        autoregressive-moving average order. *Biometrika*, 69(1), 81-94.
    Dufour, J.-M., & Jouini, T. (2005). Asymptotic distribution of a simple
        linear estimator for VARMA models in echelon form. In *Statistical
        Modeling and Analysis for Complex Data Problems* (pp. 209-240).
    Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*,
        chapters 11-12.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
import numpy.typing as npt

from ..._core import (
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
from ..._internals import (
    _ComparisonMixin,
    _StabilityTest,
    _SummaryMixin,
    _VectorAutoRegressionModel,
    _VectorInferenceMixin,
    _VectorMoments,
    _VectorPropagationMixin,
    _VectorResult,
)
from ...exceptions import SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class VARMAResult(
    _VectorResult,
    _SummaryMixin,
    _ComparisonMixin,
    _VectorInferenceMixin,
    _VectorPropagationMixin,
):
    """A fitted vector autoregressive moving average.

    The propagation surface is inherited with one override that changes
    everything downstream: :meth:`ma_representation` runs the VARMA recursion
    ``Psi_h = M_h + sum A_i Psi_{h-i}`` rather than powering the autoregressive
    companion, and since the impulse responses, both variance decompositions,
    and the historical decomposition all read the moving-average matrices
    rather than the companion, correcting the one method corrects them all.
    :meth:`forecast` is the other override, because a VARMA forecast carries
    the last ``q`` innovations across the sample boundary.

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
    """

    coefficients: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    ma_coefficients: npt.NDArray[np.float64]
    long_order: int
    refined: bool

    @property
    def ma_order(self) -> int:
        """Moving-average order ``q``."""
        return int(self.ma_coefficients.shape[0])

    def _trailing_blocks(self) -> tuple[npt.NDArray[np.float64], ...]:
        """The moving-average coefficients, in design order."""
        return tuple(self.ma_coefficients[j].T for j in range(self.ma_order))

    def _trailing_labels(self) -> tuple[str, ...]:
        """Moving-average column names, innovation lag by innovation lag."""
        return tuple(
            f"u[{source}].L{j + 1}"
            for j in range(self.ma_order)
            for source in self.names
        )

    def ma_representation(self, horizon: int = 20) -> npt.NDArray[np.float64]:
        """Moving-average matrices ``Psi_0, ..., Psi_horizon`` of the VARMA.

        Computed by the recursion ``Psi_h = M_h + sum_{i=1}^{min(h,p)} A_i
        Psi_{h-i}`` with ``M_h = 0`` beyond ``q``, which is the representation
        the impulse responses, variance decompositions, and historical
        decomposition inherited from the propagation surface all consume --
        overriding this one method is what makes every one of them correct for
        a VARMA.

        Args:
            horizon: Largest lead to return.

        Returns:
            An array of shape ``(horizon + 1, k, k)``; ``Psi_0`` is the
            identity.

        Raises:
            SpecificationError: If ``horizon`` is negative.
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

    def invertibility_check(self) -> _StabilityTest:
        """Eigenvalue verdict for the moving-average companion matrix.

        Returns:
            The :class:`_StabilityTest`; the moving-average polynomial is
            invertible exactly when every root lies inside the unit circle.
            Invertibility is what makes the innovations recoverable from the
            observed history, so a non-invertible estimate means the residuals
            -- and everything computed from them -- identify a different,
            observationally equivalent representation.
        """
        return _StabilityTest.assess_stability(self.ma_coefficients)

    @property
    def is_invertible(self) -> bool:
        """Whether every moving-average root lies inside the unit circle."""
        return self.invertibility_check().is_stable

    def forecast(self, steps: int = 1) -> npt.NDArray[np.float64]:
        """Deterministic multi-step forecasts from the end of the sample.

        The moving-average block contributes only while a forecast date is
        within ``q`` periods of the sample: the innovations after the sample
        end are set to their zero mean, and the last ``q`` estimated
        innovations decay out of the forecast one step at a time. Beyond
        ``q`` steps the iteration is exactly the autoregressive one.

        Args:
            steps: Forecast horizon, at least one.

        Returns:
            An array of shape ``(steps, k)``.

        Raises:
            SpecificationError: If ``steps`` is less than one.
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
        """Short specification label for a ranking table."""
        tail = "" if self.trend == "c" else f", trend={self.trend}"
        return f"VARMA({self.order}, {self.ma_order}{tail})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        criteria = self.information_criteria
        stability = self.stability_check()
        invertibility = self.invertibility_check()
        notes = [
            f"Stable: {self.is_stable}   max |AR companion root| = "
            f"{stability.max_modulus:.4f}",
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
    """Vector autoregressive moving average, estimated by Hannan-Rissanen.

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
    standard errors that condition on the estimated innovations.

    Args:
        endog: The ``(nobs, k)`` panel, time down the rows.
        order: The pair ``(p, q)``. ``q`` must be at least one -- with no
            moving-average block a plain
            :class:`~cultivars.var.reduced_form.VAR` is the right model, and it
            estimates in one pass instead of three.
        trend: Deterministic terms.
        names: Variable labels. Defaults to ``y1 ... yk``.

    Raises:
        SpecificationError: If the order pair is malformed or ``q`` is zero.
        DimensionError: If the sample cannot support the specification,
            including the long autoregression the first stage needs.

    Example:
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
        """Validate the sample and the specification."""
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
        """Moving-average order ``q``."""
        return self._ma_order

    @property
    def n_regressors(self) -> int:
        """Regressors per equation, including the innovation lags."""
        return super().n_regressors + self.k_endog * self._ma_order

    def _long_order(self) -> int:
        """Order of the stage-one autoregression.

        Grows like the square root of the sample, which dominates any fixed
        ``(p, q)`` asymptotically and keeps the residuals consistent for the
        innovations, capped at what the sample can identify so that the first
        stage never fails on a design its own target cannot support.
        """
        nobs = int(self._endog.shape[0])
        wanted = max(self._order + self._ma_order, int(np.ceil(np.sqrt(nobs))))
        cap = (nobs - self._n_deterministic_columns - 1) // (self.k_endog + 1)
        return max(min(wanted, cap), 1)

    def _burn_for(self, order: int) -> int:
        """Leading observations lost across all three passes."""
        return max(order, self._long_order() + self._ma_order)

    def _max_supported_lags(self) -> int:
        """Largest autoregressive order the sample can identify."""
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

        Raises:
            SpecificationError: Always.
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
        """Stack ``u_{t-1}, ..., u_{t-q}`` for rows ``start`` onward."""
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
        """Split a stage's coefficients into deterministic, AR, and MA blocks."""
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

        Returns:
            The fitted result. ``refined`` on the result records whether the
            third pass ran; it is skipped only when the second-stage
            moving-average block is not invertible.
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

        refined = bool(_StabilityTest.assess_stability(ma).is_stable)
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