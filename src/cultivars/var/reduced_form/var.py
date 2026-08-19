"""The reduced-form vector autoregression: the root of the multivariate surface.

Estimates ``y_t = D d_t + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t`` by least
squares and exposes the nine operations that make a VAR usable for inference
rather than merely fitted -- forecasting, impulse responses, variance
decomposition, historical decomposition, Granger causality, stability, lag
selection, and residual diagnostics.

Seven of those nine live on :class:`_VectorInferenceMixin` rather than here,
because they are functions of the coefficient stack, the innovation covariance,
and the residuals, and know nothing about how those were produced. That is what
lets a VARX, a panel VAR, and a VECM inherit the same surface by supplying a
different regressor block or a different solve.

One warning is repeated in the summary of every fit and belongs here too. An
*orthogonalized* impulse response is not a reduced-form object. It requires a
Cholesky factor, the Cholesky factor is lower-triangular, and triangularity is
a recursive identifying restriction on the order of ``names``. Permute the
variables and the answer changes. The reduced-form VAR reports it because
everyone wants it, and says plainly that it is a recursive SVAR whose
assumption has not been declared; :mod:`cultivars.var.structural` is where the
declaration goes.

References:
    Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
    Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from ..._core import (
    _CHOLESKY_NOTE,
    _NO_STDERR_NOTE,
    _UNSTABLE_NOTE,
    InformationCriteria,
    SummaryTable,
    deterministic_columns,
    validate_exog_matrix,
)
from ..._internals import (
    _ComparisonMixin,
    _ExogenousVectorAutoRegressionFit,
    _ExogenousVectorAutoRegressionModel,
    _SummaryMixin,
    _VectorAutoRegressionFit,
    _VectorAutoRegressionModel,
    _VectorInferenceMixin,
)
from ...exceptions import DimensionError, SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class VARResult(_SummaryMixin, _ComparisonMixin, _VectorInferenceMixin):
    """A fitted reduced-form vector autoregression.

    Declares the attributes :class:`_VectorInferenceMixin` reads and adds the
    display surface. Every substantive post-estimation operation is inherited,
    which is the point: this class is a record plus a summary, not an
    implementation.

    Attributes:
        endog: The full observed panel, shape ``(n, k)``.
        names: Variable names in column order.
        order: Autoregressive order.
        trend: Deterministic specification.
        coefficients: ``A_1, ..., A_p``, shape ``(p, k, k)``.
        deterministic: Deterministic coefficients, shape ``(d, k)``.
        sigma_u: Degrees-of-freedom-adjusted innovation covariance.
        sigma_ml: Maximum-likelihood innovation covariance.
        resid: Residuals over the effective sample, shape ``(nobs, k)``.
        fittedvalues: One-step fitted values, shape ``(nobs, k)``.
        design: The regressor matrix, shape ``(nobs, d + k * p)``.
        llf: Gaussian log-likelihood.
        nobs: Effective sample size.
        n_params: Coefficients plus the distinct covariance elements.
    """

    endog: npt.NDArray[np.float64]
    names: tuple[str, ...]
    order: int
    trend: str
    coefficients: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    sigma_u: npt.NDArray[np.float64]
    sigma_ml: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    design: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int

    @classmethod
    def _from_fit(
        cls, fit: _VectorAutoRegressionFit, model: _VectorAutoRegressionModel[VARResult]
    ) -> VARResult:
        """Assemble the public result from a raw fit and its specification."""
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
        )

    def equation(self, name: str) -> dict[str, float]:
        """One equation's coefficients, keyed by regressor.

        The unit a reader actually thinks in. A ``(p, k, k)`` stack is the right
        shape for the recursions and the wrong shape for a person: reading
        ``coefficients[1][0, 2]`` as "the effect of the third variable's second
        lag on the first equation" is a mistake waiting to happen, and this
        spells it as ``rate.L2``.

        Args:
            name: Which equation, by variable name.

        Returns:
            Deterministic terms first, then lags in order, keyed
            ``"{source}.L{lag}"``.

        Raises:
            SpecificationError: If ``name`` is not in ``names``.
        """
        if name not in self.names:
            raise SpecificationError(f"unknown variable {name!r}; expected one of {self.names}.")
        equation = self.names.index(name)
        out: dict[str, float] = {}
        labels = ["const"] if self.trend in ("c", "ct") else []
        if self.trend == "ct":
            labels.append("trend")
        for row, label in enumerate(labels):
            out[label] = float(self.deterministic[row, equation])
        for lag in range(self.order):
            for column, source in enumerate(self.names):
                out[f"{source}.L{lag + 1}"] = float(self.coefficients[lag][equation, column])
        return out

    @property
    def params(self) -> dict[str, float]:
        """Every coefficient, keyed ``"{equation}: {regressor}"``.

        Flat rather than nested because :class:`SummaryTable` renders one
        coefficient per row and the equation has to travel with the name; a
        table of ``k`` blocks each headed ``const, y1.L1, ...`` is unreadable
        once ``k`` exceeds two.
        """
        return {
            f"{equation}: {regressor}": value
            for equation in self.names
            for regressor, value in self.equation(equation).items()
        }

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        suffix = "" if self.trend == "c" else f" trend={self.trend}"
        return f"VAR({self.order}){suffix}"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        ic: InformationCriteria = self.information_criteria
        stability = self.stability_check()
        notes: list[str] = []
        if not self.is_stable:
            notes.append(
                "NOT STABLE: a companion root lies on or outside the unit circle, so "
                "impulse responses diverge rather than decay and forecasts have no "
                "limit. Every dynamic quantity below is uninterpretable."
            )
        notes.extend(
            (
                f"Stable: {self.is_stable}   max |companion root| = {stability.max_modulus:.4f}",
                "Orthogonalized impulse responses and the variance decomposition use a "
                "Cholesky factor, which imposes the recursive ordering of the variable "
                "names; that is a structural assumption, not a reduced-form result.",
                "Standard errors are not yet available for this estimator.",
            )
        )
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=(
                ("Model", self._comparison_label()),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Variables", f"{self.k_endog}"),
                ("AIC", f"{ic.aic:.3f}"),
                ("Trend", self.trend),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=tuple(notes),
        )


class VAR(_VectorAutoRegressionModel[VARResult]):
    """Reduced-form vector autoregression of order ``p``.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order. Choose it with
            :meth:`lag_order_selection` rather than by eye.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        names: Variable names in column order; defaults to ``("y1", ...)``.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> a = np.array([[0.5, 0.1], [0.0, 0.4]])
        >>> y = np.zeros((500, 2))
        >>> for t in range(1, 500):
        ...     y[t] = a @ y[t - 1] + rng.standard_normal(2)
        >>> res = VAR(y, order=1, names=("x", "z")).fit()
        >>> res.is_stable
        True
        >>> res.irf(8).shape
        (9, 2, 2)
    """

    __slots__ = ()

    def fit(self) -> VARResult:
        """Estimate the system by least squares."""
        return VARResult._from_fit(self._fit_family(), self)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class VARXResult(_SummaryMixin, _ComparisonMixin, _VectorInferenceMixin):
    """A fitted vector autoregression with exogenous regressors.

    Everything the reduced-form surface offers is inherited unchanged, because
    the exogenous block does not enter the moving-average representation:
    ``irf``, ``fevd``, ``historical_decomposition``, ``granger_causality``,
    ``stability_check``, and ``residual_diagnostics`` read the endogenous
    dynamics and are therefore the same objects they are for a plain VAR.

    :meth:`forecast` is the exception, and it is the only genuinely new
    behaviour in the model. A VARX forecast is conditional on a path for ``x``,
    and this class has no model of ``x`` to supply one, so it asks for the path
    and refuses without it.

    Attributes:
        endog: The endogenous sample.
        exog: The exogenous sample.
        names: Endogenous labels, in Cholesky order.
        exog_names: Exogenous labels.
        order: Endogenous autoregressive order.
        exog_order: Exogenous lags beyond the contemporaneous term.
        trend: Deterministic specification.
        coefficients: ``(p, k, k)`` stack of ``A_1, ..., A_p``.
        exog_coefficients: ``(s + 1, k, m)`` stack of ``B_0, ..., B_s``.
        deterministic: Deterministic coefficients, one row per term.
        sigma_u: Residual covariance with the degrees-of-freedom correction.
        sigma_ml: Residual covariance divided by the effective sample.
        resid: Residuals over the effective sample.
        fittedvalues: One-step conditional means over the effective sample.
        design: The regressor matrix as estimated.
        llf: Gaussian log-likelihood.
        nobs: Effective sample size.
        n_params: Free parameters, covariance included.
    """

    endog: npt.NDArray[np.float64]
    exog: npt.NDArray[np.float64]
    names: tuple[str, ...]
    exog_names: tuple[str, ...]
    order: int
    exog_order: int
    trend: str
    coefficients: npt.NDArray[np.float64]
    exog_coefficients: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    sigma_u: npt.NDArray[np.float64]
    sigma_ml: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    design: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int

    @classmethod
    def _from_fit(
        cls, fit: _ExogenousVectorAutoRegressionFit, model: _ExogenousVectorAutoRegressionModel
    ) -> Self:
        """Assemble the public result from the internal fit and its model."""
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

    @property
    def k_exog(self) -> int:
        """Number of exogenous variables."""
        return len(self.exog_names)

    def _trailing_blocks(self) -> tuple[npt.NDArray[np.float64], ...]:
        """The distributed-lag coefficients, in design order."""
        return tuple(self.exog_coefficients[j].T for j in range(self.exog_order + 1))

    def forecast(
        self, steps: int = 1, *, exog_future: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Point forecasts conditional on a future path for the exogenous block.

        Args:
            steps: Horizon.
            exog_future: A ``(steps, m)`` path for the exogenous variables, in
                the column order of :attr:`exog_names`. Required.

        Returns:
            A ``(steps, k)`` array of conditional means.

        Raises:
            SpecificationError: If ``steps`` is not positive, or ``exog_future``
                is omitted.
            DimensionError: If ``exog_future`` has the wrong shape.
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
        """Every coefficient, keyed ``"{equation}: {regressor}"``."""
        return {
            f"{equation}: {name}": value
            for equation in self.names
            for name, value in self.equation(equation).items()
        }

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table."""
        tail = "" if self.trend == "c" else f", trend={self.trend}"
        return f"VARX({self.order}, {self.exog_order}{tail})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
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
            _NO_STDERR_NOTE,
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
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=tuple(notes),
        )


class VARX(_ExogenousVectorAutoRegressionModel[VARXResult]):
    """Vector autoregression with exogenous regressors, estimated by least squares.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal((200, 1))
        >>> y = np.column_stack([x[:, 0] + rng.standard_normal(200), rng.standard_normal(200)])
        >>> res = VARX(y, x, order=1, exog_order=1).fit()
        >>> res.forecast(3, exog_future=np.zeros((3, 1))).shape
        (3, 2)
    """

    __slots__ = ()

    def fit(self) -> VARXResult:
        """Estimate the system and return the fitted result."""
        return VARXResult._from_fit(self._fit_family(), self)
