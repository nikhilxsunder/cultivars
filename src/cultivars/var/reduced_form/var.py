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

import numpy as np
import numpy.typing as npt

from ..._core import InformationCriteria, SummaryTable
from ..._internals import (
    _ComparisonMixin,
    _SummaryMixin,
    _VectorAutoRegressionFit,
    _VectorAutoRegressionModel,
    _VectorInferenceMixin,
)
from ...exceptions import SpecificationError


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
