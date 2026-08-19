"""The reduced-form panel vector autoregression with pooled slopes."""

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
)
from ..._internals import (
    _ComparisonMixin,
    _PanelVectorAutoRegressionModel,
    _SummaryMixin,
    _VectorAutoRegressionFit,
    _VectorInferenceMixin,
)
from ...exceptions import SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class PanelVARResult(_SummaryMixin, _ComparisonMixin, _VectorInferenceMixin):
    """A fitted panel vector autoregression with pooled slopes.

    The residual matrix is a stack of units rather than one series, and every
    inherited method that reads residual history reads it through
    :attr:`_sample_blocks`, so no lagged cross-product and no shock propagation
    ever crosses a unit boundary. ``irf``, ``fevd``, and ``stability_check``
    depend only on the pooled matrices and are unqualified.

    :meth:`forecast` is unit-specific and says so: the slopes are pooled but the
    intercept and the lag history are not, so there is no single path to return.

    Attributes:
        endog: Every unit stacked, in the order given. Not a time series.
        names: Variable labels, in Cholesky order.
        unit_names: Unit labels.
        unit_lengths: Observations per unit before lags are taken.
        effects: ``"unit"`` or ``"none"``.
        order: Autoregressive order, common to all units.
        trend: Deterministic specification, meaningful only under ``"none"``.
        coefficients: ``(p, k, k)`` stack of pooled ``A_1, ..., A_p``.
        deterministic: One row per unit under fixed effects, otherwise one row
            per deterministic term.
        sigma_u: Pooled residual covariance with the degrees-of-freedom
            correction, which counts the unit indicators.
        sigma_ml: Pooled residual covariance divided by the effective sample.
        resid: Residuals, stacked in unit order.
        fittedvalues: One-step conditional means, stacked in unit order.
        design: The regressor matrix as estimated.
        llf: Gaussian log-likelihood.
        nobs: Effective sample size, the sum of ``T_i - p``.
        n_params: Free parameters, covariance included.
    """

    endog: npt.NDArray[np.float64]
    names: tuple[str, ...]
    unit_names: tuple[str, ...]
    unit_lengths: tuple[int, ...]
    effects: str
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
        cls, fit: _VectorAutoRegressionFit, model: _PanelVectorAutoRegressionModel
    ) -> Self:
        """Assemble the public result from the internal fit and its model."""
        return cls(
            endog=model.endog,
            names=model.names,
            unit_names=model.unit_names,
            unit_lengths=model.unit_lengths,
            effects=model.effects,
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

    @property
    def n_units(self) -> int:
        """Number of units."""
        return len(self.unit_names)

    @property
    def min_time_dimension(self) -> int:
        """Observations in the shortest unit, which is what the Nickell bias tracks."""
        return min(self.unit_lengths)

    @property
    def _sample_blocks(self) -> tuple[tuple[int, int], ...]:
        """One half-open residual span per unit, in stacking order."""
        spans: list[tuple[int, int]] = []
        cursor = 0
        for length in self.unit_lengths:
            span = length - self.order
            spans.append((cursor, cursor + span))
            cursor += span
        return tuple(spans)

    def _unit_index(self, unit: str) -> int:
        """Position of a unit, or a failure naming the ones that exist."""
        if unit not in self.unit_names:
            raise SpecificationError(f"unknown unit {unit!r}; expected one of {self.unit_names}.")
        return self.unit_names.index(unit)

    def unit_series(self, unit: str) -> npt.NDArray[np.float64]:
        """One unit's observations, including the rows consumed as lags.

        Args:
            unit: A unit label.

        Returns:
            A ``(T_i, k)`` view into :attr:`endog`.

        Raises:
            SpecificationError: If the unit is unknown.
        """
        index = self._unit_index(unit)
        start = sum(self.unit_lengths[:index])
        return self.endog[start : start + self.unit_lengths[index]]

    def unit_residuals(self, unit: str) -> npt.NDArray[np.float64]:
        """One unit's residuals.

        Args:
            unit: A unit label.

        Returns:
            A ``(T_i - p, k)`` view into :attr:`resid`.

        Raises:
            SpecificationError: If the unit is unknown.
        """
        lo, hi = self._sample_blocks[self._unit_index(unit)]
        return self.resid[lo:hi]

    def _deterministic_labels(self) -> tuple[str, ...]:
        """One intercept name per unit under fixed effects, otherwise the trend block."""
        if self.effects != "unit":
            return (("const",) if self.trend in ("c", "ct") else ()) + (
                ("trend",) if self.trend == "ct" else ()
            )
        return tuple(f"const[{unit}]" for unit in self.unit_names)

    @property
    def unit_effects(self) -> dict[str, npt.NDArray[np.float64]]:
        """Each unit's intercept vector.

        Raises:
            SpecificationError: If the model was fitted without unit effects, in
                which case there are no unit intercepts to return and an empty
                mapping would read as though every unit had a zero one.
        """
        if self.effects != "unit":
            raise SpecificationError(
                "this model was fitted with effects='none', so there are no unit "
                "intercepts; the pooled deterministic terms are in params."
            )
        return {name: self.deterministic[i] for i, name in enumerate(self.unit_names)}

    def forecast(self, steps: int = 1, *, unit: str | None = None) -> npt.NDArray[np.float64]:
        """Point forecasts for one unit.

        Args:
            steps: Horizon.
            unit: Which unit to forecast. Required.

        Returns:
            A ``(steps, k)`` array of conditional means.

        Raises:
            SpecificationError: If ``steps`` is not positive, ``unit`` is
                omitted, or the unit is unknown.
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        if unit is None:
            raise SpecificationError(
                "a panel forecast is unit-specific: pass unit=<name> from "
                f"{self.unit_names}. The slopes are pooled but the intercept and the lag "
                "history are not, so there is no single path this could return."
            )
        index = self._unit_index(unit)
        series = self.unit_series(unit)
        k, p, nobs = self.k_endog, self.order, series.shape[0]
        if self.effects == "unit":
            baseline = np.tile(self.deterministic[index], (steps, 1))
        elif self.deterministic.shape[0]:
            det = deterministic_columns(self.trend, steps, start=nobs + 1)
            baseline = det @ self.deterministic
        else:
            baseline = np.zeros((steps, k), dtype=np.float64)
        history = [series[nobs - j - 1] for j in range(p)]
        out = np.empty((steps, k), dtype=np.float64)
        for h in range(steps):
            point = baseline[h].copy()
            for j in range(p):
                point = point + self.coefficients[j] @ history[j]
            out[h] = point
            history = [point, *history[: p - 1]] if p else []
        return out

    @property
    def slopes(self) -> dict[str, float]:
        """The pooled lag coefficients alone, without the unit intercepts."""
        keep = set(self._lag_labels())
        return {
            name: value
            for name, value in self.params.items()
            if name.split(": ", 1)[1] in keep
        }

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table."""
        return f"PanelVAR({self.order}, effects={self.effects})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        criteria = self.information_criteria
        stability = self.stability_check()
        notes = [
            f"Stable: {self.is_stable}   max |companion root| = {stability.max_modulus:.4f}",
            "Slopes are pooled: every unit is assumed to share the same autoregressive "
            "matrices and the same innovation covariance, and only the intercept varies.",
            "Lags are built inside each unit, so no regressor crosses a unit boundary and "
            "the effective sample is the sum of T_i - p.",
        ]
        if self.effects == "unit":
            notes.append(
                "Nickell bias: with unit dummies the lag coefficients are biased of order "
                f"1/T, and the shortest unit here has T = {self.min_time_dimension}. The "
                "bias does not shrink as the number of units grows, only as T grows, so "
                "these estimates are consistent in T and not in N. A GMM estimator in "
                "first differences is the standard remedy and is not implemented here."
            )
        notes.append(_CHOLESKY_NOTE)
        if self.effects == "unit":
            notes.append(
                f"The {self.n_units} unit intercepts are nuisance parameters and are not "
                "listed below; read them from unit_effects, or the full map from params."
            )
        if not self.is_stable:
            notes.insert(0, _UNSTABLE_NOTE)
        shown = self.slopes if self.effects == "unit" else self.params
        return SummaryTable(
            title=f"Panel VAR({self.order}) Results",
            metadata=(
                ("Model", f"PanelVAR({self.order})"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Units", f"{self.n_units}"),
                ("AIC", f"{criteria.aic:.3f}"),
                ("Variables", f"{self.k_endog}"),
                ("BIC", f"{criteria.bic:.3f}"),
                ("Effects", self.effects),
                ("HQIC", f"{criteria.hqic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("Shortest T", f"{self.min_time_dimension}"),
            ),
            columns=self._coefficient_columns(),
            rows=tuple(row for row in self._coefficient_rows() if row[0] in shown),
            notes=tuple(notes),
        )


class PanelVAR(_PanelVectorAutoRegressionModel[PanelVARResult]):
    """Panel vector autoregression with pooled slopes and unit fixed effects.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> panel = rng.standard_normal((5, 60, 2))
        >>> res = PanelVAR(panel, order=1).fit()
        >>> res.forecast(3, unit="unit1").shape
        (3, 2)
    """

    __slots__ = ()

    def fit(self) -> PanelVARResult:
        """Estimate the pooled system and return the fitted result."""
        return PanelVARResult._from_fit(self._fit_family(), self)
