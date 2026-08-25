"""Global reduced-form VAR results assembled from conditional units."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from ..._core import SummaryTable, validate_weights
from ..._internals import _SummaryMixin, _VectorPropagationMixin, _WaldTestResult, solve_global
from ...exceptions import SpecificationError
from .error_correction import VECMXResult
from .vector_autoregression import VARXResult


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class GVARResult(_SummaryMixin, _VectorPropagationMixin):
    """A closed global system assembled from conditional units.

    Deliberately not a :class:`_VectorResult`. Every other result in the package
    was estimated: it has one design matrix, one likelihood, one parameter
    count. This one was *assembled* -- the units were estimated separately and
    then linked by an algebraic solve -- so it has none of those three, and
    giving it hollow versions to fit the hierarchy would be worse than standing
    outside it. What it does have is a law of motion for every variable, which
    is why it carries the propagation surface the units individually refused.

    The assembly is exact rather than approximate. Substituting each unit's link
    matrix into its own equations and stacking gives a square system whose
    solution reproduces every unit equation to machine precision; nothing is
    re-estimated and no unit's coefficients are altered.

    What the assembly cannot do is make the units' coefficients correct. A
    unit's foreign aggregate contains other units' contemporaneous variables,
    which the global solve makes functions of that unit's own innovations, so
    ordinary least squares on a unit equation is consistent only if the
    aggregate is weakly exogenous for it. That is an assumption about the data,
    not a property of the estimator, and the bias it causes does not shrink as
    units are added. :meth:`weak_exogeneity_test` is therefore not an optional
    diagnostic here -- it is the condition under which everything else on this
    object means what it says, and the summary names any unit that fails it.

    Attributes:
        endog: The global panel, all units side by side.
        names: Global variable labels, ``"unit.variable"``.
        unit_names: Unit labels, in global column order.
        unit_sizes: Variables each unit contributes.
        order: Lags of the global system.
        trend: Deterministic specification of the global representation.
        coefficients: ``(p, k, k)`` global autoregressive matrices.
        deterministic: ``(d, k)`` global deterministic coefficients.
        contemporaneous: The ``(k, k)`` linkage matrix inverted to close the
            system, retained because its conditioning is the honest diagnostic
            for a badly specified weight matrix.
        weights: The validated cross-unit weight matrix.
        sigma_u: Covariance of the global reduced-form innovations.
        resid: The global reduced-form innovations on the common sample.
        nobs: Rows of the common sample.
        units: The fitted unit results, in global column order.
    """

    endog: npt.NDArray[np.float64]
    names: tuple[str, ...]
    unit_names: tuple[str, ...]
    unit_sizes: tuple[int, ...]
    order: int
    trend: str
    coefficients: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    contemporaneous: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]
    sigma_u: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    nobs: int
    units: tuple[VARXResult | VECMXResult, ...]

    @property
    def n_units(self) -> int:
        """Units the system links."""
        return len(self.unit_names)

    @property
    def sigma_ml(self) -> npt.NDArray[np.float64]:
        """Innovation covariance without the degrees-of-freedom correction."""
        return self.resid.T @ self.resid / self.nobs

    def _unit_index(self, unit: str) -> int:
        """Position of a unit, or a failure naming the ones that exist."""
        if unit not in self.unit_names:
            raise SpecificationError(f"unknown unit {unit!r}; expected one of {self.unit_names}.")
        return self.unit_names.index(unit)

    def unit_columns(self, unit: str) -> tuple[int, ...]:
        """Global column indices the named unit owns."""
        index = self._unit_index(unit)
        start = sum(self.unit_sizes[:index])
        return tuple(range(start, start + self.unit_sizes[index]))

    @property
    def linkage_condition(self) -> float:
        """Condition number of the contemporaneous linkage matrix.

        Large values mean two units' foreign aggregates are nearly the same
        combination of global variables, so the solve that closes the system is
        ill-posed and the global coefficients are sensitive to noise in the unit
        estimates. A weight matrix with a few dominant trading partners shared
        across many units is the usual cause.
        """
        return float(np.linalg.cond(self.contemporaneous))

    def weak_exogeneity_test(self, unit: str) -> tuple[_WaldTestResult, ...]:
        """Test that a unit's foreign aggregates ignore its disequilibria.

        The condition the whole construction rests on. If a unit's foreign
        aggregate responds to that unit's own error-correction terms then the
        aggregate is not weakly exogenous, the unit's least-squares estimates
        are inconsistent, and every global quantity assembled from them
        inherits the bias.

        The auxiliary regression puts each foreign variable's change on the
        unit's own short-run design with the contemporaneous foreign block
        removed, and tests that the error-correction coefficients are jointly
        zero. Removing the contemporaneous block is not a detail: leaving it in
        would put the dependent variable on both sides, and testing the unit's
        *residuals* instead -- the obvious first idea -- has no power at all,
        because least squares already made them orthogonal to the foreign block
        by construction.

        Args:
            unit: One of :attr:`unit_names`.

        Returns:
            One test per foreign variable, in that unit's exogenous order.

        Raises:
            SpecificationError: If the unit is unknown, was fitted in levels
                rather than as an error-correction model, or has rank zero.
        """
        index = self._unit_index(unit)
        fitted = self.units[index]
        if not isinstance(fitted, VECMXResult):
            raise SpecificationError(
                f"unit {unit!r} was fitted in levels, which has no error-correction terms "
                "for the foreign variables to respond to, so this test does not exist for "
                "it. Specify the unit as a conditional error-correction model if the "
                "weak-exogeneity assumption needs checking, which for a global system it "
                "does."
            )
        if not fitted.rank:
            raise SpecificationError(
                f"unit {unit!r} was fitted at rank zero, so it has no disequilibria for "
                "the foreign variables to respond to and nothing to test."
            )
        rank, k_exog = fitted.rank, fitted.k_exog
        width = fitted.design.shape[1]
        first = width - rank
        contemporaneous = k_exog if fitted.contemporaneous else 0
        keep = [c for c in range(width) if not (first - contemporaneous <= c < first)]
        auxiliary = fitted.design[:, keep]
        positions = [keep.index(c) for c in range(first, width)]
        restricted = np.delete(auxiliary, positions, axis=1)
        rows = auxiliary.shape[0]
        changes = np.diff(fitted.exog, axis=0)[-rows:]
        out: list[_WaldTestResult] = []
        for column, label in enumerate(fitted.exog_names):
            target = changes[:, column]
            full_resid = target - auxiliary @ np.linalg.lstsq(auxiliary, target, rcond=None)[0]
            null_resid = target - restricted @ np.linalg.lstsq(restricted, target, rcond=None)[0]
            denominator = float(full_resid @ full_resid)
            statistic = rows * (float(null_resid @ null_resid) - denominator) / denominator
            out.append(
                _WaldTestResult(
                    statistic=statistic,
                    df=rank,
                    pvalue=float(chi2.sf(statistic, rank)),
                    null=f"{label} is weakly exogenous for {unit}",
                )
            )
        return tuple(out)

    def exogeneity_table(self, *, alpha: float = 0.05) -> SummaryTable:
        """Run the weak-exogeneity test on every unit that supports it."""
        rows: list[tuple[str, ...]] = []
        skipped: list[str] = []
        failed: list[str] = []
        for unit in self.unit_names:
            try:
                tests = self.weak_exogeneity_test(unit)
            except SpecificationError:
                skipped.append(unit)
                continue
            for test in tests:
                rejected = test.reject(alpha=alpha)
                if rejected:
                    failed.append(unit)
                rows.append(
                    (
                        unit,
                        test.null.split(" is weakly")[0],
                        f"{test.statistic:.3f}",
                        f"{test.df}",
                        f"{test.pvalue:.4f}",
                        "REJECT" if rejected else "",
                    )
                )
        notes = [
            "A rejection says that unit's foreign aggregate responds to its own "
            "disequilibria, so the aggregate is not weakly exogenous, the unit's "
            "estimates are inconsistent, and every global quantity built on them "
            "inherits the bias.",
        ]
        if failed:
            notes.insert(
                0,
                f"NOT WEAKLY EXOGENOUS at the {alpha:.0%} level: "
                f"{tuple(sorted(set(failed)))}. Impulse responses and the variance "
                "decomposition from this system should not be read as they stand.",
            )
        if skipped:
            notes.append(f"Units fitted in levels and therefore untestable: {tuple(skipped)}.")
        return SummaryTable(
            title="Weak exogeneity of the foreign variables",
            metadata=(
                ("Units", f"{self.n_units}"),
                ("Tested", f"{self.n_units - len(skipped)}"),
                ("Rejections", f"{len(set(failed))}"),
                ("Level", f"{alpha:.0%}"),
            ),
            columns=("unit", "foreign variable", "statistic", "df", "p-value", ""),
            rows=tuple(rows),
            notes=tuple(notes),
        )

    def _comparison_label(self) -> str:
        """Short specification label."""
        return f"GVAR({self.order}, units={self.n_units})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        stability = self.stability_check()
        notes = [
            f"Stable: {self.is_stable}   max |companion root| = {stability.max_modulus:.4f}",
            "Assembled, not estimated: the units were fitted separately and linked by an "
            "algebraic solve that reproduces every unit equation exactly. There is no "
            "global design matrix, likelihood or parameter count, so this result carries "
            "no coefficient table and no information criteria.",
            "Impulse responses are valid only if the foreign aggregates are weakly "
            "exogenous for their units; run exogeneity_table() before reading them.",
            "Prefer generalized_irf() and generalized_fevd() over the orthogonalized "
            "pair: a recursive ordering across this many variables is not a structural "
            "assumption anyone can defend.",
        ]
        if self.linkage_condition > 1e8:
            notes.insert(
                0,
                f"ILL-CONDITIONED LINKAGE: condition number {self.linkage_condition:.3g}. "
                "Two units' foreign aggregates are nearly the same combination of global "
                "variables, so the global coefficients are unstable in the unit estimates.",
            )
        return SummaryTable(
            title=f"GVAR({self.order}) Results",
            metadata=(
                ("Model", f"GVAR({self.order})"),
                ("Units", f"{self.n_units}"),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.nobs}"),
                ("Trend", self.trend),
                ("Linkage cond.", f"{self.linkage_condition:.3g}"),
            ),
            columns=("unit", "variables", "order", "foreign block"),
            rows=tuple(
                (name, f"{size}", f"{fitted.order}", f"{fitted.k_exog}")
                for name, size, fitted in zip(
                    self.unit_names, self.unit_sizes, self.units, strict=True
                )
            ),
            notes=tuple(notes),
        )


class GVAR:
    """Link fitted conditional units into one closed global system.

    An assembler rather than an estimator, and the class says so by exposing
    :meth:`solve` instead of ``fit``. Every coefficient in the result was
    estimated by a unit; what happens here is the algebra that turns a
    collection of models each missing a law of motion for its own foreign
    variables into a single model that has one for everything.

    Units may be heterogeneous -- different orders, different variable counts,
    a mixture of levels and error-correction specifications -- because the
    linkage reads each through the levels form it implies rather than through
    the parameterization it was estimated in.
    """

    __slots__ = ("_labels", "_unit_names", "_units", "_weights")

    def __init__(
        self,
        units: Sequence[VARXResult | VECMXResult],
        weights: npt.ArrayLike,
        *,
        variable_labels: Sequence[str],
        unit_names: Sequence[str] | None = None,
    ) -> None:
        """Validate the units, the weight matrix, and the variable mapping.

        Args:
            units: Fitted unit results, in the order their variables appear in
                the global vector. Each must have been fitted against its own
                foreign aggregates as the exogenous block.
            weights: An ``(n_units, n_units)`` cross-unit weight matrix.
            variable_labels: One entry per global column giving the *shared*
                variable identity, so that a unit's foreign aggregate averages
                like with like. Repeated across units: two entries labelled
                ``"gdp"`` in different units are the same variable seen in two
                places.
            unit_names: Unit labels. Defaults to ``unit1 ... unitN``.

        Raises:
            SpecificationError: If no units are given, the labels do not match
                the global width, or the unit names repeat.
            DimensionError: If the weight matrix is the wrong shape.
        """
        self._units = tuple(units)
        if not self._units:
            raise SpecificationError("a global system needs at least one unit.")
        width = sum(unit.k_endog for unit in self._units)
        labels = tuple(str(label) for label in variable_labels)
        if len(labels) != width:
            raise SpecificationError(
                f"variable_labels must have one entry per global column ({width}); "
                f"got {len(labels)}."
            )
        self._labels = labels
        self._weights = validate_weights(weights, n_units=len(self._units))
        if unit_names is None:
            self._unit_names = tuple(f"unit{i + 1}" for i in range(len(self._units)))
        else:
            resolved = tuple(str(name) for name in unit_names)
            if len(resolved) != len(self._units):
                raise SpecificationError(
                    f"unit_names must have one entry per unit ({len(self._units)}); "
                    f"got {len(resolved)}."
                )
            if len(set(resolved)) != len(resolved):
                raise SpecificationError(f"unit_names must be unique; got {resolved}.")
            self._unit_names = resolved

    @property
    def unit_of_column(self) -> tuple[int, ...]:
        """Owning unit index for each global column."""
        return tuple(index for index, unit in enumerate(self._units) for _ in range(unit.k_endog))

    @property
    def variable_of_column(self) -> tuple[int, ...]:
        """Variable identity for each global column, as integer codes."""
        order = {label: code for code, label in enumerate(dict.fromkeys(self._labels))}
        return tuple(order[label] for label in self._labels)

    def _stacked_innovations(
        self, linkage: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Align the unit residuals and rotate them into global coordinates.

        Units may lose different numbers of leading observations to their own
        lag lengths, but every unit's residuals end on the same date, so the
        common sample is the last ``min`` rows. Rotating by the inverse linkage
        turns the unit innovations into the reduced-form innovations of the
        global system, which is what the moving-average representation
        propagates.
        """
        common = min(unit.resid.shape[0] for unit in self._units)
        stacked = np.hstack([unit.resid[-common:] for unit in self._units])
        return np.linalg.solve(linkage, stacked.T).T, common

    def solve(self) -> GVARResult:
        """Close the system and return the global result.

        Returns:
            A :class:`GVARResult`.

        Raises:
            DimensionError: If the units do not partition the global columns.
            NumericalError: If the contemporaneous linkage is singular.
        """
        levels = tuple(unit.to_varx() for unit in self._units)
        linkage, blocks, drift = solve_global(
            levels,
            weights=self._weights,
            unit_of_column=self.unit_of_column,
            variable_of_column=self.variable_of_column,
        )
        innovations, common = self._stacked_innovations(linkage)
        widest = max(unit.design.shape[1] for unit in self._units)
        panel = np.hstack([unit.endog for unit in self._units])
        names = tuple(
            f"{self._unit_names[self.unit_of_column[column]]}.{self._labels[column]}"
            for column in range(len(self._labels))
        )
        trend = ("n", "c", "ct")[min(drift.shape[0], 2)]
        return GVARResult(
            endog=panel,
            names=names,
            unit_names=self._unit_names,
            unit_sizes=tuple(unit.k_endog for unit in self._units),
            order=int(blocks.shape[0]),
            trend=trend,
            coefficients=blocks,
            deterministic=drift,
            contemporaneous=linkage,
            weights=self._weights,
            sigma_u=innovations.T @ innovations / max(common - widest, 1),
            resid=innovations,
            nobs=common,
            units=self._units,
        )
