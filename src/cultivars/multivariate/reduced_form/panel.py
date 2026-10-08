# filepath: /src/cultivars/multivariate/reduced_form/panel.py
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
r"""The panel vector autoregression: one set of dynamics, many units, pooled by least squares.

A panel VAR estimates

.. math::

   y_{it} = \mu_i + \sum_{l=1}^{p} A_l\, y_{i,t-l} + u_{it},
   \qquad u_{it} \sim (0, \Sigma_u), \qquad i = 1, \dots, N,

on :math:`N` units observed for :math:`T_i` periods each, with the
autoregressive matrices and the innovation covariance shared across units
and only the intercept allowed to differ. The payoff is sample: a system
that would be unidentifiable on one short unit is estimated on
:math:`\sum_i (T_i - p)` rows, and the inherited reduced-form surface --
coefficient tables, Granger tests, impulse responses, variance
decompositions -- comes with it unchanged. The price is the pooling
assumption itself, which is strong rather than technical: heterogeneous
dynamics estimated as if common do not average to the mean dynamics.

Two commitments shape the surface. First, the unit boundary is respected
everywhere. Lags are built inside each unit and stacked afterwards, never
the other way round -- stacking first would quietly regress each unit's
first observation on the previous unit's last, which produces a number
rather than an error -- and the result's residual matrix carries the unit
spans, so every residual-history diagnostic reads within a unit and the
forecast is unit-specific by construction: the slopes are pooled but the
intercept and the lag history are not, so there is no single path to
return and ``forecast()`` asks which unit. Second, the biases are named on
the summary rather than left for the reader to remember. With unit dummies
the estimator is least-squares dummy variables and the lag coefficients
carry the Nickell bias, of order :math:`1/T` in the shortest unit, downward
for a positive own-lag, and unaffected by the number of units; the summary
prints the shortest :math:`T` next to the warning. Without them,
heterogeneous levels load onto the lags and the pooled slopes are biased
toward persistence, which on a panel of units with different means can
nearly double an own-lag coefficient. The GMM-in-differences estimators
that avoid both are not implemented here; this module is the honest
least-squares baseline they are compared against.

Layout. :class:`PanelVAR` is a thin ``fit()`` over
``_PanelVectorAutoRegressionModel`` in ``_internals``, which validates the
panel through ``validate_panel`` in ``_core`` -- a balanced
``(n_units, nobs, k)`` array or a sequence of ``(nobs_i, k)`` arrays --
refuses a pooled constant alongside unit dummies, builds each unit's
deterministic block and lag matrix separately in ``_design`` with the trend
restarting inside every unit, and stacks them for the base's least squares
and Gaussian moments. :class:`PanelVARResult` is a full ``_VectorResult``
with the summary, comparison, inference and propagation mixins, overriding
``_sample_blocks`` so the inference layer never crosses a unit, and adding
the per-unit accessors, the unit effects and the unit-specific forecast.
The single-unit model is
:mod:`~cultivars.multivariate.reduced_form.vector_autoregression`; the
many-unit system that links heterogeneous units instead of pooling them is
:mod:`~cultivars.multivariate.reduced_form.closed_global`.

References:
    Holtz-Eakin, D., Newey, W., & Rosen, H. S. (1988). Estimating vector
    autoregressions with panel data. *Econometrica*, 56(6), 1371-1395.

    Nickell, S. (1981). Biases in dynamic models with fixed effects.
    *Econometrica*, 49(6), 1417-1426.

    Arellano, M., & Bond, S. (1991). Some tests of specification for panel
    data: Monte Carlo evidence and an application to employment equations.
    *Review of Economic Studies*, 58(2), 277-297.

    Canova, F., & Ciccarelli, M. (2013). Panel vector autoregressive models:
    A survey. In *VAR Models in Macroeconomics -- New Developments and
    Applications* (Advances in Econometrics, Vol. 32, pp. 205-246). Emerald.

Example:
    Six units of unequal length share one bivariate autoregression around
    unit means that differ by several standard deviations. The fixed-effects
    fit recovers the common slopes up to the Nickell bias and forecasts each
    unit from its own intercept and history; the fit without unit effects
    reads the level differences as persistence, which is why ``"unit"`` is
    the default:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> A = np.array([[0.5, 0.2], [0.1, 0.4]])
    >>> units = []
    >>> for i in range(6):
    ...     mu = rng.normal(0.0, 2.0, size=2)
    ...     y = np.zeros((80 + 10 * i, 2))
    ...     y[0] = mu
    ...     for t in range(1, y.shape[0]):
    ...         y[t] = mu + A @ (y[t - 1] - mu) + rng.standard_normal(2)
    ...     units.append(y)
    >>> fixed = PanelVAR(units, order=1).fit()
    >>> fixed.n_units, fixed.nobs, fixed.min_time_dimension, fixed.is_stable
    (6, 624, 80, True)
    >>> bool(np.abs(fixed.coefficients[0] - A).max() < 0.1)
    True
    >>> pooled = PanelVAR(units, order=1, effects="none", trend="c").fit()
    >>> bool(np.abs(pooled.coefficients[0] - A).max() > 0.3)
    True
    >>> fixed.forecast(2, unit="unit4").shape, len(fixed.unit_effects)
    ((2, 2), 6)
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
)
from ..._internals import (
    _ComparisonMixin,
    _PanelVectorAutoRegressionModel,
    _SummaryMixin,
    _VectorAutoRegressionFit,
    _VectorInferenceMixin,
    _VectorPropagationMixin,
    _VectorResult,
)
from ...exceptions import SpecificationError

__all__ = ["PanelVAR", "PanelVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class PanelVARResult(
    _VectorResult,
    _SummaryMixin,
    _ComparisonMixin,
    _VectorInferenceMixin,
    _VectorPropagationMixin,
):
    r"""A fitted panel vector autoregression with pooled slopes.

    Every unit :math:`i` shares the autoregressive matrices and the innovation
    covariance and differs only in its intercept,

    .. math::

       y_{it} = \mu_i + \sum_{l=1}^{p} A_l\, y_{i,t-l} + u_{it},
       \qquad u_{it} \sim (0, \Sigma_u),
       \qquad i = 1, \dots, N,

    estimated by least squares on the stacked panel with one dummy per unit
    (``effects="unit"``) or with a single pooled deterministic block
    (``effects="none"``). Lags are built inside each unit, so the effective
    sample is :math:`\sum_i (T_i - p)` and no regressor crosses a unit
    boundary.

    The residual matrix is a stack of units rather than one series, and every
    inherited method that reads residual history reads it through
    :attr:`_sample_blocks`, so no lagged cross-product and no shock propagation
    ever crosses a unit boundary. ``irf``, ``fevd``, and ``stability_check``
    depend only on the pooled matrices and are unqualified.

    :meth:`forecast` is unit-specific and says so: the slopes are pooled but the
    intercept and the lag history are not, so there is no single path to return.

    Note:
        Two biases bracket this estimator and the summary names the one that
        applies. With unit dummies the lag coefficients carry the Nickell
        bias, of order :math:`1/T` in the shortest unit and downward for a
        positive own-lag; it does not shrink as units are added. Without
        them, heterogeneous intercepts load onto the lags and the pooled
        slopes are biased *toward* persistence -- on a panel whose units
        differ in level, ``effects="none"`` can nearly double an own-lag
        coefficient. The first bias is the price of removing the second, and
        the GMM-in-differences estimators that avoid both are not implemented
        here. Pooling itself is the stronger assumption: heterogeneous
        dynamics estimated as if common do not average to the mean dynamics.

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

    See Also:
        * :class:`PanelVAR` -- the model whose ``fit()`` returns this record.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
          -- the single-unit case, whose surface this record inherits.
        * :class:`~cultivars.multivariate.reduced_form.closed_global.GVARResult`
          -- the other many-unit system in the package, which lets every
          unit keep its own dynamics and links them instead of pooling.

    References:
        Holtz-Eakin, D., Newey, W., & Rosen, H. S. (1988). Estimating vector
        autoregressions with panel data. *Econometrica*, 56(6), 1371-1395.

        Nickell, S. (1981). Biases in dynamic models with fixed effects.
        *Econometrica*, 49(6), 1417-1426.

        Canova, F., & Ciccarelli, M. (2013). Panel vector autoregressive
        models: A survey. In *VAR Models in Macroeconomics -- New
        Developments and Applications* (Advances in Econometrics, Vol. 32,
        pp. 205-246). Emerald.

    Example:
        Six units of unequal length share one autoregression around unit
        means two standard deviations apart. With unit effects the pooled
        slopes land near the truth and the own-lag of the second variable
        shows the downward Nickell bias; without them the level differences
        masquerade as persistence:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> A = np.array([[0.5, 0.2], [0.1, 0.4]])
        >>> units = []
        >>> for i in range(6):
        ...     mu = rng.normal(0.0, 2.0, size=2)
        ...     y = np.zeros((80 + 10 * i, 2))
        ...     y[0] = mu
        ...     for t in range(1, y.shape[0]):
        ...         y[t] = mu + A @ (y[t - 1] - mu) + rng.standard_normal(2)
        ...     units.append(y)
        >>> res = PanelVAR(units, order=1).fit()
        >>> res.n_units, res.unit_lengths, res.nobs, res.min_time_dimension
        (6, (80, 90, 100, 110, 120, 130), 624, 80)
        >>> res.coefficients[0].round(2)
        array([[0.53, 0.22],
               [0.14, 0.33]])
        >>> pooled = PanelVAR(units, order=1, effects="none", trend="c").fit()
        >>> bool(pooled.coefficients[0, 0, 0] > 0.8), bool(pooled.coefficients[0, 1, 1] > 0.55)
        (True, True)
        >>> sorted(res.unit_effects)[:2], res.forecast(3, unit="unit2").shape
        (['unit1', 'unit2'], (3, 2))
    """

    coefficients: npt.NDArray[np.float64]
    """The ``(p, k, k)`` pooled autoregressive matrices."""

    deterministic: npt.NDArray[np.float64]
    """``(N, k)`` unit intercepts under fixed effects, else ``(d, k)`` pooled terms."""

    unit_names: tuple[str, ...]
    """Unit labels, in stacking order."""

    unit_lengths: tuple[int, ...]
    """Observations each unit contributed before its ``order`` leading rows became lags."""

    effects: str
    """``"unit"`` for one intercept per unit, ``"none"`` for a pooled deterministic block."""

    @classmethod
    def _from_fit(
        cls, fit: _VectorAutoRegressionFit, model: _PanelVectorAutoRegressionModel
    ) -> Self:
        """Assemble the public result from the internal fit and its model.

        Args:
            fit: The packed least-squares fit on the stacked design.
            model: The model that produced it, read for the stacked sample,
                the labels, the unit layout, the effects and the order.

        Returns:
            A populated result.

        Example:
            >>> import numpy as np
            >>> panel = np.random.default_rng(0).standard_normal((3, 40, 2))
            >>> model = PanelVAR(panel, order=1, unit_names=["a", "b", "c"])
            >>> res = PanelVARResult._from_fit(model._fit_family(), model)
            >>> res.unit_names, res.effects, res.deterministic.shape
            (('a', 'b', 'c'), 'unit', (3, 2))
        """
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
        """Number of units.

        Example:
            >>> import numpy as np
            >>> panel = np.random.default_rng(0).standard_normal((4, 40, 2))
            >>> PanelVAR(panel, order=1).fit().n_units
            4
        """
        return len(self.unit_names)

    @property
    def min_time_dimension(self) -> int:
        """Observations in the shortest unit, which is what the Nickell bias tracks.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> units = [rng.standard_normal((n, 2)) for n in (60, 45, 90)]
            >>> PanelVAR(units, order=1).fit().min_time_dimension
            45
        """
        return min(self.unit_lengths)

    @property
    def _sample_blocks(self) -> tuple[tuple[int, int], ...]:
        """One half-open residual span per unit, in stacking order.

        The inference mixin reads residual history through these spans, so
        lagged cross-products never pair the last residual of one unit with
        the first of the next.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> units = [rng.standard_normal((n, 2)) for n in (60, 45)]
            >>> PanelVAR(units, order=2).fit()._sample_blocks
            ((0, 58), (58, 101))
        """
        spans: list[tuple[int, int]] = []
        cursor = 0
        for length in self.unit_lengths:
            span = length - self.order
            spans.append((cursor, cursor + span))
            cursor += span
        return tuple(spans)

    def _unit_index(self, unit: str) -> int:
        """Position of a unit, or a failure naming the ones that exist.

        Args:
            unit: One of :attr:`unit_names`.

        Returns:
            The unit's index into :attr:`unit_names`, :attr:`unit_lengths`
            and :attr:`_sample_blocks`.

        Raises:
            SpecificationError: If the unit is not one of :attr:`unit_names`.

        Example:
            >>> import numpy as np
            >>> panel = np.random.default_rng(0).standard_normal((3, 40, 2))
            >>> res = PanelVAR(panel, order=1, unit_names=["a", "b", "c"]).fit()
            >>> res._unit_index("b")
            1
            >>> res._unit_index("d")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: unknown unit 'd'; expected one of ('a', ...
        """
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

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> units = [rng.standard_normal((n, 2)) for n in (60, 45)]
            >>> res = PanelVAR(units, order=1).fit()
            >>> series = res.unit_series("unit2")
            >>> series.shape, bool(np.allclose(series, units[1]))
            ((45, 2), True)
            >>> bool(np.shares_memory(series, res.endog))
            True
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

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> units = [rng.standard_normal((n, 2)) for n in (60, 45)]
            >>> res = PanelVAR(units, order=1).fit()
            >>> res.unit_residuals("unit2").shape
            (44, 2)
            >>> bool(np.allclose(res.unit_residuals("unit2"), res.resid[59:]))
            True
        """
        lo, hi = self._sample_blocks[self._unit_index(unit)]
        return self.resid[lo:hi]

    def _deterministic_labels(self) -> tuple[str, ...]:
        """One intercept name per unit under fixed effects, otherwise the trend block.

        Example:
            >>> import numpy as np
            >>> panel = np.random.default_rng(0).standard_normal((2, 40, 2))
            >>> PanelVAR(panel, order=1, unit_names=["a", "b"]).fit()._deterministic_labels()
            ('const[a]', 'const[b]')
            >>> PanelVAR(panel, order=1, effects="none", trend="ct").fit()._deterministic_labels()
            ('const', 'trend')
        """
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

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> units = [rng.standard_normal((60, 2)) + shift for shift in (0.0, 5.0)]
            >>> res = PanelVAR(units, order=1).fit()
            >>> effects = res.unit_effects
            >>> sorted(effects), bool(np.all(effects["unit2"] > effects["unit1"] + 2.0))
            (['unit1', 'unit2'], True)
            >>> pooled = PanelVAR(units, order=1, effects="none", trend="c").fit()
            >>> pooled.unit_effects  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: this model was fitted with effects='none', ...
        """
        if self.effects != "unit":
            raise SpecificationError(
                "this model was fitted with effects='none', so there are no unit "
                "intercepts; the pooled deterministic terms are in params."
            )
        return {name: self.deterministic[i] for i, name in enumerate(self.unit_names)}

    def forecast(self, steps: int = 1, *, unit: str | None = None) -> npt.NDArray[np.float64]:
        """Point forecasts for one unit.

        Iterates the pooled autoregression from the unit's own last ``order``
        observations, with the unit's intercept under fixed effects or the
        pooled deterministic terms extended past that unit's sample
        otherwise.

        Args:
            steps: Horizon.
            unit: Which unit to forecast. Required.

        Returns:
            A ``(steps, k)`` array of conditional means.

        Raises:
            SpecificationError: If ``steps`` is not positive, ``unit`` is
                omitted, or the unit is unknown.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> units = [rng.standard_normal((60, 2)) + shift for shift in (0.0, 5.0)]
            >>> res = PanelVAR(units, order=1).fit()
            >>> path = res.forecast(2, unit="unit2")
            >>> first = res.unit_effects["unit2"] + res.coefficients[0] @ units[1][-1]
            >>> path.shape, bool(np.allclose(path[0], first))
            ((2, 2), True)
            >>> res.forecast(2)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: a panel forecast is unit-specific: pass ...
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
        """The pooled lag coefficients alone, without the unit intercepts.

        Example:
            >>> import numpy as np
            >>> panel = np.random.default_rng(0).standard_normal((3, 40, 2))
            >>> res = PanelVAR(panel, order=1).fit()
            >>> sorted(res.slopes), len(res.params)
            (['y1: y1.L1', 'y1: y2.L1', 'y2: y1.L1', 'y2: y2.L1'], 10)
        """
        keep = set(self._lag_labels())
        return {
            name: value for name, value in self.params.items() if name.split(": ", 1)[1] in keep
        }

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table.

        Example:
            >>> import numpy as np
            >>> panel = np.random.default_rng(0).standard_normal((3, 40, 2))
            >>> PanelVAR(panel, order=2).fit()._comparison_label()
            'PanelVAR(2, effects=unit)'
        """
        return f"PanelVAR({self.order}, effects={self.effects})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Under unit effects the coefficient rows are the pooled slopes only --
        the ``N`` intercepts are nuisance parameters, listed in
        :attr:`unit_effects` -- and a Nickell-bias note names the shortest
        unit; under ``effects="none"`` every parameter is listed. The
        remaining notes state the pooling assumption, the within-unit lag
        construction and the Cholesky ordering, with an unstable system
        flagged first.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> units = [rng.standard_normal((n, 2)) for n in (60, 45, 90)]
            >>> table = PanelVAR(units, order=1).fit()._summary_table()
            >>> table.title, dict(table.metadata)["Shortest T"], len(table.rows)
            ('Panel VAR(1) Results', '45', 4)
            >>> table.notes[3][:14]
            'Nickell bias: '
        """
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
    r"""Panel vector autoregression with pooled slopes and unit fixed effects.

    Every unit shares :math:`A_1, \dots, A_p` and :math:`\Sigma_u`; only the
    intercept may differ, as one dummy per unit (``effects="unit"``, the
    default, with ``trend`` forced to ``"n"`` because unit dummies already
    span the constant) or as a single pooled deterministic block
    (``effects="none"``). The panel may be balanced -- an
    ``(n_units, nobs, k)`` array -- or a sequence of ``(nobs_i, k)`` arrays
    of unequal length; lags are built inside each unit and never across a
    boundary, and the shortest unit must still supply ``order`` lags.

    Attributes:
        _endog: The units stacked into one ``(sum T_i, k)`` panel.
        _names: Variable labels.
        _order: Autoregressive order, common to all units.
        _trend: Deterministic terms, ``"n"`` under unit effects.
        _prior: The base slot for a shrinkage prior, unused by this family.
        _units: The validated per-unit arrays.
        _lengths: Observations per unit.
        _unit_names: Unit labels.
        _effects: ``"unit"`` or ``"none"``.

    See Also:
        * :class:`PanelVARResult` -- the record ``fit()`` returns.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
          -- the single-unit model.
        * :class:`~cultivars.multivariate.reduced_form.closed_global.GVAR`
          -- the alternative when units should keep their own dynamics and
          be linked rather than pooled.

    References:
        Holtz-Eakin, D., Newey, W., & Rosen, H. S. (1988). Estimating vector
        autoregressions with panel data. *Econometrica*, 56(6), 1371-1395.

        Nickell, S. (1981). Biases in dynamic models with fixed effects.
        *Econometrica*, 49(6), 1417-1426.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> panel = rng.standard_normal((5, 60, 2))
        >>> res = PanelVAR(panel, order=1).fit()
        >>> res.forecast(3, unit="unit1").shape
        (3, 2)
    """

    __slots__ = ()

    def fit(self) -> PanelVARResult:
        """Estimate the pooled system and return the fitted result.

        Returns:
            A :class:`PanelVARResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> units = [rng.standard_normal((n, 2)) for n in (60, 45, 90)]
            >>> res = PanelVAR(units, order=2, unit_names=["de", "fr", "it"]).fit()
            >>> res.unit_names, res.nobs, res.deterministic.shape, res.coefficients.shape
            (('de', 'fr', 'it'), 189, (3, 2), (2, 2, 2))
        """
        return PanelVARResult._from_fit(self._fit_family(), self)
