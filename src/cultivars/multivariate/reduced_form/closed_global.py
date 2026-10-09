# filepath: /src/cultivars/multivariate/reduced_form/closed_global.py
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
r"""The global VAR: conditional units closed into one system by a weight matrix.

A global VAR is the answer to a counting problem. A panel of :math:`N`
units with :math:`k_i` variables each has :math:`k = \sum_i k_i` variables,
and an unrestricted VAR on all of them has :math:`p k^2` autoregressive
coefficients that no sample of macro data can estimate. The GVAR
(Pesaran, Schuermann and Weiner, 2004) spends those degrees of freedom
differently: each unit is a small conditional model of its own variables
given a weighted average of the others',

.. math::

   y_{it} = \sum_{l=1}^{p_i} \Phi_{il}\, y_{i,t-l}
   + \Lambda_{i0}\, y^*_{it} + \sum_{l=1}^{q_i} \Lambda_{il}\, y^*_{i,t-l}
   + d_{it} + u_{it},
   \qquad y^*_{it} = \sum_{j} w_{ij}\, y_{jt},

estimated unit by unit, and the restriction that the rest of the world
enters only through :math:`y^*_{it}` is what makes the counting work. Each
unit alone is open -- it has no equation for the foreign aggregate it
conditions on -- but substituting :math:`z_{it} = (y_{it}', y^{*\prime}_{it})'
= W_i x_t` and stacking the units gives the square system
:math:`G_0 x_t = d_t + \sum_l G_l x_{t-l} + u_t`, and inverting
:math:`G_0` closes it: every variable in the panel now has a law of motion,
so the global system can be propagated, decomposed and forecast like any
VAR.

Two commitments shape the surface. First, the closing step is algebra, not
estimation, and the objects say so. :class:`GVAR` exposes ``solve()``
rather than ``fit``, and :class:`GVARResult` stands beside the estimated
results rather than descending from them: it has no design matrix,
likelihood, parameter count or coefficient table, because an assembled
system has none, and it refuses to carry hollow versions. What it does
carry is the full propagation surface -- moving-average representation,
impulse responses, variance decompositions, forecasts -- and the
closed-system contract, so a global system feeds straight into
:class:`~cultivars.multivariate.large_dim.spillover.Spillover` for
connectedness across units. Second, the condition the construction rests
on is tested, not assumed. Least squares on a unit equation is consistent
only if the foreign aggregate is weakly exogenous for that unit's
parameters, and that is a property of the data: a dominant unit whose
partners adjust toward it fails it, by design, and should be modelled
with a reduced foreign set. :meth:`GVARResult.weak_exogeneity_test` runs
the Johansen-style auxiliary regression for an error-correction unit,
:meth:`GVARResult.exogeneity_table` runs it for every unit that supports
it, and the summary names any unit that fails. The other diagnostic the
result keeps is :attr:`GVARResult.linkage_condition`, the condition number
of :math:`G_0`, because a weight matrix that gives two units nearly the
same foreign aggregate makes the closing solve ill-posed without making it
fail.

Layout. The units are ordinary conditional results from this package:
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARX`
in levels or
:class:`~cultivars.multivariate.reduced_form.error_correction.VECMX` in
error-correction form, each fitted against the aggregate its own row of
the weight matrix defines. :class:`GVAR` validates the weight matrix
through ``validate_weights`` in ``_core`` -- zero diagonal, non-negative
entries, unit row sums -- maps global columns to units and shared
variable labels, and in ``solve()`` reads every unit through its
``to_varx()`` levels form into ``solve_global`` in ``_internals``, which
builds the link matrices with ``link_matrix`` from ``_core``, stacks and
inverts :math:`G_0`, and returns the solved coefficient stack. The unit
residuals are aligned on the common sample, rotated through
:math:`G_0^{-1}`, and packed with the panel into :class:`GVARResult`,
which takes its summary from ``_SummaryMixin`` and its propagation surface
from ``_VectorPropagationMixin``. The estimated closed systems the global
one stands beside are :mod:`~cultivars.multivariate.reduced_form.vector_autoregression`
and :mod:`~cultivars.multivariate.reduced_form.error_correction`; the
shrinkage route to the same counting problem is
:mod:`~cultivars.multivariate.large_dim`.

References:
    Pesaran, M. H., Schuermann, T., & Weiner, S. M. (2004). Modeling
    regional interdependencies using a global error-correcting
    macroeconometric model. *Journal of Business & Economic Statistics*,
    22(2), 129-162.

    Dees, S., di Mauro, F., Pesaran, M. H., & Smith, L. V. (2007).
    Exploring the international linkages of the euro area: A global VAR
    analysis. *Journal of Applied Econometrics*, 22(1), 1-38.

    Chudik, A., & Pesaran, M. H. (2016). Theory and practice of GVAR
    modelling. *Journal of Economic Surveys*, 30(1), 165-197.

    Johansen, S. (1992). Cointegration in partial systems and the
    efficiency of single-equation analysis. *Journal of Econometrics*,
    52(3), 389-402.

Example:
    A dominant unit and two satellites that error-correct toward it, each
    fitted as a conditional error-correction model against its own
    trade-weighted aggregate and closed into one system. The satellites
    pass the weak-exogeneity test and the dominant unit fails it, which is
    the right reading: its partners adjust to it, not the other way round.
    Fed to the spillover machinery, the closed system names the dominant
    unit as the only net transmitter:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.error_correction import VECMX
    >>> from cultivars.multivariate.large_dim.spillover import Spillover
    >>> rng = np.random.default_rng(0)
    >>> us = np.cumsum(rng.standard_normal(400))
    >>> eu, jp = np.zeros(400), np.zeros(400)
    >>> for t in range(1, 400):
    ...     eu[t] = eu[t - 1] + 0.3 * (us[t - 1] - eu[t - 1]) + 0.6 * rng.standard_normal()
    ...     jp[t] = jp[t - 1] + 0.2 * (us[t - 1] - jp[t - 1]) + 0.6 * rng.standard_normal()
    >>> panel = np.column_stack([us, eu, jp])
    >>> weights = np.array([[0.0, 0.5, 0.5], [0.7, 0.0, 0.3], [0.7, 0.3, 0.0]])
    >>> units = [
    ...     VECMX(panel[:, [i]], (panel @ weights[i])[:, None], order=1, rank=1).fit()
    ...     for i in range(3)
    ... ]
    >>> names = ("us", "eu", "jp")
    >>> res = GVAR(units, weights, variable_labels=["p"] * 3, unit_names=names).solve()
    >>> res.names, res.is_stable
    (('us.p', 'eu.p', 'jp.p'), True)
    >>> [(row[0], row[-1]) for row in res.exogeneity_table().rows]
    [('us', 'REJECT'), ('eu', ''), ('jp', '')]
    >>> network = Spillover(res, horizon=10).compute()
    >>> network.transmitters(), bool(network.net[0] > 100.0)
    (('us.p',), True)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from ...engine._core import SummaryTable, validate_weights
from ...engine._internals import _SummaryMixin, _VectorPropagationMixin, _WaldTest, solve_global
from ...exceptions import SpecificationError
from .error_correction import VECMXResult
from .vector_autoregression import VARXResult

__all__ = ["GVAR", "GVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class GVARResult(_SummaryMixin, _VectorPropagationMixin):
    r"""A closed global system assembled from conditional units.

    Each unit :math:`i` was fitted on its own variables :math:`y_{it}` with a
    weighted average of the other units' variables,
    :math:`y^*_{it} = \sum_j w_{ij}\, y_{jt}`, entering as the exogenous block,
    so on its own it has no law of motion for the foreign variables it
    conditions on. Writing every unit in levels against the global vector
    :math:`x_t = (y_{1t}', \dots, y_{Nt}')'` through its link matrix
    :math:`z_{it} = (y_{it}', y^{*\prime}_{it})' = W_i x_t` and stacking,

    .. math::

       G_0\, x_t = d_t + \sum_{l=1}^{p} G_l\, x_{t-l} + u_t,
       \qquad
       x_t = G_0^{-1} d_t + \sum_{l=1}^{p} G_0^{-1} G_l\, x_{t-l} + G_0^{-1} u_t,

    where :math:`G_0` stacks :math:`[I, -\Lambda_{i0}] W_i` and :math:`G_l`
    stacks :math:`[\Phi_{il}, \Lambda_{il}] W_i`. The inversion is the moment
    the system closes: individually no unit explains its foreign aggregate,
    jointly they all do, because one unit's foreign variables are other units'
    domestic ones. This record holds :math:`G_0` as :attr:`contemporaneous`,
    the solved stack :math:`G_0^{-1} G_l` as :attr:`coefficients`, and the
    rotated innovations :math:`G_0^{-1} u_t` as :attr:`resid`.

    Deliberately not a ``_VectorResult``. Every other result in the package
    was estimated: it has one design matrix, one likelihood, one parameter
    count. This one was *assembled* -- the units were estimated separately and
    then linked by an algebraic solve -- so it has none of those three, and
    giving it hollow versions to fit the hierarchy would be worse than standing
    outside it. What it does have is a law of motion for every variable, which
    is why it carries the propagation surface the units individually refused
    and satisfies the closed-system contract they could not.

    Note:
        The assembly is exact rather than approximate: substituting each
        unit's link matrix into its own equations and stacking gives a square
        system whose solution reproduces every unit equation to machine
        precision, so ``contemporaneous @ resid.T`` is the stacked unit
        residuals and no unit's coefficients are altered. What the assembly
        cannot do is make the units' coefficients correct. A unit's foreign
        aggregate contains other units' contemporaneous variables, which the
        global solve makes functions of that unit's own innovations, so least
        squares on a unit equation is consistent only if the aggregate is
        weakly exogenous for it. That is an assumption about the data, not a
        property of the estimator, and the bias it causes does not shrink as
        units are added. :meth:`weak_exogeneity_test` is therefore not an
        optional diagnostic here -- it is the condition under which everything
        else on this object means what it says, and the summary names any unit
        that fails it. Two further conventions: the global ``order`` is the
        deepest lag any unit uses in its levels form, with shorter units
        padded by zero blocks, and ``trend`` is read off the widest
        deterministic block among the units.

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
            system.
        weights: The validated cross-unit weight matrix.
        sigma_u: Covariance of the global reduced-form innovations.
        resid: The global reduced-form innovations on the common sample.
        nobs: Rows of the common sample.
        units: The fitted unit results, in global column order.

    See Also:
        * :class:`GVAR` -- the assembler that builds this record from fitted
          units and a weight matrix through ``solve()``.
        * :class:`~cultivars.multivariate.reduced_form.error_correction.VECMXResult`
          -- the conditional error-correction unit, the only kind whose
          weak-exogeneity assumption can be tested here.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARXResult`
          -- the conditional levels unit, linked through the levels form it
          already is.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
          -- the estimated closed system this record stands beside without
          descending from.

    References:
        Pesaran, M. H., Schuermann, T., & Weiner, S. M. (2004). Modeling
        regional interdependencies using a global error-correcting
        macroeconometric model. *Journal of Business & Economic Statistics*,
        22(2), 129-162.

        Dees, S., di Mauro, F., Pesaran, M. H., & Smith, L. V. (2007).
        Exploring the international linkages of the euro area: A global VAR
        analysis. *Journal of Applied Econometrics*, 22(1), 1-38.

        Chudik, A., & Pesaran, M. H. (2016). Theory and practice of GVAR
        modelling. *Journal of Economic Surveys*, 30(1), 165-197.

    Example:
        A dominant unit and a satellite: the world price is a random walk,
        the satellite price error-corrects toward it. Each unit is fitted as
        a conditional error-correction model against the other, the weight
        matrix swaps them, and the solve closes the pair. The rotation back
        through the linkage matrix recovers the unit residuals exactly, and
        the generalized response of the satellite to a world shock climbs
        toward one as the gap closes:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.error_correction import VECMX
        >>> rng = np.random.default_rng(0)
        >>> world, sat = np.zeros(400), np.zeros(400)
        >>> for t in range(1, 400):
        ...     world[t] = world[t - 1] + rng.standard_normal()
        ...     gap = world[t - 1] - sat[t - 1]
        ...     sat[t] = sat[t - 1] + 0.3 * gap + 0.5 * rng.standard_normal()
        >>> units = [
        ...     VECMX(world[:, None], sat[:, None], order=1, rank=1).fit(),
        ...     VECMX(sat[:, None], world[:, None], order=1, rank=1).fit(),
        ... ]
        >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
        >>> names = ("world", "sat")
        >>> res = GVAR(units, swap, variable_labels=["p", "p"], unit_names=names).solve()
        >>> res.names, res.order, res.nobs
        (('world.p', 'sat.p'), 1, 399)
        >>> stacked = np.hstack([unit.resid[-res.nobs :] for unit in res.units])
        >>> bool(np.allclose(res.contemporaneous @ res.resid.T, stacked.T))
        True
        >>> response = res.generalized_irf(8)[:, 1, 0]
        >>> bool(response[1] > 0.25), bool(response[-1] > 0.9)
        (True, True)
        >>> res.exogeneity_table().metadata[2]
        ('Rejections', '1')
    """

    endog: npt.NDArray[np.float64]
    """The global panel ``(n, k)``, every unit's ``endog`` side by side in unit order."""

    names: tuple[str, ...]
    """Global column labels ``"unit.variable"``, built from the unit names and the labels."""

    unit_names: tuple[str, ...]
    """Unit labels in global column order; the keys every per-unit method accepts."""

    unit_sizes: tuple[int, ...]
    """Variables each unit contributes; the partition of the ``k`` global columns."""

    order: int
    """Lags of the global system: the deepest levels lag any unit uses."""

    trend: str
    """Deterministic specification of the global representation, ``"n"``, ``"c"`` or ``"ct"``."""

    coefficients: npt.NDArray[np.float64]
    r"""The ``(p, k, k)`` solved autoregressive stack :math:`G_0^{-1} G_l`."""

    deterministic: npt.NDArray[np.float64]
    """The ``(d, k)`` global deterministic coefficients, one row per deterministic term."""

    contemporaneous: npt.NDArray[np.float64]
    r"""The ``(k, k)`` linkage matrix :math:`G_0` whose inverse closed the system.

    Retained because its conditioning is the honest diagnostic for a badly
    specified weight matrix; see :attr:`linkage_condition`.
    """

    weights: npt.NDArray[np.float64]
    """The validated ``(n_units, n_units)`` cross-unit weight matrix the aggregates used."""

    sigma_u: npt.NDArray[np.float64]
    """Covariance of the global reduced-form innovations.

    The cross product of :attr:`resid` over the common sample less the widest
    unit design, the single degrees-of-freedom correction the system has.
    """

    resid: npt.NDArray[np.float64]
    r"""The ``(nobs, k)`` global innovations :math:`G_0^{-1} u_t` on the common sample."""

    nobs: int
    """Rows of the common sample: the shortest unit residual series."""

    units: tuple[VARXResult | VECMXResult, ...]
    """The fitted unit results in global column order, exactly as they were passed in."""

    @property
    def n_units(self) -> int:
        """Units the system links.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> units = [VARX(y[:, [i]], y[:, [1 - i]], order=1).fit() for i in (0, 1)]
            >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> names = ("us", "eu")
            >>> res = GVAR(units, swap, variable_labels=["p", "p"], unit_names=names).solve()
            >>> res.n_units
            2
        """
        return len(self.unit_names)

    @property
    def sigma_ml(self) -> npt.NDArray[np.float64]:
        """Innovation covariance without the degrees-of-freedom correction.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> units = [VARX(y[:, [i]], y[:, [1 - i]], order=1).fit() for i in (0, 1)]
            >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> names = ("us", "eu")
            >>> res = GVAR(units, swap, variable_labels=["p", "p"], unit_names=names).solve()
            >>> bool(np.allclose(res.sigma_ml * res.nobs, res.resid.T @ res.resid))
            True
            >>> bool(np.all(np.diag(res.sigma_ml) < np.diag(res.sigma_u)))
            True
        """
        return self.resid.T @ self.resid / self.nobs

    def _unit_index(self, unit: str) -> int:
        """Position of a unit, or a failure naming the ones that exist.

        Args:
            unit: One of :attr:`unit_names`.

        Returns:
            The unit's index into :attr:`unit_names`, :attr:`unit_sizes` and
            :attr:`units`.

        Raises:
            SpecificationError: If the unit is not one of :attr:`unit_names`.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> units = [VARX(y[:, [i]], y[:, [1 - i]], order=1).fit() for i in (0, 1)]
            >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> names = ("us", "eu")
            >>> res = GVAR(units, swap, variable_labels=["p", "p"], unit_names=names).solve()
            >>> res._unit_index("eu")
            1
            >>> res._unit_index("uk")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: unknown unit 'uk'; expected one of ('us', ...
        """
        if unit not in self.unit_names:
            raise SpecificationError(f"unknown unit {unit!r}; expected one of {self.unit_names}.")
        return self.unit_names.index(unit)

    def unit_columns(self, unit: str) -> tuple[int, ...]:
        """Global column indices the named unit owns.

        Args:
            unit: One of :attr:`unit_names`.

        Returns:
            The contiguous run of global columns the unit's variables occupy,
            which indexes :attr:`names`, :attr:`endog`, :attr:`coefficients`
            and every propagation array.

        Raises:
            SpecificationError: If the unit is unknown.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> units = [VARX(y[:, [i]], y[:, [1 - i]], order=1).fit() for i in (0, 1)]
            >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> names = ("us", "eu")
            >>> res = GVAR(units, swap, variable_labels=["p", "p"], unit_names=names).solve()
            >>> res.unit_columns("eu")
            (1,)
            >>> [res.names[c] for c in res.unit_columns("us")]
            ['us.p']
        """
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
        across many units is the usual cause. The summary flags the system
        when this exceeds ``1e8``.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> units = [VARX(y[:, [i]], y[:, [1 - i]], order=1).fit() for i in (0, 1)]
            >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> names = ("us", "eu")
            >>> res = GVAR(units, swap, variable_labels=["p", "p"], unit_names=names).solve()
            >>> bool(res.linkage_condition < 2.0)
            True
            >>> cond = np.linalg.cond(res.contemporaneous)
            >>> bool(np.isclose(res.linkage_condition, cond))
            True
        """
        return float(np.linalg.cond(self.contemporaneous))

    def weak_exogeneity_test(self, unit: str) -> tuple[_WaldTest, ...]:
        r"""Test that a unit's foreign aggregates ignore its disequilibria.

        The condition the whole construction rests on. If a unit's foreign
        aggregate responds to that unit's own error-correction terms then the
        aggregate is not weakly exogenous, the unit's least-squares estimates
        are inconsistent, and every global quantity assembled from them
        inherits the bias.

        The auxiliary regression puts each foreign variable's change on the
        unit's own short-run design with the contemporaneous foreign block
        removed, and tests that the error-correction coefficients are jointly
        zero,

        .. math::

           \Delta y^*_{jt} = c + \gamma'\, \hat\xi_{i,t-1} + \text{(lagged
           differences)} + \epsilon_t, \qquad H_0 : \gamma = 0,

        with :math:`\hat\xi_{i,t-1} = \hat\beta_i' z_{i,t-1}` the unit's
        estimated disequilibria and the statistic
        :math:`n\,(\mathrm{RSS}_0 - \mathrm{RSS}_1)/\mathrm{RSS}_1` referred
        to :math:`\chi^2(r)`. Removing the contemporaneous block is not a
        detail: leaving it in would put the dependent variable on both sides,
        and testing the unit's *residuals* instead -- the obvious first idea
        -- has no power at all, because least squares already made them
        orthogonal to the foreign block by construction.

        Args:
            unit: One of :attr:`unit_names`.

        Returns:
            One test per foreign variable, in that unit's exogenous order.

        Raises:
            SpecificationError: If the unit is unknown, was fitted in levels
                rather than as an error-correction model, or has rank zero.

        Note:
            The test conditions on the unit's own estimate of
            :math:`\beta_i`, so it is only as good as that estimate: a unit
            whose own variables do not error-correct identifies its
            cointegrating vector weakly, and its test should be read with
            that in mind. In the example the satellite error-corrects, its
            vector is well identified, the world price ignores the gap, and
            the test passes; for the world unit the satellite does respond to
            the gap and the test rejects, which is the right answer -- a unit
            that conditions on a variable adjusting to its own disequilibrium
            is misspecified as a conditional model, and only the satellite's
            estimates can be trusted as they stand.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.error_correction import VECMX
            >>> rng = np.random.default_rng(0)
            >>> world, sat = np.zeros(400), np.zeros(400)
            >>> for t in range(1, 400):
            ...     world[t] = world[t - 1] + rng.standard_normal()
            ...     gap = world[t - 1] - sat[t - 1]
            ...     sat[t] = sat[t - 1] + 0.3 * gap + 0.5 * rng.standard_normal()
            >>> units = [
            ...     VECMX(world[:, None], sat[:, None], order=1, rank=1).fit(),
            ...     VECMX(sat[:, None], world[:, None], order=1, rank=1).fit(),
            ... ]
            >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> names = ("world", "sat")
            >>> res = GVAR(units, swap, variable_labels=["p", "p"], unit_names=names).solve()
            >>> (test,) = res.weak_exogeneity_test("sat")
            >>> test.df, bool(test.pvalue > 0.05), test.null
            (1, True, 'x1 is weakly exogenous for sat')
            >>> (test,) = res.weak_exogeneity_test("world")
            >>> bool(test.reject(alpha=0.05))
            True
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
            >>> levels = VARX(world[:, None], sat[:, None], order=1, exog_order=1).fit()
            >>> mixed = GVAR([levels, units[1]], swap, variable_labels=["p", "p"]).solve()
            >>> mixed.weak_exogeneity_test("unit1")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: unit 'unit1' was fitted in levels, which ...
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
        out: list[_WaldTest] = []
        for column, label in enumerate(fitted.exog_names):
            target = changes[:, column]
            full_resid = target - auxiliary @ np.linalg.lstsq(auxiliary, target, rcond=None)[0]
            null_resid = target - restricted @ np.linalg.lstsq(restricted, target, rcond=None)[0]
            denominator = float(full_resid @ full_resid)
            statistic = rows * (float(null_resid @ null_resid) - denominator) / denominator
            out.append(
                _WaldTest(
                    statistic=statistic,
                    df=rank,
                    pvalue=float(chi2.sf(statistic, rank)),
                    null=f"{label} is weakly exogenous for {unit}",
                )
            )
        return tuple(out)

    def exogeneity_table(self, *, alpha: float = 0.05) -> SummaryTable:
        """Run the weak-exogeneity test on every unit that supports it.

        Units fitted in levels or at rank zero are skipped and named in a
        note rather than failing the table; a rejection anywhere puts a
        warning note first, because impulse responses from a system with a
        misspecified unit should not be read as they stand.

        Args:
            alpha: Size of each test, used for the ``REJECT`` column and the
                rejection count.

        Returns:
            A :class:`~cultivars.summary.SummaryTable` with one row per
            foreign variable per tested unit, metadata counting units, tests
            and rejections, and the notes described above.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.error_correction import VECMX
            >>> rng = np.random.default_rng(0)
            >>> world, sat = np.zeros(400), np.zeros(400)
            >>> for t in range(1, 400):
            ...     world[t] = world[t - 1] + rng.standard_normal()
            ...     gap = world[t - 1] - sat[t - 1]
            ...     sat[t] = sat[t - 1] + 0.3 * gap + 0.5 * rng.standard_normal()
            >>> units = [
            ...     VECMX(world[:, None], sat[:, None], order=1, rank=1).fit(),
            ...     VECMX(sat[:, None], world[:, None], order=1, rank=1).fit(),
            ... ]
            >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> names = ("world", "sat")
            >>> res = GVAR(units, swap, variable_labels=["p", "p"], unit_names=names).solve()
            >>> table = res.exogeneity_table()
            >>> table.columns
            ('unit', 'foreign variable', 'statistic', 'df', 'p-value', '')
            >>> [(row[0], row[-1]) for row in table.rows]
            [('world', 'REJECT'), ('sat', '')]
            >>> table.notes[0][:48]
            "NOT WEAKLY EXOGENOUS at the 5% level: ('world',)"
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
            >>> levels = VARX(world[:, None], sat[:, None], order=1, exog_order=1).fit()
            >>> mixed = GVAR([levels, units[1]], swap, variable_labels=["p", "p"]).solve()
            >>> mixed.exogeneity_table().metadata[1:3]
            (('Tested', '1'), ('Rejections', '0'))
            >>> mixed.exogeneity_table().notes[-1]
            "Units fitted in levels and therefore untestable: ('unit1',)."
        """
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
        """Short specification label.

        Returns:
            ``"GVAR(p, units=N)"``.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> units = [VARX(y[:, [i]], y[:, [1 - i]], order=1).fit() for i in (0, 1)]
            >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> names = ("us", "eu")
            >>> res = GVAR(units, swap, variable_labels=["p", "p"], unit_names=names).solve()
            >>> res._comparison_label()
            'GVAR(1, units=2)'
        """
        return f"GVAR({self.order}, units={self.n_units})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        One row per unit giving its width, order and foreign-block size; no
        coefficient table and no information criteria, because an assembled
        system has no likelihood to compute them from. The notes carry the
        stability verdict, the assembled-not-estimated statement, the
        weak-exogeneity warning and the advice to prefer the generalized
        propagation pair; an ill-conditioned linkage is flagged first.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> units = [VARX(y[:, [i]], y[:, [1 - i]], order=1).fit() for i in (0, 1)]
            >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> names = ("us", "eu")
            >>> res = GVAR(units, swap, variable_labels=["p", "p"], unit_names=names).solve()
            >>> table = res._summary_table()
            >>> table.title, table.columns
            ('GVAR(1) Results', ('unit', 'variables', 'order', 'foreign block'))
            >>> table.rows
            (('us', '1', '1', '1'), ('eu', '1', '1', '1'))
            >>> dict(table.metadata)["Variables"], table.notes[0].startswith("Stable: True")
            ('2', True)
        """
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
    r"""Link fitted conditional units into one closed global system.

    An assembler rather than an estimator, and the class says so by exposing
    :meth:`solve` instead of ``fit``. Every coefficient in the result was
    estimated by a unit; what happens here is the algebra that turns a
    collection of models each missing a law of motion for its own foreign
    variables into a single model that has one for everything. Unit
    :math:`i` was fitted on :math:`z_{it} = (y_{it}', y^{*\prime}_{it})'`
    with its foreign block the weighted average
    :math:`y^*_{it} = \sum_j w_{ij}\, y_{jt}` of the *same* variables in the
    other units; writing :math:`z_{it} = W_i x_t` against the global vector
    and stacking the units gives the square system :math:`G_0 x_t = d_t +
    \sum_l G_l x_{t-l} + u_t` that :meth:`solve` inverts.

    Units may be heterogeneous -- different orders, different variable counts,
    a mixture of levels and error-correction specifications -- because the
    linkage reads each through the levels form it implies rather than through
    the parameterization it was estimated in. What they may not be is loosely
    built: the assembly reproduces the data exactly only when each unit's
    exogenous block is precisely the aggregate the weight matrix and the
    variable labels describe -- one foreign series per own variable, in the
    unit's own variable order, averaged with that unit's row of ``weights``
    over the units whose labels match. The identity worth checking after a
    solve is the fitted one, ``endog[t] == deterministic + sum_l
    coefficients[l] @ endog[t - l] + resid[t]`` on the common sample; it
    holds to rounding when the aggregates were built as described and fails
    visibly when they were not.

    Attributes:
        _units: The fitted unit results, in global column order.
        _weights: The validated ``(n_units, n_units)`` weight matrix.
        _labels: One shared variable label per global column.
        _unit_names: One label per unit, unique.

    See Also:
        * :class:`GVARResult` -- the assembled record ``solve()`` returns.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARX`
          -- the conditional levels unit.
        * :class:`~cultivars.multivariate.reduced_form.error_correction.VECMX`
          -- the conditional error-correction unit, whose weak-exogeneity
          assumption the result can test.

    References:
        Pesaran, M. H., Schuermann, T., & Weiner, S. M. (2004). Modeling
        regional interdependencies using a global error-correcting
        macroeconometric model. *Journal of Business & Economic Statistics*,
        22(2), 129-162.

        Chudik, A., & Pesaran, M. H. (2016). Theory and practice of GVAR
        modelling. *Journal of Economic Surveys*, 30(1), 165-197.

    Example:
        Three units, one variable each, linked by a trade-weight matrix with
        a zero diagonal and unit row sums. Each unit is fitted against the
        aggregate its own weight row defines, and the assembled system
        reproduces the panel exactly and inherits the units' persistence:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((300, 3))
        >>> A = np.array([[0.5, 0.2, 0.1], [0.1, 0.5, 0.2], [0.2, 0.1, 0.5]])
        >>> for t in range(1, 300):
        ...     y[t] = A @ y[t - 1] + rng.standard_normal(3)
        >>> weights = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        >>> units = [
        ...     VARX(y[:, [i]], (y @ weights[i])[:, None], order=1).fit() for i in range(3)
        ... ]
        >>> names = ("us", "eu", "jp")
        >>> res = GVAR(units, weights, variable_labels=["gdp"] * 3, unit_names=names).solve()
        >>> res.names
        ('us.gdp', 'eu.gdp', 'jp.gdp')
        >>> fitted = res.endog[:-1] @ res.coefficients[0].T + res.deterministic[0]
        >>> bool(np.allclose(res.endog[1:], fitted + res.resid))
        True
        >>> res.is_stable, bool(np.all(np.diag(res.coefficients[0]) > 0.35))
        (True, True)
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
                foreign aggregates as the exogenous block: one foreign series
                per own variable, in the unit's own variable order, built
                with that unit's row of ``weights``.
            weights: An ``(n_units, n_units)`` cross-unit weight matrix. Row
                ``i`` says how unit ``i`` sees the rest of the world; it must
                have a zero diagonal, non-negative entries and unit row sums,
                and is checked by
                :func:`~cultivars._core._validators.validate_weights`.
            variable_labels: One entry per global column giving the *shared*
                variable identity, so that a unit's foreign aggregate averages
                like with like. Repeated across units: two entries labelled
                ``"gdp"`` in different units are the same variable seen in two
                places. A label no other unit carries gives that variable an
                all-zero foreign aggregate.
            unit_names: Unit labels. Defaults to ``unit1 ... unitN``.

        Raises:
            SpecificationError: If no units are given, the labels do not match
                the global width, the unit names repeat, or the weight matrix
                has a non-zero diagonal, negative entries or rows that do not
                sum to one.
            DimensionError: If the weight matrix is the wrong shape.
            NumericalError: If the weight matrix contains non-finite values.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> units = [VARX(y[:, [i]], y[:, [1 - i]], order=1).fit() for i in (0, 1)]
            >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> GVAR(units, swap, variable_labels=["p", "p"]).unit_of_column
            (0, 1)
            >>> GVAR(units, swap, variable_labels=["p"])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: variable_labels must have one entry per ...
            >>> GVAR(units, swap, variable_labels=["p", "p"], unit_names=("a", "a"))
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: unit_names must be unique; got ('a', 'a').
            >>> GVAR(units, np.eye(2), variable_labels=["p", "p"])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: weights must have a zero diagonal; units ...
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
        """Owning unit index for each global column.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
            >>> y = np.random.default_rng(0).standard_normal((200, 3))
            >>> wide = VARX(y[:, :2], y[:, 2:], order=1).fit()
            >>> narrow = VARX(y[:, 2:], y[:, :1], order=1).fit()
            >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> GVAR([wide, narrow], swap, variable_labels=["gdp", "inf", "gdp"]).unit_of_column
            (0, 0, 1)
        """
        return tuple(index for index, unit in enumerate(self._units) for _ in range(unit.k_endog))

    @property
    def variable_of_column(self) -> tuple[int, ...]:
        """Variable identity for each global column, as integer codes.

        Codes are assigned in order of first appearance of each label, so
        the first unit's variables are ``0, 1, ...`` and a later unit's
        column that repeats a label reuses its code.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
            >>> y = np.random.default_rng(0).standard_normal((200, 3))
            >>> wide = VARX(y[:, :2], y[:, 2:], order=1).fit()
            >>> narrow = VARX(y[:, 2:], y[:, :1], order=1).fit()
            >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> GVAR([wide, narrow], swap, variable_labels=["gdp", "inf", "gdp"]).variable_of_column
            (0, 1, 0)
        """
        order = {label: code for code, label in enumerate(dict.fromkeys(self._labels))}
        return tuple(order[label] for label in self._labels)

    def _stacked_innovations(
        self, linkage: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], int]:
        r"""Align the unit residuals and rotate them into global coordinates.

        Units may lose different numbers of leading observations to their own
        lag lengths, but every unit's residuals end on the same date, so the
        common sample is the last ``min`` rows. Rotating by the inverse linkage
        turns the unit innovations :math:`u_t` into the reduced-form
        innovations :math:`G_0^{-1} u_t` of the global system, which is what
        the moving-average representation propagates.

        Args:
            linkage: The ``(k, k)`` contemporaneous matrix :math:`G_0`.

        Returns:
            The ``(common, k)`` rotated innovations and the common sample
            length.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> deep = VARX(y[:, [0]], y[:, [1]], order=2).fit()
            >>> shallow = VARX(y[:, [1]], y[:, [0]], order=1).fit()
            >>> units = [deep, shallow]
            >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> system = GVAR(units, swap, variable_labels=["p", "p"])
            >>> innovations, common = system._stacked_innovations(np.eye(2))
            >>> innovations.shape, common, [unit.resid.shape[0] for unit in units]
            ((198, 2), 198, [198, 199])
            >>> bool(np.allclose(innovations[:, 1], units[1].resid[-198:, 0]))
            True
        """
        common = min(unit.resid.shape[0] for unit in self._units)
        stacked = np.hstack([unit.resid[-common:] for unit in self._units])
        return np.linalg.solve(linkage, stacked.T).T, common

    def solve(self) -> GVARResult:
        r"""Close the system and return the global result.

        Each unit is read through its ``to_varx()`` into the levels form it
        implies, the link matrices and the stacked :math:`G_0, G_l` are
        built and :math:`G_0` inverted by ``solve_global``, the unit
        residuals are aligned on the common sample and rotated through
        :math:`G_0^{-1}`, and the record is assembled. The global order is
        the deepest lag any unit uses, the trend label is read off the
        widest deterministic block, and the innovation covariance is divided
        by the common sample less the widest unit design.

        Returns:
            A :class:`GVARResult`.

        Raises:
            DimensionError: If the units do not partition the global columns.
            NumericalError: If the contemporaneous linkage is singular.

        Example:
            A levels unit of order two linked to an error-correction unit of
            order one; the global system takes the deeper order, the common
            sample is the shorter residual series, and the fitted identity
            holds exactly:

            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VARX
            >>> from cultivars.multivariate.reduced_form.error_correction import VECMX
            >>> rng = np.random.default_rng(0)
            >>> world, sat = np.zeros(400), np.zeros(400)
            >>> for t in range(1, 400):
            ...     world[t] = world[t - 1] + rng.standard_normal()
            ...     gap = world[t - 1] - sat[t - 1]
            ...     sat[t] = sat[t - 1] + 0.3 * gap + 0.5 * rng.standard_normal()
            >>> units = [
            ...     VARX(world[:, None], sat[:, None], order=2, exog_order=1).fit(),
            ...     VECMX(sat[:, None], world[:, None], order=1, rank=1).fit(),
            ... ]
            >>> swap = np.array([[0.0, 1.0], [1.0, 0.0]])
            >>> res = GVAR(units, swap, variable_labels=["p", "p"]).solve()
            >>> res.order, res.nobs, res.trend, res.unit_names
            (2, 398, 'c', ('unit1', 'unit2'))
            >>> x, (first, second) = res.endog, res.coefficients
            >>> fitted = x[1:-1] @ first.T + x[:-2] @ second.T + res.deterministic[0]
            >>> bool(np.allclose(x[2:], fitted + res.resid))
            True
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
