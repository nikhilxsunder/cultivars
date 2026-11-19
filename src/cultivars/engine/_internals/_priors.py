# filepath: /src/cultivars/engine/_internals/_priors.py
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
"""Priors: the moment-and-dummy interface for VAR coefficients, and the volatility hyperpriors.

Two unrelated kinds of prior live here. They share a module because both
are records that an estimator reads, not objects that estimate anything.

The first is the interface that every prior on a vector autoregression's
coefficients implements. A prior states itself as two ``(width, k)`` arrays,
a mean and a diagonal variance over the coefficient matrix, plus an optional
block of artificial observations for restrictions that no diagonal variance
can express. :class:`_PriorContext` carries what a prior must know about the
sample to produce those arrays. :class:`_Prior` is the abstract interface.
:class:`_CompositePrior` is what ``+`` returns: it pools moments as
independent information, so precisions add, and it stacks dummy rows.
:class:`_NoPrior` is the stated absence of a prior. :class:`_AdaptivePrior`
extends the interface with the hooks a Gibbs sampler drives when the
variances are latent states and not constants.

All of them index coefficients the same way. Rows are design columns: the
deterministic block first, then the lags ordered lag-major (every variable
at lag one, then every variable at lag two), then the exogenous regressors.
Columns are equations. Entry ``[n_deterministic + (l - 1) * k + j, i]`` is
therefore equation ``i``'s coefficient on lag ``l`` of variable ``j``. The
estimator that owns the design builds the context, so the layout is stated
once and no prior has to guess it.

The second kind is the pair of hyperprior records read by the
stochastic-volatility samplers: :class:`_VolatilityPrior` for the stationary
log-AR(1) law and :class:`_RandomWalkVolatilityPrior` for the random-walk
law. Each is a validated tuple of hyperparameters with a label for the
summary table and nothing else.

Layout. The concrete coefficient priors (Minnesota, normal-inverse-Wishart,
sum-of-coefficients, dummy-initial-observation, independent normal-Wishart,
and the four adaptive priors) live in :mod:`cultivars.bayes.priors`. That
module also re-exports five classes from here under public names: ``Prior``,
``PriorContext``, ``NoPrior``, ``VolatilityPrior``, and
``RandomWalkVolatilityPrior``. The context is built by the vector-model base
in ``_models``, and the samplers that read the moments and drive the
adaptive hooks are there too.

References:
    Banbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector
    auto regressions. *Journal of Applied Econometrics*, 25(1), 71-92.

    Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection for
    vector autoregressions. *The Review of Economics and Statistics*,
    97(2), 436-451.

    Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
    Likelihood inference and comparison with ARCH models. *The Review
    of Economic Studies*, 65(3), 361-393.

    Litterman, R. B. (1986). Forecasting with Bayesian vector
    autoregressions: Five years of experience. *Journal of Business &
    Economic Statistics*, 4(1), 25-38.

    Stock, J. H., & Watson, M. W. (2007). Why has U.S. inflation become
    harder to forecast? *Journal of Money, Credit and Banking*, 39(s1),
    3-33.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core._defaults import (
    _KSC_MU_PRIOR,
    _KSC_PHI_PRIOR,
    _KSC_SIGMA2_PRIOR,
    _UCSV_VOL_OF_VOL_PRIOR,
)
from .._core._validators import (
    _validate_hyperparameter_pair,
)

__all__ = [
    "_AdaptivePrior",
    "_CompositePrior",
    "_NoPrior",
    "_Prior",
    "_PriorContext",
    "_RandomWalkVolatilityPrior",
    "_VolatilityPrior",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class _PriorContext:
    """What a prior needs to know about the sample before it can state itself.

    Built once by the estimator and read by every prior in a stack, so two
    priors cannot disagree about the design's column order or a variable's
    scale. Everything here comes from the data or the specification; nothing
    comes from the prior.

    Attributes:
        k_endog: Number of modelled variables.
        order: Autoregressive order.
        scales: Per-variable residual scale, from ``_minnesota_scales`` in
            ``_core``.
        presample_mean: Mean of the first ``order`` observations, which the
            sum-of-coefficients restriction is centred on.
        k_exog: Exogenous regressors carried after the lag block.
        n_deterministic: Width of the leading deterministic block, which is a
            count rather than a flag because families disagree about it: none
            for a trendless specification, one for a constant, two for a
            constant and trend, and one per unit for a fixed-effects panel. A
            boolean would have been silently one column short for every
            ``"ct"`` model and off by ``N - 1`` for every panel, and a prior
            whose columns do not line up with the design is inert at the limits
            while still looking plausible in between -- which is worse than
            being wrong loudly.
        include_constant: Whether the design carries a leading intercept
            column. Dummy columns must match the design's own ordering, and a
            prior that guesses wrong is inert at the limits while still looking
            plausible in between -- which is worse than being wrong loudly.

    Note:
        Construction is keyword-only and validates nothing: the arrays are
        not checked against ``k_endog`` and the counts are not checked for
        sign. Because two fields are arrays, ``==`` between contexts raises
        ``ValueError`` and ``hash`` raises ``TypeError``.

        ``include_constant`` is carried but not consulted. Every prior in
        the package places its columns from ``n_deterministic`` alone, and
        nothing checks that the two fields agree.

    Example:
        A bivariate VAR(2) with a constant has five design columns, one
        deterministic column ahead of four lag columns:

        >>> import numpy as np
        >>> context = _PriorContext(
        ...     k_endog=2,
        ...     order=2,
        ...     scales=np.array([1.0, 4.0]),
        ...     presample_mean=np.array([0.5, 2.0]),
        ... )
        >>> context.width, context.lag_offset
        (5, 1)
    """

    k_endog: int
    """Number of modelled variables ``k``, which is also the number of equations.

    The second dimension of every moment array and the width of each lag.
    """
    order: int
    """Autoregressive order ``p``; the lag block is ``k_endog * order`` columns wide."""
    scales: npt.NDArray[np.float64]
    """Per-variable residual standard deviation ``s_i``, shape ``(k_endog,)``.

    Taken from univariate autoregressions, so the scale a prior is written
    against does not depend on the system it regularizes. The ratio
    ``s_i / s_j`` is what makes a cross-variable coefficient unit-free. The
    entries must be strictly positive: a zero makes that ratio infinite.
    """
    presample_mean: npt.NDArray[np.float64]
    """Mean of the first ``order`` observations of each variable, shape ``(k_endog,)``.

    The level at which the sum-of-coefficients rows are written.
    """
    k_exog: int = 0
    """Number of exogenous regressors, whose columns follow the lag block."""
    n_deterministic: int = 1
    """Width of the deterministic block that leads the design.

    Zero for a trendless specification, one for a constant, two for a
    constant and trend, one per unit for a fixed-effects panel. This is the
    count every prior uses to place its columns; :attr:`lag_offset` returns
    it.
    """
    include_constant: bool = True
    """Whether the design carries a leading intercept column.

    Informational only. No prior in the package reads it, and it is not
    checked against :attr:`n_deterministic`.
    """

    @property
    def width(self) -> int:
        """Design columns a prior's blocks must match.

        The deterministic block, then ``k_endog * order`` lag columns, then
        the exogenous columns. This is the first dimension of every moment
        array and the second dimension of every dummy design block.

        Returns:
            ``n_deterministic + k_endog * order + k_exog``.
        """
        return self.n_deterministic + self.k_endog * self.order + self.k_exog

    @property
    def lag_offset(self) -> int:
        """Design columns ahead of the endogenous lag block.

        Returns:
            The index of the first lag column, equal to ``n_deterministic``.
            Variable ``j`` at lag ``l`` (counting lags from one) sits at
            column ``lag_offset + (l - 1) * k_endog + j``.
        """
        return self.n_deterministic

    def _lead(self, rows: int, value: float = 0.0) -> npt.NDArray[np.float64]:
        """The deterministic columns a dummy block carries, one per term.

        Args:
            rows: Number of dummy rows being built.
            value: Entry to fill with. The default of zero leaves the
                deterministic terms out of the restriction the rows express.

        Returns:
            A ``(rows, n_deterministic)`` array filled with ``value``; it
            has no columns for a trendless specification.
        """
        return np.full((rows, self.n_deterministic), value, dtype=np.float64)


class _Prior(ABC):
    """A prior on a vector autoregression's coefficients.

    Stated as *moments* -- a mean and a variance over the coefficient matrix --
    rather than as artificial observations. The distinction is not stylistic.
    Banbura-Giannone-Reichlin dummy rows are convenient because stacking them
    under the sample turns the posterior into an ordinary least-squares solve,
    but that convenience is exactly what forces Litterman's cross-equation
    weight to one: a weight that differs between own and cross lags breaks the
    Kronecker structure the dummy form depends on. Choosing moments as the
    interface keeps that hyperparameter, and it is also what the sparse and
    global-local priors need, since a Gibbs sampler redraws their conditional
    variances every sweep and has nowhere to put that in a fixed block of rows.

    Dummy observations do not disappear -- a restriction on a *sum* of
    coefficients has no diagonal-variance form and must be rows -- so a prior
    supplies both hooks and leaves the one it does not use empty. That is what
    Giannone, Lenza and Primiceri do, and it is why the two coexist here.

    Addition composes, and composition flattens.

    Every array a prior returns is indexed the same way. Rows are design
    columns: the deterministic block, then the lags ordered lag-major (every
    variable at lag one, then every variable at lag two), then the exogenous
    regressors. Columns are equations. So entry ``[lag_offset + j, i]`` is
    equation ``i``'s coefficient on the first lag of variable ``j``.

    A subclass implements :meth:`coefficient_mean`,
    :meth:`coefficient_variance`, and :meth:`_label`, and overrides
    :meth:`dummy_observations` only if it restricts a combination of
    coefficients. An instance should be immutable and hold hyperparameters
    only. Everything that depends on the sample arrives through the
    :class:`_PriorContext`, which is what lets one prior object serve many
    fits.

    Note:
        ``+`` always returns a :class:`_CompositePrior`, even when one side
        is :class:`_NoPrior`. There is no ``__radd__``, so the built-in
        ``sum`` needs a prior as its start value:
        ``sum(priors, _NoPrior())``.

    Example:
        A ridge prior, the same variance on every coefficient, and what
        composing it with itself does to the moments:

        >>> import numpy as np
        >>> class Ridge(_Prior):
        ...     def coefficient_mean(self, context):
        ...         return np.zeros((context.width, context.k_endog))
        ...
        ...     def coefficient_variance(self, context):
        ...         return np.full((context.width, context.k_endog), 0.25)
        ...
        ...     def _label(self):
        ...         return "ridge(0.25)"
        >>> context = _PriorContext(
        ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
        ... )
        >>> Ridge().coefficient_variance(context).shape
        (3, 2)
        >>> stacked = Ridge() + Ridge()
        >>> stacked._label()
        'ridge(0.25) + ridge(0.25)'
        >>> float(stacked.coefficient_variance(context)[0, 0])
        0.125
    """

    __slots__ = ()

    @abstractmethod
    def coefficient_mean(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Prior mean of the coefficient matrix, shaped like the design.

        Args:
            context: What the prior needs to know about the sample.

        Returns:
            A ``(width, k)`` array in design-column order.
        """

    @abstractmethod
    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Prior variance of each coefficient, shaped like the design.

        Diagonal by construction. A prior that wants correlated coefficients
        expresses that through :meth:`dummy_observations`, where a row can
        restrict a combination.

        Args:
            context: What the prior needs to know about the sample.

        Returns:
            A ``(width, k)`` array of variances. Entries may be infinite, which
            is how a prior says it has no opinion about a coefficient.
        """

    def dummy_observations(
        self, context: _PriorContext
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Artificial rows this prior contributes, for restrictions on sums.

        Defaults to none, which is the statement that this prior's content is
        entirely in its moments.

        Args:
            context: What the prior needs to know about the sample.

        Returns:
            A ``(rows, k)`` target block and a ``(rows, width)`` design block.
        """
        return (
            np.zeros((0, context.k_endog), dtype=np.float64),
            np.zeros((0, context.width), dtype=np.float64),
        )

    @abstractmethod
    def _label(self) -> str:
        """Short description for a summary table.

        Returns:
            A one-line label naming the prior and its hyperparameters, such
            as ``"minnesota(l1=0.2, l2=0.5, l3=1, l4=100)"``.
        """

    def __add__(self, other: _Prior) -> _Prior:
        """Compose two priors.

        The components of both operands are concatenated into one flat
        :class:`_CompositePrior`, so ``(a + b) + c`` and ``a + (b + c)``
        hold the same three components in the same order.
        :class:`_NoPrior` contributes no component.

        The result is a :class:`_CompositePrior` in every case. Adding
        :class:`_NoPrior` to a single prior gives a composite of one, not
        the prior itself, and adding two :class:`_NoPrior` gives an empty
        composite whose label is the empty string.

        Args:
            other: The prior to stack on this one.

        Returns:
            The flattened composition, or ``NotImplemented`` when ``other``
            is not a prior, which Python turns into a ``TypeError``.
        """
        if not isinstance(other, _Prior):
            return NotImplemented
        return _CompositePrior(components=(*self._components(), *other._components()))

    def _components(self) -> tuple[_Prior, ...]:
        """This prior as a flat tuple, so composition does not nest.

        Returns:
            ``(self,)`` for an ordinary prior. :class:`_CompositePrior`
            returns its components and :class:`_NoPrior` returns ``()``,
            which is also how an estimator asks whether any prior was
            stated.
        """
        return (self,)


@dataclass(frozen=True, kw_only=True, slots=True)
class _CompositePrior(_Prior):
    """Several priors combined, which ``+`` returns.

    Moments multiply as precisions -- combining two opinions about the same
    coefficient is adding what each knows, which is what independent
    information does -- while dummy rows concatenate. Means combine as the
    precision-weighted average, so a prior with no opinion about a coefficient
    contributes nothing to it rather than dragging it toward zero.

    Attributes:
        components: The priors in the stack, already flattened.

    Note:
        Component variances are pooled without validation. An infinite or
        ``nan`` variance counts as no opinion (zero precision). A zero or
        negative variance is floored at ``1e-300``, which makes that
        component dogmatic about the coefficient. A coefficient on which no
        component has an opinion keeps an infinite variance and a zero mean.

        Flattening happens in ``+``, not here: a composite constructed
        directly from other composites keeps them nested.

    Example:
        Two opinions about the same coefficient, one four times as precise
        as the other:

        >>> import numpy as np
        >>> class Point(_Prior):
        ...     def __init__(self, mean, variance):
        ...         self.mean, self.variance = mean, variance
        ...
        ...     def coefficient_mean(self, context):
        ...         return np.full((context.width, context.k_endog), self.mean)
        ...
        ...     def coefficient_variance(self, context):
        ...         return np.full((context.width, context.k_endog), self.variance)
        ...
        ...     def _label(self):
        ...         return f"point({self.mean:g}, {self.variance:g})"
        >>> context = _PriorContext(
        ...     k_endog=1, order=1, scales=np.ones(1), presample_mean=np.zeros(1)
        ... )
        >>> both = Point(0.0, 0.25) + Point(2.0, 1.0)
        >>> round(float(both.coefficient_variance(context)[0, 0]), 4)
        0.2
        >>> round(float(both.coefficient_mean(context)[0, 0]), 4)
        0.4
        >>> len((both + Point(1.0, 4.0)).components)
        3
    """

    components: tuple[_Prior, ...]
    """The priors in the stack, in the order they were added.

    Flat when the composite came from ``+``. The order does not affect the
    moments; it fixes the order of the stacked dummy rows and of the parts
    of the label.
    """

    def _combined(
        self, context: _PriorContext
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Precision-weighted mean and combined variance.

        Evaluates both moment hooks of every component and pools them
        coefficient by coefficient: precisions add, and the mean is the
        precision-weighted average of the component means. Both public
        accessors call this, so asking for the mean and then the variance
        evaluates every component twice.

        Args:
            context: What the priors need to know about the sample.

        Returns:
            The combined mean and the combined variance, each ``(width, k)``.
        """
        precision = np.zeros((context.width, context.k_endog), dtype=np.float64)
        weighted = np.zeros((context.width, context.k_endog), dtype=np.float64)
        for component in self.components:
            variance = component.coefficient_variance(context)
            share = np.where(np.isfinite(variance), 1.0 / np.maximum(variance, 1e-300), 0.0)
            precision += share
            weighted += share * component.coefficient_mean(context)
        variance = np.where(precision > 0.0, 1.0 / np.maximum(precision, 1e-300), np.inf)
        mean = np.where(precision > 0.0, weighted / np.maximum(precision, 1e-300), 0.0)
        return mean, variance

    def coefficient_mean(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Precision-weighted average of the components' means.

        Args:
            context: What the priors need to know about the sample.

        Returns:
            A ``(width, k)`` array; zero where no component has an opinion.
        """
        return self._combined(context)[0]

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Reciprocal of the summed precisions.

        Args:
            context: What the priors need to know about the sample.

        Returns:
            A ``(width, k)`` array; infinite where no component has an
            opinion.
        """
        return self._combined(context)[1]

    def dummy_observations(
        self, context: _PriorContext
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Concatenate what each component contributes.

        Args:
            context: What the priors need to know about the sample.

        Returns:
            The stacked ``(rows, k)`` target block and ``(rows, width)``
            design block, in component order; zero rows when no component
            supplies any.
        """
        targets: list[npt.NDArray[np.float64]] = []
        blocks: list[npt.NDArray[np.float64]] = []
        for component in self.components:
            target, block = component.dummy_observations(context)
            targets.append(target)
            blocks.append(block)
        if not targets:
            return super(_CompositePrior, self).dummy_observations(context)
        return np.vstack(targets), np.vstack(blocks)

    def _label(self) -> str:
        """Short description for a summary table.

        Returns:
            The components' labels joined by ``" + "``; the empty string for
            an empty composite.
        """
        return " + ".join(component._label() for component in self.components)

    def _components(self) -> tuple[_Prior, ...]:
        """Flatten rather than nest.

        Returns:
            The stored components, so that adding to a composite extends it
            instead of wrapping it.
        """
        return self.components


class _NoPrior(_Prior):
    """The absence of shrinkage, which is unrestricted least squares.

    Present so that "no prior" is something a caller passes rather than
    something they express by passing nothing, and so that a stack assembled in
    a loop needs no special first case: it is the identity of composition.

    The identity holds for the moments, the dummy rows, and the component
    list, but not for the type: ``prior + _NoPrior()`` is a
    :class:`_CompositePrior` holding ``prior``, not ``prior`` itself. For an
    :class:`_AdaptivePrior` the difference matters, because the samplers
    refuse an adaptive prior inside a composition.

    Example:
        >>> import numpy as np
        >>> context = _PriorContext(
        ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
        ... )
        >>> prior = _NoPrior()
        >>> bool(np.all(np.isinf(prior.coefficient_variance(context))))
        True
        >>> prior._label(), prior._components()
        ('none', ())
    """

    __slots__ = ()

    def coefficient_mean(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Zero, which no infinite variance will ever pull toward.

        Args:
            context: What the prior needs to know about the sample.

        Returns:
            A ``(width, k)`` array of zeros.
        """
        return np.zeros((context.width, context.k_endog), dtype=np.float64)

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Infinite everywhere: no opinion about any coefficient.

        Args:
            context: What the prior needs to know about the sample.

        Returns:
            A ``(width, k)`` array of ``inf``.
        """
        return np.full((context.width, context.k_endog), np.inf, dtype=np.float64)

    def _label(self) -> str:
        """Short description for a summary table.

        Returns:
            The string ``"none"``.
        """
        return "none"

    def _components(self) -> tuple[_Prior, ...]:
        """Contribute nothing to a composition.

        Returns:
            The empty tuple, which is how an estimator recognizes that no
            prior was stated.
        """
        return ()


class _AdaptivePrior(_Prior):
    """A prior whose coefficient variances are Gibbs conditionals, not constants.

    The global-local and selection priors -- horseshoe, spike-and-slab,
    Dirichlet-Laplace, Normal-Gamma -- have no fixed variance to state: each
    coefficient's variance is a product of latent scales with their own full
    conditionals, redrawn every sweep. This class is the moments interface's
    promise kept: the static hooks report the variance *at the initial
    scales* (which is what admissibility checks need), and the sampler
    drives the three adaptive hooks -- initialize the scales, redraw them
    given the current coefficients, and read the variance they imply.

    The latent state travels as a plain mapping of named arrays owned by the
    sampler, so the prior object itself stays frozen and reusable across
    fits. Two conventions keep every subclass unit-honest and aligned with
    the family's grammar: shrinkage applies to the *standardized* coefficient
    -- the raw coefficient divided by the Minnesota scale ratio
    ``s_i / s_j``, so one latent scale means the same thing whatever the
    variables' units -- and only the endogenous lag block is shrunk, with
    deterministic and exogenous columns held at a fixed loose variance,
    exactly where the Minnesota prior leaves them.

    Adaptive priors do not compose: ``+`` builds a :class:`_CompositePrior`,
    which has no conditionals to expose, and the sampler refuses the result
    rather than silently freezing the scales.

    One sweep of a sampler under an adaptive prior is therefore: draw the
    coefficients given the variance from :meth:`_scale_variance`, divide
    them by :meth:`_unit_ratio`, pass the result to :meth:`_draw_scales`,
    and read the new variance. :meth:`_tracked` is accumulated over the kept
    sweeps and reported under :meth:`_tracked_label`.

    Note:
        The refusal applies to any :class:`_CompositePrior` that contains an
        adaptive component, a composite of one included. Adding
        :class:`_NoPrior` to an adaptive prior is enough to trigger it.
    """

    __slots__ = ()

    def coefficient_mean(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Zero everywhere: sparsity's claim is that coefficients are zero.

        There is no persistence hyperparameter here on purpose. Centering a
        selection prior away from zero would change what an exclusion means,
        so a caller who wants random-walk centering should difference the
        data or use the Minnesota family instead.

        Args:
            context: What the prior needs to know about the sample.

        Returns:
            A ``(width, k)`` array of zeros.
        """
        return np.zeros((context.width, context.k_endog), dtype=np.float64)

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """The variance at the initial scales -- finite, so shrunk-admissible.

        Evaluates :meth:`_initial_scales` with the unit ratio as the
        reference and reads the variance those scales imply. It describes
        where a sampler starts, not what it ends with: an estimator that
        reads only this hook treats the prior as a fixed Gaussian at its
        starting scales.

        Args:
            context: What the prior needs to know about the sample.

        Returns:
            A ``(width, k)`` array of finite, positive variances.
        """
        return self._scale_variance(
            self._initial_scales(context, self._unit_ratio(context)), context
        )

    def _unit_ratio(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Minnesota's unit fix as a matrix: ``s_i / s_j`` on the lag block.

        Entry ``[column, i]`` scales equation ``i``'s coefficient on that
        design column: ``s_i / s_j`` when the column is a lag of variable
        ``j``, and ``s_i`` on the deterministic and exogenous columns, where
        the coefficient itself carries the equation's units.

        Args:
            context: Supplies the scales and the column layout.

        Returns:
            A ``(width, k)`` array of ratios. The scales are not checked
            here: a zero scale gives infinite or ``nan`` entries.
        """
        scales = context.scales
        out = np.tile(scales[None, :], (context.width, 1)).astype(np.float64)
        offset = context.lag_offset
        for lag in range(context.order):
            for source in range(context.k_endog):
                column = offset + lag * context.k_endog + source
                out[column] = scales / scales[source]
        return out

    def _penalized_mask(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Ones on the endogenous lag rows, zeros elsewhere.

        Args:
            context: Supplies the column layout.

        Returns:
            A float array of shape ``(width,)``, to be broadcast across
            equations: ``1.0`` on the ``k_endog * order`` lag columns and
            ``0.0`` on the deterministic and exogenous columns.
        """
        mask = np.zeros(context.width, dtype=np.float64)
        offset = context.lag_offset
        mask[offset : offset + context.k_endog * context.order] = 1.0
        return mask

    @abstractmethod
    def _initial_scales(
        self, context: _PriorContext, reference: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
        """The latent scales the sampler starts from.

        Args:
            context: What the prior needs to know about the sample.
            reference: ``(width, k)`` data-driven reference scales for each
                coefficient -- least-squares standard errors when the
                sampler can compute them, the unit ratio otherwise. The
                spike-and-slab prior anchors its two variances here
                (George-Sun-Ni's semiautomatic default); the global-local
                priors ignore it.

        Returns:
            Named arrays, owned by the sampler and passed back verbatim.
        """

    @abstractmethod
    def _draw_scales(
        self,
        standardized: npt.NDArray[np.float64],
        context: _PriorContext,
        rng: np.random.Generator,
        scales: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """One sweep of the latent scales' full conditionals.

        Args:
            standardized: ``(width, k)`` current coefficients divided by the
                unit ratio, so the conditionals see unit-free magnitudes.
            context: What the prior needs to know about the sample.
            rng: Random generator.
            scales: The current latent state.

        Returns:
            The refreshed latent state.
        """

    @abstractmethod
    def _scale_variance(
        self, scales: dict[str, npt.NDArray[np.float64]], context: _PriorContext
    ) -> npt.NDArray[np.float64]:
        """The ``(width, k)`` coefficient variance the current scales imply.

        On the raw-coefficient scale: the unit ratio squared multiplies the
        latent variance on the lag block, and the deterministic and
        exogenous rows carry their fixed loose variance.

        Args:
            scales: The current latent state, as returned by
                :meth:`_initial_scales` or :meth:`_draw_scales`.
            context: What the prior needs to know about the sample.

        Returns:
            A ``(width, k)`` array of finite, positive variances.
        """

    @abstractmethod
    def _tracked(
        self, scales: dict[str, npt.NDArray[np.float64]], context: _PriorContext
    ) -> npt.NDArray[np.float64]:
        """The ``(width, k)`` per-coefficient diagnostic worth averaging.

        Inclusion probabilities for a selection prior, local scales for a
        global-local one; the sampler averages this over kept sweeps.

        Args:
            scales: The current latent state.
            context: What the prior needs to know about the sample.

        Returns:
            A ``(width, k)`` array in design-column order.
        """

    @abstractmethod
    def _tracked_label(self) -> str:
        """What the averaged diagnostic *is*, for the result's summary.

        Returns:
            A short noun phrase naming the quantity :meth:`_tracked`
            returns, such as ``"posterior inclusion probability"``.
        """


@dataclass(frozen=True, kw_only=True, slots=True)
class _VolatilityPrior:
    """The Kim-Shephard-Chib prior on a stationary log-AR(1) volatility law.

    ``h_{t+1} = mu + phi (h_t - mu) + sigma eta_t`` carries three
    hyperparameters, and every model in the package whose volatility
    follows this law -- the univariate stochastic-volatility model, the
    volatility-identified SVAR, the factor stochastic-volatility model --
    states its prior through this one record: Gaussian on ``mu``, Beta on
    ``(phi + 1) / 2``, inverse-gamma on ``sigma2``. The defaults are Kim,
    Shephard and Chib's (1998), which put the persistence near ``0.86`` a
    priori and keep the innovation variance small; a model that pins
    ``mu`` for identification says so and leaves that entry unread.

    Attributes:
        mu: ``(mean, variance)`` of the Gaussian prior on the log-variance
            mean.
        phi: ``(a, b)`` of the Beta prior on ``(phi + 1) / 2``, both
            positive.
        sigma2: ``(shape, rate)`` of the inverse-gamma prior on the
            log-variance innovation variance, both positive.

    Raises:
        SpecificationError: If a field is not a pair of finite numbers, or
            if a variance, Beta parameter, shape, or rate is not positive.

    Note:
        "Rate" is the rate of the gamma distribution on the precision
        ``1 / sigma2``, which is the scale of the inverse-gamma itself: the
        density is proportional to
        ``sigma2 ** -(shape + 1) * exp(-rate / sigma2)``.

    References:
        Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
        Likelihood inference and comparison with ARCH models. *The
        Review of Economic Studies*, 65(3), 361-393.

    Example:
        >>> prior = _VolatilityPrior()
        >>> prior.mu, prior.phi, prior.sigma2
        ((0.0, 10.0), (20.0, 1.5), (2.5, 0.025))
        >>> round(prior.phi_mean, 4), round(prior.sigma2_mean, 4)
        (0.8605, 0.0167)
        >>> _VolatilityPrior(phi=(5, 5)).phi_mean
        0.0
    """

    mu: tuple[float, float] = _KSC_MU_PRIOR
    """``(mean, variance)`` of the Gaussian prior on the log-variance mean ``mu``.

    The mean may take any sign; the variance must be positive. Defaults to
    ``(0.0, 10.0)``. A model that pins ``mu`` for identification does not
    read this field.
    """
    phi: tuple[float, float] = _KSC_PHI_PRIOR
    """``(a, b)`` of the Beta prior on ``(phi + 1) / 2``, both positive.

    Places the persistence on ``(-1, 1)``. The default ``(20.0, 1.5)`` has
    prior mean ``0.86``: persistent, with mass kept off the unit root.
    """
    sigma2: tuple[float, float] = _KSC_SIGMA2_PRIOR
    """``(shape, rate)`` of the inverse-gamma prior on the innovation variance, both positive.

    The default ``(2.5, 0.025)`` has prior mean ``0.0167``, a log variance
    that moves slowly. The prior mean is infinite for a shape of one or
    less.
    """

    def __post_init__(self) -> None:
        """Validate every pair and store it as floats.

        Each field is replaced by a tuple of two floats, so a list, a pair
        of integers, or any other two-element iterable of numbers is
        accepted.

        Raises:
            SpecificationError: If a field is not two finite numbers, if the
                second entry of any pair is not positive, or if the first
                entry of ``phi`` or ``sigma2`` is not positive.
        """
        object.__setattr__(
            self, "mu", _validate_hyperparameter_pair(self.mu, name="mu", positive_first=False)
        )
        object.__setattr__(
            self, "phi", _validate_hyperparameter_pair(self.phi, name="phi", positive_first=True)
        )
        object.__setattr__(
            self,
            "sigma2",
            _validate_hyperparameter_pair(self.sigma2, name="sigma2", positive_first=True),
        )

    @property
    def phi_mean(self) -> float:
        """The prior mean of the persistence, ``2a / (a + b) - 1``.

        Returns:
            The mean of ``phi`` implied by the Beta prior on
            ``(phi + 1) / 2``, a value in ``(-1, 1)``.
        """
        a, b = self.phi
        return 2.0 * a / (a + b) - 1.0

    @property
    def sigma2_mean(self) -> float:
        """Prior mean of ``sigma2``, ``rate / (shape - 1)``; ``inf`` for a shape of one or less.

        Returns:
            The inverse-gamma mean, or ``inf`` when the mean does not exist.
        """
        shape, rate = self.sigma2
        return rate / (shape - 1.0) if shape > 1.0 else float("inf")

    def _label(self) -> str:
        """Short description for a summary table.

        Returns:
            The three priors on one line. The defaults give
            ``"ksc(mu~N(0, 10), phi~Beta(20, 1.5), sigma2~IG(2.5, 0.025))"``.
        """
        return (
            f"ksc(mu~N({self.mu[0]:g}, {self.mu[1]:g}), phi~Beta({self.phi[0]:g}, "
            f"{self.phi[1]:g}), sigma2~IG({self.sigma2[0]:g}, {self.sigma2[1]:g}))"
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class _RandomWalkVolatilityPrior:
    """The prior on a random-walk log-variance law with an estimated step variance.

    ``h_{t+1} = h_t + gamma eta_t`` carries one hyperparameter, the
    variance of the step, and the unobserved-components stochastic-
    volatility model states its prior on it through this record: an
    inverse-gamma on ``gamma**2`` for each of its two volatility paths.
    The walk has no mean and no persistence to put a prior on; a model
    that fixes ``gamma`` outright does not read this record at all.

    Attributes:
        vol_of_vol: ``(shape, rate)`` of the inverse-gamma prior on the
            step variance ``gamma**2``, both positive.

    Raises:
        SpecificationError: If ``vol_of_vol`` is not a pair of finite,
            positive numbers.

    References:
        Stock, J. H., & Watson, M. W. (2007). Why has U.S. inflation become
        harder to forecast? *Journal of Money, Credit and Banking*,
        39(s1), 3-33.

    Example:
        >>> prior = _RandomWalkVolatilityPrior()
        >>> prior.vol_of_vol, prior.vol_of_vol_mean
        ((3.0, 0.04), 0.02)
        >>> _RandomWalkVolatilityPrior(vol_of_vol=(1.0, 0.04)).vol_of_vol_mean
        inf
    """

    vol_of_vol: tuple[float, float] = _UCSV_VOL_OF_VOL_PRIOR
    """``(shape, rate)`` of the inverse-gamma prior on the step variance ``gamma**2``.

    Both positive, with "rate" in the same sense as
    :attr:`_VolatilityPrior.sigma2`. The same pair is applied to both
    volatility paths. The default ``(3.0, 0.04)`` has prior mean ``0.02``,
    half the step variance of ``0.04`` that the model holds when ``gamma`` is
    fixed at its default of ``0.2``.
    """

    def __post_init__(self) -> None:
        """Validate the pair and store it as floats.

        Raises:
            SpecificationError: If ``vol_of_vol`` is not two finite,
                positive numbers.
        """
        object.__setattr__(
            self,
            "vol_of_vol",
            _validate_hyperparameter_pair(self.vol_of_vol, name="vol_of_vol", positive_first=True),
        )

    @property
    def vol_of_vol_mean(self) -> float:
        """Prior mean of ``gamma2``, ``rate / (shape - 1)``; ``inf`` for a shape of one or less.

        Returns:
            The inverse-gamma mean of the step variance, or ``inf`` when the
            mean does not exist.
        """
        shape, rate = self.vol_of_vol
        return rate / (shape - 1.0) if shape > 1.0 else float("inf")

    def _label(self) -> str:
        """Short description for a summary table.

        Returns:
            The prior on one line. The default gives
            ``"rw(gamma2~IG(3, 0.04))"``.
        """
        return f"rw(gamma2~IG({self.vol_of_vol[0]:g}, {self.vol_of_vol[1]:g}))"
