
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt



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
        scales: Per-variable residual scale, from :func:`minnesota_scales`.
        presample_mean: Mean of the first ``order`` observations, which the
            sum-of-coefficients restriction is centred on.
        k_exog: Exogenous regressors carried after the lag block.
        include_constant: Whether the design carries a leading intercept
            column. Dummy columns must match the design's own ordering, and a
            prior that guesses wrong is inert at the limits while still looking
            plausible in between -- which is worse than being wrong loudly.
    """

    k_endog: int
    order: int
    scales: npt.NDArray[np.float64]
    presample_mean: npt.NDArray[np.float64]
    k_exog: int = 0
    include_constant: bool = True

    @property
    def width(self) -> int:
        """Design columns a prior's blocks must match."""
        return int(self.include_constant) + self.k_endog * self.order + self.k_exog

    @property
    def lag_offset(self) -> int:
        """Design columns ahead of the endogenous lag block."""
        return int(self.include_constant)


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
        """Short description for a summary table."""

    def __add__(self, other: _Prior) -> _Prior:
        """Compose two priors."""
        if not isinstance(other, _Prior):
            return NotImplemented
        return _CompositePrior(components=(*self._components(), *other._components()))

    def _components(self) -> tuple[_Prior, ...]:
        """This prior as a flat tuple, so composition does not nest."""
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
    """

    components: tuple[_Prior, ...]

    def _combined(
        self, context: _PriorContext
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Precision-weighted mean and combined variance."""
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
        """Precision-weighted average of the components' means."""
        return self._combined(context)[0]

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Reciprocal of the summed precisions."""
        return self._combined(context)[1]

    def dummy_observations(
        self, context: _PriorContext
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Concatenate what each component contributes."""
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
        """Short description for a summary table."""
        return " + ".join(component._label() for component in self.components)

    def _components(self) -> tuple[_Prior, ...]:
        """Flatten rather than nest."""
        return self.components
