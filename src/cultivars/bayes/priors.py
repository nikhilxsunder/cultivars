
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._internals import _NoPrior as NoPrior
from .._internals import _Prior, _PriorContext
from ..exceptions import DimensionError, SpecificationError

__all__ = ["MinnesotaPrior", "NoPrior"]


@dataclass(frozen=True, kw_only=True, slots=True)
class MinnesotaPrior(_Prior):
    """Litterman's prior: shrink toward independent random walks.

    Five hyperparameters, and each one answers a different question.

    ``tightness`` is how much the prior is believed at all; it scales every
    variance and is the one people tune. ``cross_equation`` says how much less
    a variable's dependence on *other* variables is believed than its
    dependence on itself, which is the prior's central economic claim -- most
    of what a series does is explained by its own past -- and is the
    hyperparameter that a dummy-observation implementation cannot express.
    ``decay`` tightens longer lags toward zero, encoding that distant history
    matters less. ``exogenous`` loosens the intercept and any exogenous block,
    and defaults large: those coefficients carry levels and slopes that nobody
    means to shrink, and a tight default there is a silent and expensive
    mistake. ``sum_of_coefficients`` is the only one that cannot be a variance
    -- it restricts a *sum* of coefficients, so it enters as artificial rows.

    Prior variances, for equation ``i`` and variable ``j`` at lag ``l``:

    ==================  ==================================================
    own lag             ``(tightness / l ** decay) ** 2``
    cross lag           ``(tightness * cross_equation * s_i /
                        (l ** decay * s_j)) ** 2``
    intercept, exog     ``(tightness * exogenous * s_i) ** 2``
    ==================  ==================================================

    The scale ratio is what makes variables measured in different units
    comparable, so that shrinkage is a statement about dynamics rather than
    about whether a series is quoted in percent or in levels.

    With ``cross_equation`` other than one the prior variance no longer factors
    as a Kronecker product, so there is no Normal-inverse-Wishart closed form
    and estimation runs equation by equation conditional on the scales. That is
    Litterman's original procedure and it is correct; it is also why this prior
    and a conjugate one are different estimation paths rather than the same
    object with different numbers.

    Attributes:
        tightness: Overall confidence, ``lambda_1``. Values near 0.1 to 0.3 are
            usual for macroeconomic data; large recovers least squares.
        cross_equation: How much harder cross-variable coefficients shrink,
            ``lambda_2``. One treats them like own lags; smaller is tighter.
        decay: Lag decay, ``lambda_3``.
        exogenous: Looseness of the intercept and exogenous block,
            ``lambda_4``. Large is flat.
        sum_of_coefficients: Confidence that the variables sit at their
            pre-sample means forever, ``lambda_5``. ``None`` omits the
            restriction; larger imposes it harder.
        persistence: Prior mean of each variable's own first lag. One is the
            random-walk prior for levels; zero suits differenced data. Getting
            this wrong shrinks toward the wrong place, so tightening makes the
            estimate worse rather than better.
    """

    tightness: float = 0.2
    cross_equation: float = 0.5
    decay: float = 1.0
    exogenous: float = 100.0
    sum_of_coefficients: float | None = None
    persistence: float | Sequence[float] = 1.0

    def _persistence(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Broadcast the prior mean to one value per variable."""
        if isinstance(self.persistence, int | float):
            return np.full(context.k_endog, float(self.persistence), dtype=np.float64)
        values = np.asarray(self.persistence, dtype=np.float64).ravel()
        if values.shape != (context.k_endog,):
            raise DimensionError(
                f"persistence must be a scalar or have {context.k_endog} entries; "
                f"got shape {values.shape}."
            )
        return values

    def _check(self) -> None:
        """Reject hyperparameters that do not describe a prior.

        Raises:
            SpecificationError: If a tightness is not positive or the decay is
                negative.
        """
        for name, value in (
            ("tightness", self.tightness),
            ("cross_equation", self.cross_equation),
            ("exogenous", self.exogenous),
        ):
            if value <= 0.0:
                raise SpecificationError(f"{name} must be positive; got {value}.")
        if self.decay < 0.0:
            raise SpecificationError(f"decay must be non-negative; got {self.decay}.")
        if self.sum_of_coefficients is not None and self.sum_of_coefficients <= 0.0:
            raise SpecificationError(
                f"sum_of_coefficients must be positive when given; got {self.sum_of_coefficients}."
            )

    def coefficient_mean(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Each variable's own first lag at ``persistence``, everything else zero."""
        self._check()
        means = self._persistence(context)
        out = np.zeros((context.width, context.k_endog), dtype=np.float64)
        offset = context.lag_offset
        for index in range(context.k_endog):
            out[offset + index, index] = means[index]
        return out

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Litterman's variances, tighter for cross terms and for longer lags."""
        self._check()
        size, order = context.k_endog, context.order
        scales = context.scales
        out = np.empty((context.width, size), dtype=np.float64)
        offset = context.lag_offset
        loose = (self.tightness * self.exogenous * scales) ** 2
        if context.include_constant:
            out[0] = loose
        for lag in range(1, order + 1):
            damped = float(lag) ** self.decay
            for source in range(size):
                column = offset + (lag - 1) * size + source
                own = (self.tightness / damped) ** 2
                cross = (
                    self.tightness * self.cross_equation * scales / (damped * scales[source])
                ) ** 2
                out[column] = np.where(np.arange(size) == source, own, cross)
        for extra in range(context.k_exog):
            out[offset + size * order + extra] = loose
        return out

    def dummy_observations(
        self, context: _PriorContext
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The sum-of-coefficients rows, when that restriction is asked for.

        One row per variable, saying that a series sitting at its pre-sample
        average forever should stay there. That is a statement about the sum of
        a variable's lag coefficients, which no diagonal variance can make,
        which is why this hyperparameter alone enters as data.
        """
        self._check()
        if self.sum_of_coefficients is None:
            return super(MinnesotaPrior, self).dummy_observations(context)
        size = context.k_endog
        centre = (
            np.diag(self._persistence(context) * context.presample_mean) * self.sum_of_coefficients
        )
        block = np.hstack(
            [
                np.zeros((size, context.lag_offset), dtype=np.float64),
                np.kron(np.ones((1, context.order)), centre),
                np.zeros((size, context.k_exog), dtype=np.float64),
            ]
        )
        return centre, block

    def _label(self) -> str:
        """Short description for a summary table."""
        tail = "" if self.sum_of_coefficients is None else f", l5={self.sum_of_coefficients:g}"
        return (
            f"minnesota(l1={self.tightness:g}, l2={self.cross_equation:g}, "
            f"l3={self.decay:g}, l4={self.exogenous:g}{tail})"
        )
