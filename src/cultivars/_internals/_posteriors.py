from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..exceptions import NumericalError


@dataclass(frozen=True, kw_only=True, slots=True)
class _ConjugatePosterior:
    """Joint Normal-inverse-Wishart posterior over ``(B, Sigma)``.

    The conjugate counterpart of :class:`_PosteriorCovariance`, and it holds
    Kronecker factors again because conjugacy is exactly the condition under
    which they exist: with the prior variance factoring as ``Sigma x Omega``,
    the posterior of the coefficient matrix given ``Sigma`` is matrix normal
    with row precision ``K = Omega^-1 + X'X`` shared by every equation, and
    ``Sigma`` itself is inverse-Wishart. Cross-equation posterior dependence
    -- what the per-equation record cannot carry -- lives in that product.

    Attributes:
        coefficients: The ``(width, k)`` posterior mean, in design-column
            order.
        row_precision: The ``(width, width)`` shared row precision ``K``.
        scale: The ``(k, k)`` inverse-Wishart posterior scale.
        df: Inverse-Wishart posterior degrees of freedom.
        log_ml: Log marginal likelihood of the rows this posterior was
            updated on, given the prior -- exact, from the matrix-variate-t
            form.
    """

    coefficients: npt.NDArray[np.float64]
    row_precision: npt.NDArray[np.float64]
    scale: npt.NDArray[np.float64]
    df: float
    log_ml: float

    @property
    def sigma_mean(self) -> npt.NDArray[np.float64]:
        """Posterior mean of the innovation covariance.

        Raises:
            NumericalError: If the degrees of freedom do not support a mean.
        """
        k = int(self.scale.shape[0])
        if self.df <= k + 1:
            raise NumericalError(
                f"the inverse-Wishart mean needs df > k + 1; got df={self.df} with k={k}."
            )
        return self.scale / (self.df - k - 1.0)
