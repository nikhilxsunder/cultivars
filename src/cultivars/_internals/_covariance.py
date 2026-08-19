
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2, norm

from ..exceptions import DimensionError, SpecificationError
from ._results import _WaldTestResult


@dataclass(frozen=True, kw_only=True, slots=True)
class _CoefficientCovariance:
    """Asymptotic covariance of a multivariate least-squares coefficient matrix.

    For ``Y = Z B + U`` with ``B`` of shape ``(width, k)``, the estimator's
    asymptotic covariance is ``kron(sigma_u, inv(Z'Z))`` over ``vec(B)`` stacked
    equation by equation. This holds the two Kronecker factors rather than their
    product. The product is ``(k * width)`` square, which for a fixed-effects
    panel with two hundred units is a third of a million floats to answer a
    question about four slopes, and every quantity anyone actually asks for --
    a standard error, a Wald block over one equation's lags -- reads a handful
    of entries that both factors already carry.

    Attributes:
        coefficients: The ``(width, k)`` estimate, in design-column order.
        sigma_u: The ``(k, k)`` residual covariance, degrees-of-freedom
            corrected.
        xtx_inv: The ``(width, width)`` inverse cross-product of the design.
    """

    coefficients: npt.NDArray[np.float64]
    sigma_u: npt.NDArray[np.float64]
    xtx_inv: npt.NDArray[np.float64]

    @property
    def k_endog(self) -> int:
        """Number of equations."""
        return int(self.sigma_u.shape[0])

    @property
    def width(self) -> int:
        """Regressors per equation."""
        return int(self.xtx_inv.shape[0])

    @property
    def stderr(self) -> npt.NDArray[np.float64]:
        """Standard errors, shaped and ordered like :attr:`coefficients`.

        The diagonal of the Kronecker product factorizes, so this is an outer
        product of two diagonals rather than a square root of a large matrix.
        """
        return np.sqrt(np.outer(np.diag(self.xtx_inv), np.diag(self.sigma_u)))

    @property
    def tstat(self) -> npt.NDArray[np.float64]:
        """Coefficients divided by their standard errors.

        Referred to the normal rather than a ``t``: the asymptotics for a
        vector autoregression are normal, the residual covariance is estimated
        jointly across equations rather than one at a time, and there is no
        exact small-sample distribution here for a ``t`` to be an
        approximation of.
        """
        return self.coefficients / self.stderr

    @property
    def pvalue(self) -> npt.NDArray[np.float64]:
        """Two-sided normal p-values for the hypothesis that a coefficient is zero."""
        return 2.0 * np.asarray(norm.sf(np.abs(self.tstat)), dtype=np.float64)

    def conf_int(self, *, alpha: float = 0.05) -> tuple[npt.NDArray[np.float64], ...]:
        """Symmetric normal confidence bounds.

        Args:
            alpha: Two-sided level; ``0.05`` gives a 95 percent interval.

        Returns:
            The lower and upper bound arrays, each shaped like
            :attr:`coefficients`.

        Raises:
            SpecificationError: If ``alpha`` is not strictly inside ``(0, 1)``.
        """
        if not 0.0 < alpha < 1.0:
            raise SpecificationError(f"alpha must lie strictly in (0, 1); got {alpha}.")
        half = float(norm.ppf(1.0 - alpha / 2.0)) * self.stderr
        return self.coefficients - half, self.coefficients + half

    def to_matrix(self) -> npt.NDArray[np.float64]:
        """Materialize the full ``(k * width)`` square covariance of ``vec(B)``.

        Provided for the caller who genuinely needs the whole thing -- a
        restriction spanning several equations, a delta-method transform. Every
        method above avoids it deliberately.
        """
        return np.asarray(np.kron(self.sigma_u, self.xtx_inv), dtype=np.float64)

    def wald(self, cells: Sequence[tuple[int, int]], *, null: str) -> _WaldTestResult:
        """Test that a set of coefficients is jointly zero.

        Args:
            cells: ``(equation, regressor)`` index pairs naming the restricted
                coefficients. Both indices address :attr:`coefficients`, so a
                caller reasons in the same coordinates it reads results in and
                never has to reproduce the column-major flattening.
            null: Plain-language statement of the hypothesis, carried onto the
                result so the test says what it tested.

        Returns:
            A :class:`_WaldTestResult` with degrees of freedom equal to the
            number of restrictions.

        Raises:
            SpecificationError: If no cells are given.
            DimensionError: If any index is out of range.
        """
        if not cells:
            raise SpecificationError("a Wald test needs at least one restriction.")
        for equation, regressor in cells:
            if not 0 <= equation < self.k_endog:
                raise DimensionError(
                    f"equation index {equation} is outside 0..{self.k_endog - 1}."
                )
            if not 0 <= regressor < self.width:
                raise DimensionError(
                    f"regressor index {regressor} is outside 0..{self.width - 1}."
                )
        estimate = np.array([self.coefficients[r, i] for i, r in cells], dtype=np.float64)
        block = np.array(
            [[self.sigma_u[i, j] * self.xtx_inv[r, s] for j, s in cells] for i, r in cells],
            dtype=np.float64,
        )
        statistic = float(estimate @ np.linalg.solve(block, estimate))
        df = len(cells)
        return _WaldTestResult(
            statistic=statistic, df=df, pvalue=float(chi2.sf(statistic, df)), null=null
        )
