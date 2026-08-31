from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from scipy.stats import chi2, norm

from ..exceptions import DimensionError, NumericalError, SpecificationError
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

    _IS_POSTERIOR: ClassVar[bool] = False

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
                raise DimensionError(f"equation index {equation} is outside 0..{self.k_endog - 1}.")
            if not 0 <= regressor < self.width:
                raise DimensionError(f"regressor index {regressor} is outside 0..{self.width - 1}.")
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

    def wald_restriction(
        self, restriction: npt.NDArray[np.float64], *, null: str
    ) -> _WaldTestResult:
        """Test a general set of linear restrictions on ``vec(B)``.

        The companion to :meth:`wald`, which handles the common case where each
        restriction sets one coefficient to zero. This one takes an arbitrary
        matrix, which is what a restriction spanning several coefficients needs
        -- a long-run non-causality condition that weights the adjustment
        loadings by a cointegrating vector, an identifying restriction on a
        cointegrating space, a linear combination anyone wants a p-value for.

        Args:
            restriction: A ``(q, k * width)`` matrix over ``vec(B)`` in
                equation-major order, so equation ``i`` occupies columns
                ``i * width`` through ``(i + 1) * width - 1``.
            null: Plain-language statement of the hypothesis.

        Returns:
            A :class:`_WaldTestResult` with ``q`` degrees of freedom.

        Raises:
            DimensionError: If the matrix has the wrong width or no rows.
            NumericalError: If the restricted covariance is singular, which
                means the rows are linearly dependent and the restriction
                counts something twice.
        """
        size = self.k_endog * self.width
        if restriction.ndim != 2 or restriction.shape[1] != size:
            raise DimensionError(f"restriction must be (q, {size}); got shape {restriction.shape}.")
        if not restriction.shape[0]:
            raise DimensionError("a Wald test needs at least one restriction.")
        estimate = restriction @ self.coefficients.flatten(order="F")
        block = restriction @ self.to_matrix() @ restriction.T
        try:
            solved = np.linalg.solve(block, estimate)
        except np.linalg.LinAlgError as error:
            raise NumericalError(
                "the restricted covariance is singular; the restriction rows are linearly "
                "dependent, so at least one of them is implied by the others."
            ) from error
        statistic = float(estimate @ solved)
        df = int(restriction.shape[0])
        return _WaldTestResult(
            statistic=statistic, df=df, pvalue=float(chi2.sf(statistic, df)), null=null
        )

    @property
    def effective_parameters(self) -> float:
        """Every coefficient, since nothing was shrunk.

        Unrestricted least squares spends one parameter per design column per
        equation. The property exists so that a caller can ask the same
        question of a shrunk fit and an unshrunk one without first asking which
        it has.
        """
        return float(self.k_endog * self.width)

@dataclass(frozen=True, kw_only=True, slots=True)
class _PosteriorCovariance:
    """Per-equation posterior covariance of a coefficient matrix under a prior.

    The shrunk counterpart of :class:`_CoefficientCovariance`, and it holds a
    stack of blocks rather than two Kronecker factors because under a prior
    there are no such factors. Litterman's cross-equation weight makes the
    prior variance differ from equation to equation, so ``X'X`` enters each
    equation with a different penalty and the joint covariance stops being
    ``kron(sigma_u, inv(X'X))``.

    A consequence worth stating rather than discovering: with the innovation
    covariance treated as fixed and diagonal -- which is what running equation
    by equation assumes -- the equations are *a posteriori independent*, so the
    joint covariance is block diagonal and a Wald test spanning two equations
    sees no covariance between them. That is an approximation, it is
    Litterman's, and it is the price of keeping the cross-equation
    hyperparameter. A test confined to one equation is unaffected.

    Attributes:
        coefficients: The ``(width, k)`` posterior mean, in design-column order.
        blocks: ``(k, width, width)`` posterior covariance, one per equation.
        effective_parameters: Trace of the smoother matrix summed over
            equations -- the parameter count shrinkage actually spends, which
            is what an information criterion has to charge for rather than the
            nominal width.
    """

    coefficients: npt.NDArray[np.float64]
    blocks: npt.NDArray[np.float64]
    effective_parameters: float

    #: These are posterior moments, not sampling moments.
    _IS_POSTERIOR: ClassVar[bool] = True

    @property
    def k_endog(self) -> int:
        """Number of equations."""
        return int(self.blocks.shape[0])

    @property
    def width(self) -> int:
        """Regressors per equation."""
        return int(self.blocks.shape[1])

    @property
    def stderr(self) -> npt.NDArray[np.float64]:
        """Posterior standard deviations, shaped like :attr:`coefficients`."""
        return np.sqrt(
            np.clip(np.diagonal(self.blocks, axis1=1, axis2=2), 0.0, None)
        ).T

    @property
    def tstat(self) -> npt.NDArray[np.float64]:
        """Posterior mean over posterior standard deviation.

        Not a test statistic in the sampling sense. Under a prior this is a
        posterior signal-to-noise ratio, and reading it as a frequentist ``z``
        overstates the evidence, because the prior contributed some of the
        precision in the denominator. The summary labels the column
        accordingly.
        """
        return self.coefficients / self.stderr

    @property
    def pvalue(self) -> npt.NDArray[np.float64]:
        """Two-sided normal tail area for :attr:`tstat`."""
        return 2.0 * np.asarray(norm.sf(np.abs(self.tstat)), dtype=np.float64)

    def conf_int(self, *, alpha: float = 0.05) -> tuple[npt.NDArray[np.float64], ...]:
        """Symmetric posterior credible bounds.

        Args:
            alpha: Two-sided level; ``0.05`` gives a 95 percent interval.

        Returns:
            The lower and upper bound arrays.

        Raises:
            SpecificationError: If ``alpha`` is not strictly inside ``(0, 1)``.
        """
        if not 0.0 < alpha < 1.0:
            raise SpecificationError(f"alpha must lie strictly in (0, 1); got {alpha}.")
        half = float(norm.ppf(1.0 - alpha / 2.0)) * self.stderr
        return self.coefficients - half, self.coefficients + half

    def to_matrix(self) -> npt.NDArray[np.float64]:
        """The full ``(k * width)`` square covariance, which is block diagonal."""
        size = self.k_endog * self.width
        out = np.zeros((size, size), dtype=np.float64)
        for index in range(self.k_endog):
            lo = index * self.width
            out[lo : lo + self.width, lo : lo + self.width] = self.blocks[index]
        return out

    def wald_restriction(
        self, restriction: npt.NDArray[np.float64], *, null: str
    ) -> _WaldTestResult:
        """Test a general set of linear restrictions on ``vec(B)``.

        The block-diagonal joint covariance carries the same caveat as
        :meth:`wald`: a restriction spanning equations is evaluated as though
        they were independent, which under Litterman's fixed diagonal
        innovation covariance is what they are.

        Args:
            restriction: A ``(q, k * width)`` matrix over ``vec(B)`` in
                equation-major order.
            null: Plain-language statement of the hypothesis.

        Returns:
            A :class:`_WaldTestResult` with ``q`` degrees of freedom.

        Raises:
            DimensionError: If the matrix has the wrong width or no rows.
            NumericalError: If the restricted covariance is singular.
        """
        size = self.k_endog * self.width
        if restriction.ndim != 2 or restriction.shape[1] != size:
            raise DimensionError(
                f"restriction must be (q, {size}); got shape {restriction.shape}."
            )
        if not restriction.shape[0]:
            raise DimensionError("a Wald test needs at least one restriction.")
        estimate = restriction @ self.coefficients.flatten(order="F")
        block = restriction @ self.to_matrix() @ restriction.T
        try:
            solved = np.linalg.solve(block, estimate)
        except np.linalg.LinAlgError as error:
            raise NumericalError(
                "the restricted posterior covariance is singular; the restriction rows "
                "are linearly dependent."
            ) from error
        statistic = float(estimate @ solved)
        df = int(restriction.shape[0])
        return _WaldTestResult(
            statistic=statistic, df=df, pvalue=float(chi2.sf(statistic, df)), null=null
        )

    def wald(self, cells: Sequence[tuple[int, int]], *, null: str) -> _WaldTestResult:
        """Test that a set of coefficients is jointly zero.

        Args:
            cells: ``(equation, regressor)`` index pairs.
            null: Plain-language statement of the hypothesis.

        Returns:
            A :class:`_WaldTestResult` with one degree of freedom per cell.

        Raises:
            SpecificationError: If no cells are given.
            DimensionError: If any index is out of range.
            NumericalError: If the restricted covariance is singular.
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
            [
                [self.blocks[i][r, s] if i == j else 0.0 for j, s in cells]
                for i, r in cells
            ],
            dtype=np.float64,
        )
        try:
            solved = np.linalg.solve(block, estimate)
        except np.linalg.LinAlgError as error:
            raise NumericalError(
                "the restricted posterior covariance is singular; the restrictions are "
                "linearly dependent."
            ) from error
        statistic = float(estimate @ solved)
        df = len(cells)
        return _WaldTestResult(
            statistic=statistic, df=df, pvalue=float(chi2.sf(statistic, df)), null=null
        )
