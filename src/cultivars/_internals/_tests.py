from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from .._core import _DEFAULT_ALPHA, SummaryTable, companion_matrix
from ..exceptions import DimensionError, NumericalError, SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _JohansenRankTest:
    """Sequential trace and maximum-eigenvalue tests for the cointegrating rank.

    Both sequences read down from ``r = 0``. The trace statistic tests
    ``rank <= r`` against ``rank = k``; the maximum-eigenvalue statistic tests
    ``rank = r`` against ``rank = r + 1``. The conventional reading stops at the
    first ``r`` that is not rejected, which :meth:`selected_rank` implements and
    which is a sequential procedure whose overall size is not the nominal level
    of any single step -- a caveat the table states rather than hides.

    Attributes:
        eigenvalues: The squared canonical correlations, descending. One per
            modelled equation, since a conditional system cannot support more
            cointegrating relations than it has equations.
        trace_statistic: One entry per null ``rank <= r``, ``r = 0 .. k - 1``.
        max_eigenvalue_statistic: One entry per null ``rank = r``.
        trace_pvalue: Empirical p-values from the simulated null.
        max_eigenvalue_pvalue: Empirical p-values from the simulated null.
        nobs: Effective sample the statistics were computed on.
        deterministic: The Johansen case the null was simulated under.
        k_exog: Weakly exogenous integrated regressors the null accounted for.
        simulations: Replications behind each p-value, so its resolution is
            visible rather than implied.
        small_sample: Whether the Reinsel-Ahn degrees-of-freedom scaling was
            applied to the statistics.
    """

    eigenvalues: npt.NDArray[np.float64]
    trace_statistic: npt.NDArray[np.float64]
    max_eigenvalue_statistic: npt.NDArray[np.float64]
    trace_pvalue: npt.NDArray[np.float64]
    max_eigenvalue_pvalue: npt.NDArray[np.float64]
    nobs: int
    deterministic: str
    k_exog: int
    simulations: int
    small_sample: bool

    @property
    def k_endog(self) -> int:
        """Number of modelled equations, which bounds the rank."""
        return int(self.eigenvalues.shape[0])

    @property
    def ranks(self) -> tuple[int, ...]:
        """The null ranks tested, in order."""
        return tuple(range(self.k_endog))

    def selected_rank(self, *, alpha: float = 0.05, statistic: str = "trace") -> int:
        """First rank the sequence fails to reject.

        Args:
            alpha: Level for each step of the sequence.
            statistic: ``"trace"`` or ``"max_eigenvalue"``.

        Returns:
            The chosen rank, or ``k_endog`` if every null is rejected, which is
            the finding that the system is stationary in levels and does not
            want a vector error-correction representation at all.

        Raises:
            SpecificationError: If the level or the statistic is unrecognized.
        """
        if not 0.0 < alpha < 1.0:
            raise SpecificationError(f"alpha must lie strictly in (0, 1); got {alpha}.")
        if statistic == "trace":
            pvalues = self.trace_pvalue
        elif statistic == "max_eigenvalue":
            pvalues = self.max_eigenvalue_pvalue
        else:
            raise SpecificationError(
                f"statistic must be 'trace' or 'max_eigenvalue'; got {statistic!r}."
            )
        for rank, pvalue in enumerate(pvalues):
            if pvalue >= alpha:
                return rank
        return self.k_endog

    def to_table(self, *, alpha: float = 0.05) -> SummaryTable:
        """Render both sequences with the rank each one selects."""
        chosen = self.selected_rank(alpha=alpha)
        notes = [
            f"Read down from r = 0 and stop at the first row not rejected: trace selects "
            f"r = {chosen}, max-eigenvalue selects "
            f"r = {self.selected_rank(alpha=alpha, statistic='max_eigenvalue')}.",
            "This is a sequential procedure, so the overall probability of an incorrect "
            f"rank is not the {alpha:.0%} attached to any single row.",
            f"P-values are empirical over {self.simulations:,} draws from the simulated "
            f"asymptotic null for the '{self.deterministic}' case, so the smallest "
            f"resolvable value is {1 / self.simulations:.0e}.",
        ]
        if self.k_exog:
            notes.append(
                f"The null accounts for {self.k_exog} weakly exogenous integrated "
                "regressor(s); its critical values are materially larger than the "
                "unconditional Johansen ones and the two are not interchangeable."
            )
        if self.small_sample:
            notes.append(
                "Statistics carry the Reinsel-Ahn degrees-of-freedom scaling, which pulls "
                "the well-known upward size distortion of the asymptotic test back toward "
                "its nominal level in short samples."
            )
        return SummaryTable(
            title="Johansen cointegration rank test",
            metadata=(
                ("Observations", f"{self.nobs}"),
                ("Variables", f"{self.k_endog}"),
                ("Deterministic", self.deterministic),
                ("Exogenous I(1)", f"{self.k_exog}"),
                ("Selected rank", f"{chosen}"),
            ),
            columns=("H0", "eigenvalue", "trace", "P>trace", "max-eig", "P>max-eig"),
            rows=tuple(
                (
                    f"r <= {rank}",
                    f"{self.eigenvalues[rank]:.4f}",
                    f"{self.trace_statistic[rank]:.3f}",
                    f"{self.trace_pvalue[rank]:.4f}",
                    f"{self.max_eigenvalue_statistic[rank]:.3f}",
                    f"{self.max_eigenvalue_pvalue[rank]:.4f}",
                )
                for rank in self.ranks
            ),
            notes=tuple(notes),
        )

    def __repr__(self) -> str:
        """One line with the rank each sequence selects."""
        return (
            f"JohansenRankTest(k={self.k_endog}, nobs={self.nobs}, "
            f"deterministic={self.deterministic!r}, "
            f"trace_rank={self.selected_rank()}, "
            f"max_eigenvalue_rank={self.selected_rank(statistic='max_eigenvalue')})"
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class _LikelihoodRatioTest:
    """Verdict of a likelihood-ratio test between two nested fits.

    Attributes:
        statistic: ``2 * (llf_unrestricted - llf_restricted)``.
        df: Degrees of freedom, the difference in free parameter counts.
        pvalue: Upper-tail probability under a chi-squared null.
    """

    statistic: float
    df: int
    pvalue: float

    def reject(self, *, alpha: float = 0.05) -> bool:
        """Whether the restriction is rejected at level ``alpha``."""
        return self.pvalue < alpha

    def __repr__(self) -> str:
        """One-line verdict."""
        return (
            f"LikelihoodRatioTest(statistic={self.statistic:.4f}, df={self.df}, "
            f"pvalue={self.pvalue:.4g})"
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class _WaldTest:
    """Verdict of a chi-squared restriction test on a fitted model.

    Carries the same three numbers as :class:`_LikelihoodRatioResult` and one
    more, and that one is the reason they are separate classes. A
    likelihood-ratio test is named by its construction: two nested fits, one
    statistic, and the restriction is whatever distinguishes them, so the
    object needs no label. A Wald statistic is a *form* rather than a
    hypothesis -- the same quadratic in the same estimated covariance answers
    Granger causality, residual autocorrelation, normality, and conditional
    heteroskedasticity -- so a result that did not carry its own null would be
    four unrelated verdicts wearing one type and no way to tell them apart in
    a table.

    The distribution is asymptotic in every case. Wald statistics are also
    famously sensitive to how a nonlinear restriction is algebraically
    arranged, but every use here restricts coefficients to zero, which is
    linear and therefore invariant.

    Attributes:
        statistic: The chi-squared statistic.
        df: Degrees of freedom, the number of restrictions imposed.
        pvalue: Upper-tail probability under the chi-squared null.
        null: The hypothesis being tested, phrased so it reads as a sentence
            in a diagnostics table -- ``"gdp does not Granger-cause infl"``,
            not ``"granger"``.
    """

    statistic: float
    df: int
    pvalue: float
    null: str

    def reject(self, *, alpha: float = _DEFAULT_ALPHA) -> bool:
        """Whether the null is rejected at level ``alpha``."""
        return self.pvalue < alpha

    def __repr__(self) -> str:
        """One-line verdict at the default level, which the text names."""
        verdict = "reject" if self.reject() else "keep"
        return (
            f"WaldTestResult(statistic={self.statistic:.4f}, df={self.df}, "
            f"pvalue={self.pvalue:.4g}, {verdict} at {_DEFAULT_ALPHA:.0%}: {self.null!r})"
        )


@dataclass(frozen=True)
class _StabilityTest:
    """The outcome of a stability (or invertibility) assessment.

    Attributes:
        eigenvalues: The companion eigenvalues (complex).
        max_modulus: The largest eigenvalue modulus; ``0.0`` when there are no
            eigenvalues (``p == 0``).
        is_stable: Whether the requested stability criterion is satisfied. With
            ``allow_unit_roots=False`` this means all moduli are strictly below
            ``1 - tol``; with ``allow_unit_roots=True`` it means no modulus
            exceeds ``1 + tol``.
        n_unit_roots: Number of eigenvalues whose modulus is within ``tol`` of 1.
        n_explosive: Number of eigenvalues with modulus above ``1 + tol``.
        tol: The modulus tolerance used for classification.
    """

    eigenvalues: npt.NDArray[np.complex128]
    max_modulus: float
    is_stable: bool
    n_unit_roots: int
    n_explosive: int
    tol: float

    @classmethod
    def _trivial(cls) -> Self:
        return cls(
            eigenvalues=np.empty(0, dtype=np.complex128),
            max_modulus=0.0,
            is_stable=True,
            n_unit_roots=0,
            n_explosive=0,
            tol=0.0,
        )

    @classmethod
    def _assess(
        cls, companion: npt.NDArray[np.float64], *, tol: float, allow_unit_roots: bool
    ) -> Self:
        if tol < 0.0:
            raise SpecificationError(f"tol must be non-negative; got {tol}.")
        if companion.size == 0:
            return cls(
                eigenvalues=np.empty(0, dtype=np.complex128),
                max_modulus=0.0,
                is_stable=True,
                n_unit_roots=0,
                n_explosive=0,
                tol=tol,
            )
        eigenvalues = np.linalg.eigvals(companion).astype(np.complex128)
        if not np.all(np.isfinite(eigenvalues)):
            raise NumericalError("Companion eigenvalue computation produced non-finite values.")
        moduli = np.abs(eigenvalues)
        max_modulus = float(moduli.max())
        n_unit_roots = int(np.count_nonzero(np.abs(moduli - 1.0) <= tol))
        n_explosive = int(np.count_nonzero(moduli > 1.0 + tol))
        is_stable = (n_explosive == 0) if allow_unit_roots else (max_modulus < 1.0 - tol)
        return cls(
            eigenvalues=eigenvalues,
            max_modulus=max_modulus,
            is_stable=is_stable,
            n_unit_roots=n_unit_roots,
            n_explosive=n_explosive,
            tol=tol,
        )

    @classmethod
    def assess_stability(
        cls, ar_coeffs: npt.ArrayLike, *, tol: float = 1e-8, allow_unit_roots: bool = False
    ) -> Self:
        """Assess stationarity of an AR/VAR from its autoregressive coefficients.

        Args:
            ar_coeffs: Coefficients ``A_1, ..., A_p``; shape ``(p,)`` or ``(p, k, k)``.
            tol: Modulus tolerance for classifying unit and explosive roots.
            allow_unit_roots: If ``True``, unit roots are permitted (only strictly
                explosive roots make the model unstable). Use for VECM and other
                models that carry unit roots by design.

        Returns:
            A :class:`StabilityResult`.

        Example:
            >>> res = assess_stability([0.5])
            >>> res.is_stable
            True
            >>> round(res.max_modulus, 4)
            0.5
        """
        ar = np.asarray(ar_coeffs, dtype=np.float64)
        if ar.size == 0:
            return cls._trivial()
        return cls._assess(companion_matrix(ar), tol=tol, allow_unit_roots=allow_unit_roots)

    @classmethod
    def assess_stability_from_companion(
        cls, companion: npt.ArrayLike, *, tol: float = 1e-8, allow_unit_roots: bool = False
    ) -> Self:
        """Assess stability directly from a companion (or state-transition) matrix.

        Args:
            companion: A square matrix (e.g. a companion or an LGSS transition matrix).
            tol: Modulus tolerance for classifying unit and explosive roots.
            allow_unit_roots: If ``True``, unit roots are permitted.

        Returns:
            A :class:`StabilityResult`.

        Raises:
            DimensionError: If ``companion`` is not a square 2-D array.
        """
        mat = np.asarray(companion, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise DimensionError(f"Companion matrix must be square 2-D; got shape {mat.shape}.")
        return cls._assess(mat, tol=tol, allow_unit_roots=allow_unit_roots)

    @classmethod
    def is_stationary(cls, ar_coeffs: npt.ArrayLike, *, tol: float = 1e-8) -> bool:
        """Convenience predicate: is the AR/VAR strictly stationary?

        Example:
            >>> is_stationary([1.5])
            False
        """
        return cls.assess_stability(ar_coeffs, tol=tol, allow_unit_roots=False).is_stable

    @classmethod
    def is_invertible(cls, ma_coeffs: npt.ArrayLike, *, tol: float = 1e-8) -> bool:
        """Is an MA/ARMA invertible? (companion of the MA polynomial, roots inside).

        Args:
            ma_coeffs: MA coefficients ``M_1, ..., M_q`` in the same layout as AR
                coefficients; shape ``(q,)`` or ``(q, k, k)``.
            tol: Modulus tolerance.

        Returns:
            ``True`` iff all companion eigenvalues lie strictly inside the unit circle.
        """
        ma = np.asarray(ma_coeffs, dtype=np.float64)
        if ma.size == 0:
            return True
        return cls._assess(companion_matrix(ma), tol=tol, allow_unit_roots=False).is_stable
