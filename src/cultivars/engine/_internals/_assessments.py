# filepath: /src/cultivars/engine/_internals/_assessments.py
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
"""Assessment records: the companion-eigenvalue verdict on stability and invertibility.

A linear recursion ``y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + u_t`` is stable
exactly when every eigenvalue of its companion matrix lies strictly inside
the unit circle, which is the same statement as
``det(I - A_1 z - ... - A_p z^p)`` having no root with ``|z| <= 1``. This
module holds the one record that carries that reading,
:class:`_StabilityAssessment`, together with the class methods that produce
it. Every stationarity, invertibility, and transition-stability check in the
package routes through here, so the classification rule and the tolerance
that decides it are defined once rather than per model family.

The rule is a three-way partition of the eigenvalue moduli by a band of
half-width ``tol`` around the circle:

* inside: ``|lambda| < 1 - tol``;
* unit: ``abs(|lambda| - 1) <= tol``;
* explosive: ``|lambda| > 1 + tol``.

The strict verdict (``allow_unit_roots=False``) requires every modulus to be
inside; the permissive verdict (``allow_unit_roots=True``) requires only
that none be explosive. The strict verdict is therefore conservative: a
root that is inside the circle but within ``tol`` of it is counted as a unit
root and fails. The permissive verdict is the one a VECM asks for, since it
carries ``k - r`` unit roots by construction, and the one a state-space
transition with random-walk states asks for.

Two conventions are fixed here and assumed by every caller.

Sign. All four entry points read their argument in the autoregressive
convention, testing ``1 - c_1 z - ... - c_q z^q``. A moving-average
polynomial stored as ``theta(L) = 1 + theta_1 L + ... + theta_q L^q`` must be
negated before it is passed, :meth:`_StabilityAssessment.is_invertible`
included. The fitted results do this in ``_InvertibilityMixin``; a direct
caller has to do it by hand.

Accuracy. The eigenvalues are those of a general real matrix
(``numpy.linalg.eigvals``), and a defective eigenvalue of multiplicity ``m``
is recovered only to about ``eps ** (1 / m)``. Simple and double unit roots
land inside the default band of ``1e-8``. A triple unit root does not: the
companion of ``(1 - L)^3`` returns moduli about ``7e-6`` from one, so at the
default tolerance two of its three roots are classified as explosive. Models
that carry higher-order integration by design should pass a wider ``tol``.

Layout. The record is a frozen dataclass and is re-exported under its public
name as ``cultivars.diagnostics.stability.StabilityAssessment``. The
companion is built by :func:`~cultivars.engine._core._matrices._companion_matrix`;
``_assess`` is the single classification kernel, and ``_trivial`` is the
record of a recursion with no lags.

References:
    Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
        Analysis*. Springer. Section 2.1.1.
    Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University
        Press. Sections 1.2 and 10.1.
    Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th
        ed.). Johns Hopkins University Press. Section 7.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

from ...exceptions import DimensionError, NumericalError, SpecificationError
from .._core._matrices import _companion_matrix

__all__ = [
    "_StabilityAssessment",
]


@dataclass(frozen=True)
class _StabilityAssessment:
    """The outcome of a stability (or invertibility) assessment.

    A deterministic reading of a coefficient array or a transition matrix:
    the companion eigenvalues, their largest modulus, a count of the roots
    on and beyond the unit circle, and the verdict under the criterion the
    caller asked for. There is no null hypothesis and no sampling variation
    in it. Instances are built by the class methods, not by the constructor,
    which performs no validation and does not check that the fields agree
    with one another.

    The record does not store which criterion produced ``is_stable``. The
    counts are criterion-free, so the other verdict can always be recovered:
    the permissive one is ``n_explosive == 0`` and the strict one is
    ``max_modulus < 1 - tol``.

    The dataclass is frozen, so fields cannot be rebound, but ``eigenvalues``
    is an ordinary writeable array and is not copied on access. Because that
    field is an array, the generated ``__eq__`` raises ``ValueError`` when
    two records with more than one eigenvalue are compared and ``__hash__``
    raises ``TypeError``; compare fields, not records.

    Attributes:
        eigenvalues: The companion eigenvalues (complex), shape ``(k * p,)``
            in the order NumPy returns them, which is not sorted by modulus.
            Empty when there are no lags.
        max_modulus: The largest eigenvalue modulus; ``0.0`` when there are no
            eigenvalues (``p == 0``).
        is_stable: Whether the requested stability criterion is satisfied. With
            ``allow_unit_roots=False`` this means all moduli are strictly below
            ``1 - tol``; with ``allow_unit_roots=True`` it means no modulus
            exceeds ``1 + tol``.
        n_unit_roots: Number of eigenvalues whose modulus is within ``tol`` of 1.
        n_explosive: Number of eigenvalues with modulus above ``1 + tol``.
        tol: The modulus tolerance used for classification. Recorded as
            ``0.0`` on the trivial record that :meth:`assess_stability`
            returns for empty coefficients, whatever tolerance was passed.

    Example:
        >>> res = _StabilityAssessment.assess_stability([0.5, -0.3])
        >>> res.is_stable, res.n_unit_roots, res.n_explosive
        (True, 0, 0)
        >>> round(res.max_modulus, 4)
        0.5477
        >>> walk = _StabilityAssessment.assess_stability([1.0])
        >>> walk.is_stable, walk.n_unit_roots, walk.n_explosive
        (False, 1, 0)
    """

    eigenvalues: npt.NDArray[np.complex128]
    """Companion eigenvalues, shape ``(k * p,)``; empty when there are no lags.

    In the order NumPy returns them, which is not sorted by modulus; complex
    eigenvalues arrive as conjugate pairs. The array is writeable and is not
    copied on access, so the frozen dataclass does not protect its contents.
    """
    max_modulus: float
    """Spectral radius of the companion: the largest eigenvalue modulus.

    ``0.0`` by convention when there are no eigenvalues. This is the number
    the strict criterion compares with ``1 - tol``, and for a stable model
    it is the geometric rate at which impulse responses eventually decay.
    """
    is_stable: bool
    """Verdict under the criterion the producer was asked for.

    Strict (``allow_unit_roots=False``): ``max_modulus < 1 - tol``.
    Permissive (``allow_unit_roots=True``): ``n_explosive == 0``. The record
    does not store which one was used; both can be recomputed from the other
    fields.
    """
    n_unit_roots: int
    """Number of eigenvalues with ``abs(|lambda| - 1) <= tol``.

    Counts eigenvalues on the unit circle wherever they sit on it, so
    seasonal roots and complex pairs are included and each member of a pair
    counts once: ``1 - L^4`` reports four. A stable root within ``tol`` of
    the circle is counted here as well.
    """
    n_explosive: int
    """Number of eigenvalues with modulus above ``1 + tol``.

    Zero is the permissive criterion. A unit root of multiplicity three or
    more can be counted here at the default tolerance, because its computed
    modulus misses one by more than ``1e-8``.
    """
    tol: float
    """Half-width of the unit-root band around modulus one used to classify.

    Carried so that two assessments can be compared on the same footing.
    Recorded as ``0.0`` on the trivial record that :meth:`assess_stability`
    returns for empty coefficients, whatever tolerance was passed.
    """

    @classmethod
    def _trivial(cls) -> Self:
        """The record of a recursion with no lags (``p == 0``).

        With no companion there is nothing to classify: the eigenvalue array
        is empty, the largest modulus is ``0.0`` by convention, both counts
        are zero, and the verdict is ``True`` under either criterion.

        Returns:
            The stable, eigenvalue-free :class:`_StabilityAssessment`, with
            ``tol`` recorded as ``0.0`` because no tolerance was applied.

        Example:
            >>> res = _StabilityAssessment._trivial()
            >>> res.is_stable, res.max_modulus, res.eigenvalues.size, res.tol
            (True, 0.0, 0, 0.0)
        """
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
        """Classify the eigenvalues of a square matrix against the unit circle.

        The single kernel behind every public entry point. It takes the
        matrix as given, computes its eigenvalues, and partitions their
        moduli by the band of half-width ``tol`` around one: ``n_unit_roots``
        counts ``abs(|lambda| - 1) <= tol`` and ``n_explosive`` counts
        ``|lambda| > 1 + tol``. The verdict is ``n_explosive == 0`` when unit
        roots are allowed and ``max_modulus < 1 - tol`` otherwise, so under
        the strict criterion a root inside the circle but within ``tol`` of
        it fails.

        The caller is responsible for the shape: ``companion`` is assumed to
        be a square 2-D float array and is not checked here. An empty matrix
        yields the eigenvalue-free stable record, carrying the ``tol`` that
        was passed (unlike :meth:`_trivial`, which records ``0.0``).

        Args:
            companion: A square ``(n, n)`` real matrix, either a companion
                matrix or a state-transition matrix. Not modified.
            tol: Half-width of the unit-root band around modulus one. Must
                be non-negative; ``0.0`` counts only moduli exactly equal
                to one as unit roots.
            allow_unit_roots: If ``True``, only explosive roots make the
                verdict ``False``; if ``False``, every modulus must be
                strictly below ``1 - tol``.

        Returns:
            The :class:`_StabilityAssessment` for ``companion``.

        Raises:
            SpecificationError: If ``tol`` is negative. Checked first, so it
                is raised for an empty matrix as well.
            NumericalError: If the computed eigenvalues are not all finite.
            numpy.linalg.LinAlgError: If ``companion`` itself contains
                non-finite entries (NumPy rejects the input before the
                finiteness check above is reached), or if the eigenvalue
                iteration fails to converge.

        Example:
            >>> import numpy as np
            >>> res = _StabilityAssessment._assess(
            ...     np.array([[0.5, 1.0], [0.0, 1.0]]), tol=1e-8, allow_unit_roots=True
            ... )
            >>> res.is_stable, res.n_unit_roots, res.n_explosive
            (True, 1, 0)
        """
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

        Builds the ``(k * p, k * p)`` companion of
        ``y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + u_t`` and classifies its
        eigenvalues. Equivalently, tests whether
        ``det(I - A_1 z - ... - A_p z^p)`` has all its roots outside the
        unit circle. Deterministic terms and exogenous regressors play no
        part and must not be included in ``ar_coeffs``.

        Empty coefficients (``p == 0``) short-circuit to the trivial stable
        record before any validation: ``tol`` is not checked on that path and
        is recorded as ``0.0``.

        Args:
            ar_coeffs: Coefficients ``A_1, ..., A_p``; shape ``(p,)`` or ``(p, k, k)``.
            tol: Modulus tolerance for classifying unit and explosive roots.
                A root is a unit root when its modulus is within ``tol`` of
                one. The default suits simple and double unit roots; widen
                it for higher-order integration (see the module notes).
            allow_unit_roots: If ``True``, unit roots are permitted (only strictly
                explosive roots make the model unstable). Use for VECM and other
                models that carry unit roots by design.

        Returns:
            The :class:`_StabilityAssessment` of the companion matrix.

        Raises:
            SpecificationError: If ``tol`` is negative and ``ar_coeffs`` is
                non-empty.
            DimensionError: If ``ar_coeffs`` is neither rank 1 nor a stack of
                square matrices.
            NumericalError: If ``ar_coeffs`` contains non-finite values, or
                the eigenvalue computation returns them.

        Example:
            >>> res = _StabilityAssessment.assess_stability([0.5])
            >>> res.is_stable
            True
            >>> round(res.max_modulus, 4)
            0.5

            A bivariate VAR(1) is passed as a ``(1, 2, 2)`` stack:

            >>> import numpy as np
            >>> A = np.array([[[0.5, 0.1], [0.0, 0.4]]])
            >>> res = _StabilityAssessment.assess_stability(A)
            >>> res.is_stable, round(res.max_modulus, 2), res.eigenvalues.shape
            (True, 0.5, (2,))

            A unit root is counted, not called explosive, and passes only
            when it is allowed:

            >>> walk = _StabilityAssessment.assess_stability([1.0])
            >>> walk.is_stable, walk.n_unit_roots, walk.n_explosive
            (False, 1, 0)
            >>> _StabilityAssessment.assess_stability([1.0], allow_unit_roots=True).is_stable
            True
        """
        ar = np.asarray(ar_coeffs, dtype=np.float64)
        if ar.size == 0:
            return cls._trivial()
        return cls._assess(_companion_matrix(ar), tol=tol, allow_unit_roots=allow_unit_roots)

    @classmethod
    def assess_stability_from_companion(
        cls, companion: npt.ArrayLike, *, tol: float = 1e-8, allow_unit_roots: bool = False
    ) -> Self:
        """Assess stability directly from a companion (or state-transition) matrix.

        For callers that already hold the matrix whose powers drive the
        recursion: a companion built elsewhere, or the transition matrix of
        a linear Gaussian state-space model. No companion structure is
        assumed or verified; any square real matrix is accepted and read as
        ``x_t = T x_{t-1} + ...``.

        Unlike :meth:`assess_stability`, an empty ``(0, 0)`` matrix is not
        short-circuited: ``tol`` is validated and recorded on the returned
        record.

        Args:
            companion: A square matrix (e.g. a companion or an LGSS transition matrix).
            tol: Modulus tolerance for classifying unit and explosive roots.
            allow_unit_roots: If ``True``, unit roots are permitted. Use for
                transitions with random-walk or other integrated states.

        Returns:
            The :class:`_StabilityAssessment` of ``companion``.

        Raises:
            DimensionError: If ``companion`` is not a square 2-D array.
            SpecificationError: If ``tol`` is negative.
            NumericalError: If the eigenvalue computation returns non-finite
                values.
            numpy.linalg.LinAlgError: If ``companion`` contains non-finite
                entries. Unlike :meth:`assess_stability`, the input is not
                screened here, so NumPy's own exception propagates.

        Example:
            A local-linear-trend transition has a double unit root and
            nothing explosive:

            >>> import numpy as np
            >>> T = np.array([[1.0, 1.0], [0.0, 1.0]])
            >>> res = _StabilityAssessment.assess_stability_from_companion(
            ...     T, allow_unit_roots=True
            ... )
            >>> res.is_stable, res.n_unit_roots, res.n_explosive
            (True, 2, 0)
            >>> _StabilityAssessment.assess_stability_from_companion(T).is_stable
            False
        """
        mat = np.asarray(companion, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise DimensionError(f"Companion matrix must be square 2-D; got shape {mat.shape}.")
        return cls._assess(mat, tol=tol, allow_unit_roots=allow_unit_roots)

    @classmethod
    def is_stationary(cls, ar_coeffs: npt.ArrayLike, *, tol: float = 1e-8) -> bool:
        """Convenience predicate: is the AR/VAR strictly stationary?

        The strict verdict of :meth:`assess_stability` without the record:
        ``True`` only when every companion eigenvalue has modulus below
        ``1 - tol``. Unit roots, and stable roots within ``tol`` of the
        circle, return ``False``.

        Args:
            ar_coeffs: Coefficients ``A_1, ..., A_p``; shape ``(p,)`` or
                ``(p, k, k)``. Empty coefficients are stationary.
            tol: Modulus tolerance.

        Returns:
            ``True`` iff all companion eigenvalues lie strictly inside the
            unit circle, by more than ``tol``.

        Raises:
            SpecificationError: If ``tol`` is negative and ``ar_coeffs`` is
                non-empty.
            DimensionError: If ``ar_coeffs`` is neither rank 1 nor a stack of
                square matrices.
            NumericalError: If ``ar_coeffs`` contains non-finite values, or
                the eigenvalue computation returns them.

        Example:
            >>> _StabilityAssessment.is_stationary([1.5])
            False
            >>> _StabilityAssessment.is_stationary([0.5, -0.3])
            True
            >>> _StabilityAssessment.is_stationary([1.0])
            False
        """
        return cls.assess_stability(ar_coeffs, tol=tol, allow_unit_roots=False).is_stable

    @classmethod
    def is_invertible(cls, ma_coeffs: npt.ArrayLike, *, tol: float = 1e-8) -> bool:
        """Is an MA/ARMA invertible? (companion of the MA polynomial, roots inside).

        The same strict companion test as :meth:`is_stationary`, applied to
        moving-average coefficients. It therefore reads its argument in the
        autoregressive convention and tests ``1 - M_1 z - ... - M_q z^q``.

        Warning:
            A moving-average polynomial written the usual way,
            ``theta(L) = 1 + theta_1 L + ... + theta_q L^q``, must be passed
            **negated**. Passing ``theta`` itself tests a different
            polynomial and returns a wrong verdict whenever the two
            disagree; for ``q == 1`` they happen to coincide. The fitted
            results negate internally, so this applies only to direct calls.

        Args:
            ma_coeffs: MA coefficients ``M_1, ..., M_q`` in the same layout as AR
                coefficients; shape ``(q,)`` or ``(q, k, k)``. In terms of
                ``theta(L)`` above, ``M_j = -theta_j``. Empty coefficients
                are invertible.
            tol: Modulus tolerance.

        Returns:
            ``True`` iff all companion eigenvalues lie strictly inside the unit circle.

        Raises:
            SpecificationError: If ``tol`` is negative and ``ma_coeffs`` is
                non-empty.
            DimensionError: If ``ma_coeffs`` is neither rank 1 nor a stack of
                square matrices.
            NumericalError: If ``ma_coeffs`` contains non-finite values, or
                the eigenvalue computation returns them.

        Example:
            ``1 + 1.5 L + 0.6 L^2`` is invertible, which the method reports
            only for the negated coefficients:

            >>> import numpy as np
            >>> theta = np.array([1.5, 0.6])
            >>> _StabilityAssessment.is_invertible(-theta)
            True
            >>> _StabilityAssessment.is_invertible(theta)
            False
        """
        ma = np.asarray(ma_coeffs, dtype=np.float64)
        if ma.size == 0:
            return True
        return cls._assess(_companion_matrix(ma), tol=tol, allow_unit_roots=False).is_stable
