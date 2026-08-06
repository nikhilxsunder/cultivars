# filepath: /src/cultivars/core/stability.py
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

"""Stationarity and invertibility assessment via companion eigenvalues.

An AR/VAR is stationary iff every eigenvalue of its companion matrix lies
strictly inside the unit circle; an MA/ARMA is invertible iff the companion of
its MA polynomial satisfies the same condition. The ``allow_unit_roots`` mode
supports models that sit *on* the unit circle by construction (e.g. VECM), for
which only strictly explosive roots indicate a problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cultivars._core._companion import companion_matrix
from cultivars.exceptions import NumericalError


@dataclass(frozen=True)
class StabilityResult:
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


def _assess(
    companion: npt.NDArray[np.float64], *, tol: float, allow_unit_roots: bool
) -> StabilityResult:
    if tol < 0.0:
        from cultivars.exceptions import SpecificationError

        raise SpecificationError(f"tol must be non-negative; got {tol}.")
    if companion.size == 0:
        return StabilityResult(
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
    return StabilityResult(
        eigenvalues=eigenvalues,
        max_modulus=max_modulus,
        is_stable=is_stable,
        n_unit_roots=n_unit_roots,
        n_explosive=n_explosive,
        tol=tol,
    )


def assess_stability(
    ar_coeffs: npt.ArrayLike, *, tol: float = 1e-8, allow_unit_roots: bool = False
) -> StabilityResult:
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
    return _assess(companion_matrix(ar_coeffs), tol=tol, allow_unit_roots=allow_unit_roots)


def assess_stability_from_companion(
    companion: npt.ArrayLike, *, tol: float = 1e-8, allow_unit_roots: bool = False
) -> StabilityResult:
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
        from cultivars.exceptions import DimensionError

        raise DimensionError(f"Companion matrix must be square 2-D; got shape {mat.shape}.")
    return _assess(mat, tol=tol, allow_unit_roots=allow_unit_roots)


def is_stationary(ar_coeffs: npt.ArrayLike, *, tol: float = 1e-8) -> bool:
    """Convenience predicate: is the AR/VAR strictly stationary?

    Example:
        >>> is_stationary([1.5])
        False
    """
    return assess_stability(ar_coeffs, tol=tol, allow_unit_roots=False).is_stable


def is_invertible(ma_coeffs: npt.ArrayLike, *, tol: float = 1e-8) -> bool:
    """Is an MA/ARMA invertible? (companion of the MA polynomial, roots inside).

    Args:
        ma_coeffs: MA coefficients ``M_1, ..., M_q`` in the same layout as AR
            coefficients; shape ``(q,)`` or ``(q, k, k)``.
        tol: Modulus tolerance.

    Returns:
        ``True`` iff all companion eigenvalues lie strictly inside the unit circle.
    """
    return _assess(companion_matrix(ma_coeffs), tol=tol, allow_unit_roots=False).is_stable
