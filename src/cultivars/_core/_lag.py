# filepath: /src/cultivars/core/lag.py
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

"""Lag operator and lag-polynomial algebra.

The lag operator ``L`` shifts a series back one period: ``L y_t = y_{t-1}``.
Every autoregressive model is an equation in a lag polynomial. This module
provides :class:`LagPolynomial`, a value object over the coefficients of a
scalar or matrix polynomial ``c_0 + c_1 L + ... + c_p L**p``.

Sign convention (binding for the whole package):
    An autoregressive polynomial is the *monic* form
    ``phi(L) = I - A_1 L - ... - A_p L**p``. The stored coefficients are
    therefore ``c_0 = I`` and ``c_i = -A_i``. Use :meth:`LagPolynomial.from_ar_coeffs`
    to construct from the ``A_i`` and :meth:`LagPolynomial.ar_coeffs` to recover
    them. Stationarity is assessed on the companion matrix built from the
    ``+A_i`` (see :mod:`cultivars.core.stability`); a flipped sign here silently
    inverts every stationarity verdict.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError, SpecificationError


class LagPolynomial:
    """A scalar or matrix lag polynomial ``c_0 + c_1 L + ... + c_p L**p``.

    The coefficients are stored as an immutable float array of shape
    ``(p + 1,)`` for a scalar polynomial or ``(p + 1, k, k)`` for a matrix
    polynomial over a ``k``-dimensional system.

    Args:
        coeffs: Coefficient array, highest index = highest lag. Shape
            ``(p + 1,)`` (scalar) or ``(p + 1, k, k)`` (matrix, square).

    Raises:
        DimensionError: If ``coeffs`` is not rank 1 or rank 3, is empty, or
            (rank 3) is not square in its trailing dimensions.
        NumericalError: If ``coeffs`` contains non-finite entries.

    Example:
        >>> phi = LagPolynomial.from_ar_coeffs([0.5, -0.3])
        >>> phi.degree
        2
        >>> phi.ar_coeffs()
        array([ 0.5, -0.3])
    """

    __slots__ = ("_c",)

    # Dunder methods
    def __init__(self, coeffs: npt.ArrayLike) -> None:
        """Initialize a lag polynomial with the given coefficients."""
        c = np.asarray(coeffs, dtype=np.float64)
        if c.ndim not in (1, 3):
            raise DimensionError(
                f"LagPolynomial coefficients must be rank 1 (scalar) or rank 3 "
                f"(matrix); got rank {c.ndim} with shape {c.shape}."
            )
        if c.shape[0] < 1:
            raise DimensionError("LagPolynomial requires at least one coefficient.")
        if c.ndim == 3 and c.shape[1] != c.shape[2]:
            raise DimensionError(
                f"Matrix lag polynomial coefficients must be square; got trailing "
                f"dimensions {c.shape[1]}x{c.shape[2]}."
            )
        if not np.all(np.isfinite(c)):
            from cultivars.exceptions import NumericalError

            raise NumericalError("LagPolynomial coefficients contain non-finite values.")
        c.setflags(write=False)
        self._c = c

    def __mul__(self, other: LagPolynomial) -> LagPolynomial:
        """Polynomial product (coefficient convolution)."""
        if not isinstance(other, LagPolynomial):
            return NotImplemented
        if self.is_matrix != other.is_matrix:
            raise DimensionError("Cannot multiply a scalar and a matrix lag polynomial.")
        if self.is_matrix and self.dim != other.dim:
            raise DimensionError(
                f"Matrix polynomials have incompatible dimensions "
                f"{self.dim} and {other.dim}."
            )
        if self.is_matrix:
            k = self.dim
            out = np.zeros((self.degree + other.degree + 1, k, k), dtype=np.float64)
            for i in range(self.degree + 1):
                for j in range(other.degree + 1):
                    out[i + j] += self._c[i] @ other._c[j]
            return LagPolynomial(out)
        return LagPolynomial(np.convolve(self._c, other._c))

    def __eq__(self, other: object) -> bool:
        """Check equality of two lag polynomials."""
        if not isinstance(other, LagPolynomial):
            return NotImplemented
        return self._c.shape == other._c.shape and bool(np.array_equal(self._c, other._c))

    def __repr__(self) -> str:
        """Return a string representation of the lag polynomial."""
        return f"LagPolynomial(degree={self.degree}, dim={self.dim})"

    # Class methods
    @classmethod
    def from_ar_coeffs(cls, ar_coeffs: npt.ArrayLike) -> LagPolynomial:
        """Build the monic AR polynomial ``I - A_1 L - ... - A_p L**p``.

        Args:
            ar_coeffs: The autoregressive coefficients ``A_1, ..., A_p`` as
                they appear in ``y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + e_t``.
                Shape ``(p,)`` (scalar) or ``(p, k, k)`` (matrix).

        Returns:
            The corresponding :class:`LagPolynomial`.

        Raises:
            DimensionError: If ``ar_coeffs`` is not rank 1 or rank 3, or
                (rank 3) is not square.

        Example:
            >>> LagPolynomial.from_ar_coeffs([0.5, -0.3]).coefficients
            array([ 1. , -0.5,  0.3])
        """
        ar = np.asarray(ar_coeffs, dtype=np.float64)
        if ar.ndim == 1:
            c = np.empty(ar.shape[0] + 1, dtype=np.float64)
            c[0] = 1.0
            c[1:] = -ar
        elif ar.ndim == 3:
            if ar.shape[1] != ar.shape[2]:
                raise DimensionError(
                    f"Matrix AR coefficients must be square; got trailing "
                    f"dimensions {ar.shape[1]}x{ar.shape[2]}."
                )
            k = ar.shape[1]
            c = np.zeros((ar.shape[0] + 1, k, k), dtype=np.float64)
            c[0] = np.eye(k)
            c[1:] = -ar
        else:
            raise DimensionError(
                f"AR coefficients must be rank 1 or rank 3; got rank {ar.ndim}."
            )
        return cls(c)

    # Properties
    @property
    def degree(self) -> int:
        """The polynomial degree ``p`` (number of lags)."""
        return int(self._c.shape[0] - 1)

    @property
    def dim(self) -> int:
        """The system dimension ``k`` (``1`` for a scalar polynomial)."""
        return 1 if self._c.ndim == 1 else int(self._c.shape[1])

    @property
    def is_matrix(self) -> bool:
        """Whether this is a matrix (multivariate) polynomial."""
        return self._c.ndim == 3

    @property
    def coefficients(self) -> npt.NDArray[np.float64]:
        """A writable copy of the stored coefficient array."""
        return self._c.copy()

    # Public methods
    def ar_coeffs(self, *, tol: float = 1e-10) -> npt.NDArray[np.float64]:
        """Recover the autoregressive coefficients ``A_1, ..., A_p``.

        Requires the polynomial to be monic (``c_0 == I``), i.e. a genuine AR
        polynomial rather than an arbitrary lag polynomial.

        Args:
            tol: Absolute tolerance for the ``c_0 == I`` check.

        Returns:
            The ``A_i`` with shape ``(p,)`` (scalar) or ``(p, k, k)`` (matrix).

        Raises:
            SpecificationError: If the constant term is not the identity, so
                the polynomial is not in monic AR form.
        """
        c0 = self._c[0]
        identity = np.eye(self.dim) if self.is_matrix else np.array(1.0)
        if not np.allclose(c0, identity, atol=tol):
            raise SpecificationError(
                "ar_coeffs() requires a monic polynomial (constant term == I); "
                "this polynomial is not in autoregressive form."
            )
        return -self._c[1:].copy()

    def evaluate(self, z: complex) -> npt.NDArray[np.complex128] | complex:
        """Evaluate the polynomial at a (complex) point ``z``.

        Computes ``sum_i c_i z**i``. Evaluating at ``z = exp(-1j * omega)``
        yields the frequency-domain transfer function used in spectral methods.

        Args:
            z: The point at which to evaluate; may be complex.

        Returns:
            A complex scalar (scalar polynomial) or a ``(k, k)`` complex array
            (matrix polynomial).

        Example:
            >>> phi = LagPolynomial.from_ar_coeffs([0.5, -0.3])
            >>> round(float(phi.evaluate(1.0).real), 4)
            0.8
        """
        powers = np.power(complex(z), np.arange(self.degree + 1))
        if self.is_matrix:
            return np.tensordot(powers, self._c.astype(np.complex128), axes=(0, 0))
        return complex(np.dot(powers, self._c))

    def roots(self) -> npt.NDArray[np.complex128]:
        """Roots of a scalar polynomial ``c_0 + c_1 x + ... + c_p x**p``.

        Returns:
            The complex roots.

        Raises:
            DimensionError: If the polynomial is a matrix polynomial; use the
                companion eigenvalues in :mod:`cultivars.core.stability` instead.
        """
        if self.is_matrix:
            raise DimensionError(
                "roots() is defined only for scalar polynomials; for matrix "
                "polynomials use companion eigenvalues (cultivars.core.stability)."
            )
        return np.roots(self._c[::-1]).astype(np.complex128)

def _expand_ar(phi: npt.NDArray[np.float64], sphi: npt.NDArray[np.float64], s: int) -> npt.NDArray[np.float64]:
    """Expanded AR coefficients of phi(L) * Phi(L**s)."""
    poly = LagPolynomial.from_ar_coeffs(phi) if phi.size else LagPolynomial([1.0])
    if sphi.size:
        scoef = np.zeros(s * sphi.size + 1)
        scoef[0] = 1.0
        for k in range(sphi.size):
            scoef[(k + 1) * s] = -sphi[k]
        poly = poly * LagPolynomial(scoef)
    return poly.ar_coeffs()

def _expand_ma(theta: npt.NDArray[np.float64], stheta: npt.NDArray[np.float64], s: int) -> npt.NDArray[np.float64]:
    """Expanded MA coefficients of theta(L) * Theta(L**s) (plus-sign convention)."""
    base = np.concatenate([[1.0], theta]) if theta.size else np.array([1.0])
    poly = LagPolynomial(base)
    if stheta.size:
        scoef = np.zeros(s * stheta.size + 1)
        scoef[0] = 1.0
        for k in range(stheta.size):
            scoef[(k + 1) * s] = stheta[k]
        poly = poly * LagPolynomial(scoef)
    return poly.coefficients[1:]

def _ar_infinity(
    d: float,
    ar_params: npt.NDArray[np.float64],
    ma_params: npt.NDArray[np.float64],
    truncation: int,
) -> npt.NDArray[np.float64]:
    """Coefficients ``c_1..c_M`` of ``Pi(L) = phi(L)(1-L)**d / theta(L)`` (monic).

    The model is ``Pi(L)(y_t - mu) = eps_t`` with ``Pi(L) = 1 - c_1 L - ...``, so
    the one-step recursion is ``(y_t - mu) = sum_j c_j (y_{t-j} - mu) + eps_t``.
    """
    frac = fractional_difference_weights(d, truncation + 1)
    phi_poly = np.concatenate([[1.0], -ar_params]) if ar_params.size else np.array([1.0])
    numerator = np.convolve(phi_poly, frac)[: truncation + 1]
    theta_poly = np.concatenate([[1.0], ma_params]) if ma_params.size else np.array([1.0])
    # Series division numerator / theta_poly (theta_poly monic).
    pi = np.zeros(truncation + 1, dtype=np.float64)
    q = ma_params.size
    for i in range(truncation + 1):
        acc = numerator[i]
        for j in range(1, min(i, q) + 1):
            acc -= theta_poly[j] * pi[i - j]
        pi[i] = acc
    return -pi[1:]                       # c_j = -pi_j for the AR(inf) recursion


