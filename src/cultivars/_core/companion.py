# filepath: /src/cultivars/core/companion.py
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

"""Companion-form linearization.

A ``VAR(p)`` stacks into a first-order system whose transition matrix is the
companion matrix. The same object serves three consumers: the eigenvalue-based
stability test (:mod:`cultivars.core.stability`), the impulse-response and FEVD
recursions (which are powers of the companion), and the linear-Gaussian
state-space transition matrix when the substrate hosts a VAR.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cultivars._core.lag import LagPolynomial
from cultivars.exceptions import DimensionError, NumericalError


def _normalize_ar(ar_coeffs: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return AR coefficients as a ``(p, k, k)`` float array (scalar -> k=1)."""
    ar = np.asarray(ar_coeffs, dtype=np.float64)
    if ar.ndim == 1:
        ar = ar.reshape(ar.shape[0], 1, 1)
    elif ar.ndim == 3:
        if ar.shape[1] != ar.shape[2]:
            raise DimensionError(
                f"Matrix AR coefficients must be square; got trailing dimensions "
                f"{ar.shape[1]}x{ar.shape[2]}."
            )
    else:
        raise DimensionError(
            f"AR coefficients must be rank 1 (scalar) or rank 3 (matrix); "
            f"got rank {ar.ndim}."
        )
    if not np.all(np.isfinite(ar)):
        raise NumericalError("AR coefficients contain non-finite values.")
    return ar


def companion_matrix(ar_coeffs: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Build the companion matrix from autoregressive coefficients.

    For ``y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + u_t`` the companion is the
    ``(kp, kp)`` matrix whose top block row is ``[A_1, ..., A_p]`` and whose
    sub-diagonal is the identity.

    Args:
        ar_coeffs: Coefficients ``A_1, ..., A_p``. Shape ``(p,)`` (scalar,
            ``k = 1``) or ``(p, k, k)`` (matrix).

    Returns:
        The companion matrix of shape ``(k * p, k * p)``.

    Raises:
        DimensionError: If ``ar_coeffs`` has an unsupported rank or is not
            square (matrix case), or if ``p < 1``.
        NumericalError: If ``ar_coeffs`` contains non-finite values.

    Example:
        >>> C = companion_matrix([0.5, -0.3])
        >>> C.shape
        (2, 2)
        >>> bool(C[1, 0] == 1.0)
        True
    """
    ar = _normalize_ar(ar_coeffs)
    p, k = ar.shape[0], ar.shape[1]
    if p < 1:
        raise DimensionError("companion_matrix requires at least one lag (p >= 1).")
    size = k * p
    companion = np.zeros((size, size), dtype=np.float64)
    companion[:k, :] = np.concatenate([ar[i] for i in range(p)], axis=1)
    if p > 1:
        companion[k:, : k * (p - 1)] = np.eye(k * (p - 1))
    return companion


def companion_from_polynomial(poly: LagPolynomial) -> npt.NDArray[np.float64]:
    """Build the companion matrix from a monic AR :class:`LagPolynomial`.

    Args:
        poly: A polynomial in monic AR form (constant term == identity).

    Returns:
        The companion matrix of shape ``(k * p, k * p)``.

    Raises:
        SpecificationError: If ``poly`` is not in monic AR form.
    """
    return companion_matrix(poly.ar_coeffs())


def selector_matrix(k: int, p: int) -> npt.NDArray[np.float64]:
    """The selector ``J = [I_k, 0, ..., 0]`` extracting the leading block.

    Used to project companion powers back to the original ``k`` variables, e.g.
    the reduced-form impulse responses ``Psi_h = J @ C**h @ J.T``.

    Args:
        k: System dimension.
        p: Lag order.

    Returns:
        A ``(k, k * p)`` array.

    Raises:
        DimensionError: If ``k < 1`` or ``p < 1``.
    """
    if k < 1 or p < 1:
        raise DimensionError(f"selector_matrix requires k >= 1 and p >= 1; got k={k}, p={p}.")
    selector = np.zeros((k, k * p), dtype=np.float64)
    selector[:, :k] = np.eye(k)
    return selector
