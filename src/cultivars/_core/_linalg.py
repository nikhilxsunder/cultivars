# filepath: /src/cultivars/_core/_linalg.py
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

"""Linear-algebra primitives shared across estimators."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError, NumericalError


def ols(
    design: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], float]:
    """Least-squares fit returning coefficients and the residual sum of squares.

    Uses ``lstsq`` rather than the normal equations so a rank-deficient design
    yields the minimum-norm solution instead of raising.

    Args:
        design: Regressor matrix of shape ``(n, k)``.
        target: Response vector of shape ``(n,)``.

    Returns:
        A tuple ``(beta, ssr)``.

    Raises:
        DimensionError: If the shapes are not conformable.

    Example:
        >>> beta, ssr = ols(np.ones((5, 1)), np.arange(5.0))
        >>> float(beta[0])
        2.0
    """
    if design.ndim != 2 or design.shape[0] != target.shape[0]:
        raise DimensionError(
            f"design {design.shape} is not conformable with target {target.shape}."
        )
    beta, _residuals, _rank, _sv = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    return np.asarray(beta, dtype=np.float64), float(resid @ resid)


def psd_sqrt(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A matrix square root valid for symmetric positive-semidefinite input.

    Negative eigenvalues arising from round-off are clipped to zero, so a
    covariance that is singular or marginally indefinite still yields a usable
    factor rather than a NaN.

    Args:
        matrix: A symmetric positive-semidefinite matrix.

    Returns:
        A factor ``S`` with ``S @ S.T`` equal to ``matrix``.

    Raises:
        DimensionError: If ``matrix`` is not square.
        NumericalError: If the eigendecomposition is not finite.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise DimensionError(f"psd_sqrt requires a square matrix; got {matrix.shape}.")
    eigvals, eigvecs = np.linalg.eigh(matrix)
    if not np.all(np.isfinite(eigvals)):
        raise NumericalError("psd_sqrt eigendecomposition produced non-finite values.")
    root = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0.0, None)))
    return np.asarray(root, dtype=np.float64)


def ergodic_distribution(
    transition: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Stationary distribution of a row-stochastic Markov transition matrix.

    Solves ``pi' P = pi'`` subject to ``sum(pi) = 1`` as a constrained least
    squares problem, which stays well behaved when the chain is near-reducible
    and an eigenvector approach would return a near-degenerate solution.

    Args:
        transition: A ``(K, K)`` row-stochastic matrix.

    Returns:
        The ergodic probabilities, shape ``(K,)``.

    Raises:
        NumericalError: If no valid distribution can be recovered.

    Example:
        >>> np.round(ergodic_distribution(np.array([[0.9, 0.1], [0.2, 0.8]])), 4)
        array([0.6667, 0.3333])
    """
    k = transition.shape[0]
    augmented = np.vstack([transition.T - np.eye(k), np.ones((1, k))])
    rhs = np.zeros(k + 1)
    rhs[-1] = 1.0
    solution, _res, _rank, _sv = np.linalg.lstsq(augmented, rhs, rcond=None)
    pi = np.clip(np.asarray(solution, dtype=np.float64), 0.0, None)
    total = pi.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise NumericalError("transition matrix admits no valid ergodic distribution.")
    return np.asarray(pi / total, dtype=np.float64)
