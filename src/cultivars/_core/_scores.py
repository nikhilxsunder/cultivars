# filepath: /src/cultivars/_core/_scores.py
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

"""Scoring primitives: exact functionals of a predictive sample.

Each function here evaluates a strictly proper scoring rule -- or the
probability integral transform that calibration reads -- directly from a
predictive *sample*, using exact finite-sample identities rather than
density estimates wherever the mathematics allows. The orientation
convention is stated once: every score is negatively oriented, smaller is
better, so rules and losses can share tables without a legend. Draws are
axis zero everywhere.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError, SpecificationError


def _check_draws(draws: npt.NDArray[np.float64], realized: npt.NDArray[np.float64]) -> None:
    """Reject a draw block and realization that cannot be scored together.

    Raises:
        DimensionError: If the shapes disagree.
        SpecificationError: If fewer than two draws are offered.
    """
    if draws.ndim != 2 or realized.ndim != 1 or draws.shape[1] != realized.shape[0]:
        raise DimensionError(
            f"draws must be (n_draws, m) against a (m,) realization; got "
            f"{draws.shape} against {realized.shape}."
        )
    if draws.shape[0] < 2:
        raise SpecificationError(
            f"scoring a predictive sample needs at least two draws; got {draws.shape[0]}."
        )


def crps_from_draws(
    draws: npt.NDArray[np.float64], realized: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The continuous ranked probability score, exactly, per column.

    The empirical-distribution form ``E|X - y| - E|X - X'| / 2`` evaluated
    by the sorted identity, so the pairwise term costs a sort rather than a
    quadratic sweep. Strictly proper for distributions with finite mean;
    negatively oriented; in the units of the variable, which is why it is
    the headline score for a single series.

    Args:
        draws: ``(n_draws, m)`` predictive sample, one column per scored
            quantity.
        realized: ``(m,)`` outcomes.

    Returns:
        The ``(m,)`` scores.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> draws = rng.standard_normal((200000, 1))
        >>> value = float(crps_from_draws(draws, np.zeros(1))[0])
        >>> theory = 1.0 / np.sqrt(np.pi) * (np.sqrt(2.0) - 1.0)
        >>> bool(abs(value - theory) < 5e-3)
        True
    """
    _check_draws(draws, realized)
    count = draws.shape[0]
    ordered = np.sort(draws, axis=0)
    absolute = np.abs(draws - realized[None, :]).mean(axis=0)
    ranks = 2.0 * np.arange(1, count + 1, dtype=np.float64) - count - 1.0
    spread = (ranks[:, None] * ordered).sum(axis=0) / count**2
    return np.asarray(absolute - spread, dtype=np.float64)


def energy_score(draws: npt.NDArray[np.float64], realized: npt.NDArray[np.float64]) -> float:
    """The energy score of a multivariate predictive sample.

    ``E||X - y|| - E||X - X'|| / 2`` in the Euclidean norm: the
    multivariate generalization of the CRPS (to which it reduces exactly at
    one dimension), strictly proper, negatively oriented. Computed from the
    sample by direct pairwise norms, chunked so memory stays linear in the
    draw count.

    Args:
        draws: ``(n_draws, k)`` predictive sample.
        realized: ``(k,)`` outcome vector.

    Returns:
        The score.
    """
    _check_draws(draws, realized)
    count = draws.shape[0]
    first = float(np.linalg.norm(draws - realized[None, :], axis=1).mean())
    pairwise = 0.0
    step = max(1, 2_000_000 // (count * draws.shape[1] + 1))
    for start in range(0, count, step):
        block = draws[start : start + step]
        pairwise += float(np.linalg.norm(block[:, None, :] - draws[None, :, :], axis=2).sum())
    return first - pairwise / (2.0 * count**2)


def pit_from_draws(
    draws: npt.NDArray[np.float64], realized: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The probability integral transform of each realization, per column.

    The predictive sample's empirical distribution function evaluated at
    the outcome, with the half-count tie convention. Under a correctly
    calibrated predictive the transforms are uniform on ``(0, 1)``; the
    values are clipped away from the exact endpoints by half a draw's
    worth of probability, so a normal-quantile transform downstream cannot
    produce infinities from a finite sample.

    Args:
        draws: ``(n_draws, m)`` predictive sample.
        realized: ``(m,)`` outcomes.

    Returns:
        The ``(m,)`` transforms, interior to ``(0, 1)``.
    """
    _check_draws(draws, realized)
    count = draws.shape[0]
    below = (draws < realized[None, :]).sum(axis=0).astype(np.float64)
    ties = (draws == realized[None, :]).sum(axis=0).astype(np.float64)
    value = (below + 0.5 * ties) / count
    return np.asarray(np.clip(value, 0.5 / count, 1.0 - 0.5 / count), dtype=np.float64)


def _kernel_log_score(
    draws: npt.NDArray[np.float64], realized: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Negative log predictive density at the outcome, per column, by Gaussian kernel.

    The density is estimated on the draws with Silverman's rule of thumb
    (the smaller of the standard deviation and the interquartile range
    over 1.34, times ``0.9 n**-1/5``), floored so that a degenerate column
    still scores. The one *estimated* score in the package: the CRPS and
    energy score are exact functionals of the sample; this one is a
    kernel estimate, and the most sensitive to tail misfit.

    Args:
        draws: ``(n_draws, m)`` predictive sample.
        realized: ``(m,)`` outcomes.

    Returns:
        The ``(m,)`` negative log scores.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> value = float(_kernel_log_score(rng.standard_normal((20000, 1)), np.zeros(1))[0])
        >>> bool(abs(value - 0.5 * np.log(2.0 * np.pi)) < 0.05)
        True
    """
    _check_draws(draws, realized)
    count = draws.shape[0]
    spread = draws.std(axis=0, ddof=1)
    quartiles = np.quantile(draws, [0.25, 0.75], axis=0)
    robust = (quartiles[1] - quartiles[0]) / 1.34
    width = 0.9 * np.minimum(spread, np.where(robust > 0.0, robust, spread))
    width = np.maximum(width * count ** (-0.2), 1e-8 * np.maximum(spread, 1.0))
    gap = (draws - realized[None, :]) / width[None, :]
    peak = (-0.5 * gap**2).max(axis=0)
    total = np.exp(-0.5 * gap**2 - peak[None, :]).sum(axis=0)
    return np.asarray(
        -(peak + np.log(total) - np.log(count) - np.log(width) - 0.5 * np.log(2.0 * np.pi)),
        dtype=np.float64,
    )


def _pinball_loss(
    realized: npt.NDArray[np.float64], quantile: npt.NDArray[np.float64], tau: float
) -> npt.NDArray[np.float64]:
    """The pinball (check) loss of a ``tau``-quantile forecast, elementwise.

    ``(tau - 1{y < q}) (y - q)``: the loss whose expectation a conditional
    quantile minimizes, so it is the proper score for a quantile forecast
    the way the squared error is for a conditional mean. Negatively
    oriented; in the variable's units.

    Args:
        realized: Outcomes, any shape.
        quantile: Quantile forecasts of the same shape.
        tau: The level forecast, strictly inside ``(0, 1)``.

    Returns:
        The losses, same shape.

    Raises:
        DimensionError: If the shapes disagree.
        SpecificationError: If the level is outside ``(0, 1)``.

    Example:
        >>> _pinball_loss(np.array([1.0, -1.0]), np.zeros(2), 0.9)
        array([0.9, 0.1])
    """
    if not 0.0 < tau < 1.0:
        raise SpecificationError(f"tau must lie strictly inside (0, 1); got {tau}.")
    if realized.shape != quantile.shape:
        raise DimensionError(
            f"realized and quantile forecasts must share a shape; got {realized.shape} and "
            f"{quantile.shape}."
        )
    gap = realized - quantile
    return np.asarray(np.where(gap >= 0.0, tau * gap, (tau - 1.0) * gap), dtype=np.float64)
