# filepath: /src/cultivars/_internals/_rotations.py
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

"""Rotation sampling for set-identified structural models.

The engine behind sign restrictions, kept below the model layer for the same
reason the Kalman recursions are: the draw-and-accept loop is numerical
machinery with no economics in it, and the declaration objects above should
read like declarations.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..exceptions import SpecificationError


def _haar_rotation(rng: np.random.Generator, size: int) -> npt.NDArray[np.float64]:
    """One draw from the uniform distribution over orthogonal matrices.

    The QR decomposition of a Gaussian draw, with the rotation's columns
    sign-fixed by the diagonal of ``R`` -- without that fix the draw is not
    Haar, because plain QR resolves the sign ambiguity in a data-dependent way.

    Args:
        rng: The generator to draw from.
        size: Matrix dimension.

    Returns:
        A ``(size, size)`` orthogonal matrix.
    """
    raw = rng.standard_normal((size, size))
    q, r = np.linalg.qr(raw)
    return np.asarray(q * np.sign(np.diagonal(r)), dtype=np.float64)


def _accepted_rotations(
    factor: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    compiled: tuple[tuple[tuple[int, float], ...], ...],
    *,
    draws: int,
    budget: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], int]:
    """Draw candidate impact matrices and keep those satisfying every sign.

    A restricted column may be flipped wholesale -- a shock and its negative
    are the same rotation -- but columns are never relabeled to rescue a draw,
    so a shock's identity is its declared position.

    Args:
        factor: ``(k, k)`` Cholesky factor of the innovation covariance.
        psi: ``(n_horizons, k, k)`` moving-average matrices at the restricted
            leads only.
        compiled: Per-column ``(variable index, +-1.0)`` requirements, from
            :func:`~cultivars._core._validate_sign_patterns`.
        draws: Accepted rotations to collect.
        budget: Rotations to attempt before giving up.
        rng: The generator behind the draws.

    Returns:
        The ``(n_accepted, k, k)`` accepted impact matrices and the number of
        rotations drawn.

    Raises:
        SpecificationError: If no rotation is accepted within the budget,
            which means the declared signs are jointly unsatisfiable at this
            reduced form, or nearly so.
    """
    k = factor.shape[0]
    accepted: list[npt.NDArray[np.float64]] = []
    attempts = 0
    n_leads = psi.shape[0]
    while len(accepted) < draws and attempts < budget:
        attempts += 1
        candidate = factor @ _haar_rotation(rng, k)
        keep = True
        for column, cells in enumerate(compiled):
            responses = np.einsum("hik,k->hi", psi, candidate[:, column])
            direct = all(sign * responses[h, i] > 0.0 for h in range(n_leads) for i, sign in cells)
            if direct:
                continue
            flipped = all(
                -sign * responses[h, i] > 0.0 for h in range(n_leads) for i, sign in cells
            )
            if flipped:
                candidate[:, column] = -candidate[:, column]
                continue
            keep = False
            break
        if keep:
            accepted.append(candidate.copy())
    if not accepted:
        raise SpecificationError(
            f"no rotation satisfied the declared signs in {attempts} draws. "
            "The restrictions are jointly unsatisfiable at this reduced form, "
            "or nearly so; loosen a sign, shorten the horizons, or reconsider "
            "whether the shocks as declared can coexist."
        )
    return np.stack(accepted), attempts


def _narrative_rotations(
    factor: npt.NDArray[np.float64],
    psi: npt.NDArray[np.float64],
    compiled: tuple[tuple[tuple[int, float], ...], ...],
    *,
    shock_events: tuple[tuple[tuple[int, float], ...], ...],
    contribution_events: tuple[tuple[int, int, int, bool], ...],
    residual_events: npt.NDArray[np.float64],
    draws: int,
    budget: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], int]:
    """Draw impact matrices satisfying declared signs and declared history.

    The sign logic extends :func:`_accepted_rotations` with one fact that
    decides the algorithm: flipping a column of the impact matrix negates that
    shock's recovered series and nothing else, so a column's traditional signs
    and its narrative shock signs must be checked jointly under the flip,
    while contribution events compare absolute contributions and are
    flip-invariant -- they are checked once, after every column's sign is
    resolved.

    Args:
        factor: ``(k, k)`` Cholesky factor of the innovation covariance.
        psi: ``(n_horizons, k, k)`` moving-average matrices at the restricted
            leads.
        compiled: Per-column traditional sign requirements, length ``k``,
            empty where a column declares none.
        shock_events: Per-column narrative shock-sign requirements as
            ``(event index, +-1.0)`` pairs, length ``k``.
        contribution_events: ``(column, variable index, event index,
            overwhelming?)`` requirements.
        residual_events: ``(n_events, k)`` reduced-form residual rows at the
            event periods, in event-index order.
        draws: Accepted rotations to collect.
        budget: Rotations to attempt before giving up.
        rng: The generator behind the draws.

    Returns:
        The ``(n_accepted, k, k)`` accepted impact matrices and the number of
        rotations drawn.

    Raises:
        SpecificationError: If no rotation is accepted within the budget --
            the declared signs and the declared history are jointly
            unsatisfiable at this reduced form, or nearly so.
    """
    k = factor.shape[0]
    accepted: list[npt.NDArray[np.float64]] = []
    attempts = 0
    n_leads = psi.shape[0]
    while len(accepted) < draws and attempts < budget:
        attempts += 1
        candidate = factor @ _haar_rotation(rng, k)
        try:
            shocks = np.linalg.solve(candidate, residual_events.T)
        except np.linalg.LinAlgError:
            continue
        keep = True
        for column in range(k):
            cells = compiled[column]
            events = shock_events[column]
            if not cells and not events:
                continue
            responses = np.einsum("hik,k->hi", psi, candidate[:, column])
            direct = all(
                sign * responses[h, i] > 0.0 for h in range(n_leads) for i, sign in cells
            ) and all(sign * shocks[column, e] > 0.0 for e, sign in events)
            if direct:
                continue
            flipped = all(
                -sign * responses[h, i] > 0.0 for h in range(n_leads) for i, sign in cells
            ) and all(-sign * shocks[column, e] > 0.0 for e, sign in events)
            if flipped:
                candidate[:, column] = -candidate[:, column]
                shocks[column] = -shocks[column]
                continue
            keep = False
            break
        if not keep:
            continue
        for column, variable, event, overwhelming in contribution_events:
            contributions = np.abs(candidate[variable, :] * shocks[:, event])
            own = contributions[column]
            others = np.delete(contributions, column)
            bound = others.sum() if overwhelming else others.max()
            if not own > bound:
                keep = False
                break
        if keep:
            accepted.append(candidate.copy())
    if not accepted:
        raise SpecificationError(
            f"no rotation satisfied the declared signs and the declared "
            f"history in {attempts} draws. The narrative events contradict "
            "the traditional signs, or the history at this reduced form; "
            "re-examine the dates, or loosen a restriction."
        )
    return np.stack(accepted), attempts
