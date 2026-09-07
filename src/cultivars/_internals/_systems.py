"""Parametric state-space system builders: structural and Nelson-Siegel.

These assemble specific parametric families -- Harvey's structural
components and the dynamic Nelson-Siegel term structure -- into the
linear-Gaussian substrate. They sit one layer above
:mod:`._state_space`, which stays a leaf: the substrate knows how to
filter and smooth any linear-Gaussian system, and this module knows how
to construct the particular systems those two model families need. Each
parameter record here is the single meeting point between an objective's
flat optimization vector and the system matrices, so the objectives, the
fits, and the public results all speak in these records rather than in
raw arrays.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._parameters import _StructuralParameters


def _structural_matrices(
    params: _StructuralParameters, *, trend: str, cycle: bool, seasonal: int | None
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    dict[str, slice],
]:
    """System matrices and state layout of a structural time-series model.

    Assembles Harvey's components: a level (with or without its own
    noise, and with or without a stochastic slope), a damped stochastic
    cycle, and a stochastic trigonometric seasonal. The trend and
    seasonal blocks carry unit roots, so their initial covariance is a
    large approximate-diffuse block; the cycle block is stationary and
    initialized at its stationary covariance.

    Args:
        params: The parameter record; fields must match the specification.
        trend: ``"level"`` (local level), ``"lltrend"`` (local linear
            trend), or ``"smooth"`` (integrated random walk).
        cycle: Whether the damped stochastic cycle is present.
        seasonal: Trigonometric seasonal period, or ``None``.

    Returns:
        ``(Z, T, R, Q, H, P1, slices)`` where ``slices`` maps component
        names (``"level"``, ``"slope"``, ``"cycle"``, ``"seasonal"``) to
        their state index ranges and ``P1`` is the initial covariance.
    """
    z_parts: list[npt.NDArray[np.float64]] = []
    t_blocks: list[npt.NDArray[np.float64]] = []
    noise_vars: list[float] = []
    noise_rows: list[int] = []
    p1_diag: list[float] = []
    slices: dict[str, slice] = {}
    offset = 0

    if trend == "level":
        z_parts.append(np.array([1.0]))
        t_blocks.append(np.array([[1.0]]))
        assert params.sigma2_level is not None
        noise_vars.append(params.sigma2_level)
        noise_rows.append(offset)
        p1_diag.extend([1e6])
        slices["level"] = slice(offset, offset + 1)
        offset += 1
    else:
        z_parts.append(np.array([1.0, 0.0]))
        t_blocks.append(np.array([[1.0, 1.0], [0.0, 1.0]]))
        if trend == "lltrend":
            assert params.sigma2_level is not None
            noise_vars.append(params.sigma2_level)
            noise_rows.append(offset)
        assert params.sigma2_slope is not None
        noise_vars.append(params.sigma2_slope)
        noise_rows.append(offset + 1)
        p1_diag.extend([1e6, 1e6])
        slices["level"] = slice(offset, offset + 1)
        slices["slope"] = slice(offset + 1, offset + 2)
        offset += 2

    if cycle:
        assert params.cycle_rho is not None
        assert params.cycle_freq is not None
        assert params.sigma2_cycle is not None
        rho, freq = params.cycle_rho, params.cycle_freq
        rotation = rho * np.array(
            [
                [np.cos(freq), np.sin(freq)],
                [-np.sin(freq), np.cos(freq)],
            ]
        )
        z_parts.append(np.array([1.0, 0.0]))
        t_blocks.append(rotation)
        noise_vars.extend([params.sigma2_cycle, params.sigma2_cycle])
        noise_rows.extend([offset, offset + 1])
        stationary = params.sigma2_cycle / max(1.0 - rho**2, 1e-10)
        p1_diag.extend([stationary, stationary])
        slices["cycle"] = slice(offset, offset + 2)
        offset += 2

    if seasonal is not None:
        assert params.sigma2_seasonal is not None
        start = offset
        for harmonic in range(1, seasonal // 2 + 1):
            angle = 2.0 * np.pi * harmonic / seasonal
            if seasonal % 2 == 0 and harmonic == seasonal // 2:
                z_parts.append(np.array([1.0]))
                t_blocks.append(np.array([[-1.0]]))
                noise_vars.append(params.sigma2_seasonal)
                noise_rows.append(offset)
                p1_diag.extend([1e6])
                offset += 1
            else:
                z_parts.append(np.array([1.0, 0.0]))
                t_blocks.append(
                    np.array(
                        [
                            [np.cos(angle), np.sin(angle)],
                            [-np.sin(angle), np.cos(angle)],
                        ]
                    )
                )
                noise_vars.extend([params.sigma2_seasonal, params.sigma2_seasonal])
                noise_rows.extend([offset, offset + 1])
                p1_diag.extend([1e6, 1e6])
                offset += 2
        slices["seasonal"] = slice(start, offset)

    m = offset
    design = np.concatenate(z_parts)[None, :]
    transition = np.zeros((m, m))
    at = 0
    for block in t_blocks:
        width = block.shape[0]
        transition[at : at + width, at : at + width] = block
        at += width
    r = len(noise_vars)
    selection = np.zeros((m, r))
    for column, row in enumerate(noise_rows):
        selection[row, column] = 1.0
    state_cov = np.diag(np.asarray(noise_vars, dtype=np.float64))
    obs_cov = np.array([[params.sigma2_irregular]])
    initial_cov = np.diag(np.asarray(p1_diag, dtype=np.float64))
    return design, transition, selection, state_cov, obs_cov, initial_cov, slices
