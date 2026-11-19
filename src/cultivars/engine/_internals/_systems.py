# filepath: /src/cultivars/engine/_internals/_systems.py
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

"""Parametric state-space system builders: Harvey's structural components.

Assembles one parametric family, Harvey's structural time-series
components, into the system matrices of the linear-Gaussian substrate. It
sits one layer above ``_substrates``, which stays a leaf: the substrate
knows how to filter and smooth any linear-Gaussian system, and this module
knows how to construct the particular system a structural model needs. The
parameter record it reads is the single meeting point between an
objective's flat optimization vector and the system matrices, so the
objectives, the fits, and the public results all speak in that record and
not in raw arrays.

The dynamic Nelson-Siegel counterpart is built by the substrate's own
``_from_nelson_siegel_system`` constructor.

References:
    Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and
    the Kalman Filter*. Cambridge University Press.

    Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State
    Space Methods* (2nd ed.). Oxford University Press.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._parameters import _StructuralParameters

__all__ = [
    "_structural_matrices",
]


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

    The state is ordered level, slope (unless the trend is ``"level"``),
    the two cycle states, then the seasonal harmonics: two states per
    harmonic, and one for the Nyquist harmonic of an even period, which is
    ``seasonal - 1`` states in all. The observation loads on the level, on
    the first cycle state, and on the first state of each harmonic.

    ``trend`` is not validated here. Any value other than ``"level"``
    builds the two-state trend, and of those only ``"lltrend"`` gives the
    level its own noise.

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

    Raises:
        AssertionError: If a field of ``params`` that the requested
            components need is ``None``.

    Example:
        A local linear trend with a quarterly seasonal has five states:

        >>> from cultivars.engine._internals._parameters import _StructuralParameters
        >>> params = _StructuralParameters(
        ...     sigma2_irregular=1.0,
        ...     sigma2_level=0.1,
        ...     sigma2_slope=0.01,
        ...     cycle_rho=None,
        ...     cycle_freq=None,
        ...     sigma2_cycle=None,
        ...     sigma2_seasonal=0.05,
        ... )
        >>> Z, T, R, Q, H, P1, slices = _structural_matrices(
        ...     params, trend="lltrend", cycle=False, seasonal=4
        ... )
        >>> Z.shape, T.shape, R.shape, Q.shape
        ((1, 5), (5, 5), (5, 5), (5, 5))
        >>> slices["seasonal"]
        slice(2, 5, None)
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
