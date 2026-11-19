# filepath: /src/cultivars/engine/_internals/_solutions.py
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
"""Perturbation solutions: the policy functions of a solved equilibrium model.

One record, :class:`_PerturbationSolution`, holding the Taylor coefficients
of the state transition and the control policy around the deterministic
steady state. The solver in ``_solvers`` produces it; the emitters turn it
into a state space, the simulators iterate it, and the posterior record
recomputes it draw by draw.

References:
    Schmitt-Grohe, S., & Uribe, M. (2004). Solving dynamic general
    equilibrium models using a second-order approximation to the
    policy function. *Journal of Economic Dynamics and Control*,
    28(4), 755-775.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "_PerturbationSolution",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class _PerturbationSolution:
    """A second-order perturbation solution in Schmitt-Grohe-Uribe form.

    With ``x`` the predetermined states (deviations from steady state) and
    ``y`` the controls::

        y = g_x x + 1/2 g_xx (x kron x) + 1/2 g_ss sigma^2
        x' = h_x x + 1/2 h_xx (x kron x) + 1/2 h_ss sigma^2 + eta eps'

    Attributes:
        h_x: ``(n_x, n_x)`` first-order state transition.
        g_x: ``(n_y, n_x)`` first-order policy.
        h_xx: ``(n_x, n_x * n_x)`` second-order state coefficients, acting on
            ``x kron x``.
        g_xx: ``(n_y, n_x * n_x)`` second-order policy coefficients.
        h_ss: ``(n_x,)`` risk correction of the states.
        g_ss: ``(n_y,)`` risk correction of the controls.
        eta: ``(n_x, n_eps)`` shock loading.
        x_ss: ``(n_x,)`` steady-state states.
        y_ss: ``(n_y,)`` steady-state controls.
        order: ``1`` or ``2``; at order one every second-order block is zero.
    """

    h_x: npt.NDArray[np.float64]
    """First-order state transition, ``(n_x, n_x)``."""
    g_x: npt.NDArray[np.float64]
    """First-order control policy, ``(n_y, n_x)``."""
    h_xx: npt.NDArray[np.float64]
    """Second-order state coefficients, ``(n_x, n_x * n_x)``, acting on ``x kron x``."""
    g_xx: npt.NDArray[np.float64]
    """Second-order control coefficients, ``(n_y, n_x * n_x)``, acting on ``x kron x``."""
    h_ss: npt.NDArray[np.float64]
    """Risk correction of the states, ``(n_x,)``."""
    g_ss: npt.NDArray[np.float64]
    """Risk correction of the controls, ``(n_y,)``."""
    eta: npt.NDArray[np.float64]
    """Shock loading on the states, ``(n_x, n_eps)``."""
    x_ss: npt.NDArray[np.float64]
    """Steady-state values of the states, ``(n_x,)``."""
    y_ss: npt.NDArray[np.float64]
    """Steady-state values of the controls, ``(n_y,)``."""
    order: int
    """Approximation order, ``1`` or ``2``; at order one every second-order block is zero."""

    @property
    def n_states(self) -> int:
        """Predetermined states.

        Returns:
            The number ``n_x`` of states, read from ``h_x``.
        """
        return int(self.h_x.shape[0])

    @property
    def n_controls(self) -> int:
        """Non-predetermined variables.

        Returns:
            The number ``n_y`` of controls, read from ``g_x``.
        """
        return int(self.g_x.shape[0])

    @property
    def n_shocks(self) -> int:
        """Structural shocks.

        Returns:
            The number ``n_eps`` of shocks, read from ``eta``.
        """
        return int(self.eta.shape[1])

    @property
    def is_stable(self) -> bool:
        """Whether the first-order state transition is a contraction.

        Returns:
            ``True`` when every eigenvalue of ``h_x`` has modulus strictly
            below one. No tolerance is applied, so a unit root reads as
            unstable. Raises ``ValueError`` for a model with no states.
        """
        return bool(np.max(np.abs(np.linalg.eigvals(self.h_x))) < 1.0)
