from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


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
    g_x: npt.NDArray[np.float64]
    h_xx: npt.NDArray[np.float64]
    g_xx: npt.NDArray[np.float64]
    h_ss: npt.NDArray[np.float64]
    g_ss: npt.NDArray[np.float64]
    eta: npt.NDArray[np.float64]
    x_ss: npt.NDArray[np.float64]
    y_ss: npt.NDArray[np.float64]
    order: int

    @property
    def n_states(self) -> int:
        """Predetermined states."""
        return int(self.h_x.shape[0])

    @property
    def n_controls(self) -> int:
        """Non-predetermined variables."""
        return int(self.g_x.shape[0])

    @property
    def n_shocks(self) -> int:
        """Structural shocks."""
        return int(self.eta.shape[1])

    @property
    def is_stable(self) -> bool:
        """Whether the first-order state transition is a contraction."""
        return bool(np.max(np.abs(np.linalg.eigvals(self.h_x))) < 1.0)
