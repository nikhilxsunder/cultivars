from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._solutions import _PerturbationSolution


def _simulate_pruned(
    solution: _PerturbationSolution,
    shocks: npt.NDArray[np.float64],
    *,
    initial_first: npt.NDArray[np.float64] | None = None,
    initial_second: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Deterministic pass of the pruned system through a shock sequence.

    Args:
        solution: The perturbation solution.
        shocks: ``(T, n_eps)`` structural shocks.
        initial_first: Starting first-order state, default zero.
        initial_second: Starting second-order state, default zero.

    Returns:
        ``(states, controls)`` as ``(T, n_x)`` and ``(T, n_y)`` *levels*
        (steady state added back), dated after each period's shock.
    """
    n_x = solution.n_states
    horizon = shocks.shape[0]
    x_f = np.zeros(n_x) if initial_first is None else np.asarray(initial_first, dtype=np.float64)
    x_s = np.zeros(n_x) if initial_second is None else np.asarray(initial_second, dtype=np.float64)
    states = np.empty((horizon, n_x))
    controls = np.empty((horizon, solution.n_controls))
    for t in range(horizon):
        previous = x_f
        x_f = solution.h_x @ previous + solution.eta @ shocks[t]
        x_s = (
            solution.h_x @ x_s
            + 0.5 * solution.h_xx @ np.kron(previous, previous)
            + 0.5 * solution.h_ss
        )
        x_dev = x_f + x_s
        y_dev = solution.g_x @ x_dev + 0.5 * solution.g_xx @ np.kron(x_f, x_f) + 0.5 * solution.g_ss
        states[t] = solution.x_ss + x_dev
        controls[t] = solution.y_ss + y_dev
    return states, controls


def _impulse_responses(
    solution: _PerturbationSolution, *, horizon: int, size: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Responses of states and controls to each shock, ``(n_eps, horizon, n)``.

    At first order these are the usual linear responses. At second order
    the response depends on where the economy starts, and the convention
    here is the one Andreasen et al. call the response from the
    *stochastic* steady state: the pruned system is run without shocks
    until its second-order state settles, then a one-time impulse of
    ``size`` standard deviations is added, and the difference from the
    unshocked continuation is reported.
    """
    n_x, n_y, n_eps = solution.n_states, solution.n_controls, solution.n_shocks
    settle = 500
    quiet = np.zeros((settle, n_eps))
    states0, _ = _simulate_pruned(solution, quiet)
    base_first = np.zeros(n_x)
    base_second = states0[-1] - solution.x_ss
    zeros = np.zeros((horizon, n_eps))
    base_s, base_c = _simulate_pruned(
        solution, zeros, initial_first=base_first, initial_second=base_second
    )
    out_states = np.empty((n_eps, horizon, n_x))
    out_controls = np.empty((n_eps, horizon, n_y))
    for e in range(n_eps):
        shocks = zeros.copy()
        shocks[0, e] = size
        s_path, c_path = _simulate_pruned(
            solution, shocks, initial_first=base_first, initial_second=base_second
        )
        out_states[e] = s_path - base_s
        out_controls[e] = c_path - base_c
    return out_states, out_controls
