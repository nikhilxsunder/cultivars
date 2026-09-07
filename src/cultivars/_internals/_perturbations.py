# filepath: /src/cultivars/_internals/_perturbation.py
#
# (MIT header)
"""Perturbation solution of rational-expectations models, to second order.

A model is a system of equilibrium conditions ``E_t f(y', y, x', x) = 0``
with ``x`` the predetermined states (including any exogenous driving
processes) and ``y`` the controls, closed by ``x' = h(x, sigma) + sigma
eta eps'``. Schmitt-Grohe and Uribe (2004) showed that the Taylor
coefficients of the policy functions ``g`` and ``h`` around the
deterministic steady state solve a sequence of *linear* problems once the
first-order coefficients are known: the first order is the Blanchard-Kahn
/ Klein generalized-Schur problem, the second order in ``x`` is one linear
system in ``(g_xx, h_xx)``, and the risk correction is one linear system in
``(g_ss, h_ss)`` -- with the cross terms ``g_xs``, ``h_xs`` identically
zero.

Derivatives of ``f`` are taken numerically (central differences), so the
model is supplied as a plain callable and no symbolic layer is needed;
the cost is a step-size floor on accuracy of order ``1e-8`` in the second
derivatives, immaterial next to the perturbation error itself.

The second-order solution is carried as the *pruned* state space of Kim,
Kim, Schaumburg, and Sims (2008) in the form of Andreasen,
Fernandez-Villaverde, and Rubio-Ramirez (2018): a first-order state and a
second-order state evolve side by side, the quadratic term is fed only by
the first-order state, and the system cannot explode. That is what makes
it a well-posed customer of the nonlinear filters.

References:
    Schmitt-Grohe, S., & Uribe, M. (2004). Solving dynamic general
        equilibrium models using a second-order approximation to the
        policy function. *Journal of Economic Dynamics and Control*,
        28(4), 755-775.
    Klein, P. (2000). Using the generalized Schur form to solve a
        multivariate linear rational expectations model. *Journal of
        Economic Dynamics and Control*, 24(10), 1405-1423.
    Kim, J., Kim, S., Schaumburg, E., & Sims, C. A. (2008). Calculating
        and using second-order accurate solutions of discrete time
        dynamic equilibrium models. *Journal of Economic Dynamics and
        Control*, 32(11), 3397-3414.
    Andreasen, M. M., Fernandez-Villaverde, J., & Rubio-Ramirez, J. F.
        (2018). The pruned state-space system for non-linear DSGE
        models: Theory and empirical applications. *Review of Economic
        Studies*, 85(1), 1-49.
    Fernandez-Villaverde, J., & Rubio-Ramirez, J. F. (2007). Estimating
        macroeconomic models: A likelihood approach. *Review of Economic
        Studies*, 74(4), 1059-1087.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt
import scipy.linalg as sla

from ..exceptions import NumericalError
from ._solutions import _PerturbationSolution


def _numerical_jacobian(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    point: npt.NDArray[np.float64],
    *,
    step: float = 1e-6,
) -> npt.NDArray[np.float64]:
    """Central-difference Jacobian ``(n_out, n_in)`` of a vector map."""
    base = np.asarray(fun(point), dtype=np.float64).ravel()
    out = np.empty((base.shape[0], point.shape[0]))
    for j in range(point.shape[0]):
        h = step * max(1.0, abs(float(point[j])))
        up = point.copy()
        down = point.copy()
        up[j] += h
        down[j] -= h
        out[:, j] = (np.asarray(fun(up)).ravel() - np.asarray(fun(down)).ravel()) / (2.0 * h)
    return out


def _numerical_hessian(
    fun: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    point: npt.NDArray[np.float64],
    *,
    step: float = 1e-4,
) -> npt.NDArray[np.float64]:
    """Central-difference Hessian ``(n_out, n_in, n_in)`` of a vector map.

    Uses the symmetric four-point stencil for the off-diagonal entries and
    the three-point stencil for the diagonal, then symmetrizes.
    """
    base = np.asarray(fun(point), dtype=np.float64).ravel()
    n_out, n_in = base.shape[0], point.shape[0]
    steps = np.array([step * max(1.0, abs(float(v))) for v in point])
    out = np.empty((n_out, n_in, n_in))

    def at(shift: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(fun(point + shift), dtype=np.float64).ravel()

    for j in range(n_in):
        ej = np.zeros(n_in)
        ej[j] = steps[j]
        out[:, j, j] = (at(ej) - 2.0 * base + at(-ej)) / steps[j] ** 2
        for k in range(j + 1, n_in):
            ek = np.zeros(n_in)
            ek[k] = steps[k]
            mixed = (at(ej + ek) - at(ej - ek) - at(-ej + ek) + at(-ej - ek)) / (
                4.0 * steps[j] * steps[k]
            )
            out[:, j, k] = mixed
            out[:, k, j] = mixed
    return out


def _stack_point(
    y_next: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    x_next: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """The ``(y', y, x', x)`` argument order flattened into one vector."""
    return np.concatenate([y_next, y, x_next, x])


def _first_order(
    f_y_next: npt.NDArray[np.float64],
    f_y: npt.NDArray[np.float64],
    f_x_next: npt.NDArray[np.float64],
    f_x: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Klein's generalized-Schur solution of the linearized system.

    With ``w = (x, y)`` the system is ``A E_t w' = B w``, ``A = [f_x',
    f_y']``, ``B = -[f_x, f_y]``. Ordering the stable generalized
    eigenvalues first gives ``g_x = Z21 Z11^{-1}`` and ``h_x = Z11 S11^{-1}
    T11 Z11^{-1}``.

    Returns:
        ``(h_x, g_x)``.

    Raises:
        NumericalError: If the Blanchard-Kahn count fails (no unique
            stable solution) or the decomposition is singular.
    """
    n_x = f_x.shape[1]
    a_mat = np.hstack([f_x_next, f_y_next])
    b_mat = -np.hstack([f_x, f_y])

    def stable(
        alpha: npt.NDArray[np.complex128], beta: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.bool_]:
        # the eigenvalues of s' = S^{-1} T s are beta / alpha; keep those inside
        return np.abs(beta) < np.abs(alpha)

    try:
        s_mat, t_mat, alpha, beta, _, z_mat = sla.ordqz(a_mat, b_mat, sort=stable, output="real")
    except (ValueError, np.linalg.LinAlgError) as error:
        raise NumericalError("the generalized Schur decomposition failed.") from error
    n_stable = int(np.sum(stable(alpha, beta)))
    if n_stable != n_x:
        raise NumericalError(
            f"Blanchard-Kahn conditions fail: {n_stable} stable eigenvalues for "
            f"{n_x} predetermined states (need exactly {n_x})."
        )
    z11 = z_mat[:n_x, :n_x]
    z21 = z_mat[n_x:, :n_x]
    s11 = s_mat[:n_x, :n_x]
    t11 = t_mat[:n_x, :n_x]
    if abs(np.linalg.det(z11)) < 1e-14:
        raise NumericalError("the stable invariant subspace is not a graph over the states.")
    z11_inv = np.linalg.inv(z11)
    g_x = z21 @ z11_inv
    h_x = z11 @ np.linalg.solve(s11, t11) @ z11_inv
    return np.real(h_x), np.real(g_x)


def _second_order(
    jac: npt.NDArray[np.float64],
    hess: npt.NDArray[np.float64],
    h_x: npt.NDArray[np.float64],
    g_x: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
    *,
    n_x: int,
    n_y: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """The Schmitt-Grohe-Uribe second-order blocks.

    Returns:
        ``(h_xx, g_xx, h_ss, g_ss)`` with ``h_xx`` of shape ``(n_x, n_x**2)``
        and ``g_xx`` of shape ``(n_y, n_x**2)``, both acting on the
        row-major Kronecker square ``x kron x``.
    """
    n_eq = jac.shape[0]
    f_yn = jac[:, :n_y]
    f_y = jac[:, n_y : 2 * n_y]
    f_xn = jac[:, 2 * n_y : 2 * n_y + n_x]
    # dv/dx, v = (y', y, x', x)
    d_mat = np.vstack([g_x @ h_x, g_x, h_x, np.eye(n_x)])
    known = np.einsum("iab,aj,bk->ijk", hess, d_mat, d_mat).reshape(n_eq, n_x * n_x)
    kron_h = np.kron(h_x, h_x)
    eye_sq = np.eye(n_x * n_x)
    lhs_g = np.kron(f_yn, kron_h.T) + np.kron(f_y, eye_sq)
    lhs_h = np.kron(f_yn @ g_x + f_xn, eye_sq)
    lhs = np.hstack([lhs_g, lhs_h])
    rhs = -known.ravel()
    try:
        solution = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError as error:
        raise NumericalError("the second-order system is singular.") from error
    g_xx = solution[: n_y * n_x * n_x].reshape(n_y, n_x * n_x)
    h_xx = solution[n_y * n_x * n_x :].reshape(n_x, n_x * n_x)
    # risk correction: dv/deps at sigma = 0
    m_mat = np.vstack(
        [g_x @ eta, np.zeros((n_y, eta.shape[1])), eta, np.zeros((n_x, eta.shape[1]))]
    )
    trace_term = np.einsum("iab,ae,be->i", hess, m_mat, m_mat)
    eta_sq = (eta @ eta.T).ravel()
    rhs_ss = -(f_yn @ g_xx @ eta_sq + trace_term)
    lhs_ss = np.hstack([f_yn + f_y, f_yn @ g_x + f_xn])
    try:
        ss = np.linalg.solve(lhs_ss, rhs_ss)
    except np.linalg.LinAlgError as error:
        raise NumericalError("the risk-correction system is singular.") from error
    g_ss = ss[:n_y]
    h_ss = ss[n_y:]
    return h_xx, g_xx, h_ss, g_ss


def _discrete_lyapunov(
    transition: npt.NDArray[np.float64], noise: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """The stationary covariance ``P = T P T' + Q``."""
    try:
        return np.asarray(sla.solve_discrete_lyapunov(transition, noise), dtype=np.float64)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise NumericalError("the first-order transition has no stationary covariance.") from error





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


class _PerturbationModelSpecification(Protocol):
    """What a perturbation model must supply to be solved and estimated.

    Every method takes the structural parameter vector ``theta`` in its
    natural (constrained) space. The engine differentiates ``equations``
    numerically, so it must be smooth in its arguments near the steady
    state and must be written in the variables the approximation should
    be taken in -- write it in logs for a log-linear approximation.

    Attributes:
        parameter_names: One label per structural parameter.
        bounds: ``(d, 2)`` lower and upper bounds per parameter;
            ``-inf`` / ``inf`` for an unbounded side.
        n_states: Predetermined variables, exogenous processes included.
        n_controls: Non-predetermined variables.
    """

    parameter_names: Sequence[str]
    bounds: Sequence[tuple[float, float]]
    n_states: int
    n_controls: int

    def equations(
        self,
        theta: npt.NDArray[np.float64],
        y_next: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        x_next: npt.NDArray[np.float64],
        x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Residuals of ``E_t f(y', y, x', x) = 0``, ``n_states + n_controls`` of them."""
        ...

    def steady_state(
        self, theta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(x_ss, y_ss)`` solving the equations with expectations removed."""
        ...

    def shock_loading(self, theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """``(n_states, n_shocks)`` loading of standard-normal shocks on the states."""
        ...

    def observation(
        self, theta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """``(design, intercept, obs_cov)``: observables ``= design @ [x; y] + intercept + e``."""
        ...

    def log_prior(self, theta: npt.NDArray[np.float64]) -> float:
        """Log prior density in the constrained space; ``-inf`` outside support."""
        ...
