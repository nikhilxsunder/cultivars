from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._core import deterministic_columns
from ..exceptions import DimensionError, SpecificationError
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


def _simulate_vector_autoregression(
    beta: npt.NDArray[np.float64],
    noise: npt.NDArray[np.float64],
    *,
    order: int,
    trend: str,
    presample: npt.NDArray[np.float64],
    start: int,
) -> npt.NDArray[np.float64]:
    """Deterministic pass of a VAR through an innovation sequence.

    The recursion behind every replication and predictive path in the
    Bayesian VAR family: one ``(w, k)`` coefficient matrix in the design's
    column order -- deterministic block first, then the lag blocks -- a
    presample to start from, and the innovations already drawn.

    Args:
        beta: ``(w, k)`` coefficients, ``w = n_det + k * order``.
        noise: ``(n, k)`` innovations, one row per simulated period.
        order: Autoregressive order.
        trend: Deterministic specification, ``"n"``, ``"c"`` or ``"ct"``.
        presample: ``(order, k)`` initial values, oldest first; ignored
            when the order is zero.
        start: Time index of the first simulated row, in the convention of
            :func:`~cultivars._core.deterministic_columns`, so that a
            replication of the effective sample carries the trend values
            the sample had.

    Returns:
        The ``(n, k)`` simulated rows.

    Raises:
        DimensionError: If the coefficient width does not match the
            deterministic block and the lag blocks.
        SpecificationError: If the trend is unknown.
    """
    n, k = noise.shape
    if trend not in ("n", "c", "ct"):
        raise SpecificationError(f"trend must be 'n', 'c', or 'ct'; got {trend!r}.")
    det = deterministic_columns(trend, n, start=start)
    offset = det.shape[1]
    if beta.shape != (offset + k * order, k):
        raise DimensionError(
            f"beta has shape {beta.shape}; expected ({offset + k * order}, {k}) for "
            f"trend {trend!r}, order {order}, and {k} variables."
        )
    stack = [beta[offset + lag * k : offset + (lag + 1) * k].T for lag in range(order)]
    history = list(np.asarray(presample, dtype=np.float64)[::-1]) if order else []
    if len(history) != order:
        raise DimensionError(f"presample has {len(history)} rows; the order is {order}.")
    out = np.empty((n, k))
    level = det @ beta[:offset] + noise
    for t in range(n):
        value = level[t]
        for lag in range(order):
            value = value + stack[lag] @ history[lag]
        out[t] = value
        if order:
            history = [value, *history[:-1]]
    return out


def _simulate_stochastic_volatility(
    n: int,
    *,
    mu: float,
    phi: float,
    sigma2: float,
    mean: float,
    rng: np.random.Generator,
    nu: float | None = None,
    rho: float = 0.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """One path of the log-AR(1) stochastic-volatility model from its stationary start.

    ``h_1`` is drawn from the stationary distribution of the log variance,
    then ``h_{t+1} = mu + phi (h_t - mu) + sigma eta_t`` and ``y_t = mean +
    exp(h_t / 2) eps_t``. Heavy tails make ``eps_t`` Student-t with ``nu``
    degrees of freedom (unscaled, as the model states it); leverage
    correlates ``eps_t`` with the innovation that moves ``h_{t+1}``.

    Args:
        n: Observations to simulate.
        mu: Unconditional mean of the log variance.
        phi: Persistence, strictly inside ``(-1, 1)``.
        sigma2: Innovation variance of the log variance, positive.
        mean: The observation mean.
        rng: Random generator.
        nu: Degrees of freedom of the observation noise, or ``None`` for
            Gaussian.
        rho: Correlation between ``eps_t`` and ``eta_t``.

    Returns:
        ``(y, h)``, each ``(n,)``.

    Raises:
        SpecificationError: If the parameters leave the stationary region
            or combine heavy tails with leverage.
    """
    if not -1.0 < phi < 1.0 or sigma2 <= 0.0:
        raise SpecificationError(
            f"a stationary log variance needs |phi| < 1 and sigma2 > 0; got phi={phi}, "
            f"sigma2={sigma2}."
        )
    if nu is not None and rho != 0.0:
        raise SpecificationError("heavy tails and leverage are not offered together.")
    if nu is not None and nu <= 0.0:
        raise SpecificationError(f"nu must be positive; got {nu}.")
    if not -1.0 < rho < 1.0:
        raise SpecificationError(f"rho must lie strictly inside (-1, 1); got {rho}.")
    sigma = float(np.sqrt(sigma2))
    eps = rng.standard_normal(n)
    if nu is not None:
        eps = eps / np.sqrt(rng.gamma(0.5 * nu, 2.0 / nu, size=n))
    eta = rho * eps + float(np.sqrt(1.0 - rho**2)) * rng.standard_normal(n)
    h = np.empty(n)
    h[0] = mu + float(np.sqrt(sigma2 / (1.0 - phi**2))) * rng.standard_normal()
    for t in range(1, n):
        h[t] = mu + phi * (h[t - 1] - mu) + sigma * eta[t - 1]
    y = mean + np.exp(0.5 * h) * eps
    return np.asarray(y, dtype=np.float64), h
