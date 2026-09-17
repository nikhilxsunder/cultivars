from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from .._core import _SQRT_2_OVER_PI, deterministic_columns
from ..exceptions import DimensionError, SpecificationError
from ._matrices import _psd_factor
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


def _simulate_perturbation(
    solution: _PerturbationSolution,
    n: int,
    *,
    design: npt.NDArray[np.float64],
    intercept: npt.NDArray[np.float64],
    obs_cov: npt.NDArray[np.float64],
    rng: np.random.Generator,
    burn: int = 0,
) -> npt.NDArray[np.float64]:
    """One path of the observables of a perturbation solution.

    Standard-normal structural shocks drive the pruned system from the
    steady state; the first ``burn`` periods are discarded; the
    observables read ``design @ [states; controls] + intercept`` plus
    Gaussian measurement noise of covariance ``obs_cov``.

    Args:
        solution: The perturbation solution.
        n: Observations kept.
        design: ``(p, n_x + n_y)`` observation loading over
            ``[states; controls]``.
        intercept: ``(p,)`` observation intercept.
        obs_cov: ``(p, p)`` measurement-noise covariance; zero is allowed.
        rng: Random generator.
        burn: Periods discarded from the start.

    Returns:
        An ``(n, p)`` array.

    Raises:
        SpecificationError: If the counts are not usable.

    Example:
        >>> import numpy as np
        >>> from cultivars._internals._solutions import _PerturbationSolution
        >>> sol = _PerturbationSolution(
        ...     h_x=np.array([[0.5]]), g_x=np.array([[2.0]]), eta=np.array([[1.0]]),
        ...     h_xx=np.zeros((1, 1)), g_xx=np.zeros((1, 1)), h_ss=np.zeros(1),
        ...     g_ss=np.zeros(1), x_ss=np.zeros(1), y_ss=np.ones(1), order=1,
        ... )
        >>> y = _simulate_perturbation(
        ...     sol, 2000, design=np.array([[0.0, 1.0]]), intercept=np.zeros(1),
        ...     obs_cov=np.zeros((1, 1)), rng=np.random.default_rng(0), burn=100,
        ... )
        >>> y.shape, bool(abs(y.mean() - 1.0) < 0.3)
        ((2000, 1), True)
    """
    if n < 1 or burn < 0:
        raise SpecificationError(f"n must be positive and burn non-negative; got {n}, {burn}.")
    shocks = rng.standard_normal((burn + n, solution.n_shocks))
    states, controls = _simulate_pruned(solution, shocks)
    latent = np.hstack([states[burn:], controls[burn:]])
    p = design.shape[0]
    noise = rng.standard_normal((n, p)) @ _psd_factor(obs_cov).T
    return np.asarray(latent @ design.T + intercept + noise, dtype=np.float64)


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


def _simulate_arma(
    n: int,
    *,
    ar: npt.NDArray[np.float64],
    ma: npt.NDArray[np.float64],
    sigma: float,
    rng: np.random.Generator,
    intercept: npt.NDArray[np.float64] | None = None,
    burn: int = 0,
) -> npt.NDArray[np.float64]:
    """One path of ``u_t = sum phi_i u_{t-i} + e_t + sum theta_j e_{t-j}`` plus an intercept path.

    Started from zeros and run ``burn`` extra periods first, which is how
    a fresh sample from the stationary law is produced; ``intercept`` is
    added to the kept part after the recursion, so it is a deterministic
    mean path rather than a regressor inside the autoregression.

    Args:
        n: Observations kept.
        ar: ``(p,)`` autoregressive coefficients (expanded seasonal
            products included).
        ma: ``(q,)`` moving-average coefficients of ``1 + theta_1 L + ...``.
        sigma: Innovation standard deviation.
        rng: Random generator.
        intercept: ``(n,)`` deterministic mean path added to the kept part.
        burn: Periods discarded from the start.

    Returns:
        The ``(n,)`` path.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = _simulate_arma(500, ar=np.array([0.9]), ma=np.zeros(0), sigma=1.0, rng=rng, burn=99)
        >>> bool(abs(np.corrcoef(y[1:], y[:-1])[0, 1] - 0.9) < 0.05)
        True
    """
    if n < 1:
        raise SpecificationError(f"n must be at least 1; got {n}.")
    if burn < 0:
        raise SpecificationError(f"burn must be non-negative; got {burn}.")
    p, q = ar.shape[0], ma.shape[0]
    total = n + burn
    shocks = sigma * rng.standard_normal(total)
    u = np.zeros(total)
    for t in range(total):
        value = shocks[t]
        for i in range(min(p, t)):
            value += ar[i] * u[t - 1 - i]
        for j in range(min(q, t)):
            value += ma[j] * shocks[t - 1 - j]
        u[t] = value
    out = u[burn:]
    if intercept is not None:
        out = out + intercept
    return np.asarray(out, dtype=np.float64)


def _integrate(
    w: npt.NDArray[np.float64], d: int, capital_d: int, s: int
) -> npt.NDArray[np.float64]:
    """Undo ``combined_difference`` from zero initial levels.

    Seasonal integration is undone first, then the non-seasonal, so the
    order matches the differencing that produced ``w``.

    Example:
        >>> _integrate(np.array([1.0, 1.0, 1.0]), 1, 0, 1)
        array([1., 2., 3.])
    """
    y = np.asarray(w, dtype=np.float64)
    for _ in range(capital_d):
        out = np.empty_like(y)
        for t in range(y.shape[0]):
            out[t] = y[t] + (out[t - s] if t >= s else 0.0)
        y = out
    for _ in range(d):
        y = np.cumsum(y)
    return y


def _simulate_conditional_variance(
    n: int,
    *,
    vol: str,
    omega: float,
    alpha: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    const: float,
    ar: npt.NDArray[np.float64],
    ma: npt.NDArray[np.float64],
    rng: np.random.Generator,
    burn: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """One path of an ARMA mean with a GARCH, GJR, or EGARCH variance.

    The variance recursions mirror the estimators' exactly: linear in past
    squared residuals and variances (with the sign-asymmetry term for
    GJR), or linear in the log variance driven by standardized residuals
    for EGARCH. Pre-sample terms start at the unconditional variance for
    the linear families and at ``exp(omega / (1 - sum beta))`` for EGARCH.

    Args:
        n: Observations kept.
        vol: ``"GARCH"``, ``"GJR"`` or ``"EGARCH"``.
        omega: Variance intercept.
        alpha: Coefficients on the shock magnitude.
        gamma: Asymmetry coefficients; empty when symmetric.
        beta: Persistence coefficients.
        const: Mean intercept.
        ar: Conditional-mean AR coefficients.
        ma: Conditional-mean MA coefficients.
        rng: Random generator.
        burn: Periods discarded from the start.

    Returns:
        ``(y, sigma2)``, each ``(n,)``.

    Raises:
        SpecificationError: If the family is unknown or the counts are
            not usable.
    """
    if vol not in ("GARCH", "GJR", "EGARCH"):
        raise SpecificationError(f"vol must be 'GARCH', 'GJR', or 'EGARCH'; got {vol!r}.")
    if n < 1 or burn < 0:
        raise SpecificationError(f"n must be positive and burn non-negative; got {n}, {burn}.")
    total = n + burn
    p, o, q = alpha.size, gamma.size, beta.size
    persistence = float(alpha.sum() + beta.sum() + (0.5 * gamma.sum() if vol == "GJR" else 0.0))
    if vol == "EGARCH":
        start = float(np.exp(omega / max(1.0 - beta.sum(), 1e-6)))
    else:
        start = omega / max(1.0 - persistence, 1e-6) if persistence < 1.0 else omega
    z = rng.standard_normal(total)
    sigma2 = np.empty(total)
    resid = np.zeros(total)
    y = np.zeros(total)
    log_s2 = np.empty(total)
    for t in range(total):
        if vol == "EGARCH":
            s = omega
            for i in range(p):
                s += alpha[i] * ((abs(z[t - 1 - i]) - _SQRT_2_OVER_PI) if t - 1 - i >= 0 else 0.0)
            for k in range(o):
                s += gamma[k] * (z[t - 1 - k] if t - 1 - k >= 0 else 0.0)
            for j in range(q):
                s += beta[j] * (log_s2[t - 1 - j] if t - 1 - j >= 0 else np.log(start))
            log_s2[t] = s
            sigma2[t] = float(np.exp(s))
        else:
            s = omega
            for i in range(p):
                s += alpha[i] * (resid[t - 1 - i] ** 2 if t - 1 - i >= 0 else start)
            for k in range(o):
                if t - 1 - k >= 0:
                    s += gamma[k] * resid[t - 1 - k] ** 2 * float(resid[t - 1 - k] < 0.0)
                else:
                    s += gamma[k] * start * 0.5
            for j in range(q):
                s += beta[j] * (sigma2[t - 1 - j] if t - 1 - j >= 0 else start)
            sigma2[t] = s
        resid[t] = float(np.sqrt(sigma2[t])) * z[t]
        value = const + resid[t]
        for i in range(min(ar.size, t)):
            value += ar[i] * y[t - 1 - i]
        for j in range(min(ma.size, t)):
            value += ma[j] * resid[t - 1 - j]
        y[t] = value
    return np.asarray(y[burn:], dtype=np.float64), np.asarray(sigma2[burn:], dtype=np.float64)


def _simulate_markov_switching(
    n: int,
    *,
    transition: npt.NDArray[np.float64],
    intercepts: npt.NDArray[np.float64],
    ar_params: npt.NDArray[np.float64],
    variances: npt.NDArray[np.float64],
    rng: np.random.Generator,
    burn: int = 0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """One path of a Markov-switching autoregression and the regimes that generated it.

    The regime chain starts from its ergodic distribution and the series
    from zeros; ``burn`` periods are discarded.

    Args:
        n: Observations kept.
        transition: ``(K, K)`` matrix with ``transition[i, j] = P(s_t = j |
            s_{t-1} = i)``.
        intercepts: ``(K,)`` regime intercepts.
        ar_params: ``(K, p)`` regime autoregressive coefficients.
        variances: ``(K,)`` regime innovation variances.
        rng: Random generator.
        burn: Periods discarded from the start.

    Returns:
        ``(y, regimes)`` of shapes ``(n,)`` and ``(n,)``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y, s = _simulate_markov_switching(
        ...     200,
        ...     transition=np.array([[0.9, 0.1], [0.2, 0.8]]),
        ...     intercepts=np.array([-1.0, 1.0]),
        ...     ar_params=np.zeros((2, 0)),
        ...     variances=np.ones(2),
        ...     rng=rng,
        ...     burn=50,
        ... )
        >>> bool(y[s == 1].mean() > y[s == 0].mean())
        True
    """
    if n < 1 or burn < 0:
        raise SpecificationError(f"n must be positive and burn non-negative; got {n}, {burn}.")
    n_regimes, p = ar_params.shape
    if transition.shape != (n_regimes, n_regimes):
        raise DimensionError(
            f"transition must be ({n_regimes}, {n_regimes}); got {transition.shape}."
        )
    total = n + burn
    values, vectors = np.linalg.eig(transition.T)
    ergodic = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    ergodic = np.abs(ergodic) / np.abs(ergodic).sum()
    regimes = np.empty(total, dtype=np.int64)
    regimes[0] = rng.choice(n_regimes, p=ergodic)
    for t in range(1, total):
        regimes[t] = rng.choice(n_regimes, p=transition[regimes[t - 1]])
    y = np.zeros(total)
    shocks = rng.standard_normal(total)
    for t in range(total):
        k = regimes[t]
        value = intercepts[k] + float(np.sqrt(variances[k])) * shocks[t]
        for i in range(min(p, t)):
            value += ar_params[k, i] * y[t - 1 - i]
        y[t] = value
    return np.asarray(y[burn:], dtype=np.float64), regimes[burn:]


def _simulate_two_regime(
    n: int,
    *,
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
    order: int,
    delay: int,
    sigma: float,
    weight: Callable[[float], float],
    rng: np.random.Generator,
    burn: int = 0,
) -> npt.NDArray[np.float64]:
    """One path of a self-exciting two-regime autoregression, hard or smooth.

    ``y_t = (1 - w_t) (c_L + phi_L y) + w_t (c_U + phi_U y) + sigma e_t``
    with ``w_t = weight(y_{t-d})``: the indicator ``y_{t-d} > threshold``
    for a threshold model, a logistic or exponential function of it for a
    smooth-transition one.

    Args:
        n: Observations kept.
        lower: ``(order + 1,)`` lower-regime coefficients, intercept first.
        upper: ``(order + 1,)`` upper-regime coefficients, intercept first.
        order: Autoregressive order.
        delay: Delay of the transition variable, at least one.
        sigma: Innovation standard deviation.
        weight: Upper-regime weight as a function of ``y_{t-d}``.
        rng: Random generator.
        burn: Periods discarded from the start.

    Returns:
        The ``(n,)`` path.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = _simulate_two_regime(
        ...     300, lower=np.array([0.5, 0.3]), upper=np.array([-0.5, 0.3]), order=1, delay=1,
        ...     sigma=1.0, weight=lambda z: float(z > 0.0), rng=rng, burn=50
        ... )
        >>> y.shape
        (300,)
    """
    if n < 1 or burn < 0:
        raise SpecificationError(f"n must be positive and burn non-negative; got {n}, {burn}.")
    if delay < 1:
        raise SpecificationError(f"delay must be at least 1; got {delay}.")
    total = n + burn
    y = np.zeros(total)
    shocks = sigma * rng.standard_normal(total)
    for t in range(total):
        w = weight(float(y[t - delay])) if t >= delay else 0.5
        low, up = lower[0], upper[0]
        for i in range(min(order, t)):
            low += lower[i + 1] * y[t - 1 - i]
            up += upper[i + 1] * y[t - 1 - i]
        y[t] = (1.0 - w) * low + w * up + shocks[t]
    return np.asarray(y[burn:], dtype=np.float64)
