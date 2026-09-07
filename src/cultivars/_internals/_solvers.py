# filepath: /src/cultivars/_internals/_solvers.py
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

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.special import multigammaln

from .._core import link_matrix
from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._covariances import _PosteriorCovariance
from ._levels import _ConditionalLevels
from ._objectives import _Objective
from ._posteriors import _ConjugatePosterior
from ._priors import _Prior, _PriorContext
from ._solvers import _LinearGaussianStateSpace, _NonlinearStateSpace
from ._solutions import _PerturbationSolution


def _solve[P](objective: _Objective[P]) -> tuple[P, float]:
    """Minimize an objective from every starting point and keep the best.

    Args:
        objective: The surface to minimize. Its ``method`` and ``options``
            select the algorithm; its ``starts`` supply the initial points.

    Returns:
        A tuple ``(parameters, criterion)`` where ``criterion`` is the minimized
        value of :meth:`_Objective.__call__` at the winning start.

    Raises:
        NumericalError: If ``starts`` yields no points at all.
    """
    best_x: npt.NDArray[np.float64] | None = None
    best_f = np.inf
    for theta0 in objective.starts():
        result = minimize(objective, theta0, method=objective.method, options=objective.options)
        if float(result.fun) < best_f:
            best_f = float(result.fun)
            best_x = np.asarray(result.x, dtype=np.float64)
    if best_x is None:
        raise NumericalError("objective supplied no starting points.")
    return objective.unpack(best_x), best_f


def _maximize_likelihood[P](objective: _Objective[P]) -> tuple[P, float]:
    """Solve a negative-log-likelihood objective, returning the log-likelihood.

    A thin sign-flip over :func:`_solve`, kept as a separate name so that no
    call site has to remember which surfaces are negated and which are not.

    Args:
        objective: A surface whose criterion is a negative log-likelihood.

    Returns:
        A tuple ``(parameters, llf)`` with the maximized log-likelihood.
    """
    parameters, criterion = _solve(objective)
    return parameters, -criterion


def solve_global(
    units: Sequence[_ConditionalLevels],
    *,
    weights: npt.NDArray[np.float64],
    unit_of_column: Sequence[int],
    variable_of_column: Sequence[int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Stack conditional unit equations into one closed global autoregression.

    Each unit satisfies ``A_i0 z_it = d_it + sum_l A_il z_{i,t-l} + u_it`` with
    ``A_i0 = [I, -Lambda_i0]`` and ``A_il = [Phi_il, Lambda_il]``. Substituting
    ``z_it = W_i x_t`` writes every unit against the same global vector, and
    stacking the results gives ``G_0 x_t = d_t + sum_l G_l x_{t-l} + u_t``,
    which is square because the units' own blocks partition the global vector
    exactly once.

    The inversion of ``G_0`` is the moment the system closes. Individually no
    unit has a law of motion for its foreign variables; jointly they do, because
    one unit's foreign aggregate is a weighted sum of other units' domestic
    variables and those units have equations. Nothing is estimated here -- the
    result is an algebraic rearrangement of coefficients that were already
    fitted, and it reproduces every unit equation exactly rather than
    approximately.

    Args:
        units: One record per unit, in global column order.
        weights: A validated cross-unit weight matrix.
        unit_of_column: Owning unit index for each global column.
        variable_of_column: Variable identity for each global column.

    Returns:
        The contemporaneous matrix ``G_0``, the ``(p, k, k)`` stack of global
        autoregressive matrices ``G_0^{-1} G_l``, and the global deterministic
        coefficients.

    Raises:
        DimensionError: If the units do not partition the global columns.
        NumericalError: If ``G_0`` is singular, which means the contemporaneous
            linkage has no solution -- typically a weight matrix that makes two
            units' foreign aggregates identical.
    """
    width = len(unit_of_column)
    owners = tuple(int(u) for u in unit_of_column)
    covered = sorted(c for unit in range(len(units)) for c in range(width) if owners[c] == unit)
    if covered != list(range(width)):
        raise DimensionError(
            "the units must partition the global columns exactly once; "
            f"unit_of_column covers {covered} of {list(range(width))}."
        )
    order = max(unit.order for unit in units)
    depth = max(order, max(unit.exog_lags.shape[0] for unit in units))
    contemporaneous: list[npt.NDArray[np.float64]] = []
    lagged: list[list[npt.NDArray[np.float64]]] = [[] for _ in range(depth)]
    drift_rows = max(unit.deterministic.shape[0] for unit in units)
    drift = np.zeros((drift_rows, width), dtype=np.float64)
    cursor = 0
    for index, unit in enumerate(units):
        selector = link_matrix(
            index,
            weights=weights,
            unit_of_column=unit_of_column,
            variable_of_column=variable_of_column,
        )
        size = unit.k_endog
        contemporaneous.append(np.hstack([np.eye(size, dtype=np.float64), -unit.impact]) @ selector)
        for lag in range(depth):
            own = unit.phi[lag] if lag < unit.order else np.zeros((size, size))
            foreign = (
                unit.exog_lags[lag]
                if lag < unit.exog_lags.shape[0]
                else np.zeros((size, unit.k_exog))
            )
            lagged[lag].append(np.hstack([own, foreign]) @ selector)
        rows = unit.deterministic.shape[0]
        drift[:rows, cursor : cursor + size] = unit.deterministic
        cursor += size
    g_zero = np.vstack(contemporaneous)
    if abs(float(np.linalg.slogdet(g_zero)[0])) < 0.5:
        raise NumericalError(
            "the contemporaneous linkage matrix is singular, so the global system has no "
            "solution. The usual cause is a weight matrix that gives two units identical "
            "foreign aggregates, which leaves their contemporaneous equations linearly "
            "dependent."
        )
    blocks = np.stack([np.linalg.solve(g_zero, np.vstack(lagged[lag])) for lag in range(depth)])
    return g_zero, blocks, np.linalg.solve(g_zero, drift.T).T


def posterior_coefficients(
    target: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    prior: _Prior,
    context: _PriorContext,
) -> _PosteriorCovariance:
    """Posterior mean of the coefficient matrix, equation by equation.

    Each equation solves ``(X'X / s_i^2 + V_i^-1) b_i = X'y_i / s_i^2 +
    V_i^-1 m_i`` with ``V_i`` the prior variances for that equation and ``m_i``
    its prior mean. Running equation by equation is not an approximation and
    not a shortcut: once the cross-equation weight differs from one, the prior
    variance stops factoring as a Kronecker product, the Normal-inverse-Wishart
    conjugacy is gone, and Litterman's original per-equation procedure is the
    correct one.

    Any dummy rows the prior contributes are stacked under the sample first,
    which is how a restriction on a *sum* of coefficients enters a problem
    whose other restrictions are diagonal variances.

    Args:
        target: The ``(nobs, k)`` block being explained.
        design: The ``(nobs, width)`` regressor matrix.
        prior: The prior, possibly a composition.
        context: What the prior needs to know about the sample.

    Returns:
        A :class:`_PosteriorCovariance` carrying the posterior mean, the
        per-equation covariance, and the effective parameter count. The last is
        not decoration: shrinkage means the model spends fewer parameters than
        the design has columns, and an information criterion that charges for
        the nominal width penalizes a shrunk model for freedom it never used.

    Raises:
        DimensionError: If the design does not match the context's width.
    """
    if design.shape[1] != context.width:
        raise DimensionError(
            f"design has {design.shape[1]} columns but the prior context describes "
            f"{context.width}. The two must agree on the column order or the prior "
            "lands on the wrong coefficients."
        )
    dummy_target, dummy_design = prior.dummy_observations(context)
    full_target = np.vstack([target, dummy_target]) if dummy_target.shape[0] else target
    full_design = np.vstack([design, dummy_design]) if dummy_design.shape[0] else design
    mean = prior.coefficient_mean(context)
    variance = prior.coefficient_variance(context)
    cross = full_design.T @ full_design
    moment = full_design.T @ full_target
    out = np.empty((context.width, context.k_endog), dtype=np.float64)
    blocks = np.empty((context.k_endog, context.width, context.width), dtype=np.float64)
    effective = 0.0
    for index in range(context.k_endog):
        scale = float(context.scales[index]) ** 2
        column = variance[:, index]
        precision = np.where(np.isfinite(column), 1.0 / np.maximum(column, 1e-300), 0.0)
        left = cross / scale + np.diag(precision)
        right = moment[:, index] / scale + precision * mean[:, index]
        out[:, index] = np.linalg.solve(left, right)
        posterior = np.linalg.inv(left)
        blocks[index] = posterior
        effective += float(np.trace(posterior @ cross)) / scale
    return _PosteriorCovariance(coefficients=out, blocks=blocks, effective_parameters=effective)


def _conjugate_posterior(
    target: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    *,
    omega: npt.NDArray[np.float64],
    mean: npt.NDArray[np.float64],
    scale0: npt.NDArray[np.float64],
    df0: float,
) -> _ConjugatePosterior:
    """One exact Normal-inverse-Wishart update, with its marginal likelihood.

    The conjugate counterpart of :func:`posterior_coefficients`, and the two
    agree exactly at their overlap: with a Kronecker prior the per-equation
    solve and this joint update produce the same posterior mean, and the
    difference is what each can report -- the per-equation path keeps
    Litterman's cross-equation weight, this path keeps the joint posterior
    over ``(B, Sigma)`` and the closed-form marginal likelihood that
    hierarchical shrinkage optimizes.

    The prior is ``Sigma ~ IW(scale0, df0)`` and ``B | Sigma`` matrix normal
    with mean ``mean`` and covariance ``Sigma x diag(omega)``; the update is
    the standard one, and the marginal likelihood is the matrix-variate-t
    normalizing-constant ratio, exact rather than simulated.

    Args:
        target: The ``(n, k)`` rows to update on -- possibly zero rows, in
            which case the posterior is the prior and ``log_ml`` is zero.
        design: The ``(n, width)`` regressors for those rows.
        omega: The ``(width,)`` shared prior variance profile, positive and
            finite.
        mean: The ``(width, k)`` prior mean of the coefficient matrix.
        scale0: The ``(k, k)`` inverse-Wishart prior scale.
        df0: Inverse-Wishart prior degrees of freedom, above ``k - 1``.

    Returns:
        The :class:`_ConjugatePosterior`.

    Raises:
        NumericalError: If a posterior scale matrix is not positive definite,
            which means the prior and the rows jointly degenerate.
    """
    n, k = target.shape
    precision = 1.0 / omega
    key = design.T @ design + np.diag(precision)
    key = 0.5 * (key + key.T)
    coefficients = np.linalg.solve(key, precision[:, None] * mean + design.T @ target)
    scale = (
        scale0
        + target.T @ target
        + mean.T @ (precision[:, None] * mean)
        - coefficients.T @ key @ coefficients
    )
    scale = 0.5 * (scale + scale.T)
    df = df0 + float(n)
    sign_key, logdet_key = np.linalg.slogdet(key)
    sign_zero, logdet_zero = np.linalg.slogdet(scale0)
    sign_scale, logdet_scale = np.linalg.slogdet(scale)
    if min(sign_key, sign_zero, sign_scale) <= 0:
        raise NumericalError(
            "a Normal-inverse-Wishart update produced a non-positive-definite "
            "scale; the prior and the sample jointly degenerate."
        )
    log_ml = (
        -0.5 * n * k * np.log(np.pi)
        + 0.5 * k * (-float(logdet_key) - float(np.log(omega).sum()))
        + 0.5 * df0 * float(logdet_zero)
        - 0.5 * df * float(logdet_scale)
        + float(multigammaln(0.5 * df, k))
        - float(multigammaln(0.5 * df0, k))
    )
    return _ConjugatePosterior(
        coefficients=coefficients,
        row_precision=key,
        scale=scale,
        df=df,
        log_ml=float(log_ml),
    )


def _fista_penalized(
    gram: npt.NDArray[np.float64],
    moment: npt.NDArray[np.float64],
    *,
    lam: float,
    weights: npt.NDArray[np.float64],
    groups: tuple[npt.NDArray[np.intp], ...] | None = None,
    start: npt.NDArray[np.float64] | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> npt.NDArray[np.float64]:
    """Penalized multivariate least squares by accelerated proximal gradient.

    Minimizes ``sum_i [ b_i' G b_i / 2 - c_i' b_i ] + lam * P(B)`` over the
    ``(w, k)`` coefficient matrix, where ``P`` is a weighted elementwise
    L1 penalty -- or, when ``groups`` is given, a group penalty whose blocks
    are rows shared by every equation, shrunk by Frobenius norm. FISTA with
    adaptive restart, fully vectorized across coordinates and equations: one
    iteration is one Gram product, which is what makes a cross-validation
    path over hundreds of refits affordable in NumPy. A weight of zero
    exempts a row from the penalty, which is how deterministic terms stay
    unshrunk.

    Args:
        gram: ``(w, w)`` Gram matrix ``X'X / n`` of the (standardized)
            design.
        moment: ``(w, k)`` cross-moment ``X'Y / n``.
        lam: Penalty level, non-negative.
        weights: ``(w,)`` per-row or ``(w, k)`` per-coefficient penalty
            weights; zeros are free. Per-coefficient weights are what the
            adaptive and local-linear-approximation reweightings produce.
        groups: Row-index blocks for the group penalty; ``None`` for the
            elementwise penalty. Blocks must cover only penalized rows.
        start: Warm start, ``(w, k)``; zeros when omitted.
        max_iter: Iteration cap.
        tol: Relative change below which the iteration stops.

    Returns:
        The ``(w, k)`` minimizer.

    Raises:
        NumericalError: If the Gram matrix has no positive curvature.
    """
    width, k = moment.shape
    lipschitz = float(np.linalg.eigvalsh(gram)[-1])
    if not lipschitz > 0.0:
        raise NumericalError("the Gram matrix has no positive curvature.")
    beta = np.zeros((width, k)) if start is None else start.copy()
    point = beta.copy()
    momentum = 1.0
    step = 1.0 / lipschitz
    for _ in range(max_iter):
        gradient = gram @ point - moment
        candidate = point - step * gradient
        if groups is None:
            threshold = lam * step * (weights if weights.ndim == 2 else weights[:, None])
            updated = np.sign(candidate) * np.maximum(np.abs(candidate) - threshold, 0.0)
        else:
            updated = candidate.copy()
            for block in groups:
                norm = float(np.sqrt(np.sum(candidate[block] ** 2)))
                cut = lam * step * np.sqrt(block.size * k)
                updated[block] = (
                    candidate[block] * max(0.0, 1.0 - cut / norm) if norm > 0.0 else 0.0
                )
        momentum_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        accel = updated + ((momentum - 1.0) / momentum_next) * (updated - beta)
        if float(np.sum((updated - beta) * (point - updated))) > 0.0:
            accel = updated
            momentum_next = 1.0
        shift = float(np.abs(updated - beta).max())
        scale_ref = max(float(np.abs(updated).max()), 1.0)
        beta, point, momentum = updated, accel, momentum_next
        if shift <= tol * scale_ref:
            break
    return beta


def _spectral_factor(
    density: npt.NDArray[np.complex128],
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.float64]]:
    """Canonical spectral factorization by Wilson's (1972) algorithm.

    Given a Hermitian positive-definite spectral density sampled on the
    *full* circle ``omega_m = 2 pi m / M``, finds the minimum-phase factor
    ``psi`` with ``2 pi S(omega) = psi(omega) psi(omega)*`` and only
    non-negative-lag Fourier content, and reads off the innovation
    representation: transfer ``H = psi psi_0**(-1)`` and covariance
    ``Sigma = psi_0 psi_0*``. This is what turns a *marginal* spectral
    density -- a sub-block of a larger system's, which is generally VARMA
    and matches no finite lag stack -- back into the (transfer, covariance)
    pair Geweke's causality formulas need, and it is the standard route
    (Dhamala-Rangarajan-Ding 2008). The constant-unitary ambiguity of the
    factorization cancels in ``H`` and ``Sigma``, so no triangularization
    of ``psi_0`` is enforced.

    Args:
        density: ``(M, k, k)`` spectral density on the full circle, with
            the ``1 / (2 pi)`` normalization of :func:`spectral_matrix`.
        max_iter: Iteration cap.
        tol: Relative factorization error below which iteration stops.

    Returns:
        ``(transfer, sigma)``: the ``(M, k, k)`` causal transfer function
        and the ``(k, k)`` innovation covariance.

    Raises:
        NumericalError: If the iteration fails to converge, which for a
            smooth stable-model density means the input is degenerate.
    """
    target = np.asarray(density, dtype=np.complex128) * (2.0 * np.pi)
    m, k = int(target.shape[0]), int(target.shape[-1])
    half = m // 2
    gamma0 = np.real(target.mean(axis=0))
    try:
        psi = np.tile(np.linalg.cholesky(gamma0).astype(np.complex128), (m, 1, 1))
    except np.linalg.LinAlgError as error:
        raise NumericalError(
            "the spectral density's mean is not positive definite; there is nothing to factorize."
        ) from error

    def causal_part(values: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        coefficients = np.fft.ifft(values, axis=0)
        coefficients[0] *= 0.5
        coefficients[half:] = 0.0
        return np.asarray(np.fft.fft(coefficients, axis=0), dtype=np.complex128)

    identity = np.eye(k, dtype=np.complex128)
    for _ in range(max_iter):
        inverse = np.linalg.inv(psi)
        inner = inverse @ target @ np.conj(np.swapaxes(inverse, -1, -2)) + identity
        updated = psi @ causal_part(inner)
        gap = float(np.abs(updated - psi).max())
        scale = float(np.abs(updated).max())
        psi = updated
        if gap <= tol * max(scale, 1.0):
            break
    else:
        raise NumericalError(
            "Wilson spectral factorization failed to converge; the density "
            "is degenerate or too close to singular at some frequency."
        )
    residual_gap = float(np.abs(psi @ np.conj(np.swapaxes(psi, -1, -2)) - target).max())
    if residual_gap > 1e-6 * max(float(np.abs(target).max()), 1.0):
        raise NumericalError(
            "Wilson spectral factorization converged to a non-factor; the density is degenerate."
        )
    psi0 = np.asarray(np.fft.ifft(psi, axis=0)[0], dtype=np.complex128)
    transfer = psi @ np.linalg.inv(psi0)
    sigma = np.real(psi0 @ np.conj(psi0.T))
    return transfer, sigma


def _linear_state_space(
    solution: _PerturbationSolution,
    design: npt.NDArray[np.float64],
    intercept: npt.NDArray[np.float64],
    obs_cov: npt.NDArray[np.float64],
) -> _LinearGaussianStateSpace:
    """The first-order solution as an exact linear-Gaussian state space.

    The state is the deviation ``x - x_ss``; observables are ``design @
    [x; y] + intercept + measurement error``.
    """
    n_x = solution.n_states
    z_full = design @ np.vstack([np.eye(n_x), solution.g_x])
    d_full = design @ np.concatenate([solution.x_ss, solution.y_ss]) + intercept
    noise = solution.eta @ solution.eta.T
    return _LinearGaussianStateSpace(
        z_full,
        obs_cov,
        solution.h_x,
        np.eye(n_x),
        noise,
        obs_intercept=d_full,
        initial_state=np.zeros(n_x),
        initial_state_cov=_discrete_lyapunov(solution.h_x, noise),
    )


def _pruned_state_space(
    solution: _PerturbationSolution,
    design: npt.NDArray[np.float64],
    intercept: npt.NDArray[np.float64],
    obs_cov: npt.NDArray[np.float64],
) -> _NonlinearStateSpace:
    """The pruned second-order solution as an additive-Gaussian nonlinear state space.

    The state is ``(x_f, x_s)``: the first-order deviation and the
    second-order deviation. Noise enters only ``x_f``, so the state
    covariance is singular and the form is additive-Gaussian, which
    every filter on the substrate accepts.
    """
    n_x = solution.n_states
    h_x, h_xx, h_ss = solution.h_x, solution.h_xx, solution.h_ss
    g_x, g_xx, g_ss = solution.g_x, solution.g_xx, solution.g_ss
    x_ss, y_ss = solution.x_ss, solution.y_ss

    def kron_square(block: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.einsum("nj,nk->njk", block, block).reshape(block.shape[0], n_x * n_x)

    def transition(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        block = np.asarray(states, dtype=np.float64)
        x_f = block[:, :n_x]
        x_s = block[:, n_x:]
        next_f = x_f @ h_x.T
        next_s = x_s @ h_x.T + 0.5 * kron_square(x_f) @ h_xx.T + 0.5 * h_ss[None, :]
        return np.hstack([next_f, next_s])

    def observation(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        block = np.asarray(states, dtype=np.float64)
        x_f = block[:, :n_x]
        x_s = block[:, n_x:]
        x_dev = x_f + x_s
        y_dev = x_dev @ g_x.T + 0.5 * kron_square(x_f) @ g_xx.T + 0.5 * g_ss[None, :]
        full = np.hstack([x_dev + x_ss[None, :], y_dev + y_ss[None, :]])
        return full @ design.T + intercept[None, :]

    noise = solution.eta @ solution.eta.T
    state_cov = np.zeros((2 * n_x, 2 * n_x))
    state_cov[:n_x, :n_x] = noise
    p_first = _discrete_lyapunov(h_x, noise)
    mean_second = np.linalg.solve(np.eye(n_x) - h_x, 0.5 * (h_xx @ p_first.ravel() + h_ss))
    initial_state = np.concatenate([np.zeros(n_x), mean_second])
    initial_cov = np.zeros((2 * n_x, 2 * n_x))
    initial_cov[:n_x, :n_x] = p_first
    return _NonlinearStateSpace(
        transition,
        observation,
        state_cov=state_cov,
        obs_cov=obs_cov,
        initial_state=initial_state,
        initial_state_cov=initial_cov,
    )


def _solve_perturbation(
    equations: _Residuals,
    x_ss: npt.NDArray[np.float64],
    y_ss: npt.NDArray[np.float64],
    eta: npt.NDArray[np.float64],
    *,
    order: int,
    residual_tolerance: float = 1e-6,
) -> _PerturbationSolution:
    """Solve a model to first or second order around its steady state.

    Args:
        equations: The equilibrium conditions ``(y', y, x', x) ->
            residuals``, ``n_x + n_y`` of them.
        x_ss: Steady-state states ``(n_x,)``.
        y_ss: Steady-state controls ``(n_y,)``.
        eta: Shock loading ``(n_x, n_eps)``.
        order: ``1`` or ``2``.
        residual_tolerance: Largest steady-state residual tolerated.

    Returns:
        The :class:`_PerturbationSolution`.

    Raises:
        SpecificationError: If the order is not 1 or 2, or the equation
            count does not match the variable count.
        DimensionError: If ``eta`` is not ``(n_x, n_eps)``.
        NumericalError: If the steady state does not solve the equations
            or the solution fails.
    """
    if order not in (1, 2):
        raise SpecificationError(f"order must be 1 or 2; got {order}.")
    x_ss = np.asarray(x_ss, dtype=np.float64).ravel()
    y_ss = np.asarray(y_ss, dtype=np.float64).ravel()
    eta = np.asarray(eta, dtype=np.float64)
    n_x, n_y = x_ss.shape[0], y_ss.shape[0]
    if eta.ndim != 2 or eta.shape[0] != n_x:
        raise DimensionError(f"eta must be ({n_x}, n_eps); got shape {eta.shape}.")
    point = _stack_point(y_ss, y_ss, x_ss, x_ss)

    def stacked(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(
            equations(v[:n_y], v[n_y : 2 * n_y], v[2 * n_y : 2 * n_y + n_x], v[2 * n_y + n_x :]),
            dtype=np.float64,
        ).ravel()

    residual = stacked(point)
    if residual.shape[0] != n_x + n_y:
        raise SpecificationError(
            f"the model has {n_x} states and {n_y} controls but returns "
            f"{residual.shape[0]} equations; need {n_x + n_y}."
        )
    if not np.all(np.isfinite(residual)) or float(np.abs(residual).max()) > residual_tolerance:
        raise NumericalError(
            "the supplied steady state does not solve the equilibrium conditions "
            f"(largest residual {float(np.abs(residual).max()):.3g})."
        )
    jac = _numerical_jacobian(stacked, point)
    f_yn = jac[:, :n_y]
    f_y = jac[:, n_y : 2 * n_y]
    f_xn = jac[:, 2 * n_y : 2 * n_y + n_x]
    f_x = jac[:, 2 * n_y + n_x :]
    h_x, g_x = _first_order(f_yn, f_y, f_xn, f_x)
    if order == 1:
        return _PerturbationSolution(
            h_x=h_x,
            g_x=g_x,
            h_xx=np.zeros((n_x, n_x * n_x)),
            g_xx=np.zeros((n_y, n_x * n_x)),
            h_ss=np.zeros(n_x),
            g_ss=np.zeros(n_y),
            eta=eta,
            x_ss=x_ss,
            y_ss=y_ss,
            order=1,
        )
    hess = _numerical_hessian(stacked, point)
    h_xx, g_xx, h_ss, g_ss = _second_order(jac, hess, h_x, g_x, eta, n_x=n_x, n_y=n_y)
    return _PerturbationSolution(
        h_x=h_x,
        g_x=g_x,
        h_xx=h_xx,
        g_xx=g_xx,
        h_ss=h_ss,
        g_ss=g_ss,
        eta=eta,
        x_ss=x_ss,
        y_ss=y_ss,
        order=2,
    )
