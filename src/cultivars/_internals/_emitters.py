# filepath: /src/cultivars/_internals/_emitters.py
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
"""Emitters: parameter records to state-space systems on the nonlinear substrate.

The linear substrate builds its own systems (``_from_structural_system``,
``_from_nelson_siegel_system``) because a linear system *is* a set of
matrices. A nonlinear system is a pair of maps plus the hooks that place it
inside or outside the additive-Gaussian form, and which hooks a model needs
is a modeling statement, so those builders live here, one per public
model, each stating in its docstring which filters can read the system it
emits and why.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import digamma, gammaln, polygamma

from .._core import (
    _LOG_2PI,
    _LOG_CHI2_MEAN,
    _LOG_CHI2_VAR,
    _discrete_lyapunov,
    companion_matrix,
    fractional_difference_weights,
)
from ..exceptions import SpecificationError
from ._parameters import (
    _DecayNelsonSiegelParameters,
    _LongMemoryVolatilityParameters,
    _StochasticVolatilityParameters,
    _TrendVolatilityParameters,
)
from ._solutions import _PerturbationSolution
from ._substrates import _LinearGaussianStateSpace, _NonlinearStateSpace


def _log_scale_mixture_moments(nu: float) -> tuple[float, float]:
    """Mean and variance of ``log(lambda)`` for ``lambda ~ IG(nu/2, nu/2)``.

    Under the Student-t scale mixture ``log(eps**2) = log(z**2) + log(lambda)``,
    so the linearized measurement noise of the quasi-likelihood shifts by
    these two moments: ``E[log lambda] = log(nu/2) - psi(nu/2)`` and
    ``Var[log lambda] = psi'(nu/2)``.
    """
    half = 0.5 * nu
    return float(np.log(half) - digamma(half)), float(polygamma(1, half))


def _volatility_state_space(
    params: _StochasticVolatilityParameters,
) -> _NonlinearStateSpace:
    """The stochastic-volatility model on the nonlinear substrate.

    The state is the log variance ``h_t`` with a linear-Gaussian
    transition; the measurement ``y_t = c + exp(h_t / 2) eps_t`` is
    *multiplicative* in the state, so the Gaussian measurement density is
    replaced through ``observation_loglik`` and the emitted system is
    outside the additive form: the extended and unscented filters refuse
    it, and the particle filter is the reader. The ``observation`` map
    returns the observation mean ``c`` so the shape contract is honored;
    ``obs_cov`` is a placeholder the particle filter ignores.

    Heavy tails replace the Gaussian measurement density by the Student-t
    one. Leverage enters through ``observed_transition``: given the
    return, ``h_{t+1} = mu + phi (h_t - mu) + sigma rho eps_t`` with
    residual noise variance ``sigma2 (1 - rho**2)``, which keeps the
    transition density intact for the particle smoother. The two
    departures are not combined: with heavy tails the correlated
    innovation is the Gaussian kernel ``z_t`` rather than ``eps_t``, and
    recovering it from the return needs the mixture variable in the
    state, which this emitter does not carry.

    Args:
        params: The parameter record.

    Returns:
        The nonlinear state-space model.

    Raises:
        SpecificationError: If both ``nu`` and ``rho`` are set.
    """
    mu, phi, sigma2, mean = params.mu, params.phi, params.sigma2, params.mean
    nu, rho = params.nu, params.rho
    if nu is not None and rho != 0.0:
        raise SpecificationError(
            "heavy tails and leverage are not offered together: the leverage "
            "correlation attaches to the Gaussian kernel of the t innovation, "
            "which the return alone does not reveal. Fit one departure at a time."
        )

    def transition(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return mu + phi * (np.asarray(states, dtype=np.float64) - mu)

    def observation(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.full((np.asarray(states).shape[0], 1), mean)

    if nu is None:

        def observation_loglik(
            y: npt.NDArray[np.float64], states: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            h = np.asarray(states, dtype=np.float64)[:, 0]
            residual = float(y[0]) - mean
            return -0.5 * (_LOG_2PI + h + residual**2 * np.exp(-h))

    else:
        constant = float(gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(nu * np.pi))

        def observation_loglik(
            y: npt.NDArray[np.float64], states: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            h = np.asarray(states, dtype=np.float64)[:, 0]
            residual = float(y[0]) - mean
            return constant - 0.5 * h - 0.5 * (nu + 1.0) * np.log1p(residual**2 * np.exp(-h) / nu)

    observed_transition = None
    state_cov = sigma2
    if rho != 0.0:
        state_cov = sigma2 * (1.0 - rho**2)
        loading = float(np.sqrt(sigma2) * rho)

        def observed_transition(
            states: npt.NDArray[np.float64], y_prev: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            h = np.asarray(states, dtype=np.float64)
            previous = float(y_prev[0])
            if not np.isfinite(previous):
                return mu + phi * (h - mu)
            eps = (previous - mean) * np.exp(-0.5 * h)
            return mu + phi * (h - mu) + loading * eps

    return _NonlinearStateSpace(
        transition,
        observation,
        state_cov=[[state_cov]],
        obs_cov=[[1.0]],
        initial_state=[mu],
        initial_state_cov=[[params.stationary_variance]],
        observation_loglik=observation_loglik,
        observed_transition=observed_transition,
    )


def _quasi_volatility_state_space(
    params: _StochasticVolatilityParameters,
) -> _LinearGaussianStateSpace:
    """The Harvey-Ruiz-Shephard linearization on the linear substrate.

    Squaring and taking logs, ``log((y_t - c)**2) = h_t + log(eps_t**2)``,
    and ``log(eps_t**2)`` is replaced by a Gaussian with its mean
    ``-1.2704`` and variance ``pi**2 / 2``. The Kalman filter on this
    system yields the *quasi*-likelihood: consistent for the parameters,
    not the exact likelihood, and the estimator that seeds both the Gibbs
    sampler and the particle chain. Under Student-t noise the moments
    shift by those of ``log(lambda)`` (Ruiz 1994); the degrees of freedom
    are identified there only through the white part of the linearized
    noise variance, which the summary says.

    Args:
        params: The parameter record.

    Returns:
        The linear-Gaussian state-space model in the log-squared data.

    Raises:
        SpecificationError: If the record carries leverage, whose
            linearization needs the sign of the return as a second
            observation and is not offered here.
    """
    if params.rho != 0.0:
        raise SpecificationError(
            "the quasi-likelihood linearization discards the sign of the return, "
            "which is exactly what leverage acts through; the Harvey-Shephard "
            "sign-augmented system is not offered. Use the particle likelihood."
        )
    noise_mean, noise_var = _LOG_CHI2_MEAN, _LOG_CHI2_VAR
    if params.nu is not None:
        shift_mean, shift_var = _log_scale_mixture_moments(params.nu)
        noise_mean += shift_mean
        noise_var += shift_var
    return _LinearGaussianStateSpace(
        np.ones((1, 1)),
        np.array([[noise_var]]),
        np.array([[params.phi]]),
        np.eye(1),
        np.array([[params.sigma2]]),
        obs_intercept=np.array([noise_mean]),
        state_intercept=np.array([params.mu * (1.0 - params.phi)]),
        initial_state=np.array([params.mu]),
        initial_state_cov=np.array([[params.stationary_variance]]),
    )


def _trend_volatility_state_space(
    params: _TrendVolatilityParameters,
) -> _NonlinearStateSpace:
    """The unobserved-components stochastic-volatility model on the substrate.

    The state is ``(tau_t, h_t, q_t)``: the trend, the log variance of the
    irregular, and the log variance of the trend innovation. The trend's
    innovation variance depends on the state, so the transition noise is
    not additive: ``transition_sampler`` draws the state, and
    ``observation_loglik`` reads ``y_t ~ N(tau_t, exp(h_t))``. Both hooks
    place the system outside the additive form, so only the particle
    filter reads it; the Gibbs sampler is the model's primary estimator
    and this emitter is for the likelihood and for filtering new data.

    Args:
        params: The parameter record.

    Returns:
        The nonlinear state-space model.
    """
    g_h, g_q = float(params.gamma2_irregular), float(params.gamma2_trend)

    def transition(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(states, dtype=np.float64)

    def transition_sampler(
        states: npt.NDArray[np.float64], rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        block = np.asarray(states, dtype=np.float64)
        count = block.shape[0]
        out = np.empty_like(block)
        out[:, 0] = block[:, 0] + np.exp(0.5 * block[:, 2]) * rng.standard_normal(count)
        out[:, 1] = block[:, 1] + np.sqrt(g_h) * rng.standard_normal(count)
        out[:, 2] = block[:, 2] + np.sqrt(g_q) * rng.standard_normal(count)
        return out

    def observation(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(states, dtype=np.float64)[:, :1]

    def observation_loglik(
        y: npt.NDArray[np.float64], states: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        block = np.asarray(states, dtype=np.float64)
        gap = float(y[0]) - block[:, 0]
        return -0.5 * (_LOG_2PI + block[:, 1] + gap**2 * np.exp(-block[:, 1]))

    state_cov = np.diag([np.exp(params.q0), g_h, g_q])
    return _NonlinearStateSpace(
        transition,
        observation,
        state_cov=state_cov,
        obs_cov=[[np.exp(params.h0)]],
        initial_state=[params.tau0, params.h0, params.q0],
        initial_state_cov=np.diag([np.exp(params.h0) * 10.0, 1.0, 1.0]),
        transition_sampler=transition_sampler,
        observation_loglik=observation_loglik,
    )


def _decay_nelson_siegel_state_space(
    params: _DecayNelsonSiegelParameters,
    maturities: npt.NDArray[np.float64],
) -> _NonlinearStateSpace:
    """The dynamic Nelson-Siegel model with a time-varying decay on the substrate.

    The state is ``(level, slope, curvature, log lambda)``; the transition
    is linear-Gaussian (diagonal AR(1) in each), and the measurement
    ``y_t(tau) = L_t + S_t s(tau, lambda_t) + C_t c(tau, lambda_t)`` is
    nonlinear only through the loadings' dependence on ``lambda_t``. The
    system is additive-Gaussian, so the extended, unscented, and particle
    filters all read it; the unscented filter is the default reader
    because the loadings are smooth and bounded. The likelihoods those
    filters report are Gaussian approximations, labeled as such by the
    public result.

    Args:
        params: The parameter record.
        maturities: Strictly positive maturities, ``(p,)``.

    Returns:
        The nonlinear state-space model.
    """
    grid = np.asarray(maturities, dtype=np.float64).ravel()
    mu = np.concatenate([params.mu, [params.log_decay_mean]])
    ar = np.concatenate([params.ar, [params.decay_ar]])
    state_cov = np.zeros((4, 4))
    state_cov[:3, :3] = params.state_chol @ params.state_chol.T
    state_cov[3, 3] = params.decay_sd**2

    def transition(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        block = np.asarray(states, dtype=np.float64)
        return mu[None, :] + (block - mu[None, :]) * ar[None, :]

    def observation(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        block = np.asarray(states, dtype=np.float64)
        decay = np.exp(block[:, 3])[:, None]
        x = decay * grid[None, :]
        safe = np.where(x > 1e-10, x, 1.0)
        slope = np.where(x > 1e-10, (1.0 - np.exp(-x)) / safe, 1.0)
        curvature = slope - np.exp(-x)
        return block[:, :1] + block[:, 1:2] * slope + block[:, 2:3] * curvature

    # diagonal AR with a full innovation covariance: P_ij = Q_ij / (1 - a_i a_j)
    stationary = state_cov / (1.0 - np.outer(ar, ar))
    if params.decay_sd <= 0.0:
        stationary[3, 3] = 1e-8
    return _NonlinearStateSpace(
        transition,
        observation,
        state_cov=state_cov,
        obs_cov=np.diag(params.obs_var),
        initial_state=mu,
        initial_state_cov=stationary,
    )


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


def _long_memory_quasi_state_space(
    params: _LongMemoryVolatilityParameters, *, truncation: int
) -> _LinearGaussianStateSpace:
    """The linearized long-memory SV model as a truncated AR(inf) state space.

    ``log((y_t - c)**2) = mu + E[log eps**2] + v_t + xi_t`` with ``v_t``
    ARFIMA(1, d, 0). The fractional operator has no finite state
    representation, so ``v_t`` is written in its autoregressive form
    ``v_t = sum_j pi_j v_{t-j} + eta_t`` and cut at ``truncation`` lags: a
    companion system of that dimension on the linear substrate, exact for
    the truncated law and an approximation of the long-memory one whose
    error decays like ``truncation**(-d)``. The initial state covariance is
    the truncated law's stationary covariance. Both the truncation and the
    ``pi**2 / 2`` measurement floor are the linearization's, so what the
    Kalman filter returns on this system is a truncated quasi-likelihood.

    Args:
        params: The parameter record.
        truncation: Autoregressive lags retained, at least 1.

    Returns:
        The linear-Gaussian state-space model in the log-squared data.

    Raises:
        SpecificationError: If ``truncation`` is not positive.
    """
    if truncation < 1:
        raise SpecificationError(f"truncation must be at least 1; got {truncation}.")
    # AR(inf) of (1 - phi L)(1 - L)^d: weights of (1 - L)^d convolved with (1, -phi)
    frac = fractional_difference_weights(params.d, truncation + 1)
    poly = np.convolve(frac, np.array([1.0, -params.phi]))[: truncation + 1]
    ar = -poly[1:]
    m = truncation
    transition = companion_matrix(ar)
    selection = np.zeros((m, 1))
    selection[0, 0] = 1.0
    state_cov = np.array([[params.sigma2]])
    initial_cov = _discrete_lyapunov(transition, selection @ state_cov @ selection.T)
    design = np.zeros((1, m))
    design[0, 0] = 1.0
    return _LinearGaussianStateSpace(
        design,
        np.array([[_LOG_CHI2_VAR]]),
        transition,
        selection,
        state_cov,
        obs_intercept=np.array([params.mu + _LOG_CHI2_MEAN]),
        initial_state=np.zeros(m),
        initial_state_cov=initial_cov,
    )
