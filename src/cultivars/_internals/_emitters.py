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

from .._core import _LOG_2PI, _LOG_CHI2_MEAN, _LOG_CHI2_VAR
from ._parameters import (
    _DecayNelsonSiegelParameters,
    _StochasticVolatilityParameters,
    _TrendVolatilityParameters,
)
from ._substrates import _LinearGaussianStateSpace, _NonlinearStateSpace




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

    Args:
        params: The parameter record.

    Returns:
        The nonlinear state-space model.
    """
    mu, phi, sigma2, mean = params.mu, params.phi, params.sigma2, params.mean

    def transition(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return mu + phi * (np.asarray(states, dtype=np.float64) - mu)

    def observation(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.full((np.asarray(states).shape[0], 1), mean)

    def observation_loglik(
        y: npt.NDArray[np.float64], states: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        h = np.asarray(states, dtype=np.float64)[:, 0]
        residual = float(y[0]) - mean
        return -0.5 * (_LOG_2PI + h + residual**2 * np.exp(-h))

    return _NonlinearStateSpace(
        transition,
        observation,
        state_cov=[[sigma2]],
        obs_cov=[[1.0]],
        initial_state=[mu],
        initial_state_cov=[[params.stationary_variance]],
        observation_loglik=observation_loglik,
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
    sampler and the particle chain.

    Args:
        params: The parameter record.

    Returns:
        The linear-Gaussian state-space model in the log-squared data.
    """
    return _LinearGaussianStateSpace(
        np.ones((1, 1)),
        np.array([[_LOG_CHI2_VAR]]),
        np.array([[params.phi]]),
        np.eye(1),
        np.array([[params.sigma2]]),
        obs_intercept=np.array([_LOG_CHI2_MEAN]),
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
