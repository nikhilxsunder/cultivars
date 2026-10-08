# filepath: /src/cultivars/_core/_samplers.py
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

"""Sampling primitives: single conditional draws with no recursion over data.

The membership test for this module is the same one that keeps the Hamilton
filter out of :mod:`cultivars._core`: a primitive may draw from a
distribution or vectorize arithmetic over a sample, but the moment a
function walks the time axis -- a filter pass, a smoother pass -- it is an
engine and belongs in the internals layer. The stochastic-volatility *path*
draw therefore lives in :mod:`cultivars._internals._samplers`, composing
these primitives with the state-space engine; what lives here is everything
it needs that has no memory: the Kim-Shephard-Chib mixture constants and
indicator draw, and the conjugate covariance draws.

References:
    Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
        Likelihood inference and comparison with ARCH models. *Review of
        Economic Studies*, 65(3), 361-393.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy.linalg as sla
import scipy.special as ssp
import scipy.stats as sst
from scipy.special import gammaln

from ..exceptions import NumericalError
from ._defaults import _GIG_MAX_ROUNDS, _GIG_TINY, _KSC_MEAN, _KSC_PROB, _KSC_VAR, _OFFSET


def _draw_inverse_wishart(
    scale: npt.NDArray[np.float64], df: float, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """One inverse-Wishart draw.

    Args:
        scale: Positive-definite scale matrix.
        df: Degrees of freedom, greater than ``dim - 1``.
        rng: Random generator.

    Returns:
        A positive-definite matrix of the scale's shape.

    Raises:
        NumericalError: If the scale has lost positive definiteness, which in
            a Gibbs loop means an upstream block has collapsed.
    """
    try:
        draw = np.asarray(
            sst.invwishart.rvs(df=df, scale=scale, random_state=rng),
            dtype=np.float64,
        )
    except np.linalg.LinAlgError as error:
        raise NumericalError(
            "an inverse-Wishart scale matrix lost positive definiteness during sampling."
        ) from error
    return np.atleast_2d(draw)


def _draw_inverse_gamma(shape: float, rate: float, rng: np.random.Generator) -> float:
    """One inverse-gamma draw under the shape/rate convention.

    Args:
        shape: Shape parameter ``a``.
        rate: Rate parameter ``b``, so the mean is ``b / (a - 1)`` for
            ``a > 1``.
        rng: Random generator.

    Returns:
        A positive scalar.
    """
    return float(rate / rng.gamma(shape, 1.0))


def _draw_mixture_indicators(
    log_squared: npt.NDArray[np.float64],
    log_variance: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> npt.NDArray[np.intp]:
    """Draw the KSC mixture component behind each observation.

    Args:
        log_squared: ``log(e_t**2 + offset)``, one per period.
        log_variance: The current log-variance path ``h_t``.
        rng: Random generator.

    Returns:
        Component indices in ``0..6``, one per period.
    """
    gap = log_squared[:, None] - log_variance[:, None] - _KSC_MEAN[None, :]
    log_kernel = (
        np.log(_KSC_PROB)[None, :]
        - 0.5 * np.log(_KSC_VAR)[None, :]
        - 0.5 * gap**2 / _KSC_VAR[None, :]
    )
    log_kernel -= log_kernel.max(axis=1, keepdims=True)
    prob = np.exp(log_kernel)
    prob /= prob.sum(axis=1, keepdims=True)
    uniform = np.asarray(rng.random(int(log_squared.shape[0])), dtype=np.float64)
    return np.asarray((prob.cumsum(axis=1) < uniform[:, None]).sum(axis=1), dtype=np.intp)


def _gig_rou(
    p: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Ratio-of-uniforms with mode shift, for the concentrated region.

    Exact for ``omega = sqrt(a * b) >= 1``: the bounding box comes from the
    two stationary points of ``(x - m)**2 f(x)``, whose defining cubic is
    solved trigonometrically in the mode-scaled variable ``x / m`` so its
    roots stay order one however extreme the raw parameters are.
    """
    mode = np.where(
        p < 1.0,
        b / (np.sqrt((1.0 - p) ** 2 + a * b) + (1.0 - p)),
        ((p - 1.0) + np.sqrt((p - 1.0) ** 2 + a * b)) / a,
    )
    alpha = a * mode
    beta = b / mode
    cubic2 = -(4.0 + 2.0 * (p - 1.0) + alpha) / alpha
    cubic1 = -(beta - 2.0 * (p - 1.0)) / alpha
    cubic0 = beta / alpha
    shift = cubic2 / 3.0
    reduced_p = cubic1 - cubic2**2 / 3.0
    reduced_q = 2.0 * cubic2**3 / 27.0 - cubic2 * cubic1 / 3.0 + cubic0
    radius = np.sqrt(np.maximum(-reduced_p / 3.0, 1e-300))
    angle = np.arccos(np.clip(-reduced_q / (2.0 * radius**3), -1.0, 1.0))
    roots = np.stack(
        [2.0 * radius * np.cos((angle - 2.0 * np.pi * turn) / 3.0) - shift for turn in range(3)]
    )

    def log_kernel(
        y: npt.NDArray[np.float64], idx: npt.NDArray[np.intp] | slice
    ) -> npt.NDArray[np.float64]:
        return (p[idx] - 1.0) * np.log(y) - 0.5 * (
            alpha[idx] * (y - 1.0) + beta[idx] * (1.0 / y - 1.0)
        )

    below = np.where((roots > 0.0) & (roots < 1.0), roots, -np.inf).max(axis=0)
    above = np.where(roots > 1.0, roots, np.inf).min(axis=0)
    if not (np.all(np.isfinite(below)) and np.all(np.isfinite(above))):
        raise NumericalError(
            "a generalized-inverse-Gaussian bounding box could not be "
            "bracketed; the conditional's parameters have degenerated."
        )
    everything = slice(None)
    u_lo = (below - 1.0) * np.exp(0.5 * log_kernel(below, everything))
    u_hi = (above - 1.0) * np.exp(0.5 * log_kernel(above, everything))
    out = np.empty(p.size)
    active = np.arange(p.size)
    for _ in range(_GIG_MAX_ROUNDS):
        if not active.size:
            return out
        u = u_lo[active] + (u_hi[active] - u_lo[active]) * rng.random(active.size)
        v = np.asarray(rng.random(active.size), dtype=np.float64)
        y = 1.0 + u / v
        inside = y > 0.0
        log_height = np.full(active.size, -np.inf)
        log_height[inside] = log_kernel(y[inside], active[inside])
        accepted = 2.0 * np.log(v) <= log_height
        out[active[accepted]] = mode[active[accepted]] * y[accepted]
        active = active[~accepted]
    raise NumericalError(
        "generalized-inverse-Gaussian rejection sampling failed to accept; "
        "the conditional's parameters have degenerated."
    )


def _gig_tail(
    p: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Boundary-proposal rejection, for the diffuse region away from zero order.

    For ``omega < 1`` the density is close to the distribution it degenerates
    to, which is therefore an efficient and exactly dominating proposal: an
    inverse-Gamma for negative order accepted with probability
    ``exp(-a * x / 2)``, a Gamma for positive order accepted with
    ``exp(-b / (2 * x))``. Acceptance approaches one as ``omega`` falls.
    """
    out = np.empty(p.size)
    active = np.arange(p.size)
    for _ in range(_GIG_MAX_ROUNDS):
        if not active.size:
            return out
        order = p[active]
        negative = order < 0.0
        x = np.empty(active.size)
        x[negative] = (b[active][negative] / 2.0) / rng.gamma(-order[negative], 1.0)
        x[~negative] = rng.gamma(order[~negative], 2.0 / a[active][~negative])
        log_accept = np.where(
            negative,
            -a[active] * x / 2.0,
            -b[active] / (2.0 * np.maximum(x, 1e-300)),
        )
        accepted = np.log(rng.random(active.size)) <= log_accept
        out[active[accepted]] = x[accepted]
        active = active[~accepted]
    raise NumericalError(
        "generalized-inverse-Gaussian rejection sampling failed to accept; "
        "the conditional's parameters have degenerated."
    )


def _gig_corner(
    p: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Piecewise-hat rejection for the diffuse region near zero order.

    With ``|p| < 1/2`` and small ``omega`` the density is close to
    log-uniform between its two exponential cutoffs, so neither boundary
    distribution alone proposes well. The hat is exact and piecewise: a
    truncated inverse-Gamma below ``L``, the log-uniform envelope on
    ``[L, U]``, and a truncated Gamma above ``U``, each bounded by its
    regional supremum. Valid whenever ``omega**2 < 4 (2p + 1)(1 - 2p)``,
    which the caller's ``|p| < 0.2`` and ``omega < 1`` guarantee.
    """
    lower = 0.5 * b / (1.0 - 2.0 * p)
    upper = 2.0 * (2.0 * p + 1.0) / a
    log_m1 = (p + 0.5) * np.log(lower)
    log_m2 = (p - 0.5) * np.log(upper)
    log_m3 = np.maximum(p * np.log(lower), p * np.log(upper))
    tail1 = np.asarray(ssp.gammaincc(0.5, 1.0 - 2.0 * p), dtype=np.float64)
    tail2 = np.asarray(ssp.gammaincc(0.5, 2.0 * p + 1.0), dtype=np.float64)
    root_pi = float(np.sqrt(np.pi))
    log_z1 = log_m1 + 0.5 * (np.log(2.0) - np.log(b)) + np.log(root_pi * tail1)
    log_z2 = log_m2 + 0.5 * (np.log(2.0) - np.log(a)) + np.log(root_pi * tail2)
    log_z3 = log_m3 + np.log(np.log(upper / lower))
    peak = np.maximum(np.maximum(log_z1, log_z2), log_z3)
    w1 = np.exp(log_z1 - peak)
    w2 = np.exp(log_z2 - peak)
    w3 = np.exp(log_z3 - peak)
    total = w1 + w2 + w3
    out = np.empty(p.size)
    active = np.arange(p.size)
    for _ in range(_GIG_MAX_ROUNDS):
        if not active.size:
            return out
        pick = rng.random(active.size) * total[active]
        in1 = pick < w1[active]
        in3 = ~in1 & (pick < (w1 + w3)[active])
        in2 = ~in1 & ~in3
        x = np.empty(active.size)
        idx1, idx3, idx2 = active[in1], active[in3], active[in2]
        draw = np.asarray(
            ssp.gammainccinv(0.5, rng.random(idx1.size) * tail1[idx1]),
            dtype=np.float64,
        )
        x[in1] = b[idx1] / (2.0 * draw)
        x[in3] = np.exp(
            np.log(lower[idx3])
            + (np.log(upper[idx3]) - np.log(lower[idx3])) * rng.random(idx3.size)
        )
        draw = np.asarray(
            ssp.gammainccinv(0.5, rng.random(idx2.size) * tail2[idx2]),
            dtype=np.float64,
        )
        x[in2] = 2.0 * draw / a[idx2]
        log_f = (p[active] - 1.0) * np.log(x) - 0.5 * (a[active] * x + b[active] / x)
        log_hat = np.empty(active.size)
        log_hat[in1] = log_m1[idx1] - 1.5 * np.log(x[in1]) - b[idx1] / (2.0 * x[in1])
        log_hat[in3] = log_m3[idx3] - np.log(x[in3])
        log_hat[in2] = log_m2[idx2] - 0.5 * np.log(x[in2]) - a[idx2] * x[in2] / 2.0
        accepted = np.log(rng.random(active.size)) <= log_f - log_hat
        out[active[accepted]] = x[accepted]
        active = active[~accepted]
    raise NumericalError(
        "generalized-inverse-Gaussian rejection sampling failed to accept; "
        "the conditional's parameters have degenerated."
    )


def _draw_generalized_inverse_gaussian(
    order: npt.NDArray[np.float64] | float,
    tilt: npt.NDArray[np.float64] | float,
    chi: npt.NDArray[np.float64] | float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Elementwise generalized-inverse-Gaussian draws.

    The density is ``x**(p - 1) * exp(-(a * x + b / x) / 2)`` with ``p`` the
    order, ``a`` the tilt, and ``b`` the chi parameter -- the conditional
    that the Dirichlet-Laplace and Normal-Gamma hierarchies both land on.
    Everything here is exact rejection sampling, vectorized over elements
    because a global-local Gibbs sweep redraws thousands of these with
    element-specific parameters and a per-element library call is three
    orders of magnitude too slow. Three interior regimes get three exact
    samplers -- ratio-of-uniforms where the density is concentrated
    (``omega = sqrt(a b) >= 1``), boundary-distribution proposals where it
    is diffuse with the order away from zero, and a piecewise log-uniform
    hat where it is diffuse with the order near zero, the regime where the
    density is nearly flat on the log scale between its two exponential
    cutoffs. The two boundary faces are drawn as the distributions they
    are (``b -> 0`` a Gamma, ``a -> 0`` an inverse-Gamma) rather than
    approached.

    Args:
        order: The order ``p``; any real, broadcast against the others.
        tilt: The rate-like parameter ``a``, non-negative.
        chi: The reciprocal-rate-like parameter ``b``, non-negative.
        rng: Random generator.

    Returns:
        Positive draws in the broadcast shape.

    Raises:
        NumericalError: If a boundary face has no proper limit (``chi``
            near zero needs positive order; ``tilt`` near zero needs
            negative order), or rejection fails to accept.
    """
    p, a, b = np.broadcast_arrays(
        np.asarray(order, dtype=np.float64),
        np.asarray(tilt, dtype=np.float64),
        np.asarray(chi, dtype=np.float64),
    )
    shape = p.shape
    p = p.ravel().astype(np.float64, copy=True)
    a = a.ravel().astype(np.float64, copy=True)
    b = b.ravel().astype(np.float64, copy=True)
    out = np.empty(p.size)
    gamma_face = b <= _GIG_TINY
    inverse_face = (a <= _GIG_TINY) & ~gamma_face
    interior = ~gamma_face & ~inverse_face
    if np.any(gamma_face):
        if np.any(p[gamma_face] <= 0.0):
            raise NumericalError(
                "a generalized-inverse-Gaussian conditional degenerated: "
                "chi near zero with non-positive order has no proper limit."
            )
        out[gamma_face] = rng.gamma(p[gamma_face], 2.0 / a[gamma_face])
    if np.any(inverse_face):
        if np.any(p[inverse_face] >= 0.0):
            raise NumericalError(
                "a generalized-inverse-Gaussian conditional degenerated: "
                "tilt near zero with non-negative order has no proper limit."
            )
        out[inverse_face] = (b[inverse_face] / 2.0) / rng.gamma(-p[inverse_face], 1.0)
    if np.any(interior):
        omega = np.sqrt(a[interior] * b[interior])
        concentrated = interior.copy()
        concentrated[interior] = omega >= 1.0
        diffuse_tail = interior.copy()
        diffuse_tail[interior] = (omega < 1.0) & (np.abs(p[interior]) >= 0.2)
        diffuse_corner = interior & ~concentrated & ~diffuse_tail
        if np.any(concentrated):
            out[concentrated] = _gig_rou(p[concentrated], a[concentrated], b[concentrated], rng)
        if np.any(diffuse_tail):
            out[diffuse_tail] = _gig_tail(p[diffuse_tail], a[diffuse_tail], b[diffuse_tail], rng)
        if np.any(diffuse_corner):
            out[diffuse_corner] = _gig_corner(
                p[diffuse_corner], a[diffuse_corner], b[diffuse_corner], rng
            )
    return out.reshape(shape)


def _gamma_from_mode(mode: float, sd: float) -> tuple[float, float]:
    """Gamma shape and scale matching a stated mode and standard deviation.

    The parameterization Giannone-Lenza-Primiceri state their hyperpriors
    in. With mode ``m = (a - 1) b`` and variance ``a b^2``, the scale solves
    ``b^2 + m b - sd^2 = 0``.

    Args:
        mode: The distribution's mode, non-negative.
        sd: The distribution's standard deviation, positive.

    Returns:
        ``(shape, scale)``.
    """
    scale = 0.5 * (np.sqrt(mode**2 + 4.0 * sd**2) - mode)
    return mode / scale + 1.0, float(scale)


def _draw_factors(
    panel: npt.NDArray[np.float64],
    loadings: npt.NDArray[np.float64],
    h_factor: npt.NDArray[np.float64],
    h_idio: npt.NDArray[np.float64],
    *,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Draw the latent factors of a factor SV model, one Gaussian per period.

    Given loadings and both sets of log variances the model is linear and
    Gaussian period by period: ``f_t | y_t ~ N(m_t, V_t)`` with
    ``V_t**-1 = Lambda' D_t**-1 Lambda + H_t**-1`` and ``m_t = V_t Lambda'
    D_t**-1 y_t``, where ``D_t`` and ``H_t`` are the idiosyncratic and
    factor variances. The ``T`` small systems are solved in one batch.

    Args:
        panel: ``(T, k)`` demeaned observations.
        loadings: ``(k, r)`` loading matrix.
        h_factor: ``(T, r)`` factor log variances.
        h_idio: ``(T, k)`` idiosyncratic log variances.
        rng: Random generator.

    Returns:
        The ``(T, r)`` factor draw.
    """
    nobs, r = h_factor.shape
    weights = np.exp(-h_idio)
    precision = np.einsum("kj,tk,kl->tjl", loadings, weights, loadings)
    precision[:, np.arange(r), np.arange(r)] += np.exp(-h_factor)
    rhs = np.einsum("kj,tk->tj", loadings, weights * panel)
    mean = np.linalg.solve(precision, rhs[:, :, None])[:, :, 0]
    chol = np.linalg.cholesky(precision)
    noise = np.linalg.solve(np.transpose(chol, (0, 2, 1)), rng.standard_normal((nobs, r, 1)))[
        :, :, 0
    ]
    return np.asarray(mean + noise, dtype=np.float64)


def _draw_loading_rows(
    panel: npt.NDArray[np.float64],
    factors: npt.NDArray[np.float64],
    loadings: npt.NDArray[np.float64],
    h_idio: npt.NDArray[np.float64],
    *,
    prior_precision: float,
    rng: np.random.Generator,
) -> None:
    """Draw the free loadings row by row, in place, under the triangular convention.

    Row ``i`` loads on factors ``j < min(i, r)`` freely and, when ``i < r``,
    on factor ``i`` with the loading fixed at one; rows ``i >= r`` load on
    every factor. Each row's free block is a weighted Gaussian regression
    of the series on the factors with weights ``exp(-h_it)``, under an
    independent ``N(0, 1 / prior_precision)`` prior.

    Args:
        panel: ``(T, k)`` demeaned observations.
        factors: ``(T, r)`` current factors.
        loadings: ``(k, r)`` loading matrix, updated in place.
        h_idio: ``(T, k)`` idiosyncratic log variances.
        prior_precision: Prior precision on each free loading.
        rng: Random generator.
    """
    k, r = loadings.shape
    for i in range(k):
        n_free = min(i, r)
        if n_free == 0:
            continue
        weights = np.exp(-h_idio[:, i])
        target = panel[:, i] - (factors[:, i] if i < r else 0.0)
        design = factors[:, :n_free]
        precision = design.T @ (design * weights[:, None]) + prior_precision * np.eye(n_free)
        mean = np.linalg.solve(precision, design.T @ (weights * target))
        root = np.linalg.cholesky(np.linalg.inv(precision))
        loadings[i, :n_free] = mean + root @ rng.standard_normal(n_free)


def _scalar_ffbs(
    obs: npt.NDArray[np.float64],
    noise: npt.NDArray[np.float64],
    *,
    phi: float,
    drift: float,
    sigma2: float | npt.NDArray[np.float64],
    init_mean: float,
    init_var: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Forward-filter backward-sample one scalar Gaussian state path.

    The state follows ``h_{t+1} = drift + phi h_t + eta_t`` with
    ``Var(eta_t) = sigma2`` (a scalar, or one value per period) and starts
    at ``N(init_mean, init_var)``; the observation is ``obs_t = h_t + e_t``
    with known per-period variance ``noise_t``. This is the substrate's
    Kalman filter and simulation smoother specialized to one state and
    written out: the volatility samplers call it once per equation per
    sweep, where the general machinery's per-step overhead is two orders
    of magnitude of wasted time. Equivalence to the substrate route is
    verified, not assumed.

    Args:
        obs: Observations with the mixture means removed, ``(n,)``.
        noise: Per-period observation variances, ``(n,)``.
        phi: State persistence.
        drift: State intercept.
        sigma2: State innovation variance, scalar or ``(n,)``; entry ``t``
            drives the step from ``t`` to ``t + 1``.
        init_mean: Initial state mean.
        init_var: Initial state variance.
        rng: Random generator.

    Returns:
        One exact draw of the state path, ``(n,)``.
    """
    n = obs.shape[0]
    state_var = np.broadcast_to(np.asarray(sigma2, dtype=np.float64), (n,))
    filtered_mean = np.empty(n)
    filtered_var = np.empty(n)
    mean, var = init_mean, init_var
    for t in range(n):
        gain = var / (var + noise[t])
        mean = mean + gain * (obs[t] - mean)
        var = var * (1.0 - gain)
        filtered_mean[t] = mean
        filtered_var[t] = var
        mean = drift + phi * mean
        var = phi**2 * var + state_var[t]
    path = np.empty(n)
    path[-1] = filtered_mean[-1] + np.sqrt(max(filtered_var[-1], 0.0)) * rng.standard_normal()
    for t in range(n - 2, -1, -1):
        predicted_var = phi**2 * filtered_var[t] + state_var[t]
        pull = filtered_var[t] * phi / max(predicted_var, 1e-300)
        cond_mean = filtered_mean[t] + pull * (path[t + 1] - drift - phi * filtered_mean[t])
        cond_var = filtered_var[t] - pull * phi * filtered_var[t]
        path[t] = cond_mean + np.sqrt(max(cond_var, 0.0)) * rng.standard_normal()
    return np.asarray(path, dtype=np.float64)


def _draw_volatility_path(
    residual: npt.NDArray[np.float64],
    log_variance: npt.NDArray[np.float64],
    vol_of_vol: float,
    *,
    prior_mean: float,
    prior_var: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Draw one equation's random-walk log-variance path, KSC-conditionally.

    Given the current path, mixture indicators are drawn; conditional on
    them the observation ``log(e_t**2 + offset)`` is linear-Gaussian in
    ``h_t``, and the path is drawn exactly by the scalar forward-filter
    backward-sampler :func:`_scalar_ffbs`.

    Args:
        residual: The equation's orthogonalized residuals ``e_t``.
        log_variance: The current path, used to draw the indicators.
        vol_of_vol: Current innovation variance of the random walk.
        prior_mean: Prior mean of the initial log variance.
        prior_var: Prior variance of the initial log variance.
        rng: Random generator.

    Returns:
        A new log-variance path of the residual's length.
    """
    star = np.log(residual**2 + _OFFSET)
    components = _draw_mixture_indicators(star, log_variance, rng)
    return _scalar_ffbs(
        star - _KSC_MEAN[components],
        _KSC_VAR[components],
        phi=1.0,
        drift=0.0,
        sigma2=float(vol_of_vol),
        init_mean=float(prior_mean),
        init_var=float(prior_var),
        rng=rng,
    )


def _draw_triangular_volatility_block(
    resid: npt.NDArray[np.float64],
    a_mat: npt.NDArray[np.float64],
    h_path: npt.NDArray[np.float64],
    vol_of_vol: npt.NDArray[np.float64],
    *,
    log_diag0: npt.NDArray[np.float64],
    k_vol: float,
    a_prior_prec: float,
    rng: np.random.Generator,
) -> None:
    """One Gibbs sweep of the Primiceri volatility block, in place.

    Draws the sub-diagonal rows of the unit-lower-triangular ``A`` by
    weighted least squares on the current log variances, orthogonalizes
    the residuals, then for each equation draws the random-walk
    log-variance path by :func:`_draw_volatility_path` and its innovation
    variance by inverse-gamma. ``a_mat``, ``h_path`` and ``vol_of_vol``
    are updated in place and nothing is returned, so the two callers --
    the BVAR-SV and the TVP-VAR-SV engines -- cannot drift apart in what
    they keep. The draw order (all rows of ``A``, then path and variance
    equation by equation) is the order both engines used before the block
    was shared, so seeded runs are unchanged.

    Args:
        resid: ``(T, k)`` reduced-form residuals at the current coefficients.
        a_mat: ``(k, k)`` unit-lower-triangular contemporaneous matrix.
        h_path: ``(T, k)`` current log-variance paths.
        vol_of_vol: ``(k,)`` current innovation variances of the paths.
        log_diag0: ``(k,)`` prior means of the initial log variances.
        k_vol: Primiceri's ``k_W``; the inverse-gamma scale is ``2 k_W**2``.
        a_prior_prec: Prior precision on each free element of ``A``.
        rng: Random generator.
    """
    nobs, k = resid.shape
    for i in range(1, k):
        weights = np.exp(-h_path[:, i])
        x_reg = -resid[:, :i]
        row_precision = x_reg.T @ (x_reg * weights[:, None]) + a_prior_prec * np.eye(i)
        row_mean = np.linalg.solve(row_precision, x_reg.T @ (resid[:, i] * weights))
        root = np.linalg.cholesky(np.linalg.inv(row_precision))
        a_mat[i, :i] = row_mean + root @ rng.standard_normal(i)
    ortho = resid @ a_mat.T
    for i in range(k):
        h_path[:, i] = _draw_volatility_path(
            ortho[:, i],
            h_path[:, i],
            float(vol_of_vol[i]),
            prior_mean=float(log_diag0[i]),
            prior_var=4.0,
            rng=rng,
        )
        steps = np.diff(h_path[:, i])
        vol_of_vol[i] = _draw_inverse_gamma(
            2.0 + 0.5 * (nobs - 1.0),
            2.0 * k_vol**2 + 0.5 * float(steps @ steps),
            rng,
        )


def _draw_stationary_volatility_path(
    residual: npt.NDArray[np.float64],
    log_variance: npt.NDArray[np.float64],
    *,
    mu: float,
    phi: float,
    sigma2: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Draw a mean-reverting log-variance path, KSC-conditionally.

    The stationary counterpart of :func:`_draw_volatility_path`: the log
    variance follows ``h_{t+1} = mu + phi (h_t - mu) + sigma eta_t`` and
    starts from its stationary distribution. Given the current path,
    mixture indicators are drawn; conditional on them the observation
    ``log(e_t**2 + offset)`` is linear-Gaussian in ``h_t`` and the path is
    drawn exactly by the scalar forward-filter backward-sampler
    :func:`_scalar_ffbs`, initialized at the stationary law.

    Args:
        residual: The demeaned observations ``e_t``.
        log_variance: The current path, used to draw the indicators.
        mu: Unconditional mean of the log variance.
        phi: Persistence, strictly inside ``(-1, 1)``.
        sigma2: Innovation variance of the log variance.
        rng: Random generator.

    Returns:
        A new log-variance path of the residual's length.
    """
    star = np.log(residual**2 + _OFFSET)
    components = _draw_mixture_indicators(star, log_variance, rng)
    return _scalar_ffbs(
        star - _KSC_MEAN[components],
        _KSC_VAR[components],
        phi=float(phi),
        drift=float(mu * (1.0 - phi)),
        sigma2=float(sigma2),
        init_mean=float(mu),
        init_var=float(sigma2 / max(1.0 - phi**2, 1e-8)),
        rng=rng,
    )


def _draw_persistence(
    centered: npt.NDArray[np.float64],
    *,
    phi: float,
    sigma2: float,
    prior_phi: tuple[float, float],
    rng: np.random.Generator,
) -> float:
    """Metropolis-Hastings step for the AR(1) persistence of a centered path.

    The proposal is the Gaussian conditional of the regression of
    ``centered[1:]`` on ``centered[:-1]``; the acceptance ratio carries the
    stationary-initialization term ``sqrt(1 - phi**2)`` and the Beta prior
    on ``(phi + 1) / 2``, exactly as in Kim, Shephard, and Chib (1998,
    section 3.3). Serves both the centered path (innovation variance
    ``sigma2``) and the non-centered one (unit variance).

    Args:
        centered: The path with its level removed.
        phi: Current persistence.
        sigma2: Innovation variance of the path.
        prior_phi: ``(a, b)`` of the Beta prior on ``(phi + 1) / 2``.
        rng: Random generator.

    Returns:
        The persistence after the step.
    """
    sxx = float(centered[:-1] @ centered[:-1])
    sxy = float(centered[1:] @ centered[:-1])
    if not sxx > 0.0:
        return phi
    phi_hat = sxy / sxx
    phi_prop = float(phi_hat + rng.standard_normal() * np.sqrt(sigma2 / sxx))
    if not abs(phi_prop) < 1.0:
        return phi
    a, b = prior_phi

    def log_target_extra(value: float) -> float:
        return (
            0.5 * np.log(1.0 - value**2)
            - 0.5 * (1.0 - value**2) * centered[0] ** 2 / sigma2
            + (a - 1.0) * np.log(0.5 * (1.0 + value))
            + (b - 1.0) * np.log(0.5 * (1.0 - value))
        )

    log_ratio = log_target_extra(phi_prop) - log_target_extra(phi)
    if np.log(rng.random()) < log_ratio:
        return phi_prop
    return phi


def _draw_volatility_parameters(
    log_variance: npt.NDArray[np.float64],
    *,
    mu: float,
    phi: float,
    sigma2: float,
    prior_mu: tuple[float, float],
    prior_phi: tuple[float, float],
    prior_sigma2: tuple[float, float],
    rng: np.random.Generator,
    draw_mean: bool = True,
) -> tuple[float, float, float]:
    """One sweep of the Kim-Shephard-Chib parameter blocks given a path.

    ``mu`` is drawn from its Gaussian conditional; ``sigma2`` from its
    inverse-gamma conditional; ``phi`` by a Metropolis-Hastings step whose
    proposal is the Gaussian conditional of the AR(1) regression and
    whose acceptance ratio carries the stationary-initialization term
    ``sqrt(1 - phi**2)`` and the Beta prior on ``(phi + 1) / 2``, exactly
    as in Kim, Shephard, and Chib (1998, section 3.3).

    Args:
        log_variance: The current path ``h``.
        mu: Current mean.
        phi: Current persistence.
        sigma2: Current innovation variance.
        prior_mu: ``(mean, variance)`` of the Gaussian prior on ``mu``.
        prior_phi: ``(a, b)`` of the Beta prior on ``(phi + 1) / 2``.
        prior_sigma2: ``(shape, rate)`` of the inverse-gamma prior.
        rng: Random generator.
        draw_mean: Whether to draw mu; False holds it at the value passed in, which is how a model
            that carries the scale elsewhere (a structural impact matrix, say) pins the log-variance
            level.

    Returns:
        ``(mu, phi, sigma2)`` after one sweep.
    """
    h = log_variance
    n = h.shape[0]
    # -- mu | h, phi, sigma2 : Gaussian -------------------------------------
    if draw_mean:
        m0, v0 = prior_mu
        precision = 1.0 / v0 + ((1.0 - phi**2) + (n - 1) * (1.0 - phi) ** 2) / sigma2
        moment = (
            m0 / v0
            + ((1.0 - phi**2) * h[0] + (1.0 - phi) * float(np.sum(h[1:] - phi * h[:-1]))) / sigma2
        )
        mu = float(moment / precision + rng.standard_normal() / np.sqrt(precision))
    # -- phi | h, mu, sigma2 : MH with the regression conditional -----------
    centered = h - mu
    phi = _draw_persistence(centered, phi=phi, sigma2=sigma2, prior_phi=prior_phi, rng=rng)
    # -- sigma2 | h, mu, phi : inverse-gamma -------------------------------
    shape0, rate0 = prior_sigma2
    residual_ss = (1.0 - phi**2) * centered[0] ** 2 + float(
        np.sum((centered[1:] - phi * centered[:-1]) ** 2)
    )
    sigma2 = _draw_inverse_gamma(shape0 + 0.5 * n, rate0 + 0.5 * residual_ss, rng)
    return mu, phi, sigma2


def _interweave_volatility_parameters(
    residual: npt.NDArray[np.float64],
    log_variance: npt.NDArray[np.float64],
    *,
    mu: float,
    phi: float,
    sigma2: float,
    prior_mu: tuple[float, float],
    prior_phi: tuple[float, float],
    prior_sigma2: tuple[float, float],
    rng: np.random.Generator,
    draw_mean: bool = True,
) -> tuple[npt.NDArray[np.float64], float, float, float]:
    """The non-centered half of an ancillarity-sufficiency interweaving sweep.

    The centered parameterization ``h_t = mu + phi (h_{t-1} - mu) + sigma
    eta_t`` mixes badly when ``sigma`` is small: the path pins ``mu`` and
    ``sigma`` almost exactly, so their conditional draws barely move, and the
    inefficiency factor of ``sigma2`` runs into the hundreds or thousands.
    Kastner and Frühwirth-Schnatter (2014) showed that alternating with the
    non-centered path ``h~_t = (h_t - mu) / sigma`` -- in which ``mu`` and
    ``sigma`` are regression coefficients of the observation equation and
    the path carries neither -- and transforming back, removes the problem
    in both regimes at the cost of one extra parameter draw per sweep.

    Called after :func:`_draw_volatility_parameters`, this step maps the
    path to its non-centered form, redraws ``phi`` from the unit-variance
    AR(1) regression, redraws the mixture indicators given the current path,
    proposes ``(mu, sigma)`` from the weighted regression ``log(e_t**2 +
    offset) - m_{s_t} = mu + sigma h~_t + v_{s_t}^(1/2) xi_t`` (with the
    Gaussian prior on ``mu`` folded in and a flat prior on ``sigma``), and
    accepts it by Metropolis-Hastings against the inverse-gamma prior on
    ``sigma2`` with its Jacobian ``|sigma|``. The path is then rebuilt as
    ``mu + sigma h~``; ``sigma`` may leave the step negative, which is the
    same path with the sign of ``h~`` flipped and is immaterial to the
    centered representation that follows.

    Args:
        residual: The demeaned observations ``e_t``, already divided by any
            scale-mixture variables.
        log_variance: The current centered path ``h``.
        mu: Current mean of the log variance.
        phi: Current persistence.
        sigma2: Current innovation variance.
        prior_mu: ``(mean, variance)`` of the Gaussian prior on ``mu``.
        prior_phi: ``(a, b)`` of the Beta prior on ``(phi + 1) / 2``.
        prior_sigma2: ``(shape, rate)`` of the inverse-gamma prior.
        rng: Random generator.
        draw_mean: Whether ``mu`` is a free parameter; ``False`` regresses
            on ``sigma`` alone with the level held at ``mu``.

    Returns:
        ``(log_variance, mu, phi, sigma2)`` after the non-centered move.

    Raises:
        NumericalError: If the path carries no variation to scale by.

    References:
        Kastner, G., & Frühwirth-Schnatter, S. (2014). Ancillarity-
            sufficiency interweaving strategy (ASIS) for boosting MCMC
            estimation of stochastic volatility models. *Computational
            Statistics & Data Analysis*, 76, 408-423.
    """
    sigma = float(np.sqrt(sigma2))
    if not sigma > 0.0:
        raise NumericalError("The log-variance innovation variance collapsed to zero.")
    tilde = (log_variance - mu) / sigma
    phi = _draw_persistence(tilde, phi=phi, sigma2=1.0, prior_phi=prior_phi, rng=rng)
    star = np.log(residual**2 + _OFFSET)
    components = _draw_mixture_indicators(star, log_variance, rng)
    target = star - _KSC_MEAN[components]
    weight = 1.0 / _KSC_VAR[components]
    shape0, rate0 = prior_sigma2

    def log_prior_sigma(value: float) -> float:
        # Inverse-gamma on sigma**2 with the Jacobian of the map sigma -> sigma**2.
        return -(2.0 * shape0 + 1.0) * np.log(abs(value)) - rate0 / value**2

    if draw_mean:
        m0, v0 = prior_mu
        design = np.column_stack([np.ones_like(tilde), tilde])
        precision = (design * weight[:, None]).T @ design
        precision[0, 0] += 1.0 / v0
        moment = design.T @ (weight * target)
        moment[0] += m0 / v0
        try:
            chol = np.linalg.cholesky(precision)
        except np.linalg.LinAlgError as error:
            raise NumericalError(
                "The non-centered regression lost positive definiteness."
            ) from error
        center = np.linalg.solve(precision, moment)
        proposal = center + np.linalg.solve(chol.T, rng.standard_normal(2))
        mu_prop, sigma_prop = float(proposal[0]), float(proposal[1])
    else:
        precision_scalar = float(weight @ (tilde**2))
        if not precision_scalar > 0.0:
            raise NumericalError("The non-centered path carries no variation to scale by.")
        center_scalar = float(weight @ (tilde * (target - mu))) / precision_scalar
        mu_prop = mu
        sigma_prop = float(center_scalar + rng.standard_normal() / np.sqrt(precision_scalar))
    if sigma_prop != 0.0:
        log_ratio = log_prior_sigma(sigma_prop) - log_prior_sigma(sigma)
        if np.log(rng.random()) < log_ratio:
            mu, sigma = mu_prop, sigma_prop
    return mu + sigma * tilde, mu, phi, sigma * sigma


def _draw_scale_mixture(
    residual: npt.NDArray[np.float64],
    log_variance: npt.NDArray[np.float64],
    *,
    nu: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Draw the Student-t scale-mixture variables, one per observation.

    With ``eps_t = sqrt(lambda_t) z_t`` and ``lambda_t ~ IG(nu/2, nu/2)`` a
    priori, the conditional given the standardized residual
    ``e_t exp(-h_t / 2)`` is again inverse-gamma, with shape
    ``(nu + 1) / 2`` and scale ``(nu + e_t**2 exp(-h_t)) / 2``. Dividing
    the residual by ``sqrt(lambda_t)`` then returns the observation to the
    Gaussian form the KSC mixture step expects.

    Args:
        residual: The demeaned observations ``e_t``.
        log_variance: The current log-variance path.
        nu: Current degrees of freedom.
        rng: Random generator.

    Returns:
        The ``(n,)`` mixture variables ``lambda_t``.
    """
    standardized_sq = residual**2 * np.exp(-log_variance)
    shape = 0.5 * (nu + 1.0)
    scale = 0.5 * (nu + standardized_sq)
    return np.asarray(scale / rng.gamma(shape, 1.0, size=residual.shape[0]), dtype=np.float64)


def _draw_degrees_of_freedom(
    mixture: npt.NDArray[np.float64],
    nu: float,
    *,
    prior_rate: float,
    step: float,
    rng: np.random.Generator,
) -> tuple[float, bool]:
    """One random-walk Metropolis step on the degrees of freedom.

    The conditional of ``nu`` given the mixture variables is the product
    of ``IG(nu/2, nu/2)`` densities times the prior ``nu - 2 ~
    Exponential(prior_rate)``; the walk is on ``log(nu - 2)`` so the
    support is respected, with the log-Jacobian folded in. This is the
    same prior the particle chain places on the coordinate, so the two
    samplers target one posterior.

    Args:
        mixture: The ``(n,)`` mixture variables ``lambda_t``.
        nu: Current degrees of freedom.
        prior_rate: Rate of the exponential prior on ``nu - 2``.
        step: Standard deviation of the proposal on ``log(nu - 2)``.
        rng: Random generator.

    Returns:
        The new degrees of freedom and whether the proposal was accepted.
    """
    n = mixture.shape[0]
    sum_log = float(np.sum(np.log(mixture)))
    sum_inv = float(np.sum(1.0 / mixture))

    def log_target(value: float) -> float:
        half = 0.5 * value
        return (
            n * (half * np.log(half) - gammaln(half))
            - (half + 1.0) * sum_log
            - half * sum_inv
            - prior_rate * (value - 2.0)
            + np.log(value - 2.0)
        )

    current_z = np.log(nu - 2.0)
    proposal_z = current_z + step * rng.standard_normal()
    proposal = 2.0 + float(np.exp(proposal_z))
    log_ratio = log_target(proposal) - log_target(nu)
    if np.log(rng.uniform()) < log_ratio:
        return proposal, True
    return nu, False


def _draw_structural_rows(
    resid: npt.NDArray[np.float64],
    a_mat: npt.NDArray[np.float64],
    log_variance: npt.NDArray[np.float64],
    *,
    prior_precision: float,
    rng: np.random.Generator,
) -> None:
    """One Gibbs sweep over the rows of a structural matrix, Waggoner-Zha (2003).

    The model is ``A u_t = eps_t`` with ``eps_it ~ N(0, exp(h_it))``, so the
    likelihood in ``A`` is ``|det A|^T`` times a Gaussian kernel whose
    precision for row ``i`` is ``S_i = sum_t exp(-h_it) u_t u_t'`` plus the
    prior. The determinant is linear in any one row given the others, and
    Waggoner and Zha's construction turns that into an exact draw: rotate
    row ``i`` into coordinates where the kernel is spherical, split off the
    one direction the determinant lives on, draw that coordinate from
    ``|b|^T exp(-b**2 / 2)`` (a signed square root of a gamma variate) and
    the rest as standard normals. The draw is exact, so the sweep is Gibbs
    rather than Metropolis, and the sign of the determinant coordinate is
    drawn at random: the posterior is symmetric under column sign flips,
    and the caller normalizes signs after the fact.

    Args:
        resid: ``(T, k)`` reduced-form innovations ``u_t``.
        a_mat: ``(k, k)`` current structural matrix ``A = B**-1``; updated
            in place, row by row.
        log_variance: ``(T, k)`` current log variances of the structural
            shocks.
        prior_precision: Precision of the ``N(0, 1 / prior_precision)``
            prior on each element of ``A``.
        rng: Random generator.

    Raises:
        NumericalError: If the current ``A`` is singular, so the cofactor
            direction is undefined.
    """
    nobs, k = resid.shape
    weights = np.exp(-log_variance)
    for i in range(k):
        kernel = (resid * weights[:, i][:, None]).T @ resid + prior_precision * np.eye(k)
        chol = np.linalg.cholesky(kernel)
        det = float(np.linalg.det(a_mat))
        if not np.isfinite(det) or det == 0.0:
            raise NumericalError("the structural matrix became singular during the row sweep.")
        cofactor = det * np.linalg.inv(a_mat)[:, i]
        direction = sla.solve_triangular(chol, cofactor, lower=True)
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise NumericalError("the determinant direction of a structural row vanished.")
        basis = np.linalg.qr(np.column_stack([direction / norm, np.eye(k)]))[0][:, :k]
        if float(basis[:, 0] @ direction) < 0.0:
            basis[:, 0] = -basis[:, 0]
        coordinates = np.asarray(rng.standard_normal(k), dtype=np.float64)
        magnitude = float(np.sqrt(rng.gamma(0.5 * (nobs + 1.0), 2.0)))
        coordinates[0] = magnitude if rng.random() < 0.5 else -magnitude
        rotated = basis @ coordinates
        a_mat[i] = sla.solve_triangular(chol, rotated, lower=True, trans="T")


def _mix_predictive_paths(
    paths: Sequence[npt.NDArray[np.float64]],
    weights: npt.NDArray[np.float64],
    *,
    n_draws: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Draw from a weighted mixture of simulated predictive paths.

    Each draw picks a model with probability given by its weight and then
    one of that model's paths uniformly, with replacement, so the output is
    an equally weighted sample from the mixture predictive.

    Args:
        paths: One ``(S_m, h, k)`` array per model, all sharing ``(h, k)``.
        weights: ``(M,)`` non-negative weights summing to one.
        n_draws: Draws to return.
        rng: Random generator.

    Returns:
        ``(n_draws, h, k)`` mixed paths.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> a = np.zeros((50, 2, 1))
        >>> b = np.ones((30, 2, 1))
        >>> mixed = _mix_predictive_paths([a, b], np.array([0.0, 1.0]), n_draws=10, rng=rng)
        >>> float(mixed.mean())
        1.0
    """
    counts = rng.multinomial(n_draws, weights)
    pieces = []
    for block, count in zip(paths, counts, strict=True):
        if count:
            pieces.append(block[rng.integers(0, block.shape[0], size=count)])
    mixed = np.concatenate(pieces, axis=0)
    return np.asarray(mixed[rng.permutation(n_draws)], dtype=np.float64)


def _draw_conditional_shocks(
    restriction: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    *,
    n_draws: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Standard normal shocks conditioned on ``R eps = r`` (Waggoner & Zha, 1999).

    The conditional law of ``eps ~ N(0, I)`` given the restrictions is
    Gaussian with mean ``R'(RR')^{-1} r`` -- the minimum-norm shock that
    delivers the conditions -- and covariance ``I - R'(RR')^{-1}R``, the
    projector onto the null space of ``R``; a draw is the mean plus the
    projection of a fresh standard normal vector.

    Args:
        restriction: ``(q, m)`` restriction matrix of full row rank.
        target: ``(q,)`` targets.
        n_draws: Draws.
        rng: Random generator.

    Returns:
        ``(shocks, minimum_norm)``: ``(n_draws, m)`` conditional draws and
        the ``(m,)`` minimum-norm shock.

    Raises:
        NumericalError: If the restrictions are linearly dependent.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> R, r = np.array([[1.0, 1.0]]), np.array([2.0])
        >>> draws, star = _draw_conditional_shocks(R, r, n_draws=5, rng=rng)
        >>> star, bool(np.allclose(draws.sum(axis=1), 2.0))
        (array([1., 1.]), True)
    """
    q, m = restriction.shape
    if q == 0:
        return rng.standard_normal((n_draws, m)), np.zeros(m)
    gram = restriction @ restriction.T
    try:
        solved = np.linalg.solve(gram, np.column_stack([target, restriction]))
    except np.linalg.LinAlgError as error:
        raise NumericalError(
            "the conditions are linearly dependent: some conditioned cell is implied by the "
            "others, or more cells are conditioned than the shocks can deliver."
        ) from error
    if float(np.linalg.cond(gram)) > 1e12:
        raise NumericalError(
            "the conditions are nearly dependent; the conditional shock is not determined."
        )
    minimum_norm = restriction.T @ solved[:, 0]
    projector = np.eye(m) - restriction.T @ solved[:, 1:]
    draws = minimum_norm[None, :] + rng.standard_normal((n_draws, m)) @ projector.T
    return np.asarray(draws, dtype=np.float64), np.asarray(minimum_norm, dtype=np.float64)
