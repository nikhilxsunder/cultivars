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

import numpy as np
import numpy.typing as npt
import scipy.special as ssp
import scipy.stats as sst

from ..exceptions import NumericalError
from ._defaults import _GIG_MAX_ROUNDS, _GIG_TINY, _KSC_MEAN, _KSC_PROB, _KSC_VAR


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
