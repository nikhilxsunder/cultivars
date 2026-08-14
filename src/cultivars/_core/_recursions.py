# filepath: /src/cultivars/_core/_recursions.py
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

import numpy as np
import numpy.typing as npt

from ._defaults import _SQRT_2_OVER_PI
from ._transforms import fractional_difference_weights


def _linear_variance_recursion(
    resid: npt.NDArray[np.float64],
    omega: float,
    alpha: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    backcast: float,
) -> npt.NDArray[np.float64]:
    """Linear recursion in past squared residuals and past variances.

    Covers the symmetric and the sign-asymmetric case in one pass: the
    asymmetry block is inert when ``gamma`` is empty. Pre-sample squared
    residuals and variances are replaced by ``backcast``; the asymmetry term
    uses half the backcast, matching the unconditional probability of a
    negative shock.

    Args:
        resid: Mean residuals.
        omega: Variance intercept.
        alpha: Coefficients on past squared residuals.
        gamma: Coefficients on past squared residuals conditional on a negative
            shock; empty for the symmetric case.
        beta: Coefficients on past variances.
        backcast: Pre-sample variance.

    Returns:
        The conditional-variance path, in levels.
    """
    n, p, o, q = resid.shape[0], alpha.size, gamma.size, beta.size
    sigma2 = np.empty(n)
    r2 = resid**2
    neg = (resid < 0.0).astype(np.float64)
    for t in range(n):
        s = omega
        for i in range(p):
            s += alpha[i] * (r2[t - 1 - i] if t - 1 - i >= 0 else backcast)
        for k in range(o):
            if t - 1 - k >= 0:
                s += gamma[k] * r2[t - 1 - k] * neg[t - 1 - k]
            else:
                s += gamma[k] * backcast * 0.5
        for j in range(q):
            s += beta[j] * (sigma2[t - 1 - j] if t - 1 - j >= 0 else backcast)
        sigma2[t] = s
    return sigma2


def _log_variance_recursion(
    resid: npt.NDArray[np.float64],
    omega: float,
    alpha: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    backcast: float,
) -> npt.NDArray[np.float64]:
    """Linear recursion in the log variance, driven by standardized residuals.

    Because the recursion is in logs the variance path is positive by
    construction, so no positivity constraint is needed on the coefficients.
    The magnitude term is centered by ``E|z| = sqrt(2 / pi)`` so a
    standard-normal shock contributes zero drift.

    Args:
        resid: Mean residuals.
        omega: Log-variance intercept.
        alpha: Coefficients on the centered absolute standardized residual.
        gamma: Coefficients on the signed standardized residual.
        beta: Coefficients on the past log variance.
        backcast: Pre-sample variance, entered in logs.

    Returns:
        The conditional-variance path, exponentiated back to levels.
    """
    n, p, o, q = resid.shape[0], alpha.size, gamma.size, beta.size
    ln_sigma2 = np.empty(n)
    ln_backcast = float(np.log(backcast))
    e = np.zeros(n)
    for t in range(n):
        s = omega
        for i in range(p):
            s += alpha[i] * ((abs(e[t - 1 - i]) - _SQRT_2_OVER_PI) if t - 1 - i >= 0 else 0.0)
        for k in range(o):
            s += gamma[k] * (e[t - 1 - k] if t - 1 - k >= 0 else 0.0)
        for j in range(q):
            s += beta[j] * (ln_sigma2[t - 1 - j] if t - 1 - j >= 0 else ln_backcast)
        ln_sigma2[t] = s
        sigma = np.sqrt(np.exp(ln_sigma2[t]))
        e[t] = resid[t] / sigma if sigma > 0 else 0.0
    return np.exp(ln_sigma2)


def _arch_infinity_weights(
    phi: float, d: float, beta: float, truncation: int
) -> npt.NDArray[np.float64]:
    """Weights of the infinite-order representation of a fractional variance filter.

    Uses the Chung (1999) recursion, which is better behaved numerically than
    expanding the fractional operator and dividing polynomials.

    Args:
        phi: The short-memory numerator weight.
        d: Fractional integration order.
        beta: The denominator weight.
        truncation: Number of weights to generate.

    Returns:
        The weights ``lambda_1, ..., lambda_truncation``.

    References:
        Chung, C.-F. (1999). Estimating the fractionally integrated GARCH model.
    """
    delta = -fractional_difference_weights(d, truncation + 1)[1:]
    lam = np.empty(truncation, dtype=np.float64)
    lam[0] = phi - beta + d
    for i in range(1, truncation):
        lam[i] = beta * lam[i - 1] + (delta[i] - phi * delta[i - 1])
    return lam


def _arch_infinity_variance(
    resid: npt.NDArray[np.float64],
    omega: float,
    phi: float,
    d: float,
    beta: float,
    backcast: float,
    truncation: int,
) -> npt.NDArray[np.float64]:
    """Variance path from the infinite-order representation, truncated.

    Weights beyond the available history are applied to ``backcast``, so the
    truncation tail contributes a constant rather than being silently dropped.

    Args:
        resid: Mean residuals.
        omega: Variance intercept.
        phi: The short-memory numerator weight.
        d: Fractional integration order.
        beta: The denominator weight.
        backcast: Pre-sample variance.
        truncation: Number of weights retained.

    Returns:
        The conditional-variance path.
    """
    n = resid.shape[0]
    lam = _arch_infinity_weights(phi, d, beta, truncation)
    r2 = resid**2
    sigma2 = np.empty(n)
    intercept = omega / (1.0 - beta)
    for t in range(n):
        m = min(t, truncation)
        acc = intercept
        if m > 0:
            acc += float(np.dot(lam[:m], r2[t - 1 :: -1][:m]))
        if truncation > m:
            acc += backcast * float(lam[m:truncation].sum())
        sigma2[t] = acc
    return sigma2
