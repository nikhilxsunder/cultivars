# filepath: /src/cultivars/_core/_estimators.py
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

from functools import lru_cache

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._defaults import _LOG_2PI, _PENALTY
from ._types import CointegrationTrend


def ols(
    design: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], float]:
    """Least-squares fit returning coefficients and the residual sum of squares.

    Uses ``lstsq`` rather than the normal equations so a rank-deficient design
    yields the minimum-norm solution instead of raising.

    Args:
        design: Regressor matrix of shape ``(n, k)``.
        target: Response vector of shape ``(n,)``.

    Returns:
        A tuple ``(beta, ssr)``.

    Raises:
        DimensionError: If the shapes are not conformable.

    Example:
        >>> beta, ssr = ols(np.ones((5, 1)), np.arange(5.0))
        >>> float(beta[0])
        2.0
    """
    if design.ndim != 2 or design.shape[0] != target.shape[0]:
        raise DimensionError(
            f"design {design.shape} is not conformable with target {target.shape}."
        )
    beta, _residuals, _rank, _sv = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    return np.asarray(beta, dtype=np.float64), float(resid @ resid)


def concentrated_gaussian(ssr: float, nobs: int) -> tuple[float, float]:
    """Concentrated Gaussian variance and log-likelihood from an SSR."""
    sigma2 = ssr / nobs
    return sigma2, -0.5 * nobs * (_LOG_2PI + np.log(sigma2) + 1.0)


def periodogram(
    y: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Positive Fourier frequencies and the periodogram, with the DC term dropped.

    The series is mean-centered first, which makes the zero frequency
    uninformative; it is discarded so that log-periodogram regressions are not
    anchored by a structurally zero ordinate.

    Args:
        y: The series, shape ``(n,)``.

    Returns:
        A tuple ``(freqs, ordinates)``, each of length ``floor(n / 2)``.

    Raises:
        SpecificationError: If the series has fewer than two observations.
    """
    n = y.shape[0]
    if n < 2:
        raise SpecificationError(f"periodogram requires at least 2 observations; got {n}.")
    transform = np.fft.rfft(y - y.mean())
    ordinates = (np.abs(transform) ** 2) / n
    freqs = 2.0 * np.pi * np.arange(transform.shape[0]) / n
    return freqs[1:], ordinates[1:]


def bandwidth(nobs: int, m: int | None, exponent: float) -> int:
    """Resolve the number of Fourier frequencies for a semiparametric estimator.

    Args:
        nobs: Series length.
        m: Explicit bandwidth, or ``None`` to derive it from ``exponent``.
        exponent: Exponent in the default rule ``m = floor(n ** exponent)``.

    Returns:
        The bandwidth, never below 2.

    Raises:
        SpecificationError: If an explicit ``m`` is below 2, or ``exponent``
            does not lie in ``(0, 1)``.

    Example:
        >>> bandwidth(400, None, 0.5)
        20
    """
    if m is not None:
        if m < 2:
            raise SpecificationError(f"bandwidth m must be >= 2; got {m}.")
        return int(m)
    if not (0.0 < exponent < 1.0):
        raise SpecificationError(f"bandwidth_exponent must lie in (0, 1); got {exponent}.")
    return max(2, int(np.floor(nobs**exponent)))


def local_whittle_d(
    y: npt.NDArray[np.float64], m: int | None = None, exponent: float = 0.65
) -> tuple[float, int]:
    """Local Whittle estimate of the fractional differencing parameter.

    Args:
        y: The series.
        m: Explicit bandwidth, or ``None`` for the default rule.
        exponent: Exponent in the default bandwidth rule.

    Returns:
        A tuple ``(d_hat, m_eff)``.
    """
    from scipy.optimize import minimize_scalar

    from ._defaults import _D_MAX

    freqs, ordinates = periodogram(y)
    m_eff = min(bandwidth(y.shape[0], m, exponent), freqs.shape[0])
    lam = freqs[:m_eff]
    power = ordinates[:m_eff]
    log_lam_mean = float(np.log(lam).mean())

    def objective(d: float) -> float:
        g = float(np.mean(lam ** (2.0 * d) * power))
        if not np.isfinite(g) or g <= 0.0:
            return 1e10
        return float(np.log(g) - 2.0 * d * log_lam_mean)

    result = minimize_scalar(objective, bounds=(-_D_MAX, _D_MAX), method="bounded")
    return float(result.x), m_eff


def gph_d(
    y: npt.NDArray[np.float64], m: int | None = None, exponent: float = 0.5
) -> tuple[float, float, int]:
    """Geweke-Porter-Hudak log-periodogram estimate of ``d``.

    Args:
        y: The series.
        m: Explicit bandwidth, or ``None`` for the default rule.
        exponent: Exponent in the default bandwidth rule.

    Returns:
        A tuple ``(d_hat, se, m_eff)``.

    Raises:
        NumericalError: If the periodogram has non-positive ordinates, or the
            regressor has zero variance.
    """
    from ..exceptions import NumericalError

    freqs, ordinates = periodogram(y)
    m_eff = min(bandwidth(y.shape[0], m, exponent), freqs.shape[0])
    lam = freqs[:m_eff]
    power = ordinates[:m_eff]
    if np.any(power <= 0.0):
        raise NumericalError("periodogram has non-positive ordinates; cannot take logs.")
    regressor = -2.0 * np.log(2.0 * np.sin(lam / 2.0))
    centered = regressor - regressor.mean()
    denom = float(centered @ centered)
    if denom <= 0.0:
        raise NumericalError("degenerate GPH regression (zero regressor variance).")
    response = np.log(power)
    d_hat = float(centered @ (response - response.mean()) / denom)
    se = float(np.pi / np.sqrt(24.0 * denom))
    return d_hat, se, m_eff


def ewma_mean_square(x: npt.NDArray[np.float64], *, decay: float = 0.94, window: int = 75) -> float:
    """Exponentially weighted pre-sample variance estimate.

    Args:
        x: Mean residuals.
        decay: Exponential decay factor.
        window: Number of observations to include in the weighted mean.

    Returns:
        The weighted mean squared residual over the first ``window`` observations of ``x``.
    """
    tau = min(window, x.shape[0])
    w = decay ** np.arange(tau)
    w /= w.sum()
    return float(np.sum(w * x[:tau] ** 2))


def ergodic_distribution(
    transition: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Stationary distribution of a row-stochastic Markov transition matrix.

    Solves ``pi' P = pi'`` subject to ``sum(pi) = 1`` as a constrained least
    squares problem, which stays well behaved when the chain is near-reducible
    and an eigenvector approach would return a near-degenerate solution.

    Args:
        transition: A ``(K, K)`` row-stochastic matrix.

    Returns:
        The ergodic probabilities, shape ``(K,)``.

    Raises:
        NumericalError: If no valid distribution can be recovered.

    Example:
        >>> np.round(ergodic_distribution(np.array([[0.9, 0.1], [0.2, 0.8]])), 4)
        array([0.6667, 0.3333])
    """
    k = transition.shape[0]
    augmented = np.vstack([transition.T - np.eye(k), np.ones((1, k))])
    rhs = np.zeros(k + 1)
    rhs[-1] = 1.0
    solution, _res, _rank, _sv = np.linalg.lstsq(augmented, rhs, rcond=None)
    pi = np.clip(np.asarray(solution, dtype=np.float64), 0.0, None)
    total = pi.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise NumericalError("transition matrix admits no valid ergodic distribution.")
    return np.asarray(pi / total, dtype=np.float64)


def _gaussian_negloglik(resid: npt.NDArray[np.float64], sigma2: npt.NDArray[np.float64]) -> float:
    """Negative Gaussian log-likelihood given a conditional-variance path.

    Args:
        resid: Mean residuals.
        sigma2: The conditional-variance path, same length as ``resid``.

    Returns:
        The negative log-likelihood, or :data:`_PENALTY` if the variance path
        is non-positive or non-finite anywhere, or the sum overflows.
    """
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
        return _PENALTY
    value = 0.5 * float(np.sum(_LOG_2PI + np.log(sigma2) + resid**2 / sigma2))
    return value if np.isfinite(value) else _PENALTY


def _null_functional(
    walk: npt.NDArray[np.float64], grid: npt.NDArray[np.float64], case: CointegrationTrend
) -> npt.NDArray[np.float64]:
    """Build the regressor process the limiting distribution is written against."""
    reps, steps, _ = walk.shape
    if case == "none":
        return walk
    if case == "restricted_constant":
        return np.concatenate([walk, np.ones((reps, steps, 1))], axis=2)
    centred = walk - walk.mean(axis=1, keepdims=True)
    if case == "constant":
        return centred
    if case == "restricted_trend":
        return np.concatenate([centred, np.broadcast_to(grid - 0.5, (reps, steps, 1))], axis=2)
    ramp = np.broadcast_to(grid, (reps, steps, 1))
    ramp = ramp - ramp.mean(axis=1, keepdims=True)
    slope = (centred * ramp).sum(axis=1, keepdims=True) / (ramp * ramp).sum(axis=1, keepdims=True)
    return centred - slope * ramp


@lru_cache(maxsize=128)
def simulate_cointegration_null(
    n: int,
    case: CointegrationTrend,
    *,
    n_exog: int = 0,
    simulations: int = 25_000,
    steps: int = 500,
    seed: int = 20260819,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Draw from the asymptotic null distribution of the Johansen rank statistics.

    Both statistics converge to functionals of an ``n``-dimensional Brownian
    motion, where ``n = k - r`` is the number of common trends under the null.
    The limit has no closed form and depends on which deterministic terms the
    specification carries, so it is obtained here by simulating the functional
    directly on a discretized path.

    Simulating rather than tabulating is a deliberate choice. The published
    route is MacKinnon, Haug and Michelis (1999), who fit a two-moment gamma
    response surface so that a printed table can be compressed to three
    coefficients per cell; the approximation is excellent to the upper decile
    and drifts by most of a point by the 99.5th percentile. Having the draws in
    hand removes the reason for that compression -- the empirical distribution
    is exact up to Monte Carlo error, extends to any ``n`` without a new table,
    and reports its own resolution through ``simulations``.

    Results are memoized on the full argument tuple, so a model that tests
    several ranks pays for each ``n`` once.

    A conditional specification shifts the distribution rather than leaving it
    alone. When ``n_exog`` weakly exogenous integrated regressors enter without
    equations of their own, they widen the process being projected onto while
    contributing no innovations of their own to project, so the statistic grows
    and the critical values with it -- at five percent and one modelled common
    trend, from 8.2 with no exogenous block to 11.3 with one. Reading a
    conditional statistic against the unconditional table would over-reject
    badly, which is why this is an argument rather than a footnote.

    Args:
        n: Number of *modelled* common trends under the null. For a closed
            system that is ``k - r``; for a conditional one it is ``k_y - r``,
            counting only the equations that were estimated.
        case: One of :data:`CointegrationTrend`.
        n_exog: Weakly exogenous integrated regressors carried without
            equations. Zero recovers the standard Johansen distribution.
        simulations: Replications. Tail resolution is ``1 / simulations``.
        steps: Discretization of the unit interval. Coarse grids bias the
            statistic downward; 500 places the five percent point within about
            a tenth of a unit of the published value.
        seed: Fixed so that a p-value is reproducible.

    Returns:
        Sorted trace and maximum-eigenvalue draws.

    Raises:
        SpecificationError: If ``n`` is not positive, ``n_exog`` is negative,
            the case is unrecognized, or the simulation controls are not
            positive.
    """
    if n < 1:
        raise SpecificationError(f"n must be at least 1; got {n}.")
    if n_exog < 0:
        raise SpecificationError(f"n_exog must be non-negative; got {n_exog}.")
    if case not in CointegrationTrend.__value__:
        raise SpecificationError(
            f"case must be one of {CointegrationTrend.__value__}; got {case!r}."
        )
    if simulations < 1 or steps < 1:
        raise SpecificationError("simulations and steps must both be positive.")
    rng = np.random.default_rng(seed)
    trace = np.empty(simulations, dtype=np.float64)
    maximum = np.empty(simulations, dtype=np.float64)
    grid = (np.arange(1, steps + 1, dtype=np.float64) / steps)[:, None]
    width = n + n_exog
    chunk = max(1, min(2000, simulations))
    done = 0
    while done < simulations:
        size = min(chunk, simulations - done)
        increments = rng.standard_normal((size, steps, width)) / np.sqrt(steps)
        walk = np.cumsum(increments, axis=1)
        lagged = np.concatenate([np.zeros((size, 1, width)), walk[:, :-1]], axis=1)
        regressor = _null_functional(lagged, grid, case)
        cross = np.einsum("msi,msj->mij", increments[:, :, :n], regressor)
        gram = np.einsum("msi,msj->mij", regressor, regressor) / steps
        quad = cross @ np.linalg.solve(gram, np.swapaxes(cross, 1, 2))
        quad = (quad + np.swapaxes(quad, 1, 2)) / 2.0
        trace[done : done + size] = np.trace(quad, axis1=1, axis2=2)
        maximum[done : done + size] = np.linalg.eigvalsh(quad)[:, -1]
        done += size
    return np.sort(trace), np.sort(maximum)


def minnesota_scales(endog: npt.NDArray[np.float64], order: int) -> npt.NDArray[np.float64]:
    """Per-variable residual scale, from univariate autoregressions.

    The ``sigma_i`` every Minnesota variance is written against, and the choice
    of estimator here is structural rather than conventional. Taking the scales
    from univariate fits keeps the prior independent of the system it
    regularizes; taking them from an unrestricted vector autoregression would
    make the prior depend on an estimate that, for the large systems this prior
    exists to serve, cannot be computed without it. For ``N`` greater than
    ``T`` the unrestricted fit does not exist at all, which is precisely the
    case where the prior is doing the most work.

    Args:
        endog: The ``(nobs, k)`` sample.
        order: Autoregressive order for the univariate fits.

    Returns:
        One residual standard deviation per variable.

    Raises:
        DimensionError: If the sample is too short for the order.
    """
    rows, size = endog.shape
    if rows <= order + 1:
        raise DimensionError(
            f"a sample of {rows} rows cannot support univariate fits of order {order}."
        )
    out = np.empty(size, dtype=np.float64)
    for index in range(size):
        design = np.column_stack(
            [np.ones(rows - order)]
            + [endog[order - lag : rows - lag, index] for lag in range(1, order + 1)]
        )
        target = endog[order:, index]
        coef = np.linalg.lstsq(design, target, rcond=None)[0]
        resid = target - design @ coef
        out[index] = float(np.sqrt(resid @ resid / (resid.shape[0] - design.shape[1])))
    return out


def _cumulant_slices(
    whitened: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Third- and fourth-order cumulant slices of a whitened panel.

    For mutually independent unit-variance sources every one of these
    matrices is diagonal, and a rotation of the panel rotates them all
    congruently -- which is what turns independent-component analysis into
    joint diagonalization. The third-order slices carry skewness, the
    fourth-order slices carry excess kurtosis, and including both is what
    identifies a shock that is non-Gaussian through either.

    Args:
        whitened: The ``(nobs, k)`` panel, unit covariance by construction.

    Returns:
        ``k`` third-order slices ``M_i = E[z_i z z']`` followed by
        ``k (k + 1) / 2`` fourth-order cumulant slices ``Q_ij = E[z_i z_j z
        z'] - delta_ij I - E_ij - E_ji``, each symmetrized.
    """
    nobs, k = whitened.shape
    identity = np.eye(k)
    out: list[npt.NDArray[np.float64]] = []
    for i in range(k):
        slice_third = (whitened * whitened[:, i : i + 1]).T @ whitened / nobs
        out.append((slice_third + slice_third.T) / 2.0)
    for i in range(k):
        for j in range(i, k):
            weight = whitened[:, i] * whitened[:, j]
            raw = (whitened * weight[:, np.newaxis]).T @ whitened / nobs
            correction = np.zeros((k, k))
            if i == j:
                correction += identity
            correction[i, j] += 1.0
            correction[j, i] += 1.0
            slice_fourth = raw - correction
            out.append((slice_fourth + slice_fourth.T) / 2.0)
    return tuple(out)
