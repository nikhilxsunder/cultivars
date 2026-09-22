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

from collections.abc import Callable, Sequence
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from scipy.stats import chi2, invwishart, norm
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._converters import _as_chains
from ._defaults import (
    _BRIDGE_MAX_ITER,
    _BRIDGE_TOL,
    _CRITICAL_LEVELS,
    _LOG_2PI,
    _MHM_TAU,
    _MIN_CHAIN_DRAWS,
    _PENALTY,
)
from ._mappings import (
    _DISCREPANCIES,
    _GLS_DETREND_C,
    _KPSS_CRITICAL,
    _MACKINNON_CRITICAL_2010,
    _MACKINNON_TAU_LARGE,
    _MACKINNON_TAU_MAX,
    _MACKINNON_TAU_MIN,
    _MACKINNON_TAU_SMALL,
    _MACKINNON_TAU_STAR,
)
from ._matrices import deterministic_columns, lag_matrix
from ._transforms import _rank_normalize, _split_chains, fractional_difference_weights
from ._types import CointegrationTrend, _FTest
from ._validators import _validate_posterior_draws, bandwidth


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


def local_whittle_d(
    y: npt.NDArray[np.float64],
    m: int | None = None,
    exponent: float = 0.65,
    *,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, int]:
    """Local Whittle estimate of the fractional differencing parameter.

    Args:
        y: The series.
        m: Explicit bandwidth, or ``None`` for the default rule.
        exponent: Exponent in the default bandwidth rule.
        bounds: Search interval; ``None`` confines it to the stationary
            range ``(-_D_MAX, _D_MAX)`` an ARFIMA start value needs.

    Returns:
        A tuple ``(d_hat, m_eff)``.
    """
    from scipy.optimize import minimize_scalar

    from ._defaults import _D_MAX

    low, high = (-_D_MAX, _D_MAX) if bounds is None else bounds
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

    result = minimize_scalar(objective, bounds=(low, high), method="bounded")
    return float(result.x), m_eff


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


def principal_components(
    standardized: npt.NDArray[np.float64], count: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Principal-component scores, loadings, and variance shares of a panel.

    The normalization is the plain one: loadings are the orthonormal right
    singular vectors, scores are the panel's projection onto them, so
    ``standardized ~ scores @ loadings.T`` and the approximation is exact
    when ``count`` spans the panel. Factor models built on top are
    identified only up to rotation, and consumers say so rather than
    pretending the basis is unique.

    Args:
        standardized: The ``(nobs, n_series)`` panel, already standardized.
        count: Components to keep, at least one.

    Returns:
        ``(scores, loadings, shares)``: the ``(nobs, count)`` scores, the
        ``(n_series, count)`` orthonormal loadings, and the ``(count,)``
        explained-variance shares.
    """
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    shares = singular**2 / float(np.sum(singular**2))
    loadings = vt[:count].T
    return standardized @ loadings, loadings, shares[:count]


def _variance_ratio_test(
    shocks: npt.NDArray[np.float64], *, n_blocks: int = 10, window: int = 9
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Test, pair by pair, whether two shocks' variance ratio moves over time.

    The identifying condition of volatility-based schemes is that the
    shocks' variance paths are not proportional. Whatever the common path
    does, under proportionality the *ratio* of two shocks' variances is
    constant, so its block-by-block log estimate scatters only by
    sampling noise. The statistic is the precision-weighted dispersion of
    the block log-ratios, with each block's precision ``n_eff / 4`` read
    off a smoothed proxy of the pair's common variance so that a
    volatile common path does not masquerade as a moving ratio; under the
    null it is chi-squared with ``n_blocks - 1`` degrees of freedom.
    Calibration on Gaussian and proportional-SV draws puts the size at
    the nominal level, and the direction of any miscalibration is toward
    calling identification weak rather than strong.

    Args:
        shocks: ``(T, k)`` structural shocks.
        n_blocks: Contiguous sample blocks; at least 3.
        window: Length of the centered moving average used as the
            common-variance proxy; odd and at least 1.

    Returns:
        ``(statistics, p_values)``, each ``(k, k)`` symmetric with zeros
        (statistics) and ones (p-values) on the diagonal.

    Raises:
        SpecificationError: If the block or window settings are malformed
            or the sample is too short for them.
    """
    eps = np.asarray(shocks, dtype=np.float64)
    if eps.ndim != 2:
        raise SpecificationError(f"shocks must be (T, k); got shape {eps.shape}.")
    nobs, k = eps.shape
    if n_blocks < 3:
        raise SpecificationError(f"n_blocks must be at least 3; got {n_blocks}.")
    if window < 1 or window % 2 == 0:
        raise SpecificationError(f"window must be odd and at least 1; got {window}.")
    if nobs < 5 * n_blocks:
        raise SpecificationError(
            f"{nobs} observations cannot support {n_blocks} blocks; use fewer blocks."
        )
    edges = np.linspace(0, nobs, n_blocks + 1).astype(int)
    kernel = np.ones(window) / window
    statistics = np.zeros((k, k))
    p_values = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            common = np.convolve(eps[:, i] ** 2 + eps[:, j] ** 2, kernel, mode="same")
            ratios = np.empty(n_blocks)
            precisions = np.empty(n_blocks)
            for b in range(n_blocks):
                block = slice(edges[b], edges[b + 1])
                weight = common[block]
                ratios[b] = np.log(
                    float(np.mean(eps[block, i] ** 2)) / float(np.mean(eps[block, j] ** 2))
                )
                precisions[b] = float(weight.sum()) ** 2 / float((weight**2).sum()) / 4.0
            center = float((precisions * ratios).sum() / precisions.sum())
            stat = float((precisions * (ratios - center) ** 2).sum())
            statistics[i, j] = statistics[j, i] = stat
            p_values[i, j] = p_values[j, i] = float(chi2.sf(stat, n_blocks - 1))
    return statistics, p_values


def _gaussian_envelope(
    theta: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Mean, Cholesky factor of the covariance, and its log determinant."""
    mean = theta.mean(axis=0)
    cov = np.atleast_2d(np.cov(theta, rowvar=False))
    cov[np.diag_indices_from(cov)] += 1e-12 * max(float(np.trace(cov)), 1.0)
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as error:
        raise NumericalError("The posterior draws have a singular covariance.") from error
    log_det = 2.0 * float(np.log(np.diag(chol)).sum())
    return mean, chol, log_det


def _modified_harmonic_mean(
    draws: npt.ArrayLike, log_kernel: npt.ArrayLike, *, tau: float = _MHM_TAU
) -> tuple[float, float, float]:
    """Geweke's truncated-Gaussian harmonic-mean estimate of the log evidence.

    Args:
        draws: ``(S, d)`` posterior draws.
        log_kernel: ``(S,)`` unnormalized log posterior at each draw.
        tau: Probability mass of the Gaussian envelope retained; draws
            outside the ``tau`` ellipsoid receive zero weight.

    Returns:
        ``(log_evidence, mcse, coverage)`` where ``coverage`` is the share
        of draws inside the truncation region -- a value far from ``tau``
        says the posterior is not the Gaussian the envelope assumes.

    Raises:
        SpecificationError: If ``tau`` is not in ``(0, 1)`` or no draw
            falls inside the region.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> z = rng.standard_normal((4000, 2))
        >>> kernel = -0.5 * (z**2).sum(axis=1)
        >>> value, error, coverage = _modified_harmonic_mean(z, kernel)
        >>> round(value, 1)
        1.8
    """
    if not 0.0 < tau < 1.0:
        raise SpecificationError(f"tau must lie in (0, 1); got {tau}.")
    theta, values = _validate_posterior_draws(draws, log_kernel)
    d = theta.shape[1]
    mean, chol, log_det = _gaussian_envelope(theta)
    whitened = np.linalg.solve(chol, (theta - mean).T).T
    mahalanobis = (whitened**2).sum(axis=1)
    inside = mahalanobis <= chi2.ppf(tau, d)
    coverage = float(inside.mean())
    if not inside.any():
        raise SpecificationError("No draw falls inside the truncation ellipsoid.")
    log_weight = -0.5 * (d * _LOG_2PI + log_det + mahalanobis[inside]) - np.log(tau)
    log_terms = np.full(theta.shape[0], -np.inf)
    log_terms[inside] = log_weight - values[inside]
    shift = float(log_terms[inside].max())
    ratios = np.exp(log_terms - shift)
    mean_ratio = float(ratios.mean())
    log_inverse = shift + float(np.log(mean_ratio))
    ess = _ess_mean(ratios) if ratios.shape[0] >= 8 else float("nan")
    error = (
        float("nan")
        if np.isnan(ess) or not ess > 0.0
        else float(ratios.std(ddof=1) / (mean_ratio * np.sqrt(ess)))
    )
    return -log_inverse, error, coverage


def _bridge_sampling(
    draws: npt.ArrayLike,
    log_kernel: npt.ArrayLike,
    kernel: Callable[[npt.NDArray[np.float64]], float],
    *,
    rng: np.random.Generator,
    n_proposal: int | None = None,
    max_iter: int = _BRIDGE_MAX_ITER,
    tol: float = _BRIDGE_TOL,
) -> tuple[float, float, int]:
    """Meng-Wong bridge-sampling estimate of the log evidence.

    The draws are split in half: the first half fits the Gaussian
    proposal, the second half enters the bridge, so the proposal is not
    tuned to the same draws it is bridged against (Gronau et al., 2017).

    Args:
        draws: ``(S, d)`` posterior draws.
        log_kernel: ``(S,)`` unnormalized log posterior at each draw.
        kernel: The same unnormalized log posterior as a function of one
            ``(d,)`` point, evaluated at fresh proposal draws.
        rng: Generator for the proposal draws.
        n_proposal: Proposal draws; defaults to the number of posterior
            draws entering the bridge.
        max_iter: Iterations of the fixed-point recursion.
        tol: Relative change in the estimate at which to stop.

    Returns:
        ``(log_evidence, mcse, n_iter)``. The error is the Frühwirth-
        Schnatter (2004) approximation, which treats the posterior half as
        independent draws scaled by their effective sample size.

    Raises:
        NumericalError: If the recursion fails to converge or the kernel
            returns non-finite values at the proposal points.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> z = rng.standard_normal((4000, 2))
        >>> logk = lambda x: -0.5 * float(x @ x)
        >>> value, error, n_iter = _bridge_sampling(
        ...     z, -0.5 * (z**2).sum(axis=1), logk, rng=rng
        ... )
        >>> round(value, 1)
        1.8
    """
    theta, values = _validate_posterior_draws(draws, log_kernel)
    half = theta.shape[0] // 2
    fit_half, bridge_half, bridge_values = theta[:half], theta[half:], values[half:]
    d = theta.shape[1]
    mean, chol, log_det = _gaussian_envelope(fit_half)
    n_prop = bridge_half.shape[0] if n_proposal is None else int(n_proposal)
    if n_prop < 8:
        raise SpecificationError(f"n_proposal must be at least 8; got {n_prop}.")
    proposal = mean + (chol @ rng.standard_normal((d, n_prop))).T

    def log_proposal(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        whitened = np.linalg.solve(chol, (points - mean).T).T
        return -0.5 * (d * _LOG_2PI + log_det + (whitened**2).sum(axis=1))

    log_q_post = log_proposal(bridge_half)
    log_q_prop = log_proposal(proposal)
    log_p_prop = np.array([float(kernel(point)) for point in proposal])
    if not np.all(np.isfinite(log_p_prop)):
        raise NumericalError("The kernel returned non-finite values at proposal points.")
    n2, n1 = bridge_half.shape[0], n_prop
    s1, s2 = n1 / (n1 + n2), n2 / (n1 + n2)
    # Work with l = log p - log q throughout; r = log evidence estimate.
    l_post = bridge_values - log_q_post
    l_prop = log_p_prop - log_q_prop
    log_r = float(np.median(l_post))
    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        numerator = logsumexp(l_prop - np.logaddexp(np.log(s1) + l_prop, np.log(s2) + log_r))
        denominator = logsumexp(-np.logaddexp(np.log(s1) + l_post, np.log(s2) + log_r))
        new = float(numerator - np.log(n1) - denominator + np.log(n2))
        if abs(new - log_r) <= tol * max(1.0, abs(new)):
            log_r = new
            break
        log_r = new
    else:
        raise NumericalError(f"Bridge sampling did not converge in {max_iter} iterations.")
    # Frühwirth-Schnatter (2004) relative variance approximation.
    f1 = np.exp(l_prop - np.logaddexp(np.log(s1) + l_prop, np.log(s2) + log_r))
    f2 = np.exp(-np.logaddexp(np.log(s1) + l_post, np.log(s2) + log_r) + log_r)
    ess2 = _ess_mean(f2) if f2.shape[0] >= 8 else float(f2.shape[0])
    ess2 = float(f2.shape[0]) if np.isnan(ess2) or not ess2 > 0.0 else ess2
    rel_var = float(
        f1.var(ddof=1) / (n1 * f1.mean() ** 2) + f2.var(ddof=1) / (ess2 * f2.mean() ** 2)
    )
    return log_r, float(np.sqrt(max(rel_var, 0.0))), n_iter


def _chib_independent_normal_wishart(
    target: npt.NDArray[np.float64],
    design: npt.NDArray[np.float64],
    beta_draws: npt.NDArray[np.float64],
    sigma_draws: npt.NDArray[np.float64],
    *,
    prior_mean: npt.NDArray[np.float64],
    prior_variance: npt.NDArray[np.float64],
    prior_scale: npt.NDArray[np.float64],
    prior_df: float,
) -> tuple[float, float]:
    """Chib's log marginal likelihood for the independent Normal-Wishart VAR.

    Two blocks, no reduced runs: ``pi(Sigma* | y)`` is the average over the
    main run's coefficient draws of the inverse-Wishart conditional, and
    ``pi(beta* | Sigma*, y)`` is a Gaussian density evaluated exactly, both
    at the posterior mean. The only Monte Carlo error is in the first
    average.

    Args:
        target: ``(n, k)`` sample rows of the response.
        design: ``(n, w)`` regressor rows.
        beta_draws: ``(S, w, k)`` kept coefficient draws.
        sigma_draws: ``(S, k, k)`` kept covariance draws.
        prior_mean: ``(w, k)`` prior coefficient means.
        prior_variance: ``(w, k)`` prior coefficient variances.
        prior_scale: ``(k, k)`` inverse-Wishart scale.
        prior_df: Inverse-Wishart degrees of freedom.

    Returns:
        ``(log_marginal_likelihood, mcse)``.

    Raises:
        NumericalError: If a conditional loses positive definiteness at the
            evaluation point.
    """
    n, k = target.shape
    width = design.shape[1]
    beta_star = beta_draws.mean(axis=0)
    sigma_star = sigma_draws.mean(axis=0)
    sigma_star = 0.5 * (sigma_star + sigma_star.T)
    try:
        chol_star = np.linalg.cholesky(sigma_star)
    except np.linalg.LinAlgError as error:
        raise NumericalError("The posterior mean covariance is not positive definite.") from error
    log_det_star = 2.0 * float(np.log(np.diag(chol_star)).sum())
    sigma_star_inv = np.linalg.inv(sigma_star)
    # Likelihood at the point.
    resid = target - design @ beta_star
    loglik = -0.5 * n * (k * _LOG_2PI + log_det_star) - 0.5 * float(
        np.trace(sigma_star_inv @ (resid.T @ resid))
    )
    # Prior ordinates.
    log_prior_beta = float(
        np.sum(norm.logpdf(beta_star, loc=prior_mean, scale=np.sqrt(prior_variance)))
    )
    log_prior_sigma = float(invwishart.logpdf(sigma_star, df=prior_df, scale=prior_scale))
    # pi(Sigma* | y): average of the inverse-Wishart conditional over beta draws.
    log_terms = np.empty(beta_draws.shape[0])
    for s, beta in enumerate(beta_draws):
        resid_s = target - design @ beta
        log_terms[s] = invwishart.logpdf(
            sigma_star, df=prior_df + n, scale=prior_scale + resid_s.T @ resid_s
        )
    log_sigma_ordinate, mcse = _log_mean_mcse(log_terms)
    # pi(beta* | Sigma*, y): exact Gaussian conditional.
    gram = design.T @ design
    moment = design.T @ target
    precision_vector = (1.0 / prior_variance).T.ravel()
    big = np.kron(sigma_star_inv, gram)
    big[np.diag_indices_from(big)] += precision_vector
    rhs = (sigma_star_inv @ moment.T).ravel() + precision_vector * prior_mean.T.ravel()
    try:
        chol_big = np.linalg.cholesky(big)
    except np.linalg.LinAlgError as error:
        raise NumericalError("The coefficient conditional is not positive definite.") from error
    conditional_mean = np.linalg.solve(big, rhs)
    deviation = beta_star.T.ravel() - conditional_mean
    quad = float(deviation @ (big @ deviation))
    log_det_big = 2.0 * float(np.log(np.diag(chol_big)).sum())
    log_beta_ordinate = 0.5 * log_det_big - 0.5 * k * width * _LOG_2PI - 0.5 * quad
    log_ml = loglik + log_prior_beta + log_prior_sigma - log_sigma_ordinate - log_beta_ordinate
    return float(log_ml), float(mcse)


def _autocovariance(chains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Biased sample autocovariance of every chain, all lags, via the FFT.

    Example:
        >>> acov = _autocovariance(np.array([[1.0, -1.0, 1.0, -1.0]]))
        >>> [round(float(v), 2) for v in acov[0]]
        [1.0, -0.75, 0.5, -0.25]
    """
    n_draws = chains.shape[1]
    n_fft = 1 << int(2 * n_draws - 1).bit_length()
    centered = chains - chains.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, n=n_fft, axis=1)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft, axis=1)[:, :n_draws]
    return np.asarray(acov / n_draws, dtype=np.float64)


def _potential_scale_reduction(chains: npt.NDArray[np.float64]) -> float:
    """Classical R-hat over already split (and possibly transformed) chains.

    ``sqrt(var_hat / W)`` with ``var_hat`` the weighted average of the
    within-chain variance ``W`` and the between-chain variance ``B``.
    Returns ``nan`` when the chains carry no within-chain variance.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> round(_potential_scale_reduction(rng.standard_normal((4, 500))), 1)
        1.0
    """
    n_chains, n_draws = chains.shape
    within = float(chains.var(axis=1, ddof=1).mean())
    if not within > 0.0:
        return float("nan")
    between = float(n_draws * chains.mean(axis=1).var(ddof=1)) if n_chains > 1 else 0.0
    var_hat = (n_draws - 1) / n_draws * within + between / n_draws
    return float(np.sqrt(var_hat / within))


def _effective_sample_size(chains: npt.NDArray[np.float64]) -> float:
    """Geyer initial-monotone-sequence effective sample size of ``(M, N)`` chains.

    The pooled autocorrelation at each lag is ``1 - (W - mean acov) /
    var_hat``; consecutive lags are summed in pairs, the sequence is
    truncated at the first non-positive pair (initial positive sequence)
    and then forced non-increasing (initial monotone sequence), and the
    integrated autocorrelation time is ``-1 + 2 * sum of the pairs``. The
    estimate is capped at ``M N log10(M N)``, since an antithetic chain can
    otherwise report more effective draws than a bound the estimator's
    precision supports. Returns ``nan`` for chains without variance.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> ess = _effective_sample_size(rng.standard_normal((2, 1000)))
        >>> 1400 < ess < 2600
        True
    """
    n_chains, n_draws = chains.shape
    acov = _autocovariance(chains)
    chain_var = acov[:, 0] * n_draws / (n_draws - 1)
    within = float(chain_var.mean())
    var_hat = within * (n_draws - 1) / n_draws
    if n_chains > 1:
        var_hat += float(chains.mean(axis=1).var(ddof=1))
    if not var_hat > 0.0:
        return float("nan")
    rho = 1.0 - (within - acov.mean(axis=0)) / var_hat
    n_pairs = n_draws // 2
    pairs = rho[: 2 * n_pairs].reshape(n_pairs, 2).sum(axis=1)
    negative = np.flatnonzero(pairs <= 0.0)
    cutoff = int(negative[0]) if negative.size else n_pairs
    tau = 1.0 if cutoff == 0 else -1.0 + 2.0 * float(np.minimum.accumulate(pairs[:cutoff]).sum())
    total = n_chains * n_draws
    ess = total / tau if tau > 0.0 else float(total)
    return float(min(ess, total * np.log10(total)))


def _rhat(draws: npt.ArrayLike) -> float:
    """Rank-normalized split-R-hat with folding.

    Args:
        draws: ``(N,)`` or ``(C, N)`` kept draws of one quantity.

    Returns:
        The larger of the rank-normalized R-hat and the folded rank-normalized
        R-hat; ``nan`` for a constant chain.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> mixed = rng.standard_normal((2, 400))
        >>> rhat(mixed) < 1.02
        True
        >>> apart = np.stack([mixed[0], mixed[1] + 5.0])
        >>> rhat(apart) > 1.5
        True
    """
    chains = _as_chains(draws)
    split = _split_chains(chains)
    plain = _potential_scale_reduction(_rank_normalize(split))
    folded = _potential_scale_reduction(_rank_normalize(np.abs(split - np.median(chains))))
    if np.isnan(plain) or np.isnan(folded):
        return float("nan")
    return max(plain, folded)


def _ess_bulk(draws: npt.ArrayLike) -> float:
    """Bulk effective sample size on the rank-normalized split chains.

    Args:
        draws: ``(N,)`` or ``(C, N)`` kept draws of one quantity.

    Returns:
        Effective draws for estimating central posterior quantities; ``nan``
        for a constant chain.

    Example:
        >>> rng = np.random.default_rng(1)
        >>> x = rng.standard_normal(2000)
        >>> 1400 < ess_bulk(x) < 2600
        True
        >>> sticky = np.repeat(x[:200], 10)
        >>> ess_bulk(sticky) < 400
        True
    """
    return _effective_sample_size(_rank_normalize(_split_chains(_as_chains(draws))))


def _ess_tail(draws: npt.ArrayLike) -> float:
    """Tail effective sample size: the lesser of the 5% and 95% quantile sizes.

    Each is the effective sample size of the indicator that a draw lies
    below the corresponding pooled quantile, which is the quantity whose
    Monte Carlo error governs a credible-interval endpoint.

    Args:
        draws: ``(N,)`` or ``(C, N)`` kept draws of one quantity.

    Returns:
        Effective draws for the interval endpoints; ``nan`` for a constant
        chain.

    Example:
        >>> rng = np.random.default_rng(2)
        >>> 900 < ess_tail(rng.standard_normal(2000)) < 2600
        True
    """
    chains = _as_chains(draws)
    lower, upper = np.quantile(chains, [0.05, 0.95])
    sizes = [
        _effective_sample_size(_split_chains((chains <= q).astype(np.float64)))
        for q in (lower, upper)
    ]
    if any(np.isnan(s) for s in sizes):
        return float("nan")
    return float(min(sizes))


def _ess_mean(draws: npt.ArrayLike) -> float:
    """Effective sample size for the posterior mean, on the raw split chains.

    This is the untransformed size that enters the Monte Carlo standard
    error of the mean; the rank-normalized bulk size is the one to read for
    a convergence verdict.

    Example:
        >>> rng = np.random.default_rng(3)
        >>> 1400 < ess_mean(rng.standard_normal(2000)) < 2600
        True
    """
    return _effective_sample_size(_split_chains(_as_chains(draws)))


def _mcse_mean(draws: npt.ArrayLike) -> float:
    """Monte Carlo standard error of the posterior mean.

    ``sd / sqrt(ess_mean)``, with the standard deviation pooled across
    chains.

    Example:
        >>> rng = np.random.default_rng(4)
        >>> round(mcse_mean(rng.standard_normal(10_000)), 1)
        0.0
    """
    chains = _as_chains(draws)
    ess = _ess_mean(chains)
    if np.isnan(ess):
        return float("nan")
    return float(chains.std(ddof=1) / np.sqrt(ess))


def _geweke(
    draws: npt.ArrayLike, *, first: float = 0.1, last: float = 0.5
) -> npt.NDArray[np.float64]:
    """Geweke's early-versus-late mean-difference score, one per chain.

    The long-run variance of each segment is its sample variance times the
    ratio of its length to its effective sample size, so no spectral window
    is tuned separately.

    Args:
        draws: ``(N,)`` or ``(C, N)`` kept draws of one quantity.
        first: Fraction of each chain forming the early segment.
        last: Fraction of each chain forming the late segment.

    Returns:
        ``(C,)`` z-scores, ``nan`` where a segment is constant.

    Raises:
        SpecificationError: If the fractions are not positive, overlap, or
            leave a segment too short to estimate an autocorrelation.

    Example:
        >>> rng = np.random.default_rng(5)
        >>> z = geweke(rng.standard_normal((3, 1000)))
        >>> z.shape
        (3,)
        >>> bool(np.all(np.abs(z) < 4.0))
        True
    """
    if not (first > 0.0 and last > 0.0 and first + last <= 1.0):
        raise SpecificationError(
            f"first and last must be positive fractions summing to at most 1; got {first}, {last}."
        )
    chains = _as_chains(draws)
    n_draws = chains.shape[1]
    n_first = int(first * n_draws)
    n_last = int(last * n_draws)
    if min(n_first, n_last) < _MIN_CHAIN_DRAWS:
        raise SpecificationError(
            f"Geweke segments need at least {_MIN_CHAIN_DRAWS} draws each; got {n_first} "
            f"and {n_last} from {n_draws} draws."
        )
    scores = np.empty(chains.shape[0], dtype=np.float64)
    for c, chain in enumerate(chains):
        early = chain[:n_first]
        late = chain[n_draws - n_last :]
        variances = []
        for segment in (early, late):
            ess = _effective_sample_size(segment[None, :])
            variances.append(float("nan") if np.isnan(ess) else segment.var(ddof=1) / ess)
        denominator = float(np.sqrt(variances[0] + variances[1]))
        scores[c] = (
            float("nan") if not denominator > 0.0 else (early.mean() - late.mean()) / denominator
        )
    return scores


def _log_mean_mcse(log_terms: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Log of the mean of ``exp(log_terms)`` and its Monte Carlo standard error.

    The error uses the effective sample size of the term sequence, so a
    correlated chain of terms is not reported as if independent.

    Example:
        >>> value, error = _log_mean_mcse(np.log(np.full(1000, 2.0)))
        >>> round(value, 6), error
        (0.693147, 0.0)
    """
    n = log_terms.shape[0]
    shift = float(log_terms.max())
    ratios = np.exp(log_terms - shift)
    mean = float(ratios.mean())
    log_mean = shift + float(np.log(mean))
    if n < 8:
        return log_mean, float("nan")
    ess = _ess_mean(ratios)
    if np.isnan(ess) or not ess > 0.0:
        return log_mean, 0.0
    error = float(ratios.std(ddof=1) / (mean * np.sqrt(ess)))
    return log_mean, error


def _stacking_weights(
    log_density: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float]:
    """Log-score-optimal simplex weights over models' held-out predictive densities.

    Maximizes ``sum_t log sum_m w_m exp(l_tm)`` over the simplex (Yao,
    Vehtari, Simpson, and Gelman, 2018): the weights of the mixture of
    predictive densities that would have scored best on the evaluation
    origins. The objective is concave in ``w``; it is optimized through the
    softmax map from ``M - 1`` free coordinates, whose stationary point is
    the simplex optimum.

    Args:
        log_density: ``(T, M)`` log predictive densities, one row per
            origin and one column per model.

    Returns:
        ``(weights, score)``: the ``(M,)`` weights and the maximized mean log
        score per origin.

    Raises:
        NumericalError: If the optimizer fails.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> better = -0.5 * rng.standard_normal(200) ** 2
        >>> worse = better - 2.0
        >>> weights, score = _stacking_weights(np.column_stack([better, worse]))
        >>> round(float(weights[0]), 2)
        1.0
    """
    n_origins, n_models = log_density.shape
    shift = log_density.max(axis=1, keepdims=True)
    scaled = np.exp(log_density - shift)

    def objective(free: npt.NDArray[np.float64]) -> float:
        z = np.concatenate([free, [0.0]])
        weights = np.exp(z - z.max())
        weights /= weights.sum()
        mixture = scaled @ weights
        return -float(np.sum(np.log(np.maximum(mixture, 1e-300))))

    solution = minimize(objective, np.zeros(n_models - 1), method="L-BFGS-B")
    if not solution.success and not np.isfinite(solution.fun):
        raise NumericalError(f"Stacking weight optimization failed: {solution.message}")
    z = np.concatenate([solution.x, [0.0]])
    weights = np.exp(z - z.max())
    weights /= weights.sum()
    score = float(-(solution.fun) / n_origins + shift.mean())
    return np.asarray(weights, dtype=np.float64), score


def _standardized(panel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Each column centered and scaled to unit variance; a constant column becomes ``nan``.

    Every shape statistic is computed on this scale so that a prior
    replication wandering to ``1e80`` does not overflow its fourth power.
    """
    centered = panel - panel.mean(axis=0)
    scale = np.sqrt((centered**2).mean(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.where(scale > 0.0, centered / scale, np.nan), dtype=np.float64)


def _lag_one_autocorrelation(panel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Lag-one autocorrelation of each column of an ``(n, k)`` panel."""
    z = _standardized(panel)
    return np.asarray((z[1:] * z[:-1]).sum(axis=0) / z.shape[0])


def _excess_kurtosis(panel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Excess kurtosis of each column of an ``(n, k)`` panel."""
    return np.asarray((_standardized(panel) ** 4).mean(axis=0) - 3.0)


def _skewness(panel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Skewness of each column of an ``(n, k)`` panel."""
    return np.asarray((_standardized(panel) ** 3).mean(axis=0))


def _discrepancy_statistics(
    panel: npt.NDArray[np.float64], statistics: Sequence[str]
) -> npt.NDArray[np.float64]:
    """Evaluate named discrepancy statistics on one ``(n, k)`` panel.

    Args:
        panel: The data set, one column per variable.
        statistics: Names from :data:`_DISCREPANCIES`.

    Returns:
        An ``(m, k)`` array, one row per statistic.

    Raises:
        SpecificationError: If a name is unknown.

    Example:
        >>> panel = np.column_stack([np.arange(6.0), np.ones(6)])
        >>> _discrepancy_statistics(panel, ["mean", "sd"])
        array([[2.5       , 1.        ],
               [1.87082869, 0.        ]])
    """
    unknown = [name for name in statistics if name not in _DISCREPANCIES]
    if unknown:
        raise SpecificationError(
            f"unknown discrepancy statistic(s) {unknown}; known: {', '.join(_DISCREPANCIES)}."
        )
    data = np.asarray(panel, dtype=np.float64)
    return np.stack([_DISCREPANCIES[name](data) for name in statistics])


def _predictive_pvalues(
    observed: npt.NDArray[np.float64], replicated: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Posterior predictive p-values ``P(T(y_rep) >= T(y))`` with a tie correction.

    Args:
        observed: ``(m, k)`` statistics of the data.
        replicated: ``(R, m, k)`` statistics of the replications.

    Returns:
        ``(m, k)`` tail probabilities, ties counted as one half so that a
        statistic the model reproduces exactly reads ``0.5`` rather than
        ``1.0``. A statistic undefined on some replication is ``nan``.

    Example:
        >>> rep = np.arange(10.0).reshape(10, 1, 1)
        >>> float(_predictive_pvalues(np.array([[7.0]]), rep)[0, 0])
        0.25
    """
    above = (replicated > observed[None]).mean(axis=0)
    ties = (replicated == observed[None]).mean(axis=0)
    out = np.asarray(above + 0.5 * ties, dtype=np.float64)
    undefined = ~np.all(np.isfinite(replicated), axis=0) | ~np.isfinite(observed)
    out[undefined] = np.nan
    return out


def _berkowitz_likelihood_ratio(transformed: npt.NDArray[np.float64]) -> tuple[float, int, float]:
    """Berkowitz's (2001) likelihood ratio on the normal-quantile transforms of PITs.

    The unrestricted model is a Gaussian AR(1) with free mean, slope, and
    variance, estimated by exact conditional maximum likelihood; the null
    restricts to zero mean, zero slope, unit variance -- what a calibrated,
    independent PIT series must look like on this scale.

    Args:
        transformed: ``(T,)`` normal quantiles of the PITs.

    Returns:
        ``(statistic, df, pvalue)`` with ``df = 3``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> stat, df, p = _berkowitz_likelihood_ratio(rng.standard_normal(500))
        >>> df, bool(p > 0.01)
        (3, True)
    """
    lagged = transformed[:-1]
    current = transformed[1:]
    count = current.shape[0]
    design = np.column_stack([np.ones(count), lagged])
    coefficients, *_ = np.linalg.lstsq(design, current, rcond=None)
    residual = current - design @ coefficients
    variance = max(float(residual @ residual) / count, 1e-12)
    llf_free = -0.5 * count * (np.log(2.0 * np.pi * variance) + 1.0)
    llf_null = -0.5 * count * np.log(2.0 * np.pi) - 0.5 * float(current @ current)
    statistic = max(2.0 * (llf_free - llf_null), 0.0)
    return float(statistic), 3, float(chi2.sf(statistic, 3))


def _bartlett_long_run_variance(centered: npt.NDArray[np.float64], horizon: int) -> float:
    """Long-run variance of a demeaned series with Bartlett weights through ``horizon - 1`` lags.

    The variance an ``h``-step forecast-error functional needs: mechanically
    an MA(``h - 1``), so autocovariances through lag ``h - 1`` enter, and
    the Bartlett taper keeps the estimate positive.

    Args:
        centered: ``(T,)`` series with its mean removed.
        horizon: The forecast horizon behind the series.

    Returns:
        The long-run variance, floored at a tiny positive number.

    Example:
        >>> z = np.array([1.0, -1.0, 1.0, -1.0])
        >>> _bartlett_long_run_variance(z, 1)
        1.0
    """
    count = centered.shape[0]
    variance = float(centered @ centered) / count
    for lag in range(1, horizon):
        weight = 1.0 - lag / horizon
        variance += 2.0 * weight * float(centered[lag:] @ centered[:-lag]) / count
    return max(variance, 1e-300)


def _clark_west(
    realized: npt.NDArray[np.float64],
    restricted: npt.NDArray[np.float64],
    unrestricted: npt.NDArray[np.float64],
    *,
    horizon: int,
) -> tuple[float, float, float]:
    """Clark and West's (2007) adjusted MSPE test of a nested forecast.

    Under the null that the restricted model is true, the unrestricted
    model's forecast carries estimation noise the restricted one does not,
    so its MSPE is *expected* to be larger; the adjustment removes that
    noise term, ``(f_r - f_u)**2``, and tests whether what remains --
    ``e_r**2 - e_u**2 + (f_r - f_u)**2`` -- has a positive mean, with the
    Bartlett long-run variance through ``horizon - 1`` lags and a one-sided
    standard normal reference.

    Args:
        realized: ``(T,)`` outcomes.
        restricted: ``(T,)`` forecasts of the nested (smaller) model.
        unrestricted: ``(T,)`` forecasts of the nesting (larger) model.
        horizon: The forecast horizon behind the series.

    Returns:
        ``(statistic, pvalue, adjusted_differential)``: the t-type
        statistic, its upper-tail p-value, and the mean of the adjusted
        loss differential.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = rng.standard_normal(200)
        >>> stat, p, _ = _clark_west(y, np.zeros(200), 0.05 * rng.standard_normal(200), horizon=1)
        >>> bool(p > 0.05)
        True
    """
    adjusted = (
        (realized - restricted) ** 2
        - (realized - unrestricted) ** 2
        + (restricted - unrestricted) ** 2
    )
    count = adjusted.shape[0]
    mean = float(adjusted.mean())
    variance = _bartlett_long_run_variance(adjusted - mean, horizon)
    statistic = mean / float(np.sqrt(variance / count))
    return statistic, float(norm.sf(statistic)), mean


def _stationary_bootstrap_indices(
    count: int, *, expected_block: float, rng: np.random.Generator
) -> npt.NDArray[np.intp]:
    """Politis and Romano's (1994) stationary bootstrap resampling indices.

    Blocks start at uniform positions and continue with probability
    ``1 - 1 / expected_block``, wrapping circularly, so that the resample
    keeps the dependence of a serially correlated series.

    Args:
        count: Length of the series and of the resample.
        expected_block: Mean block length, at least one.
        rng: Random generator.

    Returns:
        ``(count,)`` indices into the series.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> idx = _stationary_bootstrap_indices(10, expected_block=3.0, rng=rng)
        >>> idx.shape, bool(idx.min() >= 0 and idx.max() < 10)
        ((10,), True)
    """
    if expected_block < 1.0:
        raise SpecificationError(f"expected_block must be at least 1; got {expected_block}.")
    continue_probability = 1.0 - 1.0 / expected_block
    out = np.empty(count, dtype=np.intp)
    out[0] = rng.integers(count)
    continued = rng.random(count - 1) < continue_probability
    fresh = rng.integers(count, size=count - 1)
    for t in range(1, count):
        out[t] = (out[t - 1] + 1) % count if continued[t - 1] else fresh[t - 1]
    return out


def _model_confidence_set(
    losses: npt.NDArray[np.float64],
    *,
    alpha: float,
    n_bootstrap: int,
    block_length: float,
    statistic: str,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """Hansen, Lunde and Nason's (2011) model confidence set by sequential elimination.

    Starting from every model, the equivalence hypothesis ``E[d_ij] = 0``
    for all pairs in the current set is tested with the range statistic
    (largest studentized pairwise differential) or the max statistic
    (largest studentized deviation from the set average), its null
    distribution taken from a stationary bootstrap of the loss panel; the
    worst model is eliminated and the test repeated until it no longer
    rejects. Each model's MCS p-value is the running maximum of the
    elimination p-values up to its own, which makes the set at level
    ``alpha`` exactly the models with p-value at least ``alpha``.

    Args:
        losses: ``(T, M)`` losses, one column per model, negatively
            oriented.
        alpha: Level of the set.
        n_bootstrap: Bootstrap replications.
        block_length: Expected block length of the stationary bootstrap.
        statistic: ``"range"`` for ``T_R`` or ``"max"`` for ``T_max``.
        rng: Random generator.

    Returns:
        ``(order, pvalues)``: the models in elimination order (the last is
        the best), and each model's MCS p-value indexed by model.

    Raises:
        SpecificationError: If the statistic is unknown or the panel is
            too small.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> base = rng.standard_normal((150, 3)) ** 2
        >>> base[:, 2] += 1.0
        >>> order, p = _model_confidence_set(
        ...     base, alpha=0.1, n_bootstrap=200, block_length=2.0, statistic="range", rng=rng
        ... )
        >>> int(order[0]), bool(p[2] < 0.1)
        (2, True)
    """
    if statistic not in ("range", "max"):
        raise SpecificationError(f"statistic must be 'range' or 'max'; got {statistic!r}.")
    count, n_models = losses.shape
    if n_models < 2:
        raise SpecificationError("a model confidence set needs at least two models.")
    if count < 2:
        raise SpecificationError("a model confidence set needs at least two evaluation origins.")
    indices = np.stack(
        [
            _stationary_bootstrap_indices(count, expected_block=block_length, rng=rng)
            for _ in range(n_bootstrap)
        ]
    )
    boot_means = losses[indices].mean(axis=1)  # (B, M)
    sample_mean = losses.mean(axis=0)  # (M,)
    remaining = list(range(n_models))
    order: list[int] = []
    elimination_p: list[float] = []
    while len(remaining) > 1:
        keep = np.array(remaining)
        mean = sample_mean[keep]
        boot = boot_means[:, keep]
        if statistic == "range":
            pair_mean = mean[:, None] - mean[None, :]
            pair_boot = boot[:, :, None] - boot[:, None, :]
            pair_var = ((pair_boot - pair_mean[None]) ** 2).mean(axis=0)
            pair_var[np.arange(len(keep)), np.arange(len(keep))] = 1.0
            scale = np.sqrt(np.maximum(pair_var, 1e-300))
            observed = float(np.abs(pair_mean / scale).max())
            simulated = np.abs((pair_boot - pair_mean[None]) / scale[None]).max(axis=(1, 2))
            deviation = (mean[:, None] - mean[None, :]).mean(axis=1)
            dev_var = ((boot - boot.mean(axis=1, keepdims=True) - deviation[None]) ** 2).mean(
                axis=0
            )
        else:
            deviation = mean - mean.mean()
            boot_dev = boot - boot.mean(axis=1, keepdims=True)
            dev_var = ((boot_dev - deviation[None]) ** 2).mean(axis=0)
            scale_dev = np.sqrt(np.maximum(dev_var, 1e-300))
            observed = float((deviation / scale_dev).max())
            simulated = ((boot_dev - deviation[None]) / scale_dev[None]).max(axis=1)
        pvalue = float((simulated >= observed).mean())
        worst = int(np.argmax(deviation / np.sqrt(np.maximum(dev_var, 1e-300))))
        order.append(int(keep[worst]))
        elimination_p.append(pvalue)
        remaining.remove(int(keep[worst]))
    order.append(remaining[0])
    elimination_p.append(1.0)
    pvalues = np.empty(n_models)
    running = 0.0
    for model, p in zip(order, elimination_p, strict=True):
        running = max(running, p)
        pvalues[model] = running
    return np.asarray(order, dtype=np.intp), pvalues


def _mincer_zarnowitz(
    realized: npt.NDArray[np.float64], forecast: npt.NDArray[np.float64], *, horizon: int
) -> tuple[float, float, float, float, float]:
    """Mincer and Zarnowitz's (1969) efficiency regression with a HAC Wald test.

    ``y_t = a + b f_t + u_t``; an unbiased and efficient forecast has
    ``(a, b) = (0, 1)``. The joint restriction is tested with a Wald
    statistic whose covariance is the Bartlett long-run covariance of the
    regressor-scaled residuals through ``horizon - 1`` lags with the
    ``T / (T - 2)`` degrees-of-freedom scaling, referred to ``F(2, T - 2)``
    as ``W / 2``. At one step the size is close to nominal from 100
    origins on; at multi-step horizons the truncated kernel understates
    the long-run variance and the test over-rejects (about 13% at nominal
    5% for a four-step horizon on 100 origins), which the caller is told.

    Args:
        realized: ``(T,)`` outcomes.
        forecast: ``(T,)`` point forecasts.
        horizon: The forecast horizon behind the series.

    Returns:
        ``(intercept, slope, statistic, pvalue, r_squared)``, the statistic
        on the chi-squared scale.

    Raises:
        NumericalError: If the forecasts are constant or the covariance
            degenerates.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> f = rng.standard_normal(200)
        >>> a, b, stat, p, r2 = _mincer_zarnowitz(f + 0.5 * rng.standard_normal(200), f, horizon=1)
        >>> bool(abs(b - 1.0) < 0.15 and p > 0.05)
        True
    """
    count = realized.shape[0]
    design = np.column_stack([np.ones(count), forecast])
    gram = design.T @ design
    if float(forecast.std()) <= 1e-12 * max(float(np.abs(forecast).max()), 1.0):
        raise NumericalError("the forecasts are constant; the efficiency regression is singular.")
    coefficients = np.linalg.solve(gram, design.T @ realized)
    residual = realized - design @ coefficients
    scores = design * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, horizon):
        weight = 1.0 - lag / horizon
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    inverse = np.linalg.inv(gram)
    covariance = inverse @ meat @ inverse * (count / (count - 2))
    deviation = coefficients - np.array([0.0, 1.0])
    try:
        statistic = float(deviation @ np.linalg.solve(covariance, deviation))
    except np.linalg.LinAlgError as error:
        raise NumericalError(
            "the HAC covariance of the efficiency regression is singular."
        ) from error
    total = float(((realized - realized.mean()) ** 2).sum())
    r_squared = 1.0 - float(residual @ residual) / total if total > 0.0 else 0.0
    return (
        float(coefficients[0]),
        float(coefficients[1]),
        statistic,
        float(f_dist.sf(0.5 * statistic, 2, count - 2)),
        r_squared,
    )


def _forecast_encompassing(
    errors_a: npt.NDArray[np.float64], errors_b: npt.NDArray[np.float64], *, horizon: int
) -> tuple[float, float, float]:
    """Harvey, Leybourne and Newbold's (1998) test that forecast A encompasses B.

    Forecast A encompasses B when the optimal combination puts zero weight
    on B, which is ``E[e_A (e_A - e_B)] = 0``. The statistic is the
    Diebold-Mariano machinery on ``d_t = e_A,t (e_A,t - e_B,t)`` with the
    Bartlett long-run variance through ``horizon - 1`` lags and the HLN
    small-sample correction, referred to a ``t`` distribution one-sided:
    a positive mean says B carries information A lacks.

    Args:
        errors_a: ``(T,)`` errors of the forecast claimed to encompass.
        errors_b: ``(T,)`` errors of the forecast claimed to be encompassed.
        horizon: The forecast horizon behind the series.

    Returns:
        ``(statistic, pvalue, weight)``: the corrected statistic, its
        upper-tail p-value, and the least-squares combination weight on B,
        ``mean(e_A (e_A - e_B)) / mean((e_A - e_B)**2)``.

    Raises:
        SpecificationError: If the two forecasts are numerically identical.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> e_a = rng.standard_normal(200)
        >>> e_b = e_a + 0.5 * rng.standard_normal(200)
        >>> stat, p, w = _forecast_encompassing(e_a, e_b, horizon=1)
        >>> bool(p > 0.05 and abs(w) < 0.2)
        True
    """
    gap = errors_a - errors_b
    if float(np.abs(gap).max()) <= 1e-14 * max(float(np.abs(errors_a).max()), 1.0):
        raise SpecificationError(
            "the two forecasts are numerically identical; encompassing is not defined."
        )
    differential = errors_a * gap
    count = differential.shape[0]
    mean = float(differential.mean())
    variance = _bartlett_long_run_variance(differential - mean, horizon)
    statistic = mean / float(np.sqrt(variance / count))
    adjust = float(np.sqrt((count + 1 - 2 * horizon + horizon * (horizon - 1) / count) / count))
    corrected = statistic * adjust
    weight = mean / float((gap**2).mean())
    return corrected, float(t_dist.sf(corrected, count - 1)), weight


def _giacomini_white(
    differential: npt.NDArray[np.float64],
    instruments: npt.NDArray[np.float64],
    *,
    horizon: int,
) -> tuple[float, float, npt.NDArray[np.float64]]:
    """Giacomini and White's (2006) conditional predictive ability Wald statistic.

    With ``d_t`` the loss differential of a forecast made at origin ``t``
    and ``h_t`` the ``q`` instruments known at that origin, the null of
    equal conditional predictive ability is ``E[d_t | F_t] = 0``, which
    implies ``E[h_t d_t] = 0``. The statistic is ``T zbar' Omega^{-1}
    zbar`` for ``z_t = h_t d_t``, chi-squared with ``q`` degrees of
    freedom, where ``Omega`` is the uncentered sample covariance of
    ``z_t`` at one step -- the paper's ``T R**2`` form -- and its HAC
    version with Bartlett weights through ``horizon - 1`` lags beyond,
    since an ``h``-step differential is mechanically an MA(``h - 1``).

    The OLS coefficients of ``d_t`` on ``h_t`` come back alongside: they
    are the paper's decision rule, whose fitted sign at a fresh ``h_t``
    says which forecaster to use next.

    Args:
        differential: ``(T,)`` loss differential, first minus second.
        instruments: ``(T, q)`` instruments aligned with it, each row
            known at that origin; a constant column makes the
            unconditional test a special case.
        horizon: The forecast horizon behind the differential.

    Returns:
        ``(statistic, pvalue, coefficients)``.

    Raises:
        NumericalError: If the instruments are collinear or the moment
            covariance is singular.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> d = rng.standard_normal(200)
        >>> h = np.column_stack([np.ones(199), d[:-1]])
        >>> stat, p, beta = _giacomini_white(d[1:], h, horizon=1)
        >>> bool(p > 0.05), beta.shape
        (True, (2,))
    """
    count, q = instruments.shape
    if np.linalg.matrix_rank(instruments) < q:
        raise NumericalError(
            "the Giacomini-White instruments are collinear; drop the redundant column."
        )
    scores = instruments * differential[:, None]
    mean = scores.mean(axis=0)
    meat = scores.T @ scores / count
    for lag in range(1, horizon):
        weight = 1.0 - lag / horizon
        cross = scores[lag:].T @ scores[:-lag] / count
        meat += weight * (cross + cross.T)
    try:
        solved = np.linalg.solve(meat, mean)
    except np.linalg.LinAlgError as error:
        raise NumericalError(
            "the conditional moment covariance is singular; the losses carry no usable "
            "variation for the Giacomini-White test."
        ) from error
    statistic = float(count * mean @ solved)
    if not np.isfinite(statistic) or statistic < 0.0:
        raise NumericalError(
            "the Giacomini-White covariance is not positive definite at this horizon; "
            "the evaluation window is too short for the Bartlett window."
        )
    coefficients = np.linalg.lstsq(instruments, differential, rcond=None)[0]
    return statistic, float(chi2.sf(statistic, q)), np.asarray(coefficients, dtype=np.float64)


def _newey_west_bandwidth(nobs: int) -> int:
    """Newey and West's (1994) rule-of-thumb Bartlett bandwidth ``floor(4 (T/100)^(2/9))``.

    Example:
        >>> _newey_west_bandwidth(100), _newey_west_bandwidth(1000)
        (4, 6)
    """
    return int(np.floor(4.0 * (nobs / 100.0) ** (2.0 / 9.0)))


def _schwert_max_lags(nobs: int) -> int:
    """Schwert's (1989) ceiling on the Dickey-Fuller augmentation, ``floor(12 (T/100)^(1/4))``.

    Example:
        >>> _schwert_max_lags(100), _schwert_max_lags(400)
        (12, 16)
    """
    return int(np.floor(12.0 * (nobs / 100.0) ** 0.25))


def _andrews_bandwidth(x: npt.NDArray[np.float64], kernel: str) -> float:
    """Andrews' (1991) automatic bandwidth from an AR(1) plug-in.

    Args:
        x: ``(T,)`` series, already demeaned.
        kernel: ``"bartlett"`` or ``"quadratic-spectral"``.

    Returns:
        The bandwidth ``S_T``; a Bartlett kernel truncates at ``floor(S_T)``.

    Example:
        >>> x = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        >>> _andrews_bandwidth(x, "bartlett") > 0
        True
    """
    nobs = x.shape[0]
    denominator = float(x[:-1] @ x[:-1])
    rho = float(x[1:] @ x[:-1]) / denominator if denominator > 0.0 else 0.0
    rho = float(np.clip(rho, -0.97, 0.97))
    if kernel == "bartlett":
        alpha = 4.0 * rho**2 / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
        return float(1.1447 * (alpha * nobs) ** (1.0 / 3.0))
    alpha = 4.0 * rho**2 / (1.0 - rho) ** 4
    return float(1.3221 * (alpha * nobs) ** (1.0 / 5.0))


def _long_run_variance(
    x: npt.NDArray[np.float64], *, kernel: str = "bartlett", bandwidth: float | None = None
) -> float:
    """Kernel estimate of the long-run variance ``sum_j gamma_j`` of a demeaned series.

    Args:
        x: ``(T,)`` series with its mean already removed.
        kernel: ``"bartlett"`` (Newey-West) or ``"quadratic-spectral"``
            (Andrews).
        bandwidth: Kernel bandwidth. ``None`` selects Newey and West's
            (1994) rule for the Bartlett kernel and Andrews' (1991) AR(1)
            plug-in for the quadratic spectral.

    Returns:
        The long-run variance, floored at a tiny positive number.

    Raises:
        SpecificationError: If the kernel is unknown or the bandwidth is
            negative.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> e = rng.standard_normal(4000)
        >>> bool(abs(_long_run_variance(e - e.mean()) - 1.0) < 0.15)
        True
    """
    if kernel not in ("bartlett", "quadratic-spectral"):
        raise SpecificationError(
            f"kernel must be 'bartlett' or 'quadratic-spectral'; got {kernel!r}."
        )
    nobs = x.shape[0]
    if bandwidth is None:
        width = (
            float(_newey_west_bandwidth(nobs))
            if kernel == "bartlett"
            else _andrews_bandwidth(x, kernel)
        )
    else:
        width = float(bandwidth)
    if width < 0.0:
        raise SpecificationError(f"bandwidth must be non-negative; got {bandwidth}.")
    gamma0 = float(x @ x) / nobs
    total = gamma0
    max_lag = nobs - 1 if kernel == "quadratic-spectral" else int(np.floor(width))
    for lag in range(1, max_lag + 1):
        if kernel == "bartlett":
            weight = 1.0 - lag / (width + 1.0)
        else:
            if width <= 0.0:
                break
            z = 6.0 * np.pi * (lag / width) / 5.0
            weight = 3.0 / z**2 * (np.sin(z) / z - np.cos(z))
            if lag > 20 * width:
                break
        gamma = float(x[lag:] @ x[:-lag]) / nobs
        total += 2.0 * weight * gamma
    return max(total, 1e-300)


def _dickey_fuller_design(
    y: npt.NDArray[np.float64], lags: int, trend: str, *, drop: int = 0
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Target and design of the augmented Dickey-Fuller regression.

    ``dy_t = d_t' gamma + rho y_{t-1} + sum_{j<=lags} b_j dy_{t-j} + e_t``,
    with the level lag in column ``0`` after the deterministic block.

    Args:
        y: ``(T,)`` series.
        lags: Augmentation lags.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        drop: Extra leading observations to discard, so that fits with
            different ``lags`` share a common effective sample.

    Returns:
        ``(target, design)`` with the level lag as the first column after
        the deterministic terms.

    Raises:
        SpecificationError: If the sample is too short.
    """
    dy = np.diff(y)
    start = max(lags, drop)
    n_eff = dy.shape[0] - start
    width = lags + 1 + (2 if trend == "ct" else 1 if trend == "c" else 0)
    if n_eff < width + 3:
        raise SpecificationError(
            f"the sample of {y.shape[0]} observations is too short for {lags} lags."
        )
    target = dy[start:]
    columns = [deterministic_columns(trend, n_eff, start=start + 2), y[start:-1, None]]
    for j in range(1, lags + 1):
        columns.append(dy[start - j : dy.shape[0] - j, None])
    return target, np.hstack(columns)


def _dickey_fuller_regression(
    y: npt.NDArray[np.float64], lags: int, trend: str, *, drop: int = 0
) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Fit the ADF regression and return the ``tau`` statistic on the level lag.

    Returns:
        ``(tau, rho, coefficients, residuals, n_eff)`` where ``rho`` is the
        coefficient on ``y_{t-1}`` and ``coefficients`` the full vector.

    Raises:
        SpecificationError: If the sample is too short.
        NumericalError: If the design is singular.
    """
    target, design = _dickey_fuller_design(y, lags, trend, drop=drop)
    n_eff, k = design.shape
    coef, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < k:
        raise NumericalError("the Dickey-Fuller design is singular.")
    resid = target - design @ coef
    sigma2 = float(resid @ resid) / (n_eff - k)
    index = 2 if trend == "ct" else 1 if trend == "c" else 0
    try:
        inverse = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError as error:
        raise NumericalError("the Dickey-Fuller design is singular.") from error
    se = float(np.sqrt(sigma2 * inverse[index, index]))
    return float(coef[index] / se), float(coef[index]), coef, resid, n_eff


def _select_dickey_fuller_lags(
    y: npt.NDArray[np.float64], max_lags: int, trend: str, method: str
) -> int:
    """Choose the ADF augmentation on a common sample.

    Args:
        y: ``(T,)`` series.
        max_lags: Largest augmentation considered.
        trend: Deterministic specification.
        method: ``"aic"``, ``"bic"`` (Gaussian criteria on the common
            sample), ``"t-stat"`` (Ng-Perron 1995 general-to-specific at
            the 10% level), or ``"maic"`` (Ng-Perron 2001 modified AIC).

    Returns:
        The chosen number of lags.

    Raises:
        SpecificationError: If the method is unknown.
    """
    if method not in ("aic", "bic", "t-stat", "maic"):
        raise SpecificationError(
            f"method must be 'aic', 'bic', 't-stat' or 'maic'; got {method!r}."
        )
    if method == "t-stat":
        for lags in range(max_lags, 0, -1):
            target, design = _dickey_fuller_design(y, lags, trend, drop=max_lags)
            coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
            resid = target - design @ coef
            n_eff, k = design.shape
            sigma2 = float(resid @ resid) / (n_eff - k)
            inverse = np.linalg.pinv(design.T @ design)
            t_last = abs(coef[-1]) / float(np.sqrt(sigma2 * inverse[-1, -1]))
            if t_last > 1.6449:
                return lags
        return 0
    best_lags, best_value = 0, np.inf
    for lags in range(max_lags + 1):
        target, design = _dickey_fuller_design(y, lags, trend, drop=max_lags)
        coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        resid = target - design @ coef
        n_eff, k = design.shape
        sigma2 = float(resid @ resid) / n_eff
        if method == "maic":
            index = 2 if trend == "ct" else 1 if trend == "c" else 0
            level = design[:, index]
            tau_k = float(coef[index] ** 2 * (level @ level)) / sigma2
            value = np.log(sigma2) + 2.0 * (tau_k + lags) / n_eff
        elif method == "aic":
            value = np.log(sigma2) + 2.0 * k / n_eff
        else:
            value = np.log(sigma2) + k * np.log(n_eff) / n_eff
        if value < best_value:
            best_lags, best_value = lags, value
    return best_lags


def _resolve_dickey_fuller_lags(
    y: npt.NDArray[np.float64],
    lags: int | None,
    max_lags: int | None,
    trend: str,
    method: str,
) -> tuple[int, str]:
    """The Dickey-Fuller augmentation to use, and a label saying how it was chosen.

    Args:
        y: ``(T,)`` series the selection runs on.
        lags: A fixed augmentation, or ``None`` to select.
        max_lags: Largest augmentation under selection; ``None`` for
            Schwert's rule.
        trend: Deterministic specification of the selection regression.
        method: Selection rule, as :func:`_select_dickey_fuller_lags`.

    Returns:
        ``(lags, label)`` with the label ``"fixed"`` or ``"<method> (max
        <ceiling>)"``.

    Raises:
        SpecificationError: If a count is negative or the method unknown.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> _resolve_dickey_fuller_lags(rng.standard_normal(200), 3, None, "c", "aic")
        (3, 'fixed')
    """
    if lags is not None:
        if lags < 0:
            raise SpecificationError(f"lags must be non-negative; got {lags}.")
        return int(lags), "fixed"
    ceiling = _schwert_max_lags(y.shape[0]) if max_lags is None else int(max_lags)
    if ceiling < 0:
        raise SpecificationError(f"max_lags must be non-negative; got {max_lags}.")
    return _select_dickey_fuller_lags(y, ceiling, trend, method), f"{method} (max {ceiling})"


def _mackinnon_pvalue(tau: float, trend: str) -> float:
    """MacKinnon's (1994) response-surface p-value of a Dickey-Fuller ``tau`` statistic.

    Example:
        >>> round(_mackinnon_pvalue(-2.86, "c"), 2)
        0.05
    """
    if tau < _MACKINNON_TAU_MIN[trend]:
        return 0.0
    if tau > _MACKINNON_TAU_MAX[trend]:
        return 1.0
    coefficients = (
        _MACKINNON_TAU_SMALL[trend]
        if tau <= _MACKINNON_TAU_STAR[trend]
        else _MACKINNON_TAU_LARGE[trend]
    )
    value = sum(c * tau**i for i, c in enumerate(coefficients))
    return float(norm.cdf(value))


def _mackinnon_critical_values(trend: str, nobs: int) -> dict[str, float]:
    """MacKinnon's (2010) finite-sample critical values of the Dickey-Fuller ``tau`` law.

    Example:
        >>> cv = _mackinnon_critical_values("c", 100)
        >>> round(cv["5%"], 2)
        -2.89
    """
    out: dict[str, float] = {}
    for label, (b0, b1, b2, b3) in zip(
        _CRITICAL_LEVELS, _MACKINNON_CRITICAL_2010[trend], strict=True
    ):
        out[label] = float(b0 + b1 / nobs + b2 / nobs**2 + b3 / nobs**3)
    return out


def _kpss_statistic(
    y: npt.NDArray[np.float64], trend: str, *, kernel: str, bandwidth: float | None
) -> tuple[float, float]:
    """Kwiatkowski-Phillips-Schmidt-Shin LM statistic and the long-run variance behind it.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> e = rng.standard_normal(500)
        >>> stat, _ = _kpss_statistic(e, "c", kernel="bartlett", bandwidth=None)
        >>> bool(stat < 0.463)
        True
    """
    nobs = y.shape[0]
    design = deterministic_columns(trend, nobs)
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    partial = np.cumsum(resid)
    lrv = _long_run_variance(resid, kernel=kernel, bandwidth=bandwidth)
    return float(partial @ partial) / (nobs**2 * lrv), lrv


def _kpss_pvalue(statistic: float, trend: str) -> float:
    """Interpolated p-value from the KPSS table, clipped to ``[0.01, 0.10]`` at the ends.

    Example:
        >>> round(_kpss_pvalue(0.463, "c"), 2)
        0.05
    """
    table = _KPSS_CRITICAL[trend]
    levels = np.array([p for p, _ in table])
    values = np.array([cv for _, cv in table])
    if statistic <= values[0]:
        return float(levels[0])
    if statistic >= values[-1]:
        return float(levels[-1])
    return float(np.interp(statistic, values, levels))


def _phillips_perron(
    y: npt.NDArray[np.float64], trend: str, *, kernel: str, bandwidth: float | None
) -> tuple[float, float, int]:
    """Phillips-Perron ``Z_t`` from the un-augmented Dickey-Fuller regression.

    Returns:
        ``(Z_t, long_run_variance, n_eff)``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> walk = np.cumsum(rng.standard_normal(400))
        >>> z, _, _ = _phillips_perron(walk, "c", kernel="bartlett", bandwidth=None)
        >>> bool(z > -2.86)
        True
    """
    tau, _, _, resid, n_eff = _dickey_fuller_regression(y, 0, trend)
    _, design = _dickey_fuller_design(y, 0, trend)
    k = design.shape[1]
    sigma2 = float(resid @ resid) / (n_eff - k)
    lrv = _long_run_variance(resid, kernel=kernel, bandwidth=bandwidth)
    index = 2 if trend == "ct" else 1 if trend == "c" else 0
    inverse = np.linalg.inv(design.T @ design)
    se = float(np.sqrt(sigma2 * inverse[index, index]))
    gamma0 = float(resid @ resid) / n_eff
    z_t = np.sqrt(gamma0 / lrv) * tau - 0.5 * (lrv - gamma0) * n_eff * se / np.sqrt(lrv * sigma2)
    return float(z_t), lrv, n_eff


def _ng_perron_statistics(
    detrended: npt.NDArray[np.float64], lags: int, trend: str
) -> dict[str, float]:
    """Ng and Perron's (2001) ``M`` statistics on a GLS-detrended series.

    ``s_AR^2`` is the autoregressive spectral density at frequency zero
    from the Dickey-Fuller regression with ``lags`` augmentation terms.

    Returns:
        ``{"MZa", "MZt", "MSB", "MPT"}``.
    """
    nobs = detrended.shape[0]
    target, design = _dickey_fuller_design(detrended, lags, "n")
    coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    s2_e = float(resid @ resid) / (design.shape[0] - design.shape[1])
    s2_ar = s2_e / (1.0 - float(np.sum(coef[1:]))) ** 2
    lagged = detrended[:-1]
    sum_sq = float(lagged @ lagged) / nobs**2
    mza = (detrended[-1] ** 2 / nobs - s2_ar) / (2.0 * sum_sq)
    msb = float(np.sqrt(sum_sq / s2_ar))
    mzt = mza * msb
    c_bar = _GLS_DETREND_C[trend]
    if trend == "c":
        mpt = (c_bar**2 * sum_sq - c_bar * detrended[-1] ** 2 / nobs) / s2_ar
    else:
        mpt = (c_bar**2 * sum_sq + (1.0 - c_bar) * detrended[-1] ** 2 / nobs) / s2_ar
    return {"MZa": float(mza), "MZt": float(mzt), "MSB": msb, "MPT": float(mpt)}


def _zivot_andrews(
    y: npt.NDArray[np.float64],
    *,
    model: str,
    lags: int | None,
    max_lags: int,
    trimming: float,
) -> tuple[float, int, int]:
    """Minimum Dickey-Fuller ``t`` over candidate one-time breaks (Zivot & Andrews, 1992).

    Args:
        y: ``(T,)`` series.
        model: ``"c"`` (intercept break), ``"t"`` (trend-slope break) or
            ``"ct"`` (both).
        lags: Fixed augmentation, or ``None`` for t-statistic selection on
            the no-break regression, held fixed across candidates.
        max_lags: Largest augmentation under selection.
        trimming: Fraction of the sample excluded at each end.

    Returns:
        ``(statistic, break_index, lags_used)`` where ``break_index`` is
        the first observation of the new regime and ``lags_used`` the
        augmentation at the minimizing date.
    """
    nobs = y.shape[0]
    first = max(int(np.floor(trimming * nobs)), 2)
    last = nobs - first
    best = (np.inf, -1, 0)
    dy = np.diff(y)
    k = _select_dickey_fuller_lags(y, max_lags, "ct", "t-stat") if lags is None else lags
    for tb in range(first, last):
        n_eff = dy.shape[0] - k
        time = np.arange(k + 2, nobs + 1, dtype=np.float64)
        columns = [np.ones((n_eff, 1)), time[:, None]]
        if model in ("c", "ct"):
            columns.append((time > tb + 1).astype(np.float64)[:, None])
        if model in ("t", "ct"):
            columns.append(np.where(time > tb + 1, time - tb - 1, 0.0)[:, None])
        columns.append(y[k:-1, None])
        for j in range(1, k + 1):
            columns.append(dy[k - j : dy.shape[0] - j, None])
        design = np.hstack(columns)
        target = dy[k:]
        coef, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank < design.shape[1]:
            continue
        resid = target - design @ coef
        sigma2 = float(resid @ resid) / (n_eff - design.shape[1])
        index = design.shape[1] - k - 1
        inverse = np.linalg.pinv(design.T @ design)
        t_stat = float(coef[index] / np.sqrt(sigma2 * inverse[index, index]))
        if t_stat < best[0]:
            best = (t_stat, tb + 1, k)
    if best[1] < 0:
        raise NumericalError("every candidate break produced a singular design.")
    return best


def _segment_ssr(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64], min_size: int
) -> npt.NDArray[np.float64]:
    """Sum of squared residuals of every admissible segment ``[i, j)``.

    Args:
        y: ``(T,)`` target.
        x: ``(T, q)`` design.
        min_size: Smallest admissible segment length.

    Returns:
        A ``(T + 1, T + 1)`` array with ``ssr[i, j]`` for ``j - i >=
        min_size`` and ``inf`` elsewhere.
    """
    nobs, q = x.shape
    out = np.full((nobs + 1, nobs + 1), np.inf)
    for i in range(nobs - min_size + 1):
        xtx = np.zeros((q, q))
        xty = np.zeros(q)
        yty = 0.0
        for j in range(i, nobs):
            row = x[j]
            xtx += np.outer(row, row)
            xty += row * y[j]
            yty += y[j] * y[j]
            if j + 1 - i >= min_size:
                try:
                    beta = np.linalg.solve(xtx, xty)
                except np.linalg.LinAlgError:
                    continue
                out[i, j + 1] = max(yty - float(beta @ xty), 0.0)
    return out


def _bai_perron_partition(
    ssr: npt.NDArray[np.float64], n_breaks: int, min_size: int
) -> tuple[float, tuple[int, ...]]:
    """Global minimizer of the segmented sum of squares with ``n_breaks`` breaks.

    Dynamic programming over the segment table (Bai & Perron, 2003).

    Returns:
        ``(ssr, break_indices)`` with each index the first observation of
        a new regime.
    """
    nobs = ssr.shape[0] - 1
    if n_breaks == 0:
        return float(ssr[0, nobs]), ()
    best = np.full((n_breaks + 1, nobs + 1), np.inf)
    argmin = np.zeros((n_breaks + 1, nobs + 1), dtype=np.int64)
    best[0] = ssr[0]
    for m in range(1, n_breaks + 1):
        for j in range((m + 1) * min_size, nobs + 1):
            candidates = (
                best[m - 1, m * min_size : j - min_size + 1]
                + ssr[m * min_size : j - min_size + 1, j]
            )
            if candidates.size == 0:
                continue
            pick = int(np.argmin(candidates))
            best[m, j] = candidates[pick]
            argmin[m, j] = pick + m * min_size
    if not np.isfinite(best[n_breaks, nobs]):
        raise SpecificationError(
            f"{n_breaks} breaks with segments of at least {min_size} do not fit {nobs} "
            "observations."
        )
    breaks = []
    j = nobs
    for m in range(n_breaks, 0, -1):
        j = int(argmin[m, j])
        breaks.append(j)
    return float(best[n_breaks, nobs]), tuple(sorted(breaks))


def _bridge_functionals(
    q: int, trimming: float, *, n_draws: int, grid: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Draws of the sup, exp and ave functionals of the Andrews (1993) limit.

    ``F(s) = ||B(s)||^2 / (s (1 - s))`` for a ``q``-dimensional Brownian
    bridge ``B`` on ``[trimming, 1 - trimming]``.

    Returns:
        ``(sup, exp, ave)`` each ``(n_draws,)``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> sup, _, _ = _bridge_functionals(1, 0.15, n_draws=200, grid=200, rng=rng)
        >>> bool(6.0 < np.quantile(sup, 0.95) < 11.0)
        True
    """
    s = np.arange(1, grid) / grid
    keep = (s >= trimming) & (s <= 1.0 - trimming)
    sup = np.empty(n_draws)
    exp = np.empty(n_draws)
    ave = np.empty(n_draws)
    batch = max(1, int(2e6 // (grid * q)))
    for start in range(0, n_draws, batch):
        size = min(batch, n_draws - start)
        increments = rng.standard_normal((size, grid, q)) / np.sqrt(grid)
        motion = np.cumsum(increments, axis=1)
        bridge = motion[:, :-1, :] - s[None, :, None] * motion[:, -1:, :]
        f = np.sum(bridge**2, axis=2) / (s * (1.0 - s))[None, :]
        window = f[:, keep]
        sup[start : start + size] = window.max(axis=1)
        exp[start : start + size] = logsumexp(0.5 * window, axis=1) - np.log(window.shape[1])
        ave[start : start + size] = window.mean(axis=1)
    return sup, exp, ave


def _simulated_critical_values(
    draws: npt.NDArray[np.float64], statistic: float, *, lower_tail: bool = False
) -> tuple[float, dict[str, float]]:
    """P-value and critical values of a statistic against simulated null draws.

    Args:
        draws: ``(R,)`` draws from the null law.
        statistic: The observed statistic.
        lower_tail: Whether rejection lies in the lower tail.

    Returns:
        ``(pvalue, critical_values)`` with the p-value the share of draws
        at or beyond the statistic in the rejection direction and the
        critical values keyed by ``_CRITICAL_LEVELS``.

    Example:
        >>> p, cv = _simulated_critical_values(np.arange(1000.0), 950.0)
        >>> round(p, 2), round(cv["5%"])
        (0.05, 949)
        >>> p, cv = _simulated_critical_values(np.arange(1000.0), 50.0, lower_tail=True)
        >>> round(p, 2), round(cv["5%"])
        (0.05, 50)
    """
    if lower_tail:
        pvalue = float(np.mean(draws <= statistic))
        values = np.quantile(draws, [0.01, 0.05, 0.10])
    else:
        pvalue = float(np.mean(draws >= statistic))
        values = np.quantile(draws, [0.99, 0.95, 0.90])
    return pvalue, {label: float(v) for label, v in zip(_CRITICAL_LEVELS, values, strict=True)}


def _recursive_residuals(
    y: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Standardized recursive residuals of Brown, Durbin and Evans (1975).

    Returns:
        ``(T - q,)`` residuals from observation ``q + 1`` on.

    Raises:
        NumericalError: If the first ``q`` rows are singular.
    """
    nobs, q = x.shape
    xtx = x[:q].T @ x[:q]
    try:
        inverse = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as error:
        raise NumericalError("the first q observations do not identify the regression.") from error
    beta = inverse @ x[:q].T @ y[:q]
    out = np.empty(nobs - q)
    for t in range(q, nobs):
        row = x[t]
        f = 1.0 + float(row @ inverse @ row)
        out[t - q] = (y[t] - float(row @ beta)) / np.sqrt(f)
        gain = inverse @ row / f
        beta = beta + gain * (y[t] - float(row @ beta))
        inverse = inverse - np.outer(gain, row @ inverse)
    return out


def _cusum_squares_quantiles(
    count: int, levels: tuple[float, ...], *, n_draws: int, rng: np.random.Generator
) -> tuple[dict[float, float], npt.NDArray[np.float64]]:
    """Null quantiles of ``max_t |S_t - t / n|`` for the CUSUM-of-squares path.

    Under the null the recursive residuals are i.i.d. Gaussian, so the
    exact finite-sample law is simulated directly.

    Returns:
        ``({level: quantile}, draws)``.
    """
    z = rng.standard_normal((n_draws, count)) ** 2
    path = np.cumsum(z, axis=1) / z.sum(axis=1, keepdims=True)
    expected = np.arange(1, count + 1) / count
    draws = np.abs(path - expected[None, :]).max(axis=1)
    return {level: float(np.quantile(draws, 1.0 - level)) for level in levels}, draws


def _nested_f_test(
    ssr_restricted: float, ssr_unrestricted: float, df_num: int, df_den: int
) -> tuple[float, float]:
    """F statistic and p-value for nested least-squares fits.

    Args:
        ssr_restricted: Residual sum of squares under the null.
        ssr_unrestricted: Residual sum of squares under the alternative.
        df_num: Restrictions.
        df_den: Residual degrees of freedom of the unrestricted fit.

    Returns:
        ``(statistic, pvalue)``.

    Raises:
        NumericalError: If the unrestricted fit is exact or the degrees of
            freedom are exhausted.

    Example:
        >>> stat, p = _nested_f_test(10.0, 5.0, 2, 20)
        >>> round(stat, 2), bool(p < 0.01)
        (10.0, True)
    """
    if df_den < 1 or df_num < 1:
        raise NumericalError(
            f"the F test has no degrees of freedom left ({df_num}, {df_den}); shorten the order."
        )
    if ssr_unrestricted <= 1e-300:
        raise NumericalError("the unrestricted regression fits exactly; the series is degenerate.")
    statistic = max((ssr_restricted - ssr_unrestricted) / df_num, 0.0) / (ssr_unrestricted / df_den)
    return float(statistic), float(f_dist.sf(statistic, df_num, df_den))


def _autoregressive_design(
    y: npt.NDArray[np.float64], order: int, delay: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Target, ``[1, lags]`` design and delayed threshold variable of a nonlinearity test.

    The sample starts at ``max(order, delay)`` so that every row has its
    full lag history and its transition variable ``y_{t - delay}``.

    Args:
        y: ``(T,)`` series.
        order: Autoregressive order ``p``.
        delay: Delay ``d`` of the transition variable.

    Returns:
        ``(target, design, transition)`` with ``design`` of shape
        ``(T - start, 1 + p)``.

    Example:
        >>> target, design, z = _autoregressive_design(np.arange(6.0), 2, 1)
        >>> target, z
        (array([2., 3., 4., 5.]), array([1., 2., 3., 4.]))
    """
    n = y.shape[0]
    start = max(order, delay)
    design = np.column_stack([np.ones(n - start), lag_matrix(y, order, start=start)])
    return y[start:], design, y[start - delay : n - delay]


def _terasvirta_lm(
    y: npt.NDArray[np.float64], order: int, delay: int
) -> tuple[_FTest, tuple[_FTest, _FTest, _FTest]]:
    """Teräsvirta's (1994) linearity test against smooth transition, with its escalation.

    The auxiliary regression of the AR(``p``) residual on ``w_t = (1,
    y_{t-1}, ..., y_{t-p})`` and ``w~_t y_{t-d}^j`` for ``j = 1, 2, 3``
    (``w~_t`` the lags alone) tests linearity through the ``3p`` cubic
    terms; the F form is used throughout because the chi-squared form
    over-rejects in the sample sizes the test meets. The sequence
    ``H04: b3 = 0``, ``H03: b2 = 0 | b3 = 0`` and ``H02: b1 = 0 | b2 = b3
    = 0`` is Teräsvirta's model-selection device: a quadratic term that
    is the strongest of the three points to ESTAR, otherwise LSTAR.

    Args:
        y: ``(T,)`` series.
        order: Autoregressive order.
        delay: Delay of the transition variable.

    Returns:
        ``((F, pvalue, df1, df2), (H04, H03, H02))``, each inner tuple
        ``(F, pvalue, df1, df2)``.

    Raises:
        NumericalError: If the auxiliary regressions are degenerate.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(300)
        >>> for t in range(1, 300):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal()
        >>> (stat, p, df1, df2), _ = _terasvirta_lm(y, 1, 1)
        >>> (df1, df2), bool(p > 0.01)
        ((3, 294), True)
    """
    target, design, z = _autoregressive_design(y, order, delay)
    lags = design[:, 1:]
    count = target.shape[0]
    _beta, ssr0 = ols(design, target)
    blocks = [lags * (z[:, None] ** j) for j in (1, 2, 3)]
    _b, ssr3 = ols(np.column_stack([design, *blocks]), target)
    _b, ssr2 = ols(np.column_stack([design, *blocks[:2]]), target)
    _b, ssr1 = ols(np.column_stack([design, blocks[0]]), target)
    df4, df3, df2 = (count - (1 + k * order) for k in (4, 3, 2))
    overall = (*_nested_f_test(ssr0, ssr3, 3 * order, df4), 3 * order, df4)
    h04 = (*_nested_f_test(ssr2, ssr3, order, df4), order, df4)
    h03 = (*_nested_f_test(ssr1, ssr2, order, df3), order, df3)
    h02 = (*_nested_f_test(ssr0, ssr1, order, df2), order, df2)
    return overall, (h04, h03, h02)


def _tsay_arranged(
    y: npt.NDArray[np.float64], order: int, delay: int
) -> tuple[float, float, int, int]:
    """Tsay's (1989) arranged-autoregression test for a threshold.

    Sort the AR(``p``) rows by the threshold variable ``y_{t-d}``, fit
    least squares recursively down the sorted sample, and collect the
    standardized one-step predictive residuals from ``m = 3 sqrt(T) + p``
    on. Under linearity those residuals are orthogonal to the regressors;
    under a threshold their mean shifts as the recursion crosses it. The
    regression of the predictive residuals on ``(1, lags)`` gives an
    ``F(p + 1, m' - p - 1)`` statistic, ``m'`` the number of predictive
    residuals.

    Args:
        y: ``(T,)`` series.
        order: Autoregressive order.
        delay: Delay of the threshold variable.

    Returns:
        ``(statistic, pvalue, df1, df2)``.

    Raises:
        NumericalError: If the arranged recursion cannot start or the
            final regression is degenerate.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(300)
        >>> for t in range(1, 300):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal()
        >>> stat, p, df1, df2 = _tsay_arranged(y, 1, 1)
        >>> df1, bool(p > 0.01)
        (2, True)
    """
    target, design, z = _autoregressive_design(y, order, delay)
    count, width = design.shape
    start = int(np.ceil(3.0 * np.sqrt(count))) + order
    if start >= count - width - 1:
        raise NumericalError(
            f"the arranged autoregression needs more than {start + width + 1} rows to start its "
            f"recursion; got {count}."
        )
    order_index = np.argsort(z, kind="stable")
    x_sorted = design[order_index]
    y_sorted = target[order_index]
    gram_inverse = np.linalg.pinv(x_sorted[:start].T @ x_sorted[:start])
    beta = gram_inverse @ x_sorted[:start].T @ y_sorted[:start]
    predictive = np.empty(count - start)
    for i in range(start, count):
        x = x_sorted[i]
        gain = gram_inverse @ x
        leverage = float(x @ gain)
        error = float(y_sorted[i] - x @ beta)
        predictive[i - start] = error / np.sqrt(1.0 + leverage)
        beta = beta + gain * error / (1.0 + leverage)
        gram_inverse = gram_inverse - np.outer(gain, gain) / (1.0 + leverage)
    regressors = x_sorted[start:]
    _b, ssr = ols(regressors, predictive)
    total = float(predictive @ predictive)
    df2 = predictive.shape[0] - width
    return (*_nested_f_test(total, ssr, width, df2), width, df2)


def _reset_test(
    y: npt.NDArray[np.float64], order: int, powers: int
) -> tuple[float, float, int, int]:
    """Ramsey's (1969) RESET on an AR(``p``): powers of the fitted values as omitted regressors.

    Args:
        y: ``(T,)`` series.
        order: Autoregressive order.
        powers: Highest power of the fitted value added; ``2`` adds the
            square alone.

    Returns:
        ``(statistic, pvalue, df1, df2)`` with ``df1 = powers - 1``.

    Raises:
        NumericalError: If the augmented regression is degenerate.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(300)
        >>> for t in range(1, 300):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal()
        >>> stat, p, df1, df2 = _reset_test(y, 1, 3)
        >>> (df1, df2), bool(p > 0.01)
        ((2, 295), True)
    """
    target, design, _z = _autoregressive_design(y, order, 1)
    beta, ssr0 = ols(design, target)
    fitted = design @ beta
    scale = float(np.abs(fitted).max())
    if scale <= 1e-12:
        raise NumericalError("the fitted values are zero; RESET has nothing to raise to a power.")
    extra = np.column_stack([(fitted / scale) ** k for k in range(2, powers + 1)])
    _b, ssr1 = ols(np.column_stack([design, extra]), target)
    df1 = powers - 1
    df2 = target.shape[0] - design.shape[1] - df1
    return (*_nested_f_test(ssr0, ssr1, df1, df2), df1, df2)


def _bds(
    y: npt.NDArray[np.float64], dimension: int, epsilon: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Brock-Dechert-Scheinkman (1996) statistics for embedding dimensions ``2 .. dimension``.

    ``W_m = sqrt(n_m) (C_m - C_1^m) / sigma_m`` with ``C_m`` the
    correlation integral of the ``m``-histories at radius ``epsilon``,
    ``C_1`` re-computed on the same ``n_m = T - m + 1`` points so the two
    are comparable, and ``sigma_m`` the Brock et al. asymptotic standard
    deviation built from the full-sample ``C`` and the three-point
    integral ``K``. The pairwise indicator matrix is formed once and the
    ``m``-history matrices follow by shifting and multiplying, so the
    cost is ``O(m T^2)`` memory-bound.

    Args:
        y: ``(T,)`` series, typically residuals.
        dimension: Largest embedding dimension.
        epsilon: Radius, in the units of ``y``.

    Returns:
        ``(statistics, pvalues)`` for ``m = 2 .. dimension``, two-sided
        standard normal p-values.

    Raises:
        NumericalError: If the radius leaves no pairs close or every pair
            close.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> w, p = _bds(rng.standard_normal(500), 3, 1.0)
        >>> w.shape, bool(np.all(p > 0.001))
        ((2,), True)
    """
    count = y.shape[0]
    close = (np.abs(y[:, None] - y[None, :]) <= epsilon).astype(np.float64)
    np.fill_diagonal(close, 0.0)
    pairs = count * (count - 1)
    c_full = float(close.sum()) / pairs
    if c_full <= 0.0 or c_full >= 1.0:
        raise NumericalError(
            f"a radius of {epsilon:g} leaves the correlation integral at {c_full:g}; choose "
            "epsilon between half and twice the standard deviation."
        )
    rows = close.sum(axis=1)
    k_full = float((rows**2 - rows).sum()) / (count * (count - 1) * (count - 2))
    statistics = np.empty(dimension - 1)
    pvalues = np.empty(dimension - 1)
    history = close.copy()
    for m in range(2, dimension + 1):
        history = history[:-1, :-1] * close[m - 1 :, m - 1 :]
        n_m = count - m + 1
        c_m = float(history.sum()) / (n_m * (n_m - 1))
        one = close[:n_m, :n_m]
        c_1 = float(one.sum()) / (n_m * (n_m - 1))
        variance = k_full**m + 2.0 * sum(k_full ** (m - j) * c_full ** (2 * j) for j in range(1, m))
        variance += (m - 1) ** 2 * c_full ** (2 * m) - m**2 * k_full * c_full ** (2 * m - 2)
        variance *= 4.0
        if variance <= 0.0:
            raise NumericalError(
                f"the BDS variance is not positive at dimension {m}; the radius is too extreme."
            )
        statistics[m - 2] = np.sqrt(n_m) * (c_m - c_1**m) / np.sqrt(variance)
        pvalues[m - 2] = 2.0 * norm.sf(abs(statistics[m - 2]))
    return statistics, pvalues


def _hansen_threshold(
    y: npt.NDArray[np.float64],
    order: int,
    delays: tuple[int, ...],
    *,
    trim: float,
    n_grid: int,
    replications: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int, npt.NDArray[np.float64]]:
    """Hansen's (1996) sup-F test of a threshold autoregression with simulated p-value.

    Under linearity the threshold is not identified, so the sup over the
    grid has no chi-squared law; Hansen replaces the response by standard
    normal draws with the regressors and candidate splits held fixed and
    reads the p-value off the simulated sup. Only the sup-statistic's
    numerator changes across draws, so each candidate split is reduced
    once to orthonormal bases of its two regime blocks and every draw is
    projected on all of them at once. The delay is searched jointly when
    several are passed, and the simulated sup ranges over the same
    delays, which is the size-correct treatment of a searched delay.

    Args:
        y: ``(T,)`` series.
        order: Autoregressive order.
        delays: Candidate delays of the threshold variable.
        trim: Fraction of the sorted threshold variable excluded at each
            end of the grid.
        n_grid: Candidate thresholds per delay, on the quantile grid.
        replications: Simulated sup statistics behind the p-value.
        rng: Random generator.

    Returns:
        ``(statistic, pvalue, threshold, delay, simulated)``; ``simulated``
        are the replicated sup statistics.

    Raises:
        NumericalError: If no admissible split exists or the linear fit is
            exact.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(200)
        >>> for t in range(1, 200):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal()
        >>> stat, p, r, d, sims = _hansen_threshold(
        ...     y, 1, (1,), trim=0.15, n_grid=50, replications=200, rng=rng
        ... )
        >>> d, sims.shape, bool(p > 0.01)
        (1, (200,), True)
    """
    n = y.shape[0]
    start = max(order, max(delays))
    target = y[start:]
    count = target.shape[0]
    design = np.column_stack([np.ones(count), lag_matrix(y, order, start=start)])
    width = design.shape[1]
    min_regime = width + 1
    q_full, _r = np.linalg.qr(design)
    ssr0 = float(target @ target) - float((q_full.T @ target) @ (q_full.T @ target))
    if ssr0 <= 1e-300:
        raise NumericalError("the linear autoregression fits exactly; the series is degenerate.")
    bases: list[npt.NDArray[np.float64]] = []
    labels: list[tuple[int, float]] = []
    for d in delays:
        z = y[start - d : n - d]
        for r in np.unique(np.quantile(z, np.linspace(trim, 1.0 - trim, n_grid))):
            lower = z <= r
            n_lo = int(lower.sum())
            if n_lo < min_regime or count - n_lo < min_regime:
                continue
            q_lo, _ = np.linalg.qr(design * lower[:, None])
            q_hi, _ = np.linalg.qr(design * (~lower)[:, None])
            bases.append(np.column_stack([q_lo, q_hi]))
            labels.append((d, float(r)))
    if not bases:
        raise NumericalError(
            "threshold grid search found no admissible split; relax trim or shorten order."
        )
    stack = np.stack(bases)  # (G, count, 2 width)
    fitted = np.einsum("gnk,n->gk", stack, target)
    ssr1 = float(target @ target) - np.einsum("gk,gk->g", fitted, fitted)
    ssr1 = np.maximum(ssr1, 1e-300)
    path = count * (ssr0 - ssr1) / ssr1
    best = int(np.argmax(path))
    statistic = float(path[best])
    draws = rng.standard_normal((count, replications))
    total = np.einsum("nr,nr->r", draws, draws)
    linear = q_full.T @ draws
    ssr0_sim = total - np.einsum("kr,kr->r", linear, linear)
    projected = np.einsum("gnk,nr->gkr", stack, draws)
    ssr1_sim = total[None, :] - np.einsum("gkr,gkr->gr", projected, projected)
    ssr1_sim = np.maximum(ssr1_sim, 1e-300)
    simulated = (count * (ssr0_sim[None, :] - ssr1_sim) / ssr1_sim).max(axis=0)
    pvalue = float((simulated >= statistic).mean())
    return statistic, pvalue, labels[best][1], labels[best][0], simulated


def _gph(y: npt.NDArray[np.float64], m: int) -> tuple[float, float]:
    """Geweke and Porter-Hudak's (1983) log-periodogram estimate of ``d`` with its standard error.

    ``log I(lambda_j) = c - d log(4 sin^2(lambda_j / 2)) + e_j`` over the
    first ``m`` Fourier frequencies; the error is asymptotically
    ``log chi^2_2 / 2`` with variance ``pi^2 / 6``, so the slope's
    standard error is ``sqrt(pi^2 / (6 sum (x_j - x_bar)^2))``, which is
    ``pi / sqrt(24 m)`` in the limit.

    Args:
        y: ``(T,)`` series.
        m: Fourier frequencies used.

    Returns:
        ``(d, se)``.

    Raises:
        NumericalError: If a periodogram ordinate is zero, so its log is
            undefined.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> d, se = _gph(rng.standard_normal(1000), 31)
        >>> bool(abs(d) < 3 * se), round(se, 3)
        (True, 0.137)
    """
    freqs, ordinates = periodogram(y)
    lam = freqs[:m]
    power = ordinates[:m]
    if np.any(power <= 0.0):
        raise NumericalError("a periodogram ordinate is zero; the series is degenerate.")
    regressor = np.log(4.0 * np.sin(lam / 2.0) ** 2)
    centered = regressor - regressor.mean()
    spread = float(centered @ centered)
    slope = float(centered @ (np.log(power) - np.log(power).mean())) / spread
    return -slope, float(np.sqrt(np.pi**2 / (6.0 * spread)))


def _exact_local_whittle(
    y: npt.NDArray[np.float64], m: int, *, bounds: tuple[float, float]
) -> tuple[float, float]:
    """Shimotsu and Phillips' (2005) exact local Whittle estimate of ``d`` with unknown mean.

    The Whittle objective is evaluated on the periodogram of ``(1 - L)^d
    (y - mu(d))`` rather than on ``lambda^{2d} I_y(lambda)``, which keeps
    the estimator consistent and ``N(0, 1/4)`` in ``sqrt(m)`` units for
    every ``d``, stationary or not, instead of only for ``d < 3/4``. The
    mean is Shimotsu's (2010) weighted estimate ``w(d) y_bar + (1 - w(d))
    y_1`` with ``w`` falling from one at ``d = 1/2`` to zero at ``d =
    3/4``, so that a unit-root series is anchored at its first value
    and a stationary one at its mean.

    Args:
        y: ``(T,)`` series.
        m: Fourier frequencies used.
        bounds: Search interval for ``d``.

    Returns:
        ``(d, se)`` with ``se = 1 / (2 sqrt(m))``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> walk = np.cumsum(rng.standard_normal(500))
        >>> d, se = _exact_local_whittle(walk, 56, bounds=(-0.5, 2.0))
        >>> bool(abs(d - 1.0) < 3 * se)
        True
    """
    n = y.shape[0]
    lam = 2.0 * np.pi * np.arange(1, m + 1) / n
    log_lam_mean = float(np.log(lam).mean())
    first = float(y[0])
    mean = float(y.mean())

    def weight(d: float) -> float:
        if d <= 0.5:
            return 1.0
        if d >= 0.75:
            return 0.0
        return 0.5 * (1.0 + np.cos(4.0 * np.pi * d))

    def objective(d: float) -> float:
        w = weight(d)
        centered = y - (w * mean + (1.0 - w) * first)
        differenced = np.convolve(centered, fractional_difference_weights(d, n))[:n]
        transform = np.fft.rfft(differenced)[1 : m + 1]
        power = (np.abs(transform) ** 2) / n
        g = float(power.mean())
        if not np.isfinite(g) or g <= 0.0:
            return 1e10
        return float(np.log(g) - 2.0 * d * log_lam_mean)

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(result.x), float(1.0 / (2.0 * np.sqrt(m)))


def _seasonal_frequencies(period: int) -> tuple[float, ...]:
    """Harmonic seasonal frequencies ``2 pi k / s`` for ``k = 1 .. s/2 - 1``.

    Example:
        >>> np.round(_seasonal_frequencies(4), 4)
        array([1.5708])
    """
    return tuple(2.0 * np.pi * k / period for k in range(1, period // 2))


def _hegy_regressors(y: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    """The HEGY level regressors at every ``t >= s - 1``, one block per unit root of ``1 - L^s``.

    Column ``0`` is ``sum_{j<s} y_{t-j}``, which keeps only the zero-frequency
    root; column ``1`` is ``sum_{j<s} cos((j + 1) pi) y_{t-j}``, which keeps
    only the Nyquist root; columns ``2k, 2k + 1`` for each harmonic
    ``theta_k = 2 pi k / s`` are ``sum_{j<s} cos((j + 1) theta_k) y_{t-j}``
    and ``-sum_{j<s} sin((j + 1) theta_k) y_{t-j}``, the pair that keeps
    the complex root at ``theta_k``. Each column is the series filtered by
    ``(1 - L^s)`` with that root's factor removed, so its coefficient in
    the HEGY regression is the test of that root alone. At ``s = 4`` the
    columns are Hylleberg et al.'s ``y_1, y_2, y_3(t-1), y_3(t)`` up to
    sign.

    Args:
        y: ``(T,)`` series.
        period: Seasonal period ``s``, even and at least 2.

    Returns:
        ``(T - s + 1, s)`` array, row ``i`` for ``t = i + s - 1``.

    Example:
        >>> r = _hegy_regressors(np.arange(1.0, 9.0), 4)
        >>> r.shape, float(r[0, 0]), float(r[0, 1])
        ((5, 4), 10.0, -2.0)
    """
    n = y.shape[0]
    rows = n - period + 1
    lags = np.column_stack([y[period - 1 - j : n - j] for j in range(period)])  # (rows, s)
    j = np.arange(1, period + 1)
    columns = [lags.sum(axis=1), lags @ np.cos(np.pi * j)]
    for theta in _seasonal_frequencies(period):
        columns.append(lags @ np.cos(theta * j))
        columns.append(-(lags @ np.sin(theta * j)))
    out = np.column_stack(columns)
    assert out.shape == (rows, period)
    return out


def _seasonal_trig_columns(period: int, nobs: int, *, start: int = 1) -> npt.NDArray[np.float64]:
    """Trigonometric seasonal regressors, a ``(cos, sin)`` pair per harmonic and ``(-1)^t`` last.

    Spans the same space as ``s - 1`` centered seasonal dummies, ordered
    by frequency so that a test of one frequency selects a contiguous
    block: two columns per harmonic, one for the Nyquist frequency.

    Args:
        period: Seasonal period ``s``.
        nobs: Rows.
        start: Time index of the first row.

    Returns:
        ``(nobs, s - 1)`` array.

    Example:
        >>> _seasonal_trig_columns(4, 4).round(6)
        array([[ 0.,  1., -1.],
               [-1.,  0.,  1.],
               [-0., -1., -1.],
               [ 1., -0.,  1.]])
    """
    t = np.arange(start, start + nobs, dtype=np.float64)
    columns = []
    for theta in _seasonal_frequencies(period):
        columns.append(np.cos(theta * t))
        columns.append(np.sin(theta * t))
    columns.append(np.cos(np.pi * t))
    return np.column_stack(columns)


def _seasonal_deterministics(
    period: int, nobs: int, trend: str, *, seasonal: bool, start: int
) -> npt.NDArray[np.float64]:
    """Deterministic block of a seasonal regression: trend terms, then seasonal terms.

    Example:
        >>> _seasonal_deterministics(4, 3, "c", seasonal=True, start=1).shape
        (3, 4)
    """
    blocks = [deterministic_columns(trend, nobs, start=start)]
    if seasonal:
        blocks.append(_seasonal_trig_columns(period, nobs, start=start))
    return np.column_stack(blocks)


def _hegy_design(
    y: npt.NDArray[np.float64], period: int, trend: str, lags: int, *, seasonal: bool, drop: int = 0
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Target, design and the column offset of the ``pi`` block in the HEGY regression.

    Args:
        y: ``(T,)`` series.
        period: Seasonal period ``s``.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        lags: Augmentation lags of the seasonal difference.
        seasonal: Whether seasonal dummies (in trigonometric form) enter.
        drop: Extra leading observations to discard, so that fits with
            different ``lags`` share a sample.

    Returns:
        ``(target, design, offset)`` with the ``s`` level regressors at
        ``design[:, offset:offset + s]``.

    Example:
        >>> target, design, offset = _hegy_design(np.arange(20.0), 4, "c", 1, seasonal=False)
        >>> target.shape, design.shape, offset
        ((15,), (15, 6), 1)
    """
    n = y.shape[0]
    start = period + max(lags, drop)
    target = y[start:] - y[start - period : n - period]
    count = target.shape[0]
    levels = _hegy_regressors(y, period)[start - period : start - period + count]
    seasonal_difference = y[period:] - y[:-period]
    augmentation = (
        lag_matrix(seasonal_difference, lags, start=start - period)
        if lags
        else np.zeros((count, 0))
    )
    deterministic = _seasonal_deterministics(
        period, count, trend, seasonal=seasonal, start=start + 1
    )
    return target, np.column_stack([deterministic, levels, augmentation]), deterministic.shape[1]


def _hegy_statistics(
    y: npt.NDArray[np.float64], period: int, trend: str, lags: int, *, seasonal: bool
) -> tuple[npt.NDArray[np.float64], int]:
    """HEGY statistics: ``t`` at the two real roots, ``F`` at each harmonic pair, and joint ``F``.

    ``(1 - L^s) y_t = d_t' gamma + sum_i pi_i x_{i,t-1} + sum_j phi_j
    (1 - L^s) y_{t-j} + e_t`` with ``x`` the :func:`_hegy_regressors`.
    The returned vector is ``[t(pi_0), t(pi_{s/2}), F_1, ..., F_{s/2-1},
    F_seasonal, F_all]`` where ``F_k`` tests the ``k``-th harmonic pair,
    ``F_seasonal`` every coefficient but ``pi_0`` and ``F_all`` every
    ``pi``. Under the null of a seasonal random walk none of these has a
    standard law, so p-values come from :func:`_hegy_null_draws`.

    Args:
        y: ``(T,)`` series.
        period: Seasonal period ``s``.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        lags: Augmentation lags of the seasonal difference.
        seasonal: Whether seasonal dummies (in trigonometric form) enter.

    Returns:
        ``(statistics, nobs)``.

    Raises:
        NumericalError: If the regression is exact or singular.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(rng.standard_normal(200))
        >>> stats, nobs = _hegy_statistics(y, 4, "c", 0, seasonal=True)
        >>> stats.shape, nobs
        ((5,), 196)
    """
    target, design, offset = _hegy_design(y, period, trend, lags, seasonal=seasonal)
    count, width = design.shape
    if count <= width:
        raise NumericalError(
            f"the HEGY regression has {width} regressors on {count} observations; shorten the lags."
        )
    beta, ssr = ols(design, target)
    df = count - width
    if ssr <= 1e-300:
        raise NumericalError("the HEGY regression fits exactly; the series is degenerate.")
    sigma2 = ssr / df
    try:
        inverse = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError as error:
        raise NumericalError(
            "the HEGY regression is singular; the series may be constant."
        ) from error
    pi = beta[offset : offset + period]
    covariance = sigma2 * inverse[offset : offset + period, offset : offset + period]

    def wald(indices: list[int]) -> float:
        block = covariance[np.ix_(indices, indices)]
        vector = pi[indices]
        return float(vector @ np.linalg.solve(block, vector)) / len(indices)

    statistics = [pi[0] / np.sqrt(covariance[0, 0]), pi[1] / np.sqrt(covariance[1, 1])]
    for k in range(period // 2 - 1):
        statistics.append(wald([2 + 2 * k, 3 + 2 * k]))
    statistics.append(wald(list(range(1, period))))
    statistics.append(wald(list(range(period))))
    return np.asarray(statistics, dtype=np.float64), count


def _select_hegy_lags(
    y: npt.NDArray[np.float64],
    period: int,
    trend: str,
    max_lags: int,
    method: str,
    *,
    seasonal: bool,
) -> int:
    """Choose the HEGY augmentation on the common sample the largest candidate leaves.

    Args:
        y: ``(T,)`` series.
        period: Seasonal period.
        trend: Deterministic specification.
        max_lags: Largest augmentation considered.
        method: ``"aic"``, ``"bic"`` (Gaussian criteria on the common
            sample) or ``"t-stat"`` (general-to-specific at the 10% level).
        seasonal: Whether seasonal dummies enter.

    Returns:
        The chosen number of lags.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(rng.standard_normal(200))
        >>> _select_hegy_lags(y, 4, "c", 4, "bic", seasonal=True)
        0
    """
    if method == "t-stat":
        for lags in range(max_lags, 0, -1):
            target, design, _ = _hegy_design(
                y, period, trend, lags, seasonal=seasonal, drop=max_lags
            )
            beta, ssr = ols(design, target)
            n_eff, k = design.shape
            sigma2 = ssr / (n_eff - k)
            inverse = np.linalg.pinv(design.T @ design)
            if abs(beta[-1]) / float(np.sqrt(sigma2 * inverse[-1, -1])) > 1.6449:
                return lags
        return 0
    best_lags, best_value = 0, np.inf
    for lags in range(max_lags + 1):
        target, design, _ = _hegy_design(y, period, trend, lags, seasonal=seasonal, drop=max_lags)
        _beta, ssr = ols(design, target)
        n_eff, k = design.shape
        penalty = 2.0 * k / n_eff if method == "aic" else k * np.log(n_eff) / n_eff
        value = np.log(max(ssr / n_eff, 1e-300)) + penalty
        if value < best_value:
            best_lags, best_value = lags, value
    return best_lags


def _hegy_null_draws(
    period: int,
    nobs: int,
    trend: str,
    *,
    seasonal: bool,
    replications: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Draws of the HEGY statistics under the seasonal random walk with the same deterministics.

    Simulating the null directly at the sample size in hand, rather than
    interpolating the published quarterly and monthly tables, gives every
    statistic its finite-sample law at once and works for any period;
    the augmentation is left at zero because under the null it changes
    the law only through the degrees of freedom.

    Args:
        period: Seasonal period.
        nobs: Length of the series the test was run on.
        trend: Deterministic specification.
        seasonal: Whether seasonal dummies enter.
        replications: Draws.
        rng: Random generator.

    Returns:
        ``(replications, s / 2 + 3)`` draws in the order of
        :func:`_hegy_statistics`.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> draws = _hegy_null_draws(4, 100, "c", seasonal=True, replications=50, rng=rng)
        >>> draws.shape, bool(np.quantile(draws[:, 0], 0.05) < -2.0)
        ((50, 5), True)
    """
    draws = np.empty((replications, period // 2 + 3))
    for r in range(replications):
        y = rng.standard_normal(nobs)
        for phase in range(period):
            y[phase::period] = np.cumsum(y[phase::period])
        draws[r], _ = _hegy_statistics(y, period, trend, 0, seasonal=seasonal)
    return draws


def _von_mises_draws(
    q: int, *, n_draws: int, grid: int, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Draws of the generalized von Mises law, ``integral ||B_q(r)||^2 dr`` over a Brownian bridge.

    The limit of the Canova-Hansen (and KPSS, at ``q = 1``) statistic.
    The 5% point at ``q = 1`` is 0.461 and at ``q = 2`` is 0.749, which
    the grid of 1000 points reproduces to two decimals.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> draws = _von_mises_draws(1, n_draws=2000, grid=500, rng=rng)
        >>> bool(0.40 < np.quantile(draws, 0.95) < 0.53)
        True
    """
    s = np.arange(1, grid + 1) / grid
    out = np.empty(n_draws)
    batch = max(1, int(2e6 // (grid * q)))
    for start in range(0, n_draws, batch):
        size = min(batch, n_draws - start)
        increments = rng.standard_normal((size, grid, q)) / np.sqrt(grid)
        motion = np.cumsum(increments, axis=1)
        bridge = motion - s[None, :, None] * motion[:, -1:, :]
        out[start : start + size] = np.sum(bridge**2, axis=(1, 2)) / grid
    return out


def _canova_hansen(
    y: npt.NDArray[np.float64],
    period: int,
    trend: str,
    lags: int,
    *,
    bandwidth: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], int]:
    """Canova and Hansen's (1995) LM statistics of stationary against unit-root seasonality.

    Regress ``y_t`` on the deterministic block (trend terms, the ``s - 1``
    trigonometric seasonal columns, ``lags`` lags of ``y``) and form, for
    the seasonal columns ``z_t`` of the frequency under test, the partial
    sums ``F_t = sum_{i<=t} z_i e_i``. The statistic ``T^{-2} sum_t F_t'
    Omega^{-1} F_t`` uses the Bartlett long-run covariance of ``z_t e_t``
    through ``bandwidth`` lags and converges to the generalized von
    Mises law with ``q`` the number of columns tested: one at the
    Nyquist frequency, two at each harmonic, ``s - 1`` jointly.

    Args:
        y: ``(T,)`` series.
        period: Seasonal period.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        lags: Lags of ``y`` among the regressors.
        bandwidth: Bartlett lags of the long-run covariance.

    Returns:
        ``(statistics, degrees, nobs)``: the statistics in the order
        ``[Nyquist, harmonic_1, ..., harmonic_{s/2-1}, joint]``, their
        ``q``, and the observations.

    Raises:
        NumericalError: If the long-run covariance is singular.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> t = np.arange(400)
        >>> y = np.cos(np.pi * t / 2) + rng.standard_normal(400)
        >>> stats, q, nobs = _canova_hansen(y, 4, "c", 0, bandwidth=4)
        >>> stats.shape, q.tolist(), nobs
        ((3,), [1, 2, 3], 400)
    """
    target = y[lags:]
    count = target.shape[0]
    deterministic = deterministic_columns(trend, count, start=lags + 1)
    seasonal = _seasonal_trig_columns(period, count, start=lags + 1)
    augmentation = lag_matrix(y, lags, start=lags) if lags else np.zeros((count, 0))
    design = np.column_stack([deterministic, seasonal, augmentation])
    beta, _ssr = ols(design, target)
    residual = target - design @ beta
    scores = seasonal * residual[:, None]
    meat = scores.T @ scores / count
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1)
        cross = scores[lag:].T @ scores[:-lag] / count
        meat += weight * (cross + cross.T)
    partial = np.cumsum(scores, axis=0)
    harmonics = period // 2 - 1
    blocks = (
        [[period - 2]] + [[2 * k, 2 * k + 1] for k in range(harmonics)] + [list(range(period - 1))]
    )
    statistics = np.empty(len(blocks))
    degrees = np.empty(len(blocks), dtype=np.int64)
    for i, indices in enumerate(blocks):
        omega = meat[np.ix_(indices, indices)]
        f = partial[:, indices]
        try:
            solved = np.linalg.solve(omega, f.T)
        except np.linalg.LinAlgError as error:
            raise NumericalError(
                "the long-run covariance of the seasonal scores is singular; the series may "
                "carry no variation at that frequency."
            ) from error
        statistics[i] = float(np.sum(f.T * solved)) / count**2
        degrees[i] = len(indices)
    return statistics, degrees, count


def _hamilton_regression(
    y: npt.NDArray[np.float64], horizon: int, lags: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Hamilton's (2018) regression filter: ``y_{t+h}`` on a constant and ``y_t, ..., y_{t-p+1}``.

    The fitted value is the trend at ``t + h`` -- what could be predicted
    ``h`` steps ahead from the level's own recent history -- and the
    residual is the cycle. Both start at observation ``h + p - 1``.

    Args:
        y: ``(T,)`` series.
        horizon: Steps ahead ``h``; 8 quarterly, 24 monthly.
        lags: Regressors ``p``; 4 quarterly, 12 monthly.

    Returns:
        ``(trend, cycle)`` each of length ``T - h - p + 1``.

    Raises:
        NumericalError: If the regression has no degrees of freedom.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(rng.standard_normal(200))
        >>> trend, cycle = _hamilton_regression(y, 8, 4)
        >>> trend.shape, bool(abs(cycle.mean()) < 1e-8)
        ((189,), True)
    """
    offset = horizon + lags - 1
    count = y.shape[0] - offset
    if count <= lags + 1:
        raise NumericalError(
            f"the Hamilton regression needs more than {offset + lags + 1} observations; got "
            f"{y.shape[0]}."
        )
    target = y[offset:]
    design = np.column_stack(
        [np.ones(count), *(y[lags - 1 - j : lags - 1 - j + count] for j in range(lags))]
    )
    beta, _ssr = ols(design, target)
    trend = design @ beta
    return trend, target - trend


def _beveridge_nelson(
    z: npt.NDArray[np.float64],
    resid: npt.NDArray[np.float64],
    ar: npt.NDArray[np.float64],
    ma: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Sum of all future forecasts of a demeaned ARMA series at each date, in companion form.

    For ``z_t = phi(L)^{-1} theta(L) e_t`` write the state ``x_t = (z_t,
    ..., z_{t-p+1}, e_t, ..., e_{t-q+1})``, ``z_t`` always present, with
    companion ``F``; then
    ``sum_{k>=1} E_t z_{t+k} = e_1' F (I - F)^{-1} x_t`` (Morley 2002).
    Added to the level, that sum is the Beveridge-Nelson trend; its
    negative is the cycle.

    Args:
        z: ``(n,)`` demeaned first differences.
        resid: ``(n,)`` innovations aligned with ``z``.
        ar: ``(p,)`` autoregressive coefficients.
        ma: ``(q,)`` moving-average coefficients.

    Returns:
        ``(n,)`` the forecast sums.

    Raises:
        NumericalError: If ``I - F`` is singular, which is a unit root in
            the differences.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> e = rng.standard_normal(300)
        >>> z = np.zeros(300)
        >>> for t in range(1, 300):
        ...     z[t] = 0.5 * z[t - 1] + e[t]
        >>> s = _beveridge_nelson(z, e, np.array([0.5]), np.zeros(0))
        >>> bool(np.allclose(s, z))
        True
    """
    p, q = ar.shape[0], ma.shape[0]
    if p + q == 0:
        return np.zeros_like(z)
    width = max(p, 1)
    size = width + q
    companion = np.zeros((size, size))
    companion[0, :p] = ar
    companion[0, width:] = ma
    for i in range(1, p):
        companion[i, i - 1] = 1.0
    for i in range(width + 1, size):
        companion[i, i - 1] = 1.0
    try:
        weights = np.linalg.solve((np.eye(size) - companion).T, companion[0])
    except np.linalg.LinAlgError as error:
        raise NumericalError(
            "the differenced series has a unit root; the Beveridge-Nelson trend is undefined."
        ) from error
    n = z.shape[0]
    state = np.zeros((n, size))
    for j in range(width):
        state[j:, j] = z[: n - j]
    for j in range(q):
        state[j:, width + j] = resid[: n - j]
    return state @ weights


def _candidate_turns(
    y: npt.NDArray[np.float64], window: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Local maxima and minima over a two-sided window of ``window`` observations.

    ``t`` is a candidate peak when ``y_t`` exceeds every ``y_{t +- k}``
    for ``k = 1 .. window``, and a candidate trough symmetrically; the
    first and last ``window`` observations cannot qualify.

    Args:
        y: ``(T,)`` series.
        window: Half-width ``K``; Harding-Pagan use 2 quarterly, Bry-
            Boschan 5 monthly.

    Returns:
        ``(peaks, troughs)`` as sorted index arrays.

    Example:
        >>> y = np.array([0.0, 1.0, 3.0, 1.0, 0.0, 2.0, 4.0, 2.0, 1.0])
        >>> peaks, troughs = _candidate_turns(y, 2)
        >>> peaks.tolist(), troughs.tolist()
        ([2, 6], [4])
    """
    n = y.shape[0]
    peaks, troughs = [], []
    for t in range(window, n - window):
        neighbours = np.concatenate([y[t - window : t], y[t + 1 : t + window + 1]])
        if np.all(y[t] > neighbours):
            peaks.append(t)
        elif np.all(y[t] < neighbours):
            troughs.append(t)
    return np.asarray(peaks, dtype=np.int64), np.asarray(troughs, dtype=np.int64)


def _alternate_turns(
    y: npt.NDArray[np.float64], peaks: npt.NDArray[np.int64], troughs: npt.NDArray[np.int64]
) -> list[tuple[int, bool]]:
    """Merge candidate peaks and troughs into a strictly alternating sequence.

    Of two or more consecutive peaks the highest survives, of consecutive
    troughs the lowest, so the sequence reads peak, trough, peak, ...

    Returns:
        ``[(index, is_peak), ...]`` in time order.

    Example:
        >>> y = np.array([0.0, 3.0, 2.0, 4.0, 0.0, 1.0])
        >>> _alternate_turns(y, np.array([1, 3]), np.array([4]))
        [(3, True), (4, False)]
    """
    turns = sorted([(int(i), True) for i in peaks] + [(int(i), False) for i in troughs])
    merged: list[tuple[int, bool]] = []
    for index, is_peak in turns:
        if merged and merged[-1][1] == is_peak:
            previous = merged[-1][0]
            better = y[index] > y[previous] if is_peak else y[index] < y[previous]
            if better:
                merged[-1] = (index, is_peak)
        else:
            merged.append((index, is_peak))
    return merged


def _enforce_durations(
    y: npt.NDArray[np.float64], turns: list[tuple[int, bool]], min_phase: int, min_cycle: int
) -> list[tuple[int, bool]]:
    """Drop turns until every phase lasts ``min_phase`` and every cycle ``min_cycle`` observations.

    Turns are removed in pairs, which keeps the sequence alternating: a
    phase shorter than the minimum loses both its ends, and a cycle
    shorter than the minimum loses the ends of its smaller-amplitude
    phase. Passes repeat until nothing changes, as in Harding-Pagan's
    BBQ.

    Example:
        >>> y = np.array([0.0, 3.0, 2.9, 3.1, 0.0, 1.0, 5.0, 1.0])
        >>> turns = [(1, True), (2, False), (3, True), (4, False), (6, True)]
        >>> _enforce_durations(y, turns, 2, 4)
        [(3, True), (4, False), (6, True)]
    """
    current = list(turns)
    while len(current) >= 2:
        short = next(
            (i for i in range(len(current) - 1) if current[i + 1][0] - current[i][0] < min_phase),
            None,
        )
        if short is not None:
            del current[short : short + 2]
            continue
        brief = next(
            (i for i in range(len(current) - 2) if current[i + 2][0] - current[i][0] < min_cycle),
            None,
        )
        if brief is None:
            break
        first = abs(y[current[brief + 1][0]] - y[current[brief][0]])
        second = abs(y[current[brief + 2][0]] - y[current[brief + 1][0]])
        start = brief if first <= second else brief + 1
        del current[start : start + 2]
    return current
