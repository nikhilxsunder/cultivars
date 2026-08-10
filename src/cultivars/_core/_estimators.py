from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._defaults import _LOG_2PI


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
