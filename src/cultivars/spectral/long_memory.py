import numpy as np
import numpy.typing as npt


def gph(
    endog: npt.ArrayLike, *, m: int | None = None, bandwidth_exponent: float = 0.5
) -> LongMemoryEstimate:
    """Estimate ``d`` by the Geweke-Porter-Hudak log-periodogram regression.

    Regresses ``log I(lambda_j)`` on ``log(4 sin^2(lambda_j / 2))`` over the
    lowest ``m`` Fourier frequencies; ``d`` is minus the slope. The error
    variance is asymptotically ``pi^2 / 6``, giving the reported standard error.

    Args:
        endog: The series (1-D array-like).
        m: Number of frequencies (bandwidth). Defaults to ``floor(n**exponent)``.
        bandwidth_exponent: Exponent for the default bandwidth (typically 0.5).

    Returns:
        A :class:`LongMemoryEstimate`.

    Raises:
        SpecificationError: If ``m`` is too small.
        NumericalError: If the regression is degenerate.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(rng.standard_normal(4096))     # d = 1 process
        >>> est = gph(np.diff(y))                        # differenced -> d ~ 0
        >>> abs(est.d) < 0.2
        True
    """
    y = _validate_series(endog)
    freqs, periodogram = _periodogram(y)
    m_eff = min(_bandwidth(y.shape[0], m, bandwidth_exponent), freqs.shape[0])
    lam = freqs[:m_eff]
    power = periodogram[:m_eff]
    if np.any(power <= 0.0):
        raise NumericalError("periodogram has non-positive ordinates; cannot take logs.")
    regressor = np.log(4.0 * np.sin(lam / 2.0) ** 2)
    response = np.log(power)
    centered = regressor - regressor.mean()
    denom = float(centered @ centered)
    if denom <= 0.0:
        raise NumericalError("degenerate GPH regression (zero regressor variance).")
    slope = float(centered @ (response - response.mean()) / denom)
    d_hat = -slope
    se = float(np.sqrt((np.pi ** 2 / 6.0) / denom))
    return LongMemoryEstimate(d=d_hat, se=se, n_freq=m_eff, method="gph")


def local_whittle(
    endog: npt.ArrayLike, *, m: int | None = None, bandwidth_exponent: float = 0.65
) -> LongMemoryEstimate:
    """Estimate ``d`` by the Robinson (1995) Gaussian semiparametric estimator.

    Minimizes the local Whittle objective
    ``R(d) = log( m^{-1} sum_j lambda_j^{2d} I(lambda_j) )
    - (2d / m) sum_j log lambda_j`` over ``|d| < 0.5``. The estimator is
    ``sqrt(m)``-consistent with asymptotic variance ``1 / 4``.

    Args:
        endog: The series (1-D array-like).
        m: Number of frequencies (bandwidth). Defaults to ``floor(n**exponent)``.
        bandwidth_exponent: Exponent for the default bandwidth (typically ~0.65).

    Returns:
        A :class:`LongMemoryEstimate`.

    Example:
        >>> rng = np.random.default_rng(1)
        >>> y = np.cumsum(rng.standard_normal(4096))
        >>> est = local_whittle(np.diff(y))
        >>> abs(est.d) < 0.2
        True
    """
    y = _validate_series(endog)
    freqs, periodogram = _periodogram(y)
    m_eff = min(_bandwidth(y.shape[0], m, bandwidth_exponent), freqs.shape[0])
    lam = freqs[:m_eff]
    power = periodogram[:m_eff]
    log_lam = np.log(lam)

    def objective(d: float) -> float:
        g = np.mean(lam ** (2.0 * d) * power)
        if not np.isfinite(g) or g <= 0.0:
            return 1e10
        return float(np.log(g) - 2.0 * d * log_lam.mean())

    result = minimize_scalar(objective, bounds=(-_D_MAX, _D_MAX), method="bounded")
    d_hat = float(result.x)
    se = float(1.0 / (2.0 * np.sqrt(m_eff)))
    return LongMemoryEstimate(d=d_hat, se=se, n_freq=m_eff, method="local_whittle")
