import numpy as np
import numpy.typing as npt


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


def psd_sqrt(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A matrix square root valid for symmetric positive-semidefinite input.

    Negative eigenvalues arising from round-off are clipped to zero, so a
    covariance that is singular or marginally indefinite still yields a usable
    factor rather than a NaN.

    Args:
        matrix: A symmetric positive-semidefinite matrix.

    Returns:
        A factor ``S`` with ``S @ S.T`` equal to ``matrix``.

    Raises:
        DimensionError: If ``matrix`` is not square.
        NumericalError: If the eigendecomposition is not finite.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise DimensionError(f"psd_sqrt requires a square matrix; got {matrix.shape}.")
    eigvals, eigvecs = np.linalg.eigh(matrix)
    if not np.all(np.isfinite(eigvals)):
        raise NumericalError("psd_sqrt eigendecomposition produced non-finite values.")
    root = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0.0, None)))
    return np.asarray(root, dtype=np.float64)


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


def forecast_ar(
    history: npt.NDArray[np.float64],
    ar_params: npt.NDArray[np.float64],
    const: float,
    trend_coeff: float,
    origin: int,
    h: int,
) -> npt.NDArray[np.float64]:
    """Recursive point forecast for an AR(p) mean model.

    Args:
        history: The observed series, most recent value last.
        ar_params: Autoregressive coefficients.
        const: Intercept, ``0.0`` when absent.
        trend_coeff: Linear-trend slope, ``0.0`` when absent.
        origin: Time index of the last observation, so the deterministic trend
            continues from the right point rather than restarting at 1.
        h: Forecast horizon.

    Returns:
        Point forecasts of shape ``(h,)``.
    """
    p = ar_params.shape[0]
    buf = list(history[-p:]) if p else []
    out = np.empty(h, dtype=np.float64)
    for step in range(h):
        value = const + trend_coeff * (origin + step + 1)
        for i in range(p):
            value += ar_params[i] * buf[-1 - i]
        out[step] = value
        buf.append(value)
    return out


def difference_series(
    y: npt.NDArray[np.float64], d: int, capital_d: int, s: int
) -> npt.NDArray[np.float64]:
    """Apply non-seasonal then seasonal differencing.

    Args:
        y: The series.
        d: Non-seasonal differencing order.
        capital_d: Seasonal differencing order.
        s: Seasonal period.

    Returns:
        The differenced series, shorter by ``d + s * capital_d``.
    """
    w = y
    if d > 0:
        w = difference(w, d)
    if capital_d > 0:
        w = seasonal_difference(w, s, capital_d)
    return w


def fractional_difference_weights(d: float, length: int) -> npt.NDArray[np.float64]:
    """First ``length`` coefficients of the operator ``(1 - L)**d``.

    The coefficients satisfy ``b_0 = 1`` and ``b_k = b_{k-1} * (k - 1 - d) / k``.
    For ``d > 0`` they are negative for ``k >= 1`` and decay like ``k**(-d-1)``,
    which is the slow decay that encodes long memory.

    Args:
        d: The fractional differencing order.
        length: Number of coefficients to return.

    Returns:
        The coefficients ``[b_0, ..., b_{length-1}]``.

    Raises:
        SpecificationError: If ``length < 1``.

    Example:
        >>> np.round(fractional_difference_weights(0.5, 4), 4)
        array([ 1.    , -0.5   , -0.125 , -0.0625])
    """
    if length < 1:
        raise SpecificationError(f"length must be >= 1; got {length}.")
    weights = np.empty(length, dtype=np.float64)
    weights[0] = 1.0
    for k in range(1, length):
        weights[k] = weights[k - 1] * (k - 1 - d) / k
    return weights


def fractional_difference(
    y: npt.ArrayLike, d: float, *, truncation: int | None = None
) -> npt.NDArray[np.float64]:
    """Apply the truncated fractional difference ``(1 - L)**d``.

    Uses all available history at each ``t``, so the series is not shortened.
    At the sample start the filter is necessarily truncated, inducing a
    transient conditioned on the same way CSS conditions on its first
    observations.

    Args:
        y: Input series (1-D array-like).
        d: Fractional differencing order.
        truncation: Maximum filter length; defaults to the series length.

    Returns:
        The fractionally differenced series, same length as ``y``.

    Raises:
        DimensionError: If ``y`` is not one-dimensional.
        NumericalError: If ``y`` contains non-finite values.
        SpecificationError: If ``truncation < 1``.

    Example:
        >>> np.round(fractional_difference(np.array([1.0, 2.0, 3.0, 4.0]), 1.0), 4)
        array([1., 1., 1., 1.])
    """
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise DimensionError(f"y must be one-dimensional; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise NumericalError("y contains non-finite values.")
    n = arr.shape[0]
    m = n if truncation is None else int(truncation)
    if m < 1:
        raise SpecificationError(f"truncation must be >= 1; got {m}.")
    weights = fractional_difference_weights(d, min(m, n))
    return np.convolve(arr, weights)[:n]


def ar_infinity(
    d: float,
    ar_params: npt.NDArray[np.float64],
    ma_params: npt.NDArray[np.float64],
    truncation: int,
) -> npt.NDArray[np.float64]:
    """Coefficients of ``Pi(L) = phi(L)(1 - L)**d / theta(L)``, in AR form.

    The model is ``Pi(L)(y_t - mu) = eps_t`` with ``Pi(L) = 1 - c_1 L - ...``,
    so the one-step recursion driving forecasts is
    ``(y_t - mu) = sum_j c_j (y_{t-j} - mu) + eps_t``.

    Args:
        d: Fractional differencing order.
        ar_params: Short-memory AR coefficients.
        ma_params: Short-memory MA coefficients.
        truncation: Number of coefficients to retain.

    Returns:
        The coefficients ``c_1, ..., c_truncation``.
    """
    frac = fractional_difference_weights(d, truncation + 1)
    phi_poly = np.concatenate([[1.0], -ar_params]) if ar_params.size else np.array([1.0])
    numerator = np.convolve(phi_poly, frac)[: truncation + 1]
    theta_poly = np.concatenate([[1.0], ma_params]) if ma_params.size else np.array([1.0])
    pi = np.zeros(truncation + 1, dtype=np.float64)
    q = ma_params.size
    for i in range(truncation + 1):
        acc = numerator[i]
        for j in range(1, min(i, q) + 1):
            acc -= theta_poly[j] * pi[i - j]
        pi[i] = acc
    return -pi[1:]


def backcast(resid: npt.NDArray[np.float64]) -> float:
    """Exponentially weighted pre-sample variance estimate.

    Args:
        resid: Mean residuals.

    Returns:
        The weighted mean squared residual over the first 75 observations.
    """
    tau = min(75, resid.shape[0])
    w = 0.94 ** np.arange(tau)
    w /= w.sum()
    return float(np.sum(w * resid[:tau] ** 2))


def softplus(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Map the real line to the positive half-line, overflow-safe."""
    return np.logaddexp(0.0, x)


def inv_softplus(x: float) -> float:
    """Inverse of :func:`_softplus`, for constructing starting values."""
    return float(np.log(np.expm1(x)))


def garch_variance(
    resid: npt.NDArray[np.float64],
    omega: float,
    alpha: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    backcast: float,
) -> npt.NDArray[np.float64]:
    """Variance recursion for the GARCH and GJR families.

    Pre-sample squared residuals and variances are replaced by ``backcast``;
    the asymmetry term uses half the backcast pre-sample, matching the
    unconditional probability of a negative shock.

    Args:
        resid: Mean residuals.
        omega: Variance intercept.
        alpha: ARCH coefficients.
        gamma: Asymmetry coefficients (empty for symmetric GARCH).
        beta: GARCH coefficients.
        backcast: Pre-sample variance.

    Returns:
        The conditional-variance path.
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


def egarch_variance(
    resid: npt.NDArray[np.float64],
    omega: float,
    alpha: npt.NDArray[np.float64],
    gamma: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    backcast: float,
) -> npt.NDArray[np.float64]:
    """Log-variance recursion for EGARCH.

    The magnitude term is centered by ``E|z| = sqrt(2 / pi)`` so that a
    standard-normal shock contributes zero drift to the log variance.

    Args:
        resid: Mean residuals.
        omega: Log-variance intercept.
        alpha: Magnitude coefficients.
        gamma: Sign (leverage) coefficients.
        beta: Log-variance persistence coefficients.
        backcast: Pre-sample variance (used in logs).

    Returns:
        The conditional-variance path, exponentiated back to levels.
    """
    n, p, o, q = resid.shape[0], alpha.size, gamma.size, beta.size
    ln_sigma2 = np.empty(n)
    ln_bc = float(np.log(backcast))
    e = np.zeros(n)
    for t in range(n):
        s = omega
        for i in range(p):
            s += alpha[i] * ((abs(e[t - 1 - i]) - _SQRT_2_OVER_PI) if t - 1 - i >= 0 else 0.0)
        for k in range(o):
            s += gamma[k] * (e[t - 1 - k] if t - 1 - k >= 0 else 0.0)
        for j in range(q):
            s += beta[j] * (ln_sigma2[t - 1 - j] if t - 1 - j >= 0 else ln_bc)
        ln_sigma2[t] = s
        sig = np.sqrt(np.exp(ln_sigma2[t]))
        e[t] = resid[t] / sig if sig > 0 else 0.0
    return np.exp(ln_sigma2)


def figarch_weights(phi: float, d: float, beta: float, truncation: int) -> npt.NDArray[np.float64]:
    """ARCH(infinity) lambda weights for FIGARCH(1, d, 1).

    Uses the Chung (1999) recursion, which is numerically better behaved than
    expanding the fractional operator and dividing polynomials.

    Args:
        phi: The ARCH weight.
        d: Fractional integration order.
        beta: The GARCH weight.
        truncation: Number of weights to generate.

    Returns:
        The weights ``lambda_1, ..., lambda_truncation``.
    """
    delta = -fractional_difference_weights(d, truncation + 1)[1:]
    lam = np.empty(truncation, dtype=np.float64)
    lam[0] = phi - beta + d
    for i in range(1, truncation):
        lam[i] = beta * lam[i - 1] + (delta[i] - phi * delta[i - 1])
    return lam


def figarch_variance(
    resid: npt.NDArray[np.float64],
    omega: float,
    phi: float,
    d: float,
    beta: float,
    backcast: float,
    truncation: int = _DEFAULT_TRUNCATION,
) -> npt.NDArray[np.float64]:
    """Variance recursion for FIGARCH via its ARCH(infinity) representation.

    Weights beyond the available history are applied to ``backcast``, so the
    truncation tail contributes a constant rather than being silently dropped.

    Args:
        resid: Mean residuals.
        omega: Variance intercept.
        phi: The ARCH weight.
        d: Fractional integration order.
        beta: The GARCH weight.
        backcast: Pre-sample variance.
        truncation: ARCH(infinity) truncation lag.

    Returns:
        The conditional-variance path.
    """
    n = resid.shape[0]
    lam = _figarch_weights(phi, d, beta, truncation)
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


def activation(z: npt.NDArray[np.float64], kind: str) -> npt.NDArray[np.float64]:
    """Hidden-layer activation."""
    if kind == "tanh":
        return np.tanh(z)
    return np.maximum(z, 0.0)


def activation_grad(z: npt.NDArray[np.float64], kind: str) -> npt.NDArray[np.float64]:
    """Derivative of :func:`_activation`."""
    if kind == "tanh":
        return 1.0 - np.tanh(z) ** 2
    return (z > 0.0).astype(np.float64)


def gaussian_llf(ssr: float, nobs: int) -> tuple[float, float]:
    """Concentrated Gaussian variance and log-likelihood from an SSR."""
    sigma2 = ssr / nobs
    return sigma2, -0.5 * nobs * (_LOG_2PI + np.log(sigma2) + 1.0)


def ms_lag_matrix(y: npt.NDArray[np.float64], order: int) -> npt.NDArray[np.float64]:
    """Lagged levels aligned to the effective sample ``t = p .. n - 1``.

    Distinct from :func:`cultivars._core._design.lag_matrix`: this variant
    returns a zero-width matrix of full length when ``order == 0``, which the
    EM loop relies on for the switching-intercept-only specification.

    Args:
        y: The series.
        order: Autoregressive order.

    Returns:
        An ``(n - order, order)`` array, or ``(n, 0)`` when ``order == 0``.
    """
    n = y.shape[0]
    if order == 0:
        return np.zeros((n, 0), dtype=np.float64)
    return np.column_stack([y[order - 1 - i : n - 1 - i] for i in range(order)])


def regime_means(
    intercepts: npt.NDArray[np.float64],
    ar_params: npt.NDArray[np.float64],
    lags: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Per-regime conditional means of shape ``(T_eff, K)``."""
    if lags.shape[1] == 0:
        return np.broadcast_to(intercepts, (lags.shape[0], intercepts.shape[0])).copy()
    return intercepts[None, :] + lags @ ar_params.T


def log_densities(
    target: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    sigma2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Gaussian log conditional densities of shape ``(T_eff, K)``."""
    resid = target[:, None] - means
    return -0.5 * (_LOG_2PI + np.log(sigma2)[None, :] + resid**2 / sigma2[None, :])


def update_transition(
    smoothed: npt.NDArray[np.float64],
    joint: npt.NDArray[np.float64],
    floor: float,
) -> npt.NDArray[np.float64]:
    """M-step transition update from expected transition counts.

    Probabilities are floored before renormalizing, so a regime that becomes
    momentarily unvisited can still be re-entered instead of being absorbed.

    Args:
        smoothed: Smoothed regime probabilities.
        joint: Smoothed joint probabilities ``Pr(S_t = i, S_{t+1} = j | y)``.
        floor: Minimum admissible probability.

    Returns:
        The updated row-stochastic transition matrix.
    """
    k = smoothed.shape[1]
    if joint.shape[0] == 0:
        return np.full((k, k), 1.0 / k)
    numer = joint.sum(axis=0)
    denom = smoothed[:-1].sum(axis=0)
    p = numer / np.clip(denom[:, None], floor, None)
    p = np.clip(p, floor, None)
    return p / p.sum(axis=1, keepdims=True)


def update_coefficients(
    target: npt.NDArray[np.float64],
    lags: npt.NDArray[np.float64],
    smoothed: npt.NDArray[np.float64],
    sigma2: npt.NDArray[np.float64],
    layout: _Layout,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """M-step coefficient update by responsibility-weighted GLS.

    All regimes are solved in one stacked system so that non-switching blocks
    are estimated jointly across regimes rather than per regime.

    Args:
        target: The effective sample.
        lags: Lagged levels.
        smoothed: Smoothed regime probabilities.
        sigma2: Current per-regime variances.
        layout: Column bookkeeping.

    Returns:
        A tuple ``(intercepts, ar_params)`` of shapes ``(K,)`` and ``(K, p)``.
    """
    n_eff = target.shape[0]
    k, p = layout.n_regimes, layout.order
    d = layout.width
    a = np.zeros((d, d), dtype=np.float64)
    b = np.zeros(d, dtype=np.float64)
    for j in range(k):
        design = np.zeros((n_eff, d), dtype=np.float64)
        design[:, layout.intercept_col(j)] = 1.0
        if p:
            design[:, layout.ar_slice(j)] = lags
        weight = smoothed[:, j] / sigma2[j]
        a += design.T @ (design * weight[:, None])
        b += design.T @ (weight * target)
    beta, *_ = np.linalg.lstsq(a, b, rcond=None)

    intercepts = np.empty(k, dtype=np.float64)
    ar_params = np.zeros((k, p), dtype=np.float64)
    for j in range(k):
        intercepts[j] = beta[layout.intercept_col(j)]
        if p:
            ar_params[j] = beta[layout.ar_slice(j)]
    return intercepts, ar_params


def update_variance(
    target: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    smoothed: npt.NDArray[np.float64],
    switching_variance: bool,
    floor: float,
) -> npt.NDArray[np.float64]:
    """M-step variance update, per regime or pooled.

    Args:
        target: The effective sample.
        means: Per-regime conditional means.
        smoothed: Smoothed regime probabilities.
        switching_variance: Whether the variance switches across regimes.
        floor: Minimum admissible variance.

    Returns:
        Per-regime variances of shape ``(K,)``.
    """
    k = smoothed.shape[1]
    sq = (target[:, None] - means) ** 2
    if switching_variance:
        sigma2 = (smoothed * sq).sum(axis=0) / np.clip(smoothed.sum(axis=0), 1e-12, None)
    else:
        sigma2 = np.full(k, float((smoothed * sq).sum() / target.shape[0]))
    return np.clip(sigma2, floor, None)


def initial_transition(
    k: int, rng: np.random.Generator, diagonal: float
) -> npt.NDArray[np.float64]:
    """Randomized persistent transition matrix for a restart.

    Args:
        k: Number of regimes.
        rng: Random generator.
        diagonal: Target self-transition probability.

    Returns:
        A row-stochastic ``(K, K)`` matrix.
    """
    off = (1.0 - diagonal) / (k - 1)
    p = np.full((k, k), off) + (diagonal - off) * np.eye(k)
    p = p * rng.uniform(0.9, 1.1, size=(k, k))
    return p / p.sum(axis=1, keepdims=True)


def ar_design(y: npt.NDArray[np.float64], order: int, start: int) -> npt.NDArray[np.float64]:
    """Intercept plus lagged levels, aligned to begin at ``start``.

    Args:
        y: The series.
        order: AR order per regime.
        start: First retained time index.

    Returns:
        An ``(n - start, order + 1)`` design matrix.
    """
    return np.column_stack(
        [
            deterministic_columns("c", y.shape[0] - start),
            lag_matrix(y, order, start=start),
        ]
    )
