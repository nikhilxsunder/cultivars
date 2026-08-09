from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class _ARFit:
    """Raw outputs of an autoregressive fit, before public result assembly.

    Attributes:
        const: Intercept, or ``None`` when ``trend == "n"``.
        trend_coeff: Linear-trend slope, or ``None`` unless ``trend == "ct"``.
        ar_params: Autoregressive coefficients.
        sigma2: Innovation variance.
        llf: Maximized log-likelihood (conditional for CSS, exact otherwise).
        nobs: Observations the likelihood was evaluated on.
        resid: One-step residuals.
        fittedvalues: One-step fitted values.
    """

    const: float | None
    trend_coeff: float | None
    ar_params: npt.NDArray[np.float64]
    sigma2: float
    llf: float
    nobs: int
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]


def _fit_ar_css(y: npt.NDArray[np.float64], order: int, trend: str) -> _ARFit:
    """Fit an AR(p) mean model by conditional least squares.

    Conditions on the first ``p`` observations, so the effective sample is
    ``n - p`` and the reported likelihood is the conditional one.

    Args:
        y: The endogenous series.
        order: Autoregressive order ``p``.
        trend: Deterministic specification (``"n"``, ``"c"``, ``"ct"``).

    Returns:
        The packed :class:`_ARFit`.
    """
    target, regressors, eff = css_design(y, order, trend)
    beta, _res, _rank, _sv = np.linalg.lstsq(regressors, target, rcond=None)
    fitted = regressors @ beta
    resid = target - fitted
    sigma2 = float(resid @ resid) / eff
    n_det = n_deterministic(trend)
    const = float(beta[0]) if trend in ("c", "ct") else None
    trend_coeff = float(beta[1]) if trend == "ct" else None
    ar_params = np.asarray(beta[n_det:], dtype=np.float64)
    llf = -0.5 * eff * (_LOG_2PI + np.log(sigma2) + 1.0)
    return _ARFit(const, trend_coeff, ar_params, sigma2, float(llf), eff, resid, fitted)


def _fit_ar_exact(y: npt.NDArray[np.float64], order: int, trend: str) -> _ARFit:
    """Fit an AR(p) mean model by exact maximum likelihood.

    Builds the companion state-space form and maximizes the Kalman likelihood,
    warm-started from the CSS fit. If the CSS estimate is explosive it is
    discarded in favour of a zero start, since an explosive warm start puts the
    optimizer outside the stationary region the reparameterization assumes.

    Args:
        y: The endogenous series.
        order: Autoregressive order ``p``.
        trend: Deterministic specification; ``"ct"`` is not supported.

    Returns:
        The packed :class:`_ARFit`, using the full sample.

    Raises:
        SpecificationError: If ``trend == "ct"``.
    """
    if trend == "ct":
        raise SpecificationError(
            "exact ML with trend='ct' is not supported in this release; "
            "use method='css' for a linear trend."
        )
    has_const = trend == "c"
    p = order
    e1 = np.zeros(p, dtype=np.float64)
    e1[0] = 1.0
    design = np.zeros((1, p), dtype=np.float64)
    design[0, 0] = 1.0
    obs_cov = np.zeros((1, 1), dtype=np.float64)
    selection = e1.reshape(p, 1)
    identity = np.eye(p, dtype=np.float64)

    warm = _fit_ar_css(y, order, trend)
    phi0 = warm.ar_params
    if not assess_stability(phi0).is_stable:
        phi0 = np.zeros(p, dtype=np.float64)
    psi0 = pack_stationary(phi0)
    log_sigma0 = np.log(warm.sigma2)
    theta0 = (
        np.concatenate([[warm.const or 0.0], psi0, [log_sigma0]])
        if has_const
        else np.concatenate([psi0, [log_sigma0]])
    )

    def unpack(
        theta: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64], float]:
        if has_const:
            const = float(theta[0])
            psi = theta[1 : 1 + p]
            log_sigma2 = float(theta[1 + p])
        else:
            const = 0.0
            psi = theta[:p]
            log_sigma2 = float(theta[p])
        return const, unpack_stationary(psi), log_sigma2

    def build(
        const: float, phi: npt.NDArray[np.float64], sigma2: float
    ) -> LinearGaussianStateSpace:
        transition = companion_matrix(phi)
        state_cov = np.array([[sigma2]], dtype=np.float64)
        state_intercept = const * e1
        initial_state = (
            np.linalg.solve(identity - transition, state_intercept)
            if has_const
            else np.zeros(p, dtype=np.float64)
        )
        return LinearGaussianStateSpace(
            design,
            obs_cov,
            transition,
            selection,
            state_cov,
            state_intercept=state_intercept,
            initial_state=initial_state,
        )

    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        const, phi, log_sigma2 = unpack(theta)
        try:
            return -build(const, phi, float(np.exp(log_sigma2))).loglikelihood(y)
        except (NumericalError, np.linalg.LinAlgError):
            return 1e10

    result = minimize(negloglik, theta0, method="L-BFGS-B")
    const, phi, log_sigma2 = unpack(np.asarray(result.x, dtype=np.float64))
    sigma2 = float(np.exp(log_sigma2))
    filtered = build(const, phi, sigma2).filter(y)
    fitted = filtered.predicted_state[:, 0].copy()
    resid = y - fitted
    return _ARFit(
        const if has_const else None,
        None,
        phi,
        sigma2,
        -float(result.fun),
        y.shape[0],
        resid,
        fitted,
    )


@dataclass(frozen=True, slots=True)
class _SARIMAXFit:
    """Raw outputs of a seasonal ARIMA-with-regressors fit."""

    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    seasonal_ar_params: npt.NDArray[np.float64]
    seasonal_ma_params: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    sigma2: float
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


def _arma_state_space(
    phi_star: npt.NDArray[np.float64],
    theta_star: npt.NDArray[np.float64],
    sigma2: float,
    obs_intercept: npt.NDArray[np.float64],
) -> LinearGaussianStateSpace:
    """Build the Harvey state-space form of an ARMA process.

    The state dimension is ``max(p, q + 1)``, which is the minimal realization:
    a larger companion would be observationally equivalent but would make the
    Kalman recursion carry redundant states.

    Args:
        phi_star: Expanded AR coefficients.
        theta_star: Expanded MA coefficients.
        sigma2: Innovation variance.
        obs_intercept: Per-observation mean shift from the regression block.

    Returns:
        The configured :class:`LinearGaussianStateSpace`.
    """
    r = max(phi_star.size, theta_star.size + 1)
    phi_full = np.zeros(r)
    phi_full[: phi_star.size] = phi_star
    transition = np.zeros((r, r))
    transition[:, 0] = phi_full
    for i in range(r - 1):
        transition[i, i + 1] = 1.0
    selection = np.zeros((r, 1))
    selection[0, 0] = 1.0
    selection[1 : 1 + theta_star.size, 0] = theta_star
    design = np.zeros((1, r))
    design[0, 0] = 1.0
    return LinearGaussianStateSpace(
        design,
        np.zeros((1, 1)),
        transition,
        selection,
        np.array([[sigma2]]),
        obs_intercept=obs_intercept.reshape(-1, 1),
    )


def _fit_sarimax(
    endog: npt.NDArray[np.float64],
    exog: npt.NDArray[np.float64] | None,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    trend: str,
) -> _SARIMAXFit:
    """Fit a seasonal ARIMA with optional regressors by exact ML.

    Starting values come from a regression of the differenced series on the
    deterministic and exogenous block, then a short AR fit to those residuals;
    an explosive AR start is replaced by zeros.

    Args:
        endog: The endogenous series.
        exog: Optional exogenous regressors, differenced alongside ``endog``.
        order: Non-seasonal ``(p, d, q)``.
        seasonal_order: Seasonal ``(P, D, Q, s)``.
        trend: Deterministic specification.

    Returns:
        The packed :class:`_SARIMAXFit` on the differenced modeling series.
    """
    p, d, q = order
    cap_p, cap_d, cap_q, s = seasonal_order
    w = _difference_series(endog, d, cap_d, s)
    n_eff = w.shape[0]
    det = deterministic_columns(trend, n_eff)
    if exog is not None:
        exog_w = _difference_series(exog, d, cap_d, s) if (d or cap_d) else exog
        design_x = np.column_stack([det, exog_w]) if det.shape[1] else exog_w
    else:
        design_x = det
    k_beta = design_x.shape[1]

    if k_beta:
        beta0, _r, _rk, _sv = np.linalg.lstsq(design_x, w, rcond=None)
        resid0 = w - design_x @ beta0
    else:
        beta0 = np.zeros(0)
        resid0 = w - w.mean()
    sigma2_0 = max(float(resid0 @ resid0) / n_eff, 1e-8)
    ar0 = np.zeros(p)
    if p:
        lag_mat = np.column_stack([resid0[p - i - 1 : n_eff - i - 1] for i in range(p)])
        tgt = resid0[p:]
        try:
            ar0 = np.asarray(np.linalg.lstsq(lag_mat, tgt, rcond=None)[0], dtype=np.float64)
            if not assess_stability(ar0).is_stable:
                ar0 = np.zeros(p)
        except np.linalg.LinAlgError:
            ar0 = np.zeros(p)

    psi = np.concatenate(
        [
            beta0,
            pack_stationary(ar0),
            np.zeros(cap_p),
            np.zeros(q),
            np.zeros(cap_q),
            [np.log(sigma2_0)],
        ]
    )
    idx = np.cumsum([k_beta, p, cap_p, q, cap_q])

    def unpack(
        vec: npt.NDArray[np.float64],
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        beta = vec[: idx[0]]
        phi = unpack_stationary(vec[idx[0] : idx[1]]) if p else np.zeros(0)
        sphi = unpack_stationary(vec[idx[1] : idx[2]]) if cap_p else np.zeros(0)
        theta_c = -unpack_stationary(vec[idx[2] : idx[3]]) if q else np.zeros(0)
        stheta = -unpack_stationary(vec[idx[3] : idx[4]]) if cap_q else np.zeros(0)
        return beta, phi, sphi, theta_c, stheta, float(np.exp(vec[idx[4]]))

    def obs_intercept(beta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return design_x @ beta if k_beta else np.zeros(n_eff)

    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        beta, phi, sphi, theta_c, stheta, sigma2 = unpack(theta)
        try:
            ss = _arma_state_space(
                expand_ar(phi, sphi, s),
                expand_ma(theta_c, stheta, s),
                sigma2,
                obs_intercept(beta),
            )
            return -ss.loglikelihood(w)
        except (NumericalError, np.linalg.LinAlgError):
            return 1e10

    result = minimize(negloglik, psi, method="L-BFGS-B")
    beta, phi, sphi, theta_c, stheta, sigma2 = unpack(np.asarray(result.x, dtype=np.float64))
    ss = _arma_state_space(
        expand_ar(phi, sphi, s), expand_ma(theta_c, stheta, s), sigma2, obs_intercept(beta)
    )
    fitted = ss.filter(w).predicted_state[:, 0] + obs_intercept(beta)
    resid = w - fitted
    return _SARIMAXFit(
        ar_params=phi,
        ma_params=theta_c,
        seasonal_ar_params=sphi,
        seasonal_ma_params=stheta,
        beta=beta,
        sigma2=sigma2,
        resid=resid,
        fittedvalues=fitted,
        llf=-float(result.fun),
        nobs=n_eff,
        n_params=k_beta + p + cap_p + q + cap_q + 1,
    )


@dataclass(frozen=True, slots=True)
class _ARFIMAFit:
    """Raw outputs of a fractionally integrated ARMA fit."""

    d: float
    mean: float | None
    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    sigma2: float
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


def _fit_arfima(
    y: npt.NDArray[np.float64], p: int, q: int, estimate_mean: bool, truncation: int
) -> _ARFIMAFit:
    """Fit an ARFIMA(p, d, q) by joint maximum likelihood.

    Warm-starts ``d`` from a local Whittle estimate, falling back to zero if
    that estimator fails; the ARMA block starts at zero, since a short-memory
    warm start fitted before ``d`` is known tends to absorb long memory.

    Args:
        y: The series.
        p: Short-memory AR order.
        q: Short-memory MA order.
        estimate_mean: Whether to estimate a mean ``mu``.
        truncation: Fractional-filter length.

    Returns:
        The packed :class:`_ARFIMAFit` on the fractionally differenced series.
    """
    n = y.shape[0]
    try:
        d0 = float(np.clip(local_whittle_d(y)[0], -_D_MAX + 1e-3, _D_MAX - 1e-3))
    except (NumericalError, SpecificationError):
        d0 = 0.0
    mu0 = float(y.mean()) if estimate_mean else 0.0
    w0 = fractional_difference(y - mu0, d0, truncation=truncation)
    log_sigma0 = float(np.log(max(float(np.var(w0)), 1e-8)))

    raw_d0 = float(np.arctanh(d0 / _D_MAX))
    parts: list[npt.NDArray[np.float64]] = []
    if estimate_mean:
        parts.append(np.array([mu0]))
    parts.extend([np.array([raw_d0]), np.zeros(p), np.zeros(q), np.array([log_sigma0])])
    theta0 = np.concatenate(parts)

    offset = 1 if estimate_mean else 0
    i_d = offset
    i_ar = i_d + 1
    i_ma = i_ar + p
    i_sig = i_ma + q

    def unpack(
        theta: npt.NDArray[np.float64],
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        mu = float(theta[0]) if estimate_mean else 0.0
        d = _D_MAX * float(np.tanh(theta[i_d]))
        phi = unpack_stationary(theta[i_ar:i_ma]) if p else np.zeros(0)
        theta_c = -unpack_stationary(theta[i_ma:i_sig]) if q else np.zeros(0)
        return mu, d, phi, theta_c, float(np.exp(theta[i_sig]))

    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        mu, d, phi, theta_c, sigma2 = unpack(theta)
        try:
            w = fractional_difference(y - mu, d, truncation=truncation)
            ss = _arma_state_space(phi, theta_c, sigma2, np.zeros(n))
            return -ss.loglikelihood(w)
        except (NumericalError, np.linalg.LinAlgError, ValueError):
            return 1e10

    result = minimize(negloglik, theta0, method="L-BFGS-B")
    mu, d, phi, theta_c, sigma2 = unpack(np.asarray(result.x, dtype=np.float64))
    w = fractional_difference(y - mu, d, truncation=truncation)
    ss = _arma_state_space(phi, theta_c, sigma2, np.zeros(n))
    fitted = ss.filter(w).predicted_state[:, 0]
    return _ARFIMAFit(
        d=d,
        mean=mu if estimate_mean else None,
        ar_params=phi,
        ma_params=theta_c,
        sigma2=sigma2,
        resid=w - fitted,
        fittedvalues=fitted,
        llf=-float(result.fun),
        nobs=n,
        n_params=offset + 1 + p + q + 1,
    )


@dataclass(frozen=True, slots=True)
class _GARCHFit:
    """Raw outputs of a conditional-variance fit.

    Attributes:
        const: Mean intercept, or ``None`` when ``mean == "zero"``.
        ar_params: Conditional-mean AR coefficients (empty when ``ar_lags == 0``).
        omega: Variance intercept.
        alpha: ARCH coefficients (the ``phi`` weight for FIGARCH).
        gamma: Asymmetry coefficients.
        beta: GARCH coefficients.
        fractional_d: Fractional integration order (FIGARCH only).
        conditional_variance: The fitted variance path.
        resid: Mean residuals.
        fittedvalues: Fitted conditional means.
        llf: Maximized Gaussian log-likelihood.
        nobs: Effective observations.
        n_params: Free parameter count.
    """

    const: float | None
    ar_params: npt.NDArray[np.float64]
    omega: float
    alpha: npt.NDArray[np.float64]
    gamma: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    fractional_d: float | None
    conditional_variance: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


def _fit_garch(
    endog: npt.NDArray[np.float64],
    p: int,
    o: int,
    q: int,
    ar_lags: int,
    include_const: bool,
    vol: str,
) -> _GARCHFit:
    """Fit a GARCH, GJR, or EGARCH model by Gaussian maximum likelihood.

    Mean and variance parameters are estimated jointly rather than in two
    steps, so the reported likelihood is the true joint one.

    Args:
        endog: The series, typically returns or residuals.
        p: ARCH order.
        o: Asymmetry order.
        q: GARCH order.
        ar_lags: Conditional-mean AR order.
        include_const: Whether to estimate a mean intercept.
        vol: Volatility family (not ``"FIGARCH"``; see :func:`_fit_figarch`).

    Returns:
        The packed :class:`_GARCHFit`.
    """
    y = endog
    n_full = y.shape[0]
    start = ar_lags
    target = y[start:]
    n = target.shape[0]
    mean_cols: list[npt.NDArray[np.float64]] = []
    if include_const:
        mean_cols.append(np.ones(n))
    for i in range(1, ar_lags + 1):
        mean_cols.append(y[start - i : n_full - i])
    mean_x = np.column_stack(mean_cols) if mean_cols else np.zeros((n, 0))
    k_mean = mean_x.shape[1]

    if k_mean:
        mean0 = np.linalg.lstsq(mean_x, target, rcond=None)[0]
        resid0 = target - mean_x @ mean0
    else:
        mean0 = np.zeros(0)
        resid0 = target.copy()
    var0 = max(float(np.var(resid0)), 1e-8)
    backcast = _backcast(resid0)

    a_init, b_init, g_init = 0.05, 0.90, 0.05
    if vol == "GARCH":
        var_raw0 = np.concatenate(
            [
                [np.log(var0 * (1 - a_init - b_init))],
                [_inv_softplus(a_init)] * p,
                [_inv_softplus(b_init)] * q,
            ]
        )
    elif vol == "GJR":
        var_raw0 = np.concatenate(
            [
                [np.log(var0 * (1 - a_init - b_init - 0.5 * g_init))],
                [_inv_softplus(a_init)] * p,
                [g_init] * o,
                [_inv_softplus(b_init)] * q,
            ]
        )
    else:
        var_raw0 = np.concatenate(
            [
                [np.log(var0) * (1 - 0.95)],
                [0.1] * p,
                [-0.05] * o,
                [0.95] * q,
            ]
        )
    theta0 = np.concatenate([mean0, var_raw0])
    m_idx = k_mean

    def unpack_var(
        v: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if vol == "GARCH":
            return (
                float(np.exp(v[0])),
                _softplus(v[1 : 1 + p]),
                np.zeros(0),
                _softplus(v[1 + p : 1 + p + q]),
            )
        if vol == "GJR":
            return (
                float(np.exp(v[0])),
                _softplus(v[1 : 1 + p]),
                v[1 + p : 1 + p + o],
                _softplus(v[1 + p + o : 1 + p + o + q]),
            )
        return (
            float(v[0]),
            v[1 : 1 + p],
            v[1 + p : 1 + p + o],
            v[1 + p + o : 1 + p + o + q],
        )

    def variance_path(
        resid: npt.NDArray[np.float64],
        omega: float,
        alpha: npt.NDArray[np.float64],
        gamma: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        if vol == "EGARCH":
            return _egarch_variance(resid, omega, alpha, gamma, beta, backcast)
        return _garch_variance(resid, omega, alpha, gamma, beta, backcast)

    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        mean = theta[:m_idx]
        resid = target - mean_x @ mean if k_mean else target
        omega, alpha, gamma, beta = unpack_var(theta[m_idx:])
        if vol == "EGARCH":
            if abs(beta.sum()) >= 0.999:
                return 1e10
        elif alpha.sum() + 0.5 * gamma.sum() + beta.sum() >= 0.999:
            return 1e10
        sigma2 = variance_path(resid, omega, alpha, gamma, beta)
        if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
            return 1e10
        ll = -0.5 * np.sum(_LOG_2PI + np.log(sigma2) + resid**2 / sigma2)
        return float(-ll) if np.isfinite(ll) else 1e10

    result = minimize(negloglik, theta0, method="L-BFGS-B")
    theta = np.asarray(result.x, dtype=np.float64)
    mean = theta[:m_idx]
    fitted = mean_x @ mean if k_mean else np.zeros(n)
    resid = target - fitted
    omega, alpha, gamma, beta = unpack_var(theta[m_idx:])
    sigma2 = variance_path(resid, omega, alpha, gamma, beta)
    return _GARCHFit(
        const=float(mean[0]) if include_const else None,
        ar_params=np.asarray(mean[1:] if include_const else mean, dtype=np.float64),
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        fractional_d=None,
        conditional_variance=sigma2,
        resid=resid,
        fittedvalues=fitted,
        llf=-float(result.fun),
        nobs=n,
        n_params=k_mean + 1 + p + o + q,
    )


def _fit_figarch(
    endog: npt.NDArray[np.float64],
    include_const: bool,
    truncation: int = _DEFAULT_TRUNCATION,
) -> _GARCHFit:
    """Fit a FIGARCH(1, d, 1) model by Gaussian maximum likelihood.

    Admissibility is checked on a short prefix of the lambda weights each
    iteration: a negative weight implies a negative conditional variance
    somewhere in the sample, so the draw is rejected before the full
    recursion runs.

    Args:
        endog: The series, typically returns or residuals.
        include_const: Whether to estimate a mean intercept.
        truncation: ARCH(infinity) truncation lag.

    Returns:
        The packed :class:`_GARCHFit`, with ``fractional_d`` populated.
    """
    n = endog.shape[0]
    mean_x = np.ones((n, 1)) if include_const else np.zeros((n, 0))
    k_mean = mean_x.shape[1]
    if k_mean:
        mean0 = np.linalg.lstsq(mean_x, endog, rcond=None)[0]
        resid0 = endog - mean_x @ mean0
    else:
        mean0 = np.zeros(0)
        resid0 = endog.copy()
    var0 = max(float(np.var(resid0)), 1e-8)
    backcast = _backcast(resid0)
    theta0 = np.concatenate([mean0, [np.log(var0 * 0.4), -1.0, -0.2, 0.4]])

    def sig(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def unpack(
        theta: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], float, float, float, float]:
        return (
            theta[:k_mean],
            float(np.exp(theta[k_mean])),
            sig(float(theta[k_mean + 1])),
            sig(float(theta[k_mean + 2])),
            sig(float(theta[k_mean + 3])),
        )

    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        mean, omega, phi, d, beta = unpack(theta)
        resid = endog - mean_x @ mean if k_mean else endog
        lam = _figarch_weights(phi, d, beta, min(truncation, 200))
        if np.any(lam < -1e-6):
            return 1e10
        sigma2 = _figarch_variance(resid, omega, phi, d, beta, backcast, truncation)
        if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
            return 1e10
        ll = -0.5 * np.sum(_LOG_2PI + np.log(sigma2) + resid**2 / sigma2)
        return float(-ll) if np.isfinite(ll) else 1e10

    result = minimize(negloglik, theta0, method="L-BFGS-B")
    mean, omega, phi, d, beta = unpack(np.asarray(result.x, dtype=np.float64))
    fitted = mean_x @ mean if k_mean else np.zeros(n)
    resid = endog - fitted
    sigma2 = _figarch_variance(resid, omega, phi, d, beta, backcast, truncation)
    return _GARCHFit(
        const=float(mean[0]) if include_const else None,
        ar_params=np.zeros(0),
        omega=omega,
        alpha=np.array([phi]),
        gamma=np.zeros(0),
        beta=np.array([beta]),
        fractional_d=d,
        conditional_variance=sigma2,
        resid=resid,
        fittedvalues=fitted,
        llf=-float(result.fun),
        nobs=n,
        n_params=k_mean + 3,
    )


@dataclass(frozen=True, slots=True)
class _ARNNFit:
    """Raw outputs of an autoregressive neural mean-function fit."""

    predictor: MeanPredictor
    sigma2: float
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


@dataclass(frozen=True, slots=True)
class _TARNNFit:
    """Raw outputs of a threshold neural mean-function fit."""

    delay: int
    threshold: float
    lower_predictor: MeanPredictor
    upper_predictor: MeanPredictor
    threshold_variable: npt.NDArray[np.float64] | None
    self_exciting: bool
    sigma2: float
    ssr: float
    n_lower: int
    n_upper: int
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


def _fit_arnn(y: npt.NDArray[np.float64], order: int, engine: MeanFunctionEngine) -> _ARNNFit:
    """Fit a neural autoregression of the given order.

    The likelihood is Gaussian with the variance concentrated out, so
    ``n_params`` counts the learner's parameters plus that variance.

    Args:
        y: The series.
        order: Autoregressive order.
        engine: The training backend.

    Returns:
        The packed :class:`_ARNNFit`.
    """
    target = y[order:]
    features = lag_matrix(y, order)
    predictor = engine.fit(features, target)
    fitted = predictor.predict(features)
    resid = target - fitted
    sigma2, llf = _gaussian_llf(float(resid @ resid), target.shape[0])
    return _ARNNFit(
        predictor=predictor,
        sigma2=sigma2,
        resid=resid,
        fittedvalues=fitted,
        llf=float(llf),
        nobs=target.shape[0],
        n_params=predictor.n_parameters + 1,
    )


def _fit_tarnn(
    y: npt.NDArray[np.float64],
    order: int,
    engine: MeanFunctionEngine,
    threshold_variable: npt.NDArray[np.float64] | None,
    delay: int,
    threshold: float | None,
    trim: float,
) -> _TARNNFit:
    """Fit a two-regime neural threshold autoregression.

    The threshold defaults to the median of the transition variable rather
    than being searched: with a nonlinear learner per regime, a grid search
    would retrain the network at every candidate split.

    Args:
        y: The series.
        order: Autoregressive order per regime.
        engine: The training backend, used once per regime.
        threshold_variable: External threshold variable, or ``None``.
        delay: Threshold delay.
        threshold: Fixed threshold, or ``None`` for the median.
        trim: Minimum regime share of the effective sample.

    Returns:
        The packed :class:`_TARNNFit`.

    Raises:
        NumericalError: If the split leaves a regime with too few observations.
    """
    n = y.shape[0]
    start = max(order, delay)
    target = y[start:]
    n_eff = target.shape[0]
    features = lag_matrix(y, order, start=start)
    base = threshold_variable if threshold_variable is not None else y
    z = base[start - delay : n - delay]
    r = float(np.median(z)) if threshold is None else float(threshold)
    lower = z <= r
    n_lo = int(lower.sum())
    n_hi = n_eff - n_lo
    if min(n_lo, n_hi) < max(2, int(trim * n_eff)):
        raise NumericalError(
            f"threshold {r} leaves a regime with too few observations ({n_lo} lower, {n_hi} upper)."
        )
    lower_predictor = engine.fit(features[lower], target[lower])
    upper_predictor = engine.fit(features[~lower], target[~lower])
    fitted = np.empty(n_eff, dtype=np.float64)
    fitted[lower] = lower_predictor.predict(features[lower])
    fitted[~lower] = upper_predictor.predict(features[~lower])
    resid = target - fitted
    ssr = float(resid @ resid)
    sigma2, llf = _gaussian_llf(ssr, n_eff)
    return _TARNNFit(
        delay=delay,
        threshold=r,
        lower_predictor=lower_predictor,
        upper_predictor=upper_predictor,
        threshold_variable=threshold_variable,
        self_exciting=threshold_variable is None,
        sigma2=sigma2,
        ssr=ssr,
        n_lower=n_lo,
        n_upper=n_hi,
        resid=resid,
        fittedvalues=fitted,
        llf=float(llf),
        nobs=n_eff,
        n_params=lower_predictor.n_parameters + upper_predictor.n_parameters + 2,
    )


@dataclass(frozen=True, slots=True)
class _MSARFit:
    """Raw outputs of a Markov-switching autoregression fit."""

    transition: npt.NDArray[np.float64]
    intercepts: npt.NDArray[np.float64]
    ar_params: npt.NDArray[np.float64]
    variances: npt.NDArray[np.float64]
    filtered_prob: npt.NDArray[np.float64]
    predicted_prob: npt.NDArray[np.float64]
    smoothed_prob: npt.NDArray[np.float64]
    ergodic_prob: npt.NDArray[np.float64]
    expected_durations: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int
    n_iter: int
    converged: bool


@dataclass(frozen=True, slots=True)
class _EMFit:
    """Internal state carried between EM restarts."""

    transition: npt.NDArray[np.float64]
    intercepts: npt.NDArray[np.float64]
    ar_params: npt.NDArray[np.float64]
    sigma2: npt.NDArray[np.float64]
    filtered_prob: npt.NDArray[np.float64]
    predicted_prob: npt.NDArray[np.float64]
    smoothed_prob: npt.NDArray[np.float64]
    llf: float
    n_iter: int
    converged: bool


def _run_em(
    target: npt.NDArray[np.float64],
    lags: npt.NDArray[np.float64],
    layout: _Layout,
    transition0: npt.NDArray[np.float64],
    intercepts0: npt.NDArray[np.float64],
    ar0: npt.NDArray[np.float64],
    sigma20: npt.NDArray[np.float64],
    var_floor: float,
    prob_floor: float,
    max_iter: int,
    tol: float,
    switching_variance: bool,
) -> _EMFit:
    """Run EM to convergence or ``max_iter`` from one set of starting values.

    Returns:
        The :class:`_EMFit` reached.

    Raises:
        NumericalError: If the log-likelihood becomes non-finite.
    """
    transition = transition0.copy()
    intercepts = intercepts0.copy()
    ar_params = ar0.copy()
    sigma2 = sigma20.copy()
    prev_llf = -np.inf
    filtered = predicted = smoothed = np.empty((0, layout.n_regimes))
    n_iter = 0
    converged = False

    for n_iter in range(1, max_iter + 1):
        means = _regime_means(intercepts, ar_params, lags)
        logd = _log_densities(target, means, sigma2)
        filt = hamilton_filter(logd, transition, initial_prob=ergodic_distribution(transition))
        smooth = kim_smoother(filt, transition)
        filtered = filt.filtered_prob
        predicted = filt.predicted_prob
        smoothed = smooth.smoothed_prob
        llf = filt.loglikelihood
        if not np.isfinite(llf):
            raise NumericalError("MS-AR log-likelihood became non-finite during EM.")
        if llf - prev_llf < tol and n_iter > 1:
            converged = True
            prev_llf = llf
            break
        prev_llf = llf
        transition = _update_transition(smoothed, smooth.smoothed_joint_prob, prob_floor)
        intercepts, ar_params = _update_coefficients(target, lags, smoothed, sigma2, layout)
        means = _regime_means(intercepts, ar_params, lags)
        sigma2 = _update_variance(target, means, smoothed, switching_variance, var_floor)

    return _EMFit(
        transition=transition,
        intercepts=intercepts,
        ar_params=ar_params,
        sigma2=sigma2,
        filtered_prob=filtered,
        predicted_prob=predicted,
        smoothed_prob=smoothed,
        llf=float(prev_llf),
        n_iter=n_iter,
        converged=converged,
    )


@dataclass(frozen=True, slots=True)
class _ThresholdFit:
    """Raw outputs of a two-regime threshold grid search."""

    delay: int
    threshold: float
    lower_params: npt.NDArray[np.float64]
    upper_params: npt.NDArray[np.float64]
    sigma2: float
    ssr: float
    n_lower: int
    n_upper: int
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


@dataclass(frozen=True, slots=True)
class _STARFit:
    """Raw outputs of a smooth-transition autoregression fit."""

    delay: int
    threshold: float
    gamma: float
    lower_params: npt.NDArray[np.float64]
    upper_params: npt.NDArray[np.float64]
    sigma2: float
    ssr: float
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


def _fit_threshold(
    y: npt.NDArray[np.float64],
    order: int,
    delays: list[int],
    trim: float,
    n_grid: int,
    threshold_var: npt.NDArray[np.float64] | None,
) -> _ThresholdFit:
    """Fit a two-regime threshold autoregression by grid search.

    Searches every ``(delay, threshold)`` pair on a trimmed quantile grid and
    keeps the pair minimizing total SSR. Splits that leave either regime with
    fewer than ``order + 2`` observations are skipped, since those coefficients
    would not be identified.

    Args:
        y: The series.
        order: AR order per regime.
        delays: Candidate threshold delays.
        trim: Fraction trimmed from each tail of the grid.
        n_grid: Number of candidate thresholds per delay.
        threshold_var: External threshold variable, or ``None`` for self-exciting.

    Returns:
        The packed :class:`_ThresholdFit`.

    Raises:
        NumericalError: If no admissible split exists.
    """
    n = y.shape[0]
    start = max(order, max(delays))
    target = y[start:]
    n_eff = target.shape[0]
    design = _ar_design(y, order, start)
    base = threshold_var if threshold_var is not None else y
    min_regime = order + 2

    best_ssr = np.inf
    best: tuple[int, float, npt.NDArray[np.float64], npt.NDArray[np.float64], int, int] | None = (
        None
    )
    for d in delays:
        z = base[start - d : n - d]
        grid = np.quantile(z, np.linspace(trim, 1.0 - trim, n_grid))
        for r in np.unique(grid):
            lower = z <= r
            n_lo = int(lower.sum())
            n_hi = n_eff - n_lo
            if n_lo < min_regime or n_hi < min_regime:
                continue
            b_lo, ssr_lo = ols(design[lower], target[lower])
            b_hi, ssr_hi = ols(design[~lower], target[~lower])
            ssr = ssr_lo + ssr_hi
            if ssr < best_ssr:
                best_ssr = ssr
                best = (d, float(r), b_lo, b_hi, n_lo, n_hi)

    if best is None:
        raise NumericalError(
            "threshold grid search found no admissible split; relax trim or shorten order."
        )
    d_star, r_star, b_lo, b_hi, n_lo, n_hi = best
    z = base[start - d_star : n - d_star]
    lower = z <= r_star
    fitted = np.where(lower, design @ b_lo, design @ b_hi)
    resid = target - fitted
    sigma2 = best_ssr / n_eff
    llf = -0.5 * n_eff * (_LOG_2PI + np.log(sigma2) + 1.0)
    return _ThresholdFit(
        delay=d_star,
        threshold=r_star,
        lower_params=b_lo,
        upper_params=b_hi,
        sigma2=sigma2,
        ssr=float(best_ssr),
        n_lower=n_lo,
        n_upper=n_hi,
        resid=resid,
        fittedvalues=fitted,
        llf=float(llf),
        nobs=n_eff,
        n_params=2 * (order + 1) + 1,
    )


def _fit_star(y: npt.NDArray[np.float64], order: int, delay: int, transition: str) -> _STARFit:
    """Fit a smooth-transition autoregression by concentrated least squares.

    The transition variable is standardized before entering the transition
    function, so ``gamma`` is scale-free and comparable across series.

    Args:
        y: The series.
        order: AR order per regime.
        delay: Transition-variable delay.
        transition: ``"logistic"`` (LSTAR) or ``"exponential"`` (ESTAR).

    Returns:
        The packed :class:`_STARFit`.

    Raises:
        NumericalError: If the transition variable has zero variance.
    """
    n = y.shape[0]
    start = max(order, delay)
    target = y[start:]
    n_eff = target.shape[0]
    design = _ar_design(y, order, start)
    z = y[start - delay : n - delay]
    sd = float(np.std(z))
    if sd == 0.0:
        raise NumericalError("transition variable has zero variance.")

    def transition_weights(gamma: float, c: float) -> npt.NDArray[np.float64]:
        u = (z - c) / sd
        if transition == "logistic":
            return 1.0 / (1.0 + np.exp(-np.clip(gamma * u, -50.0, 50.0)))
        return 1.0 - np.exp(-np.clip(gamma * u**2, 0.0, 50.0))

    def concentrated(
        gamma: float, c: float
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        g = transition_weights(gamma, c)
        regressors = np.column_stack([design * (1.0 - g)[:, None], design * g[:, None]])
        beta, ssr = ols(regressors, target)
        return ssr, beta, target - regressors @ beta

    c_seed = float(np.median(z))
    best_hard = np.inf
    for cc in np.quantile(z, np.linspace(0.15, 0.85, 50)):
        lower = z <= cc
        n_lo = int(lower.sum())
        if n_lo < order + 2 or n_eff - n_lo < order + 2:
            continue
        ssr_hard = ols(design[lower], target[lower])[1] + ols(design[~lower], target[~lower])[1]
        if ssr_hard < best_hard:
            best_hard, c_seed = ssr_hard, float(cc)

    def objective(par: npt.NDArray[np.float64]) -> float:
        return concentrated(float(np.exp(par[0])), float(par[1]))[0]

    best_ssr, best_par = np.inf, np.array([np.log(5.0), c_seed])
    for c0 in (c_seed, float(np.median(z))):
        for g0 in (2.0, 5.0, 10.0, 25.0):
            res = minimize(
                objective,
                np.array([np.log(g0), c0]),
                method="Nelder-Mead",
                options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000},
            )
            if float(res.fun) < best_ssr:
                best_ssr, best_par = float(res.fun), np.asarray(res.x, dtype=np.float64)

    gamma = float(np.exp(best_par[0]))
    c = float(best_par[1])
    ssr, beta, resid = concentrated(gamma, c)
    sigma2 = ssr / n_eff
    llf = -0.5 * n_eff * (_LOG_2PI + np.log(sigma2) + 1.0)
    return _STARFit(
        delay=delay,
        threshold=c,
        gamma=gamma,
        lower_params=beta[: order + 1],
        upper_params=beta[order + 1 :],
        sigma2=sigma2,
        ssr=ssr,
        resid=resid,
        fittedvalues=target - resid,
        llf=float(llf),
        nobs=n_eff,
        n_params=2 * (order + 1) + 2,
    )
