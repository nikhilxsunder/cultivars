from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

from .._core import _KSC_MEAN, _KSC_VAR, _OFFSET, _draw_inverse_gamma, _draw_mixture_indicators


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

    Returns:
        ``(mu, phi, sigma2)`` after one sweep.
    """
    h = log_variance
    n = h.shape[0]
    # -- mu | h, phi, sigma2 : Gaussian -------------------------------------
    m0, v0 = prior_mu
    precision = 1.0 / v0 + ((1.0 - phi**2) + (n - 1) * (1.0 - phi) ** 2) / sigma2
    moment = (
        m0 / v0
        + ((1.0 - phi**2) * h[0] + (1.0 - phi) * float(np.sum(h[1:] - phi * h[:-1]))) / sigma2
    )
    mu = float(moment / precision + rng.standard_normal() / np.sqrt(precision))
    # -- phi | h, mu, sigma2 : MH with the regression conditional -----------
    centered = h - mu
    sxx = float(centered[:-1] @ centered[:-1])
    sxy = float(centered[1:] @ centered[:-1])
    if sxx > 0.0:
        phi_hat = sxy / sxx
        phi_prop = float(phi_hat + rng.standard_normal() * np.sqrt(sigma2 / sxx))
        if abs(phi_prop) < 1.0:
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
                phi = phi_prop
    # -- sigma2 | h, mu, phi : inverse-gamma -------------------------------
    shape0, rate0 = prior_sigma2
    residual_ss = (1.0 - phi**2) * centered[0] ** 2 + float(
        np.sum((centered[1:] - phi * centered[:-1]) ** 2)
    )
    sigma2 = _draw_inverse_gamma(shape0 + 0.5 * n, rate0 + 0.5 * residual_ss, rng)
    return mu, phi, sigma2


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
