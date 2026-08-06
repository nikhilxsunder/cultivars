import numpy as np
import numpy.typing as npt


def _regime_means(
    intercepts: npt.NDArray[np.float64],
    ar_params: npt.NDArray[np.float64],
    lags: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Per-regime conditional means ``mu[t, j]`` of shape ``(T_eff, K)``."""
    if lags.shape[1] == 0:
        return np.broadcast_to(intercepts, (lags.shape[0], intercepts.shape[0])).copy()
    return intercepts[None, :] + lags @ ar_params.T


def _log_densities(
    target: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    sigma2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    resid = target[:, None] - means
    return -0.5 * (_LOG_2PI + np.log(sigma2)[None, :] + resid ** 2 / sigma2[None, :])


def _update_transition(
    smoothed: npt.NDArray[np.float64],
    joint: npt.NDArray[np.float64],
    floor: float,
) -> npt.NDArray[np.float64]:
    k = smoothed.shape[1]
    if joint.shape[0] == 0:
        return np.full((k, k), 1.0 / k)
    numer = joint.sum(axis=0)                       # (K, K) expected i->j counts
    denom = smoothed[:-1].sum(axis=0)               # (K,) expected time in i
    p = numer / np.clip(denom[:, None], floor, None)
    p = np.clip(p, floor, None)
    return p / p.sum(axis=1, keepdims=True)


def _update_coefficients(
    target: npt.NDArray[np.float64],
    lags: npt.NDArray[np.float64],
    smoothed: npt.NDArray[np.float64],
    sigma2: npt.NDArray[np.float64],
    layout: _Layout,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Responsibility-weighted GLS update; returns (intercepts (K,), ar (K, p))."""
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


def _update_variance(
    target: npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    smoothed: npt.NDArray[np.float64],
    switching_variance: bool,
    floor: float,
) -> npt.NDArray[np.float64]:
    k = smoothed.shape[1]
    sq = (target[:, None] - means) ** 2
    if switching_variance:
        num = (smoothed * sq).sum(axis=0)
        den = smoothed.sum(axis=0)
        sigma2 = num / np.clip(den, 1e-12, None)
    else:
        shared = float((smoothed * sq).sum() / target.shape[0])
        sigma2 = np.full(k, shared)
    return np.clip(sigma2, floor, None)
