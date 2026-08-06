from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core._defaults import _LOG_2PI
from .._core._design import css_design, n_deterministic


def _arma_residuals(
    w: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    theta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """CSS residuals of ``phi(L) w_t = theta(L) eps_t`` (plus-sign MA convention).

    ``eps_t = w_t - sum_i phi_i w_{t-i} - sum_j theta_j eps_{t-j}`` with pre-sample
    ``w`` and ``eps`` set to zero. ``w`` is the demeaned series.
    """
    n, p, q = w.shape[0], phi.size, theta.size
    eps = np.empty(n, dtype=np.float64)
    for t in range(n):
        value = w[t]
        for i in range(p):
            if t - 1 - i >= 0:
                value -= phi[i] * w[t - 1 - i]
        for j in range(q):
            if t - 1 - j >= 0:
                value -= theta[j] * eps[t - 1 - j]
        eps[t] = value
    return eps


def _hannan_rissanen(
    w: npt.NDArray[np.float64], p: int, q: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Two-stage Hannan-Rissanen starting values for (phi, theta)."""
    n = w.shape[0]
    if p == 0 and q == 0:
        return np.zeros(0), np.zeros(0)
    long_order = int(min(max(10, p + q + 5), n // 2 - 1))
    resid = w.copy()
    if long_order >= 1 and n > long_order:
        design = np.column_stack([w[long_order - i : n - i] for i in range(1, long_order + 1)])
        coeffs, *_ = np.linalg.lstsq(design, w[long_order:], rcond=None)
        resid = np.zeros(n)
        resid[long_order:] = w[long_order:] - design @ coeffs
    start = max(p, q)
    cols: list[npt.NDArray[np.float64]] = []
    for i in range(1, p + 1):
        cols.append(w[start - i : n - i])
    for j in range(1, q + 1):
        cols.append(resid[start - j : n - j])
    if not cols:
        return np.zeros(0), np.zeros(0)
    beta, *_ = np.linalg.lstsq(np.column_stack(cols), w[start:], rcond=None)
    return np.asarray(beta[:p], dtype=np.float64), np.asarray(beta[p:], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _ARFit:
    """Raw outputs of an autoregressive conditional-sum-of-squares fit."""
    const: float
    trend_coeff: float
    ar_params: npt.NDArray[np.float64]
    sigma2: float
    llf: float
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    nobs: int

def _fit_ar_css(y: npt.NDArray[np.float64], order: int, trend: str) -> _ARFit:
    """Fit an AR(p) mean model by conditional least squares."""
    target, design, eff = css_design(y, order, trend)
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta
    sigma2 = float(resid @ resid / eff)
    k_det = n_deterministic(trend)
    const = float(beta[0]) if trend in ("c", "ct") else 0.0
    trend_coeff = float(beta[1]) if trend == "ct" else 0.0
    ar_params = np.asarray(beta[k_det:], dtype=np.float64)
    llf = -0.5 * eff * (_LOG_2PI + np.log(sigma2) + 1.0)
    fitted = design @ beta
    return _ARFit(
        const=const, trend_coeff=trend_coeff, ar_params=ar_params,
        sigma2=sigma2, llf=float(llf), resid=resid, fittedvalues=fitted, nobs=eff,
    )

def _forecast_ar(
    history: npt.NDArray[np.float64],
    ar_params: npt.NDArray[np.float64],
    const: float,
    trend_coeff: float,
    nobs_full: int,
    h: int,
) -> npt.NDArray[np.float64]:
    """Recursive point forecast for an AR(p) mean model."""
    p = ar_params.shape[0]
    buf = list(history[-p:]) if p else []
    out = np.empty(h, dtype=np.float64)
    for step in range(h):
        t = nobs_full + step + 1
        val = const + trend_coeff * t
        for i in range(p):
            val += ar_params[i] * buf[-1 - i]
        out[step] = val
        buf.append(val)
    return out