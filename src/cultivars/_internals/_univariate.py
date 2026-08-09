# filepath: /src/cultivars/_internals/_univariate.py
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

"""Autoregressive estimation engines.

Two estimators for the same AR(p) mean model. Conditional sum of squares is a
single least-squares solve on the lagged design, fast and unconditionally
available. Exact maximum likelihood casts the model in companion state-space
form and evaluates the Kalman likelihood, which uses the first ``p``
observations that CSS discards -- material in short samples, negligible in long
ones.

The exact estimator optimizes in the Monahan reparameterization from
:mod:`cultivars._core._reparam`, so every candidate the optimizer proposes is
stationary by construction and no penalty term is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from .._core._companion import companion_matrix
from .._core._defaults import _LOG_2PI
from .._core._design import css_design, n_deterministic
from .._core._reparam import pack_stationary, unpack_stationary
from .._core._stability import assess_stability
from ..exceptions import NumericalError, SpecificationError
from ._linear_gaussian import LinearGaussianStateSpace


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


def _forecast_ar(
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
