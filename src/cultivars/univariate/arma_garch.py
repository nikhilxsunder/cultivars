# filepath: /src/cultivars/univariate/arma_garch.py
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

"""ARMA-mean conditional-variance models: ARMA-{GARCH, GJR, EGARCH, FIGARCH}.

The GARCH-family estimators in :mod:`cultivars.univariate.garch` model the
conditional mean as a constant plus optional pure-AR terms. This module supplies
the remaining piece of the univariate surface: a full **ARMA(p, q) conditional
mean** estimated *jointly* with any of the four volatility processes, so
ARMA-GARCH, ARMA-GJR, ARMA-EGARCH, and ARMA-FIGARCH are all reachable — and
FIGARCH gains a mean beyond a constant for the first time.

One mean, four backends
--------------------------------------------------------------------------
Rather than a class per (mean x variance) pair — the combinatorial explosion the
architecture forbids — a single :class:`ARMAGARCH` estimator carries the ARMA
mean and takes the volatility family as a ``vol`` strategy. The four variance
recursions are reused verbatim from :mod:`cultivars.univariate.garch`; only the
mean layer is new. The mean is computed by the conditional-sum-of-squares (CSS)
ARMA recursion rather than the exact-ML state-space form used by
:mod:`cultivars.univariate.arma`, because under GARCH innovations are
heteroskedastic and the constant-variance Kalman likelihood no longer applies;
the joint Gaussian likelihood
``-0.5 sum_t [ log(2 pi) + log(sigma2_t) + eps_t^2 / sigma2_t ]``
is maximized over the mean and variance parameters together.

Stationarity (AR) and invertibility (MA) of the mean are imposed structurally by
the partial-autocorrelation reparameterization in
:mod:`cultivars.univariate._base`, so the CSS residual recursion can never
explode and the optimizer searches an unconstrained space.

References:
    Bollerslev (1986); Glosten, Jagannathan, Runkle (1993); Nelson (1991);
    Baillie, Bollerslev & Mikkelsen (1996); Hannan & Rissanen (1982).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._base import (
    InformationCriteria,
    coeffs_to_pacf,
    information_criteria,
    pacf_to_coeffs,
)
from .garch import (
    Vol,
    _backcast,
    _egarch_variance,
    _figarch_variance,
    _garch_variance,
    _inv_softplus,
    _softplus,
)

# --------------------------------------------------------------------------
# ARMA conditional mean (conditional sum of squares)
# --------------------------------------------------------------------------




# --------------------------------------------------------------------------
# Variance-block bookkeeping (reuses garch.py recursions)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class ARMAGARCHResult(_MeanResult, _StationarityMixin, _ConditionalVarianceMixin):
    vol: str
    order: tuple[int, int]
    garch_order: tuple[int, int, int]
    const: float | None
    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    omega: float
    alpha: npt.NDArray[np.float64]
    gamma: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    fractional_d: float | None
    conditional_variance: npt.NDArray[np.float64]

# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

def _fit_arma_garch(
    y: npt.NDArray[np.float64],
    mean_p: int,
    mean_q: int,
    include_const: bool,
    vol: Vol,
    p: int,
    o: int,
    q: int,
    truncation: int,
) -> ARMAGARCHResult:
    burn = max(mean_p, mean_q)

    # Warm start: mean via Hannan-Rissanen on the demeaned series.
    c0 = float(y.mean()) if include_const else 0.0
    w0 = y - c0
    phi0, theta0_ma = _hannan_rissanen(w0, mean_p, mean_q)
    resid0 = _arma_residuals(w0, phi0, theta0_ma)[burn:]
    var0 = max(float(np.var(resid0)), 1e-8)
    backcast = _backcast(resid0)

    mean_start: list[npt.NDArray[np.float64]] = []
    if include_const:
        mean_start.append(np.array([c0]))
    mean_start.append(_pack_pacf(phi0))
    mean_start.append(_pack_pacf(theta0_ma))
    var_start = _variance_init(vol, var0, p, o, q)
    theta_init = np.concatenate([*mean_start, var_start])

    i_const = 1 if include_const else 0
    i_ar = i_const
    i_ma = i_ar + mean_p
    i_var = i_ma + mean_q

    def unpack_mean(
        theta: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        c = float(theta[0]) if include_const else 0.0
        phi = pacf_to_coeffs(np.tanh(theta[i_ar:i_ma])) if mean_p else np.zeros(0)
        theta_ma = pacf_to_coeffs(np.tanh(theta[i_ma:i_var])) if mean_q else np.zeros(0)
        return c, phi, theta_ma

    def negloglik(theta: npt.NDArray[np.float64]) -> float:
        c, phi, theta_ma = unpack_mean(theta)
        resid = _arma_residuals(y - c, phi, theta_ma)[burn:]
        omega, alpha, gamma, beta, frac_d = _unpack_variance(theta[i_var:], vol, p, o, q)
        if not _persistence_ok(vol, alpha, gamma, beta, frac_d, truncation):
            return 1e10
        try:
            sigma2 = _conditional_variance(
                resid, vol, omega, alpha, gamma, beta, frac_d, backcast, truncation
            )
        except (ValueError, FloatingPointError):
            return 1e10
        if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
            return 1e10
        ll = -0.5 * np.sum(_LOG_2PI + np.log(sigma2) + resid ** 2 / sigma2)
        return float(-ll) if np.isfinite(ll) else 1e10

    result = minimize(negloglik, theta_init, method="L-BFGS-B")
    theta = np.asarray(result.x, dtype=np.float64)
    c, phi, theta_ma = unpack_mean(theta)
    eps_full = _arma_residuals(y - c, phi, theta_ma)
    resid = eps_full[burn:]
    omega, alpha, gamma, beta, frac_d = _unpack_variance(theta[i_var:], vol, p, o, q)
    sigma2 = _conditional_variance(
        resid, vol, omega, alpha, gamma, beta, frac_d, backcast, truncation
    )
    fitted = y[burn:] - resid
    n_params = i_const + mean_p + mean_q + _variance_block_size(vol, p, o, q)
    return ARMAGARCHResult(
        vol=vol,
        order=(mean_p, mean_q),
        garch_order=(p, o, q),
        const=c if include_const else None,
        ar_params=phi,
        ma_params=theta_ma,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        fractional_d=frac_d if vol == "FIGARCH" else None,
        sigma2=sigma2,
        resid=resid,
        fittedvalues=fitted,
        llf=-float(result.fun),
        nobs=resid.shape[0],
        endog=y,
        _burn=burn,
        _n_params=n_params,
    )


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------

class ARMAGARCH:
    """ARMA(p, q) mean with a GARCH-family conditional variance.

    A single estimator spanning ARMA-GARCH, ARMA-GJR, ARMA-EGARCH, and
    ARMA-FIGARCH: the volatility family is selected by ``vol`` and the four
    variance recursions are shared with :mod:`cultivars.univariate.garch`.

    Args:
        endog: Univariate series (1-D array-like), typically a return series.
        order: Mean ARMA order ``(p, q)``. Defaults to ``(1, 0)``.
        vol: Volatility family — ``"GARCH"`` (default), ``"GJR"``, ``"EGARCH"``,
            or ``"FIGARCH"``.
        garch_order: Variance order ``(p, o, q)`` (ARCH, asymmetry, GARCH lags).
            Defaults to a sensible value per family; ignored for FIGARCH, which is
            fixed at ``(1, d, 1)``.
        mean: ``"constant"`` (default) or ``"zero"``.
        truncation: ARCH(inf) truncation lag (FIGARCH only).

    Raises:
        SpecificationError: If the orders/flags are invalid or the series is too
            short.
        DimensionError: If ``endog`` is not one-dimensional.
        NumericalError: If ``endog`` contains non-finite values.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> # AR(1) mean + GARCH(1,1) volatility.
        >>> n = 3000
        >>> eps = np.zeros(n); s2 = np.zeros(n); s2[0] = 1.0
        >>> for t in range(1, n):
        ...     s2[t] = 0.02 + 0.08 * eps[t - 1] ** 2 + 0.90 * s2[t - 1]
        ...     eps[t] = np.sqrt(s2[t]) * rng.standard_normal()
        >>> y = np.zeros(n)
        >>> for t in range(1, n):
        ...     y[t] = 0.05 + 0.5 * y[t - 1] + eps[t]
        >>> res = ARMAGARCH(y, order=(1, 0), vol="GARCH").fit()
        >>> bool(0.3 < res.ar_params[0] < 0.7)
        True
        >>> res.forecast(3).shape
        (3,)
    """

    __slots__ = ("_y", "_p", "_q", "_vol", "_go", "_const", "_truncation")

    def __init__(
        self,
        endog: npt.ArrayLike,
        order: tuple[int, int] = (1, 0),
        vol: Vol = "GARCH",
        *,
        garch_order: tuple[int, int, int] | None = None,
        mean: str = "constant",
        truncation: int = 1000,
    ) -> None:
        """Initialize the ARMA-GARCH specification."""
        y = np.asarray(endog, dtype=np.float64)
        if y.ndim != 1:
            raise DimensionError(f"endog must be one-dimensional; got shape {y.shape}.")
        if not np.all(np.isfinite(y)):
            raise NumericalError("endog contains non-finite values.")
        p_mean, q_mean = order
        if min(p_mean, q_mean) < 0:
            raise SpecificationError(f"mean order (p, q) must be non-negative; got {order}.")
        if vol not in ("GARCH", "GJR", "EGARCH", "FIGARCH"):
            raise SpecificationError(
                f"vol must be one of 'GARCH', 'GJR', 'EGARCH', 'FIGARCH'; got {vol!r}."
            )
        if mean not in ("constant", "zero"):
            raise SpecificationError(f"mean must be 'constant' or 'zero'; got {mean!r}.")
        go = garch_order if garch_order is not None else _DEFAULT_GARCH_ORDER[vol]
        gp, go_, gq = go
        if min(gp, go_, gq) < 0:
            raise SpecificationError(f"garch_order must be non-negative; got {go}.")
        if vol == "GARCH" and go_ != 0:
            raise SpecificationError("GARCH has no asymmetry term; set garch_order o = 0.")
        if vol in ("GJR", "EGARCH") and go_ < 1:
            raise SpecificationError(f"{vol} requires an asymmetry order o >= 1.")
        if truncation < 1:
            raise SpecificationError(f"truncation must be >= 1; got {truncation}.")
        min_len = 4 * (max(p_mean, q_mean) + max(gp, gq) + 1)
        if y.shape[0] < min_len:
            raise SpecificationError(
                f"series of length {y.shape[0]} is too short for ARMA{order}-{vol} "
                f"(need at least {min_len})."
            )
        self._y = y
        self._p, self._q = int(p_mean), int(q_mean)
        self._vol: Vol = vol
        self._go = (int(gp), int(go_), int(gq))
        self._const = mean == "constant"
        self._truncation = int(truncation)

    @property
    def order(self) -> tuple[int, int]:
        """The mean ARMA order ``(p, q)``."""
        return (self._p, self._q)

    @property
    def vol(self) -> Vol:
        """The volatility family."""
        return self._vol

    def fit(self) -> ARMAGARCHResult:
        """Estimate the mean and variance jointly by Gaussian maximum likelihood."""
        gp, go_, gq = self._go
        return _fit_arma_garch(
            self._y, self._p, self._q, self._const, self._vol,
            gp, go_, gq, self._truncation,
        )