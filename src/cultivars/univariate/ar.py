# filepath: /src/cultivars/univariate/ar.py
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
"""Univariate autoregressive model AR(p).

Implements ``y_t = c + delta*t + phi_1 y_{t-1} + ... + phi_p y_{t-p} + eps_t``,
``eps_t ~ N(0, sigma2)``, following cultivars' three-object discipline:

- :class:`AR` is the immutable specification (data, order, trend). It validates
  at construction and exposes ``fit``.
- Estimation is dispatched to two closed-form / optimizer routines. ``"css"``
  (conditional sum of squares) is the fast ordinary-least-squares path;
  ``"exact"`` maximizes the exact Gaussian likelihood through the linear-Gaussian
  state-space substrate (the AR(p) -> companion embedding), keeping the parameter
  search in the stationary region via a partial-autocorrelation reparameterization.
- :class:`ARResult` is the frozen, serializable result consumed by downstream
  forecasting and diagnostics.

The two estimators differ in finite samples: CSS conditions on the first ``p``
observations, exact ML models their (stationary) density too; they converge as
the sample grows.

References:
    Hamilton, J. D. (1994). *Time Series Analysis*, ch. 5 & 13.
    Monahan, J. F. (1984). A note on enforcing stationarity in ARMA models.
    *Biometrika*, 71(2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core._design import n_deterministic
from .._internals._models import _MeanResult, _StationarityMixin, _UnivariateModel
from .._internals._univariate import _fit_ar_css, _forecast_ar


@dataclass(frozen=True, kw_only=True, slots=True)
class ARResult(_MeanResult, _StationarityMixin):
    """Fitted autoregressive model result.

    Inherits ``llf``/``nobs``/``n_params`` and the information-criteria
    surface from :class:`_Result`, the endog/residual/fitted surface from
    :class:`_MeanResult`, and stationarity assessment from
    :class:`_StationarityMixin`.
    """

    order: int
    trend: str
    method: str
    const: float
    trend_coeff: float
    ar_params: npt.NDArray[np.float64]
    sigma2: float

    def forecast(self, h: int) -> npt.NDArray[np.float64]:
        if h < 1:
            raise ValueError(f"horizon must be >= 1; got {h}.")
        return _forecast_ar(
            self.endog,
            self.ar_params,
            self.const,
            self.trend_coeff,
            self.nobs + self.order,
            h,
        )


class AR(_UnivariateModel[ARResult]):
    """Autoregressive AR(p) specification fit by conditional least squares."""

    __slots__ = ("_order", "_trend")

    def __init__(self, endog: npt.ArrayLike, order: int, trend: str = "c") -> None:
        super().__init__(endog)
        if trend not in ("n", "c", "ct"):
            raise ValueError(f"trend must be one of 'n', 'c', 'ct'; got {trend!r}.")
        self._order = int(order)
        self._trend = trend
        self._ensure_length(order + 2, f"AR({order})")

    @property
    def order(self) -> int:
        return self._order

    @property
    def trend(self) -> str:
        return self._trend

    def fit(self, method: str = "css") -> ARResult:
        if method != "css":
            raise ValueError(f"unknown method {method!r}; expected 'css'.")
        fit = _fit_ar_css(self.endog, self._order, self._trend)
        return ARResult(
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=self._order + n_deterministic(self._trend) + 1,
            endog=self.endog,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            order=self._order,
            trend=self._trend,
            method=method,
            const=fit.const,
            trend_coeff=fit.trend_coeff,
            ar_params=fit.ar_params,
            sigma2=fit.sigma2,
        )
