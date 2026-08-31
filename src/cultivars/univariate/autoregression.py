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

"""Autoregressive model AR(p) -- the public specification and result.

Implements ``y_t = c + delta*t + phi_1 y_{t-1} + ... + phi_p y_{t-p} + eps_t``
with ``eps_t ~ N(0, sigma2)``, under the package's three-object discipline:
:class:`AR` is the immutable specification, estimation belongs to the family
base it inherits, and :class:`ARResult` is the frozen result the user actually
handles.

Two estimators are reachable through ``method``. ``"css"`` conditions on the
first ``p`` observations and solves by ordinary least squares; ``"exact"``
maximizes the exact Gaussian likelihood through the companion state-space
embedding, keeping the search inside the stationary region via a
partial-autocorrelation reparameterization. They differ in finite samples --
CSS drops ``p`` observations, exact ML models their stationary density too --
and converge as the sample grows.

Because CSS and exact ML report different effective sample sizes, information
criteria are only comparable across orders when every model uses ``"exact"``;
:meth:`ARResult.compare` enforces that rather than silently ranking
incomparable numbers.

References:
    Hamilton, J. D. (1994). *Time Series Analysis*, ch. 5 and 13.
    Monahan, J. F. (1984). A note on enforcing stationarity in ARMA models.
    *Biometrika*, 71(2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import InformationCriteria, SummaryTable
from .._internals import (
    _AutoRegressionFit,
    _AutoRegressionModel,
    _ComparisonMixin,
    _SeriesMixin,
    _StationarityMixin,
    _SummaryMixin,
)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ARResult(_SummaryMixin, _SeriesMixin, _ComparisonMixin, _StationarityMixin):
    """A fitted autoregression.

    Composes four capabilities: rendering, frame interop, criterion-named
    comparison, and stationarity assessment. ``repr=False`` is required -- see
    :class:`_SummaryMixin` -- so the summary is what a bare result prints.

    Attributes:
        endog: The full observed series, retained so residual and fitted paths
            can be aligned against it.
        fittedvalues: One-step fitted values over the effective sample.
        resid: One-step residuals over the effective sample.
        llf: Maximized log-likelihood, conditional for CSS and exact otherwise.
        nobs: Observations the likelihood was evaluated on.
        n_params: Free parameter count, including the innovation variance.
        order: Autoregressive order ``p``.
        trend: Deterministic specification.
        method: Estimator that produced the fit.
        const: Intercept, or ``None`` when ``trend == "n"``.
        trend_coeff: Linear-trend slope, or ``None`` unless ``trend == "ct"``.
        ar_params: Autoregressive coefficients.
        sigma2: Innovation variance.
    """

    endog: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: float
    order: int
    trend: str
    method: str
    const: float | None
    trend_coeff: float | None
    ar_params: npt.NDArray[np.float64]
    sigma2: float

    @classmethod
    def _from_fit(cls, fit: _AutoRegressionFit, model: _AutoRegressionModel[ARResult]) -> ARResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The raw estimator output.
            model: The specification that produced it, read for the fields the
                fit record deliberately does not carry.

        Returns:
            The assembled result.
        """
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            order=model.order,
            trend=model.trend,
            method=model.method,
            const=fit.const,
            trend_coeff=fit.trend_coeff,
            ar_params=fit.ar_params,
            sigma2=fit.sigma2,
        )

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order."""
        out: dict[str, float] = {}
        if self.const is not None:
            out["const"] = self.const
        if self.trend_coeff is not None:
            out["trend"] = self.trend_coeff
        for i, value in enumerate(self.ar_params, start=1):
            out[f"ar.L{i}"] = float(value)
        out["sigma2"] = self.sigma2
        return out

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        return f"AR({self.order}){'' if self.trend == 'c' else f' trend={self.trend}'}"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        ic: InformationCriteria = self.information_criteria
        stability = self.stability
        return SummaryTable(
            title=f"AR({self.order}) Results",
            metadata=(
                ("Model", f"AR({self.order})"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Method", self.method),
                ("AIC", f"{ic.aic:.3f}"),
                ("Trend", self.trend),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=(
                f"Stationary: {self.is_stationary}   "
                f"max |companion root| = {stability.max_modulus:.4f}",
                "Standard errors are not yet available for this estimator.",
            ),
        )


class AR(_AutoRegressionModel[ARResult]):
    """Autoregressive AR(p) specification.

    Args:
        endog: The endogenous series.
        order: Autoregressive order ``p``.
        trend: Deterministic specification (``"n"``, ``"c"``, ``"ct"``).
        method: ``"css"`` or ``"exact"``.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> e = rng.standard_normal(500)
        >>> y = np.zeros(500)
        >>> for t in range(1, 500):
        ...     y[t] = 0.5 * y[t - 1] + e[t]
        >>> res = AR(y, order=1, trend="c").fit()
        >>> bool(0.3 < res.ar_params[0] < 0.7)
        True
    """

    __slots__ = ()

    def fit(self) -> ARResult:
        """Estimate the model by the selected method.

        Returns:
            The fitted :class:`ARResult`.
        """
        return ARResult._from_fit(self._fit_family(), self)
