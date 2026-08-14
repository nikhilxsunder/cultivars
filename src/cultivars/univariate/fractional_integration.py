# filepath: /src/cultivars/univariate/arfima.py
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

"""ARFIMA(p, d, q) -- fractionally integrated ARMA with long memory.

The fractional operator ``(1 - L)**d`` interpolates between a stationary ARMA
(``d = 0``) and a unit-root process (``d = 1``), giving autocorrelations that
decay hyperbolically rather than geometrically. ``mu``, ``d`` and the
short-memory block are estimated jointly: the filter is re-applied at every
draw, because the differenced series is itself a function of ``d``.

Stationarity is not the short-memory test. It needs ``|d| < 0.5`` *and* a
stable autoregressive block, so :class:`ARFIMAResult` overrides the inherited
property rather than reporting a ``d = 0.9`` process as stationary because its
ARMA block happens to be well behaved. Mean reversion is the weaker and
separately useful condition ``d < 1``.

The estimator confines ``d`` to ``(-_D_MAX, _D_MAX)`` through a scaled ``tanh``
reparameterization, so an estimate at the boundary is a sign the data want a
unit root rather than a long-memory fit.

References:
    Granger, C. W. J. & Joyeux, R. (1980). An introduction to long-memory time
    series models and fractional differencing. *Journal of Time Series
    Analysis*, 1(1).
    Hosking, J. R. M. (1981). Fractional differencing. *Biometrika*, 68(1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import _D_MAX, InformationCriteria, SummaryTable
from .._internals import (
    _ComparisonMixin,
    _FractionalIntegrationFit,
    _FractionalIntegrationModel,
    _InvertibilityMixin,
    _SeriesMixin,
    _StationarityMixin,
    _SummaryMixin,
)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ARFIMAResult(
    _SummaryMixin, _SeriesMixin, _ComparisonMixin, _StationarityMixin, _InvertibilityMixin
):
    """A fitted fractionally integrated ARMA.

    Attributes:
        endog: The full observed series.
        fittedvalues: One-step fitted values on the fractionally differenced series.
        resid: One-step residuals on the fractionally differenced series.
        llf: Maximized joint log-likelihood.
        nobs: Observations the likelihood was evaluated on.
        n_params: Free parameter count, including ``d`` and the variance.
        order: Short-memory ``(p, q)``.
        truncation: Length of the fractional-difference filter.
        d: Estimated fractional integration order.
        mean: Estimated mean, or ``None`` when ``trend == "n"``.
        ar_params: Short-memory AR coefficients.
        ma_params: Short-memory MA coefficients.
        sigma2: Innovation variance.
    """

    endog: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int
    order: tuple[int, int]
    truncation: int
    d: float
    mean: float | None
    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    sigma2: float

    @classmethod
    def _from_fit(
        cls,
        fit: _FractionalIntegrationFit,
        model: _FractionalIntegrationModel[ARFIMAResult],
    ) -> ARFIMAResult:
        """Assemble the public result from a raw fit and its specification."""
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            order=model.order,
            truncation=model.truncation,
            d=fit.d,
            mean=fit.mean,
            ar_params=fit.ar_params,
            ma_params=fit.ma_params,
            sigma2=fit.sigma2,
        )

    @property
    def is_stationary(self) -> bool:
        """Whether the process is covariance stationary.

        Overrides the short-memory test because stationarity here has two
        conditions, not one: the fractional order must satisfy ``|d| < 0.5``
        *and* the short-memory autoregressive block must be stable. Inheriting
        the AR-only test would report a unit-root-like ``d = 0.9`` process as
        stationary purely because its ARMA block is well behaved.
        """
        return abs(self.d) < 0.5 and self.stability.is_stable

    @property
    def has_long_memory(self) -> bool:
        """Whether ``d`` is far enough from zero to imply hyperbolic decay."""
        return abs(self.d) > 1e-3

    @property
    def is_mean_reverting(self) -> bool:
        """Whether shocks die out, which holds on the wider range ``d < 1``."""
        return self.d < 1.0 and self.stability.is_stable

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order."""
        out: dict[str, float] = {}
        if self.mean is not None:
            out["mean"] = self.mean
        out["d"] = self.d
        for i, value in enumerate(self.ar_params, start=1):
            out[f"ar.L{i}"] = float(value)
        for i, value in enumerate(self.ma_params, start=1):
            out[f"ma.L{i}"] = float(value)
        out["sigma2"] = self.sigma2
        return out

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        p, q = self.order
        return f"ARFIMA({p}, d, {q})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        ic: InformationCriteria = self.information_criteria
        p, q = self.order
        return SummaryTable(
            title=f"ARFIMA({p}, d, {q}) Results",
            metadata=(
                ("Model", f"ARFIMA({p}, d, {q})"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("d", f"{self.d:.4f}"),
                ("AIC", f"{ic.aic:.3f}"),
                ("Truncation", f"{self.truncation}"),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=(
                f"Stationary: {self.is_stationary}   "
                f"(|d| < 0.5 and stable AR block; d = {self.d:.4f})",
                f"Mean reverting: {self.is_mean_reverting}   Long memory: {self.has_long_memory}",
                f"d is confined to (-{_D_MAX}, {_D_MAX}) by the estimator's reparameterization.",
                "Standard errors are not yet available for this estimator.",
            ),
        )


class ARFIMA(_FractionalIntegrationModel[ARFIMAResult]):
    """Fractionally integrated ARMA specification.

    Args:
        endog: The series.
        order: Short-memory ``(p, q)``.
        trend: ``"c"`` to estimate a mean, ``"n"`` to omit it.
        truncation: Fractional-filter length; defaults to the sample size.
    """

    __slots__ = ()

    def fit(self) -> ARFIMAResult:
        """Estimate ``(mu, d, phi, theta, sigma2)`` by joint maximum likelihood.

        Returns:
            The fitted :class:`ARFIMAResult`.
        """
        return ARFIMAResult._from_fit(self._fit_family(), self)
