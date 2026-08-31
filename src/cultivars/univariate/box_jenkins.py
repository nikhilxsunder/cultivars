# filepath: /src/cultivars/univariate/arma.py
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

"""ARMA / ARIMA / SARIMA / SARIMAX -- one engine behind four public fronts.

All four are the same estimator. The conditional mean is a multiplicative
seasonal ARMA in the Harvey state-space form, so the likelihood is exact and
runs through the Kalman filter; integration is applied by differencing before
estimation; deterministic terms and exogenous regressors enter the observation
intercept, which makes ``ARIMAX`` and ``SARIMAX`` nothing more than ``ARIMA``
and ``SARIMA`` with ``exog`` supplied.

:class:`SARIMAX` is therefore the real specification and the other three are
narrowing constructors over it, not separate models. That is deliberate: a user
who reaches for ``ARMA`` should not be able to pass a differencing order, and a
user who reaches for ``SARIMA`` should not have to spell out ``(0, 0, 0, 0)``.

Stationarity and invertibility are enforced structurally by the
partial-autocorrelation reparameterization, so the optimizer searches an
unconstrained space and the reported diagnostics verify the transform rather
than describing the data. Both are assessed on the *multiplied* polynomials
``phi(L)Phi(L**s)`` and ``theta(L)Theta(L**s)``; checking the non-seasonal
block alone would pass a specification with an explosive seasonal root.

References:
    Box, G., Jenkins, G., Reinsel, G. & Ljung, G. (2015). *Time Series
    Analysis: Forecasting and Control* (5th ed.).
    Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space
    Methods*, ch. 3 (the ARMA state-space form).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import InformationCriteria, SummaryTable, expand_ar, expand_ma, n_deterministic
from .._internals import (
    _BoxJenkinsFit,
    _BoxJenkinsModel,
    _ComparisonMixin,
    _InvertibilityMixin,
    _SeriesMixin,
    _StationarityMixin,
    _SummaryMixin,
)
from ..exceptions import SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ARMAResult(
    _SummaryMixin, _SeriesMixin, _ComparisonMixin, _StationarityMixin, _InvertibilityMixin
):
    """A fitted (seasonal) ARIMA, optionally with exogenous regressors.

    Attributes:
        endog: The full observed series.
        fittedvalues: One-step fitted values on the differenced modeling series.
        resid: One-step residuals on the differenced modeling series.
        llf: Maximized exact log-likelihood.
        nobs: Observations after differencing.
        n_params: Free parameter count, including the innovation variance.
        order: Non-seasonal ``(p, d, q)``.
        seasonal_order: Seasonal ``(P, D, Q, s)``.
        trend: Deterministic specification.
        k_exog: Number of exogenous regressors.
        ar_params: Non-seasonal AR coefficients.
        ma_params: Non-seasonal MA coefficients.
        seasonal_ar_params: Seasonal AR coefficients.
        seasonal_ma_params: Seasonal MA coefficients.
        beta: Coefficients on the deterministic and exogenous block.
        sigma2: Innovation variance.
    """

    endog: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: float
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    trend: str
    k_exog: int
    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    seasonal_ar_params: npt.NDArray[np.float64]
    seasonal_ma_params: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    sigma2: float

    @classmethod
    def _from_fit(cls, fit: _BoxJenkinsFit, model: _BoxJenkinsModel[ARMAResult]) -> ARMAResult:
        """Assemble the public result from a raw fit and its specification."""
        exog = model.exog
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            order=model.order,
            seasonal_order=model.seasonal_order,
            trend=model.trend,
            k_exog=0 if exog is None else int(exog.shape[1]),
            ar_params=fit.ar_params,
            ma_params=fit.ma_params,
            seasonal_ar_params=fit.seasonal_ar_params,
            seasonal_ma_params=fit.seasonal_ma_params,
            beta=fit.beta,
            sigma2=fit.sigma2,
        )

    @property
    def seasonal_period(self) -> int:
        """The seasonal period ``s``; ``0`` when no seasonal block is present."""
        return self.seasonal_order[3]

    def _stationarity_ar(self) -> npt.NDArray[np.float64]:
        """The multiplied polynomial ``phi(L) Phi(L**s)``.

        Assessing the non-seasonal block alone would pass a specification whose
        seasonal root is explosive.
        """
        return expand_ar(self.ar_params, self.seasonal_ar_params, self.seasonal_period)

    def _invertibility_ma(self) -> npt.NDArray[np.float64]:
        """The multiplied polynomial ``theta(L) Theta(L**s)``, sign-corrected."""
        return -expand_ma(self.ma_params, self.seasonal_ma_params, self.seasonal_period)

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order."""
        out: dict[str, float] = {}
        det = n_deterministic(self.trend)
        names = (["const", "trend"][:det]) + [f"x{i}" for i in range(1, self.k_exog + 1)]
        for name, value in zip(names, self.beta, strict=True):
            out[name] = float(value)
        s = self.seasonal_period
        for i, value in enumerate(self.ar_params, start=1):
            out[f"ar.L{i}"] = float(value)
        for i, value in enumerate(self.ma_params, start=1):
            out[f"ma.L{i}"] = float(value)
        for i, value in enumerate(self.seasonal_ar_params, start=1):
            out[f"ar.S.L{i * s}"] = float(value)
        for i, value in enumerate(self.seasonal_ma_params, start=1):
            out[f"ma.S.L{i * s}"] = float(value)
        out["sigma2"] = self.sigma2
        return out

    def _specification(self) -> str:
        """Compact specification label, widening only as the model does."""
        base = f"ARIMA{self.order}"
        if any(self.seasonal_order[:3]):
            base = f"SARIMA{self.order}{self.seasonal_order}"
        return f"{base}+X{self.k_exog}" if self.k_exog else base

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        return self._specification()

    def _notes(self) -> tuple[str, ...]:
        """Closing diagnostics, omitting any that the specification makes vacuous."""
        notes = [
            f"Stationary: {self.is_stationary}   max |AR root| = {self.stability.max_modulus:.4f}"
        ]
        if self.ma_params.size or self.seasonal_ma_params.size:
            notes.append(
                f"Invertible: {self.is_invertible}   "
                f"max |MA root| = {self.invertibility.max_modulus:.4f}"
            )
        notes.append("Standard errors are not yet available for this estimator.")
        return tuple(notes)

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        ic: InformationCriteria = self.information_criteria
        return SummaryTable(
            title=f"{self._specification()} Results",
            metadata=(
                ("Model", self._specification()),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Trend", self.trend),
                ("AIC", f"{ic.aic:.3f}"),
                ("Exog regressors", f"{self.k_exog}"),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=self._notes(),
        )


class SARIMAX(_BoxJenkinsModel[ARMAResult]):
    """Seasonal ARIMA with exogenous regressors -- the general specification.

    Every other class in this module narrows this one. Here alone ``exog`` is
    optional, because this is the form a user reaches for when they want the
    full surface rather than a named special case; :class:`ARIMAX` is the
    non-seasonal variant that makes regressors mandatory.

    Args:
        endog: The endogenous series.
        order: Non-seasonal ``(p, d, q)``.
        seasonal_order: Seasonal ``(P, D, Q, s)``.
        trend: Deterministic specification.
        exog: Optional exogenous regressors, differenced alongside ``endog``.
    """

    __slots__ = ()

    def fit(self) -> ARMAResult:
        """Estimate by exact maximum likelihood.

        Returns:
            The fitted :class:`ARMAResult`.
        """
        return ARMAResult._from_fit(self._fit_family(), self)


class ARIMAX(SARIMAX):
    """ARIMA(p, d, q) with exogenous regressors, no seasonal block.

    A regression with ARIMA errors: the regressors enter the observation
    intercept and the ARIMA structure describes what they leave behind. When
    ``d`` is non-zero the regressors are differenced alongside ``endog``, so
    the coefficients are interpretable on the differenced scale rather than on
    the level. A cointegrating relationship in levels therefore comes back
    correctly -- ``y = 2 x1 - x2 + I(1) error`` recovers ``2.009`` and
    ``-0.999`` at ``d = 1`` -- but a level relationship regressed against
    differenced drivers will not, and that is a property of the specification
    rather than of the estimator.

    ``exog`` is required and an empty block is rejected. That is the whole
    point of the name: an ``ARIMAX`` with no regressors is an :class:`ARIMA`,
    and silently accepting one would let a call site claim a covariate model it
    does not have.

    Args:
        endog: The endogenous series.
        order: ``(p, d, q)``.
        exog: Exogenous regressors, shape ``(nobs,)`` or ``(nobs, k)``.
        trend: Deterministic specification.

    Raises:
        SpecificationError: If ``exog`` is ``None`` or has zero columns.
        DimensionError: If ``exog`` is not aligned with ``endog``.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal((400, 1))
        >>> y = np.cumsum(2.0 * x[:, 0] + rng.standard_normal(400))
        >>> ARIMAX(y, order=(1, 1, 0), exog=x).fit().k_exog
        1
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: tuple[int, int, int],
        exog: npt.ArrayLike,
        trend: str = "c",
    ) -> None:
        """Validate the specification and the data."""
        if exog is None:
            raise SpecificationError(
                "ARIMAX requires exogenous regressors; use ARIMA for a model without them."
            )
        super().__init__(endog, order=order, trend=trend, exog=exog)
        if self.exog is None or self.exog.shape[1] == 0:
            raise SpecificationError(
                "ARIMAX requires at least one exogenous regressor; "
                "use ARIMA for a model without them."
            )


class SARIMA(SARIMAX):
    """Multiplicative seasonal ARIMA, without exogenous regressors.

    Args:
        endog: The endogenous series.
        order: Non-seasonal ``(p, d, q)``.
        seasonal_order: Seasonal ``(P, D, Q, s)``.
        trend: Deterministic specification.
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
        trend: str = "c",
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, order=order, seasonal_order=seasonal_order, trend=trend)


class ARIMA(SARIMAX):
    """ARIMA(p, d, q): non-seasonal differencing, no seasonal block, no regressors.

    Reach for :class:`ARIMAX` when there are exogenous regressors. Splitting the
    two is what makes either name informative: a reader of a call site knows
    from ``ARIMA`` alone that the fit has no external drivers.

    Args:
        endog: The endogenous series.
        order: ``(p, d, q)``.
        trend: Deterministic specification.
    """

    __slots__ = ()

    def __init__(
        self, endog: npt.ArrayLike, *, order: tuple[int, int, int], trend: str = "c"
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, order=order, trend=trend)


class ARMA(SARIMAX):
    """Stationary ARMA(p, q): no differencing, no seasonal block, no regressors.

    Args:
        endog: The endogenous series.
        order: ``(p, q)``.
        trend: Deterministic specification.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> e = rng.standard_normal(600)
        >>> y = np.zeros(600)
        >>> for t in range(1, 600):
        ...     y[t] = 0.7 * y[t - 1] + e[t] + 0.4 * e[t - 1]
        >>> res = ARMA(y, order=(1, 1)).fit()
        >>> res.is_stationary and res.is_invertible
        True
    """

    __slots__ = ()

    def __init__(self, endog: npt.ArrayLike, *, order: tuple[int, int], trend: str = "c") -> None:
        """Validate the specification and the data."""
        p, q = order
        super().__init__(endog, order=(p, 0, q), trend=trend)
