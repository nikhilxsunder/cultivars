# filepath: /src/cultivars/_internals/_fits.py
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

"""Fit records: the raw numeric output of an estimator, before result assembly.

Every class here is a frozen, slotted dataclass and nothing else. A fit carries
what an estimator produced -- coefficients, residuals, a likelihood, a
parameter count -- together with the quantities derivable from those alone. It
carries no estimation logic: how a fit is produced belongs to the model that
owns the specification, in :mod:`cultivars._internals._models`.

That separation is what keeps this module a leaf. It imports numpy, the
dataclass machinery, and one predictor protocol; nothing else in the package
depends on it in the other direction, so a fit record can be constructed,
compared, or serialized without dragging in an optimizer.

The hierarchies are shallow and share fields only where the fields genuinely
mean the same thing across the group. ``_ConditionalVarianceFit`` is the one
multi-level case: its two subclasses parameterize the variance path in ways
that are not comparable, so only the path itself and the two quantities derived
from it live on the shared base.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._predictors import MeanPredictor


@dataclass(frozen=True, kw_only=True, slots=True)
class _BaseFit(ABC):
    """Root of every fitted-result object.

    Carries the likelihood summary every estimator produces and derives the
    information criteria from it, so no subclass stores ``aic``/``bic``/``hqic``
    as fields that could drift out of sync with ``llf``.

    Attributes:
        llf: Maximized log-likelihood.
        nobs: Observations the likelihood was evaluated on.
        n_params: Free parameter count, including the innovation variance.
    """

    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _AutoRegressionFit(_BaseFit):
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


@dataclass(frozen=True, kw_only=True, slots=True)
class _BoxJenkinsFit(_BaseFit):
    """Raw outputs of a seasonal ARIMA-with-regressors fit."""

    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    seasonal_ar_params: npt.NDArray[np.float64]
    seasonal_ma_params: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    sigma2: float


@dataclass(frozen=True, kw_only=True, slots=True)
class _FractionalIntegrationFit(_BaseFit):
    """Raw outputs of a fractionally integrated ARMA fit."""

    d: float
    mean: float | None
    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    sigma2: float


@dataclass(frozen=True, kw_only=True, slots=True)
class _ConditionalVarianceFit(_BaseFit):
    """Raw outputs common to every conditional-variance fit.

    Abstract in intent: it carries only what every family in the group actually
    produces -- a mean intercept, a variance intercept, and a fitted variance
    path -- plus the two quantities derived from the path alone. Everything
    that parameterizes *how* the path was produced belongs to a subclass,
    because those blocks are not comparable across the group.

    Attributes:
        const: Mean intercept, or ``None`` when ``mean == "zero"``.
        omega: Variance intercept.
        conditional_variance: The fitted variance path.
    """

    const: float | None
    omega: float
    conditional_variance: npt.NDArray[np.float64]

    @property
    def conditional_volatility(self) -> npt.NDArray[np.float64]:
        """The fitted conditional standard deviation."""
        return np.sqrt(self.conditional_variance)

    @property
    def standardized_resid(self) -> npt.NDArray[np.float64]:
        """Mean residuals scaled by the fitted conditional volatility."""
        return self.resid / self.conditional_volatility


@dataclass(frozen=True, kw_only=True, slots=True)
class _ShortMemoryVarianceFit(_ConditionalVarianceFit):
    """Raw outputs of a finite-order variance fit: GARCH, GJR, or EGARCH.

    Attributes:
        vol: The family that produced the fit, needed because persistence is
            not the same functional of the coefficients across the three.
        ar_params: Conditional-mean AR coefficients; empty when ``ar_lags == 0``.
        alpha: Coefficients on the shock magnitude.
        gamma: Asymmetry coefficients; empty for the symmetric family.
        beta: Persistence coefficients.
    """

    vol: str
    ar_params: npt.NDArray[np.float64]
    alpha: npt.NDArray[np.float64]
    gamma: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]

    @property
    def persistence(self) -> float:
        """The decay rate of a shock to the conditional variance.

        For the level families this is the sum of the coefficients, with the
        asymmetry block at half weight -- its unconditional frequency under a
        symmetric innovation distribution. For the log-variance family it is
        the autoregressive root of the log variance alone, since the magnitude
        and sign terms are mean-zero innovations rather than persistence.
        """
        if self.vol == "EGARCH":
            return float(self.beta.sum())
        return float(self.alpha.sum() + 0.5 * self.gamma.sum() + self.beta.sum())


@dataclass(frozen=True, kw_only=True, slots=True)
class _FractionalVarianceFit(_ConditionalVarianceFit):
    """Raw outputs of a fractionally integrated variance fit.

    Attributes:
        phi: Short-memory numerator weight of the fractional polynomial.
        d: Fractional integration order.
        beta: Denominator weight.
    """

    phi: float
    d: float
    beta: float


@dataclass(frozen=True, kw_only=True, slots=True)
class _MeanFunctionFit(_BaseFit):
    """Raw outputs of a neural mean-function fit."""

    sigma2: float


@dataclass(frozen=True, kw_only=True, slots=True)
class _NeuralAutoregressionFit(_MeanFunctionFit):
    """Raw outputs of an autoregressive neural mean-function fit."""

    predictor: MeanPredictor


@dataclass(frozen=True, kw_only=True, slots=True)
class _NeuralThresholdFit(_MeanFunctionFit):
    """Raw outputs of a threshold neural mean-function fit."""

    delay: int
    threshold: float
    lower_predictor: MeanPredictor
    upper_predictor: MeanPredictor
    threshold_variable: npt.NDArray[np.float64] | None
    self_exciting: bool
    ssr: float
    n_lower: int
    n_upper: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _MarkovSwitchingFit(_BaseFit):
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
    n_iter: int
    converged: bool


@dataclass(frozen=True, kw_only=True, slots=True)
class _ThresholdFit(_BaseFit):
    """Raw outputs of a two-regime threshold grid search."""

    delay: int
    threshold: float
    lower_params: npt.NDArray[np.float64]
    upper_params: npt.NDArray[np.float64]
    sigma2: float
    ssr: float
    n_lower: int
    n_upper: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _SmoothTransitionFit(_BaseFit):
    """Raw outputs of a smooth-transition autoregression fit."""

    delay: int
    threshold: float
    gamma: float
    lower_params: npt.NDArray[np.float64]
    upper_params: npt.NDArray[np.float64]
    sigma2: float
    ssr: float
