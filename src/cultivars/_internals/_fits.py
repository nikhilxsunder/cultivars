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

from ._inferences import _CoefficientInference
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
    n_params: float


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
        ar_params: Conditional-mean autoregressive coefficients; empty when the
            order is zero.
        ma_params: Conditional-mean moving-average coefficients; empty when the
            order is zero. Present on the shared base rather than on one
            subclass because the mean layer is orthogonal to the variance
            family -- every member of the group can carry an ARMA mean.
        omega: Variance intercept.
        conditional_variance: The fitted variance path.
    """

    const: float | None
    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
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
        alpha: Coefficients on the shock magnitude.
        gamma: Asymmetry coefficients; empty for the symmetric family.
        beta: Persistence coefficients.
    """

    vol: str
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
class _NeuralAutoRegressionFit(_MeanFunctionFit):
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


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorObservedRegimeFit(_BaseFit):
    """Raw outputs shared by the observed-regime vector estimators.

    Attributes:
        delay: Delay of the transition variable.
        threshold: The regime split point, in the transition variable's units.
        threshold_values: The transition variable, aligned with ``resid``.
        lower_coefficients: ``(p, k, k)`` lag stack of the lower regime.
        upper_coefficients: ``(p, k, k)`` lag stack of the upper regime.
        lower_deterministic: Lower-regime deterministic coefficients.
        upper_deterministic: Upper-regime deterministic coefficients.
    """

    delay: int
    threshold: float
    threshold_values: npt.NDArray[np.float64]
    lower_coefficients: npt.NDArray[np.float64]
    upper_coefficients: npt.NDArray[np.float64]
    lower_deterministic: npt.NDArray[np.float64]
    upper_deterministic: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorThresholdFit(_VectorObservedRegimeFit):
    """Raw outputs of a two-regime threshold VAR grid search.

    Attributes:
        lower_sigma_u: Lower-regime innovation covariance, dof-corrected.
        upper_sigma_u: Upper-regime innovation covariance, dof-corrected.
        n_lower: Observations assigned to the lower regime.
        n_upper: Observations assigned to the upper regime.
    """

    lower_sigma_u: npt.NDArray[np.float64]
    upper_sigma_u: npt.NDArray[np.float64]
    n_lower: int
    n_upper: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorSmoothTransitionFit(_VectorObservedRegimeFit):
    """Raw outputs of a smooth-transition VAR fit.

    Attributes:
        gamma: Transition speed, per standard deviation of the transition
            variable.
        transition_scale: The standard deviation ``gamma`` is expressed
            against.
        sigma_u: Common innovation covariance, dof-corrected. One covariance
            rather than two, because smooth weights never partition the
            sample: every observation is a blend of both regimes, so a
            per-regime covariance has no subsample to be estimated from.
    """

    gamma: float
    transition_scale: float
    sigma_u: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorAutoRegressionFit(_BaseFit):
    """The estimated pieces of a reduced-form vector autoregression.

    Attributes:
        coefficients: ``(p, k, k)`` stack of ``A_1, ..., A_p``.
        deterministic: Deterministic coefficients, one row per term.
        sigma_u: Residual covariance with the degrees-of-freedom correction.
        sigma_ml: Residual covariance divided by the effective sample.
        design: The regressor matrix as estimated.
        posterior: Posterior covariance when a prior was applied, else ``None``.
        prior_label: What the prior was, for the summary.
    """

    coefficients: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    sigma_u: npt.NDArray[np.float64]
    sigma_ml: npt.NDArray[np.float64]
    design: npt.NDArray[np.float64]
    posterior: _CoefficientInference | None = None
    prior_label: str = "none"

    @property
    def k_endog(self) -> int:
        """Number of equations."""
        return int(self.sigma_u.shape[0])

    @property
    def order(self) -> int:
        """Autoregressive order."""
        return int(self.coefficients.shape[0])

    @property
    def n_regressors(self) -> int:
        """Regressors per equation."""
        return int(self.design.shape[1])


@dataclass(frozen=True, kw_only=True, slots=True)
class _ExogenousVectorAutoRegressionFit(_VectorAutoRegressionFit):
    """A fit that additionally carries the distributed-lag coefficients.

    Attributes:
        exog_coefficients: ``(s + 1, k, m)`` stack of ``B_0, ..., B_s``, laid
            out so that ``exog_coefficients[j]`` is the ``(k, m)`` matrix on
            ``x_{t-j}`` and index ``0`` is the contemporaneous term.
    """

    exog_coefficients: npt.NDArray[np.float64]

    @property
    def exog_order(self) -> int:
        """Number of exogenous lags beyond the contemporaneous term."""
        return int(self.exog_coefficients.shape[0]) - 1

    @property
    def k_exog(self) -> int:
        """Number of exogenous variables."""
        return int(self.exog_coefficients.shape[2])


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorErrorCorrectionFit(_VectorAutoRegressionFit):
    """The estimated pieces of a vector error-correction model.

    Subclasses the reduced-form fit rather than sitting beside it, because a
    vector error-correction model *is* a vector autoregression written in
    different coordinates and the levels coefficients are not an interpretation
    laid on top -- they are recoverable exactly. The inherited ``coefficients``
    and ``deterministic`` therefore carry the levels representation, computed
    once here so that no consumer has to know the folding rules, while the
    fields below carry the coordinates the model was actually estimated in.

    A conditional specification is the exception: it has no levels
    representation, so ``coefficients`` and ``deterministic`` come back
    zero-length. That is a statement, not a default, and the conditional result
    refuses to propagate it.

    Attributes:
        alpha: ``(k_y, r)`` adjustment loadings.
        beta: ``(k_y + k_x [+ 1], r)`` cointegrating vectors, the extra row
            present when the case restricts a deterministic term to the
            cointegrating space.
        gamma: ``(p - 1, k_y, k_y + k_x)`` short-run coefficients on lagged
            differences of every integrated variable.
        short_run_deterministic: ``(d_s, k_y)`` unrestricted deterministic
            terms, as they entered the regression.
        impact: ``(k_y, k_x)`` contemporaneous response to the weakly exogenous
            differences, zero-width for a closed system.
        eigenvalues: The squared canonical correlations, descending.
    """

    alpha: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    gamma: npt.NDArray[np.float64]
    short_run_deterministic: npt.NDArray[np.float64]
    impact: npt.NDArray[np.float64]
    eigenvalues: npt.NDArray[np.float64]

    @property
    def rank(self) -> int:
        """Cointegrating rank."""
        return int(self.alpha.shape[1])

    @property
    def k_exog(self) -> int:
        """Weakly exogenous integrated regressors carried without equations."""
        return int(self.impact.shape[1])

    @property
    def cointegrating_matrix(self) -> npt.NDArray[np.float64]:
        """The long-run impact matrix ``Pi = alpha beta'``, over the variables only."""
        return self.alpha @ self.beta[: self.k_endog].T
