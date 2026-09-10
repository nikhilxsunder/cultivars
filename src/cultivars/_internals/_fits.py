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
from ._parameters import (
    _DecayNelsonSiegelParameters,
    _LongMemoryVolatilityParameters,
    _NelsonSiegelParameters,
    _StochasticVolatilityParameters,
    _StructuralParameters,
    _TrendVolatilityParameters,
)
from ._predictors import MeanPredictor
from ._solutions import _PerturbationSolution


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
class _TimeVaryingFit:
    """Raw posterior output of a time-varying-parameter VAR sampler.

    Not a :class:`_BaseFit`: a Gibbs posterior has no maximized likelihood
    or parameter count to report, and pretending otherwise would feed
    information criteria numbers they do not mean.

    Attributes:
        beta_mean: ``(n, D)`` posterior mean coefficient path.
        beta_low: ``(n, D)`` posterior 16th percentile path.
        beta_high: ``(n, D)`` posterior 84th percentile path.
        beta_draws: ``(S, n, D)`` kept coefficient-path draws.
        state_cov: ``(D, D)`` posterior mean of the drift covariance ``Q``.
        sigma_u: ``(k, k)`` posterior mean innovation covariance -- constant
            for the homoskedastic model, the time average under stochastic
            volatility.
        impact_draws: ``(S, k, k)`` draws of the inverse contemporaneous
            matrix ``A^{-1}``, or ``None`` for the homoskedastic model.
        h_draws: ``(S, n, k)`` kept log-variance path draws, or ``None``.
        vol_of_vol: ``(k,)`` posterior mean random-walk variances of the log
            volatilities, or ``None``.
        resid: Residuals at the posterior mean path.
        fittedvalues: One-step means at the posterior mean path.
        nobs: Estimation sample after the training split.
        training: Rows consumed by the training prior.
        n_draws: Total sampler iterations.
        n_burn: Burn-in iterations discarded.
        thin: Keep-every-``thin`` thinning applied after burn-in.
    """

    beta_mean: npt.NDArray[np.float64]
    beta_low: npt.NDArray[np.float64]
    beta_high: npt.NDArray[np.float64]
    beta_draws: npt.NDArray[np.float64]
    state_cov: npt.NDArray[np.float64]
    sigma_u: npt.NDArray[np.float64]
    impact_draws: npt.NDArray[np.float64] | None
    h_draws: npt.NDArray[np.float64] | None
    vol_of_vol: npt.NDArray[np.float64] | None
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    nobs: int
    training: int
    n_draws: int
    n_burn: int
    thin: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorMarkovSwitchingFit(_BaseFit):
    """Raw outputs of a Markov-switching vector autoregression fit.

    Attributes:
        transition: Row-stochastic ``(M, M)`` matrix.
        coefficients: ``(M, p, k, k)`` per-regime lag stacks.
        deterministics: ``(M, d, k)`` per-regime deterministic coefficients.
        sigmas: ``(M, k, k)`` per-regime innovation covariances.
        filtered_prob: ``Pr(S_t = m | y_1..t)``, shape ``(nobs, M)``.
        predicted_prob: ``Pr(S_t = m | y_1..t-1)``, shape ``(nobs, M)``.
        smoothed_prob: ``Pr(S_t = m | y_1..T)``, shape ``(nobs, M)``.
        ergodic_prob: Stationary distribution of ``transition``.
        expected_durations: ``1 / (1 - P_mm)``, in periods.
        n_iter: EM iterations used by the refining run.
        converged: Whether the refining run met its tolerance.
    """

    transition: npt.NDArray[np.float64]
    coefficients: npt.NDArray[np.float64]
    deterministics: npt.NDArray[np.float64]
    sigmas: npt.NDArray[np.float64]
    filtered_prob: npt.NDArray[np.float64]
    predicted_prob: npt.NDArray[np.float64]
    smoothed_prob: npt.NDArray[np.float64]
    ergodic_prob: npt.NDArray[np.float64]
    expected_durations: npt.NDArray[np.float64]
    n_iter: int
    converged: bool


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


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorQuantileFit:
    """Raw outputs of a quantile VAR fit, one system per quantile.

    Not a :class:`_BaseFit`: check-loss minimization has no likelihood, so
    there is no ``llf`` to carry and no information criteria to derive, and a
    record that pretended otherwise would feed comparison machinery numbers
    that do not mean what it assumes.

    Attributes:
        quantiles: The estimated quantile levels, ascending.
        coefficient_stacks: ``(Q, p, k, k)`` lag stacks, one per quantile.
        deterministics: ``(Q, n_det, k)`` deterministic coefficients.
        fittedvalues: ``(Q, n, k)`` conditional quantile paths.
        resid: ``(Q, n, k)`` quantile residuals ``y - fitted``.
        loss: ``(Q, k)`` total check loss per equation at the optimum.
        loss_location: ``(Q, k)`` check loss of the unconditional quantile,
            the intercept-only benchmark the pseudo-``R``:sup:`1` is read
            against (Koenker & Machado 1999).
        nobs: Effective sample size.
    """

    quantiles: tuple[float, ...]
    coefficient_stacks: npt.NDArray[np.float64]
    deterministics: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    loss: npt.NDArray[np.float64]
    loss_location: npt.NDArray[np.float64]
    nobs: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorFunctionalFit:
    """Raw outputs of a functional-coefficient VAR fit.

    Not a :class:`_BaseFit`: a kernel estimator maximizes no likelihood and
    has no integer parameter count -- its complexity is the trace of the
    smoother matrix, carried here as ``effective_params``.

    Attributes:
        delay: Delay of the state variable.
        state_values: The delayed state ``z_t``, aligned with ``resid``.
        grid: ``(G,)`` state values the coefficient curves are evaluated on.
        curves: ``(G, w, k)`` local-linear level coefficients, design-major:
            ``curves[g, :, i]`` is equation ``i``'s coefficient vector at
            ``grid[g]``.
        curve_se: ``(G, w, k)`` pointwise standard errors of ``curves``.
        bandwidth: The bandwidth the curves were estimated at.
        bandwidth_searched: Whether the bandwidth came from cross-validation.
        effective_params: Trace of the smoother matrix at that bandwidth.
        sigma_u: ``(k, k)`` innovation covariance, corrected by the effective
            degrees of freedom.
        resid: Residuals of the local fits at the observed states.
        fittedvalues: One-step conditional means at the observed states.
        nobs: Effective sample size.
    """

    delay: int
    state_values: npt.NDArray[np.float64]
    grid: npt.NDArray[np.float64]
    curves: npt.NDArray[np.float64]
    curve_se: npt.NDArray[np.float64]
    bandwidth: float
    bandwidth_searched: bool
    effective_params: float
    sigma_u: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    nobs: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorConjugateFit:
    """Raw posterior output of a conjugate Bayesian VAR.

    Not a :class:`_BaseFit`: the honest scalar here is the log marginal
    likelihood, not a maximized log likelihood, and information criteria
    derived from a posterior would rank nothing meaningful.

    Attributes:
        coefficient_stack: ``(p, k, k)`` lag stack at the posterior mean.
        deterministic: ``(n_det, k)`` deterministic block at the posterior
            mean.
        beta_mean: ``(w, k)`` full posterior mean coefficient matrix.
        sigma_u: ``(k, k)`` posterior mean innovation covariance.
        beta_draws: ``(S, w, k)`` coefficient draws.
        sigma_draws: ``(S, k, k)`` covariance draws.
        log_marginal_likelihood: Log marginal likelihood of the *sample*,
            with the dummy-observation contribution divided out.
        posterior_df: Inverse-Wishart posterior degrees of freedom.
        resid: Residuals at the posterior mean, over the sample rows only.
        fittedvalues: One-step means at the posterior mean.
        nobs: Effective sample size, dummy rows excluded.
        n_dummy: Artificial rows the prior contributed.
    """

    coefficient_stack: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    beta_mean: npt.NDArray[np.float64]
    sigma_u: npt.NDArray[np.float64]
    beta_draws: npt.NDArray[np.float64]
    sigma_draws: npt.NDArray[np.float64]
    log_marginal_likelihood: float
    posterior_df: float
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    nobs: int
    n_dummy: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorHierarchicalFit(_VectorConjugateFit):
    """Raw output of the hierarchical (Giannone-Lenza-Primiceri) sampler.

    Everything the conjugate record carries -- with ``beta_mean`` and
    ``sigma_u`` now means over the retained draws, marginal over the
    hyperparameters under the full method -- plus the hyperparameter layer.
    ``log_marginal_likelihood`` is the value at the hyperparameter mode,
    conditional on it; the full model evidence would integrate over the
    hyperprior and is deliberately not estimated here.

    Attributes:
        hyper_names: One label per hyperparameter, in draw-column order.
        hyper_mode: Posterior-mode hyperparameter vector.
        hyper_draws: ``(S, d)`` kept hyperparameter draws -- empty under
            empirical Bayes.
        acceptance: Metropolis acceptance rate over the kept span; ``nan``
            under empirical Bayes.
        method: ``"full"`` or ``"empirical"``.
    """

    hyper_names: tuple[str, ...]
    hyper_mode: npt.NDArray[np.float64]
    hyper_draws: npt.NDArray[np.float64]
    acceptance: float
    method: str


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorStudentFit:
    """Raw posterior output of the Student-t Bayesian VAR sampler.

    Not a :class:`_BaseFit` and carrying no marginal likelihood: the t
    likelihood breaks the conjugacy that made the Gaussian model's evidence
    a closed form, and a simulated stand-in would not deserve the name.

    Attributes:
        coefficient_stack: ``(p, k, k)`` lag stack at the posterior mean.
        deterministic: ``(n_det, k)`` deterministic block at the posterior
            mean.
        beta_mean: ``(w, k)`` posterior mean coefficient matrix.
        sigma_u: ``(k, k)`` posterior mean *scale* matrix -- the innovation
            covariance is ``sigma_u * df / (df - 2)``.
        beta_draws: ``(S, w, k)`` kept coefficient draws.
        sigma_draws: ``(S, k, k)`` kept scale draws.
        df: Posterior mean degrees of freedom (the stated value when fixed).
        df_draws: ``(S,)`` kept degrees-of-freedom draws; empty when fixed.
        weight_mean: ``(n,)`` posterior mean latent precision weights --
            small values mark the dates the t distribution treats as
            outliers.
        resid: Residuals at the posterior mean.
        fittedvalues: One-step means at the posterior mean.
        nobs: Effective sample size.
        n_dummy: Artificial rows the prior contributed.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.
    """

    coefficient_stack: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    beta_mean: npt.NDArray[np.float64]
    sigma_u: npt.NDArray[np.float64]
    beta_draws: npt.NDArray[np.float64]
    sigma_draws: npt.NDArray[np.float64]
    df: float
    df_draws: npt.NDArray[np.float64]
    weight_mean: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    nobs: int
    n_dummy: int
    n_draws: int
    n_burn: int
    thin: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorVolatilityFit:
    """Raw posterior output of the stochastic-volatility Bayesian VAR sampler.

    Attributes:
        coefficient_stack: ``(p, k, k)`` lag stack at the posterior mean.
        deterministic: ``(n_det, k)`` deterministic block at the posterior
            mean.
        beta_mean: ``(w, k)`` posterior mean coefficient matrix.
        sigma_u: ``(k, k)`` posterior mean *end-of-sample* covariance.
        beta_draws: ``(S, w, k)`` kept coefficient draws.
        sigma_draws: ``(S, k, k)`` kept end-of-sample covariance draws.
        h_draws: ``(S, n, k)`` kept log-variance path draws.
        impact_draws: ``(S, k, k)`` kept draws of ``A^{-1}``.
        vol_of_vol: ``(k,)`` posterior mean random-walk variances of the
            log volatilities.
        resid: Residuals at the posterior mean.
        fittedvalues: One-step means at the posterior mean.
        nobs: Effective sample size.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.
    """

    coefficient_stack: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    beta_mean: npt.NDArray[np.float64]
    sigma_u: npt.NDArray[np.float64]
    beta_draws: npt.NDArray[np.float64]
    sigma_draws: npt.NDArray[np.float64]
    h_draws: npt.NDArray[np.float64]
    impact_draws: npt.NDArray[np.float64]
    vol_of_vol: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    nobs: int
    n_draws: int
    n_burn: int
    thin: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorSparseFit:
    """Raw outputs of a penalized (sparse) VAR fit.

    Not a :class:`_BaseFit`: a penalized objective maximizes no likelihood,
    and the honest complexity number is the nonzero count, not a parameter
    count an information criterion could charge for.

    Attributes:
        coefficient_stack: ``(p, k, k)`` lag stack at the selected penalty.
        deterministic: ``(n_det, k)`` unpenalized deterministic block.
        sigma_u: ``(k, k)`` residual covariance, corrected by the average
            per-equation nonzero count.
        resid: Residuals over the effective sample.
        fittedvalues: One-step means over the effective sample.
        penalty: The penalty family estimated under.
        lam: The penalty level estimated at.
        lambda_path: The candidate path, descending; empty when the caller
            stated ``lam``.
        cv_errors: Rolling one-step squared forecast error summed over the
            validation span, aligned with ``lambda_path``; empty when the
            caller stated ``lam``.
        n_nonzero: Nonzero penalized coefficients at the solution.
        nobs: Effective sample size.
    """

    coefficient_stack: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    sigma_u: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    penalty: str
    lam: float
    lambda_path: npt.NDArray[np.float64]
    cv_errors: npt.NDArray[np.float64]
    n_nonzero: int
    nobs: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorGraphicalFit(_VectorSparseFit):
    """Raw outputs of a graphical VAR fit: the sparse fit plus the precision.

    Attributes:
        precision: ``(k, k)`` sparse residual precision from symmetrized
            nodewise regressions.
        partial_correlations: ``(k, k)`` partial correlations implied by
            the precision, unit diagonal.
        lam_nodes: ``(k,)`` the per-node penalty levels cross-validation
            selected.
    """

    precision: npt.NDArray[np.float64]
    partial_correlations: npt.NDArray[np.float64]
    lam_nodes: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorGibbsFit:
    """Raw posterior output of the non-conjugate Gibbs BVAR sampler.

    Not a :class:`_BaseFit`, and -- unlike the conjugate record -- carrying
    no marginal likelihood at all: with the coefficient prior independent of
    the covariance, the evidence has no closed form, and a simulated
    stand-in (harmonic means and their relatives) would not deserve the
    name.

    Attributes:
        coefficient_stack: ``(p, k, k)`` lag stack at the posterior mean.
        deterministic: ``(n_det, k)`` deterministic block at the posterior
            mean.
        beta_mean: ``(w, k)`` posterior mean coefficient matrix.
        sigma_u: ``(k, k)`` posterior mean innovation covariance.
        beta_draws: ``(S, w, k)`` kept coefficient draws.
        sigma_draws: ``(S, k, k)`` kept covariance draws.
        shrinkage: ``(w, k)`` posterior mean of the adaptive prior's
            per-coefficient diagnostic -- inclusion probabilities for a
            selection prior, local-global scales for a global-local one --
            empty for a static prior.
        shrinkage_label: What ``shrinkage`` is; empty for a static prior.
        resid: Residuals at the posterior mean, over the sample rows only.
        fittedvalues: One-step means at the posterior mean.
        nobs: Effective sample size, dummy rows excluded.
        n_dummy: Artificial rows the prior contributed.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.
    """

    coefficient_stack: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    beta_mean: npt.NDArray[np.float64]
    sigma_u: npt.NDArray[np.float64]
    beta_draws: npt.NDArray[np.float64]
    sigma_draws: npt.NDArray[np.float64]
    shrinkage: npt.NDArray[np.float64]
    shrinkage_label: str
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    nobs: int
    n_dummy: int
    n_draws: int
    n_burn: int
    thin: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _StructuralFit:
    """Raw output of the structural time-series maximum-likelihood fit.

    Attributes:
        params: The maximized parameter record.
        llf: Exact Gaussian log-likelihood under the approximate-diffuse
            initialization the system builder documents.
        n_params: Free parameters the likelihood was maximized over.
        nobs: Observations.
        smoothed_state: Full-sample state means, ``(n, m)``.
        smoothed_state_cov: Full-sample state covariances, ``(n, m, m)``.
        slices: Component name to state index range.
    """

    params: _StructuralParameters
    llf: float
    n_params: int
    nobs: int
    smoothed_state: npt.NDArray[np.float64]
    smoothed_state_cov: npt.NDArray[np.float64]
    slices: dict[str, slice]


@dataclass(frozen=True, kw_only=True, slots=True)
class _NelsonSiegelFit:
    """Raw output of the dynamic Nelson-Siegel maximum-likelihood fit.

    Attributes:
        params: The maximized parameter record.
        llf: Exact Gaussian log-likelihood.
        n_params: Free parameters the likelihood was maximized over.
        nobs: Observations (curve dates).
        factors: Smoothed factor paths, ``(n, 3)``.
        factor_cov: Smoothed factor covariances, ``(n, 3, 3)``.
    """

    params: _NelsonSiegelParameters
    llf: float
    n_params: int
    nobs: int
    factors: npt.NDArray[np.float64]
    factor_cov: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True, slots=True)
class _StochasticVolatilityFit:
    """Raw output of a point (quasi- or particle-) likelihood SV fit.

    Attributes:
        params: The maximized parameter record.
        llf: The criterion at the optimum: the exact quasi-likelihood of
            the linearized model under ``method="qml"``, or a particle
            *estimate* of the exact likelihood under ``method="particle"``.
        method: ``"qml"`` or ``"particle"``.
        n_params: Free parameters the criterion was maximized over.
        nobs: Observations.
        log_variance: Smoothed log-variance path, ``(n,)``.
        log_variance_std: Smoothed log-variance standard deviations,
            ``(n,)``.
    """

    params: _StochasticVolatilityParameters
    llf: float
    method: str
    n_params: int
    nobs: int
    log_variance: npt.NDArray[np.float64]
    log_variance_std: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True, slots=True)
class _VolatilityDrawsFit:
    """Raw output of the Kim-Shephard-Chib Gibbs sampler for the SV model.

    Attributes:
        mu_draws: ``(S,)`` kept draws of the log-variance mean.
        phi_draws: ``(S,)`` kept draws of the persistence.
        sigma2_draws: ``(S,)`` kept draws of the log-variance innovation
            variance.
        mean_draws: ``(S,)`` kept draws of the observation mean (all zero
            when the mean is fixed).
        h_draws: ``(S, n)`` kept log-variance paths.
        nobs: Observations.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.
    """

    mu_draws: npt.NDArray[np.float64]
    phi_draws: npt.NDArray[np.float64]
    sigma2_draws: npt.NDArray[np.float64]
    mean_draws: npt.NDArray[np.float64]
    h_draws: npt.NDArray[np.float64]
    nu_draws: npt.NDArray[np.float64] | None
    nu_acceptance: float | None
    nobs: int
    n_draws: int
    n_burn: int
    thin: int

    @property
    def params(self) -> _StochasticVolatilityParameters:
        """The posterior-mean parameter record."""
        return _StochasticVolatilityParameters(
            mu=float(self.mu_draws.mean()),
            phi=float(self.phi_draws.mean()),
            sigma2=float(self.sigma2_draws.mean()),
            mean=float(self.mean_draws.mean()),
            nu=float(self.nu_draws.mean()) if self.nu_draws is not None else None,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class _ParticleChainFit:
    """Raw output of a particle marginal Metropolis-Hastings run.

    Attributes:
        theta_draws: ``(S, d)`` kept draws in the *unconstrained* space
            the chain moved in.
        loglik_draws: ``(S,)`` particle log-likelihood estimates carried
            with each kept draw.
        log_posterior_draws: ``(S,)`` the log-prior plus log-likelihood
            estimate at each kept draw.
        acceptance_rate: Fraction of proposals accepted over the whole
            run, burn-in included.
        proposal_cov: The ``(d, d)`` proposal covariance the chain ended
            with, after adaptation.
        n_particles: Particles per likelihood estimate.
        n_draws: Total iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.
    """

    theta_draws: npt.NDArray[np.float64]
    loglik_draws: npt.NDArray[np.float64]
    log_posterior_draws: npt.NDArray[np.float64]
    acceptance_rate: float
    proposal_cov: npt.NDArray[np.float64]
    n_particles: int
    n_draws: int
    n_burn: int
    thin: int

    @property
    def n_kept(self) -> int:
        """Kept draws."""
        return int(self.theta_draws.shape[0])


@dataclass(frozen=True, kw_only=True, slots=True)
class _TrendVolatilityFit:
    """Raw output of the unobserved-components stochastic-volatility Gibbs sampler.

    Attributes:
        trend_draws: ``(S, n)`` kept trend paths ``tau_t``.
        h_draws: ``(S, n)`` kept log variances of the irregular.
        q_draws: ``(S, n)`` kept log variances of the trend innovation.
        gamma2_irregular_draws: ``(S,)`` vol-of-vol draws of ``h`` (constant
            across draws when fixed).
        gamma2_trend_draws: ``(S,)`` vol-of-vol draws of ``q``.
        gamma_fixed: Whether the vol-of-vol parameters were held fixed.
        nobs: Observations.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.
    """

    trend_draws: npt.NDArray[np.float64]
    h_draws: npt.NDArray[np.float64]
    q_draws: npt.NDArray[np.float64]
    gamma2_irregular_draws: npt.NDArray[np.float64]
    gamma2_trend_draws: npt.NDArray[np.float64]
    gamma_fixed: bool
    nobs: int
    n_draws: int
    n_burn: int
    thin: int

    @property
    def params(self) -> _TrendVolatilityParameters:
        """The posterior-mean parameter record."""
        return _TrendVolatilityParameters(
            gamma2_irregular=float(self.gamma2_irregular_draws.mean()),
            gamma2_trend=float(self.gamma2_trend_draws.mean()),
            h0=float(self.h_draws[:, 0].mean()),
            q0=float(self.q_draws[:, 0].mean()),
            tau0=float(self.trend_draws[:, 0].mean()),
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class _DecayNelsonSiegelFit:
    """Raw output of the time-varying-decay Nelson-Siegel fit.

    Attributes:
        params: The maximized parameter record.
        llf: The Gaussian-approximation log-likelihood of the chosen filter
            (extended or unscented); *not* an exact likelihood.
        filter: ``"extended"`` or ``"unscented"``.
        n_params: Free parameters the likelihood was maximized over.
        nobs: Curve dates.
        factors: Smoothed factor paths, ``(n, 3)``.
        factor_cov: Smoothed factor covariances, ``(n, 3, 3)``.
        log_decay: Smoothed log-decay path, ``(n,)``.
        log_decay_std: Smoothed log-decay standard deviations, ``(n,)``.
    """

    params: _DecayNelsonSiegelParameters
    llf: float
    filter: str
    n_params: int
    nobs: int
    factors: npt.NDArray[np.float64]
    factor_cov: npt.NDArray[np.float64]
    log_decay: npt.NDArray[np.float64]
    log_decay_std: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True, slots=True)
class _PerturbationFit:
    """Raw output of a perturbation-DSGE point fit.

    Attributes:
        theta: The structural parameter vector at the optimum, in the
            model's own (constrained) space.
        solution: The perturbation solution at ``theta``.
        llf: The criterion at the optimum: exact under
            ``filter="kalman"`` (first order only), a Gaussian approximation
            under the extended/unscented filters, a particle estimate
            under ``filter="particle"``.
        filter: Which filter produced ``llf``.
        n_params: Free structural parameters.
        nobs: Observations.
        filtered_state: Filtered state means (pruned state, ``(n, 2 n_x)``
            at second order, ``(n, n_x)`` at first).
    """

    theta: npt.NDArray[np.float64]
    solution: _PerturbationSolution
    llf: float
    filter: str
    n_params: int
    nobs: int
    filtered_state: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True, slots=True)
class _VolatilityStructuralFit:
    """Raw output of the stochastic-volatility identification sampler.

    Every draw is already normalized to one labelling: columns of the
    impact matrix are matched to a reference by assignment on their
    absolute cosine similarity and signed to agree with it, so the draws
    below describe one posterior mode rather than a mixture of ``2**k k!``
    relabellings.

    Attributes:
        impact_draws: ``(S, k, k)`` kept impact matrices ``B = A**-1``.
        h_draws: ``(S, T, k)`` kept structural log-variance paths, one
            column per shock, each with zero sample mean; the level sits
            in the matching column of ``impact_draws``.
        phi_draws: ``(S, k)`` kept log-variance persistences.
        sigma2_draws: ``(S, k)`` kept log-variance innovation variances.
        relabel_rate: Fraction of kept draws whose column order had to be
            permuted to match the reference -- near zero when the shocks
            are well separated, and the first thing to read when they
            are not.
        nobs: Observations.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.
    """

    impact_draws: npt.NDArray[np.float64]
    h_draws: npt.NDArray[np.float64]
    phi_draws: npt.NDArray[np.float64]
    sigma2_draws: npt.NDArray[np.float64]
    relabel_rate: float
    nobs: int
    n_draws: int
    n_burn: int
    thin: int


@dataclass(frozen=True, kw_only=True, slots=True)
class _LongMemoryVolatilityFit:
    """Raw output of the Whittle fit of the long-memory SV model.

    Attributes:
        params: The maximized parameter record.
        llf: The Whittle criterion at the optimum, sign-flipped to read as
            a log-likelihood: the frequency-domain quasi-likelihood of the
            linearized model, comparable only with other Whittle fits.
        n_params: Free parameters the criterion was maximized over.
        nobs: Observations.
        n_frequencies: Fourier ordinates the criterion summed over.
        log_variance: Smoothed log-variance path, ``(n,)``, from the
            spectral (Wiener-Kolmogorov) smoother under a circular
            approximation.
        log_variance_std: The smoother's stationary root mean squared
            error, one value repeated ``(n,)`` times.
    """

    params: _LongMemoryVolatilityParameters
    llf: float
    n_params: int
    nobs: int
    n_frequencies: int
    log_variance: npt.NDArray[np.float64]
    log_variance_std: npt.NDArray[np.float64]
