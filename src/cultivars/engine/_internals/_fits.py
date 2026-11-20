# filepath: /src/cultivars/engine/_internals/_fits.py
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
owns the specification, in :mod:`cultivars.engine._internals._models`.

That separation is what keeps this module a leaf. It imports numpy, the
dataclass machinery, and one predictor protocol; nothing else in the package
depends on it in the other direction, so a fit record can be constructed,
compared, or serialized without dragging in an optimizer.

The hierarchies are shallow and share fields only where the fields genuinely
mean the same thing across the group. :class:`_ConditionalVarianceFit` is the one
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

__all__ = [
    "_AutoRegressionFit",
    "_BaseFit",
    "_BoxJenkinsFit",
    "_ConditionalVarianceFit",
    "_DecayNelsonSiegelFit",
    "_ExogenousVectorAutoRegressionFit",
    "_FactorVolatilityFit",
    "_FractionalIntegrationFit",
    "_FractionalVarianceFit",
    "_LongMemoryVolatilityFit",
    "_MarkovSwitchingFit",
    "_MeanFunctionFit",
    "_NelsonSiegelFit",
    "_NeuralAutoRegressionFit",
    "_NeuralThresholdFit",
    "_ParticleChainFit",
    "_PerturbationFit",
    "_ShortMemoryVarianceFit",
    "_SmoothTransitionFit",
    "_StochasticVolatilityFit",
    "_StructuralFit",
    "_ThresholdFit",
    "_TimeVaryingFit",
    "_TrendVolatilityFit",
    "_VectorAutoRegressionFit",
    "_VectorConjugateFit",
    "_VectorErrorCorrectionFit",
    "_VectorFunctionalFit",
    "_VectorGibbsFit",
    "_VectorGraphicalFit",
    "_VectorHierarchicalFit",
    "_VectorMarkovSwitchingFit",
    "_VectorObservedRegimeFit",
    "_VectorQuantileFit",
    "_VectorSmoothTransitionFit",
    "_VectorSparseFit",
    "_VectorStudentFit",
    "_VectorThresholdFit",
    "_VectorVolatilityFit",
    "_VolatilityDrawsFit",
    "_VolatilityStructuralFit",
]


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
    """In-sample fitted values, aligned with the effective sample."""
    resid: npt.NDArray[np.float64]
    """Residuals, the observations minus ``fittedvalues``."""
    llf: float
    """Maximized log-likelihood."""
    nobs: int
    """Observations the likelihood was evaluated on."""
    n_params: float
    """Free parameter count, including the innovation variance."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _AutoRegressionFit(_BaseFit):
    r"""Raw outputs of an autoregressive fit, before public result assembly.

    The fitted equation is

    .. math::

       y_t = c + \delta t + \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \varepsilon_t,
       \qquad \varepsilon_t \sim (0, \sigma^2),

    with :attr:`const` as :math:`c`, :attr:`trend_coeff` as :math:`\delta`,
    :attr:`ar_params` as :math:`(\phi_1, \ldots, \phi_p)` and :attr:`sigma2` as
    :math:`\sigma^2`. A deterministic term the specification leaves out is stored
    as ``None`` rather than as zero, so a consumer can tell an estimated zero from
    an absent term. This is the estimator's output before anything is derived
    from it: the standard errors, the stability verdict and the summary are
    assembled on the public result.

    Attributes:
        const: Intercept, or ``None`` when ``trend == "n"``.
        trend_coeff: Linear-trend slope, or ``None`` unless ``trend == "ct"``.
        ar_params: Autoregressive coefficients.
        sigma2: Innovation variance.
        llf: Maximized log-likelihood (conditional for CSS, exact otherwise).
        nobs: Observations the likelihood was evaluated on.
        resid: One-step residuals.
        fittedvalues: One-step fitted values.

    See Also:
        * :meth:`._AutoRegressionModel._fit_css` -- returns an instance.
        * :meth:`._AutoRegressionModel._fit_exact` -- returns an instance.
        * :meth:`._AutoRegressionModel._fit_family` -- returns an instance.
    """

    const: float | None
    """Intercept, or ``None`` when ``trend == "n"``."""
    trend_coeff: float | None
    """Linear-trend slope, or ``None`` unless ``trend == "ct"``."""
    ar_params: npt.NDArray[np.float64]
    """Autoregressive coefficients."""
    sigma2: float
    """Innovation variance."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _BoxJenkinsFit(_BaseFit):
    r"""Raw outputs of a seasonal ARIMA-with-regressors fit.

    The model is the regression with seasonal ARMA errors stated on
    :class:`~cultivars.engine._internals._parameters._BoxJenkinsParameters`,

    .. math::

       \phi(L)\, \Phi(L^s)\, u_t = \theta(L)\, \Theta(L^s)\, \varepsilon_t,
       \qquad u_t = w_t - x_t^\top \beta,

    and the six coefficient blocks here are that record's blocks at the optimum,
    on the natural scale. The base record adds what every fit carries: the
    maximized log likelihood, the sample it was evaluated on, the one-step
    residuals and the fitted values.
    """

    ar_params: npt.NDArray[np.float64]
    """Non-seasonal autoregressive coefficients, ``(p,)``."""
    ma_params: npt.NDArray[np.float64]
    r"""Non-seasonal moving-average coefficients, ``(q,)``, in the sign convention
    :math:`\theta(L) = 1 + \theta_1 L + \cdots`.
    """
    seasonal_ar_params: npt.NDArray[np.float64]
    """Seasonal autoregressive coefficients, ``(P,)``."""
    seasonal_ma_params: npt.NDArray[np.float64]
    """Seasonal moving-average coefficients, ``(Q,)``, in the same sign convention as
    ``ma_params``.
    """
    beta: npt.NDArray[np.float64]
    """Coefficients on the deterministic and exogenous regressors."""
    sigma2: float
    """Innovation variance."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _FractionalIntegrationFit(_BaseFit):
    r"""Raw outputs of a fractionally integrated ARMA fit.

    The model is the ARFIMA process stated on
    :class:`~cultivars.engine._internals._parameters._FractionalIntegrationParameters`,

    .. math::

       \phi(L)\, (1 - L)^d\, (y_t - \mu) = \theta(L)\, \varepsilon_t,

    with :attr:`d` the memory parameter and :attr:`mean` the level :math:`\mu`.
    The autocorrelations of such a process decay like :math:`k^{2d - 1}`, which
    is not summable for :math:`d > 0`: :attr:`d` is therefore the one number that
    separates this fit from a short-memory ARMA with the same polynomials.
    """

    d: float
    """Fractional integration order."""
    mean: float | None
    """Estimated mean, or ``None`` when the mean is not estimated."""
    ar_params: npt.NDArray[np.float64]
    """Short-memory autoregressive coefficients, ``(p,)``."""
    ma_params: npt.NDArray[np.float64]
    """Short-memory moving-average coefficients, ``(q,)``."""
    sigma2: float
    """Innovation variance."""


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
    """Mean intercept, or ``None`` when ``mean == "zero"``."""
    ar_params: npt.NDArray[np.float64]
    """Conditional-mean autoregressive coefficients; empty when the order is zero."""
    ma_params: npt.NDArray[np.float64]
    """Conditional-mean moving-average coefficients; empty when the order is zero.

    Present on the shared base rather than on one subclass because the mean layer
    is orthogonal to the variance family -- every member of the group can carry an
    ARMA mean.
    """
    omega: float
    """Variance intercept."""
    conditional_variance: npt.NDArray[np.float64]
    """The fitted variance path."""

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
    r"""Raw outputs of a finite-order variance fit: GARCH, GJR, or EGARCH.

    The three lag blocks are those of
    :class:`~cultivars.engine._internals._parameters._ConditionalVarianceParameters`
    at the optimum, entering the level recursion for GARCH and GJR and the log
    recursion for EGARCH. :attr:`vol` records which, because the same three
    arrays mean different things in each and because the persistence of the
    variance is a different functional of them:

    .. math::

       \rho = \sum_i \alpha_i + \tfrac{1}{2} \sum_k \gamma_k + \sum_j \beta_j
       \quad \text{(GARCH, GJR)},
       \qquad
       \rho = \sum_j \beta_j
       \quad \text{(EGARCH)}.

    The factor one half is the probability of a negative shock under a symmetric
    innovation law. :math:`\rho < 1` is the condition for a finite unconditional
    variance in the level recursion and, at first order, for stationarity of the
    log variance in the log recursion; :meth:`persistence` returns :math:`\rho`.

    Attributes:
        vol: The family that produced the fit, needed because persistence is
            not the same functional of the coefficients across the three.
        alpha: Coefficients on the shock magnitude.
        gamma: Asymmetry coefficients; empty for the symmetric family.
        beta: Persistence coefficients.

    See Also:
        * :meth:`._ShortMemoryVarianceModel._fit_family` -- returns an
          instance.
    """

    vol: str
    """The family that produced the fit, needed because persistence is not the same
    functional of the coefficients across the three.
    """
    alpha: npt.NDArray[np.float64]
    """Coefficients on the shock magnitude."""
    gamma: npt.NDArray[np.float64]
    """Asymmetry coefficients; empty for the symmetric family."""
    beta: npt.NDArray[np.float64]
    """Persistence coefficients."""

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
    r"""Raw outputs of a fractionally integrated variance fit.

    The variance is the truncated ARCH(:math:`\infty`) filter stated on
    :class:`~cultivars.engine._internals._parameters._FractionalVarianceParameters`,
    with weights generated by

    .. math::

       \lambda(L) = 1 - (1 - \beta L)^{-1} (1 - \phi L) (1 - L)^d.

    For :math:`0 < d < 1` the weights decay hyperbolically, so the effect of a
    squared shock on future variance dies out far more slowly than under any
    finite-order recursion without being permanent. :attr:`d` is the long-memory
    parameter of the variance; :attr:`phi` and :attr:`beta` are the short-memory
    weights.

    Attributes:
        phi: Short-memory numerator weight of the fractional polynomial.
        d: Fractional integration order.
        beta: Denominator weight.

    See Also:
        * :meth:`._FractionalVarianceModel._fit_family` -- returns an instance.
    """

    phi: float
    """Short-memory numerator weight of the fractional polynomial."""
    d: float
    """Fractional integration order."""
    beta: float
    """Denominator weight."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _MeanFunctionFit(_BaseFit):
    r"""Raw outputs of a neural mean-function fit.

    The model is a nonlinear autoregression with an additive, homoskedastic
    innovation,

    .. math::

       y_t = f(x_t) + \varepsilon_t, \qquad \varepsilon_t \sim (0, \sigma^2),

    where :math:`x_t` collects the lagged values the specification feeds the
    network and :math:`f` is the fitted mean function. Only :attr:`sigma2` is
    common to every member of the group; how :math:`f` is represented is the
    subclass's business.
    """

    sigma2: float
    """Innovation variance, concentrated out of the Gaussian likelihood."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _NeuralAutoRegressionFit(_MeanFunctionFit):
    r"""Raw outputs of an autoregressive neural mean-function fit.

    One network serves the whole sample: :math:`f` in
    :math:`y_t = f(x_t) + \varepsilon_t` is the single fitted
    :attr:`predictor`, evaluated on the lag vector :math:`x_t`.
    """

    predictor: MeanPredictor
    """The fitted conditional-mean map returned by the engine."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _NeuralThresholdFit(_MeanFunctionFit):
    r"""Raw outputs of a threshold neural mean-function fit.

    Two networks split the sample by an observed threshold variable :math:`z`,

    .. math::

       y_t = \begin{cases}
       f_L(x_t) + \varepsilon_t, & z_{t-d} \le c, \\
       f_U(x_t) + \varepsilon_t, & z_{t-d} > c,
       \end{cases}

    with :attr:`delay` as :math:`d`, :attr:`threshold` as :math:`c`, and
    :attr:`lower_predictor` and :attr:`upper_predictor` as :math:`f_L` and
    :math:`f_U`. When :attr:`self_exciting` is true the threshold variable is the
    series itself, :math:`z_t = y_t`. :attr:`n_lower` and :attr:`n_upper` count
    the observations that fell in each regime, and :attr:`ssr` is the sum of
    squared residuals at the selected threshold.
    """

    delay: int
    """Delay of the threshold variable, in periods."""
    threshold: float
    """The selected threshold."""
    lower_predictor: MeanPredictor
    """Fitted conditional-mean map of the lower regime."""
    upper_predictor: MeanPredictor
    """Fitted conditional-mean map of the upper regime."""
    threshold_variable: npt.NDArray[np.float64] | None
    """External threshold variable aligned with the effective sample, or ``None`` when
    the model is self-exciting.
    """
    self_exciting: bool
    """Whether the threshold variable is a lag of the series itself."""
    ssr: float
    """Sum of squared residuals at the selected threshold."""
    n_lower: int
    """Observations assigned to the lower regime."""
    n_upper: int
    """Observations assigned to the upper regime."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _MarkovSwitchingFit(_BaseFit):
    r"""Raw outputs of a Markov-switching autoregression fit.

    The regime :math:`S_t` is an unobserved first-order Markov chain with
    transition matrix :math:`P_{ij} = \Pr(S_t = j \mid S_{t-1} = i)`, held in
    :attr:`transition`, and every parameter of the autoregression switches with
    it,

    .. math::

       y_t = c_{S_t} + \sum_{i=1}^{p} \phi_{i, S_t}\, y_{t-i} + \sigma_{S_t} \varepsilon_t,
       \qquad \varepsilon_t \sim \mathcal{N}(0, 1),

    with :attr:`intercepts`, :attr:`ar_params` and :attr:`variances` holding
    :math:`c_j`, :math:`\phi_{i, j}` and :math:`\sigma_j^2` for each regime
    :math:`j`. The three probability paths are the standard inferences about the
    chain,

    .. math::

       \Pr(S_t = j \mid y_{1:t-1}), \qquad
       \Pr(S_t = j \mid y_{1:t}), \qquad
       \Pr(S_t = j \mid y_{1:T}),

    in :attr:`predicted_prob`, :attr:`filtered_prob` and :attr:`smoothed_prob`.
    Two summaries of the chain itself are derived from :math:`P`:
    :attr:`ergodic_prob` is its stationary distribution, the solution of
    :math:`\pi = P^\top \pi`, and :attr:`expected_durations` holds
    :math:`1 / (1 - P_{mm})`, the mean number of consecutive periods spent in
    regime :math:`m`. :attr:`n_iter` and :attr:`converged` report how the
    expectation-maximization iteration ended.
    """

    transition: npt.NDArray[np.float64]
    """Row-stochastic transition matrix, ``(K, K)``."""
    intercepts: npt.NDArray[np.float64]
    """Regime intercepts, ``(K,)``."""
    ar_params: npt.NDArray[np.float64]
    """Regime autoregressive coefficients, ``(K, p)``."""
    variances: npt.NDArray[np.float64]
    """Regime innovation variances, ``(K,)``."""
    filtered_prob: npt.NDArray[np.float64]
    """Filtered regime probabilities, ``(T, K)``."""
    predicted_prob: npt.NDArray[np.float64]
    """One-step-predicted regime probabilities, ``(T, K)``."""
    smoothed_prob: npt.NDArray[np.float64]
    """Smoothed regime probabilities, ``(T, K)``."""
    ergodic_prob: npt.NDArray[np.float64]
    """Stationary distribution of the regime chain, ``(K,)``."""
    expected_durations: npt.NDArray[np.float64]
    """Expected duration of each regime in periods, ``(K,)``."""
    n_iter: int
    """EM iterations run by the reported fit."""
    converged: bool
    """Whether the EM run met its convergence tolerance."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _ThresholdFit(_BaseFit):
    r"""Raw outputs of a two-regime threshold grid search.

    The model switches between two linear autoregressions on an observed
    threshold variable,

    .. math::

       y_t = \begin{cases}
       x_t^\top \beta_L + \varepsilon_t, & z_{t-d} \le c, \\
       x_t^\top \beta_U + \varepsilon_t, & z_{t-d} > c,
       \end{cases}

    with :attr:`delay` as :math:`d`, :attr:`threshold` as :math:`c`, and
    :attr:`lower_params` and :attr:`upper_params` as :math:`\beta_L` and
    :math:`\beta_U`. Given :math:`(d, c)` both regimes are linear, so the
    estimator is a grid search: each candidate pair is scored by the pooled sum
    of squared residuals of the two regressions, and the minimizer is kept with
    its :attr:`ssr`. The sum of squares is a step function of :math:`c`, which is
    why the search is over a grid and not by a gradient method, and why the
    threshold estimate has a non-standard limit law.
    """

    delay: int
    """Delay of the threshold variable, in periods."""
    threshold: float
    """The selected threshold."""
    lower_params: npt.NDArray[np.float64]
    """Lower-regime coefficients, the intercept first and then the autoregressive coefficients."""
    upper_params: npt.NDArray[np.float64]
    """Upper-regime coefficients, in the same layout as ``lower_params``."""
    sigma2: float
    """Innovation variance."""
    ssr: float
    """Sum of squared residuals at the selected threshold."""
    n_lower: int
    """Observations assigned to the lower regime."""
    n_upper: int
    """Observations assigned to the upper regime."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _SmoothTransitionFit(_BaseFit):
    r"""Raw outputs of a smooth-transition autoregression fit.

    The model moves between two linear autoregressions continuously,

    .. math::

       y_t = \bigl( 1 - G_t \bigr)\, x_t^\top \beta_L + G_t\, x_t^\top \beta_U
       + \varepsilon_t,
       \qquad G_t = G(z_{t-d};\, \gamma, c) \in [0, 1],

    with :attr:`delay` as :math:`d`, :attr:`threshold` as :math:`c`,
    :attr:`gamma` as the transition speed :math:`\gamma`, and
    :attr:`lower_params` and :attr:`upper_params` as :math:`\beta_L` and
    :math:`\beta_U`. The transition function :math:`G` is stated on
    :class:`~cultivars.engine._internals._parameters._SmoothTransitionParameters`.
    Given :math:`(\gamma, c)` the model is linear in the regime coefficients, so
    those are concentrated out by least squares and only the two transition
    parameters are searched numerically.
    """

    delay: int
    """Delay of the transition variable, in periods."""
    threshold: float
    """Transition location, on the scale of the transition variable."""
    gamma: float
    """Transition speed, strictly positive."""
    lower_params: npt.NDArray[np.float64]
    """Coefficients of the lower regime, the intercept first."""
    upper_params: npt.NDArray[np.float64]
    """Coefficients of the upper regime, in the same layout as ``lower_params``."""
    sigma2: float
    """Innovation variance."""
    ssr: float
    """Sum of squared residuals at the optimum."""


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
        state_cov: ``(D, D)`` posterior mean of the drift covariance :math:`Q`.
        sigma_u: ``(k, k)`` posterior mean innovation covariance -- constant
            for the homoskedastic model, the time average under stochastic
            volatility.
        impact_draws: ``(S, k, k)`` draws of the inverse contemporaneous
            matrix :math:`A^{-1}`, or ``None`` for the homoskedastic model.
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

    See Also:
        * :meth:`._TimeVaryingVectorAutoRegressionModel._fit_tvp` -- returns an
          instance.
    """

    beta_mean: npt.NDArray[np.float64]
    """``(n, D)`` posterior mean coefficient path."""
    beta_low: npt.NDArray[np.float64]
    """``(n, D)`` posterior 16th percentile path."""
    beta_high: npt.NDArray[np.float64]
    """``(n, D)`` posterior 84th percentile path."""
    beta_draws: npt.NDArray[np.float64]
    """``(S, n, D)`` kept coefficient-path draws."""
    state_cov: npt.NDArray[np.float64]
    """``(D, D)`` posterior mean of the drift covariance :math:`Q`."""
    sigma_u: npt.NDArray[np.float64]
    """``(k, k)`` posterior mean innovation covariance -- constant for the
    homoskedastic model, the time average under stochastic volatility.
    """
    impact_draws: npt.NDArray[np.float64] | None
    """``(S, k, k)`` draws of the inverse contemporaneous matrix :math:`A^{-1}`, or
    ``None`` for the homoskedastic model.
    """
    h_draws: npt.NDArray[np.float64] | None
    """``(S, n, k)`` kept log-variance path draws, or ``None``."""
    vol_of_vol: npt.NDArray[np.float64] | None
    """``(k,)`` posterior mean random-walk variances of the log volatilities, or ``None``."""
    resid: npt.NDArray[np.float64]
    """Residuals at the posterior mean path."""
    fittedvalues: npt.NDArray[np.float64]
    """One-step means at the posterior mean path."""
    nobs: int
    """Estimation sample after the training split."""
    training: int
    """Rows consumed by the training prior."""
    n_draws: int
    """Total sampler iterations."""
    n_burn: int
    """Burn-in iterations discarded."""
    thin: int
    """Keep-every-``thin`` thinning applied after burn-in."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorMarkovSwitchingFit(_BaseFit):
    r"""Raw outputs of a Markov-switching vector autoregression fit.

    Attributes:
        transition: Row-stochastic ``(M, M)`` matrix.
        coefficients: ``(M, p, k, k)`` per-regime lag stacks.
        deterministics: ``(M, d, k)`` per-regime deterministic coefficients.
        sigmas: ``(M, k, k)`` per-regime innovation covariances.
        filtered_prob: :math:`\Pr(S_t = m \mid y_{1:t})`, shape ``(nobs, M)``.
        predicted_prob: :math:`\Pr(S_t = m \mid y_{1:t-1})`, shape ``(nobs, M)``.
        smoothed_prob: :math:`\Pr(S_t = m \mid y_{1:T})`, shape ``(nobs, M)``.
        ergodic_prob: Stationary distribution of ``transition``.
        expected_durations: :math:`1 / (1 - P_{mm})`, in periods.
        n_iter: EM iterations used by the refining run.
        converged: Whether the refining run met its tolerance.

    See Also:
        * :meth:`._MarkovSwitchingVectorAutoRegressionModel._fit_markov` --
          returns an instance.
    """

    transition: npt.NDArray[np.float64]
    """Row-stochastic ``(M, M)`` matrix."""
    coefficients: npt.NDArray[np.float64]
    """``(M, p, k, k)`` per-regime lag stacks."""
    deterministics: npt.NDArray[np.float64]
    """``(M, d, k)`` per-regime deterministic coefficients."""
    sigmas: npt.NDArray[np.float64]
    """``(M, k, k)`` per-regime innovation covariances."""
    filtered_prob: npt.NDArray[np.float64]
    r""":math:`\Pr(S_t = m \mid y_{1:t})`, shape ``(nobs, M)``."""
    predicted_prob: npt.NDArray[np.float64]
    r""":math:`\Pr(S_t = m \mid y_{1:t-1})`, shape ``(nobs, M)``."""
    smoothed_prob: npt.NDArray[np.float64]
    r""":math:`\Pr(S_t = m \mid y_{1:T})`, shape ``(nobs, M)``."""
    ergodic_prob: npt.NDArray[np.float64]
    """Stationary distribution of ``transition``."""
    expected_durations: npt.NDArray[np.float64]
    """:math:`1 / (1 - P_{mm})`, in periods."""
    n_iter: int
    """EM iterations used by the refining run."""
    converged: bool
    """Whether the refining run met its tolerance."""


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
    """Delay of the transition variable."""
    threshold: float
    """The regime split point, in the transition variable's units."""
    threshold_values: npt.NDArray[np.float64]
    """The transition variable, aligned with ``resid``."""
    lower_coefficients: npt.NDArray[np.float64]
    """``(p, k, k)`` lag stack of the lower regime."""
    upper_coefficients: npt.NDArray[np.float64]
    """``(p, k, k)`` lag stack of the upper regime."""
    lower_deterministic: npt.NDArray[np.float64]
    """Lower-regime deterministic coefficients."""
    upper_deterministic: npt.NDArray[np.float64]
    """Upper-regime deterministic coefficients."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorThresholdFit(_VectorObservedRegimeFit):
    """Raw outputs of a two-regime threshold VAR grid search.

    Attributes:
        lower_sigma_u: Lower-regime innovation covariance, dof-corrected.
        upper_sigma_u: Upper-regime innovation covariance, dof-corrected.
        n_lower: Observations assigned to the lower regime.
        n_upper: Observations assigned to the upper regime.

    See Also:
        * :meth:`._ThresholdVectorAutoRegressionModel._fit_regimes` -- returns
          an instance.
    """

    lower_sigma_u: npt.NDArray[np.float64]
    """Lower-regime innovation covariance, dof-corrected."""
    upper_sigma_u: npt.NDArray[np.float64]
    """Upper-regime innovation covariance, dof-corrected."""
    n_lower: int
    """Observations assigned to the lower regime."""
    n_upper: int
    """Observations assigned to the upper regime."""


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

    See Also:
        * :meth:`._SmoothTransitionVectorAutoRegressionModel._fit_regimes` --
          returns an instance.
    """

    gamma: float
    """Transition speed, per standard deviation of the transition variable."""
    transition_scale: float
    """The standard deviation ``gamma`` is expressed against."""
    sigma_u: npt.NDArray[np.float64]
    """Common innovation covariance, dof-corrected.

    One covariance rather than two, because smooth weights never partition the
    sample: every observation is a blend of both regimes, so a per-regime
    covariance has no subsample to be estimated from.
    """


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorAutoRegressionFit(_BaseFit):
    r"""The estimated pieces of a reduced-form vector autoregression.

    In stacked form the system is :math:`Y = Z B + U`, with one row per date, the
    deterministic terms and the :math:`p` lags of all :math:`k` variables in
    :math:`Z`, and rows of :math:`U` with covariance :math:`\Sigma_u`. The record
    holds the least-squares solution split into its two parts,
    :attr:`deterministic` and the lag stack :attr:`coefficients`, together with
    two estimates of the innovation covariance from the same residuals,

    .. math::

       \hat\Sigma_u = \frac{\hat{U}^\top \hat{U}}{T - w}, \qquad
       \tilde\Sigma_u = \frac{\hat{U}^\top \hat{U}}{T},

    in :attr:`sigma_u` and :attr:`sigma_ml`, where :math:`w` is the number of
    regressors per equation, or the effective number spent when a prior or
    penalty shrinks the fit. The first carries the degrees-of-freedom
    correction; the second is the maximum-likelihood estimate, the one the
    Gaussian log likelihood is evaluated at.

    Attributes:
        coefficients: ``(p, k, k)`` stack of ``A_1, ..., A_p``.
        deterministic: Deterministic coefficients, one row per term.
        sigma_u: Residual covariance with the degrees-of-freedom correction.
        sigma_ml: Residual covariance divided by the effective sample.
        design: The regressor matrix as estimated.
        posterior: Posterior covariance when a prior was applied, else ``None``.
        prior_label: What the prior was, for the summary.

    See Also:
        * :meth:`._BayesianVectorAutoRegressionModel._fit_family` -- returns an
          instance.
        * :meth:`._FunctionalCoefficientVectorAutoRegressionModel._fit_family`
          -- returns an instance.
        * :meth:`._GibbsBayesianVectorAutoRegressionModel._fit_family` --
          returns an instance.
        * :meth:`._MarkovSwitchingVectorAutoRegressionModel._fit_family` --
          returns an instance.
    """

    coefficients: npt.NDArray[np.float64]
    """``(p, k, k)`` stack of ``A_1, ..., A_p``."""
    deterministic: npt.NDArray[np.float64]
    """Deterministic coefficients, one row per term."""
    sigma_u: npt.NDArray[np.float64]
    """Residual covariance with the degrees-of-freedom correction."""
    sigma_ml: npt.NDArray[np.float64]
    """Residual covariance divided by the effective sample."""
    design: npt.NDArray[np.float64]
    """The regressor matrix as estimated."""
    posterior: _CoefficientInference | None = None
    """Posterior covariance when a prior was applied, else ``None``."""
    prior_label: str = "none"
    """What the prior was, for the summary."""

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
            :math:`x_{t-j}` and index ``0`` is the contemporaneous term.

    See Also:
        * :meth:`._ExogenousVectorAutoRegressionModel._fit_family` -- returns
          an instance.
    """

    exog_coefficients: npt.NDArray[np.float64]
    """``(s + 1, k, m)`` stack of ``B_0, ..., B_s``, laid out so that
    ``exog_coefficients[j]`` is the ``(k, m)`` matrix on :math:`x_{t-j}` and index
    ``0`` is the contemporaneous term.
    """

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

    See Also:
        * :meth:`._ExogenousVectorErrorCorrectionModel._fit_family` -- returns
          an instance.
        * :meth:`._VectorErrorCorrectionModel._fit_family` -- returns an
          instance.
    """

    alpha: npt.NDArray[np.float64]
    """``(k_y, r)`` adjustment loadings."""
    beta: npt.NDArray[np.float64]
    """``(k_y + k_x [+ 1], r)`` cointegrating vectors, the extra row present when the
    case restricts a deterministic term to the cointegrating space.
    """
    gamma: npt.NDArray[np.float64]
    """``(p - 1, k_y, k_y + k_x)`` short-run coefficients on lagged differences of
    every integrated variable.
    """
    short_run_deterministic: npt.NDArray[np.float64]
    """``(d_s, k_y)`` unrestricted deterministic terms, as they entered the regression."""
    impact: npt.NDArray[np.float64]
    """``(k_y, k_x)`` contemporaneous response to the weakly exogenous differences,
    zero-width for a closed system.
    """
    eigenvalues: npt.NDArray[np.float64]
    """The squared canonical correlations, descending."""

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
        r"""The long-run impact matrix :math:`\Pi = \alpha \beta^\top`, over the variables only."""
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
            the intercept-only benchmark the pseudo-:math:`R`:sup:`1` is read
            against (Koenker & Machado 1999).
        nobs: Effective sample size.

    See Also:
        * :meth:`._QuantileVectorAutoRegressionModel._fit_quantile` -- returns
          an instance.
    """

    quantiles: tuple[float, ...]
    """The estimated quantile levels, ascending."""
    coefficient_stacks: npt.NDArray[np.float64]
    """``(Q, p, k, k)`` lag stacks, one per quantile."""
    deterministics: npt.NDArray[np.float64]
    """``(Q, n_det, k)`` deterministic coefficients."""
    fittedvalues: npt.NDArray[np.float64]
    """``(Q, n, k)`` conditional quantile paths."""
    resid: npt.NDArray[np.float64]
    """``(Q, n, k)`` quantile residuals ``y - fitted``."""
    loss: npt.NDArray[np.float64]
    """``(Q, k)`` total check loss per equation at the optimum."""
    loss_location: npt.NDArray[np.float64]
    """``(Q, k)`` check loss of the unconditional quantile, the intercept-only
    benchmark the pseudo-:math:`R`:sup:`1` is read against (Koenker & Machado 1999).
    """
    nobs: int
    """Effective sample size."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorFunctionalFit:
    """Raw outputs of a functional-coefficient VAR fit.

    Not a :class:`_BaseFit`: a kernel estimator maximizes no likelihood and
    has no integer parameter count -- its complexity is the trace of the
    smoother matrix, carried here as ``effective_params``.

    Attributes:
        delay: Delay of the state variable.
        state_values: The delayed state :math:`z_t`, aligned with ``resid``.
        grid: ``(G,)`` state values the coefficient curves are evaluated on.
        curves: ``(G, w, k)`` local-linear level coefficients, design-major:
            ``curves[g, :, i]`` is equation :math:`i`'s coefficient vector at
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

    See Also:
        * :meth:`._FunctionalCoefficientVectorAutoRegressionModel._fit_functional`
          -- returns an instance.
    """

    delay: int
    """Delay of the state variable."""
    state_values: npt.NDArray[np.float64]
    """The delayed state :math:`z_t`, aligned with ``resid``."""
    grid: npt.NDArray[np.float64]
    """``(G,)`` state values the coefficient curves are evaluated on."""
    curves: npt.NDArray[np.float64]
    """``(G, w, k)`` local-linear level coefficients, design-major: ``curves[g, :,
    i]`` is equation ``i``'s coefficient vector at ``grid[g]``.
    """
    curve_se: npt.NDArray[np.float64]
    """``(G, w, k)`` pointwise standard errors of ``curves``."""
    bandwidth: float
    """The bandwidth the curves were estimated at."""
    bandwidth_searched: bool
    """Whether the bandwidth came from cross-validation."""
    effective_params: float
    """Trace of the smoother matrix at that bandwidth."""
    sigma_u: npt.NDArray[np.float64]
    """``(k, k)`` innovation covariance, corrected by the effective degrees of freedom."""
    resid: npt.NDArray[np.float64]
    """Residuals of the local fits at the observed states."""
    fittedvalues: npt.NDArray[np.float64]
    """One-step conditional means at the observed states."""
    nobs: int
    """Effective sample size."""


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

    See Also:
        * :meth:`._BayesianVectorAutoRegressionModel._fit_conjugate` -- returns
          an instance.
        * :meth:`._StudentBayesianVectorAutoRegressionModel._fit_conjugate` --
          returns an instance.
    """

    coefficient_stack: npt.NDArray[np.float64]
    """``(p, k, k)`` lag stack at the posterior mean."""
    deterministic: npt.NDArray[np.float64]
    """``(n_det, k)`` deterministic block at the posterior mean."""
    beta_mean: npt.NDArray[np.float64]
    """``(w, k)`` full posterior mean coefficient matrix."""
    sigma_u: npt.NDArray[np.float64]
    """``(k, k)`` posterior mean innovation covariance."""
    beta_draws: npt.NDArray[np.float64]
    """``(S, w, k)`` coefficient draws."""
    sigma_draws: npt.NDArray[np.float64]
    """``(S, k, k)`` covariance draws."""
    log_marginal_likelihood: float
    """Log marginal likelihood of the *sample*, with the dummy-observation
    contribution divided out.
    """
    posterior_df: float
    """Inverse-Wishart posterior degrees of freedom."""
    resid: npt.NDArray[np.float64]
    """Residuals at the posterior mean, over the sample rows only."""
    fittedvalues: npt.NDArray[np.float64]
    """One-step means at the posterior mean."""
    nobs: int
    """Effective sample size, dummy rows excluded."""
    n_dummy: int
    """Artificial rows the prior contributed."""


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

    See Also:
        * :meth:`._BayesianVectorAutoRegressionModel._fit_hierarchical` --
          returns an instance.
        * :meth:`._StudentBayesianVectorAutoRegressionModel._fit_hierarchical`
          -- returns an instance.
    """

    hyper_names: tuple[str, ...]
    """One label per hyperparameter, in draw-column order."""
    hyper_mode: npt.NDArray[np.float64]
    """Posterior-mode hyperparameter vector."""
    hyper_draws: npt.NDArray[np.float64]
    """``(S, d)`` kept hyperparameter draws -- empty under empirical Bayes."""
    acceptance: float
    """Metropolis acceptance rate over the kept span; ``nan`` under empirical Bayes."""
    method: str
    """``"full"`` or ``"empirical"``."""


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

    See Also:
        * :meth:`._StudentBayesianVectorAutoRegressionModel._fit_student` --
          returns an instance.
    """

    coefficient_stack: npt.NDArray[np.float64]
    """``(p, k, k)`` lag stack at the posterior mean."""
    deterministic: npt.NDArray[np.float64]
    """``(n_det, k)`` deterministic block at the posterior mean."""
    beta_mean: npt.NDArray[np.float64]
    """``(w, k)`` posterior mean coefficient matrix."""
    sigma_u: npt.NDArray[np.float64]
    """``(k, k)`` posterior mean *scale* matrix -- the innovation covariance is
    ``sigma_u * df / (df - 2)``.
    """
    beta_draws: npt.NDArray[np.float64]
    """``(S, w, k)`` kept coefficient draws."""
    sigma_draws: npt.NDArray[np.float64]
    """``(S, k, k)`` kept scale draws."""
    df: float
    """Posterior mean degrees of freedom (the stated value when fixed)."""
    df_draws: npt.NDArray[np.float64]
    """``(S,)`` kept degrees-of-freedom draws; empty when fixed."""
    weight_mean: npt.NDArray[np.float64]
    """``(n,)`` posterior mean latent precision weights -- small values mark the dates
    the t distribution treats as outliers.
    """
    resid: npt.NDArray[np.float64]
    """Residuals at the posterior mean."""
    fittedvalues: npt.NDArray[np.float64]
    """One-step means at the posterior mean."""
    nobs: int
    """Effective sample size."""
    n_dummy: int
    """Artificial rows the prior contributed."""
    n_draws: int
    """Total sampler iterations."""
    n_burn: int
    """Burn-in discarded."""
    thin: int
    """Post-burn thinning."""


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
        impact_draws: ``(S, k, k)`` kept draws of :math:`A^{-1}`.
        vol_of_vol: ``(k,)`` posterior mean random-walk variances of the
            log volatilities.
        resid: Residuals at the posterior mean.
        fittedvalues: One-step means at the posterior mean.
        nobs: Effective sample size.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.

    See Also:
        * :meth:`._VolatilityBayesianVectorAutoRegressionModel._fit_volatility`
          -- returns an instance.
    """

    coefficient_stack: npt.NDArray[np.float64]
    """``(p, k, k)`` lag stack at the posterior mean."""
    deterministic: npt.NDArray[np.float64]
    """``(n_det, k)`` deterministic block at the posterior mean."""
    beta_mean: npt.NDArray[np.float64]
    """``(w, k)`` posterior mean coefficient matrix."""
    sigma_u: npt.NDArray[np.float64]
    """``(k, k)`` posterior mean *end-of-sample* covariance."""
    beta_draws: npt.NDArray[np.float64]
    """``(S, w, k)`` kept coefficient draws."""
    sigma_draws: npt.NDArray[np.float64]
    """``(S, k, k)`` kept end-of-sample covariance draws."""
    h_draws: npt.NDArray[np.float64]
    """``(S, n, k)`` kept log-variance path draws."""
    impact_draws: npt.NDArray[np.float64]
    """``(S, k, k)`` kept draws of :math:`A^{-1}`."""
    vol_of_vol: npt.NDArray[np.float64]
    """``(k,)`` posterior mean random-walk variances of the log volatilities."""
    resid: npt.NDArray[np.float64]
    """Residuals at the posterior mean."""
    fittedvalues: npt.NDArray[np.float64]
    """One-step means at the posterior mean."""
    nobs: int
    """Effective sample size."""
    n_draws: int
    """Total sampler iterations."""
    n_burn: int
    """Burn-in discarded."""
    thin: int
    """Post-burn thinning."""


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

    See Also:
        * :meth:`._SparseVectorAutoRegressionModel._fit_sparse` -- returns an
          instance.
    """

    coefficient_stack: npt.NDArray[np.float64]
    """``(p, k, k)`` lag stack at the selected penalty."""
    deterministic: npt.NDArray[np.float64]
    """``(n_det, k)`` unpenalized deterministic block."""
    sigma_u: npt.NDArray[np.float64]
    """``(k, k)`` residual covariance, corrected by the average per-equation nonzero count."""
    resid: npt.NDArray[np.float64]
    """Residuals over the effective sample."""
    fittedvalues: npt.NDArray[np.float64]
    """One-step means over the effective sample."""
    penalty: str
    """The penalty family estimated under."""
    lam: float
    """The penalty level estimated at."""
    lambda_path: npt.NDArray[np.float64]
    """The candidate path, descending; empty when the caller stated ``lam``."""
    cv_errors: npt.NDArray[np.float64]
    """Rolling one-step squared forecast error summed over the validation span,
    aligned with ``lambda_path``; empty when the caller stated ``lam``.
    """
    n_nonzero: int
    """Nonzero penalized coefficients at the solution."""
    nobs: int
    """Effective sample size."""


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

    See Also:
        * :meth:`._SparseVectorAutoRegressionModel._fit_graphical` -- returns
          an instance.
    """

    precision: npt.NDArray[np.float64]
    """``(k, k)`` sparse residual precision from symmetrized nodewise regressions."""
    partial_correlations: npt.NDArray[np.float64]
    """``(k, k)`` partial correlations implied by the precision, unit diagonal."""
    lam_nodes: npt.NDArray[np.float64]
    """``(k,)`` the per-node penalty levels cross-validation selected."""


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

    See Also:
        * :meth:`._GibbsBayesianVectorAutoRegressionModel._fit_gibbs` --
          returns an instance.
    """

    coefficient_stack: npt.NDArray[np.float64]
    """``(p, k, k)`` lag stack at the posterior mean."""
    deterministic: npt.NDArray[np.float64]
    """``(n_det, k)`` deterministic block at the posterior mean."""
    beta_mean: npt.NDArray[np.float64]
    """``(w, k)`` posterior mean coefficient matrix."""
    sigma_u: npt.NDArray[np.float64]
    """``(k, k)`` posterior mean innovation covariance."""
    beta_draws: npt.NDArray[np.float64]
    """``(S, w, k)`` kept coefficient draws."""
    sigma_draws: npt.NDArray[np.float64]
    """``(S, k, k)`` kept covariance draws."""
    shrinkage: npt.NDArray[np.float64]
    """``(w, k)`` posterior mean of the adaptive prior's per-coefficient diagnostic --
    inclusion probabilities for a selection prior, local-global scales for a
    global-local one -- empty for a static prior.
    """
    shrinkage_label: str
    """What ``shrinkage`` is; empty for a static prior."""
    resid: npt.NDArray[np.float64]
    """Residuals at the posterior mean, over the sample rows only."""
    fittedvalues: npt.NDArray[np.float64]
    """One-step means at the posterior mean."""
    nobs: int
    """Effective sample size, dummy rows excluded."""
    n_dummy: int
    """Artificial rows the prior contributed."""
    n_draws: int
    """Total sampler iterations."""
    n_burn: int
    """Burn-in discarded."""
    thin: int
    """Post-burn thinning."""


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

    See Also:
        * :meth:`._UnobservedComponentsModel._fit_structural` -- returns an
          instance.
    """

    params: _StructuralParameters
    """The maximized parameter record."""
    llf: float
    """Exact Gaussian log-likelihood under the approximate-diffuse initialization the
    system builder documents.
    """
    n_params: int
    """Free parameters the likelihood was maximized over."""
    nobs: int
    """Observations."""
    smoothed_state: npt.NDArray[np.float64]
    """Full-sample state means, ``(n, m)``."""
    smoothed_state_cov: npt.NDArray[np.float64]
    """Full-sample state covariances, ``(n, m, m)``."""
    slices: dict[str, slice]
    """Component name to state index range."""


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
    """The maximized parameter record."""
    llf: float
    """Exact Gaussian log-likelihood."""
    n_params: int
    """Free parameters the likelihood was maximized over."""
    nobs: int
    """Observations (curve dates)."""
    factors: npt.NDArray[np.float64]
    """Smoothed factor paths, ``(n, 3)``."""
    factor_cov: npt.NDArray[np.float64]
    """Smoothed factor covariances, ``(n, 3, 3)``."""


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

    See Also:
        * :meth:`._StochasticVolatilityModel._fit_particle` -- returns an
          instance.
        * :meth:`._StochasticVolatilityModel._fit_quasi` -- returns an
          instance.
    """

    params: _StochasticVolatilityParameters
    """The maximized parameter record."""
    llf: float
    """The criterion at the optimum: the exact quasi-likelihood of the linearized
    model under ``method="qml"``, or a particle *estimate* of the exact likelihood
    under ``method="particle"``.
    """
    method: str
    """``"qml"`` or ``"particle"``."""
    n_params: int
    """Free parameters the criterion was maximized over."""
    nobs: int
    """Observations."""
    log_variance: npt.NDArray[np.float64]
    """Smoothed log-variance path, ``(n,)``."""
    log_variance_std: npt.NDArray[np.float64]
    """Smoothed log-variance standard deviations, ``(n,)``."""


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

    See Also:
        * :meth:`._StochasticVolatilityModel._sample_gibbs` -- returns an
          instance.
    """

    mu_draws: npt.NDArray[np.float64]
    """``(S,)`` kept draws of the log-variance mean."""
    phi_draws: npt.NDArray[np.float64]
    """``(S,)`` kept draws of the persistence."""
    sigma2_draws: npt.NDArray[np.float64]
    """``(S,)`` kept draws of the log-variance innovation variance."""
    mean_draws: npt.NDArray[np.float64]
    """``(S,)`` kept draws of the observation mean (all zero when the mean is fixed)."""
    h_draws: npt.NDArray[np.float64]
    """``(S, n)`` kept log-variance paths."""
    nu_draws: npt.NDArray[np.float64] | None
    """``(S,)`` kept draws of the Student-t degrees of freedom, or ``None`` under Gaussian noise."""
    nu_acceptance: float | None
    """Acceptance rate of the Metropolis step for the degrees of freedom, or ``None``
    under Gaussian noise.
    """
    nobs: int
    """Observations."""
    n_draws: int
    """Total sampler iterations."""
    n_burn: int
    """Burn-in discarded."""
    thin: int
    """Post-burn thinning."""

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

    See Also:
        * :meth:`._ParticleMarginalChain.run` -- returns an instance.
        * :meth:`._PerturbationModel._sample_chain` -- returns an instance.
    """

    theta_draws: npt.NDArray[np.float64]
    """``(S, d)`` kept draws in the *unconstrained* space the chain moved in."""
    loglik_draws: npt.NDArray[np.float64]
    """``(S,)`` particle log-likelihood estimates carried with each kept draw."""
    log_posterior_draws: npt.NDArray[np.float64]
    """``(S,)`` the log-prior plus log-likelihood estimate at each kept draw."""
    acceptance_rate: float
    """Fraction of proposals accepted over the whole run, burn-in included."""
    proposal_cov: npt.NDArray[np.float64]
    """The ``(d, d)`` proposal covariance the chain ended with, after adaptation."""
    n_particles: int
    """Particles per likelihood estimate."""
    n_draws: int
    """Total iterations."""
    n_burn: int
    """Burn-in discarded."""
    thin: int
    """Post-burn thinning."""

    @property
    def n_kept(self) -> int:
        """Kept draws."""
        return int(self.theta_draws.shape[0])


@dataclass(frozen=True, kw_only=True, slots=True)
class _TrendVolatilityFit:
    r"""Raw output of the unobserved-components stochastic-volatility Gibbs sampler.

    Attributes:
        trend_draws: ``(S, n)`` kept trend paths :math:`\tau_t`.
        h_draws: ``(S, n)`` kept log variances of the irregular.
        q_draws: ``(S, n)`` kept log variances of the trend innovation.
        gamma2_irregular_draws: ``(S,)`` vol-of-vol draws of :math:`h` (constant
            across draws when fixed).
        gamma2_trend_draws: ``(S,)`` vol-of-vol draws of ``q``.
        gamma_fixed: Whether the vol-of-vol parameters were held fixed.
        nobs: Observations.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.

    See Also:
        * :meth:`._TrendVolatilityModel._sample_gibbs` -- returns an instance.
    """

    trend_draws: npt.NDArray[np.float64]
    r"""``(S, n)`` kept trend paths :math:`\tau_t`."""
    h_draws: npt.NDArray[np.float64]
    """``(S, n)`` kept log variances of the irregular."""
    q_draws: npt.NDArray[np.float64]
    """``(S, n)`` kept log variances of the trend innovation."""
    gamma2_irregular_draws: npt.NDArray[np.float64]
    """``(S,)`` vol-of-vol draws of :math:`h` (constant across draws when fixed)."""
    gamma2_trend_draws: npt.NDArray[np.float64]
    """``(S,)`` vol-of-vol draws of ``q``."""
    gamma_fixed: bool
    """Whether the vol-of-vol parameters were held fixed."""
    nobs: int
    """Observations."""
    n_draws: int
    """Total sampler iterations."""
    n_burn: int
    """Burn-in discarded."""
    thin: int
    """Post-burn thinning."""

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

    See Also:
        * :meth:`._DecayNelsonSiegelModel._fit_decay` -- returns an instance.
    """

    params: _DecayNelsonSiegelParameters
    """The maximized parameter record."""
    llf: float
    """The Gaussian-approximation log-likelihood of the chosen filter (extended or
    unscented); *not* an exact likelihood.
    """
    filter: str
    """``"extended"`` or ``"unscented"``."""
    n_params: int
    """Free parameters the likelihood was maximized over."""
    nobs: int
    """Curve dates."""
    factors: npt.NDArray[np.float64]
    """Smoothed factor paths, ``(n, 3)``."""
    factor_cov: npt.NDArray[np.float64]
    """Smoothed factor covariances, ``(n, 3, 3)``."""
    log_decay: npt.NDArray[np.float64]
    """Smoothed log-decay path, ``(n,)``."""
    log_decay_std: npt.NDArray[np.float64]
    """Smoothed log-decay standard deviations, ``(n,)``."""


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

    See Also:
        * :meth:`._PerturbationModel._fit_point` -- returns an instance.
    """

    theta: npt.NDArray[np.float64]
    """The structural parameter vector at the optimum, in the model's own (constrained) space."""
    solution: _PerturbationSolution
    """The perturbation solution at ``theta``."""
    llf: float
    """The criterion at the optimum: exact under ``filter="kalman"`` (first order
    only), a Gaussian approximation under the extended/unscented filters, a
    particle estimate under ``filter="particle"``.
    """
    filter: str
    """Which filter produced ``llf``."""
    n_params: int
    """Free structural parameters."""
    nobs: int
    """Observations."""
    filtered_state: npt.NDArray[np.float64]
    """Filtered state means (pruned state, ``(n, 2 n_x)`` at second order, ``(n,
    n_x)`` at first).
    """


@dataclass(frozen=True, kw_only=True, slots=True)
class _VolatilityStructuralFit:
    r"""Raw output of the stochastic-volatility identification sampler.

    Every draw is already normalized to one labelling: columns of the
    impact matrix are matched to a reference by assignment on their
    absolute cosine similarity and signed to agree with it, so the draws
    below describe one posterior mode rather than a mixture of :math:`2^k\, k!`
    relabellings.

    Attributes:
        impact_draws: ``(S, k, k)`` kept impact matrices :math:`B = A^{-1}`.
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

    See Also:
        * :meth:`._VolatilityIdentificationModel._sample` -- returns an
          instance.
    """

    impact_draws: npt.NDArray[np.float64]
    """``(S, k, k)`` kept impact matrices :math:`B = A^{-1}`."""
    h_draws: npt.NDArray[np.float64]
    """``(S, T, k)`` kept structural log-variance paths, one column per shock, each
    with zero sample mean; the level sits in the matching column of
    ``impact_draws``.
    """
    phi_draws: npt.NDArray[np.float64]
    """``(S, k)`` kept log-variance persistences."""
    sigma2_draws: npt.NDArray[np.float64]
    """``(S, k)`` kept log-variance innovation variances."""
    relabel_rate: float
    """Fraction of kept draws whose column order had to be permuted to match the
    reference -- near zero when the shocks are well separated, and the first thing
    to read when they are not.
    """
    nobs: int
    """Observations."""
    n_draws: int
    """Total sampler iterations."""
    n_burn: int
    """Burn-in discarded."""
    thin: int
    """Post-burn thinning."""


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

    See Also:
        * :meth:`._LongMemoryVolatilityModel._fit_whittle` -- returns an
          instance.
    """

    params: _LongMemoryVolatilityParameters
    """The maximized parameter record."""
    llf: float
    """The Whittle criterion at the optimum, sign-flipped to read as a log-likelihood:
    the frequency-domain quasi-likelihood of the linearized model, comparable only
    with other Whittle fits.
    """
    n_params: int
    """Free parameters the criterion was maximized over."""
    nobs: int
    """Observations."""
    n_frequencies: int
    """Fourier ordinates the criterion summed over."""
    log_variance: npt.NDArray[np.float64]
    """Smoothed log-variance path, ``(n,)``, from the spectral (Wiener-Kolmogorov)
    smoother under a circular approximation.
    """
    log_variance_std: npt.NDArray[np.float64]
    """The smoother's stationary root mean squared error, one value repeated ``(n,)`` times."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _FactorVolatilityFit:
    """Raw output of the factor stochastic-volatility Gibbs sampler.

    Attributes:
        loading_draws: ``(S, k, r)`` kept loading matrices, lower triangular
            with unit diagonal in the leading :math:`r` rows.
        factor_draws: ``(S, T, r)`` kept factor paths.
        h_factor_draws: ``(S, T, r)`` kept factor log-variance paths.
        h_idio_draws: ``(S, T, k)`` kept idiosyncratic log-variance paths.
        mu_factor_draws: ``(S, r)`` factor log-variance means.
        phi_factor_draws: ``(S, r)`` factor log-variance persistences.
        sigma2_factor_draws: ``(S, r)`` factor log-variance innovation
            variances.
        mu_idio_draws: ``(S, k)`` idiosyncratic log-variance means.
        phi_idio_draws: ``(S, k)`` idiosyncratic persistences.
        sigma2_idio_draws: ``(S, k)`` idiosyncratic innovation variances.
        means: ``(k,)`` series means removed before sampling.
        nobs: Observations.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.

    See Also:
        * :meth:`._FactorVolatilityModel._sample` -- returns an instance.
    """

    loading_draws: npt.NDArray[np.float64]
    """``(S, k, r)`` kept loading matrices, lower triangular with unit diagonal in the
    leading :math:`r` rows.
    """
    factor_draws: npt.NDArray[np.float64]
    """``(S, T, r)`` kept factor paths."""
    h_factor_draws: npt.NDArray[np.float64]
    """``(S, T, r)`` kept factor log-variance paths."""
    h_idio_draws: npt.NDArray[np.float64]
    """``(S, T, k)`` kept idiosyncratic log-variance paths."""
    mu_factor_draws: npt.NDArray[np.float64]
    """``(S, r)`` factor log-variance means."""
    phi_factor_draws: npt.NDArray[np.float64]
    """``(S, r)`` factor log-variance persistences."""
    sigma2_factor_draws: npt.NDArray[np.float64]
    """``(S, r)`` factor log-variance innovation variances."""
    mu_idio_draws: npt.NDArray[np.float64]
    """``(S, k)`` idiosyncratic log-variance means."""
    phi_idio_draws: npt.NDArray[np.float64]
    """``(S, k)`` idiosyncratic persistences."""
    sigma2_idio_draws: npt.NDArray[np.float64]
    """``(S, k)`` idiosyncratic innovation variances."""
    means: npt.NDArray[np.float64]
    """``(k,)`` series means removed before sampling."""
    nobs: int
    """Observations."""
    n_draws: int
    """Total sampler iterations."""
    n_burn: int
    """Burn-in discarded."""
    thin: int
    """Post-burn thinning."""
