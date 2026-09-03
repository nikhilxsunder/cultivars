# filepath: /src/cultivars/_internals/_results.py
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

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core import (
    _CAPACITY_WARNING,
    _SCHEMA_VERSION,
    CointegrationTrend,
    InformationCriteria,
    Regime,
    companion_matrix,
    deterministic_columns,
    validate_choice,
)
from ..exceptions import SpecificationError
from ._inferences import _CoefficientInference
from ._mixins import (
    _ComparisonMixin,
    _ConditionalVarianceMixin,
    _SeriesMixin,
    _SummaryMixin,
)
from ._tests import _LikelihoodRatioTest, _StabilityTest


@dataclass(frozen=True, kw_only=True, slots=True)
class _FittedResult:
    """Root of every fitted-result object.

    Carries the likelihood summary every estimator produces and derives the
    information criteria from it, so no subclass stores ``aic``/``bic``/``hqic``
    as fields that could drift out of sync with ``llf``.

    Attributes:
        llf: Maximized log-likelihood.
        nobs: Observations the likelihood was evaluated on.
        n_params: Free parameters, including the innovation variance. Not
            necessarily an integer: a shrunk fit spends effective rather
            than nominal freedom, and the criteria must charge what was
            actually used.
        schema_version: Serialization schema version.
    """

    llf: float
    nobs: int
    n_params: float = field(repr=False)
    schema_version: int = field(default=_SCHEMA_VERSION, repr=False)

    @property
    def information_criteria(self) -> InformationCriteria:
        """All three model-selection criteria for this fit."""
        return InformationCriteria.from_likelihood(self.llf, self.nobs, self.n_params)

    @property
    def aic(self) -> float:
        """Akaike information criterion."""
        return self.information_criteria.aic

    @property
    def bic(self) -> float:
        """Bayesian (Schwarz) information criterion."""
        return self.information_criteria.bic

    @property
    def hqic(self) -> float:
        """Hannan-Quinn information criterion."""
        return self.information_criteria.hqic


@dataclass(frozen=True, kw_only=True)
class _FilterResult:
    """Common ancestor of every forward-filtering output.

    Any filter -- Kalman, Hamilton, extended, particle -- produces a
    likelihood decomposition, and nothing else in common: the Kalman filter
    carries a continuous state and its covariance, the Hamilton filter a
    discrete regime distribution. Those are genuinely different objects, so
    only the likelihood lives here.

    Attributes:
        loglikelihood: Total log-likelihood of the data under the model.
        loglikelihood_contributions: Per-period contributions, shape ``(n,)``.
    """

    loglikelihood: float
    loglikelihood_contributions: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class _SmootherResult:
    """Common ancestor of every backward-smoothing output.

    Deliberately empty of fields: smoothed states and smoothed regime
    probabilities share no data, only the fact that both condition on the
    full sample. This exists to bound the ``S`` parameter of
    :class:`StateSpaceModel`, not to factor out shared state.
    """


@dataclass(frozen=True, kw_only=True, slots=True)
class _MeanResult(_FittedResult):
    """Adds the endog/residual/fitted surface shared by conditional-mean results.

    Attributes:
        endog: The observed series, retained so forecasts can be produced from
            the result alone without holding a reference to the model.
        resid: One-step residuals on the estimation sample.
        fittedvalues: One-step fitted values on the estimation sample.
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    resid: npt.NDArray[np.float64] = field(repr=False)
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)


@dataclass(frozen=True, kw_only=True)
class _KalmanFilterResult(_FilterResult):
    """Output of a linear-Gaussian forward filtering pass.

    Attributes:
        predicted_state: One-step-ahead predicted states ``a_{t|t-1}``, ``(n, m)``.
        predicted_state_cov: Predicted state covariances ``P_{t|t-1}``, ``(n, m, m)``.
        filtered_state: Contemporaneously filtered states ``a_{t|t}``, ``(n, m)``.
        filtered_state_cov: Filtered state covariances ``P_{t|t}``, ``(n, m, m)``.
    """

    predicted_state: npt.NDArray[np.float64]
    predicted_state_cov: npt.NDArray[np.float64]
    filtered_state: npt.NDArray[np.float64]
    filtered_state_cov: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class _DurbinKoopmanSmootherResult(_SmootherResult):
    """Output of a linear-Gaussian backward smoothing pass.

    Attributes:
        smoothed_state: Smoothed states ``a_{t|n}``, shape ``(n, m)``.
        smoothed_state_cov: Smoothed state covariances ``V_{t|n}``, ``(n, m, m)``.
    """

    smoothed_state: npt.NDArray[np.float64]
    smoothed_state_cov: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class _HamiltonFilterResult(_FilterResult):
    """Output of the Hamilton forward filter.

    Attributes:
        filtered_prob: Contemporaneous regime probabilities
            ``Pr(S_t = j | y_{1..t})``, shape ``(T, K)``.
        predicted_prob: One-step-ahead regime probabilities
            ``Pr(S_t = j | y_{1..t-1})``, shape ``(T, K)``.
        loglikelihood: Total log-likelihood ``sum_t log Pr(y_t | y_{1..t-1})``.
        loglikelihood_contributions: Per-period contributions, shape ``(T,)``.
    """

    filtered_prob: npt.NDArray[np.float64]
    predicted_prob: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class _KimSmootherResult(_SmootherResult):
    """Output of the Kim backward smoother.

    Attributes:
        smoothed_prob: Full-sample regime probabilities
            ``Pr(S_t = j | y_{1..T})``, shape ``(T, K)``.
        smoothed_joint_prob: Consecutive-pair probabilities
            ``Pr(S_t = i, S_{t+1} = j | y_{1..T})``, shape ``(T - 1, K, K)``;
            entry ``[t]`` links period ``t`` to ``t + 1``. Empty when ``T == 1``.
            These are the expected-transition weights of the EM M-step.
    """

    smoothed_prob: npt.NDArray[np.float64]
    smoothed_joint_prob: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _ConditionalVarianceResult(
    _SummaryMixin, _SeriesMixin, _ComparisonMixin, _ConditionalVarianceMixin
):
    """What every fitted conditional-variance model reports.

    Abstract in intent: it carries the mean intercept, the variance intercept
    and the fitted variance path, and derives from the path alone everything
    that does not depend on how the path was produced. :attr:`persistence` is
    declared here and implemented by each subclass, because the two arms of the
    group measure it in incompatible ways.

    Attributes:
        endog: The full observed series.
        fittedvalues: Fitted conditional means over the effective sample.
        resid: Mean residuals over the effective sample.
        llf: Maximized joint Gaussian log-likelihood.
        nobs: Effective observations.
        n_params: Free parameter count.
        mean: ``"constant"`` or ``"zero"``.
        const: Mean intercept, or ``None`` when the mean is zero.
        omega: Variance intercept.
        conditional_variance: The fitted variance path.
    """

    endog: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: float
    mean: str
    const: float | None
    omega: float
    conditional_variance: npt.NDArray[np.float64]

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """Aligned per-observation output, widened by the variance surface.

        Uses the two-argument ``super`` deliberately. ``@dataclass(slots=True)``
        builds a *new* class object and rebinds the name to it, so the
        ``__class__`` cell that zero-argument ``super()`` closes over still
        points at the original, pre-slots class -- which no subclass inherits
        from. ``super()._series()`` therefore raises ``TypeError`` the moment a
        subclass calls it, and only for subclasses, so it survives any test
        that exercises the base alone.
        """
        base = super(_ConditionalVarianceResult, self)._series()
        base["conditional_variance"] = self.conditional_variance
        base["conditional_volatility"] = self.conditional_volatility
        base["standardized_resid"] = self.standardized_resid
        return base

    @property
    def standardized_resid(self) -> npt.NDArray[np.float64]:
        """Mean residuals scaled by the fitted conditional volatility."""
        return self.resid / self.conditional_volatility

    @property
    def persistence(self) -> float:
        """The decay rate of a shock to the conditional variance.

        Raises:
            NotImplementedError: If a concrete result does not define it.
        """
        raise NotImplementedError

    @property
    def is_covariance_stationary(self) -> bool:
        """Whether the variance process has a finite unconditional level."""
        return self.persistence < 1.0

    @property
    def half_life(self) -> float:
        """Periods for a variance shock to decay by half.

        Returns:
            ``inf`` when the process is not covariance stationary, since a
            shock to an integrated variance never halves.
        """
        p = self.persistence
        if not 0.0 < p < 1.0:
            return float("inf")
        return float(np.log(0.5) / np.log(p))


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _MeanFunctionResult(_SummaryMixin, _SeriesMixin, _ComparisonMixin):
    """What every fitted learned-mean model reports.

    Deliberately declares no ``params`` mapping and no ``predict``. There is no
    parameter vector to report, and the two members route features to learners
    differently enough that a shared ``predict`` signature would have to take
    an argument one of them ignores -- so each concrete result declares the
    signature its own routing requires.

    Attributes:
        endog: The full input series.
        fittedvalues: In-sample conditional means over the effective sample.
        resid: Residuals against those means.
        llf: Concentrated Gaussian log-likelihood at the fitted mean. A
            goodness-of-fit summary, not a maximized likelihood: the learner is
            penalized, so the fit does not maximize this.
        nobs: Effective sample size after trimming for lags and delay.
        n_params: Learner parameters plus the concentrated variance, and the
            threshold where one is estimated.
        order: Number of lagged levels fed to the learner.
        sigma2: Concentrated innovation variance, ``ssr / nobs``.
        engine: Class name of the training backend.
        engine_config: Full repr of the backend, so the fit is reproducible.
    """

    endog: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: float
    order: int
    sigma2: float
    engine: str
    engine_config: str

    @property
    def n_learner_parameters(self) -> int:
        """Weights and biases the backend fitted, excluding the variance.

        Returns:
            The nominal count.

        Raises:
            NotImplementedError: If the concrete result does not supply one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must report how many parameters its learners hold."
        )

    @property
    def ssr(self) -> float:
        """Sum of squared residuals over the effective sample."""
        return float(self.resid @ self.resid)

    @property
    def r_squared(self) -> float:
        """Share of the effective sample's variance the fitted mean explains.

        The headline number for this family, since there is no coefficient
        table and the log-likelihood is not a maximized one. Computed against
        the effective sample's own total sum of squares, so it is comparable
        across specifications only when they trim the same number of leading
        observations -- which is why :meth:`compare` refuses mismatched
        samples.

        Note this is an *in-sample* figure for a flexible learner, so it
        rewards capacity: a network with enough hidden units will drive it
        toward one without having learned anything that generalizes. Read it
        next to :attr:`parameters_per_observation`.
        """
        target = self.endog[self.endog.shape[0] - self.nobs :]
        centered = target - target.mean()
        total = float(centered @ centered)
        if total <= 0.0:
            return 0.0
        return 1.0 - self.ssr / total

    @property
    def parameters_per_observation(self) -> float:
        """Nominal learner parameters divided by the effective sample size.

        The capacity canary. Weight decay means the *effective* count is lower
        than this, by an amount no closed form recovers, so treat it as an
        upper bound on how hard the learner could overfit rather than as a
        measurement of how hard it did.
        """
        return self.n_learner_parameters / self.nobs

    def _capacity_note(self) -> str:
        """One line on nominal capacity relative to the sample."""
        ratio = self.parameters_per_observation
        verdict = (
            "high -- the in-sample R-squared above is optimistic"
            if ratio > _CAPACITY_WARNING
            else "modest"
        )
        return (
            f"Capacity: {self.n_learner_parameters} learner parameters over "
            f"{self.nobs} observations ({ratio:.3f} per observation, {verdict}). "
            f"Weight decay lowers the effective count below the nominal one, so "
            f"this is an upper bound."
        )

    def _criteria_note(self) -> str:
        """One line on why the information criteria here are not comparable."""
        return (
            "AIC/BIC/HQIC penalize the nominal parameter count, which overstates "
            "a regularized learner's effective degrees of freedom. Use them to "
            "choose hidden units within one engine configuration, not to rank "
            "this model against a linear one."
        )

    def _engine_note(self) -> str:
        """One line recording the backend, since it is part of the specification."""
        return f"Trained by {self.engine_config}."

    def _summary_metadata(self) -> tuple[tuple[str, str], ...]:
        """Left/right metadata pairs shared by both members."""
        ic: InformationCriteria = self.information_criteria
        return (
            ("Model", self._comparison_label()),
            ("Log-likelihood", f"{self.llf:.3f}"),
            ("Engine", self.engine),
            ("AIC", f"{ic.aic:.3f}"),
            ("Lags", f"{self.order}"),
            ("BIC", f"{ic.bic:.3f}"),
            ("Observations", f"{self.nobs}"),
            ("HQIC", f"{ic.hqic:.3f}"),
            ("R-squared", f"{self.r_squared:.4f}"),
            ("sigma2", f"{self.sigma2:.4f}"),
        )

    def _likelihood_ratio_obstacle(self, counterpart: _ComparisonMixin) -> str | None:
        """Block every chi-squared likelihood-ratio test involving this result.

        Under the null that the learned mean is linear, a hidden unit's
        input weights do not appear in the likelihood at all -- the unit's
        output weight is zero, so nothing multiplies them -- and that output
        weight itself sits on the boundary of the parameter space. Both
        conditions independently break the chi-squared limit. The Terasvirta
        LM test exists precisely because of this, and it is what to reach for.

        The reported log-likelihood is also penalized rather than maximized, so
        a ratio of two of them is not a likelihood ratio in the first place.

        Args:
            counterpart: Ignored; the obstacle is a property of this family.

        Returns:
            The explanatory message, always.
        """
        return (
            "a chi-squared likelihood-ratio test is not valid for a learned mean "
            "function: under linearity the hidden-layer input weights are "
            "unidentified and the output weights lie on the boundary at zero, and "
            "the reported log-likelihood is penalized rather than maximized. Use "
            "the Terasvirta neural-network linearity test, or a bootstrap."
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _ObservedRegimeResult(_SummaryMixin, _SeriesMixin, _ComparisonMixin):
    """What every fitted observed-regime autoregression reports.

    Carries the two coefficient blocks, the transition variable they are
    indexed by, and the split point between them. Subclasses supply the
    regime weight, which is the one thing an abrupt and a smooth transition
    genuinely do differently.

    Attributes:
        endog: The full input series.
        fittedvalues: One-step fitted values over the effective sample.
        resid: Residuals over the effective sample.
        llf: Gaussian log-likelihood at the least-squares estimate.
        nobs: Effective sample size after trimming for lags and delay.
        n_params: Free parameters, including the innovation variance.
        order: Autoregressive order within each regime.
        delay: Delay of the transition variable.
        threshold: The regime split point, in the units of the transition
            variable.
        threshold_values: The transition variable, aligned with ``resid``.
        lower_params: Lower-regime coefficients, intercept first.
        upper_params: Upper-regime coefficients, intercept first.
        sigma2: Innovation variance, ``ssr / nobs``.
        ssr: Sum of squared residuals at the optimum.
    """

    endog: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: float
    order: int
    delay: int
    threshold: float
    threshold_values: npt.NDArray[np.float64]
    lower_params: npt.NDArray[np.float64]
    upper_params: npt.NDArray[np.float64]
    sigma2: float
    ssr: float

    @property
    def regime_weight(self) -> npt.NDArray[np.float64]:
        """Weight on the upper regime at each observation, in ``[0, 1]``.

        Returns:
            An array of length ``nobs``.

        Raises:
            NotImplementedError: If the concrete result does not define one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define regime_weight to describe its transition."
        )

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """Aligned per-observation output, widened by the regime surface.

        Uses the two-argument ``super`` deliberately. ``@dataclass(slots=True)``
        builds a *new* class object and rebinds the name to it, so the
        ``__class__`` cell that zero-argument ``super()`` closes over still
        points at the original, pre-slots class -- which no subclass inherits
        from. ``super()._series()`` would therefore raise ``TypeError`` the
        moment a subclass called it, and only for subclasses, so it would
        survive any test that exercised the base alone.

        Returns:
            The observed/fitted/residual triple plus the transition variable
            and the upper-regime weight.
        """
        base = super(_ObservedRegimeResult, self)._series()
        base["threshold_variable"] = self.threshold_values
        base["regime_weight"] = self.regime_weight
        return base

    @property
    def upper_fraction(self) -> float:
        """Share of the effective sample assigned to the upper regime.

        For a hard threshold this is the exact proportion of observations above
        the split; for a smooth transition it is the mean transition weight,
        which is the natural continuous analogue.
        """
        return float(np.mean(self.regime_weight))

    @property
    def lower_stability(self) -> _StabilityTest:
        """Companion-eigenvalue verdict for the lower-regime AR block."""
        return _StabilityTest.assess_stability(self.lower_params[1:])

    @property
    def upper_stability(self) -> _StabilityTest:
        """Companion-eigenvalue verdict for the upper-regime AR block."""
        return _StabilityTest.assess_stability(self.upper_params[1:])

    @property
    def is_regimewise_stationary(self) -> bool:
        """Whether *both* regimes are stationary read as linear autoregressions.

        Sufficient for global stationarity of the nonlinear process, but not
        necessary: a model whose inner regime is explosive and whose outer
        regime contracts is globally stationary, because excursions are pulled
        back. A ``False`` here is a prompt to check the regime blocks
        individually, not a verdict that the process explodes.
        """
        return self.lower_stability.is_stable and self.upper_stability.is_stable

    def _transition_params(self) -> dict[str, float]:
        """Parameters describing the transition itself, keyed for display."""
        return {}

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order."""
        out: dict[str, float] = {}
        for label, block in (("lower", self.lower_params), ("upper", self.upper_params)):
            out[f"{label}.const"] = float(block[0])
            for i, value in enumerate(block[1:], start=1):
                out[f"{label}.ar.L{i}"] = float(value)
        out["threshold"] = self.threshold
        out.update(self._transition_params())
        out["sigma2"] = self.sigma2
        return out

    def likelihood_ratio_test(self, unrestricted: _ComparisonMixin) -> _LikelihoodRatioTest:
        """Refuse the test: the threshold is unidentified under the null.

        Args:
            unrestricted: Ignored.

        Returns:
            Never returns.

        Raises:
            SpecificationError: Always. Under linearity the threshold and the
                delay drop out of the likelihood entirely, so the likelihood
                ratio has no chi-squared limit and the nuisance parameters are
                present only under the alternative -- the Davies problem. The
                correct procedure is a bootstrap or a supremum-type test
                (Hansen 1996), not a chi-squared tail. Returning a well-formed
                and wrong p-value here would be worse than declining.
        """
        raise SpecificationError(
            "a chi-squared likelihood-ratio test is not valid for observed-regime "
            "autoregressions: the threshold and delay are not identified under the "
            "null of linearity, so the statistic has a non-standard distribution. "
            "Use a bootstrap or a supremum-type test instead."
        )

    def _summary_metadata(self) -> tuple[tuple[str, str], ...]:
        """Left/right metadata pairs shared by both transition shapes."""
        ic: InformationCriteria = self.information_criteria
        return (
            ("Model", self._comparison_label()),
            ("Log-likelihood", f"{self.llf:.3f}"),
            ("Delay", f"{self.delay}"),
            ("AIC", f"{ic.aic:.3f}"),
            ("Threshold", f"{self.threshold:.4f}"),
            ("BIC", f"{ic.bic:.3f}"),
            ("Observations", f"{self.nobs}"),
            ("HQIC", f"{ic.hqic:.3f}"),
        )

    def _stationarity_note(self) -> str:
        """One line reporting each regime's largest companion root."""
        return (
            f"Regime-wise stationary: {self.is_regimewise_stationary}   "
            f"max |root| lower = {self.lower_stability.max_modulus:.4f}, "
            f"upper = {self.upper_stability.max_modulus:.4f}"
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorResult:
    """What every fitted multivariate result carries, whatever the family.

    Twelve fields, and the test for membership is narrow: a field belongs here
    only if it means the same thing for a reduced-form autoregression, a
    fixed-effects panel, and an error-correction model. ``coefficients`` fails
    that test -- for a VECM it is a derived levels representation and for a
    conditional model it does not exist -- so it lives on the families that have
    one rather than here, which is what lets a conditional result be a
    first-class member of this hierarchy instead of a closed one with holes.

    The two prior fields live here rather than on the families because
    shrinkage is orthogonal to specification: any of them can be estimated
    under a prior, and putting the fields on the ancestor is what gave every
    family the Bayesian mode without a line of change to its own body.

    Attributes:
        endog: The sample the model was fitted to.
        names: Variable labels, in the order the columns appear.
        order: Autoregressive order, counted in the family's own convention.
        trend: Deterministic specification.
        sigma_u: Residual covariance with the degrees-of-freedom correction.
        sigma_ml: Residual covariance divided by the effective sample.
        resid: Residuals over the effective sample.
        fittedvalues: One-step conditional means over the effective sample.
        design: The regressor matrix as estimated.
        llf: Gaussian log-likelihood.
        nobs: Effective sample size.
        n_params: Parameters the fit spends, covariance included. Not an
            integer under shrinkage, which is the point: a shrunk model uses
            less freedom than its design has columns and an information
            criterion must charge it for what it used.
        posterior: The posterior covariance when a prior was applied,
            otherwise ``None``. Carried rather than rebuilt because under a
            prior there is nothing to rebuild it from.
        prior_label: What the prior was, for the summary.
    """

    endog: npt.NDArray[np.float64]
    names: tuple[str, ...]
    order: int
    trend: str
    sigma_u: npt.NDArray[np.float64]
    sigma_ml: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    design: npt.NDArray[np.float64]
    llf: float
    nobs: float
    n_params: float
    posterior: _CoefficientInference | None = None
    prior_label: str = "none"


@dataclass(frozen=True, kw_only=True, slots=True)
class _ErrorCorrectionResult(_VectorResult):
    """The coordinates an error-correction model is actually estimated in.

    Shared by the closed and conditional families, which differ in what they can
    propagate and not in how they are parameterized. Everything here reads the
    short-run regression, so all of it is valid for both: the cointegrating
    space, the adjustment loadings, the lagged-difference coefficients, and the
    two tests that restrict them.

    Attributes:
        rank: Cointegrating rank.
        cointegration_trend: The Johansen case the model was estimated under.
        alpha: ``(k_y, r)`` adjustment loadings.
        beta: ``(k_y [+ k_x] [+ 1], r)`` cointegrating vectors.
        gamma: ``(p - 1, k_y, ...)`` coefficients on lagged differences.
        short_run_deterministic: ``(d_s, k_y)`` unrestricted deterministic terms.
        eigenvalues: The squared canonical correlations, descending.
    """

    rank: int
    cointegration_trend: CointegrationTrend
    alpha: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    gamma: npt.NDArray[np.float64]
    short_run_deterministic: npt.NDArray[np.float64]
    eigenvalues: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _VectorObservedRegimeResult(_SummaryMixin, _ComparisonMixin):
    """What every fitted observed-regime vector autoregression reports.

    The multivariate counterpart of :class:`_ObservedRegimeResult`: two full
    coefficient stacks, the transition variable they are indexed by, and the
    split point between them. Subclasses supply the regime weight and the
    per-regime innovation covariance -- the two things an abrupt and a smooth
    transition genuinely do differently.

    What this base deliberately is *not* is a :class:`_VectorResult` with the
    propagation surface. A regime model has no single coefficient stack, no
    single companion matrix, and no single moving-average representation, so
    every propagation question must name a regime, and the linear answer it
    gets holds the regime frozen: it is the response *within* that regime,
    valid to the extent the shock does not itself move the transition
    variable across the split. The honest nonlinear objects -- generalized
    impulse responses averaged over histories and simulated paths, Koop-
    Pesaran-Potter (1996) -- are simulation-based and belong to a later
    inference layer; nothing here pretends to be them.

    Attributes:
        endog: The full observed panel.
        names: Variable labels, in column order.
        order: Autoregressive order within each regime.
        trend: Deterministic specification.
        delay: Delay of the transition variable.
        threshold: The regime split point, in the transition variable's units.
        threshold_name: The transition variable's label -- a variable name
            when self-exciting, ``"external"`` otherwise.
        threshold_values: The *delayed* transition variable ``z_t``, aligned
            with ``resid``; what the regime weight is evaluated on.
        transition_series: The raw transition series, aligned with ``endog``
            -- the column of ``endog`` when self-exciting, the external
            variable otherwise. Carried because a forecast needs the last
            ``delay`` raw values, which the delayed alignment cannot supply.
        lower_coefficients: ``(p, k, k)`` lag stack of the lower regime.
        upper_coefficients: ``(p, k, k)`` lag stack of the upper regime.
        lower_deterministic: Lower-regime deterministic coefficients.
        upper_deterministic: Upper-regime deterministic coefficients.
        resid: Residuals over the effective sample.
        fittedvalues: One-step conditional means over the effective sample.
        llf: Gaussian log-likelihood at the optimum.
        nobs: Effective sample size after trimming for lags and delay.
        n_params: Free parameters, covariances and transition included.
    """

    endog: npt.NDArray[np.float64]
    names: tuple[str, ...]
    order: int
    trend: str
    delay: int
    threshold: float
    threshold_name: str
    threshold_values: npt.NDArray[np.float64]
    transition_series: npt.NDArray[np.float64]
    lower_coefficients: npt.NDArray[np.float64]
    upper_coefficients: npt.NDArray[np.float64]
    lower_deterministic: npt.NDArray[np.float64]
    upper_deterministic: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: float

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables."""
        return len(self.names)

    @property
    def regime_weight(self) -> npt.NDArray[np.float64]:
        """Weight on the upper regime at each observation, in ``[0, 1]``.

        Returns:
            An array of length ``nobs``.

        Raises:
            NotImplementedError: If the concrete result does not define one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define regime_weight to describe its transition."
        )

    def _weight_at(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The regime weight evaluated at arbitrary transition-variable values.

        Args:
            values: Transition-variable values.

        Returns:
            Weights in ``[0, 1]``, one per value.

        Raises:
            NotImplementedError: If the concrete result does not define one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define _weight_at to be forecastable."
        )

    def _regime_sigma(self, regime: Regime) -> npt.NDArray[np.float64]:
        """The innovation covariance the named regime propagates.

        Args:
            regime: ``"lower"`` or ``"upper"``.

        Returns:
            A ``(k, k)`` covariance.

        Raises:
            NotImplementedError: If the concrete result does not define one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define _regime_sigma to orthogonalize."
        )

    def _regime_stack(self, regime: Regime) -> npt.NDArray[np.float64]:
        """The named regime's lag stack, after validating the name."""
        choice = validate_choice(regime, Regime, "regime")
        return self.lower_coefficients if choice == "lower" else self.upper_coefficients

    @property
    def upper_fraction(self) -> float:
        """Share of the effective sample carried by the upper regime.

        The exact proportion above the split for a hard threshold; the mean
        transition weight for a smooth one.
        """
        return float(np.mean(self.regime_weight))

    @property
    def lower_stability(self) -> _StabilityTest:
        """Companion-eigenvalue verdict for the lower regime's lag stack."""
        return _StabilityTest.assess_stability(self.lower_coefficients)

    @property
    def upper_stability(self) -> _StabilityTest:
        """Companion-eigenvalue verdict for the upper regime's lag stack."""
        return _StabilityTest.assess_stability(self.upper_coefficients)

    @property
    def is_regimewise_stationary(self) -> bool:
        """Whether *both* regimes are stationary read as linear systems.

        Sufficient for global stationarity of the nonlinear process, but not
        necessary: a model whose inner regime is explosive and whose outer
        regime contracts is globally stationary, because excursions are
        pulled back. A ``False`` here is a prompt to inspect the regimes
        individually, not a verdict that the process explodes.
        """
        return self.lower_stability.is_stable and self.upper_stability.is_stable

    def ma_representation(self, horizon: int = 20, *, regime: Regime) -> npt.NDArray[np.float64]:
        """Moving-average matrices of one regime, held frozen.

        Args:
            horizon: Largest lead to return.
            regime: Which regime's dynamics to propagate.

        Returns:
            An array of shape ``(horizon + 1, k, k)`` with ``Psi_0 = I``.

        Raises:
            SpecificationError: If ``horizon`` is negative or the regime name
                is unknown.
        """
        if horizon < 0:
            raise SpecificationError(f"horizon must be non-negative; got {horizon}.")
        stack = self._regime_stack(regime)
        k, p = self.k_endog, self.order
        out = np.empty((horizon + 1, k, k), dtype=np.float64)
        selector = np.zeros((k, k * p), dtype=np.float64)
        selector[:, :k] = np.eye(k)
        power = np.eye(k * p, dtype=np.float64)
        companion = companion_matrix(stack)
        for h in range(horizon + 1):
            out[h] = selector @ power @ selector.T
            power = power @ companion
        return out

    def irf(
        self,
        horizon: int = 20,
        *,
        regime: Regime,
        orthogonalized: bool = True,
        cumulative: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Impulse responses within one regime, held frozen.

        Two declarations are being made at once here, and both are named in
        the signature rather than defaulted out of sight. Freezing the regime
        is a *linearization*: the response is exact only while the shock does
        not push the transition variable across the split, so read it as
        "the dynamics prevailing in this regime", not as the response of the
        nonlinear process -- that object requires simulation over histories
        (Koop-Pesaran-Potter 1996) and is deliberately not faked here.
        Orthogonalization is the same recursive identifying restriction it is
        for a linear VAR, applied with this regime's own Cholesky factor.

        Args:
            horizon: Largest lead to return.
            regime: Which regime's dynamics to propagate.
            orthogonalized: Rotate by the regime's Cholesky factor; the
                ordering of ``names`` is then an identifying assumption.
            cumulative: Return running sums.

        Returns:
            An array of shape ``(horizon + 1, k, k)``; entry ``[h, i, j]`` is
            the response of variable ``i`` at lead ``h`` to shock ``j``.
        """
        psi = self.ma_representation(horizon, regime=regime)
        out = psi @ np.linalg.cholesky(self._regime_sigma(regime)) if orthogonalized else psi
        return np.cumsum(out, axis=0) if cumulative else out

    def fevd(self, horizon: int = 20, *, regime: Regime) -> npt.NDArray[np.float64]:
        """Forecast-error variance decomposition within one regime, held frozen.

        Args:
            horizon: Largest lead to return.
            regime: Which regime's dynamics and covariance to read.

        Returns:
            An array of shape ``(horizon + 1, k, k)`` whose rows sum to one.
            Inherits both caveats of :meth:`irf`: the regime is frozen and the
            ordering of ``names`` is an identifying assumption.
        """
        theta = self.irf(horizon, regime=regime, orthogonalized=True)
        contribution = np.cumsum(theta**2, axis=0)
        return contribution / contribution.sum(axis=2, keepdims=True)

    def forecast(
        self, steps: int = 1, *, transition_future: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Iterate the fitted skeleton forward, blending regimes as it goes.

        At each step the transition weight is evaluated on the path itself --
        the forecast re-enters the transition variable when the model is
        self-exciting -- and the two regimes' one-step maps are blended by
        that weight. This is the *skeleton* forecast: exact at one step, and
        an approximation beyond, because for a nonlinear model the multi-step
        conditional mean averages the transition over future shocks rather
        than evaluating it at the mean path. The gap is small when the path
        stays clear of the threshold and largest on top of it; a simulation-
        based conditional mean belongs to a later inference layer.

        Args:
            steps: Forecast horizon, at least one.
            transition_future: Future values of an *external* transition
                variable, at least ``steps - delay + 1`` of them when
                ``steps > delay``. Ignored for a self-exciting model, which
                generates its own transition path; required for an external
                one that must see past its observed history, refused rather
                than extrapolated otherwise.

        Returns:
            An array of shape ``(steps, k)``.

        Raises:
            SpecificationError: If ``steps`` is less than one, or an external
                transition path is needed and not supplied.
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        k, p, d, n = self.k_endog, self.order, self.delay, self.endog.shape[0]
        self_exciting = self.threshold_name in self.names
        column = self.names.index(self.threshold_name) if self_exciting else -1
        future = (
            None
            if transition_future is None
            else np.atleast_1d(np.asarray(transition_future, dtype=np.float64))
        )
        if not self_exciting and steps > d:
            needed = steps - d
            if future is None:
                raise SpecificationError(
                    "the transition variable is external, so forecasting "
                    f"{steps} steps needs its future path beyond the observed "
                    f"history: pass transition_future with at least {needed} "
                    "values. This model holds no process for the transition "
                    "variable and will not extrapolate one."
                )
            if future.shape[0] < needed:
                raise SpecificationError(
                    f"transition_future has {future.shape[0]} values but "
                    f"forecasting {steps} steps at delay {d} needs at least "
                    f"{needed}."
                )
        det = deterministic_columns(self.trend, steps, start=n + 1)
        history = [self.endog[n - i - 1] for i in range(p)]
        out = np.empty((steps, k), dtype=np.float64)
        for h in range(steps):
            back = h - d
            if back < 0:
                z = float(self.transition_series[n + back])
            elif self_exciting:
                z = float(out[back, column])
            else:
                assert future is not None
                z = float(future[back])
            w = float(self._weight_at(np.asarray([z]))[0])
            lower = (
                det[h] @ self.lower_deterministic
                if self.lower_deterministic.shape[0]
                else np.zeros(k, dtype=np.float64)
            )
            upper = (
                det[h] @ self.upper_deterministic
                if self.upper_deterministic.shape[0]
                else np.zeros(k, dtype=np.float64)
            )
            for i in range(p):
                lower = lower + self.lower_coefficients[i] @ history[i]
                upper = upper + self.upper_coefficients[i] @ history[i]
            point = (1.0 - w) * lower + w * upper
            out[h] = point
            history = [point, *history[: p - 1]] if p else []
        return out

    def equation(self, name: str, *, regime: Regime) -> dict[str, float]:
        """One equation of one regime, keyed by regressor.

        Args:
            name: An endogenous variable.
            regime: Which regime's coefficients to read.

        Returns:
            Deterministic terms, then endogenous lags, in design order.

        Raises:
            SpecificationError: If the variable or regime is unknown.
        """
        if name not in self.names:
            raise SpecificationError(f"unknown variable {name!r}; expected one of {self.names}.")
        choice = validate_choice(regime, Regime, "regime")
        stack = self._regime_stack(choice)
        deterministic = self.lower_deterministic if choice == "lower" else self.upper_deterministic
        row = self.names.index(name)
        out: dict[str, float] = {}
        labels = (["const"] if self.trend in ("c", "ct") else []) + (
            ["trend"] if self.trend == "ct" else []
        )
        for j, label in enumerate(labels):
            out[label] = float(deterministic[j, row])
        for lag in range(self.order):
            for j, source in enumerate(self.names):
                out[f"{source}.L{lag + 1}"] = float(stack[lag][row, j])
        return out

    def likelihood_ratio_test(self, unrestricted: _ComparisonMixin) -> _LikelihoodRatioTest:
        """Refuse the test: the threshold is unidentified under the null.

        Args:
            unrestricted: Ignored.

        Returns:
            Never returns.

        Raises:
            SpecificationError: Always. Under linearity the threshold, the
                delay, and any transition speed drop out of the likelihood
                entirely, so the likelihood ratio has no chi-squared limit --
                the Davies problem. The correct procedure is a bootstrap or a
                supremum-type test (Hansen 1996), and returning a well-formed
                and wrong p-value here would be worse than declining.
        """
        raise SpecificationError(
            "a chi-squared likelihood-ratio test is not valid for observed-regime "
            "vector autoregressions: the threshold and delay are not identified "
            "under the null of linearity, so the statistic has a non-standard "
            "distribution. Use a bootstrap or a supremum-type test instead."
        )

    def _stationarity_note(self) -> str:
        """One line reporting each regime's largest companion root."""
        return (
            f"Regime-wise stationary: {self.is_regimewise_stationary}   "
            f"max |root| lower = {self.lower_stability.max_modulus:.4f}, "
            f"upper = {self.upper_stability.max_modulus:.4f}"
        )

    def _regime_conditional_note(self) -> str:
        """The one caveat every propagation method here carries."""
        return (
            "irf(), fevd(), and ma_representation() are regime-conditional: each "
            "freezes one regime's dynamics, which linearizes the model and is "
            "exact only while the shock does not move the transition variable "
            "across the split. Orthogonalized responses additionally impose the "
            "recursive ordering of names, an identifying assumption this reduced "
            "form has not declared."
        )

    def _summary_metadata(self) -> tuple[tuple[str, str], ...]:
        """Left/right metadata pairs shared by both transition shapes."""
        ic: InformationCriteria = self.information_criteria
        return (
            ("Model", self._comparison_label()),
            ("Log-likelihood", f"{self.llf:.3f}"),
            ("Variables", f"{self.k_endog}"),
            ("AIC", f"{ic.aic:.3f}"),
            ("Threshold", f"{self.threshold:.4f}"),
            ("BIC", f"{ic.bic:.3f}"),
            ("Transition variable", f"{self.threshold_name}.L{self.delay}"),
            ("HQIC", f"{ic.hqic:.3f}"),
            ("Observations", f"{self.nobs}"),
            ("Trend", self.trend),
        )

    def _own_lag_rows(self) -> tuple[tuple[str, ...], ...]:
        """Per-variable first-own-lag coefficients, one row per equation.

        The full stacks are ``k * k * p`` numbers per regime and belong in
        :attr:`lower_coefficients` / :attr:`upper_coefficients` or
        :meth:`equation`, not in a summary table; the own first lag is the
        one coefficient per equation a reader scans to see how the regimes
        differ.
        """
        return tuple(
            (
                name,
                f"{self.lower_coefficients[0][i, i]:.4f}",
                f"{self.upper_coefficients[0][i, i]:.4f}",
            )
            for i, name in enumerate(self.names)
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _RegimeSystemResult:
    """One regime of a fitted regime model, dressed as a closed linear system.

    The interchange view of the regime families, on the pattern
    :class:`_ConditionalLevels` set for the global model: a public result
    produces it, the identification layer consumes it, and users receive it
    without ever constructing it. Conditional on its regime the model is a
    linear VAR, and this view exposes exactly the closed-system surface an
    identification model reads -- labels, coefficients, an innovation
    covariance, residuals, and a moving-average representation -- so it
    passes anywhere a fitted VAR result passes as an identification source.

    Two honesty notes on the statistical members. ``nobs`` is the *expected*
    number of periods governed by this regime -- the sum of its weights, a
    float, because under a latent chain no date belongs to a regime with
    certainty. ``resid`` is the full-sample residuals under this regime's
    parameters, weighted by the square root of the regime weights and scaled
    so their second moment matches :attr:`sigma_u`; a scheme that only reads
    the covariance (recursive, long-run, short-run) is unaffected, while a
    scheme that reads residual moments directly is consuming a
    weight-averaged object and should be interpreted accordingly.

    Attributes:
        index: This regime's label under the producing fit's convention.
        names: Variable labels, in column order.
        order: Autoregressive order.
        trend: Deterministic specification.
        coefficients: ``(p, k, k)`` lag stack of this regime.
        deterministic: This regime's deterministic coefficients.
        sigma_u: This regime's innovation covariance.
        resid: Probability-weighted residuals, aligned with the effective
            sample.
        weight: This regime's weights -- smoothed probabilities for a
            latent chain -- one per effective row.
        nobs: Expected periods governed by this regime.
    """

    index: int
    names: tuple[str, ...]
    order: int
    trend: str
    coefficients: npt.NDArray[np.float64] = field(repr=False)
    deterministic: npt.NDArray[np.float64] = field(repr=False)
    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    resid: npt.NDArray[np.float64] = field(repr=False)
    weight: npt.NDArray[np.float64] = field(repr=False)
    nobs: float

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables."""
        return len(self.names)

    def stability_check(self) -> _StabilityTest:
        """Companion-eigenvalue verdict for this regime's lag stack."""
        return _StabilityTest.assess_stability(self.coefficients)

    @property
    def is_stable(self) -> bool:
        """Whether this regime, read as a linear VAR, is stable."""
        return self.stability_check().is_stable

    def ma_representation(self, horizon: int = 20) -> npt.NDArray[np.float64]:
        """Moving-average matrices of this regime, held frozen.

        Args:
            horizon: Largest lead to return.

        Returns:
            An array of shape ``(horizon + 1, k, k)`` with ``Psi_0 = I``.

        Raises:
            SpecificationError: If ``horizon`` is negative.
        """
        if horizon < 0:
            raise SpecificationError(f"horizon must be non-negative; got {horizon}.")
        k, p = self.k_endog, self.order
        out = np.empty((horizon + 1, k, k), dtype=np.float64)
        if p == 0:
            out[:] = 0.0
            out[0] = np.eye(k)
            return out
        selector = np.zeros((k, k * p), dtype=np.float64)
        selector[:, :k] = np.eye(k)
        power = np.eye(k * p, dtype=np.float64)
        companion = companion_matrix(self.coefficients)
        for h in range(horizon + 1):
            out[h] = selector @ power @ selector.T
            power = power @ companion
        return out
