# filepath: /src/cultivars/engine/_internals/_parameters.py
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

"""Parameter records: a flat optimizer vector, named and on its natural scale.

An optimizer or a sampler moves in a flat vector of unconstrained
coordinates. A parameter record is what that vector means: the same numbers
split into named blocks and mapped back to the scale the model is written
in, variances exponentiated, persistences squeezed into their admissible
interval, autoregressive blocks unpacked from partial autocorrelations.
Each objective builds its record from a draw, and every consumer downstream
of the objective (the system builders, the emitters, the simulators, the
fit and result records) reads the record and never the vector. That is
what keeps a transform from being applied twice or not at all.

The records are frozen, keyword-only dataclasses, one per model family.
They carry values only: no data, no orders, no validation. A record built
by hand with an inadmissible value is accepted here and fails wherever it
is first used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "_AutoRegressionParameters",
    "_BoxJenkinsParameters",
    "_ConditionalVarianceParameters",
    "_DecayNelsonSiegelParameters",
    "_FractionalIntegrationParameters",
    "_FractionalVarianceParameters",
    "_LongMemoryVolatilityParameters",
    "_NelsonSiegelParameters",
    "_SmoothTransitionParameters",
    "_StochasticVolatilityParameters",
    "_StructuralParameters",
    "_TrendVolatilityParameters",
    "_VarianceParameters",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class _AutoRegressionParameters:
    """Structured parameters of an AR(p) exact-ML draw.

    Attributes:
        const: Mean intercept; ``0.0`` when the specification has no constant.
        ar_params: Autoregressive coefficients, already mapped back from the
            partial-autocorrelation parameterization.
        sigma2: Innovation variance, already exponentiated.
    """

    const: float
    """Mean intercept; ``0.0`` when the specification has no constant."""
    ar_params: npt.NDArray[np.float64]
    """Autoregressive coefficients, ``(p,)``, on their natural scale."""
    sigma2: float
    """Innovation variance."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _BoxJenkinsParameters:
    """Structured parameters of a seasonal ARIMA-with-regressors draw.

    Attributes:
        beta: Coefficients on the deterministic and exogenous block.
        ar_params: Non-seasonal AR coefficients.
        seasonal_ar_params: Seasonal AR coefficients.
        ma_params: Non-seasonal MA coefficients, in the sign convention of the
            observation equation.
        seasonal_ma_params: Seasonal MA coefficients.
        sigma2: Innovation variance.
    """

    beta: npt.NDArray[np.float64]
    """Coefficients on the deterministic and exogenous regressors."""
    ar_params: npt.NDArray[np.float64]
    """Non-seasonal autoregressive coefficients, ``(p,)``."""
    seasonal_ar_params: npt.NDArray[np.float64]
    """Seasonal autoregressive coefficients, ``(P,)``."""
    ma_params: npt.NDArray[np.float64]
    """Non-seasonal moving-average coefficients, ``(q,)``.

    In the sign convention of the observation equation,
    ``theta(L) = 1 + theta_1 L + ...``.
    """
    seasonal_ma_params: npt.NDArray[np.float64]
    """Seasonal moving-average coefficients, ``(Q,)``, in the same sign convention."""
    sigma2: float
    """Innovation variance."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _FractionalIntegrationParameters:
    """Structured parameters of an ARFIMA(p, d, q) draw.

    Attributes:
        mean: Estimated mean; ``0.0`` when the mean is not estimated.
        d: Fractional integration order, mapped into ``(-_D_MAX, _D_MAX)``.
        ar_params: Short-memory AR coefficients.
        ma_params: Short-memory MA coefficients.
        sigma2: Innovation variance.
    """

    mean: float
    """Estimated mean; ``0.0`` when the mean is not estimated."""
    d: float
    """Fractional integration order."""
    ar_params: npt.NDArray[np.float64]
    """Short-memory autoregressive coefficients, ``(p,)``."""
    ma_params: npt.NDArray[np.float64]
    """Short-memory moving-average coefficients, ``(q,)``."""
    sigma2: float
    """Innovation variance."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _VarianceParameters:
    """Parameters every conditional-variance draw carries.

    Attributes:
        mean: Conditional-mean coefficients, intercept first when present.
        omega: Variance intercept, in levels or in logs depending on the family.
    """

    mean: npt.NDArray[np.float64]
    """Conditional-mean coefficients, intercept first when present.

    In the search coordinates of the mean layer that owns them.
    """
    omega: float
    """Variance intercept, in levels or in logs depending on the family."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _ConditionalVarianceParameters(_VarianceParameters):
    """Structured parameters of a finite-order variance draw.

    Attributes:
        alpha: Coefficients on the shock magnitude.
        gamma: Asymmetry coefficients; empty for the symmetric case.
        beta: Persistence coefficients.
    """

    alpha: npt.NDArray[np.float64]
    """Coefficients on the shock magnitude, one per shock lag."""
    gamma: npt.NDArray[np.float64]
    """Asymmetry coefficients; empty for a symmetric model."""
    beta: npt.NDArray[np.float64]
    """Persistence coefficients, one per lag of the variance."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _FractionalVarianceParameters(_VarianceParameters):
    """Structured parameters of a fractionally integrated variance draw.

    Attributes:
        phi: Short-memory numerator weight, in ``(0, 1)``.
        d: Fractional integration order, in ``(0, 1)``.
        beta: Denominator weight, in ``(0, 1)``.
    """

    phi: float
    """Short-memory numerator weight, in ``(0, 1)``."""
    d: float
    """Fractional integration order of the variance, in ``(0, 1)``."""
    beta: float
    """Denominator weight, in ``(0, 1)``."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _SmoothTransitionParameters:
    """Structured parameters of a smooth-transition draw.

    The regime coefficients are absent by design: they are concentrated out and
    recovered by :meth:`_SmoothTransitionObjective.least_squares`.

    Attributes:
        gamma: Transition speed, strictly positive.
        threshold: Transition location, on the scale of the transition variable.
    """

    gamma: float
    """Transition speed, strictly positive."""
    threshold: float
    """Transition location, on the scale of the transition variable."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _StructuralParameters:
    """The variance-and-cycle parameter record of a structural model.

    Fields for components the specification excludes are ``None``; the
    builder and the objective agree on that convention, so the record is
    the single meeting point between the flat optimizer vector and the
    system matrices.

    Attributes:
        sigma2_irregular: Observation noise variance.
        sigma2_level: Level innovation variance, or ``None`` when the
            level is smooth (no own noise).
        sigma2_slope: Slope innovation variance, or ``None`` without a
            stochastic slope.
        cycle_rho: Cycle damping in ``(0, 1)``, or ``None``.
        cycle_freq: Cycle frequency in ``(0, pi)``, or ``None``.
        sigma2_cycle: Cycle innovation variance, or ``None``.
        sigma2_seasonal: Shared seasonal innovation variance, or ``None``.
    """

    sigma2_irregular: float
    """Observation noise variance."""
    sigma2_level: float | None
    """Level innovation variance; ``None`` when the level has no noise of its own."""
    sigma2_slope: float | None
    """Slope innovation variance; ``None`` without a stochastic slope."""
    cycle_rho: float | None
    """Cycle damping, in ``(0, 1)``; ``None`` without a cycle."""
    cycle_freq: float | None
    """Cycle frequency in radians, in ``(0, pi)``; ``None`` without a cycle."""
    sigma2_cycle: float | None
    """Cycle innovation variance, shared by both cycle states; ``None`` without a cycle."""
    sigma2_seasonal: float | None
    """Seasonal innovation variance, shared by every harmonic; ``None`` without a seasonal."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _NelsonSiegelParameters:
    """The parameter record of the dynamic Nelson-Siegel state space.

    Attributes:
        decay: The loading decay ``lambda``, strictly positive.
        mu: Factor means, shape ``(3,)``.
        ar: Diagonal factor persistences, shape ``(3,)``, each in
            ``(-1, 1)``.
        state_chol: Lower-Cholesky factor of the factor innovation
            covariance, shape ``(3, 3)``.
        obs_var: Per-maturity measurement variances, shape ``(p,)``.
    """

    decay: float
    """The loading decay ``lambda``, strictly positive."""
    mu: npt.NDArray[np.float64]
    """Means of the level, slope, and curvature factors, ``(3,)``."""
    ar: npt.NDArray[np.float64]
    """Diagonal factor persistences, ``(3,)``, each in ``(-1, 1)``."""
    state_chol: npt.NDArray[np.float64]
    """Lower-Cholesky factor of the factor innovation covariance, ``(3, 3)``."""
    obs_var: npt.NDArray[np.float64]
    """Measurement variances, one per maturity, ``(p,)``."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _StochasticVolatilityParameters:
    """The parameter record of the log-AR(1) stochastic-volatility model.

    The log variance follows ``h_{t+1} = mu + phi (h_t - mu) + sigma eta_t``
    and the observation is ``y_t = c + exp(h_t / 2) eps_t``.

    Attributes:
        mu: Unconditional mean of the log variance.
        phi: Persistence of the log variance, in ``(-1, 1)``.
        sigma2: Innovation variance of the log variance, strictly positive.
        mean: The observation mean ``c``, or ``0.0`` when the mean is fixed
            at zero.
        nu: Degrees of freedom of Student-t observation noise, or ``None``
            for Gaussian noise.
        rho: Correlation between the observation and log-variance
            innovations (leverage); ``0.0`` for none.
    """

    mu: float
    """Unconditional mean of the log variance."""
    phi: float
    """Persistence of the log variance, in ``(-1, 1)``."""
    sigma2: float
    """Innovation variance of the log variance, strictly positive."""
    mean: float
    """The observation mean ``c``; ``0.0`` when the mean is fixed at zero."""
    nu: float | None = None
    """Degrees of freedom of Student-t observation noise; ``None`` for Gaussian noise."""
    rho: float = 0.0
    """Correlation between the observation and log-variance innovations; ``0.0`` for none."""

    @property
    def stationary_variance(self) -> float:
        """The unconditional variance of the log variance.

        Returns:
            ``sigma2 / (1 - phi ** 2)``. The value is meaningful only for
            ``abs(phi) < 1``, which is not checked.
        """
        return float(self.sigma2 / (1.0 - self.phi**2))

    @property
    def heavy_tailed(self) -> bool:
        """Whether the observation noise is Student-t.

        Returns:
            ``True`` when ``nu`` is set.
        """
        return self.nu is not None

    @property
    def leveraged(self) -> bool:
        """Whether the innovations are correlated.

        Returns:
            ``True`` when ``rho`` is not exactly zero.
        """
        return self.rho != 0.0


@dataclass(frozen=True, kw_only=True, slots=True)
class _TrendVolatilityParameters:
    """The parameter record of the unobserved-components stochastic-volatility model.

    Stock and Watson's (2007) trend-inflation model: ``y_t = tau_t +
    exp(h_t / 2) eps_t``, ``tau_{t+1} = tau_t + exp(q_t / 2) eta_t``, with
    the two log variances random walks.

    Attributes:
        gamma2_irregular: Random-walk variance of the irregular's log
            variance ``h_t``.
        gamma2_trend: Random-walk variance of the trend innovation's log
            variance ``q_t``.
        h0: Initial log variance of the irregular.
        q0: Initial log variance of the trend innovation.
        tau0: Initial trend level.
    """

    gamma2_irregular: float
    """Random-walk step variance of ``h_t``, the irregular's log variance."""
    gamma2_trend: float
    """Random-walk step variance of ``q_t``, the trend innovation's log variance."""
    h0: float
    """Initial log variance of the irregular."""
    q0: float
    """Initial log variance of the trend innovation."""
    tau0: float
    """Initial trend level."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _DecayNelsonSiegelParameters:
    """The dynamic Nelson-Siegel record with a time-varying loading decay.

    Extends :class:`_NelsonSiegelParameters` with an AR(1) in the log
    decay: ``log lambda_{t+1} = log_decay_mean + decay_ar (log lambda_t -
    log_decay_mean) + decay_sd eta_t``. The constant-decay model is the
    limit ``decay_ar = 0``, ``decay_sd = 0``.

    Attributes:
        mu: Factor means, shape ``(3,)``.
        ar: Diagonal factor persistences, shape ``(3,)``, each in
            ``(-1, 1)``.
        state_chol: Lower-Cholesky factor of the factor innovation
            covariance, shape ``(3, 3)``.
        obs_var: Per-maturity measurement variances, shape ``(p,)``.
        log_decay_mean: Unconditional mean of the log decay.
        decay_ar: Persistence of the log decay, in ``(-1, 1)``.
        decay_sd: Innovation standard deviation of the log decay,
            nonnegative.
    """

    mu: npt.NDArray[np.float64]
    """Means of the level, slope, and curvature factors, ``(3,)``."""
    ar: npt.NDArray[np.float64]
    """Diagonal factor persistences, ``(3,)``, each in ``(-1, 1)``."""
    state_chol: npt.NDArray[np.float64]
    """Lower-Cholesky factor of the factor innovation covariance, ``(3, 3)``."""
    obs_var: npt.NDArray[np.float64]
    """Measurement variances, one per maturity, ``(p,)``."""
    log_decay_mean: float
    """Unconditional mean of the log decay."""
    decay_ar: float
    """Persistence of the log decay, in ``(-1, 1)``."""
    decay_sd: float
    """Innovation standard deviation of the log decay, nonnegative."""

    @property
    def decay(self) -> float:
        """The unconditional (median) decay ``exp(log_decay_mean)``.

        Returns:
            The decay at the mean of its log, strictly positive.
        """
        return float(np.exp(self.log_decay_mean))


@dataclass(frozen=True, kw_only=True, slots=True)
class _LongMemoryVolatilityParameters:
    """The parameter record of the long-memory stochastic-volatility model.

    The log variance is ``h_t = mu + v_t`` with ``(1 - phi L)(1 - L)**d v_t =
    eta_t``, an ARFIMA(1, d, 0) whose fractional order ``d`` in ``(0, 0.5)``
    is what makes volatility shocks decay hyperbolically rather than
    geometrically; the observation is ``y_t = c + exp(h_t / 2) eps_t``.

    Attributes:
        mu: Level of the log variance.
        d: Fractional differencing order of the log variance, in ``(-0.5, 0.5)``.
        sigma2: Innovation variance of the fractional noise, strictly positive.
        phi: Short-memory AR(1) coefficient of the log variance, in
            ``(-1, 1)``; ``0.0`` when the law is pure fractional noise.
        mean: The observation mean ``c``, or ``0.0`` when fixed at zero.
    """

    mu: float
    """Level of the log variance."""
    d: float
    """Fractional differencing order of the log variance."""
    sigma2: float
    """Innovation variance of the fractional noise, strictly positive."""
    phi: float
    """Short-memory AR(1) coefficient of the log variance; ``0.0`` for pure fractional noise."""
    mean: float
    """The observation mean ``c``; ``0.0`` when fixed at zero."""
