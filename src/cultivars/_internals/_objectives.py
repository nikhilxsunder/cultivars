# filepath: /src/cultivars/_internals/_objectives.py
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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from .._core import (
    _D_MAX,
    _LOG_2PI,
    _PENALTY,
    OptimizerMethod,
    OptimizerOptions,
    _arch_infinity_variance,
    _arch_infinity_weights,
    _gaussian_negloglik,
    _linear_variance_recursion,
    _log_variance_recursion,
    _midas_weights,
    _orthogonal_from_angles,
    companion_matrix,
    expand_ar,
    expand_ma,
    fractional_difference,
    ols,
    sigmoid,
    softplus,
    unpack_stationary,
)
from ..exceptions import NumericalError
from ._means import _MeanLayer
from ._models import _LinearGaussianStateSpaceModel
from ._parameters import (
    _AutoRegressionParameters,
    _BoxJenkinsParameters,
    _ConditionalVarianceParameters,
    _FractionalIntegrationParameters,
    _FractionalVarianceParameters,
    _SmoothTransitionParameters,
    _VarianceParameters,
)


class _Objective[P](ABC):
    """A scalar surface an optimizer minimizes, plus the map back to parameters.

    Subclasses hold the estimation context as fields and expose the criterion
    through :meth:`__call__`, so an instance is itself the callable handed to
    :func:`scipy.optimize.minimize`.

    The type parameter ``P`` is the structured parameter record the flat
    optimization vector maps onto, so ``unpack`` is statically typed per family
    rather than returning an anonymous tuple.

    Attributes:
        method: The :func:`scipy.optimize.minimize` algorithm this surface
            wants. Overridden by objectives whose criterion is not smooth.
        options: Algorithm options passed through to the optimizer, or ``None``
            for the scipy defaults.
    """

    method: ClassVar[OptimizerMethod] = "L-BFGS-B"
    options: ClassVar[OptimizerOptions | None] = None

    __slots__ = ()

    @abstractmethod
    def starts(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Return the starting points the optimizer should try.

        Returns:
            One or more unconstrained parameter vectors. Most surfaces return a
            single warm start; a multi-modal surface returns a grid, and
            :func:`_solve` keeps the best result.
        """

    @abstractmethod
    def unpack(self, theta: npt.NDArray[np.float64]) -> P:
        """Map an unconstrained vector to the structured parameters.

        Args:
            theta: The flat vector the optimizer searches over.

        Returns:
            The parameter record, with every reparameterization inverted.
        """

    @abstractmethod
    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """Evaluate the criterion being minimized.

        Args:
            theta: The flat vector the optimizer searches over.

        Returns:
            The criterion value, or :data:`_PENALTY` if the draw is
            inadmissible or the evaluation fails numerically.
        """


@dataclass(frozen=True, kw_only=True, slots=True)
class _AutoRegressionObjective(_Objective[_AutoRegressionParameters]):
    """Exact Kalman likelihood of an AR(p) in companion state-space form.

    The observation vector, selection vector, and observation covariance are
    all determined by the first unit vector ``e_1``, so only ``e_1`` and the
    identity are stored; the rest are reshaped views built on demand.

    Attributes:
        y: The endogenous series, used at full length.
        order: Autoregressive order ``p``.
        has_const: Whether a mean intercept is estimated.
        state_unit: The first unit vector ``e_1`` of length ``p``.
        identity: The ``p`` by ``p`` identity, used to solve for the stationary
            mean of the state when a constant is present.
        theta0: The warm start, from the CSS fit.
    """

    y: npt.NDArray[np.float64]
    order: int
    has_const: bool
    state_unit: npt.NDArray[np.float64]
    identity: npt.NDArray[np.float64]
    theta0: npt.NDArray[np.float64]

    @property
    def design(self) -> npt.NDArray[np.float64]:
        """The observation matrix ``Z``, which selects the first state."""
        return self.state_unit.reshape(1, self.order)

    @property
    def selection(self) -> npt.NDArray[np.float64]:
        """The selection matrix ``R``, which loads the shock on the first state."""
        return self.state_unit.reshape(self.order, 1)

    @property
    def obs_cov(self) -> npt.NDArray[np.float64]:
        """The observation covariance ``H``, identically zero for an AR."""
        return np.zeros((1, 1), dtype=np.float64)

    def starts(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Return the single CSS warm start."""
        return (self.theta0,)

    def unpack(self, theta: npt.NDArray[np.float64]) -> _AutoRegressionParameters:
        """Split the vector into intercept, AR block, and log variance.

        Args:
            theta: ``[const?, psi_1..psi_p, log sigma2]``.

        Returns:
            The parameter record, with ``psi`` mapped back through the
            partial-autocorrelation transform so the AR block is stationary.
        """
        p = self.order
        offset = 1 if self.has_const else 0
        return _AutoRegressionParameters(
            const=float(theta[0]) if self.has_const else 0.0,
            ar_params=unpack_stationary(theta[offset : offset + p]),
            sigma2=float(np.exp(theta[offset + p])),
        )

    def state_space(self, parameters: _AutoRegressionParameters) -> _LinearGaussianStateSpaceModel:
        """Build the companion state-space form at the given parameters.

        The initial state is the stationary mean implied by the intercept, so
        the likelihood is exact rather than conditional on a diffuse start.

        Args:
            parameters: An unpacked draw.

        Returns:
            The state-space model whose likelihood is the AR(p) likelihood.
        """
        transition = companion_matrix(parameters.ar_params)
        state_intercept = parameters.const * self.state_unit
        initial_state = (
            np.linalg.solve(self.identity - transition, state_intercept)
            if self.has_const
            else np.zeros(self.order, dtype=np.float64)
        )
        return _LinearGaussianStateSpaceModel(
            self.design,
            self.obs_cov,
            transition,
            self.selection,
            np.array([[parameters.sigma2]], dtype=np.float64),
            state_intercept=state_intercept,
            initial_state=initial_state,
        )

    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """Return the negative exact log-likelihood."""
        try:
            return -self.state_space(self.unpack(theta)).loglikelihood(self.y)
        except (NumericalError, np.linalg.LinAlgError):
            return _PENALTY


@dataclass(frozen=True, kw_only=True, slots=True)
class _BoxJenkinsObjective(_Objective[_BoxJenkinsParameters]):
    """Exact Kalman likelihood of a multiplicative seasonal ARMA with regressors.

    Differencing is applied once when the objective is built, so the surface is
    defined on the stationary modeling series ``w`` and the optimizer never
    re-differences.

    Attributes:
        w: The differenced modeling series.
        design_x: The deterministic and exogenous block, differenced alongside
            ``w``; may have zero columns.
        order: Non-seasonal ``(p, d, q)``.
        seasonal_order: Seasonal ``(P, D, Q, s)``.
        theta0: The warm start.
    """

    w: npt.NDArray[np.float64]
    design_x: npt.NDArray[np.float64]
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    theta0: npt.NDArray[np.float64]

    @property
    def k_beta(self) -> int:
        """Width of the regression block."""
        return self.design_x.shape[1]

    @property
    def bounds(self) -> npt.NDArray[np.int_]:
        """Cumulative block boundaries within the flat parameter vector."""
        p, _d, q = self.order
        cap_p, _cap_d, cap_q, _s = self.seasonal_order
        return np.cumsum([self.k_beta, p, cap_p, q, cap_q])

    def starts(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Return the single warm start."""
        return (self.theta0,)

    def unpack(self, theta: npt.NDArray[np.float64]) -> _BoxJenkinsParameters:
        """Split the vector into regression, AR, seasonal AR, MA, seasonal MA, variance.

        Args:
            theta: The flat vector, blocked in the order given by
                :attr:`bounds`.

        Returns:
            The parameter record. Both MA blocks are negated after the
            stationarity transform, which is what makes them invertible in the
            observation-equation sign convention.
        """
        p = self.order[0]
        q = self.order[2]
        cap_p = self.seasonal_order[0]
        cap_q = self.seasonal_order[2]
        idx = self.bounds
        return _BoxJenkinsParameters(
            beta=theta[: idx[0]],
            ar_params=unpack_stationary(theta[idx[0] : idx[1]]) if p else np.zeros(0),
            seasonal_ar_params=(
                unpack_stationary(theta[idx[1] : idx[2]]) if cap_p else np.zeros(0)
            ),
            ma_params=-unpack_stationary(theta[idx[2] : idx[3]]) if q else np.zeros(0),
            seasonal_ma_params=(
                -unpack_stationary(theta[idx[3] : idx[4]]) if cap_q else np.zeros(0)
            ),
            sigma2=float(np.exp(theta[idx[4]])),
        )

    def obs_intercept(self, parameters: _BoxJenkinsParameters) -> npt.NDArray[np.float64]:
        """The regression component that enters the observation intercept.

        Args:
            parameters: An unpacked draw.

        Returns:
            ``design_x @ beta``, or zeros when there is no regression block.
        """
        if not self.k_beta:
            return np.zeros(self.w.shape[0], dtype=np.float64)
        return self.design_x @ parameters.beta

    def state_space(self, parameters: _BoxJenkinsParameters) -> _LinearGaussianStateSpaceModel:
        """Build the Harvey ARMA state-space form at the given parameters.

        Args:
            parameters: An unpacked draw.

        Returns:
            The state-space model for the multiplied polynomials
            ``phi(L)Phi(L**s)`` and ``theta(L)Theta(L**s)``.
        """
        s = self.seasonal_order[3]
        return _LinearGaussianStateSpaceModel._from_arma(
            expand_ar(parameters.ar_params, parameters.seasonal_ar_params, s),
            expand_ma(parameters.ma_params, parameters.seasonal_ma_params, s),
            parameters.sigma2,
            self.obs_intercept(parameters),
        )

    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """Return the negative exact log-likelihood on the differenced series."""
        try:
            return -self.state_space(self.unpack(theta)).loglikelihood(self.w)
        except (NumericalError, np.linalg.LinAlgError):
            return _PENALTY


@dataclass(frozen=True, kw_only=True, slots=True)
class _FractionalIntegrationObjective(_Objective[_FractionalIntegrationParameters]):
    """Exact likelihood of an ARFIMA(p, d, q) on the fractionally differenced series.

    ``d`` and the short-memory block are estimated jointly: the fractional
    filter is re-applied at every draw, since the differenced series depends on
    ``d``.

    Attributes:
        y: The series, at full length.
        p: Short-memory AR order.
        q: Short-memory MA order.
        estimate_mean: Whether a mean is estimated.
        truncation: Length of the fractional-difference filter.
        theta0: The warm start, with ``d`` seeded from a local Whittle estimate.
    """

    y: npt.NDArray[np.float64]
    p: int
    q: int
    estimate_mean: bool
    truncation: int
    theta0: npt.NDArray[np.float64]

    @property
    def offset(self) -> int:
        """Index of ``d`` in the flat vector: ``1`` with a mean, ``0`` without."""
        return 1 if self.estimate_mean else 0

    def starts(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Return the single warm start."""
        return (self.theta0,)

    def unpack(self, theta: npt.NDArray[np.float64]) -> _FractionalIntegrationParameters:
        """Split the vector into mean, ``d``, AR, MA, and variance.

        Args:
            theta: ``[mu?, atanh(d / _D_MAX), psi_ar, psi_ma, log sigma2]``.

        Returns:
            The parameter record. ``d`` comes back through ``tanh`` scaled by
            ``_D_MAX``, which keeps the search unconstrained while confining
            ``d`` to the range where the filter converges.
        """
        i_d = self.offset
        i_ar = i_d + 1
        i_ma = i_ar + self.p
        i_sigma = i_ma + self.q
        return _FractionalIntegrationParameters(
            mean=float(theta[0]) if self.estimate_mean else 0.0,
            d=_D_MAX * float(np.tanh(theta[i_d])),
            ar_params=unpack_stationary(theta[i_ar:i_ma]) if self.p else np.zeros(0),
            ma_params=-unpack_stationary(theta[i_ma:i_sigma]) if self.q else np.zeros(0),
            sigma2=float(np.exp(theta[i_sigma])),
        )

    def differenced(self, parameters: _FractionalIntegrationParameters) -> npt.NDArray[np.float64]:
        """Apply the fractional filter to the demeaned series.

        Args:
            parameters: An unpacked draw.

        Returns:
            The short-memory series the ARMA block is fitted to.
        """
        return fractional_difference(
            self.y - parameters.mean, parameters.d, truncation=self.truncation
        )

    def state_space(
        self, parameters: _FractionalIntegrationParameters
    ) -> _LinearGaussianStateSpaceModel:
        """Build the ARMA state-space form for the short-memory block."""
        return _LinearGaussianStateSpaceModel._from_arma(
            parameters.ar_params,
            parameters.ma_params,
            parameters.sigma2,
            np.zeros(self.y.shape[0], dtype=np.float64),
        )

    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """Return the negative log-likelihood of the differenced series."""
        parameters = self.unpack(theta)
        try:
            w = self.differenced(parameters)
            return -self.state_space(parameters).loglikelihood(w)
        except (NumericalError, np.linalg.LinAlgError, ValueError):
            return _PENALTY


@dataclass(frozen=True, kw_only=True, slots=True)
class _VarianceObjective[P: _VarianceParameters](_Objective[P]):
    """Gaussian likelihood of a conditional-mean model with a time-varying variance.

    Holds the estimation context every conditional-variance family shares: the
    series being explained, the conditional-mean design block, and the
    pre-sample variance the recursion is seeded with. The criterion is written
    once here, because it is the same for every family once the variance path
    exists — what differs is how the path is produced and which draws are
    admissible, and those are the two abstract hooks.

    Attributes:
        mean: The conditional-mean layer, which owns its own data and turns a
            parameter block into residuals. Delegating rather than holding a
            design matrix is what lets one criterion serve a regression mean
            and a moving-average recursion alike.
        backcast: Pre-sample variance, an exponentially weighted mean square.
        theta0: The warm start.
    """

    mean: _MeanLayer
    backcast: float
    theta0: npt.NDArray[np.float64]

    @property
    def target(self) -> npt.NDArray[np.float64]:
        """The series the mean model explains, as the layer trimmed it."""
        return self.mean.target

    @property
    def k_mean(self) -> int:
        """Width of the conditional-mean parameter block."""
        return self.mean.n_parameters

    def starts(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Return the single warm start."""
        return (self.theta0,)

    def residuals(self, parameters: P) -> npt.NDArray[np.float64]:
        """Mean residuals the variance recursion is driven by."""
        return self.mean.residuals(parameters.mean)

    def fitted(self, parameters: P) -> npt.NDArray[np.float64]:
        """Conditional means implied by the mean block."""
        return self.mean.fitted(parameters.mean)

    @abstractmethod
    def variance_path(
        self, resid: npt.NDArray[np.float64], parameters: P
    ) -> npt.NDArray[np.float64]:
        """Run the family's variance recursion.

        Args:
            resid: Mean residuals.
            parameters: An unpacked draw.

        Returns:
            The conditional-variance path, in levels for every family.
        """

    @abstractmethod
    def is_admissible(self, parameters: P) -> bool:
        """Whether the draw implies a usable variance path.

        Args:
            parameters: An unpacked draw.

        Returns:
            ``True`` if the draw should be evaluated rather than penalized.
        """

    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """Return the joint negative Gaussian log-likelihood."""
        parameters = self.unpack(theta)
        if not self.is_admissible(parameters):
            return _PENALTY
        resid = self.residuals(parameters)
        return _gaussian_negloglik(resid, self.variance_path(resid, parameters))


@dataclass(frozen=True, kw_only=True, slots=True)
class _ConditionalVarianceObjective(_VarianceObjective[_ConditionalVarianceParameters]):
    """Joint Gaussian likelihood of an AR mean with a finite-order variance.

    Attributes:
        vol: Family key selecting the recursion and the reparameterization.
        p: Order of the shock-magnitude block.
        o: Order of the asymmetry block.
        q: Order of the persistence block.
    """

    vol: str
    p: int
    o: int
    q: int

    def unpack(self, theta: npt.NDArray[np.float64]) -> _ConditionalVarianceParameters:
        """Split the vector into the mean block and the variance block.

        Args:
            theta: ``[mean, omega_raw, alpha_raw, gamma_raw?, beta_raw]``.

        Returns:
            The parameter record, with the family's positivity transforms
            already inverted.
        """
        p, o, q = self.p, self.o, self.q
        mean, v = theta[: self.k_mean], theta[self.k_mean :]
        if self.vol == "GARCH":
            return _ConditionalVarianceParameters(
                mean=mean,
                omega=float(np.exp(v[0])),
                alpha=softplus(v[1 : 1 + p]),
                gamma=np.zeros(0),
                beta=softplus(v[1 + p : 1 + p + q]),
            )
        if self.vol == "GJR":
            return _ConditionalVarianceParameters(
                mean=mean,
                omega=float(np.exp(v[0])),
                alpha=softplus(v[1 : 1 + p]),
                gamma=v[1 + p : 1 + p + o],
                beta=softplus(v[1 + p + o : 1 + p + o + q]),
            )
        return _ConditionalVarianceParameters(
            mean=mean,
            omega=float(v[0]),
            alpha=v[1 : 1 + p],
            gamma=v[1 + p : 1 + p + o],
            beta=v[1 + p + o : 1 + p + o + q],
        )

    def variance_path(
        self,
        resid: npt.NDArray[np.float64],
        parameters: _ConditionalVarianceParameters,
    ) -> npt.NDArray[np.float64]:
        """Run the family's variance recursion.

        Args:
            resid: Mean residuals.
            parameters: An unpacked draw.

        Returns:
            The conditional-variance path, in levels for every family.
        """
        recursion = _log_variance_recursion if self.vol == "EGARCH" else _linear_variance_recursion
        return recursion(
            resid,
            parameters.omega,
            parameters.alpha,
            parameters.gamma,
            parameters.beta,
            self.backcast,
        )

    def is_admissible(self, parameters: _ConditionalVarianceParameters) -> bool:
        """Whether the draw implies a covariance-stationary variance process.

        Args:
            parameters: An unpacked draw.

        Returns:
            ``True`` if the persistence sum is inside the unit circle. For the
            log-variance family only the persistence block matters; for the
            level families the asymmetry block counts at half weight, its
            unconditional frequency.
        """
        if self.vol == "EGARCH":
            return bool(abs(parameters.beta.sum()) < 0.999)
        weight = parameters.alpha.sum() + 0.5 * parameters.gamma.sum() + parameters.beta.sum()
        return bool(weight < 0.999)


@dataclass(frozen=True, kw_only=True, slots=True)
class _FractionalVarianceObjective(_VarianceObjective[_FractionalVarianceParameters]):
    """Gaussian likelihood of a fractionally integrated variance process.

    Attributes:
        truncation: Number of infinite-order weights retained.
    """

    truncation: int

    def unpack(self, theta: npt.NDArray[np.float64]) -> _FractionalVarianceParameters:
        """Split the vector into the mean block, intercept, and three weights."""
        k = self.k_mean
        return _FractionalVarianceParameters(
            mean=theta[:k],
            omega=float(np.exp(theta[k])),
            phi=sigmoid(float(theta[k + 1])),
            d=sigmoid(float(theta[k + 2])),
            beta=sigmoid(float(theta[k + 3])),
        )

    def variance_path(
        self,
        resid: npt.NDArray[np.float64],
        parameters: _FractionalVarianceParameters,
    ) -> npt.NDArray[np.float64]:
        """Run the truncated infinite-order variance recursion."""
        return _arch_infinity_variance(
            resid,
            parameters.omega,
            parameters.phi,
            parameters.d,
            parameters.beta,
            self.backcast,
            self.truncation,
        )

    def is_admissible(self, parameters: _FractionalVarianceParameters) -> bool:
        """Whether every retained weight is non-negative.

        A negative weight implies a negative conditional variance somewhere in
        the sample, so the draw is rejected on a short prefix before the full
        recursion runs.

        Args:
            parameters: An unpacked draw.

        Returns:
            ``True`` if no weight in the prefix is materially negative.
        """
        lam = _arch_infinity_weights(
            parameters.phi, parameters.d, parameters.beta, min(self.truncation, 200)
        )
        return not bool(np.any(lam < -1e-6))


@dataclass(frozen=True, kw_only=True, slots=True)
class _SmoothTransitionObjective(_Objective[_SmoothTransitionParameters]):
    """Concentrated sum of squares of a smooth-transition autoregression.

    Only the two transition parameters are searched; conditional on them the
    model is linear, so the regime coefficients come from one least-squares
    solve. The surface is flat in ``gamma`` far from the data and has local
    minima in the threshold, which is why the algorithm is derivative-free and
    the search is multi-start.

    The transition variable is standardized by ``scale`` before entering the
    transition function, so ``gamma`` is scale-free and comparable across
    series.

    Attributes:
        target: The series the model explains, trimmed by the lag and the delay.
        design: The per-regime regressor block, intercept first.
        z: The transition variable, aligned with ``target``.
        scale: Standard deviation of ``z``.
        transition: ``"logistic"`` or ``"exponential"``.
        seeds: Starting points, crossed over a grid of transition speeds and two
            candidate thresholds.
    """

    method: ClassVar[OptimizerMethod] = "Nelder-Mead"
    options: ClassVar[OptimizerOptions | None] = {
        "xatol": 1e-4,
        "fatol": 1e-7,
        "maxiter": 2000,
    }

    target: npt.NDArray[np.float64]
    design: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]
    scale: float
    transition: str
    seeds: tuple[npt.NDArray[np.float64], ...]

    def starts(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Return the multi-start grid."""
        return self.seeds

    def unpack(self, theta: npt.NDArray[np.float64]) -> _SmoothTransitionParameters:
        """Map ``[log gamma, c]`` to the transition parameters."""
        return _SmoothTransitionParameters(gamma=float(np.exp(theta[0])), threshold=float(theta[1]))

    def weights(self, parameters: _SmoothTransitionParameters) -> npt.NDArray[np.float64]:
        """Evaluate the transition function.

        Args:
            parameters: An unpacked draw.

        Returns:
            Weights in ``[0, 1]``, monotone in the standardized transition
            variable for the logistic form and symmetric about the threshold
            for the exponential form. The exponent is clipped so that an
            extreme ``gamma`` saturates rather than overflowing.
        """
        u = (self.z - parameters.threshold) / self.scale
        if self.transition == "logistic":
            return 1.0 / (1.0 + np.exp(-np.clip(parameters.gamma * u, -50.0, 50.0)))
        return 1.0 - np.exp(-np.clip(parameters.gamma * u**2, 0.0, 50.0))

    def regressors(self, parameters: _SmoothTransitionParameters) -> npt.NDArray[np.float64]:
        """Stack the two weighted regime blocks side by side."""
        g = self.weights(parameters)
        return np.column_stack([self.design * (1.0 - g)[:, None], self.design * g[:, None]])

    def least_squares(
        self, parameters: _SmoothTransitionParameters
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Solve for the regime coefficients at fixed transition parameters.

        Args:
            parameters: An unpacked draw.

        Returns:
            A tuple ``(ssr, beta, resid)``; ``beta`` stacks the lower-regime
            coefficients ahead of the upper-regime coefficients.
        """
        regressors = self.regressors(parameters)
        beta, ssr = ols(regressors, self.target)
        return ssr, beta, self.target - regressors @ beta

    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """Return the concentrated sum of squared residuals."""
        return self.least_squares(self.unpack(theta))[0]


@dataclass(frozen=True, kw_only=True, slots=True)
class _VectorSmoothTransitionObjective(_Objective[_SmoothTransitionParameters]):
    """Concentrated Gaussian likelihood of a smooth-transition VAR.

    The multivariate counterpart of :class:`_SmoothTransitionObjective`, with
    one substantive change: the concentrated criterion is the log-determinant
    of the residual covariance rather than a sum of squares, because with more
    than one equation the sum of squares would weight the equations by their
    innovation variances and let the noisiest series choose the transition.
    Conditional on ``(gamma, c)`` the model is linear, so the regime
    coefficients come from one multivariate least-squares solve and only the
    two transition parameters are searched -- derivative-free and multi-start,
    for the same flat-in-``gamma``, multimodal-in-``c`` reasons as the
    univariate surface.

    Attributes:
        target: ``(nobs, k)`` block being explained, trimmed for lags and
            delay.
        design: The per-regime regressor block, deterministic terms first.
        z: The transition variable, aligned with ``target``.
        scale: Standard deviation of ``z``; ``gamma`` is per unit of it.
        transition: ``"logistic"`` or ``"exponential"``.
        seeds: Starting points over ``[log gamma, c]``.
    """

    method: ClassVar[OptimizerMethod] = "Nelder-Mead"
    options: ClassVar[OptimizerOptions | None] = {
        "xatol": 1e-4,
        "fatol": 1e-7,
        "maxiter": 2000,
    }

    target: npt.NDArray[np.float64]
    design: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]
    scale: float
    transition: str
    seeds: tuple[npt.NDArray[np.float64], ...]

    def starts(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Return the multi-start grid."""
        return self.seeds

    def unpack(self, theta: npt.NDArray[np.float64]) -> _SmoothTransitionParameters:
        """Map ``[log gamma, c]`` to the transition parameters."""
        return _SmoothTransitionParameters(gamma=float(np.exp(theta[0])), threshold=float(theta[1]))

    def weights(self, parameters: _SmoothTransitionParameters) -> npt.NDArray[np.float64]:
        """Evaluate the transition function, exponent clipped against overflow."""
        u = (self.z - parameters.threshold) / self.scale
        if self.transition == "logistic":
            return 1.0 / (1.0 + np.exp(-np.clip(parameters.gamma * u, -50.0, 50.0)))
        return 1.0 - np.exp(-np.clip(parameters.gamma * u**2, 0.0, 50.0))

    def regressors(self, parameters: _SmoothTransitionParameters) -> npt.NDArray[np.float64]:
        """Stack the two weighted regime blocks side by side, lower first."""
        g = self.weights(parameters)
        return np.column_stack([self.design * (1.0 - g)[:, None], self.design * g[:, None]])

    def concentrated(
        self, parameters: _SmoothTransitionParameters
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Solve for the regime coefficients at fixed transition parameters.

        Args:
            parameters: An unpacked draw.

        Returns:
            A tuple ``(logdet, coef, resid)``: the log-determinant of the
            residual covariance (``inf`` when it is singular), the stacked
            coefficient matrix with the lower-regime rows first, and the
            residuals.
        """
        regressors = self.regressors(parameters)
        coef: npt.NDArray[np.float64] = np.linalg.lstsq(regressors, self.target, rcond=None)[0]
        resid = self.target - regressors @ coef
        sign, logdet = np.linalg.slogdet(resid.T @ resid / self.target.shape[0])
        return (float(logdet) if sign > 0 else np.inf), coef, resid

    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """Return the concentrated negative log-likelihood, up to constants."""
        return self.concentrated(self.unpack(theta))[0]


@dataclass(frozen=True, kw_only=True, slots=True)
class _MidasProfileObjective(_Objective[npt.NDArray[np.float64]]):
    """Concentrated Gaussian likelihood of a MIDAS vector autoregression.

    Only the lag-polynomial parameters are searched: conditional on them, each
    high-frequency history collapses to one regressor column per series, the
    model is linear, and every coefficient comes from a single least-squares
    solve with the innovation covariance concentrated out at its maximum. The
    surface is smooth but can be multi-modal in the decay parameter, which is
    why the algorithm is derivative-free and the search is multi-start.

    Attributes:
        target: ``(nobs, k)`` low-frequency block being explained.
        base_design: ``(nobs, d + k p)`` deterministic and endogenous-lag
            columns, fixed across the search.
        windows: ``(nobs, lags, m)`` high-frequency history behind each
            low-frequency row, most recent sub-period first.
        seeds: Starting points, two polynomial parameters per series.
    """

    method: ClassVar[OptimizerMethod] = "Nelder-Mead"
    options: ClassVar[OptimizerOptions | None] = {
        "xatol": 1e-6,
        "fatol": 1e-9,
        "maxiter": 5000,
    }

    target: npt.NDArray[np.float64]
    base_design: npt.NDArray[np.float64]
    windows: npt.NDArray[np.float64]
    seeds: tuple[npt.NDArray[np.float64], ...]

    def starts(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Return the multi-start grid over the polynomial parameters."""
        return self.seeds

    def unpack(self, theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map the flat vector to one ``(theta_1, theta_2)`` row per series."""
        return np.asarray(theta, dtype=np.float64).reshape(-1, 2)

    def compressed(self, theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """One regressor column per high-frequency series at these parameters.

        Args:
            theta: The flat vector the optimizer searches over.

        Returns:
            An ``(nobs, m)`` block of weighted sub-period sums.
        """
        lags = int(self.windows.shape[1])
        weights = np.column_stack([_midas_weights(row, lags) for row in self.unpack(theta)])
        return np.einsum("tjm,jm->tm", self.windows, weights)

    def design(self, theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Assemble the full regressor matrix at these parameters."""
        return np.column_stack([self.base_design, self.compressed(theta)])

    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """Negative log-likelihood with coefficients and covariance concentrated out."""
        if not np.all(np.isfinite(theta)):
            return _PENALTY
        design = self.design(theta)
        nobs, k = self.target.shape
        try:
            coef: npt.NDArray[np.float64] = np.linalg.lstsq(design, self.target, rcond=None)[0]
            resid = self.target - design @ coef
            sign, logdet = np.linalg.slogdet(resid.T @ resid / nobs)
        except np.linalg.LinAlgError:
            return _PENALTY
        if sign <= 0 or not np.isfinite(logdet):
            return _PENALTY
        return 0.5 * nobs * (k * _LOG_2PI + float(logdet) + k)


@dataclass(frozen=True, kw_only=True, slots=True)
class _ShortRunObjective(_Objective[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]):
    """Concentrated Gaussian likelihood of an AB-model contemporaneous structure.

    ``A u_t = B e_t`` with unit-variance shocks implies ``Sigma_u =
    A^{-1} B B' A^{-1}'``, and the reduced-form coefficients concentrate out,
    so the surface depends on the data only through the estimated innovation
    covariance and the sample size. The criterion is the exact negative
    log-likelihood, kept on the likelihood scale rather than divided through
    by the sample, so that twice the gap to the saturated model is directly
    the over-identification statistic.

    Attributes:
        sigma: The ``(k, k)`` estimated innovation covariance.
        nobs: Effective sample size behind ``sigma``.
        a_base: Fixed values of ``A``, zeros in the free cells.
        a_free: Free cell coordinates of ``A``.
        b_base: Fixed values of ``B``, zeros in the free cells.
        b_free: Free cell coordinates of ``B``.
    """

    sigma: npt.NDArray[np.float64]
    nobs: int
    a_base: npt.NDArray[np.float64]
    a_free: tuple[tuple[int, int], ...]
    b_base: npt.NDArray[np.float64]
    b_free: tuple[tuple[int, int], ...]

    def build(
        self, theta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Fill the free cells of ``A`` and ``B`` from the flat vector."""
        values = np.asarray(theta, dtype=np.float64)
        a = self.a_base.copy()
        for position, (i, j) in enumerate(self.a_free):
            a[i, j] = values[position]
        b = self.b_base.copy()
        offset = len(self.a_free)
        for position, (i, j) in enumerate(self.b_free):
            b[i, j] = values[offset + position]
        return a, b

    def starts(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Warm starts: free ``B`` cells from the Cholesky factor, free ``A`` at zero."""
        factor = np.linalg.cholesky(self.sigma)
        warm = np.concatenate(
            [
                np.zeros(len(self.a_free), dtype=np.float64),
                np.array([factor[i, j] for i, j in self.b_free], dtype=np.float64),
            ]
        )
        return (warm, 0.5 * warm + 0.01)

    def unpack(
        self, theta: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Map the flat vector back to the ``(A, B)`` pair."""
        return self.build(theta)

    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """Exact negative log-likelihood at these contemporaneous matrices."""
        if not np.all(np.isfinite(theta)):
            return _PENALTY
        a, b = self.build(theta)
        sign_a, logdet_a = np.linalg.slogdet(a)
        sign_b, logdet_b = np.linalg.slogdet(b)
        if sign_a == 0 or sign_b == 0:
            return _PENALTY
        try:
            inner = np.linalg.solve(b, a)
        except np.linalg.LinAlgError:
            return _PENALTY
        trace = float(np.trace(inner.T @ inner @ self.sigma))
        if not np.isfinite(trace):
            return _PENALTY
        k = self.sigma.shape[0]
        llf = self.nobs * (-0.5 * k * _LOG_2PI + float(logdet_a) - float(logdet_b) - 0.5 * trace)
        return -llf


@dataclass(frozen=True, kw_only=True, slots=True)
class _MixedHorizonObjective(_Objective[npt.NDArray[np.float64]]):
    """Feasibility surface for zero restrictions split across two horizons.

    The candidate impact matrix is ``chol(Sigma) Q`` with ``Q`` a rotation, so
    the covariance is reproduced identically at every point of the search and
    the criterion measures only how far the declared cells of the impact and
    long-run matrices sit from their fixed values. At an exactly identified
    pattern whose rank condition holds, the minimum is zero and the minimizer
    is the identification; a strictly positive minimum is the rank condition
    failing, and the caller reports it as exactly that.

    Attributes:
        impact_factor: ``(k, k)`` Cholesky factor of the innovation
            covariance, possibly sign-flipped in its last column when the
            caller explores the reflection component of the orthogonal group.
        long_factor: ``(k, k)`` long-run matrix times the same factor.
        impact_cells: ``(row, column, value)`` restrictions on the impact
            matrix.
        long_cells: ``(row, column, value)`` restrictions on the long-run
            matrix.
    """

    options: ClassVar[OptimizerOptions | None] = {
        "maxiter": 20000,
        "ftol": 1e-18,
        "gtol": 1e-14,
    }

    impact_factor: npt.NDArray[np.float64]
    long_factor: npt.NDArray[np.float64]
    impact_cells: tuple[tuple[int, int, float], ...]
    long_cells: tuple[tuple[int, int, float], ...]

    def starts(self) -> tuple[npt.NDArray[np.float64], ...]:
        """A small deterministic fan across the rotation group."""
        size = self.impact_factor.shape[0]
        count = size * (size - 1) // 2
        return (
            np.zeros(count, dtype=np.float64),
            np.full(count, 0.7, dtype=np.float64),
            np.full(count, -0.7, dtype=np.float64),
            np.linspace(-1.2, 1.2, count) if count > 1 else np.array([1.2]),
        )

    def unpack(self, theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map the angles to the rotation they compose."""
        return _orthogonal_from_angles(theta, self.impact_factor.shape[0])

    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """Sum of squared restriction violations at this rotation."""
        if not np.all(np.isfinite(theta)):
            return _PENALTY
        rotation = _orthogonal_from_angles(theta, self.impact_factor.shape[0])
        impact = self.impact_factor @ rotation
        long_run = self.long_factor @ rotation
        violation = 0.0
        for i, j, value in self.impact_cells:
            violation += (impact[i, j] - value) ** 2
        for i, j, value in self.long_cells:
            violation += (long_run[i, j] - value) ** 2
        return violation


@dataclass(frozen=True, kw_only=True, slots=True)
class _CoDiagonalObjective(_Objective[npt.NDArray[np.float64]]):
    """Joint approximate diagonalization of regime covariances by one rotation.

    With more than two variance regimes the constant impact matrix is
    over-identified: no single rotation exactly diagonalizes every whitened
    regime covariance in sample, and the minimized off-diagonal energy is the
    data's verdict on the constant-impact assumption. The two-regime problem
    has a closed form and never reaches this surface; the caller warm-starts
    the search there, so the angles parameterize a refinement around the
    exactly-solvable pair.

    Attributes:
        targets: The whitened regime covariances to diagonalize jointly, each
            ``(k, k)`` symmetric.
    """

    options: ClassVar[OptimizerOptions | None] = {
        "maxiter": 20000,
        "ftol": 1e-18,
        "gtol": 1e-14,
    }

    targets: tuple[npt.NDArray[np.float64], ...]

    def starts(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Start at the warm point the caller built into the targets."""
        size = self.targets[0].shape[0]
        count = size * (size - 1) // 2
        return (
            np.zeros(count, dtype=np.float64),
            np.full(count, 0.3, dtype=np.float64),
            np.full(count, -0.3, dtype=np.float64),
        )

    def unpack(self, theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Map the angles to the rotation they compose."""
        return _orthogonal_from_angles(theta, self.targets[0].shape[0])

    def __call__(self, theta: npt.NDArray[np.float64]) -> float:
        """Total squared off-diagonal energy across regimes at this rotation."""
        if not np.all(np.isfinite(theta)):
            return _PENALTY
        rotation = _orthogonal_from_angles(theta, self.targets[0].shape[0])
        energy = 0.0
        for target in self.targets:
            transformed = rotation.T @ target @ rotation
            energy += float(np.sum(transformed**2) - np.sum(np.diagonal(transformed) ** 2))
        return energy
