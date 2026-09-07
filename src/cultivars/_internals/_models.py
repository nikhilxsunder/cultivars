# filepath: /src/cultivars/_internals/_models.py
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

"""Protected base classes for model specifications and fitted results.

Two hierarchies live here, and they are deliberately separate.

Results inherit a single linear chain of *field* bases -- ``_Result`` then
``_MeanResult`` -- because multiple slotted dataclass bases that each declare
fields raise ``TypeError: multiple bases have instance lay-out conflict``.
Anything optional is a *behavior* mixin with ``__slots__ = ()`` and no fields
of its own; the concrete result declares the attribute the mixin reads.

Model specifications inherit ``_ModelBase``, which owns ``endog`` validation
and the length check. Every family base in this subpackage descends from it,
so no public constructor re-implements either.

A family base also owns its estimation: ``_build_objective`` assembles the
optimizer surface from the validated specification, and ``_fit_family`` runs it
and packs the raw outputs into a record from
:mod:`cultivars._internals._fits`. The fit records carry no estimation logic of
their own, so the specification is never destructured into a foreign call
signature.

These bases implement the structural contracts in
:mod:`cultivars._core._protocols` without importing them: the protocols are
duck-typed, so the relationship is checked by ``isinstance`` at runtime and by
``mypy`` statically, with no import edge in either direction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import ClassVar, cast

import numpy as np
import numpy.typing as npt
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.stats as sst
from scipy.optimize import linprog, minimize
from scipy.special import gammaln

from .._core import (
    _D_MAX,
    _DEFAULT_GRID,
    _DEFAULT_MAX_ITER,
    _DEFAULT_STARTS,
    _DEFAULT_TOL,
    _DEFAULT_TRIM,
    _DEFAULT_TRUNCATION,
    _LOG_2PI,
    _ROW_SUM_ATOL,
    _STUDENT_DF_GRID,
    _TINY,
    _UNRESTRICTED_TREND,
    ClosedSystemResult,
    CointegrationTrend,
    Frequency,
    Mean,
    Method,
    PanelEffects,
    Penalty,
    Transition,
    Trend,
    Vol,
    _draw_inverse_gamma,
    _draw_inverse_wishart,
    _ForwardPass,
    aggregation_weights,
    combined_difference,
    concentrated_gaussian,
    conditional_design,
    deterministic_columns,
    ergodic_distribution,
    ewma_mean_square,
    fractional_difference,
    inv_softplus,
    lag_matrix,
    local_whittle_d,
    minnesota_scales,
    n_deterministic,
    ols,
    pack_stationary,
    psd_sqrt,
    simulate_cointegration_null,
    validate_aligned,
    validate_choice,
    validate_endog,
    validate_endog_matrix,
    validate_exog,
    validate_exog_matrix,
    validate_open_interval,
    validate_order,
    validate_order_tuple,
    validate_panel,
    validate_transition,
)
from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._engines import MeanFunctionEngine, NumpyMLPEngine
from ._filters import hamilton_filter
from ._fits import (
    _AutoRegressionFit,
    _BoxJenkinsFit,
    _ExogenousVectorAutoRegressionFit,
    _FractionalIntegrationFit,
    _FractionalVarianceFit,
    _MarkovSwitchingFit,
    _NeuralAutoRegressionFit,
    _NeuralThresholdFit,
    _ShortMemoryVarianceFit,
    _SmoothTransitionFit,
    _ThresholdFit,
    _TimeVaryingFit,
    _VectorAutoRegressionFit,
    _VectorConjugateFit,
    _VectorErrorCorrectionFit,
    _VectorFunctionalFit,
    _VectorGibbsFit,
    _VectorGraphicalFit,
    _VectorHierarchicalFit,
    _VectorMarkovSwitchingFit,
    _VectorQuantileFit,
    _VectorSmoothTransitionFit,
    _VectorSparseFit,
    _VectorStudentFit,
    _VectorThresholdFit,
    _VectorVolatilityFit,
)
from ._inferences import _CoefficientInference
from ._layouts import _ParameterLayout
from ._means import _ARMAMean, _LinearMean, _MeanLayer
from ._moments import _CointegrationMoments, _VectorMoments
from ._objectives import (
    _AutoRegressionObjective,
    _BoxJenkinsObjective,
    _ConditionalVarianceObjective,
    _FractionalIntegrationObjective,
    _FractionalVarianceObjective,
    _SmoothTransitionObjective,
    _VectorSmoothTransitionObjective,
)
from ._posteriors import (
    _ConjugatePosterior,
)
from ._priors import _AdaptivePrior, _NoPrior, _Prior, _PriorContext
from ._results import (
    _DurbinKoopmanSmootherResult,
    _FilterResult,
    _HamiltonFilterResult,
    _KalmanFilterResult,
    _KimFilterResult,
    _KimSmootherResult,
    _ParticleFilterResult,
    _ParticleSmootherResult,
    _RtsSmootherResult,
    _SmootherResult,
)
from ._samplers import _draw_volatility_path
from ._selections import _LagOrderSelection
from ._smoothers import kim_smoother
from ._solvers import (
    _conjugate_posterior,
    _fista_penalized,
    _maximize_likelihood,
    _solve,
    posterior_coefficients,
)
from ._states import _ExpectationMaximizationState, _VectorExpectationMaximizationState
from ._tests import _JohansenRankTest, _StabilityTest


class _BaseModel[R](ABC):
    """Root of every model specification, fitting to a result of type ``R``.

    Owns ``endog`` coercion and validation so that a malformed series produces
    an identical error regardless of which model was constructed.

    Args:
        endog: The observed univariate series (1-D array-like).

    Raises:
        DimensionError: If ``endog`` is not one-dimensional.
        NumericalError: If ``endog`` contains non-finite values.
    """

    __slots__ = ("_endog",)

    def __init__(self, endog: npt.ArrayLike) -> None:
        """Validate and store the endogenous series."""
        self._endog = validate_endog(endog)

    @property
    def endog(self) -> npt.NDArray[np.float64]:
        """The validated endogenous series."""
        return self._endog

    def _ensure_length(self, min_len: int, label: str) -> None:
        """Reject a series too short to identify the specification.

        Args:
            min_len: Minimum admissible number of observations.
            label: Human-readable specification name for the error message.

        Raises:
            DimensionError: If the series is shorter than ``min_len``.
        """
        if self._endog.shape[0] < min_len:
            raise DimensionError(
                f"series of length {self._endog.shape[0]} is too short for {label} "
                f"(need at least {min_len})."
            )

    @abstractmethod
    def fit(self) -> R:
        """Estimate the model and return its result object."""
        ...


class _UnivariateModel[R](_BaseModel[R]):
    """Base for single-series models. Reserved for univariate-only behavior."""

    __slots__ = ()


class _StateSpaceModel[F: _FilterResult, S: _SmootherResult](ABC):
    """The operations a state-space model must expose.

    Generic in its filter and smoother result types so that implementations
    using a fundamentally different inference object -- the Hamilton filter's
    regime probabilities rather than the Kalman filter's state and covariance
    -- satisfy the same contract without either widening the return type or
    inheriting fields they cannot populate.

    Parameterization (system matrices vs. transition functions) is
    deliberately NOT part of this contract; only the operations and the
    state/observation dimensions are.
    """

    @property
    @abstractmethod
    def k_endog(self) -> int: ...
    @property
    @abstractmethod
    def k_states(self) -> int: ...
    @abstractmethod
    def filter(self, y: npt.ArrayLike) -> F: ...
    @abstractmethod
    def smooth(self, y: npt.ArrayLike) -> S: ...
    @abstractmethod
    def loglikelihood(self, y: npt.ArrayLike) -> float: ...


class _LinearGaussianStateSpaceModel(
    _StateSpaceModel[_KalmanFilterResult, _DurbinKoopmanSmootherResult]
):
    """A linear-Gaussian state-space model.

    Args:
        design: Observation matrix ``Z``; shape ``(p, m)`` or ``(n, p, m)``.
        obs_cov: Observation noise covariance ``H``; shape ``(p, p)`` or
            ``(n, p, p)``.
        transition: State transition matrix ``T``; shape ``(m, m)`` or
            ``(n, m, m)``.
        selection: State disturbance selection ``R``; shape ``(m, r)`` or
            ``(n, m, r)``.
        state_cov: State disturbance covariance ``Q``; shape ``(r, r)`` or
            ``(n, r, r)``.
        obs_intercept: Observation intercept ``d``; shape ``(p,)`` or ``(n, p)``.
            Defaults to zero.
        state_intercept: State intercept ``c``; shape ``(m,)`` or ``(n, m)``.
            Defaults to zero.
        initial_state: Prior mean ``a_1``; shape ``(m,)``. Defaults to zero.
        initial_state_cov: Prior covariance ``P_1``; shape ``(m, m)``. Defaults
            to the stationary covariance when the model is time-invariant and
            stable, otherwise a large diffuse covariance.

    Raises:
        DimensionError: If any system matrix has an inconsistent shape or rank.
        NumericalError: If any system matrix contains non-finite values.
    """

    def __init__(
        self,
        design: npt.ArrayLike,
        obs_cov: npt.ArrayLike,
        transition: npt.ArrayLike,
        selection: npt.ArrayLike,
        state_cov: npt.ArrayLike,
        *,
        obs_intercept: npt.ArrayLike | None = None,
        state_intercept: npt.ArrayLike | None = None,
        initial_state: npt.ArrayLike | None = None,
        initial_state_cov: npt.ArrayLike | None = None,
    ) -> None:
        """Initialize the linear-Gaussian state-space model."""
        self._Z = self._validate_matrix(design, "design", ndim_static=2)
        self._H = self._validate_matrix(obs_cov, "obs_cov", ndim_static=2)
        self._T = self._validate_matrix(transition, "transition", ndim_static=2)
        self._R = self._validate_matrix(selection, "selection", ndim_static=2)
        self._Q = self._validate_matrix(state_cov, "state_cov", ndim_static=2)

        p = int(self._Z.shape[-2])
        m = int(self._Z.shape[-1])
        r = int(self._R.shape[-1])

        if self._T.shape[-2:] != (m, m):
            raise DimensionError(
                f"transition must be ({m}, {m}) to match the state dimension; "
                f"got trailing shape {self._T.shape[-2:]}."
            )
        if self._H.shape[-2:] != (p, p):
            raise DimensionError(
                f"obs_cov must be ({p}, {p}) to match the observation dimension; "
                f"got trailing shape {self._H.shape[-2:]}."
            )
        if self._R.shape[-2] != m:
            raise DimensionError(
                f"selection must have {m} rows to match the state dimension; "
                f"got trailing shape {self._R.shape[-2:]}."
            )
        if self._Q.shape[-2:] != (r, r):
            raise DimensionError(
                f"state_cov must be ({r}, {r}) to match the disturbance "
                f"dimension; got trailing shape {self._Q.shape[-2:]}."
            )

        self._p = p
        self._m = m
        self._r = r

        self._d = self._validate_vector(obs_intercept, "obs_intercept", p)
        self._c = self._validate_vector(state_intercept, "state_intercept", m)

        if initial_state is None:
            self._a1 = np.zeros(m, dtype=np.float64)
        else:
            a1 = np.asarray(initial_state, dtype=np.float64)
            if a1.shape != (m,):
                raise DimensionError(f"initial_state must have shape ({m},); got {a1.shape}.")
            self._a1 = a1

        if initial_state_cov is None:
            self._P1 = self._default_initial_cov()
        else:
            p1 = np.asarray(initial_state_cov, dtype=np.float64)
            if p1.shape != (m, m):
                raise DimensionError(
                    f"initial_state_cov must have shape ({m}, {m}); got {p1.shape}."
                )
            self._P1 = p1

    # -- validation helpers ------------------------------------------------

    @staticmethod
    def _validate_matrix(
        value: npt.ArrayLike, name: str, *, ndim_static: int
    ) -> npt.NDArray[np.float64]:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim not in (ndim_static, ndim_static + 1):
            raise DimensionError(
                f"{name} must be rank {ndim_static} (time-invariant) or "
                f"{ndim_static + 1} (time-varying); got rank {arr.ndim}."
            )
        if not np.all(np.isfinite(arr)):
            raise NumericalError(f"{name} contains non-finite values.")
        return arr

    def _validate_vector(
        self, value: npt.ArrayLike | None, name: str, dim: int
    ) -> npt.NDArray[np.float64]:
        if value is None:
            return np.zeros(dim, dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim not in (1, 2) or arr.shape[-1] != dim:
            raise DimensionError(
                f"{name} must be rank 1 ({dim},) or rank 2 (n, {dim}); got shape {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise NumericalError(f"{name} contains non-finite values.")
        return arr

    def _default_initial_cov(self) -> npt.NDArray[np.float64]:
        if self._T.ndim == 2 and self._R.ndim == 2 and self._Q.ndim == 2:
            eig = np.abs(np.linalg.eigvals(self._T))
            if float(eig.max(initial=0.0)) < 1.0 - 1e-10:
                rqr = self._R @ self._Q @ self._R.T
                return np.asarray(sla.solve_discrete_lyapunov(self._T, rqr), dtype=np.float64)
        return np.eye(self._m, dtype=np.float64) * 1e6

    # -- accessors ---------------------------------------------------------

    @property
    def k_endog(self) -> int:
        """Observation dimension ``p`` (number of observed series)."""
        return self._p

    @property
    def k_states(self) -> int:
        """State dimension ``m``."""
        return self._m

    @property
    def k_posdef(self) -> int:
        """State disturbance dimension ``r``."""
        return self._r

    @staticmethod
    def _at(matrix: npt.NDArray[np.float64], t: int, static_ndim: int) -> npt.NDArray[np.float64]:
        """Return the system matrix at time ``t`` (broadcasting invariant ones)."""
        return matrix if matrix.ndim == static_ndim else matrix[t]

    def _prepare_data(self, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[1] != self._p:
            raise DimensionError(f"observations must have shape (n, {self._p}); got {arr.shape}.")
        return arr

    # -- forward pass ------------------------------------------------------

    def _forward(self, y: npt.NDArray[np.float64]) -> _ForwardPass:
        n, m = y.shape[0], self._m
        pred_a = np.zeros((n, m))
        pred_P = np.zeros((n, m, m))
        filt_a = np.zeros((n, m))
        filt_P = np.zeros((n, m, m))
        llc = np.zeros(n)
        obs_index: list[npt.NDArray[np.intp]] = []
        innovation: list[npt.NDArray[np.float64]] = []
        innovation_precision: list[npt.NDArray[np.float64]] = []
        obs_design: list[npt.NDArray[np.float64]] = []

        a = self._a1.copy()
        P = self._P1.copy()

        for t in range(n):
            pred_a[t] = a
            pred_P[t] = P

            mask = np.flatnonzero(np.isfinite(y[t]))
            obs_index.append(mask)

            if mask.size > 0:
                z_full = self._at(self._Z, t, 2)
                d_full = self._at(self._d, t, 1)
                h_full = self._at(self._H, t, 2)
                z_obs = z_full[mask]
                v = y[t][mask] - z_obs @ a - d_full[mask]
                f = z_obs @ P @ z_obs.T + h_full[np.ix_(mask, mask)]
                f = 0.5 * (f + f.T)
                sign, logdet = np.linalg.slogdet(f)
                if sign <= 0.0:
                    raise NumericalError(
                        f"innovation covariance is not positive-definite at t={t}."
                    )
                f_inv = np.asarray(np.linalg.inv(f), dtype=np.float64)
                pz = P @ z_obs.T
                gain = pz @ f_inv
                a_filt = a + gain @ v
                p_filt = P - gain @ pz.T
                p_filt = 0.5 * (p_filt + p_filt.T)
                llc[t] = -0.5 * (mask.size * _LOG_2PI + logdet + float(v @ f_inv @ v))
                innovation.append(np.asarray(v, dtype=np.float64))
                innovation_precision.append(f_inv)
                obs_design.append(z_obs)
            else:
                a_filt = a
                p_filt = P
                innovation.append(np.empty(0))
                innovation_precision.append(np.empty((0, 0)))
                obs_design.append(np.empty((0, m)))

            filt_a[t] = a_filt
            filt_P[t] = p_filt

            t_mat = self._at(self._T, t, 2)
            c_vec = self._at(self._c, t, 1)
            r_mat = self._at(self._R, t, 2)
            q_mat = self._at(self._Q, t, 2)
            a = t_mat @ a_filt + c_vec
            P = t_mat @ p_filt @ t_mat.T + r_mat @ q_mat @ r_mat.T
            P = 0.5 * (P + P.T)

        return _ForwardPass(
            pred_a,
            pred_P,
            filt_a,
            filt_P,
            llc,
            obs_index,
            innovation,
            innovation_precision,
            obs_design,
        )

    # -- public operations -------------------------------------------------

    def filter(self, y: npt.ArrayLike) -> _KalmanFilterResult:
        """Run the Kalman filter over the observation matrix.

        Args:
            y: Observations, shape ``(n,)`` for a univariate model or
                ``(n, p)``; ``numpy.nan`` marks a missing element.

        Returns:
            The :class:`_KalmanFilterResult` carrying predicted and filtered
            states with their covariances and the per-period likelihood.
        """
        data = self._prepare_data(y)
        fwd = self._forward(data)
        return _KalmanFilterResult(
            predicted_state=fwd.predicted_state,
            predicted_state_cov=fwd.predicted_state_cov,
            filtered_state=fwd.filtered_state,
            filtered_state_cov=fwd.filtered_state_cov,
            loglikelihood=float(fwd.loglik_contrib.sum()),
            loglikelihood_contributions=fwd.loglik_contrib,
        )

    def loglikelihood(self, y: npt.ArrayLike) -> float:
        """Return the total Gaussian log-likelihood of the data."""
        data = self._prepare_data(y)
        return float(self._forward(data).loglik_contrib.sum())

    def smooth(self, y: npt.ArrayLike) -> _DurbinKoopmanSmootherResult:
        """Run the Durbin-Koopman state smoother."""
        data = self._prepare_data(y)
        fwd = self._forward(data)
        n, m = data.shape[0], self._m

        smoothed_a = np.zeros((n, m))
        smoothed_P = np.zeros((n, m, m))
        r_vec = np.zeros(m)
        n_mat = np.zeros((m, m))

        for t in range(n - 1, -1, -1):
            t_mat = self._at(self._T, t, 2)
            z_obs = fwd.obs_design[t]
            if z_obs.shape[0] > 0:
                f_inv = fwd.innovation_precision[t]
                v = fwd.innovation[t]
                gain = t_mat @ fwd.predicted_state_cov[t] @ z_obs.T @ f_inv
                l_mat = t_mat - gain @ z_obs
                r_vec = z_obs.T @ f_inv @ v + l_mat.T @ r_vec
                n_mat = z_obs.T @ f_inv @ z_obs + l_mat.T @ n_mat @ l_mat
            else:
                r_vec = t_mat.T @ r_vec
                n_mat = t_mat.T @ n_mat @ t_mat

            smoothed_a[t] = fwd.predicted_state[t] + fwd.predicted_state_cov[t] @ r_vec
            v_cov = (
                fwd.predicted_state_cov[t]
                - fwd.predicted_state_cov[t] @ n_mat @ fwd.predicted_state_cov[t]
            )
            smoothed_P[t] = 0.5 * (v_cov + v_cov.T)

        return _DurbinKoopmanSmootherResult(
            smoothed_state=smoothed_a, smoothed_state_cov=smoothed_P
        )

    def _simulate_forward(
        self, n: int, rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        alpha = np.zeros((n, self._m))
        obs = np.zeros((n, self._p))
        p1_sqrt = psd_sqrt(self._P1)
        state = self._a1 + p1_sqrt @ rng.standard_normal(self._m)
        for t in range(n):
            z = self._at(self._Z, t, 2)
            d = self._at(self._d, t, 1)
            h = self._at(self._H, t, 2)
            alpha[t] = state
            obs[t] = z @ state + d + psd_sqrt(h) @ rng.standard_normal(self._p)
            t_mat = self._at(self._T, t, 2)
            c = self._at(self._c, t, 1)
            r_mat = self._at(self._R, t, 2)
            q = self._at(self._Q, t, 2)
            state = t_mat @ state + c + r_mat @ (psd_sqrt(q) @ rng.standard_normal(self._r))
        return alpha, obs

    def simulate(
        self, n: int, *, seed: int | np.random.Generator | None = None
    ) -> npt.NDArray[np.float64]:
        """Draw a length-``n`` observation path from the model."""
        if n < 1:
            raise DimensionError(f"simulate requires n >= 1; got {n}.")
        rng = np.random.default_rng(seed)
        _, obs = self._simulate_forward(n, rng)
        return obs

    def simulation_smoother(
        self,
        y: npt.ArrayLike,
        *,
        n_sims: int = 1,
        seed: int | np.random.Generator | None = None,
    ) -> npt.NDArray[np.float64]:
        """Draw states from ``p(alpha | y)`` via the Durbin-Koopman (2002) smoother.

        Returns:
            An array of shape ``(n_sims, n, m)`` of state draws.
        """
        if n_sims < 1:
            raise DimensionError(f"simulation_smoother requires n_sims >= 1; got {n_sims}.")
        data = self._prepare_data(y)
        n = data.shape[0]
        rng = np.random.default_rng(seed)
        missing = ~np.isfinite(data)

        smoothed = self.smooth(data).smoothed_state
        draws = np.zeros((n_sims, n, self._m))
        for s in range(n_sims):
            alpha_plus, y_plus = self._simulate_forward(n, rng)
            y_plus[missing] = np.nan
            smoothed_plus = self.smooth(y_plus).smoothed_state
            draws[s] = smoothed + (alpha_plus - smoothed_plus)
        return draws

    @classmethod
    def stationary_covariance(
        cls, transition: npt.ArrayLike, selection: npt.ArrayLike, state_cov: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Solve the discrete Lyapunov equation for the stationary state covariance.

        Args:
            transition: Time-invariant transition matrix ``T`` (must be stable).
            selection: Selection matrix ``R``.
            state_cov: State disturbance covariance ``Q``.

        Returns:
            The stationary covariance ``P`` solving ``P = T P T' + R Q R'``.

        Raises:
            NumericalError: If ``T`` is not stable (has an eigenvalue on or
                outside the unit circle).
        """
        t_mat = np.asarray(transition, dtype=np.float64)
        r_mat = np.asarray(selection, dtype=np.float64)
        q_mat = np.asarray(state_cov, dtype=np.float64)
        if float(np.abs(np.linalg.eigvals(t_mat)).max(initial=0.0)) >= 1.0 - 1e-12:
            raise NumericalError(
                "stationary_covariance requires a stable transition matrix "
                "(all eigenvalues strictly inside the unit circle)."
            )
        rqr = r_mat @ q_mat @ r_mat.T
        return np.asarray(sla.solve_discrete_lyapunov(t_mat, rqr), dtype=np.float64)

    @classmethod
    def _from_arma(
        cls,
        phi_star: npt.NDArray[np.float64],
        theta_star: npt.NDArray[np.float64],
        sigma2: float,
        obs_intercept: npt.NDArray[np.float64],
    ) -> _LinearGaussianStateSpaceModel:
        """Build the Harvey state-space form of an ARMA process.

        The state dimension is ``max(p, q + 1)``, which is the minimal realization:
        a larger companion would be observationally equivalent but would make the
        Kalman recursion carry redundant states.

        Args:
            phi_star: Expanded AR coefficients.
            theta_star: Expanded MA coefficients.
            sigma2: Innovation variance.
            obs_intercept: Per-observation mean shift from the regression block.

        Returns:
            The configured :class:`_LinearGaussianStateSpaceModel`.
        """
        r = max(phi_star.size, theta_star.size + 1)
        phi_full = np.zeros(r)
        phi_full[: phi_star.size] = phi_star
        transition = np.zeros((r, r))
        transition[:, 0] = phi_full
        for i in range(r - 1):
            transition[i, i + 1] = 1.0
        selection = np.zeros((r, 1))
        selection[0, 0] = 1.0
        selection[1 : 1 + theta_star.size, 0] = theta_star
        design = np.zeros((1, r))
        design[0, 0] = 1.0
        return cls(
            design,
            np.zeros((1, 1)),
            transition,
            selection,
            np.array([[sigma2]]),
            obs_intercept=obs_intercept.reshape(-1, 1),
        )

    @classmethod
    def mixed_frequency_state_space(
        cls,
        coefficients: npt.NDArray[np.float64],
        sigma_u: npt.NDArray[np.float64],
        *,
        kinds: Sequence[str],
        period: int,
        weights: Sequence[npt.ArrayLike | None] | None = None,
    ) -> _LinearGaussianStateSpaceModel:
        """Put a latent high-frequency autoregression into observable form.

        The state carries the companion of the high-frequency system, padded to at
        least ``period`` lags so that a flow reading can reach every sub-period it
        accumulates. The observation matrix is *constant*: a flow variable's row
        always computes its weighted sum, and a stock variable's row always selects
        the current sub-period. What varies is the data, not the model -- a variable
        that is not observed this sub-period is ``nan``, and the filter drops that
        row from the update by itself.

        Building it this way rather than with a time-varying observation matrix is
        what keeps the representation honest about its own content. There is one
        measurement equation per variable and it holds at every date; the calendar
        lives in the sample, where anyone can see it.

        Args:
            coefficients: ``(p, k, k)`` autoregressive matrices of the latent system.
            sigma_u: ``(k, k)`` innovation covariance of the latent system.
            kinds: One :data:`Frequency` per variable.
            period: Sub-periods per low-frequency period.
            weights: Optional per-variable flow weights; ``None`` entries take the
                default from :func:`aggregation_weights`.

        Returns:
            A configured :class:`_LinearGaussianStateSpaceModel` whose state is the
            latent path and whose observation is what the calendar reveals of it.

        Raises:
            DimensionError: If the coefficient stack and covariance disagree, or the
                kinds do not cover the variables.
            SpecificationError: If a kind or the period is unrecognized.
        """
        blocks = np.asarray(coefficients, dtype=np.float64)
        covariance = np.asarray(sigma_u, dtype=np.float64)
        if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
            raise DimensionError(f"coefficients must be (p, k, k); got shape {blocks.shape}.")
        order, size = int(blocks.shape[0]), int(blocks.shape[1])
        if covariance.shape != (size, size):
            raise DimensionError(
                f"sigma_u must be ({size}, {size}) to match the coefficients; "
                f"got {covariance.shape}."
            )
        labels = tuple(str(kind) for kind in kinds)
        if len(labels) != size:
            raise DimensionError(
                f"kinds must have one entry per variable ({size}); got {len(labels)}."
            )
        depth = max(order, period)
        width = size * depth
        transition = np.zeros((width, width), dtype=np.float64)
        for lag in range(order):
            transition[:size, lag * size : (lag + 1) * size] = blocks[lag]
        if depth > 1:
            transition[size:, : size * (depth - 1)] = np.eye(size * (depth - 1))
        selection = np.zeros((width, size), dtype=np.float64)
        selection[:size, :] = np.eye(size)
        design = np.zeros((size, width), dtype=np.float64)
        supplied = tuple(weights) if weights is not None else (None,) * size
        if len(supplied) != size:
            raise DimensionError(
                f"weights must have one entry per variable ({size}); got {len(supplied)}."
            )
        for index, kind in enumerate(labels):
            row = aggregation_weights(cast(Frequency, kind), period, weights=supplied[index])
            for sub in range(period):
                design[index, sub * size + index] = row[sub]
        return cls(
            design, np.zeros((size, size), dtype=np.float64), transition, selection, covariance
        )


class _AutoRegressionModel[R](_UnivariateModel[R]):
    """Specification surface for the autoregressive family.

    Args:
        endog: The endogenous series.
        order: Autoregressive order ``p``.
        trend: Deterministic specification (``"n"``, ``"c"``, ``"ct"``).
        method: ``"css"`` for conditional least squares, ``"exact"`` for exact
            maximum likelihood through the companion state-space form.

    Raises:
        SpecificationError: If the order is negative, the trend or method is
            unrecognized, or exact ML is requested with a linear trend.
        DimensionError: If the series is too short for the specification.
    """

    __slots__ = ("_method", "_order", "_trend")

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        trend: str = "c",
        method: str = "css",
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog)
        self._order = validate_order(order, "order")
        self._trend = validate_choice(trend, Trend, "trend")
        self._method = validate_choice(method, Method, "method")
        if self._method == "exact" and self._trend == "ct":
            raise SpecificationError(
                "exact ML with trend='ct' is not supported in this release; "
                "use method='css' for a linear trend."
            )
        self._ensure_length(self._order + 2, f"AR({self._order})")

    @property
    def order(self) -> int:
        """The autoregressive order."""
        return self._order

    @property
    def trend(self) -> str:
        """The deterministic specification."""
        return self._trend

    @property
    def method(self) -> str:
        """The estimator."""
        return self._method

    def _fit_css(self) -> _AutoRegressionFit:
        """Fit an AR(p) mean model by conditional least squares.

        Conditions on the first ``p`` observations, so the effective sample is
        ``n - p`` and the reported likelihood is the conditional one.

        Returns:
            The packed :class:`_AutoRegressionFit`.
        """
        y, order, trend = self.endog, self._order, self._trend
        target, regressors, eff = conditional_design(y, order, trend)
        beta = np.linalg.lstsq(regressors, target, rcond=None)[0]
        fitted = regressors @ beta
        resid = target - fitted
        sigma2 = float(resid @ resid) / eff
        n_det = n_deterministic(trend)
        const = float(beta[0]) if trend in ("c", "ct") else None
        trend_coeff = float(beta[1]) if trend == "ct" else None
        ar_params = np.asarray(beta[n_det:], dtype=np.float64)
        llf = -0.5 * eff * (_LOG_2PI + np.log(sigma2) + 1.0)
        return _AutoRegressionFit(
            fittedvalues=fitted,
            resid=resid,
            llf=float(llf),
            nobs=eff,
            n_params=order + n_det + 1,
            const=const,
            trend_coeff=trend_coeff,
            ar_params=ar_params,
            sigma2=sigma2,
        )

    def _build_objective(self) -> _AutoRegressionObjective:
        """Assemble the exact-ML surface for an AR(p).

        Warm-starts from the CSS fit. An explosive CSS estimate is discarded in
        favour of a zero start, since it would put the optimizer outside the
        stationary region the reparameterization assumes.

        Returns:
            The configured objective.
        """
        y, order, trend = self.endog, self._order, self._trend
        has_const = trend == "c"
        state_unit = np.zeros(order, dtype=np.float64)
        state_unit[0] = 1.0

        warm = self._fit_css()
        phi0 = warm.ar_params
        if not _StabilityTest.assess_stability(phi0).is_stable:
            phi0 = np.zeros(order, dtype=np.float64)
        psi0 = pack_stationary(phi0)
        log_sigma0 = np.log(warm.sigma2)
        theta0 = (
            np.concatenate([[warm.const or 0.0], psi0, [log_sigma0]])
            if has_const
            else np.concatenate([psi0, [log_sigma0]])
        )
        return _AutoRegressionObjective(
            y=y,
            order=order,
            has_const=has_const,
            state_unit=state_unit,
            identity=np.eye(order, dtype=np.float64),
            theta0=theta0,
        )

    def _fit_exact(self) -> _AutoRegressionFit:
        """Fit an AR(p) mean model by exact maximum likelihood.

        Returns:
            The packed fit, using the full sample.
        """
        y, order, trend = self.endog, self._order, self._trend
        objective = self._build_objective()
        parameters, llf = _maximize_likelihood(objective)
        fitted = objective.state_space(parameters).filter(y).predicted_state[:, 0].copy()
        return _AutoRegressionFit(
            fittedvalues=fitted,
            resid=y - fitted,
            llf=llf,
            nobs=y.shape[0],
            n_params=order + n_deterministic(trend) + 1,
            const=parameters.const if objective.has_const else None,
            trend_coeff=None,
            ar_params=parameters.ar_params,
            sigma2=parameters.sigma2,
        )

    def _fit_family(self) -> _AutoRegressionFit:
        """Dispatch to the estimator this specification selected."""
        return self._fit_css() if self._method == "css" else self._fit_exact()


class _FractionalIntegrationModel[R](_UnivariateModel[R]):
    """Specification surface for the fractionally integrated ARMA family.

    Args:
        endog: The series.
        order: Short-memory ``(p, q)``.
        trend: ``"c"`` to estimate a mean, ``"n"`` to omit it.
        truncation: Fractional-filter length; defaults to the sample size.

    Raises:
        SpecificationError: If an order is negative, the trend is unrecognized,
            or ``truncation`` is not positive.
        DimensionError: If the series is too short for the specification.
    """

    __slots__ = ("_const", "_p", "_q", "_truncation")

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: tuple[int, int],
        trend: str = "c",
        truncation: int | None = None,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog)
        self._p, self._q = validate_order_tuple(order, ("p", "q"))
        self._const = validate_choice(trend, Trend, "trend") == "c"
        self._truncation = (
            self.endog.shape[0]
            if truncation is None
            else validate_order(truncation, "truncation", minimum=1)
        )
        self._ensure_length(2 * (self._p + self._q) + 8, f"ARFIMA({self._p}, d, {self._q})")

    @property
    def order(self) -> tuple[int, int]:
        """The short-memory orders ``(p, q)``."""
        return (self._p, self._q)

    @property
    def truncation(self) -> int:
        """The fractional-filter length."""
        return self._truncation

    def _build_objective(self) -> _FractionalIntegrationObjective:
        """Assemble the joint-ML surface for an ARFIMA(p, d, q).

        Warm-starts ``d`` from a local Whittle estimate, falling back to zero if
        that estimator fails; the ARMA block starts at zero, since a
        short-memory warm start fitted before ``d`` is known tends to absorb the
        long memory.

        Returns:
            The configured objective.
        """
        y, p, q = self.endog, self._p, self._q
        estimate_mean, truncation = self._const, self._truncation
        try:
            d0 = float(np.clip(local_whittle_d(y)[0], -_D_MAX + 1e-3, _D_MAX - 1e-3))
        except (NumericalError, SpecificationError):
            d0 = 0.0
        mu0 = float(y.mean()) if estimate_mean else 0.0
        w0 = fractional_difference(y - mu0, d0, truncation=truncation)
        log_sigma0 = float(np.log(max(float(np.var(w0)), 1e-8)))

        parts: list[npt.NDArray[np.float64]] = []
        if estimate_mean:
            parts.append(np.array([mu0]))
        parts.extend(
            [
                np.array([float(np.arctanh(d0 / _D_MAX))]),
                np.zeros(p),
                np.zeros(q),
                np.array([log_sigma0]),
            ]
        )
        return _FractionalIntegrationObjective(
            y=y,
            p=p,
            q=q,
            estimate_mean=estimate_mean,
            truncation=truncation,
            theta0=np.concatenate(parts),
        )

    def _fit_family(self) -> _FractionalIntegrationFit:
        """Fit an ARFIMA(p, d, q) by joint maximum likelihood.

        Returns:
            The packed fit, on the fractionally differenced series.
        """
        y, p, q, estimate_mean = self.endog, self._p, self._q, self._const
        objective = self._build_objective()
        parameters, llf = _maximize_likelihood(objective)
        w = objective.differenced(parameters)
        fitted = objective.state_space(parameters).filter(w).predicted_state[:, 0]
        return _FractionalIntegrationFit(
            d=parameters.d,
            mean=parameters.mean if estimate_mean else None,
            ar_params=parameters.ar_params,
            ma_params=parameters.ma_params,
            sigma2=parameters.sigma2,
            resid=w - fitted,
            fittedvalues=fitted,
            llf=llf,
            nobs=y.shape[0],
            n_params=objective.offset + 1 + p + q + 1,
        )


class _BoxJenkinsModel[R](_UnivariateModel[R]):
    """Shared specification surface for the ARMA/ARIMA/SARIMA family.

    Args:
        endog: The endogenous series.
        order: Non-seasonal ``(p, d, q)``.
        seasonal_order: Seasonal ``(P, D, Q, s)``; defaults to no seasonal block.
        trend: Deterministic specification.
        exog: Optional exogenous regressors.

    Raises:
        SpecificationError: If any order is negative, or seasonal terms are
            requested with a period below 2.
        DimensionError: If the series is too short, or ``exog`` is misshapen.
    """

    __slots__ = (
        "_exog",
        "_order",
        "_seasonal",
        "_trend",
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str = "c",
        exog: npt.ArrayLike | None = None,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog)
        p, d, q = validate_order_tuple(order, ("p", "d", "q"))
        cap_p, cap_d, cap_q, s = validate_order_tuple(seasonal_order, ("P", "D", "Q", "s"))
        if (cap_p or cap_d or cap_q) and s < 2:
            raise SpecificationError(
                f"seasonal period s must be >= 2 when seasonal terms are present; got s={s}."
            )
        self._order = (p, d, q)
        self._seasonal = (cap_p, cap_d, cap_q, s)
        self._trend = validate_choice(trend, Trend, "trend")
        self._exog = validate_exog(exog, self.endog.shape[0])
        self._ensure_length(
            p + d + q + s * (cap_p + cap_d + cap_q) + 2,
            f"SARIMA{self._order}{self._seasonal}",
        )

    @property
    def order(self) -> tuple[int, int, int]:
        """The non-seasonal order ``(p, d, q)``."""
        return self._order

    @property
    def seasonal_order(self) -> tuple[int, int, int, int]:
        """The seasonal order ``(P, D, Q, s)``."""
        return self._seasonal

    @property
    def trend(self) -> str:
        """The deterministic specification."""
        return self._trend

    @property
    def exog(self) -> npt.NDArray[np.float64] | None:
        """The validated exogenous regressors, or ``None``."""
        return self._exog

    def _build_objective(self) -> _BoxJenkinsObjective:
        """Assemble the exact-ML surface for a seasonal ARIMA with regressors.

        Starting values come from a regression of the differenced series on the
        deterministic and exogenous block, then a short AR fit to those
        residuals; an explosive AR start is replaced by zeros.

        Returns:
            The configured objective, defined on the differenced series.
        """
        endog, exog = self.endog, self._exog
        order, seasonal_order, trend = self._order, self._seasonal, self._trend
        p, d, q = order
        cap_p, cap_d, cap_q, s = seasonal_order
        w = combined_difference(endog, d, cap_d, s)
        n_eff = w.shape[0]
        det = deterministic_columns(trend, n_eff)
        if exog is not None:
            exog_w = combined_difference(exog, d, cap_d, s) if (d or cap_d) else exog
            design_x = np.column_stack([det, exog_w]) if det.shape[1] else exog_w
        else:
            design_x = det
        k_beta = design_x.shape[1]

        if k_beta:
            beta0 = np.linalg.lstsq(design_x, w, rcond=None)[0]
            resid0 = w - design_x @ beta0
        else:
            beta0 = np.zeros(0)
            resid0 = w - w.mean()
        sigma2_0 = max(float(resid0 @ resid0) / n_eff, 1e-8)

        ar0 = np.zeros(p)
        if p:
            lag_mat = np.column_stack([resid0[p - i - 1 : n_eff - i - 1] for i in range(p)])
            try:
                ar0 = np.asarray(
                    np.linalg.lstsq(lag_mat, resid0[p:], rcond=None)[0], dtype=np.float64
                )
                if not _StabilityTest.assess_stability(ar0).is_stable:
                    ar0 = np.zeros(p)
            except np.linalg.LinAlgError:
                ar0 = np.zeros(p)

        theta0 = np.concatenate(
            [
                beta0,
                pack_stationary(ar0),
                np.zeros(cap_p),
                np.zeros(q),
                np.zeros(cap_q),
                [np.log(sigma2_0)],
            ]
        )
        return _BoxJenkinsObjective(
            w=w,
            design_x=design_x,
            order=order,
            seasonal_order=seasonal_order,
            theta0=theta0,
        )

    def _fit_family(self) -> _BoxJenkinsFit:
        """Fit a seasonal ARIMA with optional regressors by exact ML.

        Returns:
            The packed fit, on the differenced modeling series.
        """
        order, seasonal_order = self._order, self._seasonal
        objective = self._build_objective()
        parameters, llf = _maximize_likelihood(objective)
        intercept = objective.obs_intercept(parameters)
        state_space = objective.state_space(parameters)
        fitted = state_space.filter(objective.w).predicted_state[:, 0] + intercept
        p, _d, q = order
        cap_p, _cap_d, cap_q, _s = seasonal_order
        return _BoxJenkinsFit(
            ar_params=parameters.ar_params,
            ma_params=parameters.ma_params,
            seasonal_ar_params=parameters.seasonal_ar_params,
            seasonal_ma_params=parameters.seasonal_ma_params,
            beta=parameters.beta,
            sigma2=parameters.sigma2,
            resid=objective.w - fitted,
            fittedvalues=fitted,
            llf=llf,
            nobs=objective.w.shape[0],
            n_params=objective.k_beta + p + cap_p + q + cap_q + 1,
        )


class _ConditionalVarianceModel[R](_UnivariateModel[R]):
    """Specification surface shared by the conditional-variance group.

    Abstract in intent: it owns only the conditional-mean specification, which
    is the one choice every family in the group makes the same way. Variance
    orders, the family key, and the truncation lag belong to the subclasses
    that actually honour them.

    Args:
        endog: The series, typically returns or residuals.
        mean: ``"constant"`` or ``"zero"``.
        ar_lags: Conditional-mean autoregressive order.
        ma_lags: Conditional-mean moving-average order.

    Raises:
        SpecificationError: If ``mean`` is not a recognized choice or an order
            is negative.
    """

    __slots__ = ("_ar_lags", "_const", "_ma_lags")

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        mean: str = "constant",
        ar_lags: int = 0,
        ma_lags: int = 0,
    ) -> None:
        """Validate the conditional-mean specification and the data."""
        super().__init__(endog)
        self._const = validate_choice(mean, Mean, "mean") == "constant"
        self._ar_lags = validate_order(ar_lags, "ar_lags")
        self._ma_lags = validate_order(ma_lags, "ma_lags")

    @property
    def has_constant_mean(self) -> bool:
        """Whether a mean intercept is estimated."""
        return self._const

    @property
    def ar_lags(self) -> int:
        """The conditional-mean autoregressive order."""
        return self._ar_lags

    @property
    def ma_lags(self) -> int:
        """The conditional-mean moving-average order."""
        return self._ma_lags

    @property
    def mean_order(self) -> tuple[int, int]:
        """The conditional-mean order ``(ar_lags, ma_lags)``."""
        return (self._ar_lags, self._ma_lags)

    def _mean_layer(self) -> _MeanLayer:
        """Build the layer that turns a mean draw into residuals.

        A pure autoregressive mean stays on the regression layer rather than
        being routed through the ARMA recursion with a zero moving-average
        block. The two describe the same model, but the regression estimates
        the lag weights unconstrained while the recursion confines them to the
        stationary region, and switching an existing specification onto a
        constrained parameterization would change its answers for no gain --
        the recursion needs that constraint only because it feeds on its own
        output, which a matrix product does not.

        Returns:
            A :class:`_LinearMean` when ``ma_lags == 0``, else an
            :class:`_ARMAMean`.
        """
        endog, ar_lags = self.endog, self._ar_lags
        if self._ma_lags:
            return _ARMAMean(endog=endog, p=ar_lags, q=self._ma_lags, include_const=self._const)
        n_full = endog.shape[0]
        target = endog[ar_lags:]
        columns: list[npt.NDArray[np.float64]] = []
        if self._const:
            columns.append(np.ones(target.shape[0]))
        columns.extend(endog[ar_lags - i : n_full - i] for i in range(1, ar_lags + 1))
        design = (
            np.column_stack(columns)
            if columns
            else np.zeros((target.shape[0], 0), dtype=np.float64)
        )
        return _LinearMean(endog_target=target, design=design, include_const=self._const)


class _ShortMemoryVarianceModel[R](_ConditionalVarianceModel[R]):
    """Specification surface for the finite-order variance families.

    Args:
        endog: The series, typically returns or residuals.
        vol: Volatility family: ``"GARCH"``, ``"GJR"``, or ``"EGARCH"``.
        p: Order of the shock-magnitude block.
        o: Order of the asymmetry block.
        q: Order of the persistence block.
        ar_lags: Conditional-mean autoregressive order.
        ma_lags: Conditional-mean moving-average order. A non-zero value routes
            the mean through the ARMA recursion rather than the lag regression;
            see :meth:`_ConditionalVarianceModel._mean_layer`.
        mean: ``"constant"`` or ``"zero"``.

    Raises:
        SpecificationError: If an order is negative, ``GARCH`` is given a
            non-zero asymmetry order, or ``GJR``/``EGARCH`` is given ``o < 1``.
        DimensionError: If the series is too short for the specification.
    """

    __slots__ = ("_o", "_p", "_q", "_vol")

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        vol: str,
        p: int,
        o: int,
        q: int,
        ar_lags: int = 0,
        ma_lags: int = 0,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, mean=mean, ar_lags=ar_lags, ma_lags=ma_lags)
        self._vol = validate_choice(vol, Vol, "vol")
        self._p = validate_order(p, "p")
        self._o = validate_order(o, "o")
        self._q = validate_order(q, "q")
        if self._vol == "GARCH" and self._o != 0:
            raise SpecificationError("GARCH has no asymmetry term; set the asymmetry order o = 0.")
        if self._vol in ("GJR", "EGARCH") and self._o < 1:
            raise SpecificationError(f"{self._vol} requires an asymmetry order o >= 1.")
        self._ensure_length(
            max(self._p, self._o, self._q) + max(self._ar_lags, self._ma_lags) + 2,
            f"{self._vol}({self._p}, {self._o}, {self._q})",
        )

    @property
    def vol(self) -> str:
        """The volatility family."""
        return self._vol

    @property
    def order(self) -> tuple[int, int, int]:
        """The variance order ``(p, o, q)``."""
        return (self._p, self._o, self._q)

    def _build_objective(self) -> _ConditionalVarianceObjective:
        """Assemble the joint mean-and-variance surface.

        Returns:
            The configured objective.
        """
        p, o, q, vol = self._p, self._o, self._q, self._vol
        mean = self._mean_layer()
        mean0 = mean.start()
        resid0 = mean.residuals(mean0)
        var0 = max(float(np.var(resid0)), 1e-8)

        a_init, b_init, g_init = 0.05, 0.90, 0.05
        if vol == "GARCH":
            var_raw0 = np.concatenate(
                [
                    [np.log(var0 * (1 - a_init - b_init))],
                    [inv_softplus(a_init)] * p,
                    [inv_softplus(b_init)] * q,
                ]
            )
        elif vol == "GJR":
            var_raw0 = np.concatenate(
                [
                    [np.log(var0 * (1 - a_init - b_init - 0.5 * g_init))],
                    [inv_softplus(a_init)] * p,
                    [g_init] * o,
                    [inv_softplus(b_init)] * q,
                ]
            )
        else:
            var_raw0 = np.concatenate(
                [[np.log(var0) * (1 - 0.95)], [0.1] * p, [-0.05] * o, [0.95] * q]
            )

        return _ConditionalVarianceObjective(
            mean=mean,
            backcast=ewma_mean_square(resid0),
            vol=vol,
            p=p,
            o=o,
            q=q,
            theta0=np.concatenate([mean0, var_raw0]),
        )

    def _fit_family(self) -> _ShortMemoryVarianceFit:
        """Fit a GARCH, GJR, or EGARCH model by Gaussian maximum likelihood.

        Mean and variance parameters are estimated jointly rather than in two
        steps, so the reported likelihood is the true joint one.

        Returns:
            The packed fit.
        """
        vol = self._vol
        p, o, q = self._p, self._o, self._q
        objective = self._build_objective()
        parameters, llf = _maximize_likelihood(objective)
        fitted = objective.fitted(parameters)
        resid = objective.residuals(parameters)
        coefficients = objective.mean.unpack(parameters.mean)
        return _ShortMemoryVarianceFit(
            const=coefficients.const,
            omega=parameters.omega,
            vol=vol,
            ar_params=coefficients.ar,
            ma_params=coefficients.ma,
            alpha=parameters.alpha,
            gamma=parameters.gamma,
            beta=parameters.beta,
            conditional_variance=objective.variance_path(resid, parameters),
            resid=resid,
            fittedvalues=fitted,
            llf=llf,
            nobs=objective.target.shape[0],
            n_params=objective.k_mean + 1 + p + o + q,
        )


class _FractionalVarianceModel[R](_ConditionalVarianceModel[R]):
    """Specification surface for the fractionally integrated variance family.

    The variance order is fixed at ``(1, d, 1)``, so the only structural choice
    beyond the mean is how far the infinite-order representation is truncated.

    Args:
        endog: The series, typically returns or residuals.
        mean: ``"constant"`` or ``"zero"``.
        ar_lags: Conditional-mean autoregressive order.
        ma_lags: Conditional-mean moving-average order. A non-zero value routes
            the mean through the ARMA recursion rather than the lag regression;
            see :meth:`_ConditionalVarianceModel._mean_layer`.
        truncation: Infinite-order truncation lag.

    Raises:
        SpecificationError: If ``truncation`` is not positive.
        DimensionError: If the series is too short for the specification.
    """

    __slots__ = ("_truncation",)

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        mean: str = "constant",
        ar_lags: int = 0,
        ma_lags: int = 0,
        truncation: int = _DEFAULT_TRUNCATION,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, mean=mean, ar_lags=ar_lags, ma_lags=ma_lags)
        self._truncation = validate_order(truncation, "truncation", minimum=1)
        self._ensure_length(
            int(self._const) + max(self._ar_lags, self._ma_lags) + 4, "FIGARCH(1, d, 1)"
        )

    @property
    def truncation(self) -> int:
        """The infinite-order truncation lag."""
        return self._truncation

    def _build_objective(self) -> _FractionalVarianceObjective:
        """Assemble the surface for a fractionally integrated variance.

        Returns:
            The configured objective.
        """
        truncation = self._truncation
        mean = self._mean_layer()
        mean0 = mean.start()
        resid0 = mean.residuals(mean0)
        var0 = max(float(np.var(resid0)), 1e-8)
        return _FractionalVarianceObjective(
            mean=mean,
            backcast=ewma_mean_square(resid0),
            truncation=truncation,
            theta0=np.concatenate([mean0, [np.log(var0 * 0.4), -1.0, -0.2, 0.4]]),
        )

    def _fit_family(self) -> _FractionalVarianceFit:
        """Fit a FIGARCH(1, d, 1) model by Gaussian maximum likelihood.

        Returns:
            The packed fit.
        """
        objective = self._build_objective()
        parameters, llf = _maximize_likelihood(objective)
        fitted = objective.fitted(parameters)
        resid = objective.residuals(parameters)
        coefficients = objective.mean.unpack(parameters.mean)
        return _FractionalVarianceFit(
            const=coefficients.const,
            ar_params=coefficients.ar,
            ma_params=coefficients.ma,
            omega=parameters.omega,
            phi=parameters.phi,
            d=parameters.d,
            beta=parameters.beta,
            conditional_variance=objective.variance_path(resid, parameters),
            resid=resid,
            fittedvalues=fitted,
            llf=llf,
            nobs=objective.target.shape[0],
            n_params=objective.k_mean + 3,
        )


class _MeanFunctionModel[R](_UnivariateModel[R]):
    """Specification surface for neural mean-function models.

    Args:
        endog: The series.
        order: Autoregressive order.
        engine: Training backend; defaults to :class:`NumpyMLPEngine`.

    Raises:
        SpecificationError: If ``engine`` does not satisfy
            :class:`MeanFunctionEngine`, or the order is invalid.
    """

    __slots__ = (
        "_engine",
        "_order",
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        engine: MeanFunctionEngine | None = None,
    ) -> None:
        """Validate the specification, the engine, and the data."""
        super().__init__(endog)
        self._order = validate_order(order, "order", minimum=1)
        self._engine: MeanFunctionEngine = engine if engine is not None else NumpyMLPEngine()
        if not isinstance(self._engine, MeanFunctionEngine):
            raise SpecificationError(
                "engine must implement fit(features, target) -> MeanPredictor."
            )
        self._ensure_length(self._order + 3, f"AR-NN({self._order})")

    @property
    def order(self) -> int:
        """The autoregressive order."""
        return self._order

    @property
    def engine(self) -> MeanFunctionEngine:
        """The training backend."""
        return self._engine


class _NeuralAutoRegressionModel[R](_MeanFunctionModel[R]):
    __slots__ = ()

    def _fit_family(self) -> _NeuralAutoRegressionFit:
        """Fit a neural autoregression of the given order.

        The likelihood is Gaussian with the variance concentrated out, so
        ``n_params`` counts the learner's parameters plus that variance.

        Returns:
            The packed :class:`_NeuralAutoRegressionFit`.
        """
        y, order, engine = self.endog, self._order, self._engine
        target = y[order:]
        features = lag_matrix(y, order)
        predictor = engine.fit(features, target)
        fitted = predictor.predict(features)
        resid = target - fitted
        sigma2, llf = concentrated_gaussian(float(resid @ resid), target.shape[0])
        return _NeuralAutoRegressionFit(
            predictor=predictor,
            sigma2=sigma2,
            resid=resid,
            fittedvalues=fitted,
            llf=float(llf),
            nobs=target.shape[0],
            n_params=predictor.n_parameters + 1,
        )


class _NeuralThresholdModel[R](_MeanFunctionModel[R]):
    """Specification surface for two-regime neural threshold autoregressions.

    Args:
        endog: The series.
        order: Autoregressive order per regime.
        engine: Training backend, used once per regime.
        threshold_variable: External threshold variable, or ``None``.
        delay: Threshold delay.
        threshold: Fixed threshold, or ``None`` for the median.
        trim: Minimum regime share of the effective sample.
    """

    __slots__ = (
        "_delay",
        "_threshold",
        "_threshold_variable",
        "_trim",
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        engine: MeanFunctionEngine | None = None,
        threshold_variable: npt.ArrayLike | None = None,
        delay: int = 1,
        threshold: float | None = None,
        trim: float = 0.15,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, order=order, engine=engine)
        self._delay = validate_order(delay, "delay", minimum=1)
        self._threshold = None if threshold is None else float(threshold)
        self._trim = float(trim)
        self._threshold_variable = (
            None
            if threshold_variable is None
            else validate_aligned(threshold_variable, self.endog.shape[0], "threshold_variable")
        )
        self._ensure_length(2 * (self._order + 3) + self._delay, f"TAR-NN({self._order})")

    @property
    def delay(self) -> int:
        """The threshold delay."""
        return self._delay

    @property
    def self_exciting(self) -> bool:
        """Whether the threshold variable is a lag of the series itself."""
        return self._threshold_variable is None

    def _fit_family(self) -> _NeuralThresholdFit:
        """Fit a two-regime neural threshold autoregression.

        The threshold defaults to the median of the transition variable rather
        than being searched: with a nonlinear learner per regime, a grid search
        would retrain the network at every candidate split.

        Returns:
            The packed :class:`_NeuralThresholdFit`.

        Raises:
            NumericalError: If the split leaves a regime with too few observations.
        """
        y, order, engine = self.endog, self._order, self._engine
        threshold_variable, delay = self._threshold_variable, self._delay
        threshold, trim = self._threshold, self._trim
        n = y.shape[0]
        start = max(order, delay)
        target = y[start:]
        n_eff = target.shape[0]
        features = lag_matrix(y, order, start=start)
        base = threshold_variable if threshold_variable is not None else y
        z = base[start - delay : n - delay]
        r = float(np.median(z)) if threshold is None else float(threshold)
        lower = z <= r
        n_lo = int(lower.sum())
        n_hi = n_eff - n_lo
        if min(n_lo, n_hi) < max(2, int(trim * n_eff)):
            raise NumericalError(
                f"threshold {r} leaves a regime with too few observations "
                f"({n_lo} lower, {n_hi} upper)."
            )
        lower_predictor = engine.fit(features[lower], target[lower])
        upper_predictor = engine.fit(features[~lower], target[~lower])
        fitted = np.empty(n_eff, dtype=np.float64)
        fitted[lower] = lower_predictor.predict(features[lower])
        fitted[~lower] = upper_predictor.predict(features[~lower])
        resid = target - fitted
        ssr = float(resid @ resid)
        sigma2, llf = concentrated_gaussian(ssr, n_eff)
        return _NeuralThresholdFit(
            delay=delay,
            threshold=r,
            lower_predictor=lower_predictor,
            upper_predictor=upper_predictor,
            threshold_variable=threshold_variable,
            self_exciting=threshold_variable is None,
            sigma2=sigma2,
            ssr=ssr,
            n_lower=n_lo,
            n_upper=n_hi,
            resid=resid,
            fittedvalues=fitted,
            llf=float(llf),
            nobs=n_eff,
            n_params=lower_predictor.n_parameters + upper_predictor.n_parameters + 2,
        )


class _MarkovSwitchingStateSpaceModel(_StateSpaceModel[_HamiltonFilterResult, _KimSmootherResult]):
    """A parameterized Markov-switching observation model over a latent chain.

    The discrete counterpart of :class:`_LinearGaussianStateSpaceModel`, and its
    peer in every structural respect: both hold a fully specified system, both
    filter and smooth data handed to them, and neither estimates anything. What
    differs is the state space. A linear-Gaussian model carries a continuous
    state and propagates a mean and a covariance; this one carries a state drawn
    from ``{0, ..., K-1}`` and propagates a probability vector.

    That difference is why the filter is Hamilton's rather than Kalman's, and
    why the likelihood is *exact*. Conditional on the observed lags, the density
    ``Pr(y_t | S_t, y_{1..t-1})`` depends on ``S_t`` alone -- the regime enters
    contemporaneously through the intercept, not through the lags -- so the
    K-vector of filtered regime probabilities is a sufficient statistic and the
    sum over all ``K**T`` regime paths is carried out implicitly. Nothing here
    is approximated. That is a property of the intercept-switching form
    specifically: under Hamilton's original *mean*-switching parameterization
    the density at ``t`` depends on ``(S_t, ..., S_{t-p})``, and exact filtering
    then needs an augmented chain of ``K**(p+1)`` states.

    Holding this as an object rather than as a pair of loose recursions is what
    lets a fitted model be re-applied: the same instance that produced an
    estimate can filter a *different* series, which the free functions in
    :mod:`cultivars.state_space.regime_switching` cannot do because they never
    see the data. They take a density matrix; this class owns the map that
    produces one.

    Args:
        transition: Row-stochastic ``(K, K)`` matrix,
            ``transition[i, j] = Pr(S_t = j | S_{t-1} = i)``.
        intercepts: Per-regime intercepts, shape ``(K,)``.
        ar_params: Per-regime autoregressive coefficients, shape ``(K, p)``.
            A width of zero is a switching-mean model with no dynamics.
        variances: Per-regime innovation variances, shape ``(K,)``, all strictly
            positive.
        initial_prob: Distribution of ``S_1``, length ``K``. Defaults to the
            ergodic distribution of ``transition``.

    Raises:
        DimensionError: If the blocks do not agree on ``K``, or a block has the
            wrong rank.
        SpecificationError: If ``transition`` is not row-stochastic, a variance
            is not strictly positive, or ``initial_prob`` is not a distribution.
        NumericalError: If any block contains non-finite values.
    """

    __slots__ = ("_ar", "_c", "_k", "_p", "_pi0", "_sigma2", "_transition")

    def __init__(
        self,
        transition: npt.ArrayLike,
        intercepts: npt.ArrayLike,
        ar_params: npt.ArrayLike,
        variances: npt.ArrayLike,
        *,
        initial_prob: npt.ArrayLike | None = None,
    ) -> None:
        """Validate the switching system."""
        c = np.asarray(intercepts, dtype=np.float64)
        if c.ndim != 1:
            raise DimensionError(f"intercepts must be 1-D (K,); got shape {c.shape}.")
        k = int(c.shape[0])
        self._transition = validate_transition(transition, k)
        ar = np.asarray(ar_params, dtype=np.float64)
        if ar.ndim != 2 or ar.shape[0] != k:
            raise DimensionError(f"ar_params must be ({k}, p); got shape {ar.shape}.")
        sigma2 = np.asarray(variances, dtype=np.float64)
        if sigma2.shape != (k,):
            raise DimensionError(f"variances must have shape ({k},); got {sigma2.shape}.")
        for name, block in (("intercepts", c), ("ar_params", ar), ("variances", sigma2)):
            if not np.all(np.isfinite(block)):
                raise NumericalError(f"{name} contains non-finite values.")
        if np.any(sigma2 <= 0.0):
            raise SpecificationError(f"variances must be strictly positive; got {sigma2}.")

        if initial_prob is None:
            pi0 = ergodic_distribution(self._transition)
        else:
            pi0 = np.asarray(initial_prob, dtype=np.float64)
            if pi0.shape != (k,):
                raise DimensionError(f"initial_prob must have shape ({k},); got {pi0.shape}.")
            if np.any(pi0 < 0.0) or not np.isclose(pi0.sum(), 1.0, atol=_ROW_SUM_ATOL):
                raise SpecificationError("initial_prob must be a probability vector.")

        self._c = c
        self._ar = ar
        self._sigma2 = sigma2
        self._pi0 = pi0
        self._k = k
        self._p = int(ar.shape[1])

    @property
    def k_endog(self) -> int:
        """Observed dimension; one, since the chain drives a scalar series."""
        return 1

    @property
    def k_states(self) -> int:
        """Size of the state space: the number of regimes."""
        return self._k

    @property
    def n_regimes(self) -> int:
        """The number of regimes, named as the chain rather than as a dimension."""
        return self._k

    @property
    def order(self) -> int:
        """Autoregressive order of the per-regime observation equation."""
        return self._p

    @property
    def transition(self) -> npt.NDArray[np.float64]:
        """The row-stochastic regime transition matrix."""
        return self._transition

    @property
    def intercepts(self) -> npt.NDArray[np.float64]:
        """Per-regime intercepts."""
        return self._c

    @property
    def ar_params(self) -> npt.NDArray[np.float64]:
        """Per-regime autoregressive coefficients, shape ``(K, p)``."""
        return self._ar

    @property
    def variances(self) -> npt.NDArray[np.float64]:
        """Per-regime innovation variances."""
        return self._sigma2

    @property
    def initial_prob(self) -> npt.NDArray[np.float64]:
        """Distribution of the regime at the first modelled observation."""
        return self._pi0

    def effective_sample(
        self, y: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Split a series into the modelled target and its lag block.

        Args:
            y: The untrimmed series.

        Returns:
            A tuple ``(target, lags)``; ``target`` drops the first ``order``
            observations and ``lags`` is the ``(T_eff, p)`` matrix the
            conditional means are built from.

        Raises:
            DimensionError: If the series is not 1-D or is shorter than the
                autoregressive order.
        """
        series = np.asarray(y, dtype=np.float64)
        if series.ndim != 1:
            raise DimensionError(f"y must be one-dimensional; got shape {series.shape}.")
        if series.shape[0] <= self._p:
            raise DimensionError(
                f"a series of length {series.shape[0]} leaves no observations after "
                f"trimming {self._p} lags."
            )
        return series[self._p :], lag_matrix(series, self._p)

    def conditional_means(self, lags: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Per-regime conditional means ``E[y_t | S_t = j, lags]``.

        Args:
            lags: The ``(T_eff, p)`` lag block.

        Returns:
            An array of shape ``(T_eff, K)``, one column per regime.
        """
        if self._p == 0:
            return np.broadcast_to(self._c, (lags.shape[0], self._k)).copy()
        return self._c[None, :] + lags @ self._ar.T

    def log_densities(
        self, target: npt.NDArray[np.float64], means: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Gaussian log conditional densities, one column per regime.

        Args:
            target: The modelled observations, length ``T_eff``.
            means: Per-regime conditional means, shape ``(T_eff, K)``.

        Returns:
            An array of shape ``(T_eff, K)``. Logarithms rather than levels, so
            a regime that assigns a tiny density to an outlier underflows to a
            large negative number instead of to zero.
        """
        resid = target[:, None] - means
        return -0.5 * (_LOG_2PI + np.log(self._sigma2)[None, :] + resid**2 / self._sigma2[None, :])

    def density_matrix(
        self, target: npt.NDArray[np.float64], lags: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """The ``(T_eff, K)`` log-density matrix the recursions consume.

        Exposed alongside :meth:`filter` so a caller that already holds the
        trimmed target and lag block -- an EM loop, which rebuilds them once and
        reuses them across every iteration -- can skip re-deriving them from the
        raw series on each pass.
        """
        return self.log_densities(target, self.conditional_means(lags))

    def filter_densities(self, log_density: npt.NDArray[np.float64]) -> _HamiltonFilterResult:
        """Run the Hamilton filter over an already-built density matrix."""
        return hamilton_filter(log_density, self._transition, initial_prob=self._pi0)

    def smooth_densities(self, filtered: _HamiltonFilterResult) -> _KimSmootherResult:
        """Run the Kim smoother over a completed forward pass."""
        return kim_smoother(filtered, self._transition)

    def filter(self, y: npt.ArrayLike) -> _HamiltonFilterResult:
        """Filter a series, returning regime probabilities and the likelihood.

        Args:
            y: The untrimmed series; the first ``order`` observations are
                conditioned on rather than modelled.

        Returns:
            A :class:`_HamiltonFilterResult` whose probability arrays have
            ``len(y) - order`` rows.

        Raises:
            SpecificationError: If the series contains non-finite values
                -- the switching autoregression conditions on lagged
                observations, so a hole poisons every density whose lag
                window covers it.
        """
        series = np.asarray(y, dtype=np.float64)
        if not np.all(np.isfinite(series)):
            raise SpecificationError(
                "the series contains non-finite values, and the switching "
                "autoregression conditions on lagged observations, so a "
                "missing value poisons every density whose lag window "
                "covers it. Interpolate first, or move to the switching "
                "state-space engine, which handles missing rows."
            )
        target, lags = self.effective_sample(series)
        return self.filter_densities(self.density_matrix(target, lags))

    def smooth(self, y: npt.ArrayLike) -> _KimSmootherResult:
        """Filter, then smooth, returning full-sample regime probabilities."""
        return self.smooth_densities(self.filter(y))

    def loglikelihood(self, y: npt.ArrayLike) -> float:
        """The exact log-likelihood of a series under this system."""
        return self.filter(y).loglikelihood


class _MarkovSwitchingModel[R](_UnivariateModel[R]):
    """Specification surface for Markov-switching autoregressions.

    Args:
        endog: The series.
        order: Autoregressive order ``p``.
        n_regimes: Number of regimes ``K``.
        switching_mean: Whether the intercept switches.
        switching_variance: Whether the innovation variance switches.
        switching_ar: Whether the AR coefficients switch.

    Raises:
        SpecificationError: If no component switches, so no regime is
            identified, or an order is invalid.
        DimensionError: If the series is too short for ``K`` regimes.
    """

    __slots__ = (
        "_k",
        "_order",
        "_sw_ar",
        "_sw_mean",
        "_sw_var",
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        n_regimes: int = 2,
        switching_mean: bool = True,
        switching_variance: bool = True,
        switching_ar: bool = False,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog)
        self._order = validate_order(order, "order")
        self._k = validate_order(n_regimes, "n_regimes", minimum=2)
        self._sw_mean = bool(switching_mean)
        self._sw_var = bool(switching_variance)
        self._sw_ar = bool(switching_ar)
        if not (self._sw_mean or self._sw_var or self._sw_ar):
            raise SpecificationError(
                "at least one of switching_mean, switching_variance, switching_ar "
                "must be True; otherwise no regime is identified."
            )
        self._ensure_length(self._k * (self._order + 2), f"MSAR({self._order}, K={self._k})")

    @property
    def order(self) -> int:
        """The autoregressive order."""
        return self._order

    @property
    def n_regimes(self) -> int:
        """The number of regimes."""
        return self._k

    @staticmethod
    def update_transition(
        smoothed: npt.NDArray[np.float64],
        joint: npt.NDArray[np.float64],
        floor: float,
    ) -> npt.NDArray[np.float64]:
        """M-step transition update from expected transition counts.

        Probabilities are floored before renormalizing, so a regime that becomes
        momentarily unvisited can still be re-entered instead of being absorbed.

        Args:
            smoothed: Smoothed regime probabilities.
            joint: Smoothed joint probabilities ``Pr(S_t = i, S_{t+1} = j | y)``.
            floor: Minimum admissible probability.

        Returns:
            The updated row-stochastic transition matrix.
        """
        k = smoothed.shape[1]
        if joint.shape[0] == 0:
            return np.full((k, k), 1.0 / k)
        numer = joint.sum(axis=0)
        denom = smoothed[:-1].sum(axis=0)
        p = numer / np.clip(denom[:, None], floor, None)
        p = np.clip(p, floor, None)
        return p / p.sum(axis=1, keepdims=True)

    @staticmethod
    def update_coefficients(
        target: npt.NDArray[np.float64],
        lags: npt.NDArray[np.float64],
        smoothed: npt.NDArray[np.float64],
        sigma2: npt.NDArray[np.float64],
        layout: _ParameterLayout,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """M-step coefficient update by responsibility-weighted GLS.

        All regimes are solved in one stacked system so that non-switching blocks
        are estimated jointly across regimes rather than per regime.

        Args:
            target: The effective sample.
            lags: Lagged levels.
            smoothed: Smoothed regime probabilities.
            sigma2: Current per-regime variances.
            layout: Column bookkeeping.

        Returns:
            A tuple ``(intercepts, ar_params)`` of shapes ``(K,)`` and ``(K, p)``.
        """
        n_eff = target.shape[0]
        k, p = layout.n_regimes, layout.order
        d = layout.width
        a = np.zeros((d, d), dtype=np.float64)
        b = np.zeros(d, dtype=np.float64)
        for j in range(k):
            design = np.zeros((n_eff, d), dtype=np.float64)
            design[:, layout.intercept_col(j)] = 1.0
            if p:
                design[:, layout.ar_slice(j)] = lags
            weight = smoothed[:, j] / sigma2[j]
            a += design.T @ (design * weight[:, None])
            b += design.T @ (weight * target)
        beta, *_ = np.linalg.lstsq(a, b, rcond=None)

        intercepts = np.empty(k, dtype=np.float64)
        ar_params = np.zeros((k, p), dtype=np.float64)
        for j in range(k):
            intercepts[j] = beta[layout.intercept_col(j)]
            if p:
                ar_params[j] = beta[layout.ar_slice(j)]
        return intercepts, ar_params

    @staticmethod
    def update_variance(
        target: npt.NDArray[np.float64],
        means: npt.NDArray[np.float64],
        smoothed: npt.NDArray[np.float64],
        switching_variance: bool,
        floor: float,
    ) -> npt.NDArray[np.float64]:
        """M-step variance update, per regime or pooled.

        Args:
            target: The effective sample.
            means: Per-regime conditional means.
            smoothed: Smoothed regime probabilities.
            switching_variance: Whether the variance switches across regimes.
            floor: Minimum admissible variance.

        Returns:
            Per-regime variances of shape ``(K,)``.
        """
        k = smoothed.shape[1]
        sq = (target[:, None] - means) ** 2
        if switching_variance:
            sigma2 = (smoothed * sq).sum(axis=0) / np.clip(smoothed.sum(axis=0), 1e-12, None)
        else:
            sigma2 = np.full(k, float((smoothed * sq).sum() / target.shape[0]))
        return np.clip(sigma2, floor, None)

    @staticmethod
    def initial_transition(
        k: int, rng: np.random.Generator, diagonal: float
    ) -> npt.NDArray[np.float64]:
        """Randomized persistent transition matrix for a restart.

        Args:
            k: Number of regimes.
            rng: Random generator.
            diagonal: Target self-transition probability.

        Returns:
            A row-stochastic ``(K, K)`` matrix.
        """
        off = (1.0 - diagonal) / (k - 1)
        p = np.full((k, k), off) + (diagonal - off) * np.eye(k)
        p = p * rng.uniform(0.9, 1.1, size=(k, k))
        return p / p.sum(axis=1, keepdims=True)

    def _effective_sample(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The target and lag block the EM recursions run on.

        Returns:
            A tuple ``(target, lags)``, where ``target`` drops the first
            ``order`` observations and ``lags`` is the aligned lag matrix.
        """
        return self.endog[self._order :], lag_matrix(self.endog, self._order)

    def _parameter_layout(self) -> _ParameterLayout:
        """Column bookkeeping for the stacked coefficient system."""
        return _ParameterLayout(self._k, self._order, self._sw_mean, self._sw_ar)

    def _numerical_floors(self) -> tuple[float, float]:
        """Lower bounds on the variance and on any transition probability.

        The variance floor is scaled by the sample variance so that it means the
        same thing regardless of the units the series is measured in; the
        probability floor keeps a momentarily unvisited regime re-enterable
        instead of absorbing.

        Returns:
            A tuple ``(var_floor, prob_floor)``.
        """
        return 1e-8 * float(np.var(self.endog)) + 1e-12, 1e-8

    def _run_em(
        self,
        transition0: npt.NDArray[np.float64],
        intercepts0: npt.NDArray[np.float64],
        ar0: npt.NDArray[np.float64],
        sigma20: npt.NDArray[np.float64],
        *,
        max_iter: int,
        tol: float,
    ) -> _ExpectationMaximizationState:
        """Run EM to convergence or ``max_iter`` from one set of starting values.

        The E-step assembles the current draw into a
        :class:`_MarkovSwitchingStateSpace` and asks it to filter and smooth;
        the M-step updates the transition matrix, the coefficients, and the
        variances in that order, and the conditional means are recomputed
        between the coefficient and variance updates so the variance sees the
        new means.

        The state space is rebuilt each iteration rather than mutated, because
        it is frozen and validating -- so the E-step can only ever run on a
        system whose transition matrix is row-stochastic and whose variances
        are positive, which is exactly the invariant an EM loop can lose. It is
        handed the precomputed target and lag block through
        :meth:`_MarkovSwitchingStateSpace.density_matrix` rather than the raw
        series, so the trim is paid once for the whole run.

        Args:
            transition0: Starting transition matrix.
            intercepts0: Starting per-regime intercepts.
            ar0: Starting per-regime AR coefficients.
            sigma20: Starting per-regime variances.
            max_iter: Iteration cap.
            tol: Convergence tolerance on the log-likelihood increment.

        Returns:
            The :class:`_ExpectationMaximizationState` reached.

        Raises:
            NumericalError: If the log-likelihood becomes non-finite.
        """
        target, lags = self._effective_sample()
        layout = self._parameter_layout()
        var_floor, prob_floor = self._numerical_floors()

        transition = transition0.copy()
        intercepts = intercepts0.copy()
        ar_params = ar0.copy()
        sigma2 = sigma20.copy()
        prev_llf = -np.inf
        filtered = predicted = smoothed = np.empty((0, layout.n_regimes))
        n_iter = 0
        converged = False

        for n_iter in range(1, max_iter + 1):
            space = _MarkovSwitchingStateSpaceModel(transition, intercepts, ar_params, sigma2)
            filt = space.filter_densities(space.density_matrix(target, lags))
            smooth = space.smooth_densities(filt)
            filtered = filt.filtered_prob
            predicted = filt.predicted_prob
            smoothed = smooth.smoothed_prob
            llf = filt.loglikelihood
            if not np.isfinite(llf):
                raise NumericalError("MS-AR log-likelihood became non-finite during EM.")
            if llf - prev_llf < tol and n_iter > 1:
                converged = True
                prev_llf = llf
                break
            prev_llf = llf
            transition = self.update_transition(smoothed, smooth.smoothed_joint_prob, prob_floor)
            intercepts, ar_params = self.update_coefficients(target, lags, smoothed, sigma2, layout)
            means = _MarkovSwitchingStateSpaceModel(
                transition, intercepts, ar_params, sigma2
            ).conditional_means(lags)
            sigma2 = self.update_variance(target, means, smoothed, self._sw_var, var_floor)

        return _ExpectationMaximizationState(
            transition=transition,
            intercepts=intercepts,
            ar_params=ar_params,
            sigma2=sigma2,
            filtered_prob=filtered,
            predicted_prob=predicted,
            smoothed_prob=smoothed,
            llf=float(prev_llf),
            n_iter=n_iter,
            converged=converged,
        )

    def _fit_family(
        self,
        *,
        max_iter: int = _DEFAULT_MAX_ITER,
        tol: float = _DEFAULT_TOL,
        n_init: int = _DEFAULT_STARTS,
        screen_iter: int = 15,
        seed: int | np.random.Generator | None = None,
    ) -> _MarkovSwitchingFit:
        """Estimate by EM with multi-start screening.

        Args:
            max_iter: Maximum EM iterations for the refined winning start.
            tol: Convergence tolerance on the log-likelihood increment.
            n_init: Number of random starts to screen.
            screen_iter: Iterations used to score each screening start.
            seed: Seed or generator for the random starts.

        Returns:
            The packed :class:`_MarkovSwitchingFit`, regimes ordered by intercept.

        Raises:
            NumericalError: If every start fails to produce a finite likelihood.
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        y = self.endog
        order, k = self._order, self._k
        layout = self._parameter_layout()
        target, lags = self._effective_sample()
        var_floor, _prob_floor = self._numerical_floors()
        total_var = float(np.var(y))

        def cluster_sigma2(
            intercepts: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            assign = np.argmin(np.abs(target[:, None] - intercepts[None, :]), axis=1)
            sig = np.empty(k, dtype=np.float64)
            for j in range(k):
                group = target[assign == j]
                sig[j] = float(np.var(group)) if group.size > 1 else total_var
            if not self._sw_var:
                sig[:] = sig.mean()
            return np.clip(sig, var_floor, None)

        def make_start(
            index: int,
        ) -> tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ]:
            if index == 0:
                intercepts = np.quantile(y, np.linspace(0.5 / k, 1.0 - 0.5 / k, k))
                diagonal = 0.9
            else:
                intercepts = np.sort(rng.choice(target, size=k, replace=False))
                diagonal = float(rng.uniform(0.8, 0.95))
            return (
                self.initial_transition(k, rng, diagonal),
                intercepts,
                np.zeros((k, order), dtype=np.float64),
                cluster_sigma2(intercepts),
            )

        best: _ExpectationMaximizationState | None = None
        for index in range(max(n_init, 1)):
            transition0, intercepts0, ar_start, sigma20 = make_start(index)
            try:
                fit = self._run_em(
                    transition0,
                    intercepts0,
                    ar_start,
                    sigma20,
                    max_iter=screen_iter,
                    tol=tol,
                )
            except NumericalError:
                continue
            if best is None or fit.llf > best.llf:
                best = fit
        if best is None:
            raise NumericalError("MS-AR estimation failed for every start.")

        refined = self._run_em(
            best.transition,
            best.intercepts,
            best.ar_params,
            best.sigma2,
            max_iter=max_iter,
            tol=tol,
        )
        fit = refined if refined.llf >= best.llf else best

        perm = np.argsort(fit.intercepts)
        transition = fit.transition[np.ix_(perm, perm)]
        intercepts = fit.intercepts[perm]
        ar_params = fit.ar_params[perm]
        variances = fit.sigma2[perm]
        smoothed = fit.smoothed_prob[:, perm]
        space = _MarkovSwitchingStateSpaceModel(transition, intercepts, ar_params, variances)
        fitted = (smoothed * space.conditional_means(lags)).sum(axis=1)
        return _MarkovSwitchingFit(
            transition=transition,
            intercepts=intercepts,
            ar_params=ar_params,
            variances=variances,
            filtered_prob=fit.filtered_prob[:, perm],
            predicted_prob=fit.predicted_prob[:, perm],
            smoothed_prob=smoothed,
            ergodic_prob=ergodic_distribution(transition),
            expected_durations=1.0 / np.clip(1.0 - np.diag(transition), 1e-12, None),
            resid=target - fitted,
            fittedvalues=fitted,
            llf=fit.llf,
            nobs=target.shape[0],
            n_params=k * (k - 1) + layout.n_intercept + layout.n_ar + (k if self._sw_var else 1),
            n_iter=fit.n_iter,
            converged=fit.converged,
        )


class _ThresholdModel[R](_UnivariateModel[R]):
    """Shared specification surface for SETAR/TAR grid-search models.

    Args:
        endog: The series.
        order: AR order per regime.
        delay: Threshold delay; ``None`` searches ``1..order``.
        trim: Fraction trimmed from each tail of the grid.
        n_grid: Number of candidate thresholds per delay.
        threshold_variable: External threshold variable; ``None`` is self-exciting.
    """

    __slots__ = ("_delays", "_n_grid", "_order", "_threshold_variable", "_trim")

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        delay: int | None = None,
        trim: float = _DEFAULT_TRIM,
        n_grid: int = _DEFAULT_GRID,
        threshold_variable: npt.ArrayLike | None = None,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog)
        self._order = validate_order(order, "order", minimum=1)
        self._trim = validate_open_interval(trim, "trim", low=0.0, high=0.5)
        self._n_grid = validate_order(n_grid, "n_grid", minimum=1)
        self._delays = (
            [validate_order(delay, "delay", minimum=1)]
            if delay is not None
            else list(range(1, self._order + 1))
        )
        self._threshold_variable = (
            None
            if threshold_variable is None
            else validate_aligned(threshold_variable, self.endog.shape[0], "threshold_variable")
        )
        self._ensure_length(
            2 * (self._order + 2) + max(self._delays), f"threshold AR({self._order})"
        )

    @property
    def order(self) -> int:
        """AR order per regime."""
        return self._order

    @property
    def delay(self) -> int | None:
        """The fixed delay, or ``None`` when the delay is searched."""
        return self._delays[0] if len(self._delays) == 1 else None

    @property
    def self_exciting(self) -> bool:
        """Whether the threshold variable is a lag of the series itself."""
        return self._threshold_variable is None

    def _fit_family(self) -> _ThresholdFit:
        """Fit a two-regime threshold autoregression by grid search.

        Searches every ``(delay, threshold)`` pair on a trimmed quantile grid and
        keeps the pair minimizing total SSR. Splits that leave either regime with
        fewer than ``order + 2`` observations are skipped, since those coefficients
        would not be identified.

        Returns:
            The packed :class:`_ThresholdFit`.

        Raises:
            NumericalError: If no admissible split exists.
        """
        y, order, delays = self.endog, self._order, self._delays
        trim, n_grid = self._trim, self._n_grid
        threshold_var = self._threshold_variable
        n = y.shape[0]
        start = max(order, max(delays))
        target = y[start:]
        n_eff = target.shape[0]
        design = np.column_stack(
            [deterministic_columns("c", y.shape[0] - start), lag_matrix(y, order, start=start)]
        )
        base = threshold_var if threshold_var is not None else y
        min_regime = order + 2

        best_ssr = np.inf
        best: (
            tuple[int, float, npt.NDArray[np.float64], npt.NDArray[np.float64], int, int] | None
        ) = None
        for d in delays:
            z = base[start - d : n - d]
            grid = np.quantile(z, np.linspace(trim, 1.0 - trim, n_grid))
            for r in np.unique(grid):
                lower = z <= r
                n_lo = int(lower.sum())
                n_hi = n_eff - n_lo
                if n_lo < min_regime or n_hi < min_regime:
                    continue
                b_lo, ssr_lo = ols(design[lower], target[lower])
                b_hi, ssr_hi = ols(design[~lower], target[~lower])
                ssr = ssr_lo + ssr_hi
                if ssr < best_ssr:
                    best_ssr = ssr
                    best = (d, float(r), b_lo, b_hi, n_lo, n_hi)

        if best is None:
            raise NumericalError(
                "threshold grid search found no admissible split; relax trim or shorten order."
            )
        d_star, r_star, b_lo, b_hi, n_lo, n_hi = best
        z = base[start - d_star : n - d_star]
        lower = z <= r_star
        fitted = np.where(lower, design @ b_lo, design @ b_hi)
        resid = target - fitted
        sigma2 = best_ssr / n_eff
        llf = -0.5 * n_eff * (_LOG_2PI + np.log(sigma2) + 1.0)
        return _ThresholdFit(
            delay=d_star,
            threshold=r_star,
            lower_params=b_lo,
            upper_params=b_hi,
            sigma2=sigma2,
            ssr=float(best_ssr),
            n_lower=n_lo,
            n_upper=n_hi,
            resid=resid,
            fittedvalues=fitted,
            llf=float(llf),
            nobs=n_eff,
            n_params=2 * (order + 1) + 1,
        )


class _SmoothTransitionModel[R](_UnivariateModel[R]):
    """Shared specification surface for LSTAR/ESTAR smooth-transition models.

    Args:
        endog: The series.
        order: AR order per regime.
        transition: ``"logistic"`` or ``"exponential"``.
        delay: Transition-variable delay.
    """

    __slots__ = ("_delay", "_order", "_transition")

    def __init__(
        self, endog: npt.ArrayLike, *, order: int, transition: str, delay: int = 1
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog)
        self._order = validate_order(order, "order", minimum=1)
        self._delay = validate_order(delay, "delay", minimum=1)
        self._transition = validate_choice(transition, Transition, "transition")
        self._ensure_length(2 * (self._order + 2) + self._delay, f"STAR({self._order})")

    @property
    def order(self) -> int:
        """AR order per regime."""
        return self._order

    @property
    def delay(self) -> int:
        """The transition-variable delay."""
        return self._delay

    @property
    def transition(self) -> str:
        """The transition function family."""
        return self._transition

    def _build_objective(self) -> _SmoothTransitionObjective:
        """Assemble the concentrated least-squares surface.

        The threshold seed comes from the best hard split over a grid of
        interior quantiles, which is a cheap approximation to the smooth
        problem and lands the multi-start near the right basin.

        Returns:
            The configured objective.

        Raises:
            NumericalError: If the transition variable has zero variance.
        """
        y, order = self.endog, self._order
        delay, transition = self._delay, self._transition
        n = y.shape[0]
        start = max(order, delay)
        target = y[start:]
        n_eff = target.shape[0]
        design = np.column_stack(
            [deterministic_columns("c", n_eff), lag_matrix(y, order, start=start)]
        )
        z = y[start - delay : n - delay]
        scale = float(np.std(z))
        if scale == 0.0:
            raise NumericalError("transition variable has zero variance.")

        c_seed = float(np.median(z))
        best_hard = np.inf
        for candidate in np.quantile(z, np.linspace(0.15, 0.85, 50)):
            lower = z <= candidate
            n_lo = int(lower.sum())
            if n_lo < order + 2 or n_eff - n_lo < order + 2:
                continue
            ssr_hard = ols(design[lower], target[lower])[1] + ols(design[~lower], target[~lower])[1]
            if ssr_hard < best_hard:
                best_hard, c_seed = ssr_hard, float(candidate)

        seeds = tuple(
            np.array([np.log(gamma0), c0])
            for c0 in (c_seed, float(np.median(z)))
            for gamma0 in (2.0, 5.0, 10.0, 25.0)
        )
        return _SmoothTransitionObjective(
            target=target,
            design=design,
            z=z,
            scale=scale,
            transition=transition,
            seeds=seeds,
        )

    def _fit_family(self) -> _SmoothTransitionFit:
        """Fit a smooth-transition autoregression by concentrated least squares.

        Returns:
            The packed fit.

        Raises:
            NumericalError: If the transition variable has zero variance.
        """
        order, delay = self._order, self._delay
        objective = self._build_objective()
        parameters, ssr = _solve(objective)
        _ssr, beta, resid = objective.least_squares(parameters)
        n_eff = objective.target.shape[0]
        sigma2 = ssr / n_eff
        return _SmoothTransitionFit(
            delay=delay,
            threshold=parameters.threshold,
            gamma=parameters.gamma,
            lower_params=beta[: order + 1],
            upper_params=beta[order + 1 :],
            sigma2=sigma2,
            ssr=ssr,
            resid=resid,
            fittedvalues=objective.target - resid,
            llf=float(-0.5 * n_eff * (_LOG_2PI + np.log(sigma2) + 1.0)),
            nobs=n_eff,
            n_params=2 * (order + 1) + 2,
        )


class _MultivariateModel[R](_BaseModel[R]):
    """Base for models over a panel of series observed on a common index.

    Differs from :class:`_UnivariateModel` only in what ``endog`` is allowed to
    be, which is enough to need its own root: :meth:`_BaseModel.__init__`
    validates a one-dimensional series, and every vector model needs a
    ``(nobs, k)`` matrix instead. Everything else the base offers -- the stored
    series, the length guard, the abstract ``fit`` -- carries over untouched,
    because the time axis is axis zero in both cases.

    Args:
        endog: The observed panel, shape ``(nobs, k)``. A 1-D input is promoted
            to a single column, so a one-variable VAR is reachable without a
            reshape.

    Raises:
        DimensionError: If ``endog`` is not two-dimensional after promotion, or
            has no more observations than variables.
        NumericalError: If ``endog`` contains non-finite values.
    """

    __slots__ = ()

    def __init__(self, endog: npt.ArrayLike) -> None:
        """Validate and store the endogenous panel."""
        self._endog = validate_endog_matrix(endog)

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables."""
        return int(self._endog.shape[1])


class _VectorAutoRegressionModel[R](_MultivariateModel[R]):
    """The specification space of a reduced-form vector autoregression."""

    __slots__ = ("_endog", "_names", "_order", "_prior", "_trend")

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
        prior: _Prior | None = None,
    ) -> None:
        """Validate the sample and the specification.

        Args:
            endog: The ``(nobs, k)`` panel, time down the rows.
            order: Autoregressive order.
            trend: Deterministic terms: none, constant, or constant and trend.
            names: One label per variable. Defaults to ``y1 ... yk``.
            prior: Shrinkage toward a stated belief about the coefficients.
                ``None`` is unrestricted least squares. A prior belongs here
                rather than on ``fit`` because it changes which specifications
                are *admissible*: a proper prior makes a system estimable that
                has more regressors than observations, so the length guard
                below has to know about it.

        Raises:
            SpecificationError: If the order, trend, or names are malformed.
            DimensionError: If the sample cannot support the specification.
        """
        self._endog = validate_endog_matrix(endog)
        k = self._endog.shape[1]
        if int(order) != order or order < 0:
            raise SpecificationError(f"order must be an integer >= 0; got {order!r}.")
        self._order = int(order)
        self._trend: str = validate_choice(trend, Trend, "trend")
        self._names = self._resolve_names(names, k, "names", "y")
        self._prior: _Prior = _NoPrior() if prior is None else prior
        need = (
            self._rows_lost(self._order) + 1
            if self.is_shrunk
            else self._rows_lost(self._order) + self.n_regressors + 1
        )
        if self._endog.shape[0] < need:
            raise DimensionError(
                f"a sample of {self._endog.shape[0]} rows is too short for "
                f"{type(self).__name__}({self._order}); it needs at least {need}."
                + (
                    ""
                    if self.is_shrunk
                    else " A proper prior would make this specification estimable."
                )
            )

    @staticmethod
    def _resolve_names(
        names: Sequence[str] | None, count: int, label: str, prefix: str
    ) -> tuple[str, ...]:
        """Default or validate a label tuple.

        Three call sites want the same three rules -- right count, no
        duplicates, generated stems when omitted -- for variables, exogenous
        regressors, and units. Static rather than bound because it reads nothing
        from the instance, which is what lets a subclass call it before
        ``super().__init__`` has populated any state.

        Args:
            names: Caller-supplied labels, or ``None``.
            count: How many labels the specification requires.
            label: Argument name, for error messages.
            prefix: Stem for generated labels.

        Returns:
            One label per item.

        Raises:
            SpecificationError: If the count is wrong or the labels repeat.
        """
        if names is None:
            return tuple(f"{prefix}{i + 1}" for i in range(count))
        resolved = tuple(str(name) for name in names)
        if len(resolved) != count:
            raise SpecificationError(
                f"{label} must have one entry per item ({count}); got {len(resolved)}."
            )
        if len(set(resolved)) != count:
            raise SpecificationError(f"{label} must be unique; got {resolved}.")
        return resolved

    @property
    def prior(self) -> _Prior:
        """The prior on the coefficients; :class:`NoPrior` when unrestricted."""
        return self._prior

    @property
    def is_shrunk(self) -> bool:
        """Whether a prior contributes anything to this estimate."""
        return bool(self._prior._components())

    def _prior_context(self) -> _PriorContext:
        """Everything the prior needs to know about this sample.

        Built by the model rather than by the prior, so that the design's
        column order is stated once by whoever owns the design.
        """
        return _PriorContext(
            k_endog=self.k_endog,
            order=self._order,
            scales=minnesota_scales(self._endog, self._order),
            presample_mean=self._endog[: self._order].mean(axis=0),
            k_exog=self.n_regressors - self._n_deterministic_columns - self.k_endog * self._order,
            n_deterministic=self._n_deterministic_columns,
            include_constant=self._n_deterministic_columns > 0,
        )

    @property
    def endog(self) -> npt.NDArray[np.float64]:
        """The validated sample."""
        return self._endog

    @property
    def order(self) -> int:
        """Autoregressive order."""
        return self._order

    @property
    def trend(self) -> str:
        """Deterministic specification."""
        return self._trend

    @property
    def names(self) -> tuple[str, ...]:
        """Variable labels."""
        return self._names

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables."""
        return int(self._endog.shape[1])

    @property
    def _n_deterministic_columns(self) -> int:
        """Width of the leading deterministic block.

        Split out from :func:`n_deterministic` because a fixed-effects panel
        replaces the trend specification with one indicator per unit, and every
        offset downstream is expressed against this number rather than against
        the trend string.
        """
        return n_deterministic(self._trend)

    @property
    def n_regressors(self) -> int:
        """Regressors per equation."""
        return self._n_deterministic_columns + self.k_endog * self._order

    def _burn_for(self, order: int) -> int:
        """Leading observations each series loses at a candidate order.

        Args:
            order: Autoregressive order under consideration.

        Returns:
            The number of leading observations no equation can be written for.
        """
        return order

    def _rows_lost(self, order: int) -> int:
        """Design rows the whole sample loses at a candidate order.

        Distinct from :meth:`_burn_for` only when there is more than one series:
        a panel of ``N`` units loses ``order`` observations from each of them.

        Args:
            order: Autoregressive order under consideration.

        Returns:
            Total rows dropped from the stacked sample.
        """
        return self._burn_for(order)

    def _max_supported_lags(self) -> int:
        """Largest order this sample can identify on a common effective sample."""
        free = int(self._endog.shape[0]) - self._n_deterministic_columns - 1
        return max(free // (self.k_endog + 1), 0)

    def _design(
        self, order: int | None = None, *, trim: int = 0
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
        """Build the target and regressor matrix.

        Args:
            order: Autoregressive order; defaults to the fitted order.
            trim: Leading observations to discard from each series before the
                lags are formed. Lag-order selection uses this to hold the
                effective sample fixed while the order varies, so that the
                criteria are comparable rather than merely computed.

        Returns:
            The target block, the design, and the row count of both.

        Raises:
            DimensionError: If the trimmed sample is shorter than the order.
        """
        lags = self._order if order is None else order
        panel = self._endog[trim:]
        nobs = panel.shape[0]
        if nobs <= lags:
            raise DimensionError(f"{nobs} observations is too few for a design of order {lags}.")
        effective = nobs - lags
        det = deterministic_columns(self._trend, effective, start=trim + lags + 1)
        return panel[lags:], np.column_stack([det, lag_matrix(panel, lags)]), effective

    @staticmethod
    def _least_squares(
        target: npt.NDArray[np.float64], design: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Solve the multivariate regression and return coefficients and residuals."""
        coef: npt.NDArray[np.float64] = np.linalg.lstsq(design, target, rcond=None)[0]
        return coef, target - design @ coef

    def _gaussian_moments(
        self, target: npt.NDArray[np.float64], design: npt.NDArray[np.float64]
    ) -> _VectorMoments:
        """Least squares plus the concentrated Gaussian likelihood.

        Args:
            target: The ``(nobs, k)`` block being explained.
            design: The ``(nobs, width)`` regressor matrix.

        Returns:
            A :class:`_VectorMoments` record.

        Raises:
            DimensionError: If the design is not overidentified.
            NumericalError: If the residual covariance is singular, which means
                one equation is an exact linear combination of the others.
        """
        nobs, width = design.shape
        k = target.shape[1]
        if nobs <= width:
            raise DimensionError(
                f"{nobs} observations cannot identify {width} coefficients per equation."
            )
        coef, resid = self._least_squares(target, design)
        cross = resid.T @ resid
        sigma_ml = cross / nobs
        sign, logdet = np.linalg.slogdet(sigma_ml)
        if sign <= 0:
            raise NumericalError("the residual covariance is singular.")
        llf = -0.5 * nobs * k * _LOG_2PI - 0.5 * nobs * float(logdet) - 0.5 * nobs * k
        return _VectorMoments(
            coef=coef,
            resid=resid,
            fittedvalues=design @ coef,
            sigma_u=cross / (nobs - width),
            sigma_ml=sigma_ml,
            llf=float(llf),
            nobs=int(nobs),
            width=int(width),
        )

    def _shrunk_moments(
        self, target: npt.NDArray[np.float64], design: npt.NDArray[np.float64]
    ) -> tuple[_VectorMoments, _CoefficientInference]:
        """Posterior mean and the Gaussian quantities that follow from it.

        The residual covariance and the likelihood are computed at the
        posterior mean rather than at the least-squares one, and the
        degrees-of-freedom correction charges the *effective* parameter count.
        Charging the nominal width would penalize a shrunk model for freedom it
        never used, which is the whole point of shrinking.

        Args:
            target: The ``(nobs, k)`` block being explained.
            design: The ``(nobs, width)`` regressor matrix.

        Returns:
            The moments record and the posterior covariance behind it.

        Raises:
            NumericalError: If the residual covariance is singular.
        """
        posterior = posterior_coefficients(target, design, self._prior, self._prior_context())
        coef = posterior.coefficients
        fitted = design @ coef
        resid = target - fitted
        nobs, k = target.shape
        cross = resid.T @ resid
        sigma_ml = cross / nobs
        sign, logdet = np.linalg.slogdet(sigma_ml)
        if sign <= 0:
            raise NumericalError("the residual covariance is singular.")
        llf = -0.5 * nobs * k * _LOG_2PI - 0.5 * nobs * float(logdet) - 0.5 * nobs * k
        spent = posterior.effective_parameters / k
        moments = _VectorMoments(
            coef=coef,
            resid=resid,
            fittedvalues=fitted,
            sigma_u=cross / max(nobs - spent, 1.0),
            sigma_ml=sigma_ml,
            llf=float(llf),
            nobs=int(nobs),
            width=int(design.shape[1]),
        )
        return moments, posterior

    def _lag_blocks(self, coef: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Slice the endogenous lag coefficients out as ``A_1, ..., A_p``.

        Args:
            coef: The full ``(width, k)`` coefficient matrix.

        Returns:
            A ``(p, k, k)`` stack, empty when the order is zero.
        """
        offset, k, p = self._n_deterministic_columns, self.k_endog, self._order
        if not p:
            return np.zeros((0, k, k), dtype=np.float64)
        return np.stack([coef[offset + i * k : offset + (i + 1) * k, :].T for i in range(p)])

    def _fit_family(self) -> _VectorAutoRegressionFit:
        """Estimate the system, by least squares or under the prior."""
        k = self.k_endog
        target, design, _ = self._design()
        posterior: _CoefficientInference | None = None
        if self.is_shrunk:
            moments, posterior = self._shrunk_moments(target, design)
            spent = posterior.effective_parameters
        else:
            moments = self._gaussian_moments(target, design)
            spent = float(k * moments.width)
        return _VectorAutoRegressionFit(
            coefficients=self._lag_blocks(moments.coef),
            deterministic=moments.coef[: self._n_deterministic_columns],
            sigma_u=moments.sigma_u,
            sigma_ml=moments.sigma_ml,
            design=design,
            resid=moments.resid,
            fittedvalues=moments.fittedvalues,
            llf=moments.llf,
            nobs=moments.nobs,
            n_params=spent + k * (k + 1) / 2,
            posterior=posterior,
            prior_label=self._prior._label(),
        )

    def lag_order_selection(self, max_lags: int | None = None) -> _LagOrderSelection:
        """Score every order from zero to ``max_lags`` on one common sample.

        Args:
            max_lags: Highest order to score. Defaults to the largest the
                sample supports.

        Returns:
            A :class:`_LagOrderSelection` holding all four criteria.

        Raises:
            SpecificationError: If ``max_lags`` exceeds what the sample supports.
            NumericalError: If the candidates do not share one effective sample,
                which would make the criteria incomparable.
        """
        supported = self._max_supported_lags()
        top = supported if max_lags is None else int(max_lags)
        if top < 0 or top > supported:
            raise SpecificationError(
                f"max_lags {top} exceeds what this sample supports (maximum {supported})."
            )
        k = self.k_endog
        top_burn = self._burn_for(top)
        curves: dict[str, list[float]] = {name: [] for name in ("aic", "bic", "hqic", "fpe")}
        effective = -1
        for candidate in range(top + 1):
            target, design, nobs = self._design(
                candidate, trim=top_burn - self._burn_for(candidate)
            )
            if effective < 0:
                effective = nobs
            elif nobs != effective:
                raise NumericalError(
                    f"order {candidate} produced {nobs} rows against {effective} for the "
                    "shorter orders; the common-sample construction failed."
                )
            width = design.shape[1]
            _, resid = self._least_squares(target, design)
            sigma = resid.T @ resid / effective
            logdet = float(np.linalg.slogdet(sigma)[1])
            free = k * width
            curves["aic"].append(logdet + 2.0 * free / effective)
            curves["bic"].append(logdet + np.log(effective) * free / effective)
            curves["hqic"].append(logdet + 2.0 * np.log(np.log(effective)) * free / effective)
            curves["fpe"].append(
                float(np.linalg.det(sigma) * ((effective + width) / (effective - width)) ** k)
            )
        return _LagOrderSelection(
            max_lags=top,
            nobs=effective,
            **{name: np.asarray(values, dtype=np.float64) for name, values in curves.items()},
        )


class _PanelVectorAutoRegressionModel[R](_VectorAutoRegressionModel[R]):
    """A vector autoregression estimated across units with pooled slopes.

    Every unit shares the autoregressive matrices and the innovation
    covariance; only the intercept is allowed to differ. That is the whole
    content of the specification, and it is a strong assumption rather than a
    technicality -- heterogeneous dynamics estimated as if pooled do not average
    to the mean dynamics.

    The one mechanical rule that matters is that lags are built inside each unit
    and never across a boundary. Stacking first and lagging afterwards would
    quietly regress each unit's first observation on the previous unit's last,
    which produces a number rather than an error.

    With unit effects the estimator is least-squares dummy variables, so the lag
    coefficients carry the Nickell bias: of order ``1/T``, downward for a
    positive own-lag, and unaffected by the number of units. The result says so
    on its summary rather than reporting a clean-looking coefficient.
    """

    __slots__ = ("_effects", "_lengths", "_unit_names", "_units")

    def __init__(
        self,
        panel: npt.ArrayLike | Sequence[npt.ArrayLike],
        *,
        order: int,
        effects: PanelEffects = "unit",
        trend: Trend = "n",
        names: Sequence[str] | None = None,
        unit_names: Sequence[str] | None = None,
    ) -> None:
        """Validate the panel and the specification.

        Args:
            panel: A ``(n_units, nobs, k)`` array, or a sequence of ``(nobs_i, k)``
                arrays for an unbalanced panel.
            order: Autoregressive order, common to all units.
            effects: ``"unit"`` for one intercept per unit, ``"none"`` for a
                single deterministic block shared by the pool.
            trend: Deterministic terms, used only when ``effects="none"``.
            names: Variable labels.
            unit_names: Unit labels. Defaults to ``unit1 ... unitN``.

        Raises:
            SpecificationError: If ``effects`` is unrecognized, unit effects are
                combined with a pooled constant, or the labels are malformed.
            DimensionError: If the panel is malformed or a unit is too short to
                supply the lags.
        """
        self._units = validate_panel(panel)
        self._lengths = tuple(int(unit.shape[0]) for unit in self._units)
        if effects not in ("none", "unit"):
            raise SpecificationError(f"effects must be one of ('none', 'unit'); got {effects!r}.")
        self._effects: str = effects
        if effects == "unit" and trend in ("c", "ct"):
            raise SpecificationError(
                "unit effects already span the intercept, so trend must be 'n' when "
                f"effects='unit'; got trend={trend!r}. A pooled constant alongside unit "
                "dummies is exactly collinear and the two are not separately identified."
            )
        self._unit_names = self._resolve_names(unit_names, len(self._units), "unit_names", "unit")
        super().__init__(np.vstack(self._units), order=order, trend=trend, names=names)
        shortest = min(self._lengths)
        if shortest <= self._order:
            raise DimensionError(
                f"unit {self._unit_names[self._lengths.index(shortest)]!r} has {shortest} "
                f"observations, which cannot supply {self._order} lags; lags are never "
                "taken across a unit boundary, so the shortest unit binds."
            )

    @property
    def units(self) -> tuple[npt.NDArray[np.float64], ...]:
        """The per-unit series, in the order given."""
        return self._units

    @property
    def unit_names(self) -> tuple[str, ...]:
        """Unit labels."""
        return self._unit_names

    @property
    def unit_lengths(self) -> tuple[int, ...]:
        """Observations per unit, before lags are taken."""
        return self._lengths

    @property
    def effects(self) -> str:
        """Which intercepts the specification allows to vary."""
        return self._effects

    @property
    def n_units(self) -> int:
        """Number of units."""
        return len(self._units)

    @property
    def _n_deterministic_columns(self) -> int:
        """Unit indicators under fixed effects, otherwise the trend block."""
        return self.n_units if self._effects == "unit" else n_deterministic(self._trend)

    def _rows_lost(self, order: int) -> int:
        """Rows lost across the stack: ``order`` from each unit."""
        return self.n_units * order

    def _max_supported_lags(self) -> int:
        """Largest order the pool supports, bounded by the shortest unit."""
        total = sum(self._lengths)
        free = total - self._n_deterministic_columns - 1
        cap = free // (self.n_units + self.k_endog)
        return max(min(cap, min(self._lengths) - 1), 0)

    def _unit_deterministic(self, index: int, rows: int, *, start: int) -> npt.NDArray[np.float64]:
        """The deterministic block one unit contributes.

        Args:
            index: Position of the unit.
            rows: Rows this unit contributes to the design.
            start: Time index of the unit's first modelled row.

        Returns:
            A ``(rows, _n_deterministic_columns)`` block.
        """
        if self._effects != "unit":
            return deterministic_columns(self._trend, rows, start=start)
        block = np.zeros((rows, self.n_units), dtype=np.float64)
        block[:, index] = 1.0
        return block

    def _design(
        self, order: int | None = None, *, trim: int = 0
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
        """Build each unit's design separately and stack them.

        Args:
            order: Autoregressive order; defaults to the fitted order.
            trim: Leading observations to discard from *each* unit first.

        Returns:
            The stacked target, the stacked design, and their row count.

        Raises:
            DimensionError: If any unit is too short after the trim.
        """
        lags = self._order if order is None else order
        targets: list[npt.NDArray[np.float64]] = []
        designs: list[npt.NDArray[np.float64]] = []
        for index, unit in enumerate(self._units):
            trimmed = unit[trim:]
            nobs = trimmed.shape[0]
            if nobs <= lags:
                raise DimensionError(
                    f"unit {self._unit_names[index]!r} has {nobs} usable observations, too "
                    f"few for a design of order {lags}."
                )
            det = self._unit_deterministic(index, nobs - lags, start=trim + lags + 1)
            targets.append(trimmed[lags:])
            designs.append(np.column_stack([det, lag_matrix(trimmed, lags)]))
        target = np.vstack(targets)
        return target, np.vstack(designs), int(target.shape[0])


class _ExogenousVectorAutoRegressionModel[R](_VectorAutoRegressionModel[R]):
    """A vector autoregression with a distributed lag of weakly exogenous regressors.

    The design gains one block of ``k_exog`` columns per exogenous lag, placed
    *after* the endogenous lags rather than beside the deterministic terms. That
    ordering is load-bearing: every offset the base computes -- the
    deterministic slice, the lag slice, the Wald index -- is written against a
    lag block that starts immediately after the deterministic block, and
    appending keeps all of them correct without a single override.

    Two things the exogenous block does not do. It does not enter the
    moving-average representation, because ``x`` is conditioned on rather than
    shocked, so impulse responses and the variance decomposition are exactly the
    endogenous ones. And it does not extend a forecast, because there is no
    model of ``x`` here to extend it with.
    """

    __slots__ = ("_exog", "_exog_names", "_exog_order")

    def __init__(
        self,
        endog: npt.ArrayLike,
        exog: npt.ArrayLike,
        *,
        order: int,
        exog_order: int = 0,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
        exog_names: Sequence[str] | None = None,
    ) -> None:
        """Validate both samples and the specification.

        Args:
            endog: The ``(nobs, k)`` endogenous panel.
            exog: The ``(nobs, m)`` exogenous block, aligned on the same index.
            order: Endogenous autoregressive order.
            exog_order: Exogenous lags beyond the contemporaneous term.
            trend: Deterministic terms.
            names: Endogenous labels.
            exog_names: Exogenous labels. Defaults to ``x1 ... xm``.

        Raises:
            SpecificationError: If ``exog_order`` is malformed, the labels are
                malformed, or an endogenous and an exogenous label collide.
            DimensionError: If the two samples are not aligned or the sample is
                too short for the specification.
        """
        rows = validate_endog_matrix(endog).shape[0]
        self._exog = validate_exog(exog, nobs=rows)
        if int(exog_order) != exog_order or exog_order < 0:
            raise SpecificationError(f"exog_order must be an integer >= 0; got {exog_order!r}.")
        self._exog_order = int(exog_order)
        self._exog_names = self._resolve_names(exog_names, self._exog.shape[1], "exog_names", "x")
        super().__init__(endog, order=order, trend=trend, names=names)
        overlap = set(self._names) & set(self._exog_names)
        if overlap:
            raise SpecificationError(
                "names and exog_names must not overlap, or a coefficient table cannot say "
                f"which block a row came from; both contain {tuple(sorted(overlap))}."
            )

    @property
    def exog(self) -> npt.NDArray[np.float64]:
        """The validated exogenous block."""
        return self._exog

    @property
    def exog_order(self) -> int:
        """Exogenous lags beyond the contemporaneous term."""
        return self._exog_order

    @property
    def exog_names(self) -> tuple[str, ...]:
        """Exogenous labels."""
        return self._exog_names

    @property
    def k_exog(self) -> int:
        """Number of exogenous variables."""
        return int(self._exog.shape[1])

    @property
    def n_regressors(self) -> int:
        """Regressors per equation, including the distributed lag."""
        return super().n_regressors + self.k_exog * (self._exog_order + 1)

    def _burn_for(self, order: int) -> int:
        """Leading observations lost, which the longer of the two orders sets."""
        return max(order, self._exog_order)

    def _max_supported_lags(self) -> int:
        """Largest endogenous order the sample can identify."""
        free = (
            int(self._endog.shape[0])
            - self._n_deterministic_columns
            - self.k_exog * (self._exog_order + 1)
            - 1
        )
        return max(free // (self.k_endog + 1), 0)

    def _exog_block(self, exog: npt.NDArray[np.float64], burn: int) -> npt.NDArray[np.float64]:
        """Stack ``x_t, x_{t-1}, ..., x_{t-s}`` for rows ``burn`` onward."""
        nobs = exog.shape[0]
        return np.column_stack([exog[burn - j : nobs - j] for j in range(self._exog_order + 1)])

    def _design(
        self, order: int | None = None, *, trim: int = 0
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
        """Build the target and the three-block regressor matrix.

        Args:
            order: Endogenous order; defaults to the fitted order.
            trim: Leading observations to discard from both samples first.

        Returns:
            The target block, the design, and the row count of both.

        Raises:
            DimensionError: If the trimmed sample cannot supply both sets of lags.
        """
        lags = self._order if order is None else order
        burn = max(lags, self._exog_order)
        panel = self._endog[trim:]
        exog = self._exog[trim:]
        nobs = panel.shape[0]
        if nobs <= burn:
            raise DimensionError(
                f"{nobs} observations is too few for a design of order {lags} with "
                f"{self._exog_order} exogenous lags."
            )
        effective = nobs - burn
        det = deterministic_columns(self._trend, effective, start=trim + burn + 1)
        design = np.column_stack(
            [det, lag_matrix(panel, lags, start=burn), self._exog_block(exog, burn)]
        )
        return panel[burn:], design, effective

    def _fit_family(self) -> _ExogenousVectorAutoRegressionFit:
        """Estimate the system by multivariate least squares."""
        k, m = self.k_endog, self.k_exog
        target, design, _ = self._design()
        moments = self._gaussian_moments(target, design)
        start = self._n_deterministic_columns + k * self._order
        exog_blocks = np.stack(
            [
                moments.coef[start + j * m : start + (j + 1) * m, :].T
                for j in range(self._exog_order + 1)
            ]
        )
        return _ExogenousVectorAutoRegressionFit(
            coefficients=self._lag_blocks(moments.coef),
            exog_coefficients=exog_blocks,
            deterministic=moments.coef[: self._n_deterministic_columns],
            sigma_u=moments.sigma_u,
            sigma_ml=moments.sigma_ml,
            design=design,
            resid=moments.resid,
            fittedvalues=moments.fittedvalues,
            llf=moments.llf,
            nobs=moments.nobs,
            n_params=k * moments.width + k * (k + 1) // 2,
        )


class _VectorErrorCorrectionModel[R](_VectorAutoRegressionModel[R]):
    """A vector autoregression in levels, reparameterized around its unit roots.

    ``order`` counts lags of the *levels* system, matching the VAR it is a
    reparameterization of, so a VECM of order ``p`` carries ``p - 1`` lagged
    differences. Holding the levels convention is what lets the inherited
    :meth:`lag_order_selection` mean what it says: the standard way to choose
    ``p`` for a VECM is to choose it for the unrestricted levels VAR, and that
    is exactly the method this class inherits without touching.

    ``cointegration_trend`` is Johansen's five-case classification rather than
    the three-value trend the levels VAR takes, because a constant or trend can
    sit either inside the cointegrating space or outside it and the two have
    different implications for the long run. The unrestricted remainder maps
    onto the base class's ``trend``, which is what the short-run regression and
    the lag-order criteria see.

    An optional ``exog`` block makes this the conditional specification of
    Pesaran, Shin and Smith: weakly exogenous integrated regressors that share
    the cointegrating space and enter the lagged differences, but carry no
    equations of their own. It lives on this class rather than a subclass
    because a closed system is the case where the block happens to be empty,
    and duplicating the eigenvalue problem to say so would be the wrong kind of
    honesty.

    Estimation is Johansen's reduced-rank maximum likelihood: concentrate out
    the short-run terms, solve the eigenvalue problem for the cointegrating
    space, then -- and this is the part that keeps the class small -- take the
    remaining parameters from an ordinary multivariate regression on
    ``[deterministic | lagged differences | error-correction terms]``. That last
    step is not an approximation. By the Frisch-Waugh-Lovell theorem it
    reproduces Johansen's ``alpha`` exactly, which means the entire inference
    layer built for the least-squares families applies here, conditional on a
    ``beta`` that converges fast enough for the conditioning to be free.
    """

    __slots__ = ("_contemporaneous", "_exog", "_exog_names", "_rank", "_trend_case")

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        rank: int,
        cointegration_trend: CointegrationTrend = "constant",
        exog: npt.ArrayLike | None = None,
        contemporaneous: bool = True,
        names: Sequence[str] | None = None,
        exog_names: Sequence[str] | None = None,
    ) -> None:
        """Validate the sample, the lag length, and the rank.

        Args:
            endog: The ``(nobs, k)`` panel of levels, time down the rows.
            order: Lags of the levels system; the VECM carries ``order - 1``
                lagged differences.
            rank: Cointegrating rank. Up to ``k - 1`` for a closed system, since
                rank ``k`` is stationarity in levels; up to ``k`` when an
                exogenous block is present, where it means every modelled
                variable cointegrates with that block.
            cointegration_trend: One of Johansen's five cases.
            exog: An optional ``(nobs, k_x)`` block of weakly exogenous
                integrated regressors, carried in the cointegrating space and
                the lagged differences but given no equations of their own.
                ``None`` is the closed system.
            contemporaneous: Whether the current exogenous difference enters the
                short-run equation. Ignored without an exogenous block.
            names: Endogenous variable labels.
            exog_names: Exogenous labels. Defaults to ``x1 ... xkx``.

        Raises:
            SpecificationError: If the order is below one, the case is
                unrecognized, the labels collide, or the rank is outside its
                admissible range.
            DimensionError: If the sample cannot support the specification.
        """
        self._trend_case: str = validate_choice(
            cointegration_trend, CointegrationTrend, "cointegration_trend"
        )
        if int(order) != order or order < 1:
            raise SpecificationError(
                f"order counts lags of the levels system and must be at least 1; got {order!r}."
            )
        self._rank = -1
        self._contemporaneous = bool(contemporaneous)
        rows = int(validate_endog_matrix(endog).shape[0])
        self._exog = (
            np.zeros((rows, 0), dtype=np.float64)
            if exog is None
            else validate_exog_matrix(exog, nobs=rows)
        )
        self._exog_names = self._resolve_names(exog_names, self._exog.shape[1], "exog_names", "x")
        super().__init__(
            endog,
            order=int(order),
            trend=_UNRESTRICTED_TREND[self._trend_case],  # type: ignore[arg-type]
            names=names,
        )
        k = self.k_endog
        overlap = set(self._names) & set(self._exog_names)
        if overlap:
            raise SpecificationError(
                "names and exog_names must not overlap, or a coefficient table cannot say "
                f"which block a row came from; both contain {tuple(sorted(overlap))}."
            )
        upper = k if self._exog.shape[1] else k - 1
        if int(rank) != rank or not 0 <= rank <= upper:
            raise SpecificationError(
                f"rank must be an integer in 0..{upper}; got {rank!r}."
                + (
                    ""
                    if self._exog.shape[1]
                    else f" A rank of {k} is an unrestricted stationary system, which is a "
                    "VAR in levels rather than an error-correction model."
                )
            )
        self._rank = int(rank)

    @property
    def cointegration_trend(self) -> str:
        """The Johansen case."""
        return self._trend_case

    @property
    def rank(self) -> int:
        """Cointegrating rank."""
        return self._rank

    @property
    def exog(self) -> npt.NDArray[np.float64]:
        """The weakly exogenous block, zero-width for a closed system."""
        return self._exog

    @property
    def exog_names(self) -> tuple[str, ...]:
        """Exogenous labels."""
        return self._exog_names

    @property
    def k_exog(self) -> int:
        """Weakly exogenous integrated regressors."""
        return int(self._exog.shape[1])

    @property
    def contemporaneous(self) -> bool:
        """Whether the current exogenous difference enters the short-run equation."""
        return self._contemporaneous and bool(self.k_exog)

    @property
    def k_cointegrating(self) -> int:
        """Rows of ``beta``: every integrated variable plus any restricted term."""
        return (
            self.k_endog
            + self.k_exog
            + int(self._trend_case in ("restricted_constant", "restricted_trend"))
        )

    @property
    def _n_short_run_lags(self) -> int:
        """Lagged differences in the short-run equation."""
        return self._order - 1

    @property
    def n_regressors(self) -> int:
        """Short-run regressors per equation."""
        return (
            n_deterministic(self._trend)
            + (self.k_endog + self.k_exog) * self._n_short_run_lags
            + self.k_exog * int(self.contemporaneous)
            + max(self._rank, 0)
        )

    def _cointegration_moments(self) -> _CointegrationMoments:
        """Concentrate out the short-run terms and solve the eigenvalue problem.

        Returns:
            A :class:`_CointegrationMoments` record, independent of the rank.

        Raises:
            DimensionError: If the sample is too short for the lag length.
            NumericalError: If the lagged-levels second moment is singular,
                which means a variable is redundant.
        """
        k = self.k_endog
        joint = np.column_stack([self._endog, self._exog]) if self.k_exog else self._endog
        nobs_total = joint.shape[0]
        order = self._order
        effective = nobs_total - order
        if effective <= 0:
            raise DimensionError(
                f"a sample of {nobs_total} rows cannot support {order} levels lags."
            )
        diffs = np.diff(joint, axis=0)
        differences = diffs[order - 1 :, :k]
        levels = joint[order - 1 : nobs_total - 1]
        blocks = [
            diffs[order - i - 1 : nobs_total - i - 1] for i in range(1, self._n_short_run_lags + 1)
        ]
        if self.contemporaneous:
            blocks.append(diffs[order - 1 :, k:])
        index = np.arange(order, nobs_total, dtype=np.float64)[:, None]
        if self._trend_case == "restricted_constant":
            levels = np.column_stack([levels, np.ones(effective)])
        elif self._trend_case == "restricted_trend":
            levels = np.column_stack([levels, index])
        det = deterministic_columns(self._trend, effective, start=order + 1)
        short_run = (
            np.column_stack([det, *blocks]) if blocks or det.shape[1] else np.zeros((effective, 0))
        )
        if short_run.shape[1]:
            r0 = differences - short_run @ np.linalg.lstsq(short_run, differences, rcond=None)[0]
            r1 = levels - short_run @ np.linalg.lstsq(short_run, levels, rcond=None)[0]
        else:
            r0, r1 = differences, levels
        s00 = r0.T @ r0 / effective
        s01 = r0.T @ r1 / effective
        s11 = r1.T @ r1 / effective
        try:
            factor = np.linalg.cholesky(s11)
        except np.linalg.LinAlgError as error:
            raise NumericalError(
                "the lagged-levels second moment is singular; one variable is a linear "
                "combination of the others."
            ) from error
        inverse = np.linalg.inv(factor)
        quad = inverse @ s01.T @ np.linalg.solve(s00, s01) @ inverse.T
        eigenvalues, vectors = np.linalg.eigh((quad + quad.T) / 2.0)
        order_desc = np.argsort(eigenvalues)[::-1]
        return _CointegrationMoments(
            eigenvalues=np.clip(eigenvalues[order_desc][:k], 0.0, 1.0 - 1e-15),
            eigenvectors=inverse.T @ vectors[:, order_desc],
            levels=levels,
            differences=differences,
            short_run=short_run,
            s00=s00,
            nobs=effective,
        )

    def rank_test(
        self,
        *,
        small_sample: bool = False,
        simulations: int = 25_000,
        steps: int = 500,
        seed: int = 20260819,
    ) -> _JohansenRankTest:
        """Trace and maximum-eigenvalue tests for every candidate rank.

        Independent of the ``rank`` this model was constructed with: the
        eigenvalue problem does not know about it, so the test is a property of
        the data and the lag length alone and can be read before committing to
        a specification.

        With an exogenous block the null distribution is the conditional one of
        Pesaran, Shin and Smith, which the simulator is told about through
        ``n_exog``. Its critical values are materially larger, so a conditional
        statistic read against the unconditional table would over-reject.

        Args:
            small_sample: Apply the Reinsel-Ahn scaling ``(T - (k_y + k_x) * p) / T``.
                The asymptotic test over-rejects in short samples -- around
                seven percent at a nominal five in a three-variable system with
                four hundred observations -- and this pulls it back.
            simulations: Replications behind each p-value.
            steps: Discretization of the simulated Brownian path.
            seed: Fixed so a p-value is reproducible.

        Returns:
            A :class:`_JohansenRankTest`.
        """
        moments = self._cointegration_moments()
        k, effective = self.k_endog, moments.nobs
        span = k + self.k_exog
        scale = (effective - span * self._order) / effective if small_sample else 1.0
        logs = np.log1p(-moments.eigenvalues[:k])
        trace = np.array(
            [-effective * scale * logs[rank:].sum() for rank in range(k)], dtype=np.float64
        )
        maximum = np.array([-effective * scale * logs[rank] for rank in range(k)], dtype=np.float64)
        trace_p = np.empty(k, dtype=np.float64)
        maximum_p = np.empty(k, dtype=np.float64)
        for rank in range(k):
            trace_null, maximum_null = simulate_cointegration_null(
                k - rank,
                self._trend_case,
                n_exog=self.k_exog,
                simulations=simulations,
                steps=steps,
                seed=seed,
            )
            trace_p[rank] = float((trace_null >= trace[rank]).mean())
            maximum_p[rank] = float((maximum_null >= maximum[rank]).mean())
        return _JohansenRankTest(
            eigenvalues=moments.eigenvalues[:k],
            trace_statistic=trace,
            max_eigenvalue_statistic=maximum,
            trace_pvalue=trace_p,
            max_eigenvalue_pvalue=maximum_p,
            nobs=effective,
            deterministic=self._trend_case,
            k_exog=self.k_exog,
            simulations=simulations,
            small_sample=small_sample,
        )

    def _fit_family(self) -> _VectorErrorCorrectionFit:
        """Estimate the system at the specified rank.

        Returns:
            A :class:`_VectorErrorCorrectionFit`.

        Raises:
            DimensionError: If the short-run design is not overidentified.
            NumericalError: If the residual covariance is singular.
        """
        moments = self._cointegration_moments()
        k, rank, m = self.k_endog, self._rank, self.k_exog
        span = k + m
        beta = moments.eigenvectors[:, :rank]
        correction = moments.levels @ beta
        design = np.column_stack([moments.short_run, correction])
        least_squares: _VectorMoments = self._gaussian_moments(moments.differences, design)
        width_det = n_deterministic(self._trend)
        lags = self._n_short_run_lags
        gamma = (
            np.stack(
                [
                    least_squares.coef[width_det + i * span : width_det + (i + 1) * span, :].T
                    for i in range(lags)
                ]
            )
            if lags
            else np.zeros((0, k, span), dtype=np.float64)
        )
        cursor = width_det + span * lags
        impact = (
            least_squares.coef[cursor : cursor + m, :].T
            if self.contemporaneous
            else np.zeros((k, 0), dtype=np.float64)
        )
        cursor += m * int(self.contemporaneous)
        alpha = least_squares.coef[cursor:, :].T
        short_run_det = least_squares.coef[:width_det]
        return _VectorErrorCorrectionFit(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            short_run_deterministic=short_run_det,
            impact=impact,
            eigenvalues=moments.eigenvalues,
            coefficients=self._levels_blocks(alpha, beta, gamma),
            deterministic=self._levels_deterministic(alpha, beta, short_run_det),
            sigma_u=least_squares.sigma_u,
            sigma_ml=least_squares.sigma_ml,
            design=design,
            resid=least_squares.resid,
            fittedvalues=least_squares.fittedvalues,
            llf=least_squares.llf,
            nobs=least_squares.nobs,
            n_params=k * least_squares.width + k * (k + 1) // 2,
        )

    def _levels_blocks(
        self,
        alpha: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
        gamma: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """The levels autoregressive matrices this parameterization implies.

        From ``A_1 = I + Pi + Gamma_1``, ``A_i = Gamma_i - Gamma_{i-1}``, and
        ``A_p = -Gamma_{p-1}``. Recovering them here rather than in the result
        is what lets every dynamic method downstream -- impulse responses, the
        variance decomposition, forecasts -- be inherited from the reduced-form
        surface instead of reimplemented in error-correction coordinates.

        A conditional specification has no such representation and gets a
        zero-length stack, which is the statement that there are no ``A_i`` --
        not a default, and not something a caller should propagate. Closing the
        system requires a model for the exogenous block, which is what a global
        vector autoregression supplies by stacking units.
        """
        k, p = self.k_endog, self._order
        if self.k_exog:
            return np.zeros((0, k, k), dtype=np.float64)
        blocks = np.zeros((p, k, k), dtype=np.float64)
        blocks[0] = np.eye(k) + alpha @ beta[:k].T
        if p > 1:
            blocks[0] = blocks[0] + gamma[0]
            for i in range(1, p - 1):
                blocks[i] = gamma[i] - gamma[i - 1]
            blocks[p - 1] = -gamma[p - 2]
        return blocks

    def _levels_deterministic(
        self,
        alpha: npt.NDArray[np.float64],
        beta: npt.NDArray[np.float64],
        short_run: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Deterministic coefficients expressed in levels coordinates.

        A restricted constant enters the data as ``alpha`` times the extra row
        of ``beta``; in the levels representation it is an ordinary intercept.
        Folding it out here means a forecast does not have to know which
        Johansen case produced the model. A conditional specification has no
        levels representation to fold into, and says so with a zero-length
        block.
        """
        k = self.k_endog
        case = self._trend_case
        if self.k_exog:
            return np.zeros((0, k), dtype=np.float64)
        if case == "none":
            return np.zeros((0, k), dtype=np.float64)
        if case in ("constant", "trend"):
            return short_run
        restricted = np.asarray(alpha @ beta[k], dtype=np.float64).reshape(1, k)
        if case == "restricted_constant":
            return restricted
        return np.vstack([short_run, restricted])


class _ExogenousVectorErrorCorrectionModel[R](_VectorErrorCorrectionModel[R]):
    """The conditional case, with the exogenous block required rather than optional.

    Adds nothing to the estimator. The base already carries an exogenous block
    through the cointegrating space, the lagged differences, the contemporaneous
    term, and the conditional null distribution, because a closed system is the
    special case where that block is empty rather than a different model. This
    subclass exists to make the requirement visible in the signature: a
    conditional specification without exogenous variables is a VECM, and
    silently accepting ``None`` here would let a caller believe they had asked
    for something they had not.
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        exog: npt.ArrayLike,
        *,
        order: int,
        rank: int,
        cointegration_trend: CointegrationTrend = "constant",
        contemporaneous: bool = True,
        names: Sequence[str] | None = None,
        exog_names: Sequence[str] | None = None,
    ) -> None:
        """Validate both samples and the specification.

        Args:
            endog: The ``(nobs, k_y)`` modelled variables.
            exog: The ``(nobs, k_x)`` weakly exogenous integrated regressors.
            order: Lags of the levels system.
            rank: Cointegrating rank, from ``0`` to ``k_y``.
            cointegration_trend: One of Johansen's five cases.
            contemporaneous: Whether the current exogenous difference enters.
            names: Modelled variable labels.
            exog_names: Exogenous labels.

        Raises:
            SpecificationError: If the specification is malformed.
            DimensionError: If the samples are misaligned or too short.
        """
        super().__init__(
            endog,
            order=order,
            rank=rank,
            cointegration_trend=cointegration_trend,
            exog=exog,
            contemporaneous=contemporaneous,
            names=names,
            exog_names=exog_names,
        )


class _ObservedRegimeVectorModel[R](_VectorAutoRegressionModel[R]):
    """Shared specification surface for observed-regime vector models.

    Adds one thing to the linear specification: the transition variable and
    its delay. The variable is named rather than defaulted -- a univariate
    threshold model can plausibly self-excite on its own past, but a system
    has ``k`` candidate drivers and choosing one is economics, not a default
    the estimator should quietly make.

    Args:
        endog: The observed panel.
        order: Autoregressive order within each regime, at least one.
        transition_variable: A variable name from ``names`` (the regime is
            driven by that variable's own lag -- self-exciting) or an aligned
            external series.
        delay: Delay of the transition variable.
        trend: Deterministic terms per regime.
        names: One label per variable.

    Raises:
        SpecificationError: If the specification is malformed or the named
            transition variable is unknown.
        DimensionError: If the sample cannot support two regimes.
    """

    __slots__ = ("_delays", "_threshold_name", "_transition_series")

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        transition_variable: str | npt.ArrayLike,
        delay: int | None,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
    ) -> None:
        """Validate the linear specification, then the transition driver."""
        super().__init__(endog, order=order, trend=trend, names=names)
        if self._order < 1:
            raise SpecificationError(
                f"an observed-regime model needs order >= 1; got {self._order}."
            )
        if isinstance(transition_variable, str):
            if transition_variable not in self._names:
                raise SpecificationError(
                    f"unknown transition variable {transition_variable!r}; "
                    f"expected one of {self._names} or an aligned external series."
                )
            self._threshold_name = transition_variable
            self._transition_series = self._endog[:, self._names.index(transition_variable)]
        else:
            self._threshold_name = "external"
            self._transition_series = validate_aligned(
                transition_variable, self._endog.shape[0], "transition_variable"
            )
        if delay is None:
            if not self.self_exciting:
                raise SpecificationError(
                    "the delay is searched only for a self-exciting model; with an "
                    "external transition variable the lag is a modelling choice "
                    "with economic content, so state it: pass delay explicitly."
                )
            self._delays = list(range(1, self._order + 1))
        else:
            self._delays = [validate_order(delay, "delay", minimum=1)]
        burn = self._rows_lost(self._order) + max(self._delays) - self._order
        need = 2 * (self.n_regressors + 1) + burn
        if self._endog.shape[0] < need:
            raise DimensionError(
                f"a sample of {self._endog.shape[0]} rows is too short for two "
                f"{type(self).__name__} regimes of order {self._order}; it needs "
                f"at least {need}."
            )

    @property
    def self_exciting(self) -> bool:
        """Whether the transition variable is a column of the system itself."""
        return self._threshold_name != "external"

    @property
    def transition_name(self) -> str:
        """The transition variable's label."""
        return self._threshold_name

    @property
    def transition_series(self) -> npt.NDArray[np.float64]:
        """The raw transition series, aligned with ``endog``."""
        return self._transition_series

    @property
    def delay(self) -> int | None:
        """The fixed delay, or ``None`` when the delay is searched."""
        return self._delays[0] if len(self._delays) == 1 else None

    def _regime_design(
        self, delay: int
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        int,
    ]:
        """Target, design, and aligned transition values at one delay.

        Args:
            delay: Candidate delay.

        Returns:
            The ``(n_eff, k)`` target, the ``(n_eff, width)`` design, the
            aligned delayed transition values, and the start row.
        """
        y, order = self._endog, self._order
        n = y.shape[0]
        start = max(order, delay)
        n_eff = n - start
        det = deterministic_columns(self._trend, n_eff, start=start + 1)
        design = np.column_stack([det, lag_matrix(y, order, start=start)])
        z = self._transition_series[start - delay : n - delay]
        return y[start:], design, z, start

    def _split_moments(
        self,
        target: npt.NDArray[np.float64],
        design: npt.NDArray[np.float64],
    ) -> (
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float]
        | None
    ):
        """One regime's least squares and its Gaussian log-likelihood.

        Args:
            target: The regime's rows of the target block.
            design: The regime's rows of the design.

        Returns:
            ``(coef, resid, sigma_ml, llf)``, or ``None`` when the residual
            covariance is singular and the split must be skipped.
        """
        nobs, k = target.shape
        coef, resid = self._least_squares(target, design)
        sigma_ml = resid.T @ resid / nobs
        sign, logdet = np.linalg.slogdet(sigma_ml)
        if sign <= 0:
            return None
        llf = -0.5 * nobs * k * _LOG_2PI - 0.5 * nobs * float(logdet) - 0.5 * nobs * k
        return coef, resid, sigma_ml, float(llf)


class _ThresholdVectorAutoRegressionModel[R](_ObservedRegimeVectorModel[R]):
    """Specification and grid-search engine of a two-regime threshold VAR.

    The sum-of-squares surface is a step function of the threshold --
    piecewise constant, nowhere differentiable in it -- so the estimator is
    an exhaustive grid over trimmed quantiles of the transition variable,
    with regime-wise multivariate least squares at each candidate. The
    criterion is the total Gaussian log-likelihood with a separate innovation
    covariance per regime, which is the multivariate replacement for total
    SSR: it weights the equations by their own noise rather than letting the
    noisiest series choose the split, and it lets the covariance itself
    switch, which for financial-conditions regimes is half the point.

    Args:
        endog: The observed panel.
        order: Autoregressive order within each regime.
        transition_variable: A variable name (self-exciting) or an aligned
            external series.
        delay: Threshold delay; ``None`` searches ``1..order`` jointly with
            the threshold, and is allowed only when self-exciting.
        trim: Fraction trimmed from each tail of the quantile grid, so that
            neither regime is estimated from a handful of extreme points.
        n_grid: Candidate thresholds per delay.
        trend: Deterministic terms per regime.
        names: One label per variable.
    """

    __slots__ = ("_n_grid", "_trim")

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        transition_variable: str | npt.ArrayLike,
        delay: int | None = None,
        trim: float = _DEFAULT_TRIM,
        n_grid: int = _DEFAULT_GRID,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(
            endog,
            order=order,
            transition_variable=transition_variable,
            delay=delay,
            trend=trend,
            names=names,
        )
        self._trim = validate_open_interval(trim, "trim", low=0.0, high=0.5)
        self._n_grid = validate_order(n_grid, "n_grid", minimum=1)

    def _fit_regimes(self) -> _VectorThresholdFit:
        """Fit both regimes and the split by exhaustive grid search.

        Returns:
            The packed :class:`_VectorThresholdFit`.

        Raises:
            NumericalError: If no admissible split exists.
        """
        k = self.k_endog
        width = self.n_regressors
        min_regime = width + 1
        best_llf = -np.inf
        best: tuple[int, float] | None = None
        for d in self._delays:
            target, design, z, _ = self._regime_design(d)
            n_eff = target.shape[0]
            grid = np.quantile(z, np.linspace(self._trim, 1.0 - self._trim, self._n_grid))
            for r in np.unique(grid):
                lower = z <= r
                n_lo = int(lower.sum())
                if n_lo < min_regime or n_eff - n_lo < min_regime:
                    continue
                lo = self._split_moments(target[lower], design[lower])
                hi = self._split_moments(target[~lower], design[~lower])
                if lo is None or hi is None:
                    continue
                llf = lo[3] + hi[3]
                if llf > best_llf:
                    best_llf = llf
                    best = (d, float(r))
        if best is None:
            raise NumericalError(
                "threshold grid search found no admissible split; relax trim, "
                "shorten the order, or supply a longer sample."
            )
        d_star, r_star = best
        target, design, z, _ = self._regime_design(d_star)
        lower = z <= r_star
        lo = self._split_moments(target[lower], design[lower])
        hi = self._split_moments(target[~lower], design[~lower])
        assert lo is not None and hi is not None
        n_lo, n_hi = int(lower.sum()), int((~lower).sum())
        fitted = np.empty_like(target)
        fitted[lower] = design[lower] @ lo[0]
        fitted[~lower] = design[~lower] @ hi[0]
        offset = self._n_deterministic_columns
        return _VectorThresholdFit(
            delay=d_star,
            threshold=r_star,
            threshold_values=z,
            lower_coefficients=self._lag_blocks(lo[0]),
            upper_coefficients=self._lag_blocks(hi[0]),
            lower_deterministic=lo[0][:offset],
            upper_deterministic=hi[0][:offset],
            lower_sigma_u=lo[1].T @ lo[1] / max(n_lo - width, 1),
            upper_sigma_u=hi[1].T @ hi[1] / max(n_hi - width, 1),
            n_lower=n_lo,
            n_upper=n_hi,
            resid=target - fitted,
            fittedvalues=fitted,
            llf=lo[3] + hi[3],
            nobs=target.shape[0],
            n_params=2.0 * k * width + float(k * (k + 1)) + 1.0,
        )


class _SmoothTransitionVectorAutoRegressionModel[R](_ObservedRegimeVectorModel[R]):
    """Specification and estimation engine of a smooth-transition VAR.

    Conditional on the transition speed and location the model is linear, so
    the regime coefficients are concentrated out by one multivariate solve and
    only ``(gamma, c)`` is searched, on the log-determinant criterion of
    :class:`_VectorSmoothTransitionObjective`.

    Args:
        endog: The observed panel.
        order: Autoregressive order within each regime.
        transition_variable: A variable name (self-exciting) or an aligned
            external series.
        transition: ``"logistic"`` or ``"exponential"``.
        delay: Delay of the transition variable.
        trend: Deterministic terms per regime.
        names: One label per variable.
    """

    __slots__ = ("_transition",)

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        transition_variable: str | npt.ArrayLike,
        transition: Transition = "logistic",
        delay: int = 1,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(
            endog,
            order=order,
            transition_variable=transition_variable,
            delay=delay,
            trend=trend,
            names=names,
        )
        self._transition: str = validate_choice(transition, Transition, "transition")

    @property
    def transition(self) -> str:
        """The transition function family."""
        return self._transition

    def _build_objective(self) -> _VectorSmoothTransitionObjective:
        """Assemble the concentrated surface, seeded near the best hard split.

        Returns:
            The configured objective.

        Raises:
            NumericalError: If the transition variable has zero variance.
        """
        delay = self._delays[0]
        target, design, z, _ = self._regime_design(delay)
        scale = float(np.std(z))
        if scale == 0.0:
            raise NumericalError("the transition variable has zero variance.")
        width = design.shape[1]
        min_regime = width + 1
        n_eff = target.shape[0]
        c_seed = float(np.median(z))
        best_hard = -np.inf
        for candidate in np.quantile(z, np.linspace(0.15, 0.85, 50)):
            lower = z <= candidate
            n_lo = int(lower.sum())
            if n_lo < min_regime or n_eff - n_lo < min_regime:
                continue
            lo = self._split_moments(target[lower], design[lower])
            hi = self._split_moments(target[~lower], design[~lower])
            if lo is None or hi is None:
                continue
            if lo[3] + hi[3] > best_hard:
                best_hard, c_seed = lo[3] + hi[3], float(candidate)
        seeds = tuple(
            np.array([np.log(gamma0), c0])
            for c0 in (c_seed, float(np.median(z)))
            for gamma0 in (2.0, 5.0, 10.0, 25.0)
        )
        return _VectorSmoothTransitionObjective(
            target=target,
            design=design,
            z=z,
            scale=scale,
            transition=self._transition,
            seeds=seeds,
        )

    def _fit_regimes(self) -> _VectorSmoothTransitionFit:
        """Fit the transition by concentrated maximum likelihood.

        Returns:
            The packed :class:`_VectorSmoothTransitionFit`.

        Raises:
            NumericalError: If the transition variable has zero variance or
                every start lands on a singular covariance.
        """
        k = self.k_endog
        delay = self._delays[0]
        objective = self._build_objective()
        parameters, logdet = _solve(objective)
        if not np.isfinite(logdet):
            raise NumericalError(
                "every start of the smooth-transition search produced a singular "
                "residual covariance; the regimes are not separable on this sample."
            )
        _, coef, resid = objective.concentrated(parameters)
        n_eff = objective.target.shape[0]
        width = objective.design.shape[1]
        offset = self._n_deterministic_columns
        llf = -0.5 * n_eff * k * _LOG_2PI - 0.5 * n_eff * logdet - 0.5 * n_eff * k
        return _VectorSmoothTransitionFit(
            delay=delay,
            threshold=parameters.threshold,
            threshold_values=objective.z,
            gamma=parameters.gamma,
            transition_scale=objective.scale,
            lower_coefficients=self._lag_blocks(coef[:width]),
            upper_coefficients=self._lag_blocks(coef[width:]),
            lower_deterministic=coef[:width][:offset],
            upper_deterministic=coef[width:][:offset],
            sigma_u=resid.T @ resid / max(n_eff - 2 * width, 1),
            resid=resid,
            fittedvalues=objective.target - resid,
            llf=float(llf),
            nobs=n_eff,
            n_params=2.0 * k * width + k * (k + 1) / 2.0 + 2.0,
        )


class _FunctionalCoefficientVectorAutoRegressionModel[R](_ObservedRegimeVectorModel[R]):
    """Specification and local-linear engine of a functional-coefficient VAR.

    Subclassing :class:`_ObservedRegimeVectorModel` is deliberate: what that
    base actually contributes is the observed-transition-driver surface --
    the named-or-external state variable, its delay, and the aligned design
    construction -- and this model is the nonparametric limit of the family
    built on it. A threshold VAR gives every state value one of two systems
    and a smooth-transition VAR a blend of two anchors; here every state
    value gets its own system, with smoothness in the state the only
    restriction. Nothing regime-shaped is inherited beyond the driver.

    Estimation is local-linear kernel regression (Cai, Fan & Yao 2000): at
    each evaluation point the coefficients and their state-derivatives are
    solved from one Epanechnikov-weighted least squares on the augmented
    design ``[x_t, x_t (z_t - u)]``, all equations sharing the weights. The
    bandwidth -- the model's one tuning constant -- is chosen by exact
    leave-one-out cross-validation unless the caller states it.

    Args:
        endog: The observed panel.
        order: Autoregressive order, at least one.
        transition_variable: A variable name from ``names`` (the state is
            that variable's own lag) or an aligned external series.
        delay: Delay of the state variable. Defaults to one; there is no
            delay search here, because the bandwidth is this model's tuning
            axis and searching both would let the smoother trade one against
            the other invisibly.
        trend: Deterministic terms.
        names: One label per variable.

    Raises:
        SpecificationError: If the specification is malformed.
        DimensionError: If the sample cannot support local-linear fits.
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        transition_variable: str | npt.ArrayLike,
        delay: int = 1,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
    ) -> None:
        """Validate the linear and driver specification, then the sample length."""
        super().__init__(
            endog,
            order=order,
            transition_variable=transition_variable,
            delay=delay,
            trend=trend,
            names=names,
        )
        n_eff = self._endog.shape[0] - max(self._order, delay)
        need = 3 * 2 * self.n_regressors
        if n_eff < need:
            raise DimensionError(
                f"an effective sample of {n_eff} rows is too short for "
                f"local-linear estimation with {2 * self.n_regressors} "
                f"coefficients per window; it needs at least {need}."
            )

    @staticmethod
    def _epanechnikov(scaled: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The Epanechnikov kernel on already-bandwidth-scaled distances."""
        return np.where(np.abs(scaled) < 1.0, 0.75 * (1.0 - scaled**2), 0.0)

    @property
    def _state_in_design(self) -> bool:
        """Whether the state variable is exactly a column of the design.

        True for a self-exciting model whose delay does not exceed the
        order: the state ``z_t`` is then the design's own lag column
        ``name.L{delay}``, and the intercept's local-linear interaction
        ``1 * (z - u) = name.L{delay} - u`` is *exactly* collinear with the
        design. This is a structural fact about the model, not a numerical
        accident -- with the state among the regressors, the intercept
        function and the state's own coefficient function are only jointly
        identified at first order -- so the redundant regressor is dropped
        by construction rather than left to a pseudoinverse to resolve
        silently.
        """
        return (
            self._n_deterministic_columns > 0
            and self.self_exciting
            and self._delays[0] <= self._order
        )

    def _interaction_columns(self, design: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The design columns whose state-derivatives are identified.

        The intercept column is excluded when the state is a design column
        (see :attr:`_state_in_design`); its coefficient is then fitted
        local-constant while every other coefficient stays local-linear.
        """
        return design[:, 1:] if self._state_in_design else design

    def _local_fit(
        self,
        target: npt.NDArray[np.float64],
        design: npt.NDArray[np.float64],
        interactions: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        u: float,
        bandwidth: float,
    ) -> (
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ]
        | None
    ):
        """One local-linear weighted least squares, at one evaluation point.

        Args:
            target: The ``(n, k)`` target block.
            design: The ``(n, w)`` linear design.
            interactions: The design columns given state-derivatives, from
                :meth:`_interaction_columns`.
            z: The aligned state values.
            u: Evaluation point.
            bandwidth: Kernel bandwidth.

        Returns:
            ``(coef, gram, weights, augmented)`` -- the coefficients with
            the ``w`` levels leading, the weighted Gram matrix, the kernel
            weights, and the augmented design -- or ``None`` when the window
            holds too few points to identify the local system, or holds them
            in a numerically degenerate arrangement.
        """
        weights = self._epanechnikov((z - u) / bandwidth)
        width = design.shape[1] + interactions.shape[1]
        if int(np.count_nonzero(weights)) <= width:
            return None
        augmented = np.column_stack([design, interactions * (z - u)[:, None]])
        weighted = augmented * weights[:, None]
        gram = weighted.T @ augmented
        eigenvalues = np.linalg.eigvalsh(gram)
        if eigenvalues[0] <= 1e-9 * max(float(eigenvalues[-1]), _TINY):
            return None
        coef = np.linalg.solve(gram, weighted.T @ target)
        return coef, gram, weights, augmented

    def _cross_validation_score(
        self,
        target: npt.NDArray[np.float64],
        design: npt.NDArray[np.float64],
        interactions: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        scored: npt.NDArray[np.bool_],
        bandwidth: float,
    ) -> float:
        """Exact leave-one-out squared error of one candidate bandwidth.

        The deletion residual is computed from the full-sample local fit via
        the smoother diagonal, ``(y_t - yhat_t) / (1 - H_tt)`` -- exact for a
        linear smoother, so no refit per left-out point is needed. Only the
        ``scored`` points -- those inside the trimmed state range, where the
        curves will actually be read -- enter the criterion, so an isolated
        extreme state neither vetoes every candidate nor drags the selection
        toward the oversmoothing that lone point would demand.

        Args:
            target: The ``(n, k)`` target block.
            design: The ``(n, w)`` linear design.
            interactions: The design columns given state-derivatives.
            z: The aligned state values.
            scored: Which observations enter the criterion.
            bandwidth: Candidate bandwidth.

        Returns:
            The summed squared deletion residuals over the scored points, or
            ``inf`` when any scored local system is unidentified or
            interpolates its own point.
        """
        total = 0.0
        for t in range(target.shape[0]):
            if not scored[t]:
                continue
            out = self._local_fit(target, design, interactions, z, float(z[t]), bandwidth)
            if out is None:
                return float("inf")
            coef, gram, weights, augmented = out
            row = augmented[t]
            try:
                leverage = float(weights[t] * (row @ np.linalg.solve(gram, row)))
            except np.linalg.LinAlgError:
                return float("inf")
            if leverage >= 1.0 - 1e-8:
                return float("inf")
            deletion = (target[t] - row @ coef) / (1.0 - leverage)
            total += float(deletion @ deletion)
        return total

    def _fit_functional(
        self,
        *,
        bandwidth: float | None,
        n_grid: int,
        trim: float,
    ) -> _VectorFunctionalFit:
        """Estimate the coefficient curves, selecting the bandwidth if unstated.

        Args:
            bandwidth: Kernel bandwidth in the state variable's units, or
                ``None`` to select by leave-one-out cross-validation on a
                grid around the Silverman-scaled rule of thumb.
            n_grid: Evaluation points for the reported curves.
            trim: Quantile trimmed from each end of the state range before
                the curve grid is laid down, so the curves are not read into
                regions the kernel cannot populate.

        Everything local is anchored to the trimmed state range: the
        curves are evaluated on it, the cross-validation criterion is scored
        on it, and observations whose state falls outside it -- or whose
        window is degenerate -- take the nearest estimated system for their
        fitted value rather than vetoing the whole fit, which is what one
        isolated state excursion would otherwise do at any honest bandwidth.

        Returns:
            The packed :class:`_VectorFunctionalFit`.

        Raises:
            SpecificationError: If the grid or trim is malformed, or a
                stated bandwidth is not positive.
            NumericalError: If no candidate bandwidth identifies every local
                system on the trimmed range, or the stated one does not.
        """
        if n_grid < 2:
            raise SpecificationError(f"n_grid must be at least 2; got {n_grid}.")
        if not 0.0 <= trim < 0.5:
            raise SpecificationError(f"trim must lie in [0, 0.5); got {trim}.")
        delay = self._delays[0]
        target, design, z, _ = self._regime_design(delay)
        interactions = self._interaction_columns(design)
        n_eff, k = target.shape
        width = design.shape[1]
        low, high = (float(q) for q in np.quantile(z, (trim, 1.0 - trim)))
        scored = (z >= low) & (z <= high)
        searched = bandwidth is None
        if bandwidth is None:
            rule = 2.34 * float(np.std(z)) * n_eff ** (-0.2)
            candidates = rule * np.geomspace(0.3, 3.0, 13)
            scores = [
                self._cross_validation_score(target, design, interactions, z, scored, float(h))
                for h in candidates
            ]
            best = int(np.argmin(scores))
            if not np.isfinite(scores[best]):
                raise NumericalError(
                    "no candidate bandwidth identified every local system on "
                    "the trimmed state range; the state variable is too "
                    "sparse for local-linear estimation at this order. State "
                    "a larger bandwidth."
                )
            bandwidth = float(candidates[best])
        elif bandwidth <= 0.0:
            raise SpecificationError(f"bandwidth must be positive; got {bandwidth}.")
        grid = np.linspace(low, high, n_grid)
        curves = np.empty((n_grid, width, k))
        curve_se = np.empty((n_grid, width, k))
        for g, u in enumerate(grid):
            out = self._local_fit(target, design, interactions, z, float(u), bandwidth)
            if out is None:
                raise NumericalError(
                    f"the local system at state value {u:.6g} is unidentified "
                    f"at bandwidth {bandwidth:.6g}; widen the bandwidth or "
                    "raise trim."
                )  # inside the trimmed range, so this is a genuine gap
            coef, gram, weights, augmented = out
            curves[g] = coef[:width]
            inverse = np.linalg.inv(gram)
            core = inverse @ (augmented.T @ (augmented * (weights**2)[:, None])) @ inverse
            local_resid = target - augmented @ coef
            variance = (weights[:, None] * local_resid**2).sum(axis=0) / weights.sum()
            curve_se[g] = np.sqrt(np.outer(np.maximum(np.diag(core)[:width], 0.0), variance))
        fitted = np.empty_like(target)
        effective = 0.0
        for t in range(n_eff):
            out = (
                self._local_fit(target, design, interactions, z, float(z[t]), bandwidth)
                if scored[t]
                else None
            )
            if out is None:
                position = int(np.clip(np.searchsorted(grid, z[t]), 1, n_grid - 1))
                left, right = grid[position - 1], grid[position]
                share = float(np.clip((z[t] - left) / (right - left), 0.0, 1.0))
                blended = (1.0 - share) * curves[position - 1] + share * curves[position]
                fitted[t] = design[t] @ blended
                continue
            coef, gram, weights, augmented = out
            row = augmented[t]
            fitted[t] = row @ coef
            effective += float(weights[t] * (row @ np.linalg.solve(gram, row)))
        resid = target - fitted
        dof = max(n_eff - effective, 1.0)
        return _VectorFunctionalFit(
            delay=delay,
            state_values=z,
            grid=grid,
            curves=curves,
            curve_se=curve_se,
            bandwidth=float(bandwidth),
            bandwidth_searched=searched,
            effective_params=effective,
            sigma_u=resid.T @ resid / dof,
            resid=resid,
            fittedvalues=fitted,
            nobs=n_eff,
        )


class _QuantileVectorAutoRegressionModel[R](_VectorAutoRegressionModel[R]):
    """Specification and estimation engine of a quantile VAR.

    Each equation's coefficient vector minimizes the check loss at each
    requested quantile level -- an exact linear program (Koenker & Bassett
    1978), solved by HiGHS in its standard primal form: split the residual
    into its positive and negative parts, price them at ``tau`` and
    ``1 - tau``, and constrain ``X beta + u - v = y``. Equations share one
    design and separate at estimation because the check loss is additive
    across them; quantile levels separate for the same reason, so the fit is
    ``k * Q`` independent programs over one design.

    Args:
        endog: The observed panel.
        order: Autoregressive order.
        trend: Deterministic terms.
        names: One label per variable.

    Raises:
        SpecificationError: If the specification is malformed.
        DimensionError: If the sample cannot support the specification.
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
    ) -> None:
        """Validate the linear specification, without the prior surface.

        A quantile VAR is re-specified here rather than inherited verbatim
        because the base's shrinkage prior is a Gaussian belief about
        conditional-mean coefficients; it has no meaning for a check-loss
        program, so the surface refuses it by not offering it.
        """
        super().__init__(endog, order=order, trend=trend, names=names)

    @staticmethod
    def _check_loss(resid: npt.NDArray[np.float64], quantile: float) -> npt.NDArray[np.float64]:
        """Total check loss per column of a residual block."""
        return np.where(resid >= 0.0, quantile * resid, (quantile - 1.0) * resid).sum(axis=0)

    @staticmethod
    def _quantile_regression(
        target: npt.NDArray[np.float64],
        design: npt.NDArray[np.float64],
        quantile: float,
    ) -> npt.NDArray[np.float64]:
        """One equation's exact quantile regression, as a linear program.

        Args:
            target: The ``(n,)`` regressand.
            design: The ``(n, w)`` regressor matrix.
            quantile: The quantile level, strictly inside ``(0, 1)``.

        Returns:
            The ``(w,)`` coefficient vector.

        Raises:
            NumericalError: If the program does not solve to optimality.
        """
        n, width = design.shape
        cost = np.concatenate([np.zeros(width), np.full(n, quantile), np.full(n, 1.0 - quantile)])
        identity = sps.eye_array(n, format="csc")
        equality = sps.hstack([sps.csc_array(design), identity, -identity], format="csc")
        bounds: list[tuple[float | None, float | None]] = [(None, None)] * width + [(0.0, None)] * (
            2 * n
        )
        # HiGHS accepts a sparse A_eq at runtime (scipy documents this); the
        # published stubs admit only dense inputs, so the cast records a stub
        # gap, not a type weakening -- densifying here would cost O(n^2)
        # memory for a constraint matrix that is (w + 2) / (w + 2n) dense.
        solution = linprog(
            cost,
            A_eq=cast("npt.NDArray[np.float64]", equality),
            b_eq=target,
            bounds=bounds,
            method="highs",
        )
        if not solution.success or solution.x is None:
            raise NumericalError(
                f"the quantile regression program at tau={quantile} did not "
                f"solve to optimality: {solution.message}"
            )
        return np.asarray(solution.x[:width], dtype=np.float64)

    def _fit_quantile(self, quantiles: Sequence[float]) -> _VectorQuantileFit:
        """Estimate every equation at every requested quantile level.

        Args:
            quantiles: Distinct quantile levels, each strictly inside
                ``(0, 1)``.

        Returns:
            The packed :class:`_VectorQuantileFit`, quantiles ascending.

        Raises:
            SpecificationError: If the levels are empty, repeated, or
                outside the open unit interval.
            NumericalError: If any program does not solve.
        """
        raw = tuple(
            validate_open_interval(float(q), "quantiles", low=0.0, high=1.0) for q in quantiles
        )
        if not raw:
            raise SpecificationError("quantiles must name at least one level.")
        if len(set(raw)) != len(raw):
            raise SpecificationError(f"quantiles must be distinct; got {raw}.")
        taus = tuple(sorted(raw))
        target, design, n_eff = self._design()
        k, width, offset = self.k_endog, self.n_regressors, self._n_deterministic_columns
        n_taus = len(taus)
        stacks = np.empty((n_taus, self._order, k, k))
        deterministics = np.empty((n_taus, offset, k))
        fitted = np.empty((n_taus, n_eff, k))
        loss = np.empty((n_taus, k))
        loss_location = np.empty((n_taus, k))
        for qi, tau in enumerate(taus):
            coef = np.empty((width, k))
            for i in range(k):
                coef[:, i] = self._quantile_regression(target[:, i], design, tau)
            fitted[qi] = design @ coef
            stacks[qi] = self._lag_blocks(coef)
            deterministics[qi] = coef[:offset]
            loss[qi] = self._check_loss(target - fitted[qi], tau)
            location = np.quantile(target, tau, axis=0)
            loss_location[qi] = self._check_loss(target - location, tau)
        return _VectorQuantileFit(
            quantiles=taus,
            coefficient_stacks=stacks,
            deterministics=deterministics,
            fittedvalues=fitted,
            resid=target[None] - fitted,
            loss=loss,
            loss_location=loss_location,
            nobs=n_eff,
        )


class _GibbsBayesianVectorAutoRegressionModel[R](_VectorAutoRegressionModel[R]):
    """Estimation engine of the non-conjugate Gibbs-sampled BVAR.

    The coefficient prior here is stated *independently* of the innovation
    covariance -- the independent Normal-Wishart pairing -- which is what
    admits every prior the conjugate model must refuse: Litterman's
    cross-equation weight, and the adaptive shrinkage hierarchies whose
    variances are latent states rather than constants. The price is paid
    honestly. The posterior is reached by Gibbs sampling instead of exactly
    -- coefficients given covariance by one *joint* generalized-least-squares
    draw across all equations (exact, ordering-free), covariance given
    coefficients by inverse-Wishart, and, for an adaptive prior, its scale
    hierarchy by its own exact conditionals -- and no marginal likelihood is
    reported, because with the prior independent of the covariance the
    evidence has no closed form and a simulated stand-in would not deserve
    the name.

    The joint coefficient draw factorizes nothing, so each sweep costs a
    Cholesky of the ``(k * w, k * w)`` conditional precision. That is the
    known cost of exactness at this generality (the corrigendum literature
    is the cautionary tale for shortcuts), and it bounds the comfortable
    system size well below the conjugate model's.

    Dummy-observation rows from a static prior are stacked under the sample
    and weighted by ``Sigma`` like any other row, which is exactly the
    conjugate model's treatment; adaptive priors state no dummies by
    construction.
    """

    __slots__ = ()

    def _gibbs_static_inputs(
        self, context: _PriorContext
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Validate a static prior and return its moments and dummy rows.

        Returns:
            ``(mean, variance, dummy_target, dummy_design)``.

        Raises:
            SpecificationError: If the prior is improper or mixes an
                adaptive component into a composition.
        """
        prior = self._prior
        if any(isinstance(part, _AdaptivePrior) for part in prior._components()):
            raise SpecificationError(
                "adaptive shrinkage priors do not compose: a composition has "
                "no scale conditionals to sample. Pass the adaptive prior "
                "alone, or compose only moment-and-dummy priors."
            )
        variance = prior.coefficient_variance(context)
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
            raise SpecificationError(
                "the prior leaves some coefficient variances infinite or "
                "non-positive, so it is improper and has no posterior to "
                "sample from; give every coefficient a finite prior variance."
            )
        dummy_target, dummy_design = prior.dummy_observations(context)
        return prior.coefficient_mean(context), variance, dummy_target, dummy_design

    def _fit_gibbs(
        self,
        *,
        n_draws: int,
        n_burn: int,
        thin: int,
        seed: int | np.random.Generator | None,
    ) -> _VectorGibbsFit:
        """Gibbs over coefficients, covariance, and any adaptive scale layer.

        Args:
            n_draws: Total sampler iterations.
            n_burn: Burn-in iterations discarded.
            thin: Keep every ``thin``-th post-burn draw.
            seed: Seed or generator.

        Returns:
            The packed :class:`_VectorGibbsFit`.

        Raises:
            SpecificationError: If the draw bookkeeping is inconsistent or
                the prior is unusable.
            NumericalError: If a conditional draw collapses.
        """
        if n_draws <= n_burn:
            raise SpecificationError(f"n_draws ({n_draws}) must exceed n_burn ({n_burn}).")
        if thin < 1:
            raise SpecificationError(f"thin must be at least 1; got {thin}.")
        prior = self._prior
        if not prior._components():
            raise SpecificationError(
                "a Bayesian VAR needs a proper prior; construct with "
                "prior=IndependentNormalWishartPrior(...), an adaptive "
                "shrinkage prior, or a composition -- an improper prior has "
                "no posterior to sample from."
            )
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        target, design, n_eff = self._design()
        context = self._prior_context()
        k, width = self.k_endog, self.n_regressors
        adaptive = prior if isinstance(prior, _AdaptivePrior) else None
        if adaptive is not None:
            gram_raw = design.T @ design
            ridge = 1e-8 * float(np.trace(gram_raw)) / width
            inverse = np.linalg.inv(gram_raw + ridge * np.eye(width))
            reference = (
                np.sqrt(np.maximum(np.diag(inverse), 0.0))[:, None] * context.scales[None, :]
            )
            ratio = adaptive._unit_ratio(context)
            state = adaptive._initial_scales(context, reference)
            mean = adaptive.coefficient_mean(context)
            variance = adaptive._scale_variance(state, context)
            dummy_target = np.zeros((0, k), dtype=np.float64)
            dummy_design = np.zeros((0, width), dtype=np.float64)
        else:
            mean, variance, dummy_target, dummy_design = self._gibbs_static_inputs(context)
            ratio = np.ones((width, k), dtype=np.float64)
            state = {}
        n_dummy = int(dummy_target.shape[0])
        if n_dummy:
            full_target = np.vstack([target, dummy_target])
            full_design = np.vstack([design, dummy_design])
        else:
            full_target, full_design = target, design
        n_samp = n_eff + n_dummy
        gram = full_design.T @ full_design
        moment = full_design.T @ full_target
        scale0 = np.diag(context.scales**2)
        df0 = float(k + 2)
        sigma = scale0.copy()
        mean_vector = mean.T.ravel()
        keep = (n_draws - n_burn + thin - 1) // thin
        beta_kept = np.empty((keep, width, k))
        sigma_kept = np.empty((keep, k, k))
        tracked_sum = np.zeros((width, k))
        kept = 0
        for iteration in range(n_draws):
            precision_vector = (1.0 / variance).T.ravel()
            sigma_inv = np.linalg.inv(sigma)
            big = np.kron(sigma_inv, gram)
            big[np.diag_indices_from(big)] += precision_vector
            rhs = (sigma_inv @ moment.T).ravel() + precision_vector * mean_vector
            try:
                chol = np.linalg.cholesky(big)
            except np.linalg.LinAlgError as error:
                raise NumericalError(
                    "the joint coefficient conditional lost positive "
                    "definiteness; the sampler has collapsed."
                ) from error
            solution = np.linalg.solve(big, rhs)
            shock = np.asarray(rng.standard_normal(k * width), dtype=np.float64)
            beta = (solution + np.linalg.solve(chol.T, shock)).reshape(k, width).T
            resid_all = full_target - full_design @ beta
            sigma = _draw_inverse_wishart(scale0 + resid_all.T @ resid_all, df0 + n_samp, rng)
            if adaptive is not None:
                state = adaptive._draw_scales(beta / ratio, context, rng, state)
                variance = adaptive._scale_variance(state, context)
            if iteration >= n_burn and (iteration - n_burn) % thin == 0:
                beta_kept[kept] = beta
                sigma_kept[kept] = sigma
                if adaptive is not None:
                    tracked_sum += adaptive._tracked(state, context)
                kept += 1
        beta_mean = beta_kept.mean(axis=0)
        fitted = design @ beta_mean
        if adaptive is not None:
            shrinkage = tracked_sum / keep
            shrinkage_label = adaptive._tracked_label()
        else:
            shrinkage = np.zeros((0, 0), dtype=np.float64)
            shrinkage_label = ""
        return _VectorGibbsFit(
            coefficient_stack=self._lag_blocks(beta_mean),
            deterministic=beta_mean[: self._n_deterministic_columns],
            beta_mean=beta_mean,
            sigma_u=sigma_kept.mean(axis=0),
            beta_draws=beta_kept,
            sigma_draws=sigma_kept,
            shrinkage=shrinkage,
            shrinkage_label=shrinkage_label,
            resid=target - fitted,
            fittedvalues=fitted,
            nobs=n_eff,
            n_dummy=n_dummy,
            n_draws=n_draws,
            n_burn=n_burn,
            thin=thin,
        )


class _SparseVectorAutoRegressionModel[R](_VectorAutoRegressionModel[R]):
    """Estimation engine of the penalized (sparse) VAR family.

    One solver serves five penalties. The lasso and the lag-group penalty
    are solved directly by accelerated proximal gradient on the whole
    coefficient matrix at once; the adaptive lasso reweights from a ridge
    pilot; SCAD and MCP are solved by local linear approximation (Zou & Li
    2008) -- a short sequence of weighted lasso solves whose weights are the
    nonconvex penalty's derivative at the current solution, which is the
    standard route to their oracle behavior without nonconvex optimization
    folklore. Deterministic terms are never penalized, and both the design
    and the targets are standardized internally so one penalty level means
    the same thing in every equation.

    The penalty level, when unstated, is chosen by rolling-origin one-step
    forecast cross-validation with refitting at every origin (the
    Nicholson-Matteson-Bien scheme): genuine out-of-sample errors, warm
    starts along the path keeping the cost civil.
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
    ) -> None:
        """Validate the linear specification, without the prior surface.

        A penalized VAR is re-specified here rather than inherited verbatim
        because shrinkage arrives through the penalty, not through a
        Gaussian prior; offering both would make one silently modify the
        other.
        """
        super().__init__(endog, order=order, trend=trend, names=names)

    def _lag_groups(self) -> tuple[npt.NDArray[np.intp], ...]:
        """Row blocks of the lag-group penalty: one block per lag matrix."""
        offset, k = self._n_deterministic_columns, self.k_endog
        return tuple(
            np.arange(offset + lag * k, offset + (lag + 1) * k, dtype=np.intp)
            for lag in range(self._order)
        )

    @staticmethod
    def _nonconvex_weights(
        beta: npt.NDArray[np.float64],
        lam: float,
        penalty: str,
        mask: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Local-linear-approximation weights at the current solution.

        The derivative of the SCAD (``a = 3.7``) or MCP (``gamma = 3``)
        penalty at each coefficient's magnitude, scaled to the unit-weight
        convention of the solver, and masked so free rows stay free.
        """
        magnitude = np.abs(beta)
        if penalty == "scad":
            a = 3.7
            slope = np.where(
                magnitude <= lam,
                1.0,
                np.maximum(a * lam - magnitude, 0.0) / ((a - 1.0) * lam),
            )
        else:
            gamma = 3.0
            slope = np.maximum(1.0 - magnitude / (gamma * lam), 0.0)
        return slope * mask[:, None]

    def _solve_penalized(
        self,
        gram: npt.NDArray[np.float64],
        moment: npt.NDArray[np.float64],
        *,
        penalty: str,
        lam: float,
        mask: npt.NDArray[np.float64],
        groups: tuple[npt.NDArray[np.intp], ...] | None,
        start: npt.NDArray[np.float64] | None,
    ) -> npt.NDArray[np.float64]:
        """One penalized solve at one penalty level, penalty family dispatched.

        Args:
            gram: ``(w, w)`` standardized Gram matrix.
            moment: ``(w, k)`` standardized cross-moment.
            penalty: The penalty family.
            lam: Penalty level.
            mask: ``(w,)`` ones on penalized rows, zeros on free rows.
            groups: Lag-group blocks, for the group penalty only.
            start: Warm start.

        Returns:
            The ``(w, k)`` standardized solution.
        """
        if penalty in ("lasso", "group"):
            return _fista_penalized(
                gram,
                moment,
                lam=lam,
                weights=mask,
                groups=groups if penalty == "group" else None,
                start=start,
            )
        if penalty == "adaptive":
            ridge = np.linalg.solve(gram + 0.01 * np.eye(gram.shape[0]), moment)
            weights = mask[:, None] / np.maximum(np.abs(ridge), 1e-3)
            return _fista_penalized(gram, moment, lam=lam, weights=weights, start=start)
        beta = _fista_penalized(gram, moment, lam=lam, weights=mask, start=start)
        for _ in range(2):
            weights = self._nonconvex_weights(beta, lam, penalty, mask)
            beta = _fista_penalized(gram, moment, lam=lam, weights=weights, start=beta)
        return beta

    def _fit_sparse(
        self,
        *,
        penalty: str,
        lam: float | None,
        n_lambdas: int,
        lambda_min_ratio: float,
    ) -> _VectorSparseFit:
        """Estimate the penalized system, selecting the level if unstated.

        Args:
            penalty: One of ``lasso``, ``adaptive``, ``scad``, ``mcp``,
                ``group``.
            lam: Penalty level, or ``None`` for rolling-origin selection.
            n_lambdas: Candidate levels on the geometric path.
            lambda_min_ratio: Smallest candidate as a fraction of the level
                that zeroes everything.

        Returns:
            The packed :class:`_VectorSparseFit`.

        Raises:
            SpecificationError: If the penalty name, level, or path
                specification is malformed, or the sample cannot support
                the rolling validation.
        """
        choice = validate_choice(penalty, Penalty, "penalty")
        if lam is not None and lam <= 0.0:
            raise SpecificationError(f"lam must be positive when given; got {lam}.")
        if n_lambdas < 2 or not 0.0 < lambda_min_ratio < 1.0:
            raise SpecificationError(
                f"the path needs n_lambdas >= 2 and lambda_min_ratio in "
                f"(0, 1); got {n_lambdas}, {lambda_min_ratio}."
            )
        if choice == "group" and self._order == 0:
            raise SpecificationError("the lag-group penalty needs order >= 1.")
        target, design, n_eff = self._design()
        offset, k, width = self._n_deterministic_columns, self.k_endog, self.n_regressors
        x_scale = design.std(axis=0, ddof=0)
        x_scale[:offset] = 1.0
        x_scale = np.where(x_scale > 0.0, x_scale, 1.0)
        y_scale = target.std(axis=0, ddof=0)
        y_scale = np.where(y_scale > 0.0, y_scale, 1.0)
        xs = design / x_scale
        ys = target / y_scale
        mask = np.ones(width)
        mask[:offset] = 0.0
        groups = self._lag_groups() if choice == "group" else None
        if lam is None:
            held_out = min(40, max(20, n_eff // 5))
            if n_eff - held_out < width // 2 + 5:
                held_out = n_eff - (width // 2 + 5)
            if held_out < 5:
                raise SpecificationError(
                    "the sample is too short for rolling-origin penalty "
                    "selection; state lam explicitly."
                )
            origin = n_eff - held_out
            gram_sum = xs[:origin].T @ xs[:origin]
            moment_sum = xs[:origin].T @ ys[:origin]
            reference = np.abs(moment_sum / origin)[mask > 0.0]
            lam_max = float(reference.max()) * 1.05
            path = np.geomspace(lam_max, lam_max * lambda_min_ratio, n_lambdas)
            errors = np.zeros(n_lambdas)
            warm: list[npt.NDArray[np.float64] | None] = [None] * n_lambdas
            for t in range(origin, n_eff):
                gram = gram_sum / t
                moment = moment_sum / t
                for position, level in enumerate(path):
                    warm[position] = self._solve_penalized(
                        gram,
                        moment,
                        penalty=choice,
                        lam=float(level),
                        mask=mask,
                        groups=groups,
                        start=warm[position],
                    )
                    forecast_error = ys[t] - xs[t] @ warm[position]
                    errors[position] += float(forecast_error @ forecast_error)
                gram_sum += np.outer(xs[t], xs[t])
                moment_sum += np.outer(xs[t], ys[t])
            best = int(np.argmin(errors))
            lam = float(path[best])
        else:
            path = np.zeros(0)
            errors = np.zeros(0)
        gram = xs.T @ xs / n_eff
        moment = xs.T @ ys / n_eff
        solution = self._solve_penalized(
            gram, moment, penalty=choice, lam=lam, mask=mask, groups=groups, start=None
        )
        coef = solution * y_scale[None, :] / x_scale[:, None]
        fitted = design @ coef
        resid = target - fitted
        nonzero = int(np.count_nonzero(solution[offset:]))
        spent = offset * k + nonzero
        return _VectorSparseFit(
            coefficient_stack=self._lag_blocks(coef),
            deterministic=coef[:offset],
            sigma_u=resid.T @ resid / max(n_eff - spent / k, 1.0),
            resid=resid,
            fittedvalues=fitted,
            penalty=choice,
            lam=float(lam),
            lambda_path=path,
            cv_errors=errors,
            n_nonzero=nonzero,
            nobs=n_eff,
        )

    def _fit_graphical(
        self,
        *,
        lam: float | None,
        n_lambdas: int,
        lambda_min_ratio: float,
        n_folds: int,
        seed: int | np.random.Generator | None,
    ) -> _VectorGraphicalFit:
        """Sparse dynamics, then a sparse residual precision by nodewise lasso.

        Stage one is the lasso VAR of :meth:`_fit_sparse`; stage two runs
        Meinshausen-Buhlmann nodewise regressions on its residuals -- each
        residual on all the others, lasso-penalized, penalty chosen by plain
        K-fold cross-validation, which is legitimate here precisely because
        residual rows carry no serial ordering worth respecting once the
        dynamics are removed. The precision follows from the nodewise slopes
        and is symmetrized by averaging.

        Args:
            lam: Stage-one penalty level, or ``None`` for rolling selection.
            n_lambdas: Candidate levels for both stages' paths.
            lambda_min_ratio: Path floor as a fraction of the zeroing level.
            n_folds: Cross-validation folds for the nodewise stage.
            seed: Seed or generator for the fold shuffle.

        Returns:
            The packed :class:`_VectorGraphicalFit`.

        Raises:
            SpecificationError: If a stage's specification is malformed.
        """
        if n_folds < 2:
            raise SpecificationError(f"n_folds must be at least 2; got {n_folds}.")
        stage_one = self._fit_sparse(
            penalty="lasso",
            lam=lam,
            n_lambdas=n_lambdas,
            lambda_min_ratio=lambda_min_ratio,
        )
        resid = stage_one.resid
        n, k = resid.shape
        rng = np.random.default_rng(seed)
        assignment = rng.permutation(n) % n_folds
        slopes = np.zeros((k, k))
        node_variance = np.empty(k)
        lam_nodes = np.empty(k)
        ones = np.ones(1)
        for i in range(k):
            others = np.delete(np.arange(k), i)
            x_node = resid[:, others]
            scale = x_node.std(axis=0, ddof=0)
            scale = np.where(scale > 0.0, scale, 1.0)
            x_node = x_node / scale
            y_node = resid[:, i][:, None]
            lam_max = float(np.abs(x_node.T @ y_node / n).max()) * 1.05
            path = np.geomspace(lam_max, lam_max * lambda_min_ratio, n_lambdas)
            errors = np.zeros(n_lambdas)
            for fold in range(n_folds):
                train = assignment != fold
                rows = int(train.sum())
                gram = x_node[train].T @ x_node[train] / rows
                moment = x_node[train].T @ y_node[train] / rows
                start = None
                for position, level in enumerate(path):
                    start = _fista_penalized(
                        gram,
                        moment,
                        lam=float(level),
                        weights=ones.repeat(k - 1),
                        start=start,
                    )
                    held = y_node[~train] - x_node[~train] @ start
                    errors[position] += float((held**2).sum())
            best = int(np.argmin(errors))
            lam_nodes[i] = float(path[best])
            gram = x_node.T @ x_node / n
            moment = x_node.T @ y_node / n
            gamma = _fista_penalized(gram, moment, lam=lam_nodes[i], weights=ones.repeat(k - 1))[
                :, 0
            ]
            node_resid = resid[:, i] - (x_node * gamma[None, :]).sum(axis=1)
            node_variance[i] = float(np.mean(node_resid**2))
            slopes[i, others] = gamma / scale
        precision = np.diag(1.0 / np.maximum(node_variance, 1e-12))
        for i in range(k):
            precision[i, :] -= slopes[i, :] / max(node_variance[i], 1e-12)
            precision[i, i] = 1.0 / max(node_variance[i], 1e-12)
        precision = 0.5 * (precision + precision.T)
        diagonal = np.sqrt(np.diag(precision))
        partial = -precision / np.outer(diagonal, diagonal)
        np.fill_diagonal(partial, 1.0)
        return _VectorGraphicalFit(
            coefficient_stack=stage_one.coefficient_stack,
            deterministic=stage_one.deterministic,
            sigma_u=stage_one.sigma_u,
            resid=stage_one.resid,
            fittedvalues=stage_one.fittedvalues,
            penalty=stage_one.penalty,
            lam=stage_one.lam,
            lambda_path=stage_one.lambda_path,
            cv_errors=stage_one.cv_errors,
            n_nonzero=stage_one.n_nonzero,
            nobs=stage_one.nobs,
            precision=precision,
            partial_correlations=partial,
            lam_nodes=lam_nodes,
        )


class _IdentificationModel[R](ABC):
    """Base for identification models over a fitted closed reduced-form system.

    The structural counterpart of :class:`_BaseModel`: where an estimation
    model constructs with a sample and exposes ``fit``, an identification
    model constructs with a *fitted result* and exposes ``identify``. The
    constructor checks the one contract every scheme shares -- that the source
    is a closed system, exposing the propagation surface an identification
    reads -- so a conditional family fails here with an explanation rather
    than deep inside a scheme with an attribute error.
    """

    __slots__ = ("_source",)

    _REQUIRED: ClassVar[tuple[str, ...]] = (
        "names",
        "nobs",
        "sigma_u",
        "resid",
        "coefficients",
        "ma_representation",
    )

    def __init__(self, result: ClosedSystemResult) -> None:
        """Validate that the source result is a closed system and store it.

        Args:
            result: The fitted reduced-form result to identify.

        Raises:
            SpecificationError: If the result lacks part of the closed-system
                surface, which is what a conditional family -- a VARX viewed
                through its conditional mixin, a lone global unit -- looks
                like from here.
        """
        missing = [name for name in self._REQUIRED if not hasattr(result, name)]
        if missing:
            raise SpecificationError(
                f"identification needs a closed reduced-form result exposing "
                f"{self._REQUIRED}; this one lacks {missing}. A conditional "
                "family has no closed system to identify -- close it first, "
                "the way a global model closes its units."
            )
        self._source = result

    @property
    def source(self) -> ClosedSystemResult:
        """The fitted reduced-form result being identified."""
        return self._source

    @property
    def names(self) -> tuple[str, ...]:
        """Variable labels of the source system."""
        return self._source.names

    @property
    def k_endog(self) -> int:
        """Number of variables in the source system."""
        return self._source.k_endog

    @abstractmethod
    def identify(self) -> R:
        """Apply the scheme and return the structural view."""


class _MarkovSwitchingVectorAutoRegressionModel[R](_VectorAutoRegressionModel[R]):
    """Specification and EM engine of a Markov-switching vector autoregression.

    Rides on the same Hamilton filter and Kim smoother as the univariate
    family -- both consume a log-density matrix and a transition matrix and
    never see the data, so the only genuinely new machinery here is the
    density (multivariate Gaussian per regime) and the M-step. The M-step is
    exact under partial switching: all regimes' coefficient slabs are solved
    in one generalized-Sylvester system, weighted by the smoothed
    probabilities and each regime's inverse covariance, so a non-switching
    block is estimated jointly across regimes rather than per regime and then
    averaged.

    Args:
        endog: The observed panel.
        order: Autoregressive order within each regime.
        n_regimes: Number of latent regimes ``M``, at least two.
        switching_mean: Whether the deterministic block switches.
        switching_variance: Whether the innovation covariance switches.
        switching_ar: Whether the lag coefficients switch.
        trend: Deterministic terms per regime.
        names: One label per variable.

    Raises:
        SpecificationError: If no component switches, so no regime is
            identified, or the specification is malformed.
        DimensionError: If the sample cannot support ``M`` regimes.
    """

    __slots__ = ("_m", "_sw_ar", "_sw_mean", "_sw_var")

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        n_regimes: int = 2,
        switching_mean: bool = True,
        switching_variance: bool = True,
        switching_ar: bool = False,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, order=order, trend=trend, names=names)
        self._m = validate_order(n_regimes, "n_regimes", minimum=2)
        self._sw_mean = bool(switching_mean)
        self._sw_var = bool(switching_variance)
        self._sw_ar = bool(switching_ar)
        if not (self._sw_mean or self._sw_var or self._sw_ar):
            raise SpecificationError(
                "at least one of switching_mean, switching_variance, "
                "switching_ar must be True; otherwise no regime is identified."
            )
        if self._sw_mean and self._n_deterministic_columns == 0:
            raise SpecificationError(
                "switching_mean is meaningless with trend='n': there is no "
                "deterministic block to switch. Set switching_mean=False or "
                "choose a trend."
            )
        need = self._m * (self.n_regressors + self.k_endog + 1) + self._order
        if self._endog.shape[0] < need:
            raise DimensionError(
                f"a sample of {self._endog.shape[0]} rows is too short for "
                f"{self._m} regimes of a {type(self).__name__}({self._order}); "
                f"it needs at least {need}."
            )

    @property
    def n_regimes(self) -> int:
        """Number of latent regimes."""
        return self._m

    @property
    def switching_mean(self) -> bool:
        """Whether the deterministic block switches."""
        return self._sw_mean

    @property
    def switching_variance(self) -> bool:
        """Whether the innovation covariance switches."""
        return self._sw_var

    @property
    def switching_ar(self) -> bool:
        """Whether the lag coefficients switch."""
        return self._sw_ar

    @property
    def label_ordering(self) -> str:
        """Which quantity regimes are sorted by, ascending.

        A mixture likelihood is invariant to relabelling regimes, so the fit
        imposes an ordering to make two runs comparable: the first variable's
        intercept when the mean switches, the log-determinant of the
        innovation covariance when only scale does (regime 0 is then the
        quiet regime), and the first variable's own first-lag coefficient
        otherwise.
        """
        if self._sw_mean:
            return "first-variable intercept"
        if self._sw_var:
            return "covariance log-determinant"
        return "first own-lag coefficient"

    def _selection_matrices(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Per-regime maps from the stacked slab matrix to full coefficients.

        The free parameters are slabs: one deterministic block per regime or
        one shared, one lag block per regime or one shared. ``S_m`` maps the
        ``(q, k)`` stacked slab matrix to regime ``m``'s full ``(w, k)``
        coefficient matrix as ``B_m = S_m @ theta``.

        Returns:
            One ``(w, q)`` selection matrix per regime.
        """
        d = self._n_deterministic_columns
        lagw = self.k_endog * self._order
        n_det_slabs = self._m if (self._sw_mean and d) else (1 if d else 0)
        n_ar_slabs = self._m if (self._sw_ar and lagw) else (1 if lagw else 0)
        q = d * n_det_slabs + lagw * n_ar_slabs
        out: list[npt.NDArray[np.float64]] = []
        for m in range(self._m):
            s = np.zeros((d + lagw, q), dtype=np.float64)
            if d:
                i = m if self._sw_mean else 0
                s[:d, i * d : (i + 1) * d] = np.eye(d)
            if lagw:
                j = m if self._sw_ar else 0
                offset = d * n_det_slabs
                s[d:, offset + j * lagw : offset + (j + 1) * lagw] = np.eye(lagw)
            out.append(s)
        return tuple(out)

    @staticmethod
    def _log_densities(
        target: npt.NDArray[np.float64],
        design: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64],
        sigmas: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Per-regime multivariate Gaussian log-densities, shape ``(T, M)``.

        Raises:
            NumericalError: If a regime covariance is not positive definite.
        """
        n_eff, k = target.shape
        m = coefficients.shape[0]
        out = np.empty((n_eff, m), dtype=np.float64)
        for j in range(m):
            resid = target - design @ coefficients[j]
            try:
                factor = np.linalg.cholesky(sigmas[j])
            except np.linalg.LinAlgError as error:
                raise NumericalError(
                    f"regime {j}'s innovation covariance lost positive definiteness during EM."
                ) from error
            logdet = 2.0 * float(np.sum(np.log(np.diagonal(factor))))
            quad = np.sum(np.linalg.solve(factor, resid.T) ** 2, axis=0)
            out[:, j] = -0.5 * (k * _LOG_2PI + logdet + quad)
        return out

    def _update_coefficients(
        self,
        target: npt.NDArray[np.float64],
        design: npt.NDArray[np.float64],
        smoothed: npt.NDArray[np.float64],
        sigmas: npt.NDArray[np.float64],
        selections: tuple[npt.NDArray[np.float64], ...],
    ) -> npt.NDArray[np.float64]:
        """M-step coefficient update by probability-weighted GLS.

        Minimizes ``sum_m tr[Sigma_m^{-1} (Y - X S_m theta)' W_m
        (Y - X S_m theta)]`` over the stacked slab matrix ``theta``, which is
        the generalized Sylvester system ``sum_m G_m theta Sigma_m^{-1} = C``
        with ``G_m = S_m' X' W_m X S_m``, solved through its Kronecker form.
        With every block switching the system is block-diagonal and collapses
        to per-regime weighted least squares; with shared blocks the coupling
        through ``Sigma_m^{-1}`` is exactly what per-regime-then-average
        would get wrong.

        Returns:
            The updated ``(M, w, k)`` stack of full coefficient matrices.
        """
        k = self.k_endog
        q = selections[0].shape[1]
        a = np.zeros((q * k, q * k), dtype=np.float64)
        b = np.zeros((q, k), dtype=np.float64)
        for m in range(self._m):
            weighted = design * smoothed[:, m][:, None]
            gram = selections[m].T @ (design.T @ weighted) @ selections[m]
            inverse = np.linalg.inv(sigmas[m])
            a += np.kron(inverse, gram)
            b += selections[m].T @ (weighted.T @ target) @ inverse
        theta = np.linalg.solve(a, b.T.ravel()).reshape((k, q)).T
        return np.stack([s @ theta for s in selections])

    def _update_sigmas(
        self,
        target: npt.NDArray[np.float64],
        design: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64],
        smoothed: npt.NDArray[np.float64],
        floor: float,
    ) -> npt.NDArray[np.float64]:
        """M-step covariance update, per regime or pooled.

        Returns:
            The updated ``(M, k, k)`` covariance stack, ridged by ``floor``
            so a momentarily starved regime stays positive definite.
        """
        k = self.k_endog
        n_eff = target.shape[0]
        out = np.empty((self._m, k, k), dtype=np.float64)
        pooled = np.zeros((k, k), dtype=np.float64)
        for m in range(self._m):
            resid = target - design @ coefficients[m]
            weighted = resid * smoothed[:, m][:, None]
            cross = weighted.T @ resid
            if self._sw_var:
                out[m] = cross / max(float(smoothed[:, m].sum()), 1e-12)
            else:
                pooled += cross
        if not self._sw_var:
            out[:] = pooled / n_eff
        return out + floor * np.eye(k)

    def _run_em(
        self,
        transition0: npt.NDArray[np.float64],
        coefficients0: npt.NDArray[np.float64],
        sigmas0: npt.NDArray[np.float64],
        *,
        max_iter: int,
        tol: float,
    ) -> _VectorExpectationMaximizationState:
        """Run EM to convergence or ``max_iter`` from one set of starts.

        Raises:
            NumericalError: If the log-likelihood becomes non-finite or a
                covariance loses positive definiteness.
        """
        target, design, _ = self._design()
        selections = self._selection_matrices()
        floor = 1e-8 * float(np.mean(np.var(self._endog, axis=0))) + 1e-12
        prob_floor = 1e-8
        transition = transition0.copy()
        coefficients = coefficients0.copy()
        sigmas = sigmas0.copy()
        prev_llf = -np.inf
        filtered = predicted = smoothed = np.empty((0, self._m))
        n_iter = 0
        converged = False
        for n_iter in range(1, max_iter + 1):
            density = self._log_densities(target, design, coefficients, sigmas)
            filt = hamilton_filter(density, transition)
            smooth = kim_smoother(filt, transition)
            filtered = filt.filtered_prob
            predicted = filt.predicted_prob
            smoothed = smooth.smoothed_prob
            llf = filt.loglikelihood
            if not np.isfinite(llf):
                raise NumericalError("MS-VAR log-likelihood became non-finite during EM.")
            if llf - prev_llf < tol and n_iter > 1:
                converged = True
                prev_llf = llf
                break
            prev_llf = llf
            transition = _MarkovSwitchingModel.update_transition(
                smoothed, smooth.smoothed_joint_prob, prob_floor
            )
            coefficients = self._update_coefficients(target, design, smoothed, sigmas, selections)
            sigmas = self._update_sigmas(target, design, coefficients, smoothed, floor)
        return _VectorExpectationMaximizationState(
            transition=transition,
            coefficients=coefficients,
            sigmas=sigmas,
            filtered_prob=filtered,
            predicted_prob=predicted,
            smoothed_prob=smoothed,
            llf=float(prev_llf),
            n_iter=n_iter,
            converged=converged,
        )

    def _label_permutation(
        self,
        coefficients: npt.NDArray[np.float64],
        sigmas: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.intp]:
        """The regime ordering :attr:`label_ordering` names, as a permutation."""
        if self._sw_mean:
            keys = coefficients[:, 0, 0]
        elif self._sw_var:
            keys = np.array([np.linalg.slogdet(sigma)[1] for sigma in sigmas])
        else:
            d = self._n_deterministic_columns
            keys = coefficients[:, d, 0]
        return np.argsort(keys)

    def _fit_markov(
        self,
        *,
        max_iter: int,
        tol: float,
        n_init: int,
        screen_iter: int,
        seed: int | np.random.Generator | None,
    ) -> _VectorMarkovSwitchingFit:
        """Estimate by EM with multi-start screening.

        Start zero is the linear fit with regimes separated along whichever
        block switches -- intercepts spread by the residual scale, covariances
        scaled geometrically, lag blocks damped and amplified -- and further
        starts perturb it randomly. Each start is screened briefly and only
        the best is refined, exactly the univariate protocol.

        Raises:
            NumericalError: If every start fails to produce a finite
                likelihood.
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        target, design, n_eff = self._design()
        k, m = self.k_endog, self._m
        d = self._n_deterministic_columns
        moments = self._gaussian_moments(target, design)
        base_coef = np.stack([moments.coef.copy() for _ in range(m)])
        base_sigma = np.stack([moments.sigma_ml.copy() for _ in range(m)])
        scale = np.std(moments.resid, axis=0)
        if self._sw_mean and d:
            spread = np.linspace(-1.0, 1.0, m)
            for j in range(m):
                base_coef[j, 0] += spread[j] * scale
        if self._sw_var:
            factors = np.geomspace(0.5, 2.0, m)
            for j in range(m):
                base_sigma[j] *= factors[j]
        if self._sw_ar and self._order and not self._sw_mean:
            damp = np.linspace(0.8, 1.2, m)
            for j in range(m):
                base_coef[j, d:] *= damp[j]

        def start(
            index: int,
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            if index == 0:
                return (
                    _MarkovSwitchingModel.initial_transition(m, rng, 0.9),
                    base_coef,
                    base_sigma,
                )
            noise = rng.standard_normal(base_coef.shape) * 0.3
            coef = base_coef + noise * np.std(base_coef, axis=(0, 1), keepdims=True)
            sigma = base_sigma * rng.uniform(0.5, 2.0, size=(m, 1, 1))
            return (
                _MarkovSwitchingModel.initial_transition(m, rng, float(rng.uniform(0.8, 0.95))),
                coef,
                sigma,
            )

        best: _VectorExpectationMaximizationState | None = None
        for index in range(max(n_init, 1)):
            transition0, coef0, sigma0 = start(index)
            try:
                state = self._run_em(transition0, coef0, sigma0, max_iter=screen_iter, tol=tol)
            except NumericalError:
                continue
            if best is None or state.llf > best.llf:
                best = state
        if best is None:
            raise NumericalError("MS-VAR estimation failed for every start.")
        refined = self._run_em(
            best.transition, best.coefficients, best.sigmas, max_iter=max_iter, tol=tol
        )
        state = refined if refined.llf >= best.llf else best

        perm = self._label_permutation(state.coefficients, state.sigmas)
        transition = state.transition[np.ix_(perm, perm)]
        coefficients = state.coefficients[perm]
        sigmas = state.sigmas[perm]
        smoothed = state.smoothed_prob[:, perm]
        fitted = np.einsum("tm,mtk->tk", smoothed, design @ coefficients)
        n_det_slabs = m if (self._sw_mean and d) else (1 if d else 0)
        n_ar_slabs = m if (self._sw_ar and self._order) else (1 if self._order else 0)
        q = d * n_det_slabs + k * self._order * n_ar_slabs
        return _VectorMarkovSwitchingFit(
            transition=transition,
            coefficients=np.stack([self._lag_blocks(coefficients[j]) for j in range(m)]),
            deterministics=coefficients[:, :d, :].copy(),
            sigmas=sigmas,
            filtered_prob=state.filtered_prob[:, perm],
            predicted_prob=state.predicted_prob[:, perm],
            smoothed_prob=smoothed,
            ergodic_prob=ergodic_distribution(transition),
            expected_durations=1.0 / np.clip(1.0 - np.diag(transition), 1e-12, None),
            resid=target - fitted,
            fittedvalues=fitted,
            llf=state.llf,
            nobs=n_eff,
            n_params=float(m * (m - 1))
            + float(k * q)
            + (m if self._sw_var else 1) * k * (k + 1) / 2.0,
            n_iter=state.n_iter,
            converged=state.converged,
        )


class _TimeVaryingVectorAutoRegressionModel[R](_VectorAutoRegressionModel[R]):
    """Specification and Gibbs engine of a time-varying-parameter VAR.

    The coefficient vector follows a random walk, which puts the model one
    representation away from the linear-Gaussian substrate: the state is the
    stacked coefficient vector, the observation matrix is the lagged design,
    and every coefficient-path draw is one call to the Durbin-Koopman
    simulation smoother. The homoskedastic sampler is two conjugate blocks on
    top of that (drift covariance and innovation covariance, both
    inverse-Wishart); the stochastic-volatility sampler replaces the constant
    covariance with Primiceri's triangular factorization ``Sigma_t = A^{-1}
    H_t A^{-T}`` and adds the KSC volatility block from :mod:`._samplers`.

    Priors follow Primiceri: a training sample is split off the front, its
    OLS estimates calibrate the coefficient prior and the drift-covariance
    scale, and it is then *discarded* from the estimation sample rather than
    used twice.

    Args:
        endog: The observed panel.
        order: Autoregressive order.
        training: Rows consumed by the training prior. ``None`` uses
            ``max(width + k + 2, min(40, a third of the effective sample))``.
        trend: Deterministic terms.
        names: One label per variable.

    Raises:
        SpecificationError: If the specification or training split is
            malformed.
        DimensionError: If the sample cannot support the split.
    """

    __slots__ = ("_training",)

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        training: int | None = None,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
    ) -> None:
        """Validate the specification and the training split."""
        super().__init__(endog, order=order, trend=trend, names=names)
        n_eff = self._endog.shape[0] - self._order
        floor = self.n_regressors + self.k_endog + 2
        if training is None:
            resolved = max(floor, min(40, n_eff // 3))
        else:
            resolved = validate_order(training, "training", minimum=1)
        if resolved < floor:
            raise SpecificationError(
                f"a training sample of {resolved} rows cannot identify the "
                f"prior; it needs at least {floor}."
            )
        if n_eff - resolved < 2 * self.n_regressors:
            raise DimensionError(
                f"after a training split of {resolved} rows, {n_eff - resolved} "
                "estimation rows remain; the time-varying model needs at least "
                f"{2 * self.n_regressors}."
            )
        self._training = resolved

    @property
    def training(self) -> int:
        """Rows consumed by the training prior."""
        return self._training

    def _stacked_design(self, design: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The ``(n, k, D)`` observation matrices ``I_k (x) x_t'``."""
        n, w = design.shape
        k = self.k_endog
        z = np.zeros((n, k, k * w), dtype=np.float64)
        for i in range(k):
            z[:, i, i * w : (i + 1) * w] = design
        return z

    def _training_prior(
        self,
        target: npt.NDArray[np.float64],
        design: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """OLS on the training rows: prior mean, prior covariance, and Sigma.

        Returns:
            ``(b0, v0, sigma0)``: the stacked coefficient prior mean ``(D,)``,
            its covariance ``(D, D)`` as ``kron(Sigma, (X'X)^{-1})``, and the
            training innovation covariance.
        """
        moments = self._gaussian_moments(target, design)
        b0 = moments.coef.T.ravel()
        xtx_inv = np.linalg.inv(design.T @ design)
        v0 = np.kron(moments.sigma_u, xtx_inv)
        v0 = 0.5 * (v0 + v0.T)
        return b0, v0, np.asarray(moments.sigma_u, dtype=np.float64)

    @staticmethod
    def _triangularize(
        sigma: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Split a covariance as ``A^{-1} D A^{-T}`` with unit-lower ``A^{-1}``.

        Returns:
            ``(a, log_diag)``: the unit-lower-triangular ``A`` whose rows
            orthogonalize the residuals, and the log of the diagonal
            variances.
        """
        chol = np.linalg.cholesky(sigma)
        scale = np.diagonal(chol)
        lower = chol / scale[None, :]
        a = np.linalg.inv(lower)
        return a, np.log(scale**2)

    def _fit_tvp(
        self,
        *,
        sv: bool,
        n_draws: int,
        n_burn: int,
        thin: int,
        k_drift: float,
        k_vol: float,
        seed: int | np.random.Generator | None,
    ) -> _TimeVaryingFit:
        """Run the Gibbs sampler and summarize the posterior.

        Args:
            sv: Whether the innovation covariance carries stochastic
                volatility through the triangular factorization.
            n_draws: Total sampler iterations.
            n_burn: Burn-in iterations discarded.
            thin: Keep every ``thin``-th post-burn draw.
            k_drift: Primiceri's ``k_Q``: the prior scale of coefficient
                drift, as a fraction of the training coefficient uncertainty.
            k_vol: Primiceri's ``k_W``: the prior scale of the log-volatility
                random walk.
            seed: Seed or generator.

        Returns:
            The packed :class:`_TimeVaryingFit`.

        Raises:
            SpecificationError: If the draw bookkeeping is inconsistent.
            NumericalError: If a conditional draw collapses.
        """
        if n_draws <= n_burn:
            raise SpecificationError(f"n_draws ({n_draws}) must exceed n_burn ({n_burn}).")
        if thin < 1:
            raise SpecificationError(f"thin must be at least 1; got {thin}.")
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        full_target, full_design, _ = self._design()
        tau = self._training
        b0, v0, sigma0 = self._training_prior(full_target[:tau], full_design[:tau])
        target = full_target[tau:]
        design = full_design[tau:]
        n, k = target.shape
        d_state = b0.shape[0]
        z = self._stacked_design(design)

        df_q = float(max(tau, d_state + 2))
        q_scale0 = k_drift**2 * df_q * v0
        df_h = float(k + 2)
        h_scale0 = sigma0 * df_h

        q = 25.0 * q_scale0 / df_q
        sigma = sigma0.copy()
        a_mat, log_diag0 = self._triangularize(sigma0)
        h_path = np.tile(log_diag0, (n, 1))
        vol_of_vol = np.full(k, 2.0 * k_vol**2)
        a_prior_prec = 0.1

        keep = (n_draws - n_burn + thin - 1) // thin
        beta_kept = np.zeros((keep, n, d_state))
        impact_kept = np.zeros((keep, k, k)) if sv else None
        h_kept = np.zeros((keep, n, k)) if sv else None
        q_sum = np.zeros_like(q)
        sigma_sum = np.zeros((k, k))
        w_sum = np.zeros(k)
        kept = 0

        identity = np.eye(d_state)
        for it in range(n_draws):
            if sv:
                a_inv = np.linalg.inv(a_mat)
                obs_cov = np.einsum("ij,tj,kj->tik", a_inv, np.exp(h_path), a_inv)
            else:
                obs_cov = sigma
            space = _LinearGaussianStateSpaceModel(
                z,
                obs_cov,
                identity,
                identity,
                q,
                initial_state=b0,
                initial_state_cov=4.0 * v0,
            )
            beta = space.simulation_smoother(target, n_sims=1, seed=rng)[0]

            drift = np.diff(beta, axis=0)
            q = _draw_inverse_wishart(q_scale0 + drift.T @ drift, df_q + n - 1.0, rng)

            resid = target - np.einsum("tkd,td->tk", z, beta)
            if sv:
                for i in range(1, k):
                    weights = np.exp(-h_path[:, i])
                    x_reg = -resid[:, :i]
                    precision = x_reg.T @ (x_reg * weights[:, None]) + a_prior_prec * np.eye(i)
                    mean = np.linalg.solve(precision, x_reg.T @ (resid[:, i] * weights))
                    root = np.linalg.cholesky(np.linalg.inv(precision))
                    a_mat[i, :i] = mean + root @ rng.standard_normal(i)
                ortho = resid @ a_mat.T
                for i in range(k):
                    h_path[:, i] = _draw_volatility_path(
                        ortho[:, i],
                        h_path[:, i],
                        float(vol_of_vol[i]),
                        prior_mean=float(log_diag0[i]),
                        prior_var=4.0,
                        rng=rng,
                    )
                    delta = np.diff(h_path[:, i])
                    vol_of_vol[i] = _draw_inverse_gamma(
                        2.0 + 0.5 * (n - 1),
                        2.0 * k_vol**2 + 0.5 * float(delta @ delta),
                        rng,
                    )
            else:
                sigma = _draw_inverse_wishart(h_scale0 + resid.T @ resid, df_h + n, rng)

            if it >= n_burn and (it - n_burn) % thin == 0:
                beta_kept[kept] = beta
                q_sum += q
                if sv:
                    a_inv = np.linalg.inv(a_mat)
                    assert impact_kept is not None and h_kept is not None
                    impact_kept[kept] = a_inv
                    h_kept[kept] = h_path
                    average = np.einsum("ij,j,kj->ik", a_inv, np.exp(h_path).mean(axis=0), a_inv)
                    sigma_sum += average
                    w_sum += vol_of_vol
                else:
                    sigma_sum += sigma
                kept += 1

        beta_kept = beta_kept[:kept]
        beta_mean = beta_kept.mean(axis=0)
        beta_low = np.quantile(beta_kept, 0.16, axis=0)
        beta_high = np.quantile(beta_kept, 0.84, axis=0)
        fitted = np.einsum("tkd,td->tk", z, beta_mean)
        return _TimeVaryingFit(
            beta_mean=beta_mean,
            beta_low=beta_low,
            beta_high=beta_high,
            beta_draws=beta_kept,
            state_cov=q_sum / max(kept, 1),
            sigma_u=sigma_sum / max(kept, 1),
            impact_draws=None if impact_kept is None else impact_kept[:kept],
            h_draws=None if h_kept is None else h_kept[:kept],
            vol_of_vol=(w_sum / max(kept, 1)) if sv else None,
            resid=target - fitted,
            fittedvalues=fitted,
            nobs=n,
            training=tau,
            n_draws=n_draws,
            n_burn=n_burn,
            thin=thin,
        )


class _BayesianVectorAutoRegressionModel[R](_VectorAutoRegressionModel[R]):
    """Estimation engine of the conjugate Normal-inverse-Wishart VAR.

    The same specification surface as the linear base -- including the prior
    argument, which here is load-bearing rather than optional -- with a
    posterior instead of a point estimate. Conjugacy is a property of the
    prior, not of this class, and it is *verified* rather than assumed: the
    prior's stated variances must factor as ``sigma_i^2 * omega_col``, which
    is exactly Litterman's structure with the cross-equation weight at one.
    A prior that breaks the factorization (a Minnesota prior with
    ``cross_equation != 1``) is refused with directions to the per-equation
    point path, because approximating it here would silently change what the
    hyperparameter means.

    The marginal likelihood is the sample's, not the stacked pseudo-sample's:
    dummy observations are part of the *prior*, so their contribution is
    divided out (Giannone, Lenza & Primiceri 2015), and the number reported
    is comparable across priors with different dummy rows.
    """

    __slots__ = ()

    def _conjugate_inputs(
        self,
        prior: _Prior | None = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Validate a prior's conjugacy and extract its Kronecker pieces.

        Args:
            prior: The prior to decompose; the model's own when omitted.
                The hierarchical sampler passes a fresh prior per
                hyperparameter value through this hook.

        Returns:
            ``(omega, mean, scale0, dummy_target, dummy_design)`` -- the
            shared row-variance profile, the prior mean, the inverse-Wishart
            prior scale, and the prior's artificial rows.

        Raises:
            SpecificationError: If the prior is absent, improper, or not
                Kronecker-factorable.
        """
        context = self._prior_context()
        prior = self._prior if prior is None else prior
        if not prior._components():
            raise SpecificationError(
                "a Bayesian VAR needs a proper prior; construct with "
                "prior=NormalInverseWishartPrior(...) or a composition, not "
                "with no prior at all -- an improper prior has no marginal "
                "likelihood and no posterior to draw from."
            )
        variance = prior.coefficient_variance(context)
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
            raise SpecificationError(
                "the prior leaves some coefficient variances infinite or "
                "non-positive, so it is improper and has no marginal "
                "likelihood. Give every coefficient a finite prior variance "
                "-- NormalInverseWishartPrior does."
            )
        scales = context.scales
        ratio = variance / scales[None, :] ** 2
        omega = np.asarray(ratio.mean(axis=1), dtype=np.float64)
        spread = float(np.max(np.abs(ratio - omega[:, None]) / omega[:, None]))
        if spread > 1e-8:
            raise SpecificationError(
                "the prior variance does not factor as sigma_i^2 * omega_col, "
                "so Normal-inverse-Wishart conjugacy does not hold -- a "
                "Minnesota prior with cross_equation != 1 does this by "
                "design. Either pin cross_equation to one (that is "
                "NormalInverseWishartPrior), or keep the weight and use the "
                "per-equation point path VAR(..., prior=...) instead."
            )
        mean = prior.coefficient_mean(context)
        scale0 = np.diag(scales**2)
        dummy_target, dummy_design = prior.dummy_observations(context)
        return omega, mean, scale0, dummy_target, dummy_design

    def _conjugate_pieces(
        self,
        target: npt.NDArray[np.float64],
        design: npt.NDArray[np.float64],
        prior: _Prior | None = None,
    ) -> tuple[_ConjugatePosterior, float, int]:
        """Posterior, dummy-corrected log marginal likelihood, and dummy count.

        Args:
            target: The effective-sample target block.
            design: The effective-sample design.
            prior: The prior to update under; the model's own when omitted.

        Returns:
            ``(posterior, log_ml, n_dummy)``.

        Raises:
            SpecificationError: If the prior is unusable.
            NumericalError: If a posterior scale degenerates.
        """
        omega, mean, scale0, dummy_target, dummy_design = self._conjugate_inputs(prior)
        df0 = float(self.k_endog + 2)
        n_dummy = int(dummy_target.shape[0])
        if n_dummy:
            full_target = np.vstack([target, dummy_target])
            full_design = np.vstack([design, dummy_design])
        else:
            full_target, full_design = target, design
        posterior = _conjugate_posterior(
            full_target, full_design, omega=omega, mean=mean, scale0=scale0, df0=df0
        )
        log_ml = posterior.log_ml
        if n_dummy:
            log_ml -= _conjugate_posterior(
                dummy_target, dummy_design, omega=omega, mean=mean, scale0=scale0, df0=df0
            ).log_ml
        return posterior, float(log_ml), n_dummy

    @staticmethod
    def _draw_conjugate(
        posterior: _ConjugatePosterior,
        n_draws: int,
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Independent draws of ``(B, Sigma)`` from one exact posterior.

        Args:
            posterior: The Normal-inverse-Wishart posterior to draw from.
            n_draws: Draws to produce.
            rng: Random generator.

        Returns:
            ``(beta_draws, sigma_draws)`` of shapes ``(S, w, k)`` and
            ``(S, k, k)``.
        """
        width = int(posterior.row_precision.shape[0])
        k = int(posterior.scale.shape[0])
        key_chol = np.linalg.cholesky(posterior.row_precision)
        row_factor = np.linalg.solve(key_chol, np.eye(width)).T
        beta_draws = np.empty((n_draws, width, k))
        sigma_draws = np.empty((n_draws, k, k))
        for s in range(n_draws):
            sigma = _draw_inverse_wishart(posterior.scale, posterior.df, rng)
            sigma_draws[s] = sigma
            shock = np.asarray(rng.standard_normal((width, k)), dtype=np.float64)
            beta_draws[s] = (
                posterior.coefficients + row_factor @ shock @ np.linalg.cholesky(sigma).T
            )
        return beta_draws, sigma_draws

    def _fit_conjugate(
        self,
        *,
        n_draws: int,
        seed: int | np.random.Generator | None,
    ) -> _VectorConjugateFit:
        """Update the Normal-inverse-Wishart posterior exactly, then draw.

        Args:
            n_draws: Posterior draws of ``(B, Sigma)`` to retain.
            seed: Seed or generator, for reproducibility.

        Returns:
            The packed :class:`_VectorConjugateFit`.

        Raises:
            SpecificationError: If the prior is unusable or the draw count
                is not positive.
            NumericalError: If a posterior scale degenerates.
        """
        if n_draws < 1:
            raise SpecificationError(f"n_draws must be positive; got {n_draws}.")
        target, design, n_eff = self._design()
        posterior, log_ml, n_dummy = self._conjugate_pieces(target, design)
        rng = np.random.default_rng(seed)
        beta_draws, sigma_draws = self._draw_conjugate(posterior, n_draws, rng)
        coef = posterior.coefficients
        fitted = design @ coef
        return _VectorConjugateFit(
            coefficient_stack=self._lag_blocks(coef),
            deterministic=coef[: self._n_deterministic_columns],
            beta_mean=coef,
            sigma_u=posterior.sigma_mean,
            beta_draws=beta_draws,
            sigma_draws=sigma_draws,
            log_marginal_likelihood=log_ml,
            posterior_df=posterior.df,
            resid=target - fitted,
            fittedvalues=fitted,
            nobs=n_eff,
            n_dummy=n_dummy,
        )

    def _fit_hierarchical(
        self,
        *,
        factory: Callable[[npt.NDArray[np.float64]], _Prior],
        hyper_names: tuple[str, ...],
        hyper_shapes: npt.NDArray[np.float64],
        hyper_scales: npt.NDArray[np.float64],
        start: npt.NDArray[np.float64],
        method: str,
        n_draws: int,
        n_burn: int,
        seed: int | np.random.Generator | None,
    ) -> _VectorHierarchicalFit:
        """Giannone-Lenza-Primiceri: treat the tightnesses as unknowns too.

        The hyperparameter posterior is ``p(theta | Y) proportional to
        p(Y | theta) p(theta)``, with ``p(Y | theta)`` the closed-form
        marginal likelihood and independent Gamma hyperpriors on the
        positive hyperparameters. Both treatments start from the posterior
        mode, found by direct optimization in log space. Empirical Bayes
        stops there and draws ``(B, Sigma)`` conditional on the mode; the
        full treatment runs an adaptive random-walk Metropolis chain over
        ``log theta`` and draws one ``(B, Sigma)`` from the exact
        conditional posterior at every kept state, so the retained draws
        marginalize over the hyperparameters and the bands stop pretending
        the tightness was known.

        Args:
            factory: Maps a hyperparameter vector to the prior it names.
            hyper_names: One label per hyperparameter.
            hyper_shapes: Gamma hyperprior shape per hyperparameter.
            hyper_scales: Gamma hyperprior scale per hyperparameter.
            start: Initial hyperparameter vector, positive.
            method: ``"full"`` or ``"empirical"``.
            n_draws: Kept ``(B, Sigma)`` draws.
            n_burn: Burn-in Metropolis iterations (full method only).
            seed: Seed or generator.

        Returns:
            The packed :class:`_VectorHierarchicalFit`.

        Raises:
            SpecificationError: If the method is unknown or the counts are
                not positive.
            NumericalError: If the mode search fails outright.
        """
        if method not in ("full", "empirical"):
            raise SpecificationError(f"method must be 'full' or 'empirical'; got {method!r}.")
        if n_draws < 1 or n_burn < 0:
            raise SpecificationError(
                f"n_draws must be positive and n_burn non-negative; got {n_draws}, {n_burn}."
            )
        target, design, n_eff = self._design()
        d = len(hyper_names)

        def pieces(theta: npt.NDArray[np.float64]) -> tuple[_ConjugatePosterior, float, int]:
            return self._conjugate_pieces(target, design, factory(theta))

        def log_posterior(
            theta: npt.NDArray[np.float64],
        ) -> tuple[float, _ConjugatePosterior | None, int]:
            try:
                posterior, log_ml, n_dummy = pieces(theta)
            except NumericalError:
                return -np.inf, None, 0
            log_hyper = float(np.sum(sst.gamma.logpdf(theta, hyper_shapes, scale=hyper_scales)))
            return log_ml + log_hyper, posterior, n_dummy

        def negative(log_theta: npt.NDArray[np.float64]) -> float:
            value, _, _ = log_posterior(np.exp(log_theta))
            return -value if np.isfinite(value) else 1e12

        search = minimize(negative, np.log(start), method="Nelder-Mead")
        mode = np.asarray(np.exp(search.x), dtype=np.float64)
        mode_value, mode_posterior, mode_dummy = log_posterior(mode)
        if mode_posterior is None:
            raise NumericalError(
                "the hyperparameter mode search ended at a degenerate point; "
                "the sample cannot support this hierarchy."
            )
        _, mode_lml, _ = pieces(mode)
        rng = np.random.default_rng(seed)
        if method == "empirical":
            beta_draws, sigma_draws = self._draw_conjugate(mode_posterior, n_draws, rng)
            hyper_draws = np.zeros((0, d))
            acceptance = float("nan")
        else:
            current = np.log(mode)
            current_value = mode_value
            current_posterior = mode_posterior
            step = 0.2
            accepted = 0
            proposed = 0
            beta_draws = np.empty((n_draws, self.n_regressors, self.k_endog))
            sigma_draws = np.empty((n_draws, self.k_endog, self.k_endog))
            hyper_draws = np.empty((n_draws, d))
            kept = 0
            for iteration in range(n_burn + n_draws):
                candidate = current + step * np.asarray(rng.standard_normal(d), dtype=np.float64)
                value, posterior, _ = log_posterior(np.exp(candidate))
                proposed += 1
                if np.log(rng.random()) < value - current_value:
                    current, current_value = candidate, value
                    assert posterior is not None
                    current_posterior = posterior
                    accepted += 1
                if iteration < n_burn:
                    if (iteration + 1) % 50 == 0:
                        rate = accepted / proposed
                        step *= 1.1 if rate > 0.3 else 0.9
                        accepted = proposed = 0
                    continue
                block, cov = self._draw_conjugate(current_posterior, 1, rng)
                beta_draws[kept] = block[0]
                sigma_draws[kept] = cov[0]
                hyper_draws[kept] = np.exp(current)
                kept += 1
            acceptance = accepted / proposed if proposed else float("nan")
        beta_mean = beta_draws.mean(axis=0)
        fitted = design @ beta_mean
        return _VectorHierarchicalFit(
            coefficient_stack=self._lag_blocks(beta_mean),
            deterministic=beta_mean[: self._n_deterministic_columns],
            beta_mean=beta_mean,
            sigma_u=sigma_draws.mean(axis=0),
            beta_draws=beta_draws,
            sigma_draws=sigma_draws,
            log_marginal_likelihood=mode_lml,
            posterior_df=mode_posterior.df,
            resid=target - fitted,
            fittedvalues=fitted,
            nobs=n_eff,
            n_dummy=mode_dummy,
            hyper_names=hyper_names,
            hyper_mode=mode,
            hyper_draws=hyper_draws,
            acceptance=float(acceptance),
            method=method,
        )


class _StudentBayesianVectorAutoRegressionModel[R](_BayesianVectorAutoRegressionModel[R]):
    """Estimation engine of the Bayesian VAR with Student-t innovations.

    The t distribution enters as its scale mixture of normals: each
    observation carries a latent precision weight ``w_t ~ Gamma(nu/2,
    nu/2)``, and conditional on the weights the model is exactly the
    conjugate Normal-inverse-Wishart VAR on rescaled rows. The Gibbs sampler
    therefore alternates three exact conditionals: ``(B, Sigma)`` from the
    weighted conjugate update, the weights from their Gamma conditionals,
    and -- when the degrees of freedom are not stated -- ``nu`` from its
    conditional on a fixed grid, which is exact and immune to tuning.

    Conjugacy of the prior is required and verified exactly as in the
    Gaussian model: the weighted update is still a Normal-inverse-Wishart
    update. Dummy observations ride along unweighted -- they are prior
    content, and the tails belong to the data.
    """

    __slots__ = ()

    def _draw_degrees(
        self,
        weights: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> float:
        """Draw the degrees of freedom from their exact grid conditional.

        ``p(nu | w)`` is a product of Gamma densities in the weights with a
        flat prior on the grid; evaluating it on a fixed grid and sampling
        the normalized probabilities is an exact Gibbs step, trades no
        correctness for tuning, and caps the tail index at the grid's top --
        beyond which the t and the Gaussian are indistinguishable anyway.

        Args:
            weights: The current latent precision weights.
            rng: Random generator.

        Returns:
            The drawn degrees of freedom.
        """
        n = weights.shape[0]
        log_sum = float(np.log(weights).sum())
        total = float(weights.sum())
        half = 0.5 * _STUDENT_DF_GRID
        log_kernel = (
            n * (half * np.log(half) - gammaln(half)) + (half - 1.0) * log_sum - half * total
        )
        log_kernel -= log_kernel.max()
        probability = np.exp(log_kernel)
        probability /= probability.sum()
        return float(rng.choice(_STUDENT_DF_GRID, p=probability))

    def _fit_student(
        self,
        *,
        df: float | None,
        n_draws: int,
        n_burn: int,
        thin: int,
        seed: int | np.random.Generator | None,
    ) -> _VectorStudentFit:
        """Gibbs on the scale-mixture representation of the t likelihood.

        Args:
            df: Degrees of freedom, above two -- or ``None`` to give the
                tail index a posterior of its own on a fixed grid.
            n_draws: Total sampler iterations.
            n_burn: Burn-in iterations discarded.
            thin: Keep every ``thin``-th post-burn draw.
            seed: Seed or generator.

        Returns:
            The packed :class:`_VectorStudentFit`.

        Raises:
            SpecificationError: If the draw bookkeeping is inconsistent, the
                stated degrees of freedom do not admit a covariance, or the
                prior is unusable.
            NumericalError: If a conditional draw collapses.
        """
        if n_draws <= n_burn:
            raise SpecificationError(f"n_draws ({n_draws}) must exceed n_burn ({n_burn}).")
        if thin < 1:
            raise SpecificationError(f"thin must be at least 1; got {thin}.")
        if df is not None and df <= 2.0:
            raise SpecificationError(
                f"df must exceed 2 for the innovations to have a covariance; "
                f"got {df}. Pass df=None to give the tail index a posterior."
            )
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        target, design, n_eff = self._design()
        omega, mean, scale0, dummy_target, dummy_design = self._conjugate_inputs()
        k = self.k_endog
        width = self.n_regressors
        df0 = float(k + 2)
        n_dummy = int(dummy_target.shape[0])
        weights = np.ones(n_eff)
        nu = float(df) if df is not None else 10.0
        keep = (n_draws - n_burn + thin - 1) // thin
        beta_kept = np.empty((keep, width, k))
        sigma_kept = np.empty((keep, k, k))
        df_kept = np.empty(keep)
        weight_sum = np.zeros(n_eff)
        kept = 0
        for iteration in range(n_draws):
            root = np.sqrt(weights)
            rows_target = target * root[:, None]
            rows_design = design * root[:, None]
            if n_dummy:
                rows_target = np.vstack([rows_target, dummy_target])
                rows_design = np.vstack([rows_design, dummy_design])
            posterior = _conjugate_posterior(
                rows_target, rows_design, omega=omega, mean=mean, scale0=scale0, df0=df0
            )
            beta_block, sigma_block = self._draw_conjugate(posterior, 1, rng)
            beta, sigma = beta_block[0], sigma_block[0]
            resid = target - design @ beta
            quadratic = np.einsum("ti,ij,tj->t", resid, np.linalg.inv(sigma), resid)
            weights = np.asarray(
                rng.gamma(0.5 * (nu + k), 2.0 / (nu + quadratic)), dtype=np.float64
            )
            if df is None:
                nu = self._draw_degrees(weights, rng)
            if iteration >= n_burn and (iteration - n_burn) % thin == 0:
                beta_kept[kept] = beta
                sigma_kept[kept] = sigma
                df_kept[kept] = nu
                weight_sum += weights
                kept += 1
        beta_mean = beta_kept.mean(axis=0)
        fitted = design @ beta_mean
        return _VectorStudentFit(
            coefficient_stack=self._lag_blocks(beta_mean),
            deterministic=beta_mean[: self._n_deterministic_columns],
            beta_mean=beta_mean,
            sigma_u=sigma_kept.mean(axis=0),
            beta_draws=beta_kept,
            sigma_draws=sigma_kept,
            df=float(df_kept.mean()),
            df_draws=df_kept if df is None else np.zeros(0),
            weight_mean=weight_sum / max(kept, 1),
            resid=target - fitted,
            fittedvalues=fitted,
            nobs=n_eff,
            n_dummy=n_dummy,
            n_draws=n_draws,
            n_burn=n_burn,
            thin=thin,
        )


class _VolatilityBayesianVectorAutoRegressionModel[R](_VectorAutoRegressionModel[R]):
    """Estimation engine of the Bayesian VAR with stochastic volatility.

    Constant coefficients, drifting covariance: ``Sigma_t = A^{-1} H_t
    A^{-T}`` with a constant unit-lower ``A`` and random-walk log variances
    -- the Carriero-Clark-Marcellino object, assembled from the same blocks
    as the Primiceri sampler with the coefficient drift switched off. The
    coefficient draw is the *joint* generalized-least-squares conditional
    over all equations, exact by construction, which sidesteps the
    equation-at-a-time factorization whose original ordering required the
    2022 corrigendum.

    Conjugacy is not needed here and not demanded: the GLS draw handles any
    diagonal prior variance, so Litterman's cross-equation weight is
    admissible again. What is refused is dummy-observation content -- an
    artificial row has no date, so under time-varying volatility it has no
    covariance to be weighted by.
    """

    __slots__ = ()

    def _volatility_inputs(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Validate the prior and return its moments.

        Returns:
            ``(mean, variance)``, each ``(w, k)``.

        Raises:
            SpecificationError: If the prior is absent, improper, or
                contributes dummy observations.
        """
        context = self._prior_context()
        prior = self._prior
        if not prior._components():
            raise SpecificationError(
                "a stochastic-volatility BVAR needs a proper prior; construct "
                "with prior=MinnesotaPrior(...) or another finite-variance "
                "prior."
            )
        variance = prior.coefficient_variance(context)
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
            raise SpecificationError(
                "the prior leaves some coefficient variances infinite or "
                "non-positive; give every coefficient a finite prior "
                "variance."
            )
        dummy_target, _ = prior.dummy_observations(context)
        if dummy_target.shape[0]:
            raise SpecificationError(
                "dummy observations have no date, so under time-varying "
                "volatility they have no covariance to be weighted by; state "
                "this prior entirely in moments (drop sum-of-coefficients "
                "and dummy-initial-observation components)."
            )
        return prior.coefficient_mean(context), variance

    def _fit_volatility(
        self,
        *,
        n_draws: int,
        n_burn: int,
        thin: int,
        k_vol: float,
        seed: int | np.random.Generator | None,
    ) -> _VectorVolatilityFit:
        """Gibbs over coefficients, orthogonalization, and volatility paths.

        Args:
            n_draws: Total sampler iterations.
            n_burn: Burn-in iterations discarded.
            thin: Keep every ``thin``-th post-burn draw.
            k_vol: Prior scale of the log-volatility random walk.
            seed: Seed or generator.

        Returns:
            The packed :class:`_VectorVolatilityFit`.

        Raises:
            SpecificationError: If the draw bookkeeping is inconsistent or
                the prior is unusable.
            NumericalError: If a conditional draw collapses.
        """
        if n_draws <= n_burn:
            raise SpecificationError(f"n_draws ({n_draws}) must exceed n_burn ({n_burn}).")
        if thin < 1:
            raise SpecificationError(f"thin must be at least 1; got {thin}.")
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        target, design, n_eff = self._design()
        prior_mean, prior_variance = self._volatility_inputs()
        k, width = self.k_endog, self.n_regressors
        context = self._prior_context()
        point = posterior_coefficients(target, design, self._prior, context).coefficients
        resid0 = target - design @ point
        sigma0 = resid0.T @ resid0 / max(n_eff - width, 1)
        a_mat, log_diag0 = _TimeVaryingVectorAutoRegressionModel._triangularize(sigma0)
        h_path = np.tile(log_diag0, (n_eff, 1))
        vol_of_vol = np.full(k, k_vol**2)
        a_prior_prec = 0.1
        beta = point
        prior_precision_vector = (1.0 / prior_variance).T.ravel()
        prior_mean_vector = prior_mean.T.ravel()
        keep = (n_draws - n_burn + thin - 1) // thin
        beta_kept = np.empty((keep, width, k))
        sigma_kept = np.empty((keep, k, k))
        h_kept = np.empty((keep, n_eff, k))
        impact_kept = np.empty((keep, k, k))
        vol_kept = np.empty((keep, k))
        kept = 0
        for iteration in range(n_draws):
            inverse_sigmas = np.einsum("ji,tj,jl->til", a_mat, np.exp(-h_path), a_mat)
            precision = np.einsum("tij,ta,tb->iajb", inverse_sigmas, design, design).reshape(
                k * width, k * width
            )
            precision[np.diag_indices_from(precision)] += prior_precision_vector
            moment = np.einsum("til,tl,ta->ia", inverse_sigmas, target, design).ravel()
            moment = moment + prior_precision_vector * prior_mean_vector
            solution = np.linalg.solve(precision, moment)
            chol = np.linalg.cholesky(precision)
            shock = np.asarray(rng.standard_normal(k * width), dtype=np.float64)
            draw = solution + np.linalg.solve(chol.T, shock)
            beta = draw.reshape(k, width).T

            resid = target - design @ beta
            for i in range(1, k):
                weights = np.exp(-h_path[:, i])
                x_reg = -resid[:, :i]
                row_precision = x_reg.T @ (x_reg * weights[:, None]) + a_prior_prec * np.eye(i)
                row_mean = np.linalg.solve(row_precision, x_reg.T @ (resid[:, i] * weights))
                root = np.linalg.cholesky(np.linalg.inv(row_precision))
                a_mat[i, :i] = row_mean + root @ rng.standard_normal(i)
            ortho = resid @ a_mat.T
            for i in range(k):
                h_path[:, i] = _draw_volatility_path(
                    ortho[:, i],
                    h_path[:, i],
                    float(vol_of_vol[i]),
                    prior_mean=float(log_diag0[i]),
                    prior_var=4.0,
                    rng=rng,
                )
                steps = np.diff(h_path[:, i])
                vol_of_vol[i] = _draw_inverse_gamma(
                    2.0 + 0.5 * (n_eff - 1.0),
                    2.0 * k_vol**2 + 0.5 * float(steps @ steps),
                    rng,
                )
            if iteration >= n_burn and (iteration - n_burn) % thin == 0:
                a_inv = np.linalg.inv(a_mat)
                beta_kept[kept] = beta
                impact_kept[kept] = a_inv
                h_kept[kept] = h_path
                vol_kept[kept] = vol_of_vol
                sigma_kept[kept] = (a_inv * np.exp(h_path[-1])[None, :]) @ a_inv.T
                kept += 1
        beta_mean = beta_kept.mean(axis=0)
        fitted = design @ beta_mean
        return _VectorVolatilityFit(
            coefficient_stack=self._lag_blocks(beta_mean),
            deterministic=beta_mean[: self._n_deterministic_columns],
            beta_mean=beta_mean,
            sigma_u=sigma_kept.mean(axis=0),
            beta_draws=beta_kept,
            sigma_draws=sigma_kept,
            h_draws=h_kept,
            impact_draws=impact_kept,
            vol_of_vol=vol_kept.mean(axis=0),
            resid=target - fitted,
            fittedvalues=fitted,
            nobs=n_eff,
            n_draws=n_draws,
            n_burn=n_burn,
            thin=thin,
        )


class _NonlinearStateSpaceModel:
    """A nonlinear (and optionally non-Gaussian) state-space model.

    The additive-Gaussian core is::

        y_t = g(alpha_t) + eps_t,      eps_t ~ N(0, R)
        alpha_{t+1} = f(alpha_t) + eta_t,   eta_t ~ N(0, Q)

    with ``f`` and ``g`` arbitrary *batched* maps: each takes a
    ``(n, m)`` block of states and returns ``(n, m)`` or ``(n, p)``
    rows, which is what lets sigma points and particle clouds move
    through them without Python loops. Three filters read it, in
    increasing generality and decreasing exactness-of-assumptions:
    the extended filter linearizes ``f`` and ``g`` by central-difference
    Jacobians; the unscented filter propagates deterministic sigma
    points (Julier-Uhlmann), exact through linear maps and third-order
    accurate through smooth ones; the particle filter (bootstrap, or
    auxiliary in the Pitt-Shephard form) makes no smoothness or
    Gaussianity assumption at all and returns an unbiased likelihood
    *estimate* rather than a likelihood.

    Non-Gaussian models enter through two hooks. ``transition_sampler``
    replaces the additive-Gaussian state draw, and ``observation_loglik``
    replaces the Gaussian measurement density -- a stochastic-volatility
    model, whose measurement noise is multiplicative, is the canonical
    customer. A model carrying either hook is outside the additive form,
    so the extended and unscented filters *refuse* it rather than
    linearizing an assumption that no longer holds; the particle filter
    is the honest tool there, and the refusal says so.

    Missing observations are ``numpy.nan`` rows: every filter skips the
    update and carries the prediction, matching the linear substrate's
    convention.
    """

    def __init__(
        self,
        transition: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        observation: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        *,
        state_cov: npt.ArrayLike,
        obs_cov: npt.ArrayLike,
        initial_state: npt.ArrayLike,
        initial_state_cov: npt.ArrayLike,
        transition_sampler: Callable[
            [npt.NDArray[np.float64], np.random.Generator], npt.NDArray[np.float64]
        ]
        | None = None,
        observation_loglik: Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ]
        | None = None,
    ) -> None:
        """Validate the system and probe the callables' shape contract.

        Args:
            transition: Batched state map ``f``: ``(n, m) -> (n, m)``.
            observation: Batched measurement map ``g``: ``(n, m) -> (n, p)``.
            state_cov: State noise covariance ``Q``, ``(m, m)``.
            obs_cov: Measurement noise covariance ``R``, ``(p, p)``. Ignored
                by the particle filter when ``observation_loglik`` is given.
            initial_state: Prior mean ``a_1``, ``(m,)``.
            initial_state_cov: Prior covariance ``P_1``, ``(m, m)``.
            transition_sampler: Optional replacement for the additive state
                draw: ``(particles (n, m), rng) -> (n, m)``.
            observation_loglik: Optional replacement for the Gaussian
                measurement density: ``(y (p,), particles (n, m)) -> (n,)``
                log-densities.

        Raises:
            DimensionError: If a matrix or a probed callable's output has
                an inconsistent shape.
            NumericalError: If a covariance is not finite or not symmetric
                positive semidefinite.
        """
        self._Q = np.asarray(state_cov, dtype=np.float64)
        self._R_cov = np.asarray(obs_cov, dtype=np.float64)
        self._a1 = np.asarray(initial_state, dtype=np.float64).ravel()
        self._P1 = np.asarray(initial_state_cov, dtype=np.float64)
        m = self._a1.shape[0]
        if self._Q.shape != (m, m) or self._P1.shape != (m, m):
            raise DimensionError(
                f"state_cov and initial_state_cov must be ({m}, {m}) to match "
                f"the state; got {self._Q.shape} and {self._P1.shape}."
            )
        for label, matrix in (
            ("state_cov", self._Q),
            ("obs_cov", self._R_cov),
            ("initial_state_cov", self._P1),
        ):
            if not np.all(np.isfinite(matrix)):
                raise NumericalError(f"{label} must be finite.")
            if float(np.abs(matrix - matrix.T).max()) > 1e-10:
                raise NumericalError(f"{label} must be symmetric.")
            if matrix.size and float(np.linalg.eigvalsh(matrix)[0]) < -1e-10:
                raise NumericalError(f"{label} must be positive semidefinite.")
        probe = np.vstack([self._a1, self._a1])
        moved = np.asarray(transition(probe), dtype=np.float64)
        if moved.shape != (2, m):
            raise DimensionError(
                f"transition must map (n, {m}) states to (n, {m}); a probe "
                f"batch of 2 returned shape {moved.shape}."
            )
        seen = np.asarray(observation(probe), dtype=np.float64)
        p = int(self._R_cov.shape[0])
        if seen.ndim != 2 or seen.shape[0] != 2 or seen.shape[1] != p:
            raise DimensionError(
                f"observation must map (n, {m}) states to (n, {p}) to match "
                f"obs_cov; a probe batch of 2 returned shape {seen.shape}."
            )
        self._f = transition
        self._g = observation
        self._m = m
        self._p = p
        self._sampler = transition_sampler
        self._obs_loglik = observation_loglik

    @property
    def k_states(self) -> int:
        """State dimension."""
        return self._m

    @property
    def k_endog(self) -> int:
        """Observation dimension."""
        return self._p

    @property
    def is_additive_gaussian(self) -> bool:
        """Whether the model is in the additive-Gaussian form all filters accept."""
        return self._sampler is None and self._obs_loglik is None

    def _prepare(self, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Coerce data to ``(n, p)``, promoting a 1-D series when ``p`` is one.

        Raises:
            DimensionError: If the data cannot match the observation
                dimension.
            SpecificationError: If a row is partially observed -- this
                engine treats missingness whole-row, and silently
                discarding the observed elements would misstate the
                likelihood.
        """
        data = np.asarray(y, dtype=np.float64)
        if data.ndim == 1 and self._p == 1:
            data = data[:, None]
        if data.ndim != 2 or data.shape[1] != self._p:
            raise DimensionError(f"data must be (n, {self._p}); got shape {np.asarray(y).shape}.")
        finite = np.isfinite(data)
        partial = finite.any(axis=1) & ~finite.all(axis=1)
        if bool(partial.any()):
            raise SpecificationError(
                f"{int(partial.sum())} row(s) (first at index "
                f"{int(np.flatnonzero(partial)[0])}) are partially "
                "observed; this engine treats missingness whole-row, and "
                "silently discarding the observed elements would misstate "
                "the likelihood. Pass fully observed or fully missing "
                "rows, or use the linear-Gaussian substrate, whose filter "
                "supports element-wise missingness."
            )
        return data

    def _refuse_hooks(self, filter_name: str) -> None:
        """Refuse a linearizing filter on a model outside the additive form.

        Raises:
            SpecificationError: If a custom hook is present.
        """
        if not self.is_additive_gaussian:
            raise SpecificationError(
                f"the {filter_name} filter is defined only for the "
                "additive-Gaussian form, and this model carries a custom "
                "transition sampler or observation likelihood; linearizing "
                "an assumption that no longer holds would return confident "
                "nonsense. Use particle_filter, which assumes neither."
            )

    def _jacobian(
        self,
        func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        point: npt.NDArray[np.float64],
        out_dim: int,
    ) -> npt.NDArray[np.float64]:
        """Central-difference Jacobian of a batched map at one point."""
        m = self._m
        steps = np.sqrt(np.finfo(np.float64).eps) * np.maximum(np.abs(point), 1.0)
        forward = np.tile(point, (m, 1)) + np.diag(steps)
        backward = np.tile(point, (m, 1)) - np.diag(steps)
        high = np.asarray(func(forward), dtype=np.float64)
        low = np.asarray(func(backward), dtype=np.float64)
        return np.asarray((high - low).T / (2.0 * steps)[None, :], dtype=np.float64).reshape(
            out_dim, m
        )

    def _gaussian_update(
        self,
        y_row: npt.NDArray[np.float64],
        predicted_obs: npt.NDArray[np.float64],
        innovation_cov: npt.NDArray[np.float64],
        cross_cov: npt.NDArray[np.float64],
        mean: npt.NDArray[np.float64],
        cov: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """One Gaussian measurement update from moment inputs.

        Returns:
            ``(filtered_mean, filtered_cov, log_contribution)``.

        Raises:
            NumericalError: If the innovation covariance degenerates.
        """
        try:
            chol = np.linalg.cholesky(innovation_cov)
        except np.linalg.LinAlgError as error:
            raise NumericalError(
                "the innovation covariance lost positive definiteness; the filter has diverged."
            ) from error
        gain = sla.cho_solve((chol, True), cross_cov.T).T
        residual = y_row - predicted_obs
        white = sla.solve_triangular(chol, residual, lower=True)
        contribution = -0.5 * (
            self._p * _LOG_2PI + 2.0 * float(np.sum(np.log(np.diag(chol)))) + float(white @ white)
        )
        filtered_mean = mean + gain @ residual
        filtered_cov = cov - gain @ innovation_cov @ gain.T
        filtered_cov = 0.5 * (filtered_cov + filtered_cov.T)
        return filtered_mean, filtered_cov, contribution

    def extended_filter(self, y: npt.ArrayLike) -> _KalmanFilterResult:
        """The extended Kalman filter: linearize, then filter exactly.

        First-order accurate in the nonlinearity; exact when ``f`` and
        ``g`` are linear, in which case it *is* the Kalman filter.
        Jacobians come from central differences, so the maps need to be
        smooth at the working point but need not expose derivatives.

        Args:
            y: Data, ``(n, p)``; ``numpy.nan`` rows are missing.

        Returns:
            The standard filter record; the log-likelihood is the
            linearized approximation.

        Raises:
            SpecificationError: If the model carries custom hooks.
            NumericalError: If the filter diverges.
        """
        self._refuse_hooks("extended")
        data = self._prepare(y)
        n, m = data.shape[0], self._m
        predicted = np.empty((n, m))
        predicted_cov = np.empty((n, m, m))
        filtered = np.empty((n, m))
        filtered_cov = np.empty((n, m, m))
        contributions = np.zeros(n)
        mean, cov = self._a1.copy(), self._P1.copy()
        for t in range(n):
            predicted[t], predicted_cov[t] = mean, cov
            if np.all(np.isfinite(data[t])):
                jac = self._jacobian(self._g, mean, self._p)
                center = np.asarray(self._g(mean[None, :]), dtype=np.float64)[0]
                innovation_cov = jac @ cov @ jac.T + self._R_cov
                mean, cov, contributions[t] = self._gaussian_update(
                    data[t], center, innovation_cov, cov @ jac.T, mean, cov
                )
            filtered[t], filtered_cov[t] = mean, cov
            jac_f = self._jacobian(self._f, mean, m)
            mean = np.asarray(self._f(mean[None, :]), dtype=np.float64)[0]
            cov = jac_f @ cov @ jac_f.T + self._Q
        return _KalmanFilterResult(
            loglikelihood=float(contributions.sum()),
            loglikelihood_contributions=contributions,
            predicted_state=predicted,
            predicted_state_cov=predicted_cov,
            filtered_state=filtered,
            filtered_state_cov=filtered_cov,
        )

    def _sigma_points(
        self, mean: npt.NDArray[np.float64], cov: npt.NDArray[np.float64], scale: float
    ) -> npt.NDArray[np.float64]:
        """The ``2m + 1`` unscented points around one moment pair."""
        root = psd_sqrt((self._m + scale) * cov)
        return np.vstack([mean[None, :], mean + root.T, mean - root.T])

    def _unscented_weights(
        self, alpha: float, beta: float, kappa: float
    ) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Scaling constant and mean/covariance sigma-point weights.

        Raises:
            SpecificationError: If the spread is not positive.
        """
        if alpha <= 0.0:
            raise SpecificationError(f"alpha must be positive; got {alpha}.")
        m = self._m
        lam = alpha**2 * (m + kappa) - m
        mean_weights = np.full(2 * m + 1, 1.0 / (2.0 * (m + lam)))
        mean_weights[0] = lam / (m + lam)
        cov_weights = mean_weights.copy()
        cov_weights[0] += 1.0 - alpha**2 + beta
        return lam, mean_weights, cov_weights

    @staticmethod
    def _smoother_gain(
        cross: npt.NDArray[np.float64], predicted_cov: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """The Rauch gain ``C P_{t+1|t}^{-1}`` from a cross-covariance.

        Raises:
            NumericalError: If the predicted covariance degenerates.
        """
        try:
            chol = np.linalg.cholesky(predicted_cov)
        except np.linalg.LinAlgError as error:
            raise NumericalError(
                "a predicted state covariance lost positive definiteness; "
                "the backward pass cannot form the smoother gain."
            ) from error
        return sla.cho_solve((chol, True), cross.T).T

    def unscented_filter(
        self,
        y: npt.ArrayLike,
        *,
        alpha: float = 1e-1,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> _KalmanFilterResult:
        """The unscented Kalman filter: deterministic sigma points, no Jacobians.

        Exact through linear maps for any parameter setting, and accurate
        to third order in the Taylor sense through smooth nonlinear ones
        (Julier-Uhlmann); the standard additive-noise form.

        Args:
            y: Data, ``(n, p)``; ``numpy.nan`` rows are missing.
            alpha: Sigma-point spread; small values keep the points near
                the mean.
            beta: Prior-distribution weighting; ``2`` is optimal for a
                Gaussian.
            kappa: Secondary scaling; ``0`` is the standard default.

        Returns:
            The standard filter record; the log-likelihood is the
            unscented approximation.

        Raises:
            SpecificationError: If the model carries custom hooks or the
                spread is not positive.
            NumericalError: If the filter diverges.
        """
        self._refuse_hooks("unscented")
        lam, mean_weights, cov_weights = self._unscented_weights(alpha, beta, kappa)
        data = self._prepare(y)
        n, m = data.shape[0], self._m
        predicted = np.empty((n, m))
        predicted_cov = np.empty((n, m, m))
        filtered = np.empty((n, m))
        filtered_cov = np.empty((n, m, m))
        contributions = np.zeros(n)
        mean, cov = self._a1.copy(), self._P1.copy()
        for t in range(n):
            predicted[t], predicted_cov[t] = mean, cov
            if np.all(np.isfinite(data[t])):
                points = self._sigma_points(mean, cov, lam)
                seen = np.asarray(self._g(points), dtype=np.float64)
                center = mean_weights @ seen
                gap_obs = seen - center[None, :]
                gap_state = points - mean[None, :]
                innovation_cov = gap_obs.T @ (cov_weights[:, None] * gap_obs) + self._R_cov
                cross = gap_state.T @ (cov_weights[:, None] * gap_obs)
                mean, cov, contributions[t] = self._gaussian_update(
                    data[t], center, innovation_cov, cross, mean, cov
                )
            filtered[t], filtered_cov[t] = mean, cov
            points = self._sigma_points(mean, cov, lam)
            moved = np.asarray(self._f(points), dtype=np.float64)
            mean = mean_weights @ moved
            gap = moved - mean[None, :]
            cov = gap.T @ (cov_weights[:, None] * gap) + self._Q
            cov = 0.5 * (cov + cov.T)
        return _KalmanFilterResult(
            loglikelihood=float(contributions.sum()),
            loglikelihood_contributions=contributions,
            predicted_state=predicted,
            predicted_state_cov=predicted_cov,
            filtered_state=filtered,
            filtered_state_cov=filtered_cov,
        )

    def _rts_backward(
        self,
        outcome: _KalmanFilterResult,
        cross_at: Callable[[int], npt.NDArray[np.float64]],
        method: str,
    ) -> _RtsSmootherResult:
        """Rauch's backward recursion from a forward pass and a gain rule."""
        n, m = outcome.filtered_state.shape
        smoothed = np.empty((n, m))
        smoothed_cov = np.empty((n, m, m))
        smoothed[-1] = outcome.filtered_state[-1]
        smoothed_cov[-1] = outcome.filtered_state_cov[-1]
        for t in range(n - 2, -1, -1):
            gain = self._smoother_gain(cross_at(t), outcome.predicted_state_cov[t + 1])
            smoothed[t] = outcome.filtered_state[t] + gain @ (
                smoothed[t + 1] - outcome.predicted_state[t + 1]
            )
            shrink = smoothed_cov[t + 1] - outcome.predicted_state_cov[t + 1]
            cov = outcome.filtered_state_cov[t] + gain @ shrink @ gain.T
            smoothed_cov[t] = 0.5 * (cov + cov.T)
        return _RtsSmootherResult(
            smoothed_state=smoothed,
            smoothed_state_cov=smoothed_cov,
            method=method,
        )

    def extended_smoother(self, y: npt.ArrayLike) -> _RtsSmootherResult:
        """The extended Rauch-Tung-Striebel smoother.

        Runs the extended filter, then Rauch's backward recursion with the
        transition Jacobian evaluated at each filtered mean. The
        linearization error therefore compounds through both passes: exact
        when ``f`` and ``g`` are linear, degrading faster than the forward
        filter as curvature grows. When this and the unscented smoother
        disagree materially, trust the unscented one.

        Args:
            y: Data, ``(n, p)``; ``numpy.nan`` rows are missing.

        Returns:
            The :class:`_RtsSmootherResult` with ``method="extended"``.

        Raises:
            SpecificationError: If the model carries custom hooks.
            NumericalError: If either pass degenerates.
        """
        outcome = self.extended_filter(y)

        def cross_at(t: int) -> npt.NDArray[np.float64]:
            jac = self._jacobian(self._f, outcome.filtered_state[t], self._m)
            return np.asarray(outcome.filtered_state_cov[t] @ jac.T, dtype=np.float64)

        return self._rts_backward(outcome, cross_at, "extended")

    def unscented_smoother(
        self,
        y: npt.ArrayLike,
        *,
        alpha: float = 1e-1,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> _RtsSmootherResult:
        """The unscented Rauch-Tung-Striebel smoother.

        Runs the unscented filter, then Rauch's backward recursion with
        the filtered-to-predicted cross-covariance rebuilt from sigma
        points at each filtered moment pair -- the same deterministic
        points the forward pass used, so the two passes share one
        approximation rather than stacking two different ones. Exact
        through linear maps, like its filter.

        Args:
            y: Data, ``(n, p)``; ``numpy.nan`` rows are missing.
            alpha: Sigma-point spread; must match the intended filter's.
            beta: Prior-distribution weighting.
            kappa: Secondary scaling.

        Returns:
            The :class:`_RtsSmootherResult` with ``method="unscented"``.

        Raises:
            SpecificationError: If the model carries custom hooks or the
                spread is not positive.
            NumericalError: If either pass degenerates.
        """
        self._refuse_hooks("unscented")
        lam, mean_weights, cov_weights = self._unscented_weights(alpha, beta, kappa)
        outcome = self.unscented_filter(y, alpha=alpha, beta=beta, kappa=kappa)

        def cross_at(t: int) -> npt.NDArray[np.float64]:
            points = self._sigma_points(
                outcome.filtered_state[t], outcome.filtered_state_cov[t], lam
            )
            moved = np.asarray(self._f(points), dtype=np.float64)
            center = mean_weights @ moved
            return np.asarray(
                (points - outcome.filtered_state[t][None, :]).T
                @ (cov_weights[:, None] * (moved - center[None, :])),
                dtype=np.float64,
            )

        return self._rts_backward(outcome, cross_at, "unscented")

    def _draw_states(
        self, particles: npt.NDArray[np.float64], rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        """One transition draw per particle."""
        if self._sampler is not None:
            return np.asarray(self._sampler(particles, rng), dtype=np.float64)
        noise = rng.standard_normal(particles.shape) @ psd_sqrt(self._Q).T
        return np.asarray(self._f(particles), dtype=np.float64) + noise

    def _measure(
        self, y_row: npt.NDArray[np.float64], particles: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Per-particle measurement log-density."""
        if self._obs_loglik is not None:
            return np.asarray(self._obs_loglik(y_row, particles), dtype=np.float64)
        residual = y_row[None, :] - np.asarray(self._g(particles), dtype=np.float64)
        chol = np.linalg.cholesky(self._R_cov)
        white = sla.solve_triangular(chol, residual.T, lower=True)
        return np.asarray(
            -0.5
            * (
                self._p * _LOG_2PI
                + 2.0 * float(np.sum(np.log(np.diag(chol))))
                + np.sum(white**2, axis=0)
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _systematic_resample(
        weights: npt.NDArray[np.float64], rng: np.random.Generator
    ) -> npt.NDArray[np.intp]:
        """Systematic resampling indices for normalized weights."""
        count = weights.shape[0]
        positions = (rng.random() + np.arange(count)) / count
        return np.asarray(np.searchsorted(np.cumsum(weights), positions), dtype=np.intp).clip(
            0, count - 1
        )

    def particle_filter(
        self,
        y: npt.ArrayLike,
        *,
        n_particles: int = 2000,
        method: str = "bootstrap",
        ess_threshold: float = 0.5,
        seed: int | np.random.Generator | None = None,
    ) -> _ParticleFilterResult:
        """Sequential Monte Carlo: the filter that assumes nothing smooth.

        The bootstrap filter (Gordon-Salmond-Smith 1993) propagates
        particles through the transition and reweights by the measurement
        density, resampling systematically when the effective sample size
        falls below the stated fraction. The auxiliary variant
        (Pitt-Shephard 1999) pre-selects particles by where the transition
        *expects* them to land -- worthwhile when the measurement is
        informative *and* the transition is tight enough that the
        expectation predicts the landing point. When transition noise
        dominates, the point anchors mispredict, the second-stage weights
        degenerate, and the bootstrap filter is the better tool; watch the
        effective sample size. Both return an unbiased
        *estimate* of the likelihood; its Monte Carlo noise is real, and
        seed-to-seed spread is the honest error bar.

        Both methods observe the substrate's initial-state convention: the
        cloud at the first period is drawn from ``N(a_1, P_1)`` and
        weighted directly, with no transition applied before the first
        observation (the auxiliary look-ahead therefore starts at the
        second).

        Args:
            y: Data, ``(n, p)``; ``numpy.nan`` rows are missing.
            n_particles: Cloud size, at least 2.
            method: ``"bootstrap"`` or ``"auxiliary"``. The auxiliary
                variant needs the additive-Gaussian transition or a
                ``transition`` map to anchor its look-ahead.
            ess_threshold: Resample when the effective sample size falls
                below this fraction of the cloud (bootstrap only; the
                auxiliary filter resamples every step by construction).
            seed: Seed or generator.

        Returns:
            The :class:`_ParticleFilterResult`.

        Raises:
            SpecificationError: If the method or bookkeeping is malformed.
            NumericalError: If every particle's weight underflows -- the
                cloud has degenerated and the estimate is meaningless.
        """
        if method not in ("bootstrap", "auxiliary"):
            raise SpecificationError(f"method must be 'bootstrap' or 'auxiliary'; got {method!r}.")
        if n_particles < 2:
            raise SpecificationError(f"n_particles must be at least 2; got {n_particles}.")
        if not 0.0 < ess_threshold <= 1.0:
            raise SpecificationError(f"ess_threshold must be in (0, 1]; got {ess_threshold}.")
        data = self._prepare(y)
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        n, m, count = data.shape[0], self._m, int(n_particles)
        particles = self._a1[None, :] + rng.standard_normal((count, m)) @ psd_sqrt(self._P1).T
        log_weights = np.full(count, -np.log(count))
        filtered = np.empty((n, m))
        spread = np.empty((n, m))
        ess = np.empty(n)
        contributions = np.zeros(n)
        for t in range(n):
            observed = bool(np.all(np.isfinite(data[t])))
            if method == "auxiliary" and observed and t > 0:
                anchors = np.asarray(self._f(particles), dtype=np.float64)
                first = log_weights + self._measure(data[t], anchors)
                peak = float(first.max())
                if not np.isfinite(peak):
                    raise NumericalError(
                        "every particle's look-ahead weight underflowed; the cloud has degenerated."
                    )
                stage = np.exp(first - peak)
                total = float(stage.sum())
                indices = self._systematic_resample(stage / total, rng)
                particles = self._draw_states(particles[indices], rng)
                second = self._measure(data[t], particles) - self._measure(
                    data[t], anchors[indices]
                )
                peak2 = float(second.max())
                if not np.isfinite(peak2):
                    raise NumericalError(
                        "every particle's second-stage weight underflowed; "
                        "the cloud has degenerated."
                    )
                contributions[t] = (
                    peak + np.log(total) + peak2 + float(np.log(np.mean(np.exp(second - peak2))))
                )
                scaled = np.exp(second - peak2)
                normalized = scaled / float(scaled.sum())
                log_weights = np.log(np.maximum(normalized, 1e-300))
            else:
                if t > 0:
                    particles = self._draw_states(particles, rng)
                if observed:
                    log_weights = log_weights + self._measure(data[t], particles)
                peak = float(log_weights.max())
                if not np.isfinite(peak):
                    raise NumericalError(
                        "every particle's weight underflowed; the cloud has degenerated."
                    )
                scaled = np.exp(log_weights - peak)
                total = float(scaled.sum())
                if observed:
                    contributions[t] = peak + np.log(total)
                normalized = scaled / total
                log_weights = np.log(np.maximum(normalized, 1e-300))
            filtered[t] = normalized @ particles
            gap = particles - filtered[t][None, :]
            spread[t] = np.sqrt(np.maximum(normalized @ gap**2, 0.0))
            ess[t] = 1.0 / float(np.sum(normalized**2))
            if method == "bootstrap" and ess[t] < ess_threshold * count:
                indices = self._systematic_resample(normalized, rng)
                particles = particles[indices]
                log_weights = np.full(count, -np.log(count))
        return _ParticleFilterResult(
            loglikelihood=float(contributions.sum()),
            loglikelihood_contributions=contributions,
            filtered_state=filtered,
            filtered_state_std=spread,
            effective_sample_size=ess,
            n_particles=count,
            method=method,
        )

    def particle_smoother(
        self,
        y: npt.ArrayLike,
        *,
        n_particles: int = 500,
        ess_threshold: float = 0.5,
        seed: int | np.random.Generator | None = None,
    ) -> _ParticleSmootherResult:
        """Forward-filtering backward-smoothing (Doucet-Godsill-Andrieu).

        A bootstrap forward pass stores every weighted cloud, and the
        backward recursion reweights each of them through the transition
        density ``N(f(x), Q)``. That density is why the method's reach
        differs from the particle filter's: a custom ``observation_loglik``
        is fine (it only enters the forward weights), but a custom
        ``transition_sampler`` exposes draws without a density and is
        refused, as is a singular ``Q``.

        The costs are quadratic and stated rather than hidden: time is
        ``O(n_particles**2)`` per period and the forward clouds are held
        in full, so the default cloud is smaller than the filter's --
        raise it only knowing both bills scale with its square and its
        size respectively.

        Args:
            y: Data, ``(n, p)``; ``numpy.nan`` rows are missing.
            n_particles: Cloud size, at least 2.
            ess_threshold: Forward-pass resampling trigger, as in the
                bootstrap filter.
            seed: Seed or generator.

        Returns:
            The :class:`_ParticleSmootherResult`.

        Raises:
            SpecificationError: If the model carries a transition sampler
                hook or the bookkeeping is malformed.
            NumericalError: If ``Q`` is singular or the cloud degenerates.
        """
        if self._sampler is not None:
            raise SpecificationError(
                "the particle smoother reweights through the transition "
                "*density*, and a model with a custom transition sampler "
                "exposes only draws; without the density the backward "
                "weights are undefined. Filtering remains available."
            )
        if n_particles < 2:
            raise SpecificationError(f"n_particles must be at least 2; got {n_particles}.")
        if not 0.0 < ess_threshold <= 1.0:
            raise SpecificationError(f"ess_threshold must be in (0, 1]; got {ess_threshold}.")
        try:
            chol_q = np.linalg.cholesky(self._Q)
        except np.linalg.LinAlgError as error:
            raise NumericalError(
                "state_cov is singular, so the transition density the "
                "backward weights need is degenerate; the particle "
                "smoother requires a nondegenerate transition."
            ) from error
        data = self._prepare(y)
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        n, m, count = data.shape[0], self._m, int(n_particles)
        log_det_q = 2.0 * float(np.sum(np.log(np.diag(chol_q))))
        particles = self._a1[None, :] + rng.standard_normal((count, m)) @ psd_sqrt(self._P1).T
        log_weights = np.full(count, -np.log(count))
        clouds = np.empty((n, count, m))
        weights = np.empty((n, count))
        for t in range(n):
            if t > 0:
                particles = self._draw_states(particles, rng)
            if np.all(np.isfinite(data[t])):
                log_weights = log_weights + self._measure(data[t], particles)
            peak = float(log_weights.max())
            if not np.isfinite(peak):
                raise NumericalError(
                    "every particle's weight underflowed; the cloud has degenerated."
                )
            scaled = np.exp(log_weights - peak)
            normalized = scaled / float(scaled.sum())
            clouds[t] = particles
            weights[t] = normalized
            if 1.0 / float(np.sum(normalized**2)) < ess_threshold * count:
                indices = self._systematic_resample(normalized, rng)
                particles = particles[indices]
                log_weights = np.full(count, -np.log(count))
            else:
                log_weights = np.log(np.maximum(normalized, 1e-300))
        smoothed = np.empty((n, m))
        spread = np.empty((n, m))
        backward = weights[n - 1]
        smoothed[-1] = backward @ clouds[-1]
        gap = clouds[-1] - smoothed[-1][None, :]
        spread[-1] = np.sqrt(np.maximum(backward @ gap**2, 0.0))
        for t in range(n - 2, -1, -1):
            anchors = np.asarray(self._f(clouds[t]), dtype=np.float64)
            diff = clouds[t + 1][:, None, :] - anchors[None, :, :]
            white = sla.solve_triangular(chol_q, diff.reshape(-1, m).T, lower=True)
            log_density = -0.5 * (
                m * _LOG_2PI + log_det_q + np.sum(white**2, axis=0).reshape(count, count)
            )
            log_filtered = np.log(np.maximum(weights[t], 1e-300))
            joint = log_filtered[None, :] + log_density
            peak_rows = joint.max(axis=1, keepdims=True)
            log_predictive = peak_rows[:, 0] + np.log(np.sum(np.exp(joint - peak_rows), axis=1))
            log_backward = np.log(np.maximum(backward, 1e-300))
            carry = log_backward[:, None] + log_density - log_predictive[:, None]
            peak_cols = carry.max(axis=0, keepdims=True)
            folded = peak_cols[0] + np.log(np.sum(np.exp(carry - peak_cols), axis=0))
            log_smoothed = log_filtered + folded
            peak = float(log_smoothed.max())
            if not np.isfinite(peak):
                raise NumericalError(
                    "every backward weight underflowed; the smoothed cloud has degenerated."
                )
            scaled = np.exp(log_smoothed - peak)
            backward = scaled / float(scaled.sum())
            smoothed[t] = backward @ clouds[t]
            gap = clouds[t] - smoothed[t][None, :]
            spread[t] = np.sqrt(np.maximum(backward @ gap**2, 0.0))
        return _ParticleSmootherResult(
            smoothed_state=smoothed,
            smoothed_state_std=spread,
            n_particles=count,
        )

    @classmethod
    def _lag_stack_state_space(
        cls,
        mean_map: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        *,
        order: int,
        sigma2: float,
        measurement_variance: float,
        center: float,
        spread: float,
    ) -> _NonlinearStateSpaceModel:
        """Wrap a nonlinear autoregressive mean as a noisily observed state space.

        The shared emitter behind the observation-driven nonlinear models
        (threshold, smooth-transition, neural): the latent state is the
        ``order``-deep lag stack of the *true* series, the transition applies
        the fitted conditional mean to the stack and shifts it, and the
        observation reads the stack's first component through Gaussian
        measurement error. With zero measurement error the model is
        observation-driven and filtering is vacuous, which is why the variance
        is required and must be strictly positive -- the emitted object is the
        errors-in-variables reading of the fit, not a restatement of it.

        State noise enters only the first stack component, so the state
        covariance is singular whenever ``order > 1``; the three filters and
        both Rauch smoothers accept that, and the particle smoother's
        nondegenerate-transition refusal fires honestly.

        Args:
            mean_map: Batched conditional mean: an ``(n, order)`` block of lag
                rows ``[y_{t-1}, ..., y_{t-order}]`` to ``(n,)`` means.
            order: Depth of the lag stack, at least 1.
            sigma2: Fitted innovation variance, strictly positive.
            measurement_variance: Observation noise variance, strictly
                positive.
            center: Initial state mean, applied to every stack component
                (typically the sample mean).
            spread: Initial per-component state variance (typically the sample
                variance).

        Returns:
            The :class:`_NonlinearStateSpaceModel`.

        Raises:
            SpecificationError: If the variances are not strictly positive or
                the order is not at least 1.
        """
        if order < 1:
            raise SpecificationError(f"order must be at least 1; got {order}.")
        if not sigma2 > 0.0:
            raise SpecificationError(f"sigma2 must be strictly positive; got {sigma2}.")
        if not measurement_variance > 0.0:
            raise SpecificationError(
                "measurement_variance must be strictly positive: with no "
                "measurement error the model is observation-driven, its state "
                "is the data, and filtering it is vacuous. State the noise "
                "the emitted system is observed through."
            )

        def transition(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            block = np.asarray(states, dtype=np.float64)
            head = np.asarray(mean_map(block), dtype=np.float64).reshape(-1, 1)
            if order == 1:
                return head
            return np.hstack([head, block[:, : order - 1]])

        def observation(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.asarray(states, dtype=np.float64)[:, :1]

        state_cov = np.zeros((order, order))
        state_cov[0, 0] = float(sigma2)
        return cls(
            transition,
            observation,
            state_cov=state_cov,
            obs_cov=[[float(measurement_variance)]],
            initial_state=np.full(order, float(center)),
            initial_state_cov=np.eye(order) * max(float(spread), 1e-8),
        )


class _RegimeSwitchingLinearStateSpaceModel(_StateSpaceModel[_KimFilterResult, _KimSmootherResult]):
    """A linear-Gaussian state space whose system matrices switch by regime.

    The continuous-state half of the regime-switching layer -- the
    :class:`_MarkovSwitchingStateSpaceModel` docstring's promised
    extension. Each regime ``j`` of a ``K``-state Markov chain carries its
    own linear-Gaussian system::

        y_t         = Z_j alpha_t + d_j + eps_t,   eps_t ~ N(0, H_j)
        alpha_{t+1} = T_j alpha_t + c_j + eta_t,   eta_t ~ N(0, Q_j)

    Exact filtering requires tracking every regime *path* -- ``K**t``
    Kalman filters by period ``t`` -- so Kim (1994) collapses: run the
    ``K * K`` per-pair Kalman updates, mix by the Hamilton regime
    probabilities, and collapse each regime's mixture back to one moment
    pair by moment matching. The likelihood is therefore an
    *approximation*, and is labeled one; it is exact in the two limits
    that matter for verification -- identical regimes (it is the Kalman
    filter) and no state persistence (it is the Hamilton filter) -- and
    the build measures its gap against brute-force path enumeration where
    that is affordable.

    Missing observations are ``numpy.nan`` rows: the update is skipped and
    regime probabilities evolve by the chain alone.
    """

    def __init__(
        self,
        *,
        design: npt.ArrayLike,
        obs_cov: npt.ArrayLike,
        transition: npt.ArrayLike,
        state_cov: npt.ArrayLike,
        regime_transition: npt.ArrayLike,
        obs_intercept: npt.ArrayLike | None = None,
        state_intercept: npt.ArrayLike | None = None,
        initial_state: npt.ArrayLike | None = None,
        initial_state_cov: npt.ArrayLike | None = None,
        initial_regime_prob: npt.ArrayLike | None = None,
    ) -> None:
        """Validate the per-regime systems and the chain.

        Args:
            design: Per-regime observation matrices ``Z``, ``(K, p, m)``.
            obs_cov: Per-regime observation covariances ``H``, ``(K, p, p)``.
            transition: Per-regime state transitions ``T``, ``(K, m, m)``.
            state_cov: Per-regime state covariances ``Q``, ``(K, m, m)``.
            regime_transition: Row-stochastic chain matrix ``P``, ``(K, K)``,
                with ``P[i, j] = Pr(S_t = j | S_{t-1} = i)``.
            obs_intercept: Per-regime ``d``, ``(K, p)``; zero by default.
            state_intercept: Per-regime ``c``, ``(K, m)``; zero by default.
            initial_state: Prior state mean, ``(m,)``; zero by default.
            initial_state_cov: Prior state covariance, ``(m, m)``; identity
                scaled large by default.
            initial_regime_prob: Initial regime distribution, ``(K,)``; the
                chain's stationary distribution by default.

        Raises:
            DimensionError: If shapes disagree.
            NumericalError: If the chain matrix is not row-stochastic.
        """
        self._Z = np.asarray(design, dtype=np.float64)
        if self._Z.ndim != 3:
            raise DimensionError(f"design must be (K, p, m); got shape {self._Z.shape}.")
        k_regimes, p, m = self._Z.shape
        self._H = np.asarray(obs_cov, dtype=np.float64)
        self._T = np.asarray(transition, dtype=np.float64)
        self._Q = np.asarray(state_cov, dtype=np.float64)
        for label, matrix, shape in (
            ("obs_cov", self._H, (k_regimes, p, p)),
            ("transition", self._T, (k_regimes, m, m)),
            ("state_cov", self._Q, (k_regimes, m, m)),
        ):
            if matrix.shape != shape:
                raise DimensionError(f"{label} must be {shape}; got {matrix.shape}.")
        self._P = np.asarray(regime_transition, dtype=np.float64)
        if self._P.shape != (k_regimes, k_regimes):
            raise DimensionError(
                f"regime_transition must be ({k_regimes}, {k_regimes}); got {self._P.shape}."
            )
        if not np.allclose(self._P.sum(axis=1), 1.0, atol=1e-8) or np.any(self._P < -1e-12):
            raise NumericalError(
                "regime_transition must be row-stochastic with non-negative entries."
            )
        self._d = (
            np.zeros((k_regimes, p))
            if obs_intercept is None
            else np.asarray(obs_intercept, dtype=np.float64).reshape(k_regimes, p)
        )
        self._c = (
            np.zeros((k_regimes, m))
            if state_intercept is None
            else np.asarray(state_intercept, dtype=np.float64).reshape(k_regimes, m)
        )
        self._a1 = (
            np.zeros(m)
            if initial_state is None
            else np.asarray(initial_state, dtype=np.float64).reshape(m)
        )
        self._P1 = (
            np.eye(m) * 1e2
            if initial_state_cov is None
            else np.asarray(initial_state_cov, dtype=np.float64).reshape(m, m)
        )
        if initial_regime_prob is None:
            values, vectors = np.linalg.eig(self._P.T)
            pick = int(np.argmin(np.abs(values - 1.0)))
            stationary = np.real(vectors[:, pick])
            stationary = np.abs(stationary) / float(np.abs(stationary).sum())
            self._pi1 = stationary
        else:
            self._pi1 = np.asarray(initial_regime_prob, dtype=np.float64).reshape(k_regimes)
        self._k = k_regimes
        self._p = p
        self._m = m

    @property
    def k_regimes(self) -> int:
        """Number of regimes."""
        return self._k

    @property
    def k_states(self) -> int:
        """State dimension."""
        return self._m

    @property
    def k_endog(self) -> int:
        """Observation dimension."""
        return self._p

    @property
    def regime_transition(self) -> npt.NDArray[np.float64]:
        """The row-stochastic regime chain matrix ``P``, as a copy."""
        return self._P.copy()

    def filter(self, y: npt.ArrayLike) -> _KimFilterResult:
        """Kim's (1994) approximate forward pass.

        Args:
            y: Data, ``(n, p)``; a 1-D series is promoted when ``p`` is
                one, and ``numpy.nan`` rows are missing.

        Returns:
            The :class:`_KimFilterResult`; its log-likelihood is the
            collapse approximation.

        Raises:
            NumericalError: If an innovation covariance degenerates.
        """
        data = np.asarray(y, dtype=np.float64)
        if data.ndim == 1 and self._p == 1:
            data = data[:, None]
        if data.ndim != 2 or data.shape[1] != self._p:
            raise DimensionError(f"data must be (n, {self._p}); got shape {np.asarray(y).shape}.")
        finite = np.isfinite(data)
        partial = finite.any(axis=1) & ~finite.all(axis=1)
        if bool(partial.any()):
            raise SpecificationError(
                f"{int(partial.sum())} row(s) (first at index "
                f"{int(np.flatnonzero(partial)[0])}) are partially "
                "observed; this engine treats missingness whole-row, and "
                "silently discarding the observed elements would misstate "
                "the likelihood. Pass fully observed or fully missing "
                "rows, or use the linear-Gaussian substrate, whose filter "
                "supports element-wise missingness."
            )
        n, k, m, p = data.shape[0], self._k, self._m, self._p
        means = np.tile(self._a1, (k, 1))
        covs = np.tile(self._P1, (k, 1, 1))
        probs = self._pi1.copy()
        filtered_prob = np.empty((n, k))
        predicted_prob = np.empty((n, k))
        filtered_state = np.empty((n, m))
        filtered_cov = np.empty((n, m, m))
        regime_state = np.empty((n, k, m))
        contributions = np.zeros(n)
        for t in range(n):
            observed = bool(np.all(np.isfinite(data[t])))
            pair_mean = np.empty((k, k, m))
            pair_cov = np.empty((k, k, m, m))
            pair_log = np.full((k, k), -np.inf)
            for i in range(k):
                for j in range(k):
                    if t == 0:
                        mean = self._a1.copy()
                        cov = self._P1.copy()
                    else:
                        mean = self._c[j] + self._T[j] @ means[i]
                        cov = self._T[j] @ covs[i] @ self._T[j].T + self._Q[j]
                    loglik = 0.0
                    if observed:
                        center = self._Z[j] @ mean + self._d[j]
                        innovation_cov = self._Z[j] @ cov @ self._Z[j].T + self._H[j]
                        try:
                            chol = np.linalg.cholesky(innovation_cov)
                        except np.linalg.LinAlgError as error:
                            raise NumericalError(
                                "an innovation covariance lost positive "
                                "definiteness; the filter has diverged."
                            ) from error
                        residual = data[t] - center
                        white = sla.solve_triangular(chol, residual, lower=True)
                        loglik = -0.5 * (
                            p * _LOG_2PI
                            + 2.0 * float(np.sum(np.log(np.diag(chol))))
                            + float(white @ white)
                        )
                        gain = sla.cho_solve((chol, True), self._Z[j] @ cov).T
                        mean = mean + gain @ residual
                        cov = cov - gain @ innovation_cov @ gain.T
                        cov = 0.5 * (cov + cov.T)
                    pair_mean[i, j] = mean
                    pair_cov[i, j] = cov
                    prior = self._pi1[j] / k if t == 0 else self._P[i, j] * probs[i]
                    pair_log[i, j] = np.log(max(prior, 1e-300)) + loglik
            predicted_prob[t] = self._pi1 if t == 0 else probs @ self._P
            peak = float(pair_log.max())
            weights = np.exp(pair_log - peak)
            total = float(weights.sum())
            contributions[t] = peak + np.log(total)
            weights /= total
            probs = weights.sum(axis=0)
            for j in range(k):
                share = max(float(probs[j]), 1e-300)
                mixed = (weights[:, j, None] * pair_mean[:, j]).sum(axis=0) / share
                spread = np.zeros((m, m))
                for i in range(k):
                    gap = pair_mean[i, j] - mixed
                    spread += weights[i, j] * (pair_cov[i, j] + np.outer(gap, gap))
                means[j] = mixed
                covs[j] = spread / share
            filtered_prob[t] = probs
            regime_state[t] = means
            filtered_state[t] = probs @ means
            overall = np.zeros((m, m))
            for j in range(k):
                gap = means[j] - filtered_state[t]
                overall += probs[j] * (covs[j] + np.outer(gap, gap))
            filtered_cov[t] = overall
        return _KimFilterResult(
            loglikelihood=float(contributions.sum()),
            loglikelihood_contributions=contributions,
            filtered_prob=filtered_prob,
            predicted_prob=predicted_prob,
            filtered_state=filtered_state,
            filtered_state_cov=filtered_cov,
            regime_state=regime_state,
        )

    def smooth(self, y: npt.ArrayLike) -> _KimSmootherResult:
        """Kim's backward pass over the regime probabilities.

        Runs the forward filter, then the discrete backward recursion the
        package already owns for Hamilton-filter models, applied to the
        collapse-approximate filtered probabilities. Smoothing the
        *continuous* state of a switching model requires a second collapse
        approximation on the backward pass and is deliberately not
        offered; the regime chronology is what smoothing is for here.

        Args:
            y: Data, ``(n, p)``; ``numpy.nan`` rows are missing.

        Returns:
            The :class:`_KimSmootherResult` over regimes.
        """
        forward = self.filter(y)
        record = _HamiltonFilterResult(
            loglikelihood=forward.loglikelihood,
            loglikelihood_contributions=forward.loglikelihood_contributions,
            filtered_prob=forward.filtered_prob,
            predicted_prob=forward.predicted_prob,
        )
        return kim_smoother(record, self._P)

    def loglikelihood(self, y: npt.ArrayLike) -> float:
        """The collapse-approximate log-likelihood."""
        return self.filter(y).loglikelihood
