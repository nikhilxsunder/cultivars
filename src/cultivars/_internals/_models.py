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

import numpy as np
import numpy.typing as npt
import scipy.linalg as sla

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
    Mean,
    Method,
    Transition,
    Trend,
    Vol,
    _ForwardPass,
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
    n_deterministic,
    ols,
    pack_stationary,
    psd_sqrt,
    validate_aligned,
    validate_choice,
    validate_endog,
    validate_exog,
    validate_open_interval,
    validate_order,
    validate_order_tuple,
    validate_transition,
)
from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._engines import MeanFunctionEngine, NumpyMLPEngine
from ._filters import hamilton_filter
from ._fits import (
    _AutoRegressionFit,
    _BoxJenkinsFit,
    _FractionalIntegrationFit,
    _FractionalVarianceFit,
    _MarkovSwitchingFit,
    _NeuralAutoRegressionFit,
    _NeuralThresholdFit,
    _ShortMemoryVarianceFit,
    _SmoothTransitionFit,
    _ThresholdFit,
)
from ._layouts import _ParameterLayout
from ._means import _ARMAMean, _LinearMean, _MeanLayer
from ._objectives import (
    _AutoRegressionObjective,
    _BoxJenkinsObjective,
    _ConditionalVarianceObjective,
    _FractionalIntegrationObjective,
    _FractionalVarianceObjective,
    _SmoothTransitionObjective,
)
from ._results import (
    _DurbinKoopmanSmootherResult,
    _FilterResult,
    _HamiltonFilterResult,
    _KalmanFilterResult,
    _KimSmootherResult,
    _SmootherResult,
    _StabilityResult,
)
from ._smoothers import kim_smoother
from ._solvers import _maximize_likelihood, _solve
from ._states import _ExpectationMaximizationState


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
        if not _StabilityResult.assess_stability(phi0).is_stable:
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
                if not _StabilityResult.assess_stability(ar0).is_stable:
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
        """
        target, lags = self.effective_sample(y)
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
