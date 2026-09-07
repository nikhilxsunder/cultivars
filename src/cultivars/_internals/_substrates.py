# filepath: /src/cultivars/_internals/_substrates.py
#

"""State-space substrates: the linear-Gaussian and nonlinear engines.

Split out of ``_models`` so that objectives, samplers, and models can all
import the substrates without a cycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import scipy.linalg as sla

from .._core import (
    _LOG_2PI,
    Frequency,
    _ForwardPass,
    _nelson_siegel_loadings,
    aggregation_weights,
    psd_sqrt,
    validate_transition,
    ergodic_distribution,
    _ROW_SUM_ATOL,
    lag_matrix,
)
from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._parameters import _NelsonSiegelParameters, _StructuralParameters
from ._results import (
    _DurbinKoopmanSmootherResult,
    _FilterResult,
    _KalmanFilterResult,
    _ParticleFilterResult,
    _ParticleSmootherResult,
    _RtsSmootherResult,
    _SmootherResult,
    _KimFilterResult,
    _KimSmootherResult,
    _HamiltonFilterResult,
)
from ._filters import hamilton_filter
from ._smoothers import kim_smoother
from ._systems import _structural_matrices


class _StateSpace[F: _FilterResult, S: _SmootherResult](ABC):
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


class _LinearGaussianStateSpace(
    _StateSpace[_KalmanFilterResult, _DurbinKoopmanSmootherResult]
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
    ) -> _LinearGaussianStateSpace:
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
            The configured :class:`_LinearGaussianStateSpace`.
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
    ) -> _LinearGaussianStateSpace:
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
            A configured :class:`_LinearGaussianStateSpace` whose state is the
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

    @classmethod
    def _from_structural_system(
        cls, params: _StructuralParameters, *, trend: str, cycle: bool, seasonal: int | None
    ) -> tuple[_LinearGaussianStateSpace, dict[str, slice]]:
        """The structural model as a linear-Gaussian state space.

        Args:
            params: The parameter record.
            trend: ``"level"``, ``"lltrend"``, or ``"smooth"``.
            cycle: Whether the cycle is present.
            seasonal: Seasonal period, or ``None``.

        Returns:
            The state-space model and the component layout.
        """
        design, transition, selection, state_cov, obs_cov, initial_cov, slices = (
            _structural_matrices(params, trend=trend, cycle=cycle, seasonal=seasonal)
        )
        model = cls(
            design,
            obs_cov,
            transition,
            selection,
            state_cov,
            initial_state_cov=initial_cov,
        )
        return model, slices

    @classmethod
    def _from_nelson_siegel_system(
        cls, params: _NelsonSiegelParameters, maturities: npt.NDArray[np.float64]
    ) -> _LinearGaussianStateSpace:
        """The dynamic Nelson-Siegel model as a linear-Gaussian state space.

        The state is the factor vector (level, slope, curvature) with
        diagonal AR(1) dynamics around its mean, observed through the
        Nelson-Siegel loadings at the given maturities with diagonal
        measurement noise. The factor dynamics are stationary by
        construction, so the substrate's stationary initialization applies.

        Args:
            params: The parameter record.
            maturities: Strictly positive maturities, shape ``(p,)``.

        Returns:
            The state-space model.
        """
        loadings = _nelson_siegel_loadings(maturities, params.decay)
        transition = np.diag(params.ar)
        state_intercept = (np.eye(3) - transition) @ params.mu
        state_cov = params.state_chol @ params.state_chol.T
        return cls(
            loadings,
            np.diag(params.obs_var),
            transition,
            np.eye(3),
            state_cov,
            state_intercept=state_intercept,
            initial_state=params.mu,
            initial_state_cov=state_cov / (1.0 - np.outer(params.ar, params.ar)),
        )


class _MarkovSwitchingStateSpace(_StateSpace[_HamiltonFilterResult, _KimSmootherResult]):
    """A parameterized Markov-switching observation model over a latent chain.

    The discrete counterpart of :class:`_LinearGaussianStateSpace`, and its
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


class _NonlinearStateSpace:
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
    ) -> _NonlinearStateSpace:
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
            The :class:`_NonlinearStateSpace`.

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


class _RegimeSwitchingLinearStateSpace(_StateSpace[_KimFilterResult, _KimSmootherResult]):
    """A linear-Gaussian state space whose system matrices switch by regime.

    The continuous-state half of the regime-switching layer -- the
    :class:`_MarkovSwitchingStateSpace` docstring's promised
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
