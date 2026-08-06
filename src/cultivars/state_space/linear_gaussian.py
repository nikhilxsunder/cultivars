# filepath: /src/cultivars/state_space/linear_gaussian.py
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
"""Linear-Gaussian state-space model with Kalman filter and smoother.

Implements the state-space form (Durbin & Koopman, 2012, notation)::

    y_t     = Z_t alpha_t + d_t + eps_t,   eps_t ~ N(0, H_t)
    alpha_{t+1} = T_t alpha_t + c_t + R_t eta_t,   eta_t ~ N(0, Q_t)
    alpha_1 ~ N(a_1, P_1)

with ``y_t`` of dimension ``p`` (``k_endog``), ``alpha_t`` of dimension ``m``
(``k_states``), and ``eta_t`` of dimension ``r`` (``k_posdef``). Every system
matrix may be **time-invariant** (2-D) or **time-varying** (3-D, leading time
axis); intercepts ``c_t`` / ``d_t`` may be 1-D or 2-D. Missing observations are
encoded as ``numpy.nan`` and handled by collapsing each period to its observed
sub-vector.

Provided operations:

- :meth:`LinearGaussianStateSpace.filter` — Kalman filter (predicted and
  filtered states, per-period and total log-likelihood).
- :meth:`LinearGaussianStateSpace.smooth` — Durbin-Koopman state smoother.
- :meth:`LinearGaussianStateSpace.loglikelihood` — likelihood-only fast path.
- :meth:`LinearGaussianStateSpace.simulate` — forward simulation.
- :meth:`LinearGaussianStateSpace.simulation_smoother` — the Durbin-Koopman
  (2002) mean-corrected simulation smoother, drawing states given the data.

References:
    Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space
    Methods* (2nd ed.). Oxford University Press.
    Durbin, J. & Koopman, S. J. (2002). A simple and efficient simulation
    smoother for state space time series analysis. *Biometrika*, 89(3).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.linalg as sla

from ..exceptions import DimensionError, NumericalError
from ..state_space.base import FilterResult, SmootherResult, StateSpaceModel


class LinearGaussianStateSpace(StateSpaceModel):
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
                raise DimensionError(
                    f"initial_state must have shape ({m},); got {a1.shape}."
                )
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
                f"{name} must be rank 1 ({dim},) or rank 2 (n, {dim}); "
                f"got shape {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise NumericalError(f"{name} contains non-finite values.")
        return arr

    def _default_initial_cov(self) -> npt.NDArray[np.float64]:
        if self._T.ndim == 2 and self._R.ndim == 2 and self._Q.ndim == 2:
            eig = np.abs(np.linalg.eigvals(self._T))
            if float(eig.max(initial=0.0)) < 1.0 - 1e-10:
                rqr = self._R @ self._Q @ self._R.T
                return np.asarray(
                    sla.solve_discrete_lyapunov(self._T, rqr), dtype=np.float64
                )
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
            raise DimensionError(
                f"observations must have shape (n, {self._p}); got {arr.shape}."
            )
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
            pred_a, pred_P, filt_a, filt_P, llc,
            obs_index, innovation, innovation_precision, obs_design,
        )

    # -- public operations -------------------------------------------------

    def filter(self, y: npt.ArrayLike) -> FilterResult:
        """Run the Kalman filter. See :class:`~cultivars.state_space.base.FilterResult`.

        Example:
            >>> import numpy as np
            >>> m = LinearGaussianStateSpace(
            ...     design=[[1.0]], obs_cov=[[0.0]],
            ...     transition=[[0.5]], selection=[[1.0]], state_cov=[[1.0]],
            ... )
            >>> res = m.filter(np.array([[0.1], [0.2], [-0.3]]))
            >>> res.filtered_state.shape
            (3, 1)
        """
        data = self._prepare_data(y)
        fwd = self._forward(data)
        return FilterResult(
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

    def smooth(self, y: npt.ArrayLike) -> SmootherResult:
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

        return SmootherResult(smoothed_state=smoothed_a, smoothed_state_cov=smoothed_P)

    def _simulate_forward(
        self, n: int, rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        alpha = np.zeros((n, self._m))
        obs = np.zeros((n, self._p))
        p1_sqrt = _psd_sqrt(self._P1)
        state = self._a1 + p1_sqrt @ rng.standard_normal(self._m)
        for t in range(n):
            z = self._at(self._Z, t, 2)
            d = self._at(self._d, t, 1)
            h = self._at(self._H, t, 2)
            alpha[t] = state
            obs[t] = z @ state + d + _psd_sqrt(h) @ rng.standard_normal(self._p)
            t_mat = self._at(self._T, t, 2)
            c = self._at(self._c, t, 1)
            r_mat = self._at(self._R, t, 2)
            q = self._at(self._Q, t, 2)
            state = t_mat @ state + c + r_mat @ (_psd_sqrt(q) @ rng.standard_normal(self._r))
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
