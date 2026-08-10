from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._layouts import _ParameterLayout


@dataclass(frozen=True, slots=True)
class _ExpectationMaximizationState:
    """Internal state carried between EM restarts."""

    transition: npt.NDArray[np.float64]
    intercepts: npt.NDArray[np.float64]
    ar_params: npt.NDArray[np.float64]
    sigma2: npt.NDArray[np.float64]
    filtered_prob: npt.NDArray[np.float64]
    predicted_prob: npt.NDArray[np.float64]
    smoothed_prob: npt.NDArray[np.float64]
    llf: float
    n_iter: int
    converged: bool

    @classmethod
    def _run_em(
        cls,
        target: npt.NDArray[np.float64],
        lags: npt.NDArray[np.float64],
        layout: _ParameterLayout,
        transition0: npt.NDArray[np.float64],
        intercepts0: npt.NDArray[np.float64],
        ar0: npt.NDArray[np.float64],
        sigma20: npt.NDArray[np.float64],
        var_floor: float,
        prob_floor: float,
        max_iter: int,
        tol: float,
        switching_variance: bool,
    ) -> _ExpectationMaximizationState:
        """Run EM to convergence or ``max_iter`` from one set of starting values.

        Returns:
            The :class:`_ExpectationMaximizationState` reached.

        Raises:
            NumericalError: If the log-likelihood becomes non-finite.
        """
        transition = transition0.copy()
        intercepts = intercepts0.copy()
        ar_params = ar0.copy()
        sigma2 = sigma20.copy()
        prev_llf = -np.inf
        filtered = predicted = smoothed = np.empty((0, layout.n_regimes))
        n_iter = 0
        converged = False

        for n_iter in range(1, max_iter + 1):
            means = _regime_means(intercepts, ar_params, lags)
            logd = _log_densities(target, means, sigma2)
            filt = hamilton_filter(logd, transition, initial_prob=ergodic_distribution(transition))
            smooth = kim_smoother(filt, transition)
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
            transition = _update_transition(smoothed, smooth.smoothed_joint_prob, prob_floor)
            intercepts, ar_params = _update_coefficients(target, lags, smoothed, sigma2, layout)
            means = _regime_means(intercepts, ar_params, lags)
            sigma2 = _update_variance(target, means, smoothed, switching_variance, var_floor)

        return cls(
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
