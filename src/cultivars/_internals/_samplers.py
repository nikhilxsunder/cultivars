from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._core import _KSC_MEAN, _KSC_VAR, _OFFSET, _draw_mixture_indicators
from ._models import _LinearGaussianStateSpaceModel


def _draw_volatility_path(
    residual: npt.NDArray[np.float64],
    log_variance: npt.NDArray[np.float64],
    vol_of_vol: float,
    *,
    prior_mean: float,
    prior_var: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Draw one equation's random-walk log-variance path, KSC-conditionally.

    Given the current path, mixture indicators are drawn; conditional on
    them the observation ``log(e_t**2 + offset)`` is linear-Gaussian in
    ``h_t``, and the path is drawn exactly with the Durbin-Koopman
    simulation smoother on a one-dimensional state space.

    Args:
        residual: The equation's orthogonalized residuals ``e_t``.
        log_variance: The current path, used to draw the indicators.
        vol_of_vol: Current innovation variance of the random walk.
        prior_mean: Prior mean of the initial log variance.
        prior_var: Prior variance of the initial log variance.
        rng: Random generator.

    Returns:
        A new log-variance path of the residual's length.
    """
    n = residual.shape[0]
    star = np.log(residual**2 + _OFFSET)
    components = _draw_mixture_indicators(star, log_variance, rng)
    space = _LinearGaussianStateSpaceModel(
        np.ones((1, 1)),
        _KSC_VAR[components].reshape(n, 1, 1),
        np.eye(1),
        np.eye(1),
        np.array([[vol_of_vol]]),
        obs_intercept=_KSC_MEAN[components].reshape(n, 1),
        initial_state=np.array([prior_mean]),
        initial_state_cov=np.array([[prior_var]]),
    )
    draw = space.simulation_smoother(star.reshape(n, 1), n_sims=1, seed=rng)
    return np.asarray(draw[0, :, 0], dtype=np.float64)
