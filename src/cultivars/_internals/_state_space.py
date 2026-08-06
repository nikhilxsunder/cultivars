

from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class _ForwardPass(NamedTuple):
    predicted_state: npt.NDArray[np.float64]
    predicted_state_cov: npt.NDArray[np.float64]
    filtered_state: npt.NDArray[np.float64]
    filtered_state_cov: npt.NDArray[np.float64]
    loglik_contrib: npt.NDArray[np.float64]
    obs_index: list[npt.NDArray[np.intp]]
    innovation: list[npt.NDArray[np.float64]]
    innovation_precision: list[npt.NDArray[np.float64]]
    obs_design: list[npt.NDArray[np.float64]]
