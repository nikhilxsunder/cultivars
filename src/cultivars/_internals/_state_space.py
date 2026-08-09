# filepath: /src/cultivars/_internals/_state_space.py
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

"""State-space model contract and shared result containers.

Every model in cultivars that admits a state-space representation composes
through this contract. The abstract :class:`StateSpaceModel` defines the four
operations a state-space model must expose — filtering, smoothing, likelihood
evaluation, and simulation — without committing to *how* they are computed. The
linear-Gaussian implementation uses the Kalman filter and the Durbin-Koopman
smoother; future nonlinear and regime-switching implementations will satisfy
the same contract with the extended/unscented Kalman filter, particle methods,
or the Hamilton filter.

Deliberately, the *parameterization* (system matrices vs. transition functions)
is NOT part of this contract — only the operations and the state/observation
dimensions are. This keeps the contract honest across linear and nonlinear
models. Concrete subclasses own their parameterization:

- ``LinearGaussianStateSpace(Z, H, T, R, Q, ...)`` — this module's companion
  implementation. A reduced-form VAR embeds here with ``T`` set to the companion
  matrix (see :mod:`cultivars.core.companion`).
- A future ``TVPVARStateSpace`` supplies *time-varying* ``T_t`` / ``Q_t``; the
  linear-Gaussian filter already accepts time-indexed system matrices so that
  consumer needs no change to the filter loop — only a different parameterization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class FilterResult:
    """Output of a forward filtering pass.

    Attributes:
        predicted_state: One-step-ahead predicted states ``a_{t|t-1}``,
            shape ``(n, m)``.
        predicted_state_cov: Predicted state covariances ``P_{t|t-1}``,
            shape ``(n, m, m)``.
        filtered_state: Contemporaneously filtered states ``a_{t|t}``,
            shape ``(n, m)``.
        filtered_state_cov: Filtered state covariances ``P_{t|t}``,
            shape ``(n, m, m)``.
        loglikelihood: The total Gaussian log-likelihood of the data under the
            model, summing only over observed dimensions.
        loglikelihood_contributions: Per-period log-likelihood contributions,
            shape ``(n,)``; zero at fully-missing periods.
    """

    predicted_state: npt.NDArray[np.float64]
    predicted_state_cov: npt.NDArray[np.float64]
    filtered_state: npt.NDArray[np.float64]
    filtered_state_cov: npt.NDArray[np.float64]
    loglikelihood: float
    loglikelihood_contributions: npt.NDArray[np.float64]


@dataclass(frozen=True)
class SmootherResult:
    """Output of a backward smoothing pass.

    Attributes:
        smoothed_state: Smoothed states ``a_{t|n}``, shape ``(n, m)``.
        smoothed_state_cov: Smoothed state covariances ``V_{t|n}``,
            shape ``(n, m, m)``.
    """

    smoothed_state: npt.NDArray[np.float64]
    smoothed_state_cov: npt.NDArray[np.float64]


class StateSpaceModel(ABC):
    """Abstract contract for a state-space model.

    Subclasses implement the filter/smooth/likelihood/simulate operations with a
    concrete algorithm. All data arrays use time along ``axis=0`` and encode
    missing observations as ``numpy.nan``.
    """

    @property
    @abstractmethod
    def k_endog(self) -> int:
        """Observation dimension ``p`` (number of observed series)."""

    @property
    @abstractmethod
    def k_states(self) -> int:
        """State dimension ``m``."""

    @abstractmethod
    def filter(self, y: npt.ArrayLike) -> FilterResult:
        """Run the forward filter and return states and the log-likelihood."""

    @abstractmethod
    def smooth(self, y: npt.ArrayLike) -> SmootherResult:
        """Run the backward smoother and return smoothed states."""

    @abstractmethod
    def loglikelihood(self, y: npt.ArrayLike) -> float:
        """Return the Gaussian log-likelihood of the data (fast path)."""

    @abstractmethod
    def simulate(
        self, n: int, *, seed: int | np.random.Generator | None = None
    ) -> npt.NDArray[np.float64]:
        """Draw a length-``n`` sample path of observations from the model."""

    @abstractmethod
    def simulation_smoother(
        self,
        y: npt.ArrayLike,
        *,
        n_sims: int = 1,
        seed: int | np.random.Generator | None = None,
    ) -> npt.NDArray[np.float64]:
        """Draw states from their conditional distribution given the data."""
