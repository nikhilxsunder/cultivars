# filepath: /src/cultivars/state_space/__init__.py
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
"""State-space substrates: linear-Gaussian, nonlinear, and regime-switching.

Four engines share one grammar -- construct a fully specified system, then
hand it data to ``filter``, ``smooth``, and evaluate:

- :class:`LinearGaussianStateSpaceModel` -- Kalman filter, Durbin-Koopman
  smoother, simulation smoother; the likelihood is exact, and missingness
  is element-wise.
- :class:`MarkovSwitchingStateSpaceModel` -- a discrete latent chain over
  an autoregressive observation model; Hamilton filter, Kim smoother, and
  an exact likelihood.
- :class:`RegimeSwitchingStateSpaceModel` -- a linear-Gaussian state whose
  system matrices switch with the chain; Kim's (1994) collapse, and a
  likelihood labeled as that approximation.
- :class:`NonlinearStateSpaceModel` -- arbitrary transition and
  measurement maps; extended, unscented, and particle filters, each with
  a matching smoother, chosen by name because their answers differ in
  kind.

The first three sign the shared :class:`StateSpaceModel` contract; the
nonlinear engine deliberately does not, for reasons its docstring states.
The generic discrete recursions :func:`hamilton_filter` and
:func:`kim_smoother` are exported for switching models built outside this
package.
"""

from __future__ import annotations

from .linear_gaussian import (
    DurbinKoopmanSmootherResult,
    KalmanFilterResult,
    LinearGaussianSSM,
)
from .nonlinear import (
    NonlinearSSM,
    ParticleFilterResult,
    ParticleSmootherResult,
    RtsSmootherResult,
)
from .regime_switching import (
    HamiltonFilterResult,
    KimFilterResult,
    KimSmootherResult,
    MarkovSwitchingSSM,
    RegimeSwitchingLinearSSM,
)

__all__ = [
    "DurbinKoopmanSmootherResult",
    "HamiltonFilterResult",
    "KalmanFilterResult",
    "KimFilterResult",
    "KimSmootherResult",
    "LinearGaussianSSM",
    "MarkovSwitchingSSM",
    "NonlinearSSM",
    "ParticleFilterResult",
    "ParticleSmootherResult",
    "RegimeSwitchingLinearSSM",
    "RtsSmootherResult",
]
