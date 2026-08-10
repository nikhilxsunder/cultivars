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

from ._internals import (
    _DurbinKoopmanSmootherResult as DurbinKoopmanSmootherResult,
    _KalmanFilterResult as KalmanFilterResult,
    _LinearGaussianStateSpaceModel as LinearGaussianStateSpaceModel,
)

__all__ = [
    "LinearGaussianStateSpaceModel",
    "KalmanFilterResult",
    "DurbinKoopmanSmootherResult"
]


