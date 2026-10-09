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
r"""Linear-Gaussian state-space model with Kalman filter and smoother.

The state-space form, in Durbin and Koopman's notation,

.. math::

   y_t = Z_t \alpha_t + d_t + \varepsilon_t,
   \qquad \varepsilon_t \sim N(0, H_t),

   \alpha_{t+1} = T_t \alpha_t + c_t + R_t \eta_t,
   \qquad \eta_t \sim N(0, Q_t),
   \qquad \alpha_1 \sim N(a_1, P_1),

with :math:`y_t` of dimension :math:`p` (``k_endog``), :math:`\alpha_t`
of dimension :math:`m` (``k_states``) and :math:`\eta_t` of dimension
:math:`r` (``k_posdef``). The Kalman filter runs the prediction and
update recursions

.. math::

   v_t = y_t - Z_t a_t - d_t,\quad F_t = Z_t P_t Z_t' + H_t,\quad
   K_t = T_t P_t Z_t' F_t^{-1},

   a_{t+1} = T_t a_t + c_t + K_t v_t,\qquad
   P_{t+1} = T_t P_t (T_t - K_t Z_t)' + R_t Q_t R_t',

and the prediction-error decomposition
:math:`\log L = -\tfrac12 \sum_t \bigl(p_t \log 2\pi + \log|F_t| + v_t' F_t^{-1} v_t\bigr)`
is the exact Gaussian likelihood; the Durbin-Koopman smoother runs the
backward recursion in :math:`r_t` and :math:`N_t` to give
:math:`\hat\alpha_t = E(\alpha_t \mid y_{1:n})` and its variance without
inverting any filtered covariance; and the simulation smoother draws
:math:`\alpha \mid y` by the mean-correction device of Durbin and
Koopman (2002), one unconditional simulation and one extra smoothing
pass per draw. Every system matrix may be time-invariant (2-D) or
time-varying (3-D with a leading time axis), intercepts may be 1-D or
2-D, and a missing observation is ``numpy.nan``: each period collapses
to its observed sub-vector, so a wholly missing period is a pure
prediction step with zero likelihood contribution.

Two commitments shape the surface. First, the model is the matrices
and nothing else: there is no parameter vector, no estimator, and no
``fit`` -- a model is constructed fully specified, handed data, and
evaluated, so that the univariate and multivariate models built on top
of it (unobserved components, dynamic factors, mixed frequency, the
term structure) own their parameterizations and this substrate owns
only the recursions. Second, the initial condition is stated, not
guessed: ``initial_state_cov`` defaults to the stationary covariance
solving :math:`P = TPT' + RQR'` when the transition is time-invariant
and stable, and to a large finite diffuse prior (:math:`10^6 I`)
otherwise, which is the approximate diffuse initialization rather than
the exact one -- adequate for a random-walk level, and something to
know when a likelihood is compared across models with different
numbers of nonstationary states.

Layout. :class:`LinearGaussianSSM` is the model, constructed from
``design``, ``obs_cov``, ``transition``, ``selection`` and
``state_cov`` with the optional intercepts and initial conditions; its
``filter`` returns :class:`KalmanFilterResult` with predicted and
filtered states, their covariances, and the per-period log-likelihood
contributions; ``smooth`` returns :class:`DurbinKoopmanSmootherResult`
with smoothed states and covariances; ``loglikelihood`` is the
likelihood-only path, ``simulate`` the forward simulation, and
``simulation_smoother`` the conditional draws with shape
``(n_sims, n, m)``. The classmethod ``stationary_covariance`` solves
the Lyapunov equation on its own. All three names are public aliases of
``_LinearGaussianStateSpace``, ``_KalmanFilterResult`` and
``_DurbinKoopmanSmootherResult`` in ``_internals``, where the forward
pass ``_forward`` and its ``_ForwardPass`` record live; the nonlinear
and regime-switching engines that share the ``filter``/``smooth``
grammar are :mod:`~cultivars.state_space.nonlinear` and
:mod:`~cultivars.state_space.regime_switching`.

References:
    Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State
    Space Methods*, 2nd ed. Oxford University Press.

    Durbin, J., & Koopman, S. J. (2002). A simple and efficient
    simulation smoother for state space time series analysis.
    *Biometrika*, 89(3), 603-615.

    Harvey, A. C. (1989). *Forecasting, Structural Time Series Models
    and the Kalman Filter*. Cambridge University Press.

Example:
    A local level with signal-to-noise ratio :math:`q = 0.1`. The
    filter's predicted variance converges to the steady state
    :math:`\bar P = (q + \sqrt{q^2 + 4q})/2`, the smoother halves the
    filter's error, and a run of missing observations costs nothing in
    likelihood:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> level = np.cumsum(np.sqrt(0.1) * rng.standard_normal(300))
    >>> y = level + rng.standard_normal(300)
    >>> model = LinearGaussianSSM(
    ...     design=[[1.0]], obs_cov=[[1.0]], transition=[[1.0]],
    ...     selection=[[1.0]], state_cov=[[0.1]],
    ... )
    >>> filtered = model.filter(y)
    >>> steady = float((0.1 + np.sqrt(0.1**2 + 4 * 0.1)) / 2)
    >>> round(float(filtered.predicted_state_cov[-1, 0, 0]), 4), round(steady, 4)
    (0.3702, 0.3702)
    >>> smoothed = model.smooth(y)
    >>> filter_mse = np.mean((filtered.filtered_state[:, 0] - level) ** 2)
    >>> smooth_mse = np.mean((smoothed.smoothed_state[:, 0] - level) ** 2)
    >>> bool(smooth_mse < 0.6 * filter_mse)
    True
    >>> gappy = y.copy()
    >>> gappy[100:120] = np.nan
    >>> float(model.filter(gappy).loglikelihood_contributions[100:120].sum())
    0.0
    >>> model.simulation_smoother(y, n_sims=50, seed=0).shape
    (50, 300, 1)
"""

from __future__ import annotations

from ..engine._internals import (
    _DurbinKoopmanSmootherResult as DurbinKoopmanSmootherResult,
)
from ..engine._internals import (
    _KalmanFilterResult as KalmanFilterResult,
)
from ..engine._internals import (
    _LinearGaussianStateSpace as LinearGaussianSSM,
)

__all__ = ["DurbinKoopmanSmootherResult", "KalmanFilterResult", "LinearGaussianSSM"]
