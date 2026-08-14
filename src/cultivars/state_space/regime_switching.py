# filepath: /src/cultivars/state_space/regime_switching.py
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
"""Discrete Markov-switching inference: the Hamilton filter and Kim smoother.

Regime-switching models in cultivars separate two concerns:

- The *regime chain* — a first-order ``K``-state Markov process with transition
  matrix ``P`` — and the recursions that infer regime probabilities from a
  sequence of per-regime conditional data densities. The Hamilton (1989) forward
  filter and the Kim (1994) backward smoother are generic: they do not know what
  model produced the densities. They live here.
- The *observation model* — how ``Pr(y_t | S_t = j, past)`` is formed — is
  model-specific (a switching AR mean, a switching variance, a switching VAR).
  Each model supplies the ``(T, K)`` matrix of log conditional densities and
  consumes the filter/smoother output. See :mod:`cultivars.univariate.ms_ar`.

This factoring is what lets one filter serve MS-AR today and MS-VAR tomorrow:
only the density map changes. A future continuous-state
``RegimeSwitchingStateSpace`` — the Kim-Nelson filter, which runs one Kalman
filter per regime and collapses the state across regimes at each step — layers
directly on top of these same two recursions.

Conventions:

- The transition matrix is **row-stochastic**:
  ``transition[i, j] = Pr(S_t = j | S_{t-1} = i)`` and each row sums to one.
- Regime-probability vectors are length ``K`` and the one-step-ahead prediction
  is ``xi_pred = xi_filt @ transition``.
- Conditional densities are passed in **logarithms** so that near-zero regime
  likelihoods (small variances, outliers) do not underflow; the filter combines
  them with a log-sum-exp.

References:
    Hamilton, J. D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. *Econometrica*, 57(2).
    Hamilton, J. D. (1994). *Time Series Analysis*, ch. 22.
    Kim, C.-J. (1994). Dynamic linear models with Markov-switching.
    *Journal of Econometrics*, 60(1-2).
    Kim, C.-J. & Nelson, C. R. (1999). *State-Space Models with Regime
    Switching*.
"""

from __future__ import annotations
