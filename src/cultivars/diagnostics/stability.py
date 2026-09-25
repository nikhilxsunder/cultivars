# filepath: /src/cultivars/diagnostics/stability.py
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
r"""Stability and invertibility: the companion eigenvalues that decide whether a model can be used.

An autoregression :math:`y_t = A_1 y_{t-1} + \cdots + A_p y_{t-p} + u_t`
is stable, and so has a convergent moving-average representation, a
finite unconditional variance, and forecasts that settle, exactly when
every eigenvalue of its companion matrix

.. math::

   \mathbf{A} = \begin{pmatrix}
     A_1 & A_2 & \cdots & A_{p-1} & A_p \\
     I   & 0   & \cdots & 0       & 0   \\
     0   & I   & \cdots & 0       & 0   \\
     \vdots & & \ddots & & \vdots \\
     0   & 0   & \cdots & I       & 0
   \end{pmatrix}

lies strictly inside the unit circle. A moving-average polynomial is
invertible under the same condition on its own companion, with the
sign convention flipped, and a state-space transition matrix is
assessed the same way. This is not a hypothesis test but a
deterministic reading of a fitted model: there is no null, no p-value,
and no sampling variation in the verdict, only the eigenvalues, their
largest modulus, and a count of how many sit on or beyond the circle.
It is nonetheless the diagnostic every other one presumes. An impulse
response computed from an explosive companion diverges rather than
decays, a forecast from one is meaningless at any horizon, and a
Ljung-Box statistic on its residuals is a number without a law, which
is why every model result exposes the assessment and why the reading
belongs among the diagnostics.

Two commitments shape the surface. First, a unit root is a category of
its own, not an explosive root that happens to be small. The record
counts eigenvalues within ``tol`` of the circle separately from those
beyond it, and ``is_stable`` answers the question the model asked: a
VAR asks that every modulus be strictly below one, while a VECM, which
carries :math:`k - r` unit roots by construction, asks only that
nothing be explosive and passes ``allow_unit_roots=True``. Second, the
verdict names its tolerance. Whether a modulus of :math:`0.99999999`
is a unit root or a stable root is a numerical question, and the
record carries the ``tol`` that decided it so that two assessments can
be compared on the same footing.

Layout. :class:`StabilityTest` is the record, re-exported here under
its public name from ``_internals``, where the models that produce it
live. It is reached from a fitted result: ``stability`` and
``invertibility`` on the ARMA-family results, ``stability_check`` and
``is_stable`` on the VAR family and the error-correction models, and
the Markov-switching results per regime; its class methods
:meth:`~StabilityTest.assess_stability`,
:meth:`~StabilityTest.assess_stability_from_companion`,
:meth:`~StabilityTest.is_stationary`, and
:meth:`~StabilityTest.is_invertible` read a coefficient array or a
companion matrix directly. The companion is built by ``_core``'s
``companion_matrix``; the eigenvalues are NumPy's.

References:
    Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
    Analysis*. Springer. Chapter 2.

    Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University
    Press. Chapter 1.

    Brockwell, P. J., & Davis, R. A. (1991). *Time Series: Theory and
    Methods* (2nd ed.). Springer. Chapter 3.

Example:
    A fitted ARMA reports its autoregressive and moving-average verdicts
    separately, and the record can be built from coefficients alone;
    a unit root is counted, not called explosive:

    >>> import numpy as np
    >>> from cultivars.univariate.box_jenkins import ARIMA
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros(200)
    >>> for t in range(1, 200):
    ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal()
    >>> res = ARIMA(y, order=(1, 0, 1)).fit()
    >>> res.stability.is_stable, res.invertibility.is_stable
    (True, True)
    >>> round(res.stability.max_modulus, 2)
    0.75
    >>> walk = StabilityTest.assess_stability([1.0])
    >>> walk.is_stable, walk.n_unit_roots, walk.n_explosive
    (False, 1, 0)
    >>> StabilityTest.assess_stability([1.0], allow_unit_roots=True).is_stable
    True
"""

from __future__ import annotations

from .._internals import _StabilityAssessment as StabilityTest

__all__ = ["StabilityTest"]
