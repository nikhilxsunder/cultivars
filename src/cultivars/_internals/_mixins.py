# filepath: /src/cultivars/_internals/_mixins.py
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

import numpy as np
import numpy.typing as npt

from ._results import _StabilityResult


class _StationarityMixin:
    """Stationarity assessment over an autoregressive polynomial.

    The concrete result declares ``ar_params``. Results whose stationarity is
    governed by a *composed* polynomial -- seasonal times non-seasonal AR in
    SARIMA -- override :meth:`_stationarity_ar` to return that expansion;
    assessing the raw non-seasonal block alone would silently pass a model with
    an explosive seasonal root.

    Not applied to Markov-switching results: their ``ar_params`` is ``(K, p)``
    and no single companion polynomial describes the process.
    """

    __slots__ = ()

    ar_params: npt.NDArray[np.float64]

    def _stationarity_ar(self) -> npt.NDArray[np.float64]:
        """The polynomial whose roots determine stationarity."""
        return self.ar_params

    @property
    def stability(self) -> _StabilityResult:
        """Full eigenvalue verdict for the autoregressive polynomial."""
        return _StabilityResult.assess_stability(self._stationarity_ar())

    @property
    def is_stationary(self) -> bool:
        """Whether every companion eigenvalue lies inside the unit circle."""
        return self.stability.is_stable


class _ConditionalVarianceMixin:
    """Volatility surface for results carrying a conditional-variance path.

    Reads ``conditional_variance``, which is deliberately a different name from
    the scalar ``sigma2`` on homoskedastic mean models: one is a path of shape
    ``(nobs,)``, the other a single float, and overloading one name across both
    shapes invites silent shape bugs downstream.
    """

    __slots__ = ()

    conditional_variance: npt.NDArray[np.float64]

    @property
    def conditional_volatility(self) -> npt.NDArray[np.float64]:
        """Conditional standard deviation, ``sqrt(conditional_variance)``."""
        return np.sqrt(self.conditional_variance)
