# filepath: /src/cultivars/_internals/_parameters.py
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

"""Optimizer surfaces: the estimation context an objective function closes over.

An objective function is a closure over the estimation context — the data, the
design blocks, the pre-sample constants, the specification orders. This module
makes that context an object instead of a lexical scope, so the criterion, the
parameter map, and the intermediate quantities every fit method needs after
convergence are all reachable by name rather than trapped inside a ``def``
nested in a classmethod.

Each concrete objective is a frozen, slotted dataclass whose fields are exactly
the variables the old closures captured, and whose methods are exactly the old
closures. ``__call__`` makes the instance directly passable to
:func:`scipy.optimize.minimize` with no ``functools.partial`` and no ``args=``.

The optimizer configuration lives on the objective, not at the call site,
because the choice of algorithm is a property of the surface: a concentrated
sum of squares that is non-smooth in the threshold wants Nelder-Mead, while a
reparameterized Gaussian likelihood wants L-BFGS-B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, kw_only=True, slots=True)
class _AutoRegressionParameters:
    """Structured parameters of an AR(p) exact-ML draw.

    Attributes:
        const: Mean intercept; ``0.0`` when the specification has no constant.
        ar_params: Autoregressive coefficients, already mapped back from the
            partial-autocorrelation parameterization.
        sigma2: Innovation variance, already exponentiated.
    """

    const: float
    ar_params: npt.NDArray[np.float64]
    sigma2: float


@dataclass(frozen=True, kw_only=True, slots=True)
class _BoxJenkinsParameters:
    """Structured parameters of a seasonal ARIMA-with-regressors draw.

    Attributes:
        beta: Coefficients on the deterministic and exogenous block.
        ar_params: Non-seasonal AR coefficients.
        seasonal_ar_params: Seasonal AR coefficients.
        ma_params: Non-seasonal MA coefficients, in the sign convention of the
            observation equation.
        seasonal_ma_params: Seasonal MA coefficients.
        sigma2: Innovation variance.
    """

    beta: npt.NDArray[np.float64]
    ar_params: npt.NDArray[np.float64]
    seasonal_ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    seasonal_ma_params: npt.NDArray[np.float64]
    sigma2: float


@dataclass(frozen=True, kw_only=True, slots=True)
class _FractionalIntegrationParameters:
    """Structured parameters of an ARFIMA(p, d, q) draw.

    Attributes:
        mean: Estimated mean; ``0.0`` when the mean is not estimated.
        d: Fractional integration order, mapped into ``(-_D_MAX, _D_MAX)``.
        ar_params: Short-memory AR coefficients.
        ma_params: Short-memory MA coefficients.
        sigma2: Innovation variance.
    """

    mean: float
    d: float
    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    sigma2: float


@dataclass(frozen=True, kw_only=True, slots=True)
class _VarianceParameters:
    """Parameters every conditional-variance draw carries.

    Attributes:
        mean: Conditional-mean coefficients, intercept first when present.
        omega: Variance intercept, in levels or in logs depending on the family.
    """

    mean: npt.NDArray[np.float64]
    omega: float


@dataclass(frozen=True, kw_only=True, slots=True)
class _ConditionalVarianceParameters(_VarianceParameters):
    """Structured parameters of a finite-order variance draw.

    Attributes:
        alpha: Coefficients on the shock magnitude.
        gamma: Asymmetry coefficients; empty for the symmetric case.
        beta: Persistence coefficients.
    """

    alpha: npt.NDArray[np.float64]
    gamma: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True, slots=True)
class _FractionalVarianceParameters(_VarianceParameters):
    """Structured parameters of a fractionally integrated variance draw.

    Attributes:
        phi: Short-memory numerator weight, in ``(0, 1)``.
        d: Fractional integration order, in ``(0, 1)``.
        beta: Denominator weight, in ``(0, 1)``.
    """

    phi: float
    d: float
    beta: float


@dataclass(frozen=True, kw_only=True, slots=True)
class _SmoothTransitionParameters:
    """Structured parameters of a smooth-transition draw.

    The regime coefficients are absent by design: they are concentrated out and
    recovered by :meth:`_SmoothTransitionObjective.least_squares`.

    Attributes:
        gamma: Transition speed, strictly positive.
        threshold: Transition location, on the scale of the transition variable.
    """

    gamma: float
    threshold: float
