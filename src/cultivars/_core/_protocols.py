# filepath: /src/cultivars/_core/_protocols.py
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

"""Structural contracts satisfied by fitted results and model specifications.

These are the duck-typed interfaces the toolkit subpackages (diagnostics,
forecast, spectral) program against. They are deliberately *structural*
rather than nominal: a user-defined result that happens to expose ``llf``,
``nobs`` and the criteria surface is accepted by every diagnostic without
subclassing anything from this package.

The nominal counterparts -- the base classes that concrete results actually
inherit -- live in :mod:`cultivars._internals._models` and implement these
protocols.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ._containers import InformationCriteria


@runtime_checkable
class FittedResult(Protocol):
    """The universal contract of any fitted model result."""

    llf: float
    nobs: int

    @property
    def information_criteria(self) -> InformationCriteria: ...
    @property
    def aic(self) -> float: ...
    @property
    def bic(self) -> float: ...
    @property
    def hqic(self) -> float: ...


@runtime_checkable
class Forecaster(Protocol):
    """Anything that produces point forecasts over a horizon."""

    def forecast(self, h: int, /) -> npt.NDArray[np.float64]: ...


@runtime_checkable
class MeanModelResult(FittedResult, Protocol):
    """A fitted conditional-mean model: residuals, fitted values, forecasts."""

    endog: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]

    def forecast(self, h: int, /) -> npt.NDArray[np.float64]: ...


@runtime_checkable
class VolatilityResult(FittedResult, Protocol):
    """A fitted model carrying a time-varying conditional-variance path."""

    conditional_variance: npt.NDArray[np.float64]

    @property
    def conditional_volatility(self) -> npt.NDArray[np.float64]: ...


@runtime_checkable
class TimeSeriesModel[R](Protocol):
    """A model specification that validates data and fits to a result ``R``."""

    @property
    def endog(self) -> npt.NDArray[np.float64]: ...
    def fit(self) -> R: ...


@runtime_checkable
class ClosedSystemResult(Protocol):
    """What an identification strategy reads from a fitted reduced-form result.

    Deliberately the *propagation* subset rather than the full result surface:
    identification turns reduced-form innovations into structural shocks, and
    everything that requires is the innovation covariance, the residuals, the
    autoregressive representation, and the labels. Any closed result in the
    package satisfies this -- a VAR, a VARMA, a panel, an error-correction
    model through its levels form, a mixed-frequency filter -- and so does any
    user-defined result that exposes the same members.
    """

    names: tuple[str, ...]
    nobs: int
    sigma_u: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    coefficients: npt.NDArray[np.float64]

    @property
    def k_endog(self) -> int: ...
    def ma_representation(self, horizon: int = ...) -> npt.NDArray[np.float64]: ...


class Identification[R](Protocol):
    """An identification scheme: reduced-form innovations in, a structural view out.

    The strategy-object contract behind ``result.identify(strategy)``. A scheme
    is a declaration -- which restrictions turn correlated innovations into
    economic shocks -- and this protocol is what lets that declaration be
    passed to any closed reduced-form result rather than being welded to one
    model family. Point-identified schemes return a single structural result;
    set-identified schemes return the accepted set.
    """

    def identify(self, result: ClosedSystemResult) -> R: ...
