# filepath: /src/cultivars/_internals/_models.py
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

"""Protected base classes and mixins for the model/result hierarchy.

Two field-carrying result bases form a single linear chain
(``_Result -> _MeanResult``); orthogonal capabilities are added by
behavior-only mixins (no fields, ``__slots__ = ()``) so that, e.g., an
ARMA-GARCH result composes a mean surface and a variance surface without a
diamond. Every result is ``frozen``, ``kw_only`` (so subclasses add fields
without reordering the base's required ones), and ``slots`` (validated to pickle
and to compose with the mixins). The bases satisfy the structural contracts in
:mod:`cultivars._core._protocols`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core._containers import InformationCriteria, information_criteria
from .._core._defaults import _SCHEMA_VERSION
from .._core._stability import StabilityResult, assess_stability
from .._core._validators import validate_endog
from ..exceptions import DimensionError


# ---- result field bases (single linear chain) ----
@dataclass(frozen=True, kw_only=True, slots=True)
class _Result:
    """Root of every fitted-result object: likelihood, sample size, criteria."""
    llf: float
    nobs: int
    n_params: int = field(repr=False)
    schema_version: int = field(default=_SCHEMA_VERSION, repr=False)

    @property
    def information_criteria(self) -> InformationCriteria:
        return information_criteria(self.llf, self.nobs, self.n_params)
    @property
    def aic(self) -> float:
        return self.information_criteria.aic
    @property
    def bic(self) -> float:
        return self.information_criteria.bic
    @property
    def hqic(self) -> float:
        return self.information_criteria.hqic


@dataclass(frozen=True, kw_only=True, slots=True)
class _MeanResult(_Result):
    """Adds the endog/residual/fitted surface shared by conditional-mean results."""
    endog: npt.NDArray[np.float64] = field(repr=False)
    resid: npt.NDArray[np.float64] = field(repr=False)
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)


# ---- behavior mixins (no fields; concrete result declares the attribute) ----
class _StationarityMixin:
    """Stationarity assessment over an autoregressive polynomial.

    The concrete result declares ``ar_params``. Results whose stationarity
    is governed by a *composed* polynomial (e.g. seasonal x non-seasonal AR
    in SARIMA) override :meth:`_stationarity_ar` to return that expansion.
    """
    __slots__ = ()
    ar_params: npt.NDArray[np.float64]

    def _stationarity_ar(self) -> npt.NDArray[np.float64]:
        return self.ar_params
    @property
    def stability(self) -> StabilityResult:
        return assess_stability(self._stationarity_ar())
    @property
    def is_stationary(self) -> bool:
        return self.stability.is_stable

class _ConditionalVarianceMixin:
    """Volatility surface for results carrying a conditional-variance path.

    Distinct from a scalar innovation variance (``sigma2``): the concrete
    result declares ``conditional_variance`` as the time-varying path.
    """
    __slots__ = ()
    conditional_variance: npt.NDArray[np.float64]
    @property
    def conditional_volatility(self) -> npt.NDArray[np.float64]:
        return np.sqrt(self.conditional_variance)

# ---- model spec bases ----
class _ModelBase[R](ABC):
    """Root of every model specification. Validates ``endog`` and fits to ``R``."""
    __slots__ = ("_endog",)
    def __init__(self, endog: npt.ArrayLike) -> None:
        self._endog = validate_endog(endog)
    @property
    def endog(self) -> npt.NDArray[np.float64]:
        return self._endog
    def _ensure_length(self, min_len: int, label: str) -> None:
        if self._endog.shape[0] < min_len:
            raise DimensionError(
                f"series of length {self._endog.shape[0]} is too short for {label} "
                f"(need at least {min_len})."
            )
    @abstractmethod
    def fit(self) -> R: ...

class _UnivariateModel[R](_ModelBase[R]):
    __slots__ = ()