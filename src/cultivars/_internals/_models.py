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

"""Protected base classes for model specifications and fitted results.

Two hierarchies live here, and they are deliberately separate.

Results inherit a single linear chain of *field* bases -- ``_Result`` then
``_MeanResult`` -- because multiple slotted dataclass bases that each declare
fields raise ``TypeError: multiple bases have instance lay-out conflict``.
Anything optional is a *behavior* mixin with ``__slots__ = ()`` and no fields
of its own; the concrete result declares the attribute the mixin reads.

Model specifications inherit ``_ModelBase``, which owns ``endog`` validation
and the length check. Every family base in this subpackage descends from it,
so no public constructor re-implements either.

These bases implement the structural contracts in
:mod:`cultivars._core._protocols` without importing them: the protocols are
duck-typed, so the relationship is checked by ``isinstance`` at runtime and by
``mypy`` statically, with no import edge in either direction.
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


@dataclass(frozen=True, kw_only=True, slots=True)
class _Result:
    """Root of every fitted-result object.

    Carries the likelihood summary every estimator produces and derives the
    information criteria from it, so no subclass stores ``aic``/``bic``/``hqic``
    as fields that could drift out of sync with ``llf``.

    Attributes:
        llf: Maximized log-likelihood.
        nobs: Observations the likelihood was evaluated on.
        n_params: Free parameter count, including the innovation variance.
        schema_version: Serialization schema version.
    """

    llf: float
    nobs: int
    n_params: int = field(repr=False)
    schema_version: int = field(default=_SCHEMA_VERSION, repr=False)

    @property
    def information_criteria(self) -> InformationCriteria:
        """All three model-selection criteria for this fit."""
        return information_criteria(self.llf, self.nobs, self.n_params)

    @property
    def aic(self) -> float:
        """Akaike information criterion."""
        return self.information_criteria.aic

    @property
    def bic(self) -> float:
        """Bayesian (Schwarz) information criterion."""
        return self.information_criteria.bic

    @property
    def hqic(self) -> float:
        """Hannan-Quinn information criterion."""
        return self.information_criteria.hqic


@dataclass(frozen=True, kw_only=True, slots=True)
class _MeanResult(_Result):
    """Adds the endog/residual/fitted surface shared by conditional-mean results.

    Attributes:
        endog: The observed series, retained so forecasts can be produced from
            the result alone without holding a reference to the model.
        resid: One-step residuals on the estimation sample.
        fittedvalues: One-step fitted values on the estimation sample.
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    resid: npt.NDArray[np.float64] = field(repr=False)
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)


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
    def stability(self) -> StabilityResult:
        """Full eigenvalue verdict for the autoregressive polynomial."""
        return assess_stability(self._stationarity_ar())

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


class _ModelBase[R](ABC):
    """Root of every model specification, fitting to a result of type ``R``.

    Owns ``endog`` coercion and validation so that a malformed series produces
    an identical error regardless of which model was constructed.

    Args:
        endog: The observed univariate series (1-D array-like).

    Raises:
        DimensionError: If ``endog`` is not one-dimensional.
        NumericalError: If ``endog`` contains non-finite values.
    """

    __slots__ = ("_endog",)

    def __init__(self, endog: npt.ArrayLike) -> None:
        """Validate and store the endogenous series."""
        self._endog = validate_endog(endog)

    @property
    def endog(self) -> npt.NDArray[np.float64]:
        """The validated endogenous series."""
        return self._endog

    def _ensure_length(self, min_len: int, label: str) -> None:
        """Reject a series too short to identify the specification.

        Args:
            min_len: Minimum admissible number of observations.
            label: Human-readable specification name for the error message.

        Raises:
            DimensionError: If the series is shorter than ``min_len``.
        """
        if self._endog.shape[0] < min_len:
            raise DimensionError(
                f"series of length {self._endog.shape[0]} is too short for {label} "
                f"(need at least {min_len})."
            )

    @abstractmethod
    def fit(self) -> R:
        """Estimate the model and return its result object."""
        ...


class _UnivariateModel[R](_ModelBase[R]):
    """Base for single-series models. Reserved for univariate-only behavior."""

    __slots__ = ()
