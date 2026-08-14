# filepath: /src/cultivars/_internals/_results.py
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

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt

from .._core import (
    _SCHEMA_VERSION,
    InformationCriteria,
    companion_matrix,
)
from ..exceptions import DimensionError, NumericalError, SpecificationError


@dataclass(frozen=True)
class _StabilityResult:
    """The outcome of a stability (or invertibility) assessment.

    Attributes:
        eigenvalues: The companion eigenvalues (complex).
        max_modulus: The largest eigenvalue modulus; ``0.0`` when there are no
            eigenvalues (``p == 0``).
        is_stable: Whether the requested stability criterion is satisfied. With
            ``allow_unit_roots=False`` this means all moduli are strictly below
            ``1 - tol``; with ``allow_unit_roots=True`` it means no modulus
            exceeds ``1 + tol``.
        n_unit_roots: Number of eigenvalues whose modulus is within ``tol`` of 1.
        n_explosive: Number of eigenvalues with modulus above ``1 + tol``.
        tol: The modulus tolerance used for classification.
    """

    eigenvalues: npt.NDArray[np.complex128]
    max_modulus: float
    is_stable: bool
    n_unit_roots: int
    n_explosive: int
    tol: float

    @classmethod
    def _trivial(cls) -> Self:
        return cls(
            eigenvalues=np.empty(0, dtype=np.complex128),
            max_modulus=0.0,
            is_stable=True,
            n_unit_roots=0,
            n_explosive=0,
            tol=0.0,
        )

    @classmethod
    def _assess(
        cls, companion: npt.NDArray[np.float64], *, tol: float, allow_unit_roots: bool
    ) -> Self:
        if tol < 0.0:
            raise SpecificationError(f"tol must be non-negative; got {tol}.")
        if companion.size == 0:
            return cls(
                eigenvalues=np.empty(0, dtype=np.complex128),
                max_modulus=0.0,
                is_stable=True,
                n_unit_roots=0,
                n_explosive=0,
                tol=tol,
            )
        eigenvalues = np.linalg.eigvals(companion).astype(np.complex128)
        if not np.all(np.isfinite(eigenvalues)):
            raise NumericalError("Companion eigenvalue computation produced non-finite values.")
        moduli = np.abs(eigenvalues)
        max_modulus = float(moduli.max())
        n_unit_roots = int(np.count_nonzero(np.abs(moduli - 1.0) <= tol))
        n_explosive = int(np.count_nonzero(moduli > 1.0 + tol))
        is_stable = (n_explosive == 0) if allow_unit_roots else (max_modulus < 1.0 - tol)
        return cls(
            eigenvalues=eigenvalues,
            max_modulus=max_modulus,
            is_stable=is_stable,
            n_unit_roots=n_unit_roots,
            n_explosive=n_explosive,
            tol=tol,
        )

    @classmethod
    def assess_stability(
        cls, ar_coeffs: npt.ArrayLike, *, tol: float = 1e-8, allow_unit_roots: bool = False
    ) -> Self:
        """Assess stationarity of an AR/VAR from its autoregressive coefficients.

        Args:
            ar_coeffs: Coefficients ``A_1, ..., A_p``; shape ``(p,)`` or ``(p, k, k)``.
            tol: Modulus tolerance for classifying unit and explosive roots.
            allow_unit_roots: If ``True``, unit roots are permitted (only strictly
                explosive roots make the model unstable). Use for VECM and other
                models that carry unit roots by design.

        Returns:
            A :class:`StabilityResult`.

        Example:
            >>> res = assess_stability([0.5])
            >>> res.is_stable
            True
            >>> round(res.max_modulus, 4)
            0.5
        """
        ar = np.asarray(ar_coeffs, dtype=np.float64)
        if ar.size == 0:
            return cls._trivial()
        return cls._assess(companion_matrix(ar), tol=tol, allow_unit_roots=allow_unit_roots)

    @classmethod
    def assess_stability_from_companion(
        cls, companion: npt.ArrayLike, *, tol: float = 1e-8, allow_unit_roots: bool = False
    ) -> Self:
        """Assess stability directly from a companion (or state-transition) matrix.

        Args:
            companion: A square matrix (e.g. a companion or an LGSS transition matrix).
            tol: Modulus tolerance for classifying unit and explosive roots.
            allow_unit_roots: If ``True``, unit roots are permitted.

        Returns:
            A :class:`StabilityResult`.

        Raises:
            DimensionError: If ``companion`` is not a square 2-D array.
        """
        mat = np.asarray(companion, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise DimensionError(f"Companion matrix must be square 2-D; got shape {mat.shape}.")
        return cls._assess(mat, tol=tol, allow_unit_roots=allow_unit_roots)

    @classmethod
    def is_stationary(cls, ar_coeffs: npt.ArrayLike, *, tol: float = 1e-8) -> bool:
        """Convenience predicate: is the AR/VAR strictly stationary?

        Example:
            >>> is_stationary([1.5])
            False
        """
        return cls.assess_stability(ar_coeffs, tol=tol, allow_unit_roots=False).is_stable

    @classmethod
    def is_invertible(cls, ma_coeffs: npt.ArrayLike, *, tol: float = 1e-8) -> bool:
        """Is an MA/ARMA invertible? (companion of the MA polynomial, roots inside).

        Args:
            ma_coeffs: MA coefficients ``M_1, ..., M_q`` in the same layout as AR
                coefficients; shape ``(q,)`` or ``(q, k, k)``.
            tol: Modulus tolerance.

        Returns:
            ``True`` iff all companion eigenvalues lie strictly inside the unit circle.
        """
        ma = np.asarray(ma_coeffs, dtype=np.float64)
        if ma.size == 0:
            return True
        return cls._assess(companion_matrix(ma), tol=tol, allow_unit_roots=False).is_stable


@dataclass(frozen=True, kw_only=True, slots=True)
class _FittedResult:
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
        return InformationCriteria.from_likelihood(self.llf, self.nobs, self.n_params)

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


@dataclass(frozen=True, kw_only=True)
class _FilterResult:
    """Common ancestor of every forward-filtering output.

    Any filter -- Kalman, Hamilton, extended, particle -- produces a
    likelihood decomposition, and nothing else in common: the Kalman filter
    carries a continuous state and its covariance, the Hamilton filter a
    discrete regime distribution. Those are genuinely different objects, so
    only the likelihood lives here.

    Attributes:
        loglikelihood: Total log-likelihood of the data under the model.
        loglikelihood_contributions: Per-period contributions, shape ``(n,)``.
    """

    loglikelihood: float
    loglikelihood_contributions: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class _SmootherResult:
    """Common ancestor of every backward-smoothing output.

    Deliberately empty of fields: smoothed states and smoothed regime
    probabilities share no data, only the fact that both condition on the
    full sample. This exists to bound the ``S`` parameter of
    :class:`StateSpaceModel`, not to factor out shared state.
    """


@dataclass(frozen=True, kw_only=True, slots=True)
class _MeanResult(_FittedResult):
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


@dataclass(frozen=True, kw_only=True)
class _KalmanFilterResult(_FilterResult):
    """Output of a linear-Gaussian forward filtering pass.

    Attributes:
        predicted_state: One-step-ahead predicted states ``a_{t|t-1}``, ``(n, m)``.
        predicted_state_cov: Predicted state covariances ``P_{t|t-1}``, ``(n, m, m)``.
        filtered_state: Contemporaneously filtered states ``a_{t|t}``, ``(n, m)``.
        filtered_state_cov: Filtered state covariances ``P_{t|t}``, ``(n, m, m)``.
    """

    predicted_state: npt.NDArray[np.float64]
    predicted_state_cov: npt.NDArray[np.float64]
    filtered_state: npt.NDArray[np.float64]
    filtered_state_cov: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class _DurbinKoopmanSmootherResult(_SmootherResult):
    """Output of a linear-Gaussian backward smoothing pass.

    Attributes:
        smoothed_state: Smoothed states ``a_{t|n}``, shape ``(n, m)``.
        smoothed_state_cov: Smoothed state covariances ``V_{t|n}``, ``(n, m, m)``.
    """

    smoothed_state: npt.NDArray[np.float64]
    smoothed_state_cov: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class _HamiltonFilterResult(_FilterResult):
    """Output of the Hamilton forward filter.

    Attributes:
        filtered_prob: Contemporaneous regime probabilities
            ``Pr(S_t = j | y_{1..t})``, shape ``(T, K)``.
        predicted_prob: One-step-ahead regime probabilities
            ``Pr(S_t = j | y_{1..t-1})``, shape ``(T, K)``.
        loglikelihood: Total log-likelihood ``sum_t log Pr(y_t | y_{1..t-1})``.
        loglikelihood_contributions: Per-period contributions, shape ``(T,)``.
    """

    filtered_prob: npt.NDArray[np.float64]
    predicted_prob: npt.NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class _KimSmootherResult(_SmootherResult):
    """Output of the Kim backward smoother.

    Attributes:
        smoothed_prob: Full-sample regime probabilities
            ``Pr(S_t = j | y_{1..T})``, shape ``(T, K)``.
        smoothed_joint_prob: Consecutive-pair probabilities
            ``Pr(S_t = i, S_{t+1} = j | y_{1..T})``, shape ``(T - 1, K, K)``;
            entry ``[t]`` links period ``t`` to ``t + 1``. Empty when ``T == 1``.
            These are the expected-transition weights of the EM M-step.
    """

    smoothed_prob: npt.NDArray[np.float64]
    smoothed_joint_prob: npt.NDArray[np.float64]
