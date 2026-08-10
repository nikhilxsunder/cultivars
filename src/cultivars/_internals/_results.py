from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core._containers import InformationCriteria
from .._core._defaults import _SCHEMA_VERSION


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
