import numpy as np
import numpy.typing as npt

from .._core._stability import StabilityResult, assess_stability


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
