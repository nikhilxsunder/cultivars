from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from .._core._validators import validate_aligned, validate_open_interval, validate_order
from ._models import _UnivariateModel

@dataclass(frozen=True, slots=True)
class _ThresholdFit:
    """Raw outputs of a two-regime threshold-autoregression grid search."""
    delay: int
    threshold: float
    lower_params: npt.NDArray[np.float64]
    upper_params: npt.NDArray[np.float64]
    sigma2: float
    ssr: float
    n_lower: int
    n_upper: int
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int

class _ThresholdModel[R](_UnivariateModel[R]):
    """Shared specification surface for SETAR/TAR grid-search models.

    SETAR is the self-exciting case (the threshold variable is a lag of the
    series); TAR supplies an external threshold variable. Both search the
    same trimmed grid, so both share this base.
    """
    __slots__ = ("_order", "_delays", "_trim", "_n_grid", "_threshold_variable")

    def __init__(
        self, endog: npt.ArrayLike, *, order: int, delay: int | None = None,
        trim: float = 0.15, n_grid: int = 300,
        threshold_variable: npt.ArrayLike | None = None,
    ) -> None:
        super().__init__(endog)
        self._order = validate_order(order, "order", minimum=1)
        self._trim = validate_open_interval(trim, "trim", low=0.0, high=0.5)
        self._n_grid = validate_order(n_grid, "n_grid", minimum=1)
        self._delays = (
            [validate_order(delay, "delay", minimum=1)]
            if delay is not None else list(range(1, self._order + 1))
        )
        self._threshold_variable = (
            None if threshold_variable is None
            else validate_aligned(
                threshold_variable, self.endog.shape[0], "threshold_variable"
            )
        )
        self._ensure_length(
            2 * (self._order + 1) + max(self._delays), f"threshold AR({self._order})"
        )

    @property
    def order(self) -> int:
        return self._order
    @property
    def self_exciting(self) -> bool:
        return self._threshold_variable is None

    def _fit_family(self) -> _ThresholdFit:
        """Run the shared grid-search engine for this specification."""
        return _fit_threshold(
            self.endog, self._order, self._delays, self._trim, self._n_grid,
            self._threshold_variable,
        )