from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from .._core._validators import validate_choice, validate_exog, validate_order_tuple
from ..exceptions import SpecificationError
from ._models import _UnivariateModel

_TRENDS = ("n", "c", "ct")

@dataclass(frozen=True, slots=True)
class _SARIMAXFit:
    """Raw outputs of a (seasonal) ARIMA-with-regressors fit."""
    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]
    seasonal_ar_params: npt.NDArray[np.float64]
    seasonal_ma_params: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    sigma2: float
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int

class _SARIMAXModel[R](_UnivariateModel[R]):
    """Shared specification surface for the ARMA/ARIMA/SARIMA family.

    ARMA, ARIMA and SARIMA are the same state-space estimator under three
    parameterizations; this base owns the validation and the engine call,
    and the public leaves supply only their defaults.
    """
    __slots__ = ("_order", "_seasonal", "_trend", "_exog")

    def __init__(
        self, endog: npt.ArrayLike, *, order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str = "c", exog: npt.ArrayLike | None = None,
    ) -> None:
        super().__init__(endog)
        p, d, q = validate_order_tuple(order, ("p", "d", "q"))
        cap_p, cap_d, cap_q, s = validate_order_tuple(
            seasonal_order, ("P", "D", "Q", "s")
        )
        if (cap_p or cap_d or cap_q) and s < 2:
            raise SpecificationError(
                f"seasonal period s must be >= 2 when seasonal terms are present; got s={s}."
            )
        self._order = (p, d, q)
        self._seasonal = (cap_p, cap_d, cap_q, s)
        self._trend = validate_choice(trend, _TRENDS, "trend")
        self._exog = validate_exog(exog, self.endog.shape[0])
        self._ensure_length(
            p + d + q + s * (cap_p + cap_d + cap_q) + 2,
            f"SARIMA{self._order}{self._seasonal}",
        )

    @property
    def order(self) -> tuple[int, int, int]:
        return self._order
    @property
    def seasonal_order(self) -> tuple[int, int, int, int]:
        return self._seasonal
    @property
    def trend(self) -> str:
        return self._trend
    @property
    def exog(self) -> npt.NDArray[np.float64] | None:
        return self._exog

    def _fit_family(self) -> _SARIMAXFit:
        """Run the shared state-space engine for this specification."""
        return _fit_sarimax(
            self.endog, self._exog, self._order, self._seasonal, self._trend
        )