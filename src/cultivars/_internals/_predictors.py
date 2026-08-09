from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class MeanPredictor(Protocol):
    """A fitted, callable conditional-mean map produced by an engine.

    The contract is intentionally minimal: predict a conditional mean for a
    design matrix of lagged features, and report the number of free
    parameters so the consuming model can form information criteria.
    """

    @property
    def n_parameters(self) -> int:
        """Number of free parameters, used for AIC/BIC in the consuming model."""
        ...

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Conditional means for an ``(n, k)`` feature matrix, shape ``(n,)``."""
        ...


@dataclass(frozen=True)
class _FittedMLP:
    """A fitted single-hidden-layer perceptron; a concrete :class:`MeanPredictor`.

    Carries its own standardization constants so :meth:`predict` reproduces
    training-time scaling exactly on new data.
    """

    w1: npt.NDArray[np.float64]
    b1: npt.NDArray[np.float64]
    w2: npt.NDArray[np.float64]
    b2: float
    x_mean: npt.NDArray[np.float64]
    x_scale: npt.NDArray[np.float64]
    y_mean: float
    y_scale: float
    activation: str
    _n_parameters: int = field(repr=False, default=0)

    @property
    def n_parameters(self) -> int:
        """Total weight and bias count."""
        return self._n_parameters

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Conditional means for ``features`` of shape ``(n, k)``.

        Raises:
            DimensionError: If the feature width does not match training.
        """
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.w1.shape[0]:
            raise DimensionError(f"features must be (n, {self.w1.shape[0]}); got shape {x.shape}.")
        xs = (x - self.x_mean) / self.x_scale
        hidden = _activation(xs @ self.w1 + self.b1, self.activation)
        out = hidden @ self.w2 + self.b2
        return np.asarray(out * self.y_scale + self.y_mean, dtype=np.float64)
