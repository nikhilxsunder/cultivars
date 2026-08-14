# filepath: /src/cultivars/_internals/_predictors.py
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
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError
from ._engines import NumpyMLPEngine


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
        hidden = NumpyMLPEngine._activation(xs @ self.w1 + self.b1, self.activation)
        out = hidden @ self.w2 + self.b2
        return np.asarray(out * self.y_scale + self.y_mean, dtype=np.float64)
