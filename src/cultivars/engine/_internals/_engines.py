# filepath: /src/cultivars/engine/_internals/_engines.py
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
"""Engines: the training backends that turn features and a target into a predictor.

The neural mean-function models split training from prediction. An engine
owns training: given a design matrix of lagged values and a target it fits
a learner and returns a predictor, which the model then treats as given
(:mod:`~cultivars.engine._internals._predictors`). :class:`MeanFunctionEngine`
is the contract, one method wide, so that a learner from another library can be
adapted to it. :class:`NumpyMLPEngine` is the reference implementation, a
single-hidden-layer perceptron trained by L-BFGS with an analytic gradient and
no dependency beyond NumPy and SciPy.

The reference objective, on standardized inputs and target, is::

    L = (1 / 2n) * sum_i (m(x_i) - y_i)^2 + (alpha / 2) * (|w1|^2 + |w2|^2)

with the biases left unpenalized.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from ...exceptions import DimensionError, NumericalError, SpecificationError
from .._core._defaults import _DEFAULT_MAX_ITER
from ._predictors import MeanPredictor, _FittedMLP

__all__ = [
    "MeanFunctionEngine",
    "NumpyMLPEngine",
]


@runtime_checkable
class MeanFunctionEngine(Protocol):
    """The plug-in training backend for a nonlinear mean function.

    Any object exposing ``fit(features, target) -> MeanPredictor`` satisfies
    the contract, so a torch / jax / sklearn learner drops in by adapting to
    this one method.

    See Also:
        * :class:`~cultivars.univariate.mean_function.MeanFunctionEngine` --
          the public name of this class.
    """

    def fit(
        self, features: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
    ) -> MeanPredictor:
        """Fit the learner and return a :class:`MeanPredictor`.

        Args:
            features: Design matrix, ``(n, k)``.
            target: Response, ``(n,)``, aligned with the rows of
                ``features``.

        Returns:
            A fitted predictor. The models that call this do not check the
            return against the
            :class:`~cultivars.engine._internals._predictors.MeanPredictor`
            protocol.
        """
        ...


@dataclass(frozen=True)
class NumpyMLPEngine:
    """Reference :class:`MeanFunctionEngine`: an L-BFGS-trained MLP.

    One hidden layer, linear output, L2 penalty on weights but not biases.
    Inputs and target are standardized internally for conditioning. The
    objective is non-convex, so training restarts from several random
    initializations and keeps the best; the analytic gradient is supplied to
    the optimizer rather than relying on finite differences.

    Args:
        hidden_units: Number of hidden units.
        activation: ``"tanh"`` or ``"relu"``.
        alpha: L2 penalty on the weights.
        max_iter: Maximum L-BFGS iterations per restart.
        n_restarts: Number of random initializations.
        seed: Seed for the initialization RNG.

    Raises:
        SpecificationError: If ``hidden_units < 1``, ``n_restarts < 1``, or the
            activation is unrecognized.

    Note:
        ``alpha`` and ``max_iter`` are not validated, and the optimizer's
        convergence flag is not consulted: a restart that stops at the
        iteration cap is accepted like any other.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal((80, 2))
        >>> y = np.tanh(x[:, 0]) + 0.1 * rng.standard_normal(80)
        >>> predictor = NumpyMLPEngine(hidden_units=3).fit(x, y)
        >>> predictor.n_parameters, predictor.predict(x).shape
        (13, (80,))
    """

    hidden_units: int = 8
    """Number of hidden units :math:`h`; at least 1."""
    activation: str = "tanh"
    """Hidden activation, ``"tanh"`` or ``"relu"``."""
    alpha: float = 1e-4
    """L2 penalty on both weight blocks; the biases are not penalized."""
    max_iter: int = _DEFAULT_MAX_ITER
    """Maximum L-BFGS iterations per restart."""
    n_restarts: int = 3
    """Number of random initializations; the lowest final loss is kept."""
    seed: int | None = 0
    """Seed of the initialization generator.

    The default of ``0`` makes a fit reproducible; ``None`` draws fresh
    entropy on every call.
    """

    def __post_init__(self) -> None:
        """Validate the engine configuration.

        Raises:
            SpecificationError: If ``hidden_units < 1``, ``n_restarts < 1``,
                or the activation is unrecognized.
        """
        if self.hidden_units < 1:
            raise SpecificationError(f"hidden_units must be >= 1; got {self.hidden_units}.")
        if self.n_restarts < 1:
            raise SpecificationError(f"n_restarts must be >= 1; got {self.n_restarts}.")
        if self.activation not in ("tanh", "relu"):
            raise SpecificationError(
                f"activation must be 'tanh' or 'relu'; got {self.activation!r}."
            )

    def fit(self, features: npt.NDArray[np.float64], target: npt.NDArray[np.float64]) -> _FittedMLP:
        """Train the MLP and return a fitted predictor.

        Standardizes the features column by column and the target, with a
        scale of one substituted for a column or target that does not vary,
        then minimizes the penalized squared error from ``n_restarts``
        random starts and keeps the lowest loss.

        Args:
            features: Design matrix of shape ``(n, k)``.
            target: Response of shape ``(n,)``.

        Returns:
            The fitted :class:`~cultivars.engine._internals._predictors._FittedMLP`.

        Raises:
            DimensionError: If the shapes are not conformable.
            NumericalError: If inputs are non-finite, or no restart returns
                a finite loss. Reaching the iteration cap is not an error.
        """
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        if x.ndim != 2:
            raise DimensionError(f"features must be 2-D (n, k); got shape {x.shape}.")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise DimensionError(
                f"target must be 1-D and aligned with features; got {y.shape} vs {x.shape}."
            )
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise NumericalError("features/target contain non-finite values.")

        n, k = x.shape
        h = self.hidden_units
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x_scale = np.where(x_std > 0.0, x_std, 1.0)
        y_mean = float(y.mean())
        y_scale = float(y.std()) if y.std() > 0.0 else 1.0
        xs = (x - x_mean) / x_scale
        ys = (y - y_mean) / y_scale

        n_w1, n_b1, n_w2 = k * h, h, h
        size = n_w1 + n_b1 + n_w2 + 1

        def unpack(
            theta: npt.NDArray[np.float64],
        ) -> tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            float,
        ]:
            """Split a flat parameter vector into ``(w1, b1, w2, b2)``."""
            return (
                theta[:n_w1].reshape(k, h),
                theta[n_w1 : n_w1 + n_b1],
                theta[n_w1 + n_b1 : n_w1 + n_b1 + n_w2],
                float(theta[-1]),
            )

        def loss_and_grad(
            theta: npt.NDArray[np.float64],
        ) -> tuple[float, npt.NDArray[np.float64]]:
            """Penalized half mean squared error and its gradient at ``theta``."""
            w1, b1, w2, b2 = unpack(theta)
            z1 = xs @ w1 + b1
            a1 = _FittedMLP._activation(z1, self.activation)
            resid = (a1 @ w2 + b2) - ys
            mse = 0.5 * float(resid @ resid) / n
            penalty = 0.5 * self.alpha * (float(w1.ravel() @ w1.ravel()) + float(w2 @ w2))
            d_pred = resid / n
            g_w2 = a1.T @ d_pred + self.alpha * w2
            g_b2 = float(d_pred.sum())
            d_z1 = np.outer(d_pred, w2) * _FittedMLP._activation_grad(z1, self.activation)
            g_w1 = xs.T @ d_z1 + self.alpha * w1
            g_b1 = d_z1.sum(axis=0)
            return mse + penalty, np.concatenate([g_w1.ravel(), g_b1, g_w2, [g_b2]])

        rng = np.random.default_rng(self.seed)
        best_theta: npt.NDArray[np.float64] | None = None
        best_loss = np.inf
        w1_scale = np.sqrt(1.0 / k)
        for _ in range(self.n_restarts):
            theta0 = np.concatenate(
                [
                    rng.normal(0.0, w1_scale, size=n_w1),
                    np.zeros(n_b1),
                    rng.normal(0.0, 0.5, size=n_w2),
                    [0.0],
                ]
            )
            result = minimize(
                loss_and_grad,
                theta0,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": self.max_iter},
            )
            if float(result.fun) < best_loss:
                best_loss = float(result.fun)
                best_theta = np.asarray(result.x, dtype=np.float64)
        if best_theta is None:
            raise NumericalError("MLP training produced no finite solution.")

        w1, b1, w2, b2 = unpack(best_theta)
        return _FittedMLP(
            w1=w1.copy(),
            b1=b1.copy(),
            w2=w2.copy(),
            b2=b2,
            x_mean=x_mean,
            x_scale=x_scale,
            y_mean=y_mean,
            y_scale=y_scale,
            activation=self.activation,
            _n_parameters=size,
        )
