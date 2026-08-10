from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class MeanFunctionEngine(Protocol):
    """The plug-in training backend for a nonlinear mean function.

    Any object exposing ``fit(features, target) -> MeanPredictor`` satisfies
    the contract, so a torch / jax / sklearn learner drops in by adapting to
    this one method.
    """

    def fit(
        self, features: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
    ) -> MeanPredictor:
        """Fit the learner and return a :class:`MeanPredictor`."""
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
    """

    hidden_units: int = 8
    activation: str = "tanh"
    alpha: float = 1e-4
    max_iter: int = _DEFAULT_MAX_ITER
    n_restarts: int = 3
    seed: int | None = 0

    def __post_init__(self) -> None:
        """Validate the engine configuration."""
        if self.hidden_units < 1:
            raise SpecificationError(f"hidden_units must be >= 1; got {self.hidden_units}.")
        if self.n_restarts < 1:
            raise SpecificationError(f"n_restarts must be >= 1; got {self.n_restarts}.")
        if self.activation not in ("tanh", "relu"):
            raise SpecificationError(
                f"activation must be 'tanh' or 'relu'; got {self.activation!r}."
            )

    @staticmethod
    def _activation(z: npt.NDArray[np.float64], kind: str) -> npt.NDArray[np.float64]:
        """Hidden-layer activation."""
        if kind == "tanh":
            return np.tanh(z)
        return np.maximum(z, 0.0)

    @staticmethod
    def _activation_grad(z: npt.NDArray[np.float64], kind: str) -> npt.NDArray[np.float64]:
        """Derivative of :func:`_activation`."""
        if kind == "tanh":
            return 1.0 - np.tanh(z) ** 2
        return (z > 0.0).astype(np.float64)

    def fit(self, features: npt.NDArray[np.float64], target: npt.NDArray[np.float64]) -> _FittedMLP:
        """Train the MLP and return a fitted predictor.

        Args:
            features: Design matrix of shape ``(n, k)``.
            target: Response of shape ``(n,)``.

        Returns:
            The fitted :class:`_FittedMLP`.

        Raises:
            DimensionError: If the shapes are not conformable.
            NumericalError: If inputs are non-finite, or no restart converges.
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
            return (
                theta[:n_w1].reshape(k, h),
                theta[n_w1 : n_w1 + n_b1],
                theta[n_w1 + n_b1 : n_w1 + n_b1 + n_w2],
                float(theta[-1]),
            )

        def loss_and_grad(
            theta: npt.NDArray[np.float64],
        ) -> tuple[float, npt.NDArray[np.float64]]:
            w1, b1, w2, b2 = unpack(theta)
            z1 = xs @ w1 + b1
            a1 = self._activation(z1, self.activation)
            resid = (a1 @ w2 + b2) - ys
            mse = 0.5 * float(resid @ resid) / n
            penalty = 0.5 * self.alpha * (float(w1.ravel() @ w1.ravel()) + float(w2 @ w2))
            d_pred = resid / n
            g_w2 = a1.T @ d_pred + self.alpha * w2
            g_b2 = float(d_pred.sum())
            d_z1 = np.outer(d_pred, w2) * self._activation_grad(z1, self.activation)
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
