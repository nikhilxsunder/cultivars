# filepath: /src/cultivars/_internals/_mean_function.py
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

"""Nonlinear conditional-mean functions with a plug-in training backend.

AR-NN and TAR-NN replace the linear conditional mean with a learned function
of lagged values. The learner is not fixed: any object satisfying
:class:`MeanFunctionEngine` -- one method, ``fit(features, target)`` returning
a :class:`MeanPredictor` -- drops in, so a torch, jax, or scikit-learn model
is adapted by writing a wrapper rather than by modifying this package.

:class:`NumpyMLPEngine` is the shipped reference implementation and is
deliberately unprivileged: AR-NN and TAR-NN never reference it by type, only
through the protocol. Both protocols are ``runtime_checkable``, so a
malformed engine is rejected at construction rather than mid-fit.

TAR-NN trains one predictor per regime on the same lagged design, which is
why it consumes roughly twice the parameters of AR-NN at the same order --
reflected in ``n_params`` and therefore in the information criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from .._core._defaults import _DEFAULT_MAX_ITER, _LOG_2PI
from .._core._design import lag_matrix
from .._core._validators import validate_aligned, validate_order
from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._models import _UnivariateModel


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


def _activation(z: npt.NDArray[np.float64], kind: str) -> npt.NDArray[np.float64]:
    """Hidden-layer activation."""
    if kind == "tanh":
        return np.tanh(z)
    return np.maximum(z, 0.0)


def _activation_grad(z: npt.NDArray[np.float64], kind: str) -> npt.NDArray[np.float64]:
    """Derivative of :func:`_activation`."""
    if kind == "tanh":
        return 1.0 - np.tanh(z) ** 2
    return (z > 0.0).astype(np.float64)


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
            a1 = _activation(z1, self.activation)
            resid = (a1 @ w2 + b2) - ys
            mse = 0.5 * float(resid @ resid) / n
            penalty = 0.5 * self.alpha * (float(w1.ravel() @ w1.ravel()) + float(w2 @ w2))
            d_pred = resid / n
            g_w2 = a1.T @ d_pred + self.alpha * w2
            g_b2 = float(d_pred.sum())
            d_z1 = np.outer(d_pred, w2) * _activation_grad(z1, self.activation)
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


def _gaussian_llf(ssr: float, nobs: int) -> tuple[float, float]:
    """Concentrated Gaussian variance and log-likelihood from an SSR."""
    sigma2 = ssr / nobs
    return sigma2, -0.5 * nobs * (_LOG_2PI + np.log(sigma2) + 1.0)


@dataclass(frozen=True, slots=True)
class _ARNNFit:
    """Raw outputs of an autoregressive neural mean-function fit."""

    predictor: MeanPredictor
    sigma2: float
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


@dataclass(frozen=True, slots=True)
class _TARNNFit:
    """Raw outputs of a threshold neural mean-function fit."""

    delay: int
    threshold: float
    lower_predictor: MeanPredictor
    upper_predictor: MeanPredictor
    threshold_variable: npt.NDArray[np.float64] | None
    self_exciting: bool
    sigma2: float
    ssr: float
    n_lower: int
    n_upper: int
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int


def _fit_arnn(y: npt.NDArray[np.float64], order: int, engine: MeanFunctionEngine) -> _ARNNFit:
    """Fit a neural autoregression of the given order.

    The likelihood is Gaussian with the variance concentrated out, so
    ``n_params`` counts the learner's parameters plus that variance.

    Args:
        y: The series.
        order: Autoregressive order.
        engine: The training backend.

    Returns:
        The packed :class:`_ARNNFit`.
    """
    target = y[order:]
    features = lag_matrix(y, order)
    predictor = engine.fit(features, target)
    fitted = predictor.predict(features)
    resid = target - fitted
    sigma2, llf = _gaussian_llf(float(resid @ resid), target.shape[0])
    return _ARNNFit(
        predictor=predictor,
        sigma2=sigma2,
        resid=resid,
        fittedvalues=fitted,
        llf=float(llf),
        nobs=target.shape[0],
        n_params=predictor.n_parameters + 1,
    )


def _fit_tarnn(
    y: npt.NDArray[np.float64],
    order: int,
    engine: MeanFunctionEngine,
    threshold_variable: npt.NDArray[np.float64] | None,
    delay: int,
    threshold: float | None,
    trim: float,
) -> _TARNNFit:
    """Fit a two-regime neural threshold autoregression.

    The threshold defaults to the median of the transition variable rather
    than being searched: with a nonlinear learner per regime, a grid search
    would retrain the network at every candidate split.

    Args:
        y: The series.
        order: Autoregressive order per regime.
        engine: The training backend, used once per regime.
        threshold_variable: External threshold variable, or ``None``.
        delay: Threshold delay.
        threshold: Fixed threshold, or ``None`` for the median.
        trim: Minimum regime share of the effective sample.

    Returns:
        The packed :class:`_TARNNFit`.

    Raises:
        NumericalError: If the split leaves a regime with too few observations.
    """
    n = y.shape[0]
    start = max(order, delay)
    target = y[start:]
    n_eff = target.shape[0]
    features = lag_matrix(y, order, start=start)
    base = threshold_variable if threshold_variable is not None else y
    z = base[start - delay : n - delay]
    r = float(np.median(z)) if threshold is None else float(threshold)
    lower = z <= r
    n_lo = int(lower.sum())
    n_hi = n_eff - n_lo
    if min(n_lo, n_hi) < max(2, int(trim * n_eff)):
        raise NumericalError(
            f"threshold {r} leaves a regime with too few observations ({n_lo} lower, {n_hi} upper)."
        )
    lower_predictor = engine.fit(features[lower], target[lower])
    upper_predictor = engine.fit(features[~lower], target[~lower])
    fitted = np.empty(n_eff, dtype=np.float64)
    fitted[lower] = lower_predictor.predict(features[lower])
    fitted[~lower] = upper_predictor.predict(features[~lower])
    resid = target - fitted
    ssr = float(resid @ resid)
    sigma2, llf = _gaussian_llf(ssr, n_eff)
    return _TARNNFit(
        delay=delay,
        threshold=r,
        lower_predictor=lower_predictor,
        upper_predictor=upper_predictor,
        threshold_variable=threshold_variable,
        self_exciting=threshold_variable is None,
        sigma2=sigma2,
        ssr=ssr,
        n_lower=n_lo,
        n_upper=n_hi,
        resid=resid,
        fittedvalues=fitted,
        llf=float(llf),
        nobs=n_eff,
        n_params=lower_predictor.n_parameters + upper_predictor.n_parameters + 2,
    )


class _MeanFunctionModel[R](_UnivariateModel[R]):
    """Specification surface for neural mean-function models.

    Args:
        endog: The series.
        order: Autoregressive order.
        engine: Training backend; defaults to :class:`NumpyMLPEngine`.

    Raises:
        SpecificationError: If ``engine`` does not satisfy
            :class:`MeanFunctionEngine`, or the order is invalid.
    """

    __slots__ = (
        "_engine",
        "_order",
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        engine: MeanFunctionEngine | None = None,
    ) -> None:
        """Validate the specification, the engine, and the data."""
        super().__init__(endog)
        self._order = validate_order(order, "order", minimum=1)
        self._engine: MeanFunctionEngine = engine if engine is not None else NumpyMLPEngine()
        if not isinstance(self._engine, MeanFunctionEngine):
            raise SpecificationError(
                "engine must implement fit(features, target) -> MeanPredictor."
            )
        self._ensure_length(self._order + 3, f"AR-NN({self._order})")

    @property
    def order(self) -> int:
        """The autoregressive order."""
        return self._order

    @property
    def engine(self) -> MeanFunctionEngine:
        """The training backend."""
        return self._engine


class _ThresholdMeanFunctionModel[R](_MeanFunctionModel[R]):
    """Specification surface for two-regime neural threshold autoregressions.

    Args:
        endog: The series.
        order: Autoregressive order per regime.
        engine: Training backend, used once per regime.
        threshold_variable: External threshold variable, or ``None``.
        delay: Threshold delay.
        threshold: Fixed threshold, or ``None`` for the median.
        trim: Minimum regime share of the effective sample.
    """

    __slots__ = (
        "_delay",
        "_threshold",
        "_threshold_variable",
        "_trim",
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        engine: MeanFunctionEngine | None = None,
        threshold_variable: npt.ArrayLike | None = None,
        delay: int = 1,
        threshold: float | None = None,
        trim: float = 0.15,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, order=order, engine=engine)
        self._delay = validate_order(delay, "delay", minimum=1)
        self._threshold = None if threshold is None else float(threshold)
        self._trim = float(trim)
        self._threshold_variable = (
            None
            if threshold_variable is None
            else validate_aligned(threshold_variable, self.endog.shape[0], "threshold_variable")
        )
        self._ensure_length(2 * (self._order + 3) + self._delay, f"TAR-NN({self._order})")

    @property
    def delay(self) -> int:
        """The threshold delay."""
        return self._delay

    @property
    def self_exciting(self) -> bool:
        """Whether the threshold variable is a lag of the series itself."""
        return self._threshold_variable is None

    def _fit_family(self) -> _TARNNFit:
        """Run the two-regime neural engine for this specification."""
        return _fit_tarnn(
            self.endog,
            self._order,
            self._engine,
            self._threshold_variable,
            self._delay,
            self._threshold,
            self._trim,
        )
