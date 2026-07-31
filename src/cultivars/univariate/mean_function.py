# filepath: /src/cultivars/univariate/mean_function.py
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
"""Nonlinear autoregressive mean functions: AR-NN and TAR-NN.

Two neural conditional-mean models that reuse the lag/threshold structure the
linear univariate family already establishes, but replace the linear map from
lags to the conditional mean with a learned nonlinear one:

- :class:`ARNN` — a neural-network autoregression, ``y_t = g(y_{t-1}, ...,
  y_{t-p}) + eps_t``. The classic single-hidden-layer feedforward NNAR (Hornik;
  Terasvirta et al. 2005) when the default engine is used, but ``g`` is whatever
  the supplied engine learns.
- :class:`TARNN` — a threshold neural-network autoregression: a
  :mod:`cultivars.univariate.threshold`-style hard regime split (SETAR/TAR
  indicator on a delayed threshold variable) with an *independent* nonlinear
  mean per regime, ``y_t = g_{low}(...)`` for ``z_{t-d} <= r`` and
  ``g_{high}(...)`` otherwise.

The learner is a plug-in
--------------------------------------------------------------------------
The network itself is **not** hard-coded. AR-NN and TAR-NN depend only on the
:class:`MeanFunctionEngine` Protocol — a structural interface with a single
``fit(features, target) -> MeanPredictor`` method — and the fitted object they
consume is any :class:`MeanPredictor` (``predict`` + ``n_parameters``). This is
a deliberate seam: cultivars specifies *the contract a learner must satisfy*,
not a particular deep-learning framework, so the core stays free of a torch /
jax / sklearn dependency. The engine is an interface, not a model: it is
intentionally outside the model catalogue in ``.docs/models.md``, which tracks
estimable models rather than the plumbing they compose over.

:class:`NumpyMLPEngine` is a small, dependency-free reference implementation of
that Protocol (a single-hidden-layer perceptron trained by L-BFGS with analytic
backprop) so the models are usable and testable out of the box. Swapping in a
GPU-trained network is a matter of writing a ``fit`` method that returns
something with ``predict`` and ``n_parameters`` — no change to AR-NN / TAR-NN.

References:
    Terasvirta, T., van Dijk, D., & Medeiros, M. (2005). Linear models, smooth
    transition autoregressions, and neural networks for forecasting
    macroeconomic time series. *International Journal of Forecasting*, 21(4).
    Kuan, C.-M. & White, H. (1994). Artificial neural networks: an econometric
    perspective. *Econometric Reviews*, 13(1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._base import InformationCriteria, information_criteria

# --------------------------------------------------------------------------
# Plug-in engine interface (Protocol only — no model lives here)
# --------------------------------------------------------------------------

@runtime_checkable
class MeanPredictor(Protocol):
    """A fitted, callable conditional-mean map produced by an engine.

    The contract is intentionally minimal: predict a conditional mean for a
    design matrix of lagged features, and report the number of free parameters
    so downstream models can form information criteria.
    """

    @property
    def n_parameters(self) -> int:
        """Number of free parameters (used for AIC/BIC in the consuming model)."""
        ...

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return conditional means for the ``(n, k)`` feature matrix, shape ``(n,)``."""
        ...


@runtime_checkable
class MeanFunctionEngine(Protocol):
    """The plug-in training backend for a nonlinear mean function.

    Any object exposing ``fit(features, target) -> MeanPredictor`` satisfies the
    contract, so a torch / jax / sklearn learner drops in by adapting to this one
    method. cultivars ships :class:`NumpyMLPEngine` as a reference implementation;
    it is not privileged — AR-NN / TAR-NN never reference it by type.
    """

    def fit(
        self,
        features: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
    ) -> MeanPredictor:
        """Fit the learner to ``(features, target)`` and return a :class:`MeanPredictor`."""
        ...


# --------------------------------------------------------------------------
# Reference engine: a single-hidden-layer MLP (numpy + scipy only)
# --------------------------------------------------------------------------

def _activation(z: npt.NDArray[np.float64], kind: Activation) -> npt.NDArray[np.float64]:
    if kind == "tanh":
        return np.tanh(z)
    return np.maximum(z, 0.0)


def _activation_grad(z: npt.NDArray[np.float64], kind: Activation) -> npt.NDArray[np.float64]:
    if kind == "tanh":
        return 1.0 - np.tanh(z) ** 2
    return (z > 0.0).astype(np.float64)


@dataclass(frozen=True)
class _FittedMLP:
    """Fitted single-hidden-layer perceptron; a concrete :class:`MeanPredictor`."""

    w1: npt.NDArray[np.float64]
    b1: npt.NDArray[np.float64]
    w2: npt.NDArray[np.float64]
    b2: float
    x_mean: npt.NDArray[np.float64]
    x_scale: npt.NDArray[np.float64]
    y_mean: float
    y_scale: float
    activation: Activation
    _n_parameters: int = field(repr=False, default=0)

    @property
    def n_parameters(self) -> int:
        """Total weight and bias count."""
        return self._n_parameters

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Conditional means for ``features`` of shape ``(n, k)``."""
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.w1.shape[0]:
            raise DimensionError(
                f"features must be (n, {self.w1.shape[0]}); got shape {x.shape}."
            )
        xs = (x - self.x_mean) / self.x_scale
        hidden = _activation(xs @ self.w1 + self.b1, self.activation)
        out = hidden @ self.w2 + self.b2
        return np.asarray(out * self.y_scale + self.y_mean, dtype=np.float64)


@dataclass(frozen=True)
class NumpyMLPEngine:
    """Reference :class:`MeanFunctionEngine`: an L-BFGS-trained MLP (no extra deps).

    A single hidden layer with ``hidden_units`` units, ``tanh`` or ``relu``
    activation, a linear output, and an L2 weight penalty. Inputs and target are
    standardized internally for conditioning. Because the objective is
    non-convex, training restarts from several random initializations and keeps
    the best.

    Args:
        hidden_units: Number of hidden units.
        activation: ``"tanh"`` (default) or ``"relu"``.
        alpha: L2 penalty on the weights (not the biases).
        max_iter: Maximum L-BFGS iterations per restart.
        n_restarts: Number of random initializations; the best loss wins.
        seed: Seed for the initialization RNG (reproducibility).

    Raises:
        SpecificationError: If ``hidden_units < 1`` or ``n_restarts < 1``.
    """

    hidden_units: int = 8
    activation: Activation = "tanh"
    alpha: float = 1e-4
    max_iter: int = 500
    n_restarts: int = 3
    seed: int | None = 0

    def __post_init__(self) -> None:
        if self.hidden_units < 1:
            raise SpecificationError(f"hidden_units must be >= 1; got {self.hidden_units}.")
        if self.n_restarts < 1:
            raise SpecificationError(f"n_restarts must be >= 1; got {self.n_restarts}.")
        if self.activation not in ("tanh", "relu"):
            raise SpecificationError(
                f"activation must be 'tanh' or 'relu'; got {self.activation!r}."
            )

    def fit(
        self,
        features: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
    ) -> _FittedMLP:
        """Train the MLP and return a fitted :class:`MeanPredictor`."""
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
        x_scale = np.where(x.std(axis=0) > 0.0, x.std(axis=0), 1.0)
        y_mean = float(y.mean())
        y_scale = float(y.std()) if y.std() > 0.0 else 1.0
        xs = (x - x_mean) / x_scale
        ys = (y - y_mean) / y_scale

        n_w1, n_b1, n_w2 = k * h, h, h
        size = n_w1 + n_b1 + n_w2 + 1

        def unpack(theta: npt.NDArray[np.float64]) -> tuple[
            npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float
        ]:
            w1 = theta[:n_w1].reshape(k, h)
            b1 = theta[n_w1 : n_w1 + n_b1]
            w2 = theta[n_w1 + n_b1 : n_w1 + n_b1 + n_w2]
            b2 = float(theta[-1])
            return w1, b1, w2, b2

        def loss_and_grad(theta: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64]]:
            w1, b1, w2, b2 = unpack(theta)
            z1 = xs @ w1 + b1                       # (n, h)
            a1 = _activation(z1, self.activation)   # (n, h)
            pred = a1 @ w2 + b2                      # (n,)
            resid = pred - ys
            mse = 0.5 * float(resid @ resid) / n
            penalty = 0.5 * self.alpha * (float(w1.ravel() @ w1.ravel()) + float(w2 @ w2))
            loss = mse + penalty

            d_pred = resid / n                       # (n,)
            g_w2 = a1.T @ d_pred + self.alpha * w2   # (h,)
            g_b2 = float(d_pred.sum())
            d_a1 = np.outer(d_pred, w2)              # (n, h)
            d_z1 = d_a1 * _activation_grad(z1, self.activation)
            g_w1 = xs.T @ d_z1 + self.alpha * w1     # (k, h)
            g_b1 = d_z1.sum(axis=0)                  # (h,)
            grad = np.concatenate([g_w1.ravel(), g_b1, g_w2, [g_b2]])
            return loss, grad

        rng = np.random.default_rng(self.seed)
        best_theta: npt.NDArray[np.float64] | None = None
        best_loss = np.inf
        w1_scale = np.sqrt(1.0 / k)
        for _ in range(self.n_restarts):
            theta0 = np.concatenate([
                rng.normal(0.0, w1_scale, size=n_w1),
                np.zeros(n_b1),
                rng.normal(0.0, 0.5, size=n_w2),
                [0.0],
            ])
            result = minimize(loss_and_grad, theta0, jac=True, method="L-BFGS-B",
                              options={"maxiter": self.max_iter})
            if float(result.fun) < best_loss:
                best_loss = float(result.fun)
                best_theta = np.asarray(result.x, dtype=np.float64)

        if best_theta is None:
            raise NumericalError("MLP training produced no finite solution.")

        w1, b1, w2, b2 = unpack(best_theta)
        return _FittedMLP(
            w1=w1.copy(), b1=b1.copy(), w2=w2.copy(), b2=b2,
            x_mean=x_mean, x_scale=x_scale, y_mean=y_mean, y_scale=y_scale,
            activation=self.activation, _n_parameters=size,
        )


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _validate_endog(endog: npt.ArrayLike) -> npt.NDArray[np.float64]:
    y = np.asarray(endog, dtype=np.float64)
    if y.ndim != 1:
        raise DimensionError(f"endog must be one-dimensional; got shape {y.shape}.")
    if not np.all(np.isfinite(y)):
        raise NumericalError("endog contains non-finite values.")
    return y


def _lagged_design(y: npt.NDArray[np.float64], order: int, start: int) -> npt.NDArray[np.float64]:
    """Rows ``[y_{t-1}, ..., y_{t-p}]`` for ``t = start .. n-1``, shape ``(n-start, p)``."""
    n = y.shape[0]
    return np.column_stack([y[start - i : n - i] for i in range(1, order + 1)])


def _gaussian_llf(ssr: float, nobs: int) -> tuple[float, float]:
    """Concentrated Gaussian (sigma2, loglik) from a sum of squared residuals."""
    sigma2 = ssr / nobs
    llf = -0.5 * nobs * (_LOG_2PI + np.log(sigma2) + 1.0)
    return sigma2, llf


# --------------------------------------------------------------------------
# AR-NN
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ARNNResult:
    """Fitted neural-network autoregression.

    Attributes:
        order: The autoregressive order ``p``.
        sigma2: Residual variance.
        llf: Gaussian conditional log-likelihood.
        nobs: Effective observations (``len(endog) - order``).
        resid: One-step residuals.
        fittedvalues: One-step fitted conditional means.
        endog: The original series (kept for forecasting).
        schema_version: Serialization schema version.
    """

    order: int
    sigma2: float
    llf: float
    nobs: int
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    endog: npt.NDArray[np.float64] = field(repr=False)
    predictor: MeanPredictor = field(repr=False, default=None)  # type: ignore[assignment]
    _n_params: int = field(repr=False, default=0)
    schema_version: int = _SCHEMA_VERSION

    @property
    def information_criteria(self) -> InformationCriteria:
        """AIC/BIC/HQIC (parameter count taken from the fitted engine)."""
        return information_criteria(self.llf, self.nobs, self._n_params)

    @property
    def aic(self) -> float:
        return self.information_criteria.aic

    @property
    def bic(self) -> float:
        return self.information_criteria.bic

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Conditional mean for lagged-feature rows ``[y_{t-1}, ..., y_{t-p}]``."""
        return self.predictor.predict(np.asarray(features, dtype=np.float64))

    def forecast(self, h: int) -> npt.NDArray[np.float64]:
        """Return ``h``-step-ahead point forecasts by iterating the network.

        Args:
            h: Forecast horizon; a positive integer.

        Returns:
            Point forecasts of shape ``(h,)``.

        Raises:
            DimensionError: If ``h`` is not positive.
        """
        if h < 1:
            raise DimensionError(f"forecast horizon h must be >= 1; got {h}.")
        p = self.order
        history = list(self.endog)
        out = np.empty(h, dtype=np.float64)
        for k in range(h):
            window = np.array([[history[-i] for i in range(1, p + 1)]], dtype=np.float64)
            value = float(self.predictor.predict(window)[0])
            out[k] = value
            history.append(value)
        return out


class ARNN:
    """Neural-network autoregression AR-NN(p).

    Models ``y_t = g(y_{t-1}, ..., y_{t-p}) + eps_t`` with ``g`` learned by the
    supplied :class:`MeanFunctionEngine`.

    Args:
        endog: Univariate series (1-D array-like).
        order: Autoregressive order ``p`` (a positive integer).
        engine: The training backend. Defaults to :class:`NumpyMLPEngine`; pass
            any object satisfying :class:`MeanFunctionEngine` to swap learners.

    Raises:
        SpecificationError: If ``order`` is not a positive integer or the series
            is too short.
        DimensionError: If ``endog`` is not one-dimensional.
        NumericalError: If ``endog`` contains non-finite values.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(600)
        >>> for t in range(1, 600):
        ...     y[t] = 0.6 * np.tanh(2.0 * y[t - 1]) + 0.1 * rng.standard_normal()
        >>> res = ARNN(y, order=1).fit()
        >>> res.forecast(3).shape
        (3,)
        >>> bool(res.sigma2 < np.var(y))          # explains variation
        True
    """

    __slots__ = ("_y", "_order", "_engine")

    def __init__(
        self,
        endog: npt.ArrayLike,
        order: int,
        engine: MeanFunctionEngine | None = None,
    ) -> None:
        """Initialize the AR-NN specification."""
        self._y = _validate_endog(endog)
        if not isinstance(order, (int, np.integer)) or order < 1:
            raise SpecificationError(f"order must be a positive integer; got {order!r}.")
        if self._y.shape[0] <= order + 1:
            raise SpecificationError(
                f"series of length {self._y.shape[0]} is too short for AR-NN({order})."
            )
        self._order = int(order)
        self._engine: MeanFunctionEngine = engine if engine is not None else NumpyMLPEngine()

    @property
    def order(self) -> int:
        """The autoregressive order ``p``."""
        return self._order

    def fit(self) -> ARNNResult:
        """Fit the network to the lagged design and return an :class:`ARNNResult`."""
        y, p = self._y, self._order
        target = y[p:]
        design = _lagged_design(y, p, p)
        predictor = self._engine.fit(design, target)
        fitted = np.asarray(predictor.predict(design), dtype=np.float64)
        resid = target - fitted
        nobs = target.shape[0]
        sigma2, llf = _gaussian_llf(float(resid @ resid), nobs)
        return ARNNResult(
            order=p,
            sigma2=sigma2,
            llf=llf,
            nobs=nobs,
            resid=resid,
            fittedvalues=fitted,
            endog=y,
            predictor=predictor,
            _n_params=int(predictor.n_parameters),
        )


# --------------------------------------------------------------------------
# TAR-NN
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TARNNResult:
    """Fitted threshold neural-network autoregression (2 regimes).

    Attributes:
        order: AR order per regime.
        delay: Threshold delay ``d`` (regime set by the threshold variable at ``t-d``).
        threshold: The regime-splitting threshold ``r``.
        sigma2: Residual variance.
        ssr: Total sum of squared residuals.
        n_lower: Observations in the lower regime (``z <= r``).
        n_upper: Observations in the upper regime.
        llf: Gaussian conditional log-likelihood.
        nobs: Effective observations used.
        resid: One-step residuals.
        fittedvalues: One-step fitted conditional means.
        self_exciting: Whether the threshold variable is a lag of the series.
        endog: The original series (kept for forecasting).
        schema_version: Serialization schema version.
    """

    order: int
    delay: int
    threshold: float
    sigma2: float
    ssr: float
    n_lower: int
    n_upper: int
    llf: float
    nobs: int
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    self_exciting: bool
    endog: npt.NDArray[np.float64] = field(repr=False)
    lower_predictor: MeanPredictor = field(repr=False, default=None)  # type: ignore[assignment]
    upper_predictor: MeanPredictor = field(repr=False, default=None)  # type: ignore[assignment]
    threshold_variable: npt.NDArray[np.float64] | None = field(repr=False, default=None)
    _n_params: int = field(repr=False, default=0)
    schema_version: int = _SCHEMA_VERSION

    @property
    def information_criteria(self) -> InformationCriteria:
        """AIC/BIC/HQIC (both regime networks plus the threshold parameter)."""
        return information_criteria(self.llf, self.nobs, self._n_params)

    @property
    def aic(self) -> float:
        return self.information_criteria.aic

    @property
    def bic(self) -> float:
        return self.information_criteria.bic

    def forecast(self, h: int) -> npt.NDArray[np.float64]:
        """Return ``h``-step-ahead point forecasts by iterating the regime networks.

        Only supported for the self-exciting case (threshold variable is a lag of
        the series), where future regimes are determined by forecasted values.

        Args:
            h: Forecast horizon; a positive integer.

        Returns:
            Point forecasts of shape ``(h,)``.

        Raises:
            DimensionError: If ``h`` is not positive.
            SpecificationError: If the model used an external threshold variable
                (future threshold values are unknown).
        """
        if h < 1:
            raise DimensionError(f"forecast horizon h must be >= 1; got {h}.")
        if not self.self_exciting:
            raise SpecificationError(
                "forecast is only available for the self-exciting TAR-NN; an "
                "external threshold variable has no known future path."
            )
        p, d = self.order, self.delay
        history = list(self.endog)
        out = np.empty(h, dtype=np.float64)
        for k in range(h):
            window = np.array([[history[-i] for i in range(1, p + 1)]], dtype=np.float64)
            z = history[-d]
            predictor = self.lower_predictor if z <= self.threshold else self.upper_predictor
            value = float(predictor.predict(window)[0])
            out[k] = value
            history.append(value)
        return out


class TARNN:
    """Threshold neural-network autoregression TAR-NN (2 regimes).

    A hard regime split on a delayed threshold variable, with an independent
    nonlinear mean per regime. The threshold variable is a lag of the series
    (self-exciting) unless ``threshold_variable`` is supplied. When ``threshold``
    is not given, the split ``(delay, r)`` is located by a fast **linear**
    SETAR/TAR grid search (:mod:`cultivars.univariate.threshold`), after which each
    regime is fit nonlinearly by the engine — locating the threshold with the
    cheap linear proxy avoids retraining networks across a whole grid.

    Args:
        endog: Univariate series (1-D array-like).
        order: AR order per regime (a positive integer).
        engine: Training backend (default :class:`NumpyMLPEngine`); a fresh
            :meth:`MeanFunctionEngine.fit` is called per regime.
        threshold_variable: Optional external threshold variable aligned with
            ``endog``; if omitted the model is self-exciting.
        delay: Threshold delay ``d``. Defaults to 1 (or is searched over
            ``1..order`` by the linear proxy when ``threshold`` is not given).
        threshold: Fixed threshold ``r``; if ``None`` it is estimated.

    Raises:
        SpecificationError: If ``order`` is not positive or the series is too short.
        DimensionError: If shapes are inconsistent.
        NumericalError: If inputs contain non-finite values.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(800)
        >>> for t in range(1, 800):
        ...     if y[t - 1] <= 0.0:
        ...         y[t] = 0.7 * y[t - 1] + 0.1 * rng.standard_normal()
        ...     else:
        ...         y[t] = -0.6 * y[t - 1] + 0.1 * rng.standard_normal()
        >>> res = TARNN(y, order=1).fit()
        >>> res.n_lower > 0 and res.n_upper > 0
        True
    """

    __slots__ = ("_y", "_order", "_engine", "_z", "_delay", "_threshold")

    def __init__(
        self,
        endog: npt.ArrayLike,
        order: int,
        engine: MeanFunctionEngine | None = None,
        *,
        threshold_variable: npt.ArrayLike | None = None,
        delay: int = 1,
        threshold: float | None = None,
    ) -> None:
        """Initialize the TAR-NN specification."""
        self._y = _validate_endog(endog)
        if not isinstance(order, (int, np.integer)) or order < 1:
            raise SpecificationError(f"order must be a positive integer; got {order!r}.")
        if delay < 1:
            raise SpecificationError(f"delay must be >= 1; got {delay}.")
        z: npt.NDArray[np.float64] | None = None
        if threshold_variable is not None:
            z = np.asarray(threshold_variable, dtype=np.float64)
            if z.shape != self._y.shape:
                raise DimensionError(
                    f"threshold_variable must match endog shape {self._y.shape}; got {z.shape}."
                )
            if not np.all(np.isfinite(z)):
                raise NumericalError("threshold_variable contains non-finite values.")
        if self._y.shape[0] <= 4 * (order + delay):
            raise SpecificationError(
                f"series of length {self._y.shape[0]} is too short for TAR-NN({order})."
            )
        self._order = int(order)
        self._engine: MeanFunctionEngine = engine if engine is not None else NumpyMLPEngine()
        self._z = z
        self._delay = int(delay)
        self._threshold = None if threshold is None else float(threshold)

    @property
    def order(self) -> int:
        """The AR order per regime."""
        return self._order

    def _locate_split(self) -> tuple[int, float]:
        """Return ``(delay, threshold)`` — user-provided or via a linear proxy."""
        if self._threshold is not None:
            return self._delay, self._threshold
        # Fast linear SETAR/TAR grid to place the split; networks are not retrained here.
        from .threshold import SETAR, TAR

        if self._z is None:
            linear = SETAR(self._y, self._order, delay=self._delay).fit()
        else:
            linear = TAR(self._y, self._order, self._z, delay=self._delay).fit()
        return linear.delay, linear.threshold

    def fit(self) -> TARNNResult:
        """Fit an independent network per regime and return a :class:`TARNNResult`."""
        y, p = self._y, self._order
        delay, threshold = self._locate_split()
        n = y.shape[0]
        start = max(p, delay)
        target = y[start:]
        design = _lagged_design(y, p, start)
        base = self._z if self._z is not None else y
        z = base[start - delay : n - delay]
        lower = z <= threshold
        n_lo, n_hi = int(lower.sum()), int((~lower).sum())
        if n_lo <= p + 1 or n_hi <= p + 1:
            raise NumericalError(
                f"threshold r={threshold:.4g} leaves a regime too small "
                f"(lower={n_lo}, upper={n_hi}); provide a different threshold."
            )

        lower_predictor = self._engine.fit(design[lower], target[lower])
        upper_predictor = self._engine.fit(design[~lower], target[~lower])
        fitted = np.empty_like(target)
        fitted[lower] = lower_predictor.predict(design[lower])
        fitted[~lower] = upper_predictor.predict(design[~lower])
        resid = target - fitted
        nobs = target.shape[0]
        ssr = float(resid @ resid)
        sigma2, llf = _gaussian_llf(ssr, nobs)
        n_params = (
            int(lower_predictor.n_parameters) + int(upper_predictor.n_parameters) + 1
        )
        return TARNNResult(
            order=p,
            delay=delay,
            threshold=threshold,
            sigma2=sigma2,
            ssr=ssr,
            n_lower=n_lo,
            n_upper=n_hi,
            llf=llf,
            nobs=nobs,
            resid=resid,
            fittedvalues=fitted,
            self_exciting=self._z is None,
            endog=y,
            lower_predictor=lower_predictor,
            upper_predictor=upper_predictor,
            threshold_variable=self._z,
            _n_params=n_params,
        )