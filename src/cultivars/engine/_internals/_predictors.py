# filepath: /src/cultivars/engine/_internals/_predictors.py
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
"""Predictors: the fitted conditional-mean maps that training engines hand back.

A nonlinear mean-function model (a neural autoregression, or its threshold
variant) splits into two roles. An *engine* owns training: it takes a design
matrix of lagged values and a target and fits some learner. A *predictor*
is what training leaves behind, a frozen map from features to conditional
means that the model then treats as given. The model evaluates it for the
fitted values, counts its parameters for the information criteria, and
wraps it in a nonlinear state space, where it is called on whole batches of
particle states. This module defines the second role.

:class:`MeanPredictor` is the contract, held to the two members the
consuming models use, so that a learner written against torch, jax, or
scikit-learn can satisfy it without inheriting from anything here. The
contract is structural and shallow: it is a ``runtime_checkable`` protocol,
so ``isinstance`` asks only whether an object has attributes named
``n_parameters`` and ``predict``. Signatures, shapes, and return types are
not checked, and the models that consume a predictor do not run even that
check on what an engine returns.

:class:`_FittedMLP` is the one predictor the package ships, the output of
the reference NumPy engine. It is a single-hidden-layer perceptron with a
linear output, evaluated on standardized inputs and mapped back to the
scale of the target::

    z = (x - x_mean) / x_scale
    m(x) = y_mean + y_scale * (act(z @ w1 + b1) @ w2 + b2)

The standardization constants are those of the training sample and travel
with the record, which is what makes a prediction on new data use exactly
the scaling the weights were trained under. The activation and its
derivative are static methods on the record, not helpers of the engine,
because the engine's analytic gradient must differentiate the same function
the forward pass evaluates.

Layout. Engines, :class:`~cultivars.engine._internals._engines.NumpyMLPEngine`
among them, live in ``_engines`` and return predictors; the fit records in
``_fits`` carry them; the mean-function models in ``_models`` consume them.
``MeanPredictor`` has a public name because it appears in the signatures of
:mod:`cultivars.univariate.mean_function`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ...exceptions import DimensionError

__all__ = [
    "MeanPredictor",
    "_FittedMLP",
]


@runtime_checkable
class MeanPredictor(Protocol):
    """A fitted, callable conditional-mean map produced by an engine.

    The contract is intentionally minimal: predict a conditional mean for a
    design matrix of lagged features, and report the number of free
    parameters so the consuming model can form information criteria.

    Nothing else is assumed of an implementation. It need not be a neural
    network, need not be differentiable, and need not subclass this
    protocol: any object with the two members below is a ``MeanPredictor``.
    It should be immutable once returned, since results hold it and call it
    for as long as they live, and ``predict`` should be a pure function of
    its argument, since the particle filter calls it repeatedly on different
    batches and assumes the same map each time.

    Note:
        ``isinstance(obj, MeanPredictor)`` is a presence check on the two
        attribute names. It passes for an object whose ``predict`` takes the
        wrong arguments or whose ``n_parameters`` is not an integer.

    See Also:
        * :class:`~cultivars.engine._internals._engines.MeanFunctionEngine`
          -- the training contract whose ``fit`` returns one of these.
        * :class:`_FittedMLP` -- the reference implementation.

    Example:
        Any object with the two members qualifies:

        >>> import numpy as np
        >>> class Constant:
        ...     n_parameters = 1
        ...
        ...     def predict(self, features):
        ...         return np.full(len(features), 2.0)
        >>> isinstance(Constant(), MeanPredictor)
        True
        >>> isinstance(object(), MeanPredictor)
        False
    """

    @property
    def n_parameters(self) -> int:
        """Number of free parameters, used for AIC/BIC in the consuming model.

        Counts the learner's own parameters only. The consuming model adds
        the innovation variance itself, so a predictor that included it
        would be charged for it twice.

        Returns:
            The parameter count, a non-negative integer.
        """
        ...

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Conditional means for an ``(n, k)`` feature matrix, shape ``(n,)``.

        Args:
            features: Design matrix, one row per evaluation point and one
                column per feature, in the column order the predictor was
                trained on. Any number of rows must be accepted, a batch of
                particle states included.

        Returns:
            The conditional mean at each row, shape ``(n,)``, on the scale
            of the training target.
        """
        ...


@dataclass(frozen=True)
class _FittedMLP:
    """A fitted single-hidden-layer perceptron; a concrete :class:`MeanPredictor`.

    Carries its own standardization constants so :meth:`predict` reproduces
    training-time scaling exactly on new data.

    With ``k`` features and ``h`` hidden units the map is::

        z = (x - x_mean) / x_scale
        m(x) = y_mean + y_scale * (act(z @ w1 + b1) @ w2 + b2)

    The record is what
    :meth:`~cultivars.engine._internals._engines.NumpyMLPEngine.fit` returns
    and is not meant to be built by hand. Construction validates nothing:
    the shapes are not checked against one another, ``activation`` is not
    checked against the two supported names, and the parameter count is
    taken as given.

    Attributes:
        w1: Input-to-hidden weights, ``(k, h)``.
        b1: Hidden biases, ``(h,)``.
        w2: Hidden-to-output weights, ``(h,)``.
        b2: Output bias.
        x_mean: Training means of the features, ``(k,)``.
        x_scale: Training standard deviations of the features, ``(k,)``.
        y_mean: Training mean of the target.
        y_scale: Training standard deviation of the target.
        activation: Hidden activation, ``"tanh"`` or ``"relu"``.

    Note:
        The dataclass is frozen, but its array fields are ordinary writeable
        arrays. As with every frozen record that holds arrays, ``==`` between
        two instances raises ``ValueError`` and ``hash`` raises ``TypeError``.

    Example:
        A two-feature, two-unit network written out by hand. The first row
        sits at the training means, so only the hidden biases contribute:

        >>> import numpy as np
        >>> mlp = _FittedMLP(
        ...     w1=np.array([[1.0, -1.0], [0.5, 0.5]]),
        ...     b1=np.array([0.0, 0.1]),
        ...     w2=np.array([2.0, -1.0]),
        ...     b2=0.5,
        ...     x_mean=np.array([1.0, 2.0]),
        ...     x_scale=np.array([2.0, 4.0]),
        ...     y_mean=10.0,
        ...     y_scale=3.0,
        ...     activation="tanh",
        ...     _n_parameters=9,
        ... )
        >>> mlp.predict(np.array([[1.0, 2.0], [3.0, 6.0]])).round(4).tolist()
        [11.201, 18.0707]
        >>> round(10.0 + 3.0 * (0.5 - float(np.tanh(0.1))), 4)
        11.201
        >>> mlp.n_parameters, isinstance(mlp, MeanPredictor)
        (9, True)
    """

    w1: npt.NDArray[np.float64]
    """Input-to-hidden weights, ``(k, h)``: one row per feature, one column per hidden unit.

    Applied to the standardized features as ``z @ w1``. Its first dimension
    is the feature width that :meth:`predict` checks new data against.
    """
    b1: npt.NDArray[np.float64]
    """Hidden-layer biases, ``(h,)``."""
    w2: npt.NDArray[np.float64]
    """Hidden-to-output weights, ``(h,)``."""
    b2: float
    """Output bias, on the standardized scale of the target."""
    x_mean: npt.NDArray[np.float64]
    """Column means of the training features, ``(k,)``."""
    x_scale: npt.NDArray[np.float64]
    """Column standard deviations of the training features, ``(k,)``.

    The reference engine stores ``1.0`` for a column with no variation, so
    the division in :meth:`predict` is safe. A hand-built record with a zero
    entry divides by zero.
    """
    y_mean: float
    """Mean of the training target, added back to every prediction."""
    y_scale: float
    """Standard deviation of the training target; ``1.0`` when the target was constant."""
    activation: str
    """Hidden activation, ``"tanh"`` or ``"relu"``.

    Not validated on the record: any value other than ``"tanh"`` is
    evaluated as ReLU.
    """
    _n_parameters: int = field(repr=False, default=0)
    """Parameter count reported by :attr:`n_parameters`; kept out of the repr.

    Supplied by the engine as ``k * h + 2 * h + 1``. It is not derived from
    the arrays, so a record built without it reports zero parameters.
    """

    @property
    def n_parameters(self) -> int:
        """Total weight and bias count.

        For ``k`` features and ``h`` hidden units this is
        ``k * h + h + h + 1``: both weight blocks, the hidden biases, and
        the output bias. The standardization constants are not counted. The
        value is the one stored at construction, not a count of the arrays.

        Returns:
            The stored parameter count; ``0`` if none was supplied.
        """
        return self._n_parameters

    @staticmethod
    def _activation(z: npt.NDArray[np.float64], kind: str) -> npt.NDArray[np.float64]:
        """Hidden-layer activation.

        Shared by :meth:`predict` and by the reference engine's objective,
        so the function trained is the function evaluated.

        Args:
            z: Pre-activations, any shape.
            kind: ``"tanh"`` for the hyperbolic tangent. Every other value,
                ``"relu"`` and unrecognized names alike, gives the rectifier
                ``max(z, 0)``.

        Returns:
            The activations, same shape as ``z``.

        Example:
            >>> import numpy as np
            >>> _FittedMLP._activation(np.array([-1.0, 0.0, 2.0]), "relu").tolist()
            [0.0, 0.0, 2.0]
        """
        if kind == "tanh":
            return np.tanh(z)
        return np.maximum(z, 0.0)

    @staticmethod
    def _activation_grad(z: npt.NDArray[np.float64], kind: str) -> npt.NDArray[np.float64]:
        """Derivative of :meth:`_activation`.

        Used by the reference engine's analytic gradient; :meth:`predict`
        does not call it.

        Args:
            z: Pre-activations, any shape.
            kind: ``"tanh"`` gives ``1 - tanh(z) ** 2``. Every other value
                gives the rectifier's derivative, the indicator of
                ``z > 0``, which is ``0`` at the kink.

        Returns:
            The elementwise derivative, same shape as ``z``.

        Example:
            >>> import numpy as np
            >>> _FittedMLP._activation_grad(np.array([-1.0, 0.0, 2.0]), "relu").tolist()
            [0.0, 0.0, 1.0]
        """
        if kind == "tanh":
            return 1.0 - np.tanh(z) ** 2
        return (z > 0.0).astype(np.float64)

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Conditional means for ``features`` of shape ``(n, k)``.

        Standardizes the features with the training constants, runs the
        forward pass, and maps the output back to the scale of the target.
        The input is coerced to a float array, so nested lists and integer
        arrays are accepted. Values are not checked for finiteness: a
        ``nan`` feature gives a ``nan`` prediction for that row.

        Args:
            features: Feature matrix, ``(n, k)`` with ``k`` the training
                width, columns in training order. ``n`` may be zero.

        Returns:
            The conditional means, shape ``(n,)``.

        Raises:
            DimensionError: If the feature width does not match training.
        """
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.w1.shape[0]:
            raise DimensionError(f"features must be (n, {self.w1.shape[0]}); got shape {x.shape}.")
        xs = (x - self.x_mean) / self.x_scale
        hidden = self._activation(xs @ self.w1 + self.b1, self.activation)
        out = hidden @ self.w2 + self.b2
        return np.asarray(out * self.y_scale + self.y_mean, dtype=np.float64)
