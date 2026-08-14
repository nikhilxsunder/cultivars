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

"""Learned conditional means: AR-NN and TAR-NN.

Every other family in this package writes the conditional mean as a formula in
a handful of named coefficients. These two do not. The mean is a function
*learned* from the lagged levels by a plug-in training backend, and what comes
back is a callable, not a parameter vector -- so these results have no
coefficient table at all, and the entire summary is diagnostics.

That absence is the design, not a gap. A single-hidden-layer network's weights
are individually meaningless: permuting the hidden units, or flipping the sign
of a unit's input weights together with its output weight, leaves the fitted
function identical. Printing them would invite interpretation of numbers that
carry none. What the results expose instead is the fitted function itself
through ``predict``, plus the quantities that actually tell you whether the fit
is any good -- fit share, capacity relative to sample size, and the engine that
produced it.

The two members differ in how the sample is partitioned:

1. :class:`ARNN` learns one function of the ``p`` lagged levels over the whole
   sample. The nonlinearity is entirely inside the learner.
2. :class:`TARNN` splits on an observed threshold variable and learns a
   *separate* function per regime, so the nonlinearity is partly explicit -- an
   abrupt change of function at the split -- and partly inside each learner.
   The threshold is fixed, at the median by default, rather than searched:
   with a nonlinear learner per regime, a grid search would retrain the network
   at every candidate split.

The training backend is a plug-in. Anything exposing
``fit(features, target) -> MeanPredictor`` satisfies
:class:`~cultivars._internals._engines.MeanFunctionEngine`, so a torch, jax, or
scikit-learn learner drops in by adapting one method. The package ships one
reference implementation and maintains no others; the result records which
engine trained it, because with a plug-in backend the engine is part of the
specification and a summary that omitted it would not identify the model.

Three things are reported with more care than usual, all consequences of the
learner being nonlinear and regularized.

Information criteria are on genuinely shaky ground here. AIC and BIC penalize a
*count* of free parameters, but a weight-decayed network does not have that
many effective degrees of freedom -- the penalty shrinks the effective
dimension below the nominal weight count, by an amount that depends on the
penalty and the data. The counts are still reported, and they are still useful
for choosing hidden units *within* one engine configuration, but they are not
comparable to a linear model's and should not be read as one.

Likelihood-ratio tests are refused outright. Under the null that the network is
linear, the input-to-hidden weights are entirely unidentified and the
hidden-to-output weights sit on the boundary at zero, so the statistic is
neither chi-squared nor pivotal. This is the same obstacle the threshold family
hits, arriving by a different route.

And the reported log-likelihood is a concentrated Gaussian one evaluated at a
*penalized* fit, so it is a goodness-of-fit summary rather than a maximized
likelihood. :attr:`MeanFunctionResult.r_squared` is the honest headline number.

References:
    Terasvirta, T., Lin, C.-F. & Granger, C. W. J. (1993). Power of the
    neural network linearity test. *Journal of Time Series Analysis*, 14(2).
    Kuan, C.-M. & White, H. (1994). Artificial neural networks: an econometric
    perspective. *Econometric Reviews*, 13(1).
    Trapletti, A., Leisch, F. & Hornik, K. (2000). Stationary and integrated
    autoregressive neural network processes. *Neural Computation*, 12(10).
    Moody, J. (1992). The effective number of parameters: an analysis of
    generalization and regularization in nonlinear learning systems. *NeurIPS*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import SummaryTable, trailing_lag
from .._internals import (
    MeanPredictor,
    _MeanFunctionResult,
    _NeuralAutoRegressionFit,
    _NeuralAutoRegressionModel,
    _NeuralThresholdFit,
    _NeuralThresholdModel,
)
from ..exceptions import DimensionError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ARNNResult(_MeanFunctionResult):
    """A fitted neural autoregression.

    Attributes:
        predictor: The learned conditional-mean map, callable through
            :meth:`predict`.
    """

    predictor: MeanPredictor

    @classmethod
    def _from_fit(
        cls,
        fit: _NeuralAutoRegressionFit,
        model: _NeuralAutoRegressionModel[ARNNResult],
    ) -> ARNNResult:
        """Assemble the public result from a raw fit and its specification."""
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            order=model.order,
            sigma2=fit.sigma2,
            engine=type(model.engine).__name__,
            engine_config=repr(model.engine),
            predictor=fit.predictor,
        )

    @property
    def n_learner_parameters(self) -> int:
        """Weights and biases in the single fitted network."""
        return self.predictor.n_parameters

    def predict(self, features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Conditional means for a matrix of lagged levels.

        Args:
            features: Shape ``(n, order)``, columns ordered
                ``[y_{t-1}, ..., y_{t-order}]`` to match how the learner was
                trained. Passing them in the other order will not raise -- the
                widths match -- it will silently return nonsense, so build the
                matrix with :func:`~cultivars._core.lag_matrix` rather than by
                hand.

        Returns:
            An array of length ``n``.

        Raises:
            DimensionError: If the feature matrix is not two-dimensional or its
                width does not match ``order``.
        """
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.order:
            raise DimensionError(f"features must have shape (n, {self.order}); got {x.shape}.")
        return self.predictor.predict(x)

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        return f"AR-NN({self.order})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        The coefficient table is empty by design; :class:`SummaryTable` omits
        the block entirely when there are no rows, so the summary reads as
        header plus diagnostics rather than as a table with a hole in it.
        """
        return SummaryTable(
            title=f"AR-NN({self.order}) Results",
            metadata=self._summary_metadata(),
            notes=(
                "No coefficient table: the conditional mean is a learned function, "
                "and its weights are unidentified up to permuting and sign-flipping "
                "hidden units. Call predict() to use the fitted function.",
                self._capacity_note(),
                self._criteria_note(),
                self._engine_note(),
            ),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class TARNNResult(_MeanFunctionResult):
    """A fitted two-regime neural threshold autoregression.

    Shares vocabulary with :class:`~cultivars.univariate.threshold.SETARResult`
    -- a delay, a threshold, a lower and an upper regime -- but deliberately
    does not share its base class. That base exposes per-regime companion
    eigenvalues, and a learned regime has no autoregressive polynomial to take
    eigenvalues of; inheriting a stationarity property that could only lie is
    the failure mode this package refuses everywhere else.

    Attributes:
        delay: Delay of the threshold variable.
        threshold: The split point, in the units of the threshold variable.
        threshold_values: The threshold variable, aligned with ``resid``.
        self_exciting: Whether the threshold variable is a lag of the series.
        lower_predictor: Learned mean for observations at or below the split.
        upper_predictor: Learned mean for observations above it.
        n_lower: Observations assigned to the lower regime.
        n_upper: Observations assigned to the upper regime.
    """

    delay: int
    threshold: float
    threshold_values: npt.NDArray[np.float64]
    self_exciting: bool
    lower_predictor: MeanPredictor
    upper_predictor: MeanPredictor
    n_lower: int
    n_upper: int

    @classmethod
    def _from_fit(
        cls, fit: _NeuralThresholdFit, model: _NeuralThresholdModel[TARNNResult]
    ) -> TARNNResult:
        """Assemble the public result from a raw fit and its specification."""
        external = fit.threshold_variable
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            order=model.order,
            sigma2=fit.sigma2,
            engine=type(model.engine).__name__,
            engine_config=repr(model.engine),
            delay=fit.delay,
            threshold=fit.threshold,
            threshold_values=trailing_lag(
                model.endog if external is None else external,
                delay=fit.delay,
                length=fit.nobs,
            ),
            self_exciting=fit.self_exciting,
            lower_predictor=fit.lower_predictor,
            upper_predictor=fit.upper_predictor,
            n_lower=fit.n_lower,
            n_upper=fit.n_upper,
        )

    @property
    def n_learner_parameters(self) -> int:
        """Weights and biases across both regime networks."""
        return self.lower_predictor.n_parameters + self.upper_predictor.n_parameters

    @property
    def regime_weight(self) -> npt.NDArray[np.float64]:
        """Indicator of the upper regime: ``1.0`` above the threshold, else ``0.0``.

        Strict on the upper side, matching the estimator's ``z <= r``
        assignment to the lower regime, so an observation landing exactly on
        the threshold is routed here the same way it was during training.
        """
        return (self.threshold_values > self.threshold).astype(np.float64)

    @property
    def upper_fraction(self) -> float:
        """Share of the effective sample assigned to the upper regime."""
        return self.n_upper / self.nobs

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """Aligned per-observation output, widened by the regime split.

        Uses the two-argument ``super`` deliberately. ``@dataclass(slots=True)``
        builds a *new* class object and rebinds the name to it, so the
        ``__class__`` cell that zero-argument ``super()`` closes over still
        points at the original, pre-slots class -- which no subclass inherits
        from.

        Returns:
            The observed/fitted/residual triple plus the threshold variable and
            the regime indicator.
        """
        base = super(TARNNResult, self)._series()
        base["threshold_variable"] = self.threshold_values
        base["regime_weight"] = self.regime_weight
        return base

    def predict(
        self,
        features: npt.NDArray[np.float64],
        threshold_values: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Conditional means, routing each row to its regime's learner.

        Takes the threshold variable explicitly rather than deriving it from
        ``features``. For a self-exciting model with ``delay <= order`` it
        could be read off a column, but not when the delay exceeds the order
        and not at all when the threshold variable is external -- so requiring
        it keeps one code path that is correct in every configuration.

        Args:
            features: Shape ``(n, order)``, columns ordered
                ``[y_{t-1}, ..., y_{t-order}]``.
            threshold_values: Length ``n``, the threshold variable already
                lagged by ``delay``.

        Returns:
            An array of length ``n``.

        Raises:
            DimensionError: If the feature matrix has the wrong shape, or
                ``threshold_values`` does not match its row count.
        """
        x = np.asarray(features, dtype=np.float64)
        z = np.asarray(threshold_values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.order:
            raise DimensionError(f"features must have shape (n, {self.order}); got {x.shape}.")
        if z.shape != (x.shape[0],):
            raise DimensionError(
                f"threshold_values must have shape ({x.shape[0]},); got {z.shape}."
            )
        lower = z <= self.threshold
        out = np.empty(x.shape[0], dtype=np.float64)
        if lower.any():
            out[lower] = self.lower_predictor.predict(x[lower])
        if (~lower).any():
            out[~lower] = self.upper_predictor.predict(x[~lower])
        return out

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        family = "TAR-NN" if self.self_exciting else "TAR-NN/x"
        return f"{family}({self.order}, d={self.delay})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        notes = [
            "No coefficient table: each regime's conditional mean is a learned "
            "function, and its weights are unidentified up to permuting and "
            "sign-flipping hidden units. Call predict() to use the fitted model.",
            f"Threshold {self.threshold:.4f} at delay {self.delay}: "
            f"{self.n_lower} observations below, {self.n_upper} above "
            f"({100.0 * self.upper_fraction:.1f}% upper).",
            "The threshold is fixed, not searched -- a grid search would retrain "
            "both networks at every candidate split -- so it carries no sampling "
            "uncertainty of the kind a searched threshold would.",
        ]
        if not self.self_exciting:
            notes.append("The threshold variable is external, not a lag of the series.")
        notes.extend((self._capacity_note(), self._criteria_note(), self._engine_note()))
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=self._summary_metadata(),
            notes=tuple(notes),
        )


class ARNN(_NeuralAutoRegressionModel[ARNNResult]):
    """Neural autoregression: one learned function of ``order`` lagged levels.

    Args:
        endog: The series.
        order: Number of lagged levels fed to the learner.
        engine: Training backend. Anything exposing
            ``fit(features, target) -> MeanPredictor`` qualifies; defaults to
            the package's reference L-BFGS-trained perceptron.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(500)
        >>> for t in range(1, 500):
        ...     y[t] = 0.8 * np.tanh(2.0 * y[t - 1]) + 0.3 * rng.standard_normal()
        >>> res = ARNN(y, order=1).fit()
        >>> bool(res.r_squared > 0.3)
        True
    """

    __slots__ = ()

    def fit(self) -> ARNNResult:
        """Train the mean function and concentrate out the variance."""
        return ARNNResult._from_fit(self._fit_family(), self)


class TARNN(_NeuralThresholdModel[TARNNResult]):
    """Neural threshold autoregression: one learned function per regime.

    Args:
        endog: The series.
        order: Lagged levels fed to each regime's learner.
        engine: Training backend, invoked once per regime.
        threshold_variable: External threshold variable aligned with ``endog``;
            ``None`` makes the model self-exciting on ``y_{t-delay}``.
        delay: Delay applied to the threshold variable.
        threshold: Fixed split point; ``None`` uses the median of the
            threshold variable.
        trim: Minimum share of the effective sample each regime must hold, so
            a lopsided split cannot leave one network trained on a handful of
            points.
    """

    __slots__ = ()

    def fit(self) -> TARNNResult:
        """Split on the threshold, train one function per regime."""
        return TARNNResult._from_fit(self._fit_family(), self)
