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
r"""Learned conditional means: AR-NN and TAR-NN.

Every other family in this package writes the conditional mean as a
formula in a handful of named coefficients. These two do not. The mean
is a function *learned* from the lagged levels by a plug-in training
backend,

.. math::

   y_t = f(y_{t-1}, \ldots, y_{t-p}) + \varepsilon_t,
   \qquad \varepsilon_t \sim N(0, \sigma^2),

with the reference engine a single-hidden-layer perceptron

.. math::

   f(x) = b + \sum_{h=1}^{H} w_h \tanh(a_h' x + c_h),

trained by penalized least squares and the variance concentrated out,
:math:`\hat\sigma^2 = \text{SSR} / n`. What comes back is a callable,
not a parameter vector -- so these results have no coefficient table at
all, and the entire summary is diagnostics.

That absence is the design, not a gap. The network's weights are
individually meaningless: permuting the hidden units, or flipping the
sign of a unit's input weights together with its output weight, leaves
:math:`f` identical. Printing them would invite interpretation of
numbers that carry none. What the results expose instead is the fitted
function itself through ``predict``, plus the quantities that actually
tell you whether the fit is any good -- fit share, capacity relative to
sample size, and the engine that produced it.

The two members differ in how the sample is partitioned.
:class:`ARNN` learns one function of the :math:`p` lagged levels over
the whole sample, so the nonlinearity is entirely inside the learner.
:class:`TARNN` splits on an observed threshold variable
:math:`z_{t-d}` at a fixed :math:`r` and learns a *separate* function
per regime, so the nonlinearity is partly explicit -- an abrupt change
of function at the split -- and partly inside each learner. The
threshold is fixed, at the median by default, rather than searched:
with a nonlinear learner per regime, a grid search would retrain both
networks at every candidate split.

Two commitments shape the surface. First, the training backend is a
plug-in. Anything exposing ``fit(features, target) -> MeanPredictor``
satisfies the ``MeanFunctionEngine`` protocol, so a torch, jax, or
scikit-learn learner drops in by adapting one method. The package ships
one reference implementation and maintains no others; the result
records which engine trained it, because with a plug-in backend the
engine is part of the specification and a summary that omitted it
would not identify the model. Second, three things are reported with
more care than usual, all consequences of the learner being nonlinear
and regularized. Information criteria penalize a *count* of free
parameters, but a weight-decayed network does not have that many
effective degrees of freedom -- the penalty shrinks the effective
dimension below the nominal weight count, by an amount that depends on
the penalty and the data -- so the counts are reported, are useful for
choosing hidden units *within* one engine configuration, and are not
comparable to a linear model's. A likelihood-ratio test against a
linear null is refused: under linearity the input-to-hidden weights are
unidentified and the hidden-to-output weights sit on the boundary at
zero, so the statistic is neither chi-squared nor pivotal, the same
obstacle the threshold family hits by a different route. And the
reported log-likelihood is a concentrated Gaussian one at a *penalized*
fit, a goodness-of-fit summary rather than a maximized likelihood;
``r_squared`` is the headline number.

Layout. :class:`ARNN` and :class:`TARNN` validate ``order`` through
``validate_order`` and the engine on ``_MeanFunctionModel`` in
``_internals``; :class:`TARNN` adds ``delay``, ``threshold``, ``trim``
and an aligned ``threshold_variable`` on ``_NeuralThresholdModel``.
``_fit_family`` builds the features with ``lag_matrix`` from ``_core``,
calls ``engine.fit`` once (or once per regime, after ``trailing_lag``
produces the split variable), and packs a ``_NeuralAutoRegressionFit``
or ``_NeuralThresholdFit``. ``NumpyMLPEngine`` in
``_internals._engines`` is the reference backend, returning a
``MeanPredictor`` from ``_internals._predictors``. The results extend
``_MeanFunctionResult``, whose ``_capacity_note``, ``_criteria_note``
and ``_engine_note`` are the shared diagnostics, and each emits its
mean as a ``NonlinearSSM`` through
``NonlinearSSM._lag_stack_state_space``. Parametric nonlinear means
with coefficients are :mod:`~cultivars.univariate.threshold` and
:mod:`~cultivars.univariate.smooth_transition`.

References:
    Kuan, C.-M., & White, H. (1994). Artificial neural networks: An
    econometric perspective. *Econometric Reviews*, 13(1), 1-91.

    Terasvirta, T., Lin, C.-F., & Granger, C. W. J. (1993). Power of the
    neural network linearity test. *Journal of Time Series Analysis*,
    14(2), 209-220.

    Trapletti, A., Leisch, F., & Hornik, K. (2000). Stationary and
    integrated autoregressive neural network processes. *Neural
    Computation*, 12(10), 2427-2450.

    Moody, J. (1992). The effective number of parameters: An analysis of
    generalization and regularization in nonlinear learning systems. In
    *Advances in Neural Information Processing Systems 4*, 847-854.

Example:
    Capacity chosen by BIC within one engine, and why the same BIC must
    not be read against the linear fit:

    >>> import numpy as np
    >>> from cultivars.univariate.autoregression import AR
    >>> from cultivars._internals._engines import NumpyMLPEngine
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros(600)
    >>> for t in range(1, 600):
    ...     y[t] = 0.8 * np.tanh(2.0 * y[t - 1]) + 0.3 * rng.standard_normal()
    >>> fits = [ARNN(y, order=1, engine=NumpyMLPEngine(hidden_units=h)).fit() for h in (2, 8, 16)]
    >>> [fit.n_learner_parameters for fit in fits]
    [7, 25, 49]
    >>> fits[0].compare(*fits[1:], criterion="bic").rows[0][2]
    '8'
    >>> linear = AR(y, order=1).fit()
    >>> bool(fits[0].ssr < np.sum(linear.resid**2))
    True
    >>> bool(fits[1].information_criteria.bic > linear.information_criteria.bic)
    True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..engine._core import SummaryTable, trailing_lag
from ..engine._internals import (
    MeanPredictor,
    _MeanFunctionResult,
    _NeuralAutoRegressionFit,
    _NeuralAutoRegressionModel,
    _NeuralThresholdFit,
    _NeuralThresholdModel,
)
from ..exceptions import DimensionError, SpecificationError
from ..state_space.nonlinear import NonlinearSSM

__all__ = ["ARNN", "TARNN", "ARNNResult", "TARNNResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ARNNResult(_MeanFunctionResult):
    r"""A fitted neural autoregression.

    The model :math:`y_t = f(y_{t-1}, \ldots, y_{t-p}) + \varepsilon_t`
    with :math:`f` a learned function and
    :math:`\varepsilon_t \sim N(0, \sigma^2)`, the variance concentrated
    out of a penalized least-squares fit. There is no coefficient vector;
    the record carries the fitted function as a callable, the fit
    statistics, and the engine that produced it, since with a plug-in
    backend the engine is part of the specification.

    Attributes:
        predictor: The learned conditional-mean map, callable through
            :meth:`predict`.

    Note:
        The inherited fields are ``endog``, ``fittedvalues``, ``resid``,
        ``llf``, ``nobs``, ``n_params``, ``order``, ``sigma2``, ``engine``
        and ``engine_config``; the inherited properties ``r_squared``,
        ``ssr`` and ``parameters_per_observation`` read them.
        ``fittedvalues`` and ``resid`` are aligned with the last ``nobs =
        len(endog) - order`` observations. ``llf`` is a concentrated
        Gaussian likelihood at a *penalized* fit, so the information
        criteria compare capacity within one engine configuration and
        not against a linear model; ``r_squared`` is the headline number.

    See Also:
        * :class:`ARNN` -- the specification that produces this.
        * :class:`TARNNResult` -- one learned function per regime.
        * :class:`~cultivars.univariate.autoregression.ARResult` -- the
          linear case, with coefficients and verdicts.
        * :class:`~cultivars.state_space.nonlinear.NonlinearSSM` -- what
          :meth:`nonlinear_state_space` emits.

    Example:
        A one-lag map :math:`0.8 \tanh(2 y_{t-1})` recovered on a grid:

        >>> import numpy as np
        >>> from cultivars._core import lag_matrix
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(500)
        >>> for t in range(1, 500):
        ...     y[t] = 0.8 * np.tanh(2.0 * y[t - 1]) + 0.3 * rng.standard_normal()
        >>> res = ARNN(y, order=1).fit()
        >>> res.nobs, res.n_learner_parameters, res.engine
        (499, 25, 'NumpyMLPEngine')
        >>> bool(np.allclose(res.predict(lag_matrix(y, 1)), res.fittedvalues))
        True
        >>> grid = np.array([[-1.0], [0.0], [1.0]])
        >>> bool(np.max(np.abs(res.predict(grid) - 0.8 * np.tanh(2.0 * grid[:, 0]))) < 0.15)
        True
    """

    predictor: MeanPredictor
    """The trained network; use :meth:`predict` rather than calling it."""

    @classmethod
    def _from_fit(
        cls,
        fit: _NeuralAutoRegressionFit,
        model: _NeuralAutoRegressionModel[ARNNResult],
    ) -> ARNNResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The raw estimator output.
            model: The specification, read for the series, the order and
                the engine.

        Returns:
            The assembled result.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> model = ARNN(rng.standard_normal(200), order=2)
            >>> res = ARNNResult._from_fit(model._fit_family(), model)
            >>> res.order, res.nobs, res.engine
            (2, 198, 'NumpyMLPEngine')
        """
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
        """Weights and biases in the single fitted network.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> ARNN(rng.standard_normal(200), order=3).fit().n_learner_parameters
            41
        """
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

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = ARNN(rng.standard_normal(200), order=2).fit()
            >>> res.predict(np.zeros((4, 2))).shape
            (4,)
            >>> res.predict(np.zeros((4, 3)))
            Traceback (most recent call last):
                ...
            cultivars.exceptions.DimensionError: features must have shape (n, 2); got (4, 3).
        """
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.order:
            raise DimensionError(f"features must have shape (n, {self.order}); got {x.shape}.")
        return self.predictor.predict(x)

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> ARNN(rng.standard_normal(200), order=2).fit()._comparison_label()
            'AR-NN(2)'
        """
        return f"AR-NN({self.order})"

    def nonlinear_state_space(self, measurement_variance: float) -> NonlinearSSM:
        """The learned mean as a noisily observed state space.

        The errors-in-variables reading of the fit: the latent state is
        the ``order``-deep lag stack of the *true* series, the transition
        applies the trained network to the stack, and the observation
        reads the current level through Gaussian measurement error of the
        stated variance -- required and strictly positive, because with
        none the model is observation-driven and filtering it is vacuous.
        The network is smooth, so all three filters are available; the
        unscented filter avoids differencing through the network. The
        initial state is diffuse at the sample mean and variance. One
        caveat is the network's own: outside the range of states seen in
        training, the learned mean is extrapolation, and so is every
        filtered path that wanders there.

        Args:
            measurement_variance: Observation noise variance the emitted
                system is read through, strictly positive.

        Returns:
            The :class:`NonlinearSSM`.

        Raises:
            SpecificationError: If the variance is not strictly positive.

        Example:
            Filtering a noisy observation of the series back toward it:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros(300)
            >>> for t in range(1, 300):
            ...     y[t] = 0.8 * np.tanh(2.0 * y[t - 1]) + 0.3 * rng.standard_normal()
            >>> ssm = ARNN(y, order=1).fit().nonlinear_state_space(0.05)
            >>> ssm.k_states, ssm.k_endog
            (1, 1)
            >>> noisy = y + np.sqrt(0.05) * rng.standard_normal(300)
            >>> filtered = ssm.unscented_filter(noisy[:, None]).filtered_state[:, 0]
            >>> bool(np.mean((filtered - y) ** 2) < np.mean((noisy - y) ** 2))
            True
        """
        predictor = self.predictor

        def mean_map(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.asarray(predictor.predict(states), dtype=np.float64)

        return NonlinearSSM._lag_stack_state_space(
            mean_map,
            order=self.order,
            sigma2=self.sigma2,
            measurement_variance=float(measurement_variance),
            center=float(np.mean(self.endog)),
            spread=float(np.var(self.endog)),
        )

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        The coefficient table is empty by design; :class:`SummaryTable` omits
        the block entirely when there are no rows, so the summary reads as
        header plus diagnostics rather than as a table with a hole in it.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> table = ARNN(rng.standard_normal(200), order=1).fit()._summary_table()
            >>> table.title, len(table.rows), len(table.notes), table.metadata[2]
            ('AR-NN(1) Results', 0, 4, ('Engine', 'NumpyMLPEngine'))
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
    r"""A fitted two-regime neural threshold autoregression.

    The model

    .. math::

        y_t = \begin{cases}
            f_L(y_{t-1}, \ldots, y_{t-p}) + \varepsilon_t, & z_{t-d} \le r, \\
            f_U(y_{t-1}, \ldots, y_{t-p}) + \varepsilon_t, & z_{t-d} > r,
        \end{cases}

    with :math:`z` the series itself (self-exciting) or an external
    variable, :math:`r` a fixed threshold, and :math:`f_L`, :math:`f_U`
    two networks trained separately on their regime's subsample under
    one pooled innovation variance.

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

    Note:
        The inherited fields and properties are those of
        :class:`ARNNResult`. The effective sample drops
        ``max(order, delay)`` leading observations so both the feature
        lags and the threshold lag exist. The threshold is fixed at
        construction (the median of the lagged threshold variable by
        default), not searched, so it carries no sampling uncertainty of
        the kind a searched threshold would, and ``n_params`` counts the
        two networks and the variance only.

    See Also:
        * :class:`TARNN` -- the specification that produces this.
        * :class:`ARNNResult` -- one function over the whole sample.
        * :class:`~cultivars.univariate.threshold.SETARResult` -- the
          linear-per-regime model, with a searched threshold and
          per-regime stability verdicts.

    Example:
        A regime map with slope 0.6 below zero and -0.4 above, recovered
        at a fixed split:

        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> z = np.zeros(600)
        >>> for t in range(1, 600):
        ...     slope = 0.6 if z[t - 1] <= 0 else -0.4
        ...     z[t] = slope * z[t - 1] + 0.3 * rng.standard_normal()
        >>> res = TARNN(z, order=1, threshold=0.0).fit()
        >>> res.nobs, res.n_lower + res.n_upper, res.self_exciting
        (599, 599, True)
        >>> point = np.array([[-1.0], [1.0]])
        >>> means = res.predict(point, point[:, 0])
        >>> bool(abs(means[0] + 0.6) < 0.2), bool(abs(means[1] - -0.4) < 0.2)
        (True, True)
    """

    delay: int
    """Lag applied to the threshold variable before the split."""

    threshold: float
    """The split point, in the units of the threshold variable."""

    threshold_values: npt.NDArray[np.float64]
    """``(nobs,)`` lagged threshold variable, aligned with ``resid``."""

    self_exciting: bool
    """``True`` when the threshold variable is a lag of ``endog``."""

    lower_predictor: MeanPredictor
    """Network trained on the lower-regime subsample."""

    upper_predictor: MeanPredictor
    """Network trained on the upper-regime subsample."""

    n_lower: int
    """Effective-sample observations with ``z <= threshold``."""

    n_upper: int
    """Effective-sample observations with ``z > threshold``."""

    @classmethod
    def _from_fit(
        cls, fit: _NeuralThresholdFit, model: _NeuralThresholdModel[TARNNResult]
    ) -> TARNNResult:
        """Assemble the public result from a raw fit and its specification.

        The threshold variable is re-lagged here from the series or the
        external variable so that the stored copy is aligned with the
        effective sample rather than the full one.

        Args:
            fit: The raw estimator output.
            model: The specification, read for the series, the order and
                the engine.

        Returns:
            The assembled result.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> model = TARNN(rng.standard_normal(300), order=1, delay=2)
            >>> res = TARNNResult._from_fit(model._fit_family(), model)
            >>> res.delay, res.nobs, res.threshold_values.shape
            (2, 298, (298,))
        """
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
        """Weights and biases across both regime networks.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = TARNN(rng.standard_normal(300), order=1).fit()
            >>> res.n_learner_parameters == 2 * res.lower_predictor.n_parameters
            True
        """
        return self.lower_predictor.n_parameters + self.upper_predictor.n_parameters

    @property
    def regime_weight(self) -> npt.NDArray[np.float64]:
        """Indicator of the upper regime: ``1.0`` above the threshold, else ``0.0``.

        Strict on the upper side, matching the estimator's ``z <= r``
        assignment to the lower regime, so an observation landing exactly on
        the threshold is routed here the same way it was during training.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = TARNN(rng.standard_normal(300), order=1).fit()
            >>> int(res.regime_weight.sum()) == res.n_upper
            True
        """
        return (self.threshold_values > self.threshold).astype(np.float64)

    @property
    def upper_fraction(self) -> float:
        """Share of the effective sample assigned to the upper regime.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = TARNN(rng.standard_normal(300), order=1).fit()
            >>> bool(abs(res.upper_fraction - 0.5) < 0.01)
            True
        """
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

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> list(TARNN(rng.standard_normal(300), order=1).fit()._series())
            ['observed', 'fitted', 'resid', 'threshold_variable', 'regime_weight']
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

        Example:
            The in-sample fit reproduced from the lag matrix and the
            stored threshold variable:

            >>> import numpy as np
            >>> from cultivars._core import lag_matrix
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(300)
            >>> res = TARNN(y, order=1).fit()
            >>> means = res.predict(lag_matrix(y, 1), res.threshold_values)
            >>> bool(np.allclose(means, res.fittedvalues))
            True
            >>> res.predict(lag_matrix(y, 1), res.threshold_values[:5])
            Traceback (most recent call last):
                ...
            cultivars.exceptions.DimensionError: threshold_values must have shape (299,); got (5,).
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

    def nonlinear_state_space(self, measurement_variance: float) -> NonlinearSSM:
        """The fitted two-network threshold mean as a noisily observed state space.

        The errors-in-variables reading of the fit: the latent state is
        the lag stack of the *true* series (deep enough to hold both the
        feature lags and the transition lag), the transition routes each
        state to its regime's network by the fitted threshold (``z <= r``
        to the lower, as in training), and the observation reads the
        current level through Gaussian measurement error of the stated
        variance, required and strictly positive because with none the
        model is observation-driven and filtering it is vacuous. The hard
        switch makes the transition non-smooth at the threshold, so the
        particle filter is the natural reader; the extrapolation caveat of
        the single-network emitter applies to each regime's network on its
        own training subsample.

        Args:
            measurement_variance: Observation noise variance the emitted
                system is read through, strictly positive.

        Returns:
            The :class:`NonlinearSSM`.

        Raises:
            SpecificationError: If the threshold variable is external --
                the emitted transition can depend only on the state, so
                only self-exciting fits have a closed state-space form --
                or if the variance is not strictly positive.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(300)
            >>> TARNN(y, order=1, delay=2).fit().nonlinear_state_space(0.1).k_states
            2
            >>> external = TARNN(y, order=1, threshold_variable=rng.standard_normal(300)).fit()
            >>> external.nonlinear_state_space(0.1)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: the threshold variable is an external ...
        """
        if not self.self_exciting:
            raise SpecificationError(
                "the threshold variable is an external series, so the "
                "regime at each step depends on something outside the "
                "state and the transition map has no closed form; only "
                "self-exciting fits emit a state space."
            )
        order, delay = self.order, self.delay
        depth = max(order, delay)

        def mean_map(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.asarray(
                self.predict(states[:, :order], states[:, delay - 1]),
                dtype=np.float64,
            )

        return NonlinearSSM._lag_stack_state_space(
            mean_map,
            order=depth,
            sigma2=self.sigma2,
            measurement_variance=float(measurement_variance),
            center=float(np.mean(self.endog)),
            spread=float(np.var(self.endog)),
        )

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(300)
            >>> TARNN(y, order=2, delay=1).fit()._comparison_label()
            'TAR-NN(2, d=1)'
            >>> TARNN(y, order=1, threshold_variable=np.arange(300.0)).fit()._comparison_label()
            'TAR-NN/x(1, d=1)'
        """
        family = "TAR-NN" if self.self_exciting else "TAR-NN/x"
        return f"{family}({self.order}, d={self.delay})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        No coefficient rows, as for :class:`ARNNResult`; the notes add the
        split and its regime counts, the fixed-threshold caveat, and the
        external-variable flag when it applies.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> table = TARNN(rng.standard_normal(300), order=1).fit()._summary_table()
            >>> table.title, len(table.rows), len(table.notes), table.notes[1][:9]
            ('TAR-NN(1, d=1) Results', 0, 6, 'Threshold')
        """
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
    r"""Neural autoregression: one learned function of ``order`` lagged levels.

    The model :math:`y_t = f(y_{t-1}, \ldots, y_{t-p}) + \varepsilon_t`,
    with :math:`f` fitted by the engine on the lag matrix and the
    innovation variance concentrated out. The default engine is a
    single-hidden-layer perceptron trained by L-BFGS with an L2 penalty
    on the weights and a handful of random restarts, so the fit is
    deterministic for a fixed ``seed`` and the capacity is set by
    ``hidden_units``. Order selection by the information criteria is
    meaningful within one engine configuration only.

    Attributes:
        _endog: The validated series.
        _order: Number of lagged levels.
        _engine: The training backend.

    Args:
        endog: The series.
        order: Number of lagged levels fed to the learner.
        engine: Training backend. Anything exposing
            ``fit(features, target) -> MeanPredictor`` qualifies; defaults to
            the package's reference L-BFGS-trained perceptron.

    Raises:
        SpecificationError: If ``order`` is not a positive integer.
        DimensionError: If ``endog`` is not one-dimensional or is too
            short for the order.
        NumericalError: If ``endog`` contains non-finite values.

    See Also:
        * :class:`ARNNResult` -- what :meth:`fit` returns.
        * :class:`TARNN` -- one function per regime.
        * :class:`~cultivars.univariate.autoregression.AR` -- the linear
          model this generalizes.
        * :class:`~cultivars.univariate.smooth_transition.LSTAR` -- a
          parametric nonlinearity with interpretable coefficients.

    References:
        Kuan, C.-M., & White, H. (1994). Artificial neural networks: An
        econometric perspective. *Econometric Reviews*, 13(1), 1-91.

        Trapletti, A., Leisch, F., & Hornik, K. (2000). Stationary and
        integrated autoregressive neural network processes. *Neural
        Computation*, 12(10), 2427-2450.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(500)
        >>> for t in range(1, 500):
        ...     y[t] = 0.8 * np.tanh(2.0 * y[t - 1]) + 0.3 * rng.standard_normal()
        >>> res = ARNN(y, order=1).fit()
        >>> bool(res.r_squared > 0.3)
        True

        The learned map beats the linear one on the same lags:

        >>> from cultivars.univariate.autoregression import AR
        >>> linear = AR(y, order=1).fit()
        >>> bool(res.ssr < np.sum(linear.resid**2))
        True
    """

    __slots__ = ()

    def fit(self) -> ARNNResult:
        """Train the mean function and concentrate out the variance.

        Returns:
            The fitted :class:`ARNNResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = ARNN(rng.standard_normal(200), order=2).fit()
            >>> res.nobs, res.order, bool(res.sigma2 > 0)
            (198, 2, True)
        """
        return ARNNResult._from_fit(self._fit_family(), self)


class TARNN(_NeuralThresholdModel[TARNNResult]):
    r"""Neural threshold autoregression: one learned function per regime.

    The sample is split at a fixed threshold on a lagged variable --
    the series itself or an external one -- and a separate network is
    trained on each side, with one pooled innovation variance. The
    threshold defaults to the median of the lagged threshold variable so
    the two training sets are balanced; ``trim`` refuses a split that
    would leave either network with less than that share of the
    effective sample. The threshold is not searched: with a nonlinear
    learner per regime, a grid search would retrain both networks at
    every candidate, and the resulting split would carry a sampling
    uncertainty the record could not report.

    Attributes:
        _endog: The validated series.
        _order: Number of lagged levels.
        _engine: The training backend.
        _threshold_variable: The external variable, or ``None``.
        _delay: Delay of the threshold variable.
        _threshold: The fixed split, or ``None`` for the median.
        _trim: Minimum regime share.

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

    Raises:
        SpecificationError: If ``order`` or ``delay`` is not a positive
            integer.
        DimensionError: If ``endog`` is not one-dimensional, is too short,
            or ``threshold_variable`` is not aligned with it.
        NumericalError: If a series contains non-finite values, or the
            threshold leaves a regime with fewer than ``trim`` of the
            effective sample (raised by :meth:`fit`).

    See Also:
        * :class:`TARNNResult` -- what :meth:`fit` returns.
        * :class:`ARNN` -- one function over the whole sample.
        * :class:`~cultivars.univariate.threshold.SETAR` -- linear regimes
          with a searched threshold.

    References:
        Tong, H. (1990). *Non-linear Time Series: A Dynamical System
        Approach*. Oxford University Press.

        Terasvirta, T., Lin, C.-F., & Granger, C. W. J. (1993). Power of
        the neural network linearity test. *Journal of Time Series
        Analysis*, 14(2), 209-220.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> z = np.zeros(600)
        >>> for t in range(1, 600):
        ...     slope = 0.6 if z[t - 1] <= 0 else -0.4
        ...     z[t] = slope * z[t - 1] + 0.3 * rng.standard_normal()
        >>> res = TARNN(z, order=1, threshold=0.0).fit()
        >>> res.threshold, res.n_lower > res.n_upper
        (0.0, True)
        >>> TARNN(z, order=1, threshold=5.0).fit()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.NumericalError: threshold 5.0 leaves a regime with too few ...
    """

    __slots__ = ()

    def fit(self) -> TARNNResult:
        """Split on the threshold, train one function per regime.

        Returns:
            The fitted :class:`TARNNResult`.

        Raises:
            NumericalError: If the split leaves a regime below ``trim``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = TARNN(rng.standard_normal(300), order=1).fit()
            >>> res.nobs, res.n_lower, res.n_upper
            (299, 150, 149)
        """
        return TARNNResult._from_fit(self._fit_family(), self)
