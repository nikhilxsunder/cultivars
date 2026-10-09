# filepath: /src/cultivars/univariate/smooth_transition.py
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
r"""Smooth-transition autoregressions: LSTAR and ESTAR.

Two linear autoregressions blended by a weight that moves continuously
with a lagged value of the series,

.. math::

   y_t = \bigl(1 - G(y_{t-d})\bigr)\,\phi_L' x_t + G(y_{t-d})\,\phi_U' x_t
   + \varepsilon_t,
   \qquad x_t = (1, y_{t-1}, \ldots, y_{t-p})',

with the logistic transition
:math:`G(z) = \bigl(1 + \exp(-\gamma (z - c)/s)\bigr)^{-1}` -- monotone,
so the regimes are low and high -- or the exponential transition
:math:`G(z) = 1 - \exp(-\gamma (z - c)^2 / s^2)` -- symmetric about
:math:`c`, so the regimes are near and far from it. :math:`s` is the
standard deviation of the transition variable, so :math:`\gamma` is a
speed per standard deviation and comparable across series; as
:math:`\gamma \to \infty` the logistic weight becomes an indicator and
the model is the hard-switch :mod:`~cultivars.univariate.threshold`
model with two extra parameters. Conditional on :math:`(\gamma, c)` the
model is linear in :math:`(\phi_L, \phi_U)`, so estimation concentrates
the regime blocks out by least squares and searches only the two
transition parameters,

.. math::

   (\hat\gamma, \hat c) = \arg\min_{\gamma, c}\;
   \min_{\phi_L, \phi_U} \sum_t \varepsilon_t(\gamma, c, \phi_L, \phi_U)^2,

by derivative-free multi-start, because the surface is flat in
:math:`\gamma` away from the data and multimodal in :math:`c`.

Two commitments shape the surface. First, the transition is reported on
a scale a reader can judge: ``gamma`` per standard deviation with the
scale retained, and a flag, ``is_effectively_abrupt``, raised when the
fitted speed has driven the weight to an indicator and the likelihood is
flat in ``gamma`` -- the smooth model has then said nothing a
:class:`~cultivars.univariate.threshold.SETAR` would not, and the
summary says so. Second, nonlinearity is not tested here. Under
linearity the threshold, the delay and the speed drop out of the
likelihood, so a chi-squared likelihood ratio against the linear
autoregression has no limit distribution (the Davies problem); the
result's ``likelihood_ratio_test`` refuses, the criteria rank
specifications without testing them, and the Terasvirta linearity test
is :func:`~cultivars.diagnostics.nonlinearity.terasvirta`.

Layout. :class:`LSTAR` and :class:`ESTAR` fix the transition family and
delegate to ``_SmoothTransitionModel`` in ``_internals``, which
validates ``order`` and ``delay`` through ``validate_order`` and the
family through ``validate_choice`` against
:data:`~cultivars.typing.Transition`; ``_build_objective`` lays out the
regressors with ``deterministic_columns`` and ``lag_matrix``, seeds the
threshold from the best hard split over interior quantiles, and hands
a ``_SmoothTransitionObjective`` -- whose ``least_squares`` is ``ols``
from ``_core`` on the weighted blocks -- to ``_maximize_likelihood``
under Nelder-Mead. The packed ``_SmoothTransitionFit`` is assembled by
:meth:`STARResult._from_fit`; the result extends
``_ObservedRegimeResult``, which supplies the per-regime stability
verdicts, the refusal of the likelihood-ratio test and ``simulate``
through ``_simulate_two_regime``, and emits its mean as a
:class:`~cultivars.state_space.nonlinear.NonlinearSSM` through
``NonlinearSSM._lag_stack_state_space``. The hard-switch counterpart is
:mod:`~cultivars.univariate.threshold`; the latent-regime counterpart is
:mod:`~cultivars.univariate.regime_switching`; the vector form is
:mod:`cultivars.multivariate.regime_switching`.

References:
    Terasvirta, T. (1994). Specification, estimation, and evaluation of
    smooth transition autoregressive models. *Journal of the American
    Statistical Association*, 89(425), 208-218.

    van Dijk, D., Terasvirta, T., & Franses, P. H. (2002). Smooth
    transition autoregressive models: A survey of recent developments.
    *Econometric Reviews*, 21(1), 1-47.

    Davies, R. B. (1987). Hypothesis testing when a nuisance parameter
    is present only under the alternative. *Biometrika*, 74(1), 33-43.

Example:
    A smooth switch fitted three ways; the smooth model wins on BIC, the
    hard switch is close, and the linear fit is not in the race:

    >>> import numpy as np
    >>> from cultivars.univariate.autoregression import AR
    >>> from cultivars.univariate.threshold import SETAR
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros(800)
    >>> for t in range(1, 800):
    ...     w = 1.0 / (1.0 + np.exp(-10.0 * y[t - 1]))
    ...     y[t] = (1 - w) * (0.5 + 0.6 * y[t - 1]) + w * (-0.5 - 0.6 * y[t - 1])
    ...     y[t] += 0.3 * rng.standard_normal()
    >>> smooth = LSTAR(y, order=1).fit()
    >>> hard = SETAR(y, order=1).fit()
    >>> linear = AR(y, order=1).fit()
    >>> bic = [fit.information_criteria.bic for fit in (smooth, hard, linear)]
    >>> bool(bic[0] < bic[1] < bic[2]), bool(bic[2] - bic[0] > 400)
    (True, True)
    >>> smooth.is_effectively_abrupt, bool(abs(smooth.threshold - hard.threshold) < 0.2)
    (False, True)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..engine._core import SummaryTable, trailing_lag
from ..engine._internals import (
    _ObservedRegimeResult,
    _SmoothTransitionFit,
    _SmoothTransitionModel,
)
from ..state_space.nonlinear import NonlinearSSM

__all__ = ["ESTAR", "LSTAR", "STARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class STARResult(_ObservedRegimeResult):
    r"""A fitted smooth-transition autoregression.

    The model blends two linear autoregressions by a weight that moves
    smoothly with a lagged value of the series,

    .. math::

        y_t = \bigl(1 - G(z_t)\bigr)\,\phi_L' x_t + G(z_t)\,\phi_U' x_t
        + \varepsilon_t,
        \qquad x_t = (1, y_{t-1}, \ldots, y_{t-p})',
        \qquad z_t = y_{t-d},

    with the logistic transition
    :math:`G(z) = \bigl(1 + \exp(-\gamma (z - c) / s)\bigr)^{-1}` and the
    exponential transition :math:`G(z) = 1 - \exp(-\gamma (z - c)^2 / s^2)`,
    where :math:`s` is the sample standard deviation of :math:`z`. The
    logistic weight is monotone in :math:`z`, so the regimes are "low"
    and "high"; the exponential weight is symmetric about :math:`c`, so
    both tails share a regime and the middle is the other. Conditional on
    :math:`(\gamma, c)` the model is linear, so the regime blocks are
    concentrated out by least squares and only the two transition
    parameters are searched; ``gamma`` is reported per standard deviation
    of the transition variable so it is comparable across series, with
    the scale retained so the weight path can be reproduced.

    Attributes:
        transition: ``"logistic"`` or ``"exponential"``.
        gamma: Transition speed, in units of standard deviations of the
            transition variable.
        transition_scale: The standard deviation ``gamma`` is expressed
            against, retained so the weight path can be reproduced and so
            ``gamma`` can be read on the original scale.

    Note:
        The inherited fields are ``endog``, ``fittedvalues``, ``resid``,
        ``llf``, ``nobs``, ``n_params``, ``order``, ``delay``,
        ``threshold``, ``threshold_values``, ``lower_params``,
        ``upper_params``, ``sigma2`` and ``ssr``; the inherited properties
        ``upper_fraction``, ``lower_stability``, ``upper_stability`` and
        ``is_regimewise_stationary`` read the regime blocks as linear
        autoregressions, and ``likelihood_ratio_test`` refuses because the
        threshold and delay are unidentified under linearity. The
        effective sample drops ``max(order, delay)`` leading observations.
        ``n_params`` counts the two regime blocks, the threshold, ``gamma``
        and the variance. Standard errors are not yet reported by this
        estimator.

    See Also:
        * :class:`LSTAR`, :class:`ESTAR` -- the specifications that produce
          this.
        * :class:`~cultivars.univariate.threshold.SETARResult` -- the hard
          switch this smooths; :attr:`is_effectively_abrupt` says when the
          two coincide.
        * :class:`~cultivars.state_space.nonlinear.NonlinearSSM` -- what
          :meth:`nonlinear_state_space` emits.

    References:
        Terasvirta, T. (1994). Specification, estimation, and evaluation
        of smooth transition autoregressive models. *Journal of the
        American Statistical Association*, 89(425), 208-218.

        van Dijk, D., Terasvirta, T., & Franses, P. H. (2002). Smooth
        transition autoregressive models: A survey of recent developments.
        *Econometric Reviews*, 21(1), 1-47.

    Example:
        A logistic transition between a rising and a falling regime,
        recovered with its threshold:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(600)
        >>> for t in range(1, 600):
        ...     w = 1.0 / (1.0 + np.exp(-10.0 * y[t - 1]))
        ...     y[t] = (1 - w) * (0.5 + 0.6 * y[t - 1]) + w * (-0.5 - 0.6 * y[t - 1])
        ...     y[t] += 0.3 * rng.standard_normal()
        >>> res = LSTAR(y, order=1).fit()
        >>> res.transition, res.nobs
        ('logistic', 599)
        >>> list(res.params)
        ['lower.const', 'lower.ar.L1', 'upper.const', 'upper.ar.L1', 'threshold', 'gamma', 'sigma2']
        >>> bool(abs(res.threshold) < 0.1), bool(abs(res.gamma / res.transition_scale - 10.0) < 4.0)
        (True, True)
        >>> bool(res.lower_params[1] > 0.4), bool(res.upper_params[1] < -0.4)
        (True, True)
    """

    transition: str
    """``"logistic"`` or ``"exponential"``."""

    gamma: float
    """Transition speed per standard deviation of the transition variable."""

    transition_scale: float
    """Sample standard deviation of the transition variable."""

    @classmethod
    def _from_fit(
        cls, fit: _SmoothTransitionFit, model: _SmoothTransitionModel[STARResult]
    ) -> STARResult:
        """Assemble the public result from a raw fit and its specification.

        The transition variable is re-lagged from the series so that the
        stored copy is aligned with the effective sample, and its standard
        deviation is recomputed from that copy, which is the scale the
        estimator standardized by.

        Args:
            fit: The raw estimator output.
            model: The specification, read for the series, the order and
                the transition family.

        Returns:
            The assembled result.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> model = LSTAR(rng.standard_normal(300), order=2, delay=2)
            >>> res = STARResult._from_fit(model._fit_family(), model)
            >>> res.order, res.delay, res.nobs, res.threshold_values.shape
            (2, 2, 298, (298,))
        """
        values = trailing_lag(model.endog, delay=fit.delay, length=fit.nobs)
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            order=model.order,
            delay=fit.delay,
            threshold=fit.threshold,
            threshold_values=values,
            lower_params=fit.lower_params,
            upper_params=fit.upper_params,
            sigma2=fit.sigma2,
            ssr=fit.ssr,
            transition=model.transition,
            gamma=fit.gamma,
            transition_scale=float(np.std(values)),
        )

    @property
    def regime_weight(self) -> npt.NDArray[np.float64]:
        """The transition function evaluated over the sample.

        Reproduces the estimator's own weighting exactly, including the
        clipping of the exponent -- an extreme ``gamma`` saturates the weight
        rather than overflowing ``exp``. The logistic form is monotone in the
        standardized transition variable, so the two regimes are asymmetric;
        the exponential form is symmetric about the threshold, so both tails
        share a regime and the middle is the other.

        Example:
            The weights reproduce the fitted values from the two blocks:

            >>> import numpy as np
            >>> from cultivars._core import lag_matrix
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(300)
            >>> res = LSTAR(y, order=1).fit()
            >>> w = res.regime_weight
            >>> base = np.column_stack([np.ones(res.nobs), lag_matrix(y, 1)])
            >>> blend = (1 - w) * (base @ res.lower_params) + w * (base @ res.upper_params)
            >>> bool(np.allclose(blend, res.fittedvalues)), bool(0 <= w.min() <= w.max() <= 1)
            (True, True)
        """
        u = (self.threshold_values - self.threshold) / self.transition_scale
        if self.transition == "logistic":
            return 1.0 / (1.0 + np.exp(-np.clip(self.gamma * u, -50.0, 50.0)))
        return 1.0 - np.exp(-np.clip(self.gamma * u**2, 0.0, 50.0))

    def _weight_at(self, z: float) -> float:
        """The transition function at one value of the transition variable.

        The scalar counterpart of :attr:`regime_weight`, used by the
        inherited ``simulate`` to read the regime weight off the simulated
        past; the clipping is the same.

        Args:
            z: The transition variable.

        Returns:
            The upper-regime weight in ``[0, 1]``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = LSTAR(rng.standard_normal(300), order=1).fit()
            >>> round(res._weight_at(res.threshold), 6)
            0.5
            >>> res.simulate(50, seed=1).shape
            (50,)
        """
        u = (z - self.threshold) / self.transition_scale
        if self.transition == "logistic":
            return float(1.0 / (1.0 + np.exp(-np.clip(self.gamma * u, -50.0, 50.0))))
        return float(1.0 - np.exp(-np.clip(self.gamma * u**2, 0.0, 50.0)))

    @property
    def is_effectively_abrupt(self) -> bool:
        """Whether the fitted transition is so fast it is a hard threshold.

        A large ``gamma`` drives the logistic weight to an indicator, at which
        point the smooth model is a :class:`SETAR` with two extra parameters
        and a flat likelihood in ``gamma``. Reported so the flatness is visible
        rather than mistaken for a converged estimate.

        Example:
            A hard switch fitted as a smooth one lands at an enormous
            ``gamma`` and the same threshold as :class:`SETAR`:

            >>> import numpy as np
            >>> from cultivars.univariate.threshold import SETAR
            >>> rng = np.random.default_rng(2)
            >>> y = np.zeros(800)
            >>> for t in range(1, 800):
            ...     mean = 0.5 + 0.7 * y[t - 1] if y[t - 1] <= 0 else -0.5 - 0.6 * y[t - 1]
            ...     y[t] = mean + 0.3 * rng.standard_normal()
            >>> smooth, hard = LSTAR(y, order=1).fit(), SETAR(y, order=1).fit()
            >>> smooth.is_effectively_abrupt, bool(abs(smooth.threshold - hard.threshold) < 0.05)
            (True, True)
        """
        return self.transition == "logistic" and self.gamma > 100.0

    def nonlinear_state_space(self, measurement_variance: float) -> NonlinearSSM:
        """The fitted smooth-transition mean as a noisily observed state space.

        The errors-in-variables reading of the fit: the latent state is
        the lag stack of the *true* series (deep enough to hold both the
        autoregression and the transition lag), the transition blends the
        two regime autoregressions by the fitted weight function --
        reproduced exactly, exponent clipping included -- and the
        observation reads the current level through Gaussian measurement
        error of the stated variance, which is required and strictly
        positive because with none the model is observation-driven and
        filtering it is vacuous. Unlike the hard-threshold emitter this
        transition is smooth, so the extended and unscented filters are
        trustworthy readers alongside the particle filter; the initial
        state is diffuse at the sample mean and variance.

        Args:
            measurement_variance: Observation noise variance the emitted
                system is read through, strictly positive.

        Returns:
            The :class:`~cultivars.state_space.nonlinear.NonlinearSSM`.

        Raises:
            SpecificationError: If the variance is not strictly positive.

        Example:
            Filtering a noisy observation of the series back toward it:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros(400)
            >>> for t in range(1, 400):
            ...     w = 1.0 / (1.0 + np.exp(-10.0 * y[t - 1]))
            ...     y[t] = (1 - w) * (0.5 + 0.6 * y[t - 1]) + w * (-0.5 - 0.6 * y[t - 1])
            ...     y[t] += 0.3 * rng.standard_normal()
            >>> ssm = LSTAR(y, order=1).fit().nonlinear_state_space(0.05)
            >>> ssm.k_states, ssm.k_endog
            (1, 1)
            >>> noisy = y + np.sqrt(0.05) * rng.standard_normal(400)
            >>> filtered = ssm.unscented_filter(noisy[:, None]).filtered_state[:, 0]
            >>> bool(np.mean((filtered - y) ** 2) < np.mean((noisy - y) ** 2))
            True
            >>> ssm = LSTAR(y, order=1).fit().nonlinear_state_space(0.0)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: measurement_variance must be strictly ...
        """
        lower = np.asarray(self.lower_params, dtype=np.float64)
        upper = np.asarray(self.upper_params, dtype=np.float64)
        order, delay = self.order, self.delay
        threshold, gamma = self.threshold, self.gamma
        scale, kind = self.transition_scale, self.transition
        depth = max(order, delay)

        def mean_map(states: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            base = np.column_stack([np.ones(states.shape[0]), states[:, :order]])
            u = (states[:, delay - 1] - threshold) / scale
            if kind == "logistic":
                weight = 1.0 / (1.0 + np.exp(-np.clip(gamma * u, -50.0, 50.0)))
            else:
                weight = 1.0 - np.exp(-np.clip(gamma * u**2, 0.0, 50.0))
            return np.asarray(
                (1.0 - weight) * (base @ lower) + weight * (base @ upper),
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

    def _transition_params(self) -> dict[str, float]:
        """The transition speed, inserted between the threshold and the variance.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = ESTAR(rng.standard_normal(300), order=1).fit()
            >>> list(res._transition_params()), list(res.params)[-3:]
            (['gamma'], ['threshold', 'gamma', 'sigma2'])
        """
        return {"gamma": self.gamma}

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(300)
            >>> LSTAR(y, order=2).fit()._comparison_label()
            'LSTAR(2, d=1)'
            >>> ESTAR(y, order=1, delay=2).fit()._comparison_label()
            'ESTAR(1, d=2)'
        """
        family = "LSTAR" if self.transition == "logistic" else "ESTAR"
        return f"{family}({self.order}, d={self.delay})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        The coefficient table is :attr:`params` in order; the notes give
        the regime-wise stationarity verdict with each block's largest
        companion modulus, the transition family with ``gamma`` on both
        scales and the mean upper weight, a flatness warning when the fit
        is effectively abrupt, and the reminder that the criteria rank
        specifications but do not test for nonlinearity.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> table = ESTAR(rng.standard_normal(300), order=1).fit()._summary_table()
            >>> table.title, len(table.rows), table.notes[1][:23]
            ('ESTAR(1, d=1) Results', 7, 'Transition: exponential')
        """
        shape = (
            "monotone in the transition variable"
            if self.transition == "logistic"
            else "symmetric about the threshold"
        )
        notes = [
            self._stationarity_note(),
            f"Transition: {self.transition}, {shape}. "
            f"gamma = {self.gamma:.4f} per s.d. of the transition variable "
            f"(s.d. = {self.transition_scale:.4f}); "
            f"mean upper weight = {self.upper_fraction:.3f}.",
        ]
        if self.is_effectively_abrupt:
            notes.append(
                "gamma is large enough that the transition is effectively a hard "
                "threshold; the likelihood is nearly flat in gamma here, so prefer "
                "SETAR unless the smooth form is required."
            )
        notes.append(
            "The threshold and delay are not identified under linearity, so information "
            "criteria rank specifications but do not test for nonlinearity."
        )
        notes.append("Standard errors are not yet available for this estimator.")
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=self._summary_metadata(),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=tuple(notes),
        )


class LSTAR(_SmoothTransitionModel[STARResult]):
    r"""Logistic smooth-transition autoregression.

    The weight :math:`G(z) = \bigl(1 + \exp(-\gamma (z - c) / s)\bigr)^{-1}`
    rises monotonically through the threshold, so the two regimes are
    genuinely different states and the dynamics differ above and below.
    Use this when the asymmetry has a direction -- contractions behaving
    differently from expansions, for instance. The transition variable
    is the series' own lag :math:`y_{t-d}`; :math:`\gamma` and :math:`c`
    are searched by multi-start Nelder-Mead over a grid of speeds and
    two candidate thresholds, with the regime blocks concentrated out by
    least squares at each step.

    Attributes:
        _endog: The validated series.
        _order: Autoregressive order within each regime.
        _delay: Delay of the transition variable.
        _transition: ``"logistic"``.

    Args:
        endog: The series.
        order: Autoregressive order within each regime.
        delay: Delay of the transition variable ``y_{t-d}``.

    Raises:
        SpecificationError: If ``order`` or ``delay`` is not a positive
            integer.
        DimensionError: If ``endog`` is not one-dimensional or has fewer
            than ``2(order + 2) + delay`` observations.
        NumericalError: If ``endog`` contains non-finite values, or (at
            :meth:`fit`) the transition variable has zero variance.

    See Also:
        * :class:`STARResult` -- what :meth:`fit` returns.
        * :class:`ESTAR` -- the symmetric transition.
        * :class:`~cultivars.univariate.threshold.SETAR` -- the hard-switch
          limit, which this fit reports through
          :attr:`STARResult.is_effectively_abrupt`.

    References:
        Terasvirta, T. (1994). Specification, estimation, and evaluation
        of smooth transition autoregressive models. *Journal of the
        American Statistical Association*, 89(425), 208-218.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(600)
        >>> for t in range(1, 600):
        ...     w = 1.0 / (1.0 + np.exp(-10.0 * y[t - 1]))
        ...     y[t] = (1 - w) * (0.5 + 0.6 * y[t - 1]) + w * (-0.5 - 0.6 * y[t - 1])
        ...     y[t] += 0.3 * rng.standard_normal()
        >>> res = LSTAR(y, order=1).fit()
        >>> res._comparison_label(), bool(abs(res.threshold) < 0.1)
        ('LSTAR(1, d=1)', True)
        >>> bool(res.lower_params[1] > 0.4), bool(res.upper_params[1] < -0.4)
        (True, True)
    """

    __slots__ = ()

    def __init__(self, endog: npt.ArrayLike, *, order: int, delay: int = 1) -> None:
        """Validate the specification and the data.

        Args:
            endog: The series.
            order: Autoregressive order within each regime.
            delay: Delay of the transition variable.

        Raises:
            SpecificationError: If ``order`` or ``delay`` is invalid.
            DimensionError: If the series has the wrong shape or is too
                short.
            NumericalError: If the series is not finite.

        Example:
            >>> import numpy as np
            >>> model = LSTAR(np.arange(50.0), order=2, delay=3)
            >>> model.order, model.delay, model.transition
            (2, 3, 'logistic')
            >>> LSTAR(np.arange(50.0), order=0)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: order must be >= 1; got 0.
        """
        super().__init__(endog, order=order, transition="logistic", delay=delay)

    def fit(self) -> STARResult:
        """Estimate the transition by concentrated least squares.

        Returns:
            The fitted :class:`STARResult`.

        Raises:
            NumericalError: If the transition variable has zero variance.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = LSTAR(rng.standard_normal(300), order=1).fit()
            >>> res.transition, res.nobs, res.n_params
            ('logistic', 299, 6)
            >>> LSTAR(np.ones(100), order=1).fit()
            Traceback (most recent call last):
                ...
            cultivars.exceptions.NumericalError: transition variable has zero variance.
        """
        return STARResult._from_fit(self._fit_family(), self)


class ESTAR(_SmoothTransitionModel[STARResult]):
    r"""Exponential smooth-transition autoregression.

    The weight :math:`G(z) = 1 - \exp(-\gamma (z - c)^2 / s^2)` is
    symmetric about the threshold, so both tails share one regime and
    the middle is the other. This is the natural form for mean reversion
    that strengthens with distance from equilibrium -- the canonical
    application is a real exchange rate under transaction costs, where
    small deviations persist and large ones are arbitraged away. In the
    result's vocabulary the "lower" regime is the middle (weight near 0
    at :math:`z = c`) and the "upper" regime is the tails.

    Attributes:
        _endog: The validated series.
        _order: Autoregressive order within each regime.
        _delay: Delay of the transition variable.
        _transition: ``"exponential"``.

    Args:
        endog: The series.
        order: Autoregressive order within each regime.
        delay: Delay of the transition variable ``y_{t-d}``.

    Raises:
        SpecificationError: If ``order`` or ``delay`` is not a positive
            integer.
        DimensionError: If ``endog`` is not one-dimensional or has fewer
            than ``2(order + 2) + delay`` observations.
        NumericalError: If ``endog`` contains non-finite values, or (at
            :meth:`fit`) the transition variable has zero variance.

    See Also:
        * :class:`STARResult` -- what :meth:`fit` returns.
        * :class:`LSTAR` -- the monotone transition.
        * :mod:`~cultivars.diagnostics.unit_roots` -- the linear
          unit-root tests this model is usually set against.

    References:
        Terasvirta, T. (1994). Specification, estimation, and evaluation
        of smooth transition autoregressive models. *Journal of the
        American Statistical Association*, 89(425), 208-218.

        Kapetanios, G., Shin, Y., & Snell, A. (2003). Testing for a unit
        root in the nonlinear STAR framework. *Journal of Econometrics*,
        112(2), 359-379.

    Example:
        Near-unit-root persistence close to equilibrium, fast reversion
        far from it:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> z = np.zeros(800)
        >>> for t in range(1, 800):
        ...     w = 1.0 - np.exp(-3.0 * z[t - 1] ** 2)
        ...     z[t] = (1 - w) * 0.95 * z[t - 1] + w * 0.2 * z[t - 1] + 0.3 * rng.standard_normal()
        >>> res = ESTAR(z, order=1).fit()
        >>> res._comparison_label(), bool(abs(res.threshold) < 0.3)
        ('ESTAR(1, d=1)', True)
        >>> bool(res.lower_params[1] > 0.8), bool(res.upper_params[1] < 0.5)
        (True, True)
        >>> near = np.abs(res.threshold_values - res.threshold) < 0.2
        >>> bool(res.regime_weight[near].mean() < res.regime_weight[~near].mean())
        True
    """

    __slots__ = ()

    def __init__(self, endog: npt.ArrayLike, *, order: int, delay: int = 1) -> None:
        """Validate the specification and the data.

        Args:
            endog: The series.
            order: Autoregressive order within each regime.
            delay: Delay of the transition variable.

        Raises:
            SpecificationError: If ``order`` or ``delay`` is invalid.
            DimensionError: If the series has the wrong shape or is too
                short.
            NumericalError: If the series is not finite.

        Example:
            >>> import numpy as np
            >>> model = ESTAR(np.arange(50.0), order=1)
            >>> model.transition, model.delay
            ('exponential', 1)
            >>> ESTAR(np.arange(6.0), order=1)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.DimensionError: series of length 6 is too short for STAR(1) ...
        """
        super().__init__(endog, order=order, transition="exponential", delay=delay)

    def fit(self) -> STARResult:
        """Estimate the transition by concentrated least squares.

        Returns:
            The fitted :class:`STARResult`.

        Raises:
            NumericalError: If the transition variable has zero variance.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = ESTAR(rng.standard_normal(300), order=2, delay=2).fit()
            >>> res.transition, res.nobs, list(res.params)[:3]
            ('exponential', 298, ['lower.const', 'lower.ar.L1', 'lower.ar.L2'])
        """
        return STARResult._from_fit(self._fit_family(), self)
