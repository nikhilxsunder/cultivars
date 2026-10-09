# filepath: /src/cultivars/multivariate/nonlinear/smooth_transition.py
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
r"""The smooth-transition vector autoregression: regimes as a dial, not a switch.

An STVAR replaces the threshold VAR's indicator with a transition function
taking values in :math:`[0, 1]`,

.. math::

   y_t = (1 - G(z_t))\,\bigl(c_L + \textstyle\sum_{i=1}^{p} A_{L,i}\, y_{t-i}\bigr)
   + G(z_t)\,\bigl(c_U + \textstyle\sum_{i=1}^{p} A_{U,i}\, y_{t-i}\bigr) + u_t,

so that at every date the system is a convex blend of two regimes, with
the blend weight read off an observable :math:`z_t = s_{t-d}`. The
canonical modern use is Auerbach and Gorodnichenko's fiscal multipliers,
where the weight is a smoothed output-growth measure and the finding is
that government spending works differently in recessions than in
expansions -- a statement the smooth form can make with dates that are
seventy percent recession, which no hard split can represent. One class
carries both transition shapes, because the multivariate literature names
the model STVAR whichever :math:`G` it uses:

.. math::

   G(z) = \bigl(1 + e^{-\gamma (z - c)/\sigma_z}\bigr)^{-1}
   \quad\text{(logistic)},
   \qquad
   G(z) = 1 - e^{-\gamma (z - c)^2/\sigma_z^2}
   \quad\text{(exponential)}.

The logistic weight is monotone in the transition variable, so the regimes
are directional states -- recession versus expansion -- and the
exponential weight is symmetric about the threshold, so both tails share a
regime and the middle is the other. (The univariate family splits these
into LSTAR and ESTAR because those are established names; LSTVAR and
ESTVAR are not.) The speed :math:`\gamma` is stated per standard deviation
of :math:`z` so that one value means the same thing on any series.

Two commitments shape the surface. First, estimation concentrates:
conditional on :math:`(\gamma, c)` the model is linear, so the regime
coefficients come from one weighted multivariate least-squares solve and
only the pair is searched, derivative-free and multi-start -- from the
best hard split and from the median, at four initial speeds -- on the
log-determinant of the residual covariance. The innovation covariance is
common across regimes, and the fit record says why: smooth weights never
partition the sample, so a per-regime covariance has no subsample to be
estimated from. Second, the weak identification of the transition is
reported rather than hidden. The likelihood is nearly flat in
:math:`\gamma` once the weight is effectively an indicator, so a very
large fitted speed is a lower bound and
:attr:`STVARResult.is_effectively_abrupt` says when the smooth model has
collapsed onto a threshold VAR with two spare parameters; and because
:math:`(\gamma, c)` are not identified under linearity (Davies), the
chi-squared linearity test is refused and information criteria rank
specifications without testing for nonlinearity. Everything the threshold
module says about propagation carries over: ``irf``, ``fevd`` and
``ma_representation`` are regime-conditional, holding one regime frozen.

Layout. :class:`STVAR` validates on
``_SmoothTransitionVectorAutoRegressionModel`` in ``_internals``, which
takes the named-or-external transition variable, its delay and the
aligned design from ``_ObservedRegimeVectorModel``, adds the transition
family, and builds a ``_VectorSmoothTransitionObjective`` from
``_internals._objectives`` seeded by a grid of hard splits;
``_fit_regimes`` runs it through ``_solve`` from ``_internals._solvers``,
reads the concentrated coefficients and residuals at the optimum, and
packs a ``_VectorSmoothTransitionFit``. :class:`STVARResult` extends
``_VectorObservedRegimeResult`` -- the two coefficient stacks, the
regime-conditional propagation surface, the forecast, regime-wise
stability and comparison by information criteria -- supplying the weight
function and the single covariance the smooth form admits. The abrupt
member of the family is :mod:`~cultivars.multivariate.nonlinear.threshold`;
the nonparametric check on the transition's shape is
:mod:`~cultivars.multivariate.nonlinear.functional_coefficient`; the
univariate models are :mod:`~cultivars.univariate.smooth_transition`.

References:
    Auerbach, A. J., & Gorodnichenko, Y. (2012). Measuring the output
    responses to fiscal policy. *American Economic Journal: Economic
    Policy*, 4(2), 1-27.

    Terasvirta, T., & Yang, Y. (2014). Specification, estimation and
    evaluation of vector smooth transition autoregressive models with
    applications. *CREATES Research Paper* 2014-08.

    Granger, C. W. J., & Terasvirta, T. (1993). *Modelling Nonlinear
    Economic Relationships*. Oxford University Press.

    Davies, R. B. (1987). Hypothesis testing when a nuisance parameter is
    present only under the alternative. *Biometrika*, 74(1), 33-43.

Example:
    A system whose dynamics slide from a persistent regime to a
    mean-reverting one as ``y1`` rises: the smooth fit recovers the two
    regimes and a moderate speed, the weight path is a dial rather than a
    switch, and the regime-conditional responses differ in sign:

    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> y = np.zeros((400, 2))
    >>> for t in range(1, 400):
    ...     g = 1.0 / (1.0 + np.exp(-2.0 * y[t - 1, 0]))
    ...     y[t] = ((1.0 - g) * 0.6 - g * 0.3) * y[t - 1] + rng.standard_normal(2)
    >>> res = STVAR(y, order=1, transition_variable="y1").fit()
    >>> res.is_effectively_abrupt, bool(1.0 < res.gamma < 3.0)
    (False, True)
    >>> weight = res.regime_weight
    >>> bool(np.mean((weight > 0.1) & (weight < 0.9)) > 0.4)
    True
    >>> lower = res.irf(2, regime="lower", orthogonalized=False)[1, 0, 0]
    >>> upper = res.irf(2, regime="upper", orthogonalized=False)[1, 0, 0]
    >>> bool(lower > 0.4), bool(upper < 0.0)
    (True, True)
    >>> "STVAR(1, logistic, d=1) Results" in str(res.summary())
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ...engine._core import Regime, SummaryTable
from ...engine._internals import (
    _SmoothTransitionVectorAutoRegressionModel,
    _VectorObservedRegimeResult,
    _VectorSmoothTransitionFit,
)

__all__ = ["STVAR", "STVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class STVARResult(_VectorObservedRegimeResult):
    r"""A fitted smooth-transition vector autoregression.

    The estimate of

    .. math::

       y_t = (1 - G(z_t))\,\bigl(c_L + \textstyle\sum_{i} A_{L,i}\, y_{t-i}\bigr)
       + G(z_t)\,\bigl(c_U + \textstyle\sum_{i} A_{U,i}\, y_{t-i}\bigr) + u_t,
       \qquad z_t = s_{t-d},

    with the transition weight either logistic or exponential in the
    standardized distance from the threshold,

    .. math::

       G(z) = \bigl(1 + e^{-\gamma (z - c)/\sigma_z}\bigr)^{-1}
       \quad\text{or}\quad
       G(z) = 1 - e^{-\gamma (z - c)^2/\sigma_z^2},

    estimated by concentrated maximum likelihood: given :math:`(\gamma, c)`
    the two regimes' coefficients come from one weighted least-squares
    solve, and only the pair is searched on the log-determinant of the
    residual covariance. The record extends the observed-regime surface --
    the two coefficient stacks, regime-conditional ``irf``, ``fevd``,
    ``ma_representation`` and ``forecast``, the stability of each regime,
    and comparison by information criteria -- with the transition's own
    parameters and the single covariance the smooth form admits.

    Note:
        ``gamma`` is reported per standard deviation of the transition
        variable; divide by ``transition_scale`` for the speed on the
        original scale. The logistic weight equals one half at the
        threshold and is monotone in :math:`z`, so the regimes are
        directional; the exponential weight equals zero at the threshold
        and one far from it on either side, so the "upper" regime is the
        tails and the "lower" regime the middle. The likelihood is nearly
        flat in :math:`\gamma` once the weight is effectively an indicator,
        so a very large fitted ``gamma`` is a lower bound rather than an
        estimate, and :attr:`is_effectively_abrupt` says when that has
        happened. Information criteria rank specifications but do not test
        for nonlinearity, because :math:`(\gamma, c)` are not identified
        under linearity. Standard errors are not yet available.

    Attributes:
        transition: ``"logistic"`` or ``"exponential"``.
        gamma: Transition speed, per standard deviation of the transition
            variable.
        transition_scale: The standard deviation ``gamma`` is expressed
            against, retained so the weight path can be reproduced and so
            ``gamma`` can be read on the original scale.
        sigma_u: Common innovation covariance, dof-corrected. One covariance
            rather than two: smooth weights never partition the sample, so a
            per-regime covariance has no subsample to be estimated from.

    See Also:
        * :class:`STVAR` -- the model that produces this record.
        * :class:`~cultivars.multivariate.nonlinear.threshold.TVARResult` --
          the abrupt limit, with a covariance per regime.
        * :class:`~cultivars.univariate.smooth_transition.STARResult` -- the
          univariate counterpart, split into LSTAR and ESTAR.

    References:
        Auerbach, A. J., & Gorodnichenko, Y. (2012). Measuring the output
        responses to fiscal policy. *American Economic Journal: Economic
        Policy*, 4(2), 1-27.

        Terasvirta, T., & Yang, Y. (2014). Specification, estimation and
        evaluation of vector smooth transition autoregressive models with
        applications. *CREATES Research Paper* 2014-08.

    Example:
        A bivariate system whose own-lag coefficient slides from 0.6 to
        -0.3 as a logistic function of ``y1``'s lag with speed 2 per unit;
        the fit recovers both regimes, a speed of the right size per
        standard deviation, and a weight path that spends a real share of
        the sample in between:

        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> y = np.zeros((400, 2))
        >>> for t in range(1, 400):
        ...     g = 1.0 / (1.0 + np.exp(-2.0 * y[t - 1, 0]))
        ...     a = (1.0 - g) * 0.6 + g * -0.3
        ...     y[t] = a * y[t - 1] + rng.standard_normal(2)
        >>> res = STVAR(y, order=1, transition_variable="y1").fit()
        >>> res.transition, res.delay, res.threshold_name, res.nobs, res.n_params
        ('logistic', 1, 'y1', 399, 17.0)
        >>> bool(1.0 < res.gamma < 3.0), bool(abs(res.threshold) < 0.5), res.is_effectively_abrupt
        (True, True, False)
        >>> lower, upper = np.diag(res.lower_coefficients[0]), np.diag(res.upper_coefficients[0])
        >>> bool(np.all(np.abs(lower - 0.6) < 0.15)), bool(np.all(np.abs(upper + 0.3) < 0.25))
        (True, True)
        >>> weight = res.regime_weight
        >>> between = np.mean((weight > 0.2) & (weight < 0.8))
        >>> bool(0.2 < res.upper_fraction < 0.5), bool(between > 0.25)
        (True, True)
        >>> res.sigma_u.shape, bool(np.allclose(res._weight_at(np.array([res.threshold])), 0.5))
        ((2, 2), True)
    """

    transition: str
    """The transition family: ``"logistic"`` or ``"exponential"``."""
    gamma: float
    """Transition speed per standard deviation of the transition variable."""
    transition_scale: float
    """Standard deviation of the delayed transition variable, the unit of ``gamma``."""
    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    """``(k, k)`` common innovation covariance, dof-corrected. Kept out of the repr."""

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorSmoothTransitionFit,
        model: _SmoothTransitionVectorAutoRegressionModel[STVARResult],
    ) -> STVARResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The packed concentrated-likelihood fit.
            model: The validated specification it was estimated on.

        Returns:
            The frozen :class:`STVARResult`.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> model = STVAR(y, order=1, transition_variable="y2", transition="exponential")
            >>> res = STVARResult._from_fit(model._fit_regimes(), model)
            >>> res.transition, res.threshold_name, res.delay
            ('exponential', 'y2', 1)
        """
        return cls(
            endog=model.endog,
            names=model.names,
            order=model.order,
            trend=model.trend,
            delay=fit.delay,
            threshold=fit.threshold,
            threshold_name=model.transition_name,
            threshold_values=fit.threshold_values,
            transition_series=model.transition_series,
            lower_coefficients=fit.lower_coefficients,
            upper_coefficients=fit.upper_coefficients,
            lower_deterministic=fit.lower_deterministic,
            upper_deterministic=fit.upper_deterministic,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            transition=model.transition,
            gamma=fit.gamma,
            transition_scale=fit.transition_scale,
            sigma_u=fit.sigma_u,
        )

    @property
    def regime_weight(self) -> npt.NDArray[np.float64]:
        """The transition function evaluated over the sample.

        Reproduces the estimator's own weighting exactly, including the
        clipping of the exponent -- an extreme ``gamma`` saturates the weight
        rather than overflowing ``exp``. The logistic form is monotone in the
        standardized transition variable, so the two regimes are directional;
        the exponential form is symmetric about the threshold, so both tails
        share a regime and the middle is the other.

        Returns:
            An array of length ``nobs`` in ``[0, 1]``, aligned with ``resid``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(1)
            >>> y = np.zeros((400, 2))
            >>> for t in range(1, 400):
            ...     g = 1.0 / (1.0 + np.exp(-2.0 * y[t - 1, 0]))
            ...     y[t] = ((1.0 - g) * 0.6 - g * 0.3) * y[t - 1] + rng.standard_normal(2)
            >>> res = STVAR(y, order=1, transition_variable="y1").fit()
            >>> weight = res.regime_weight
            >>> weight.shape, bool(np.all((0.0 <= weight) & (weight <= 1.0)))
            ((399,), True)
            >>> order = np.argsort(res.threshold_values)
            >>> bool(np.all(np.diff(weight[order]) >= 0.0))
            True
            >>> bool(np.isclose(weight.mean(), res.upper_fraction))
            True
        """
        return self._weight_at(self.threshold_values)

    def _weight_at(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The transition function at arbitrary transition values.

        Args:
            values: Transition-variable values, in the original units.

        Returns:
            The weight on the upper regime at each value, in ``[0, 1]``.

        Example:
            The logistic weight is one half at the threshold; the
            exponential weight is zero there and symmetric about it:

            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> logistic = STVAR(y, order=1, transition_variable="y1").fit()
            >>> bool(np.isclose(logistic._weight_at(np.array([logistic.threshold]))[0], 0.5))
            True
            >>> exponential = STVAR(
            ...     y, order=1, transition_variable="y1", transition="exponential"
            ... ).fit()
            >>> c = exponential.threshold
            >>> at = exponential._weight_at(np.array([c, c - 1.0, c + 1.0]))
            >>> bool(at[0] == 0.0), bool(np.isclose(at[1], at[2])), bool(0.0 < at[1] <= 1.0)
            (True, True, True)
        """
        u = (values - self.threshold) / self.transition_scale
        if self.transition == "logistic":
            return 1.0 / (1.0 + np.exp(-np.clip(self.gamma * u, -50.0, 50.0)))
        return 1.0 - np.exp(-np.clip(self.gamma * u**2, 0.0, 50.0))

    def _regime_sigma(self, regime: Regime) -> npt.NDArray[np.float64]:
        """The common innovation covariance, whichever regime is named.

        Args:
            regime: ``"lower"`` or ``"upper"``; ignored, since the smooth
                model estimates one covariance.

        Returns:
            ``sigma_u``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> res = STVAR(y, order=1, transition_variable="y1").fit()
            >>> res._regime_sigma("lower") is res._regime_sigma("upper") is res.sigma_u
            True
        """
        return self.sigma_u

    @property
    def is_effectively_abrupt(self) -> bool:
        """Whether the fitted transition is so fast it is a hard threshold.

        A large ``gamma`` drives the logistic weight to an indicator, at which
        point the smooth model is a
        :class:`~cultivars.multivariate.nonlinear.threshold.TVAR` with two
        extra parameters and a flat likelihood in ``gamma``. Reported so the
        flatness is visible rather than mistaken for a converged estimate.
        Always ``False`` for the exponential form, whose large-``gamma``
        limit is not a hard split of the sample.

        Example:
            A hard-threshold truth drives ``gamma`` far past the cutoff:

            >>> import numpy as np
            >>> rng = np.random.default_rng(2)
            >>> y = np.zeros((300, 2))
            >>> for t in range(1, 300):
            ...     a = 0.7 if y[t - 1, 0] <= 0.0 else -0.3
            ...     y[t] = a * y[t - 1] + rng.standard_normal(2)
            >>> res = STVAR(y, order=1, transition_variable="y1").fit()
            >>> res.is_effectively_abrupt, bool(res.gamma > 100.0)
            (True, True)
            >>> "effectively a hard threshold" in str(res.summary())
            True
        """
        return self.transition == "logistic" and self.gamma > 100.0

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> STVAR(y, order=2, transition_variable="y1", delay=2).fit()._comparison_label()
            'STVAR(2, logistic, d=2)'
        """
        return f"STVAR({self.order}, {self.transition}, d={self.delay})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Returns:
            The observed-regime rows (own first lag in each regime) and
            metadata, with ``gamma`` appended; the notes state regime-wise
            stationarity, the transition's shape, speed, scale and mean
            upper weight, the common-covariance rationale, an abruptness
            warning when ``is_effectively_abrupt``, the non-identification
            of the transition under linearity, the regime-conditional
            reading of the propagation methods, and the absence of standard
            errors.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((200, 2))
            >>> table = STVAR(y, order=1, transition_variable="y1").fit()._summary_table()
            >>> table.title, table.columns, len(table.rows)
            ('STVAR(1, logistic, d=1) Results', ('equation', 'lower own L1', 'upper own L1'), 2)
            >>> table.metadata[-1][0], len(table.notes) in (6, 7)
            ('Gamma', True)
            >>> table.notes[1].startswith("Transition: logistic, monotone")
            True
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
            "The innovation covariance is common across regimes: smooth "
            "weights never partition the sample, so a per-regime covariance "
            "has no subsample to be estimated from.",
        ]
        if self.is_effectively_abrupt:
            notes.append(
                "gamma is large enough that the transition is effectively a "
                "hard threshold; the likelihood is nearly flat in gamma here, "
                "so prefer TVAR unless the smooth form is required."
            )
        notes.append(
            "The threshold, delay, and transition speed are not identified "
            "under linearity, so information criteria rank specifications but "
            "do not test for nonlinearity."
        )
        notes.append(self._regime_conditional_note())
        notes.append("Standard errors are not yet available for this estimator.")
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=(*self._summary_metadata(), ("Gamma", f"{self.gamma:.4f}")),
            columns=("equation", "lower own L1", "upper own L1"),
            rows=self._own_lag_rows(),
            notes=tuple(notes),
        )


class STVAR(_SmoothTransitionVectorAutoRegressionModel[STVARResult]):
    r"""Smooth-transition vector autoregression, in the Auerbach-Gorodnichenko mould.

    At every date the system is a convex blend of two regimes, weighted by a
    logistic or exponential function of an observable -- a named column's
    own lag, or an external series such as a smoothed growth measure. The
    weight :math:`G(z_t)` lies in :math:`[0, 1]`, so a date can be, say,
    seventy percent recession, which no hard split can represent. Estimation
    concentrates the regime coefficients out of the likelihood and searches
    only the transition's speed and location, derivative-free and
    multi-start, from a seed near the best hard split.

    Attributes:
        _endog: The observed panel, from the base model.
        _order: The autoregressive order ``p``, from the base model.
        _trend: The deterministic specification, from the base model.
        _names: The variable labels, from the base model.
        _delays: The single delay, as a one-element tuple, from the
            observed-regime base.
        _threshold_name: The transition variable's label, from the
            observed-regime base.
        _transition_series: The aligned transition series, from the
            observed-regime base.
        _transition: The transition family, ``"logistic"`` or
            ``"exponential"``.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order within each regime.
        transition_variable: A variable name from ``names`` (self-exciting)
            or an aligned external series.
        transition: ``"logistic"`` for directional regimes, ``"exponential"``
            for distance-from-threshold regimes.
        delay: Delay of the transition variable.
        trend: Deterministic terms per regime.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Raises:
        SpecificationError: If the specification is malformed or
            ``transition`` is not one of the two families.
        DimensionError: If the sample cannot support two regimes.

    See Also:
        * :class:`STVARResult` -- the record ``fit`` returns.
        * :class:`~cultivars.multivariate.nonlinear.threshold.TVAR` -- the
          abrupt two-regime model this one smooths.
        * :class:`~cultivars.multivariate.nonlinear.functional_coefficient.FunctionalCoefficientVAR`
          -- the nonparametric specification check on the transition's
          shape.
        * :class:`~cultivars.univariate.smooth_transition.LSTAR` -- the
          univariate logistic model.

    References:
        Auerbach, A. J., & Gorodnichenko, Y. (2012). Measuring the output
        responses to fiscal policy. *American Economic Journal: Economic
        Policy*, 4(2), 1-27.

        Terasvirta, T., & Yang, Y. (2014). Specification, estimation and
        evaluation of vector smooth transition autoregressive models with
        applications. *CREATES Research Paper* 2014-08.

        Granger, C. W. J., & Terasvirta, T. (1993). *Modelling Nonlinear
        Economic Relationships*. Oxford University Press.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((300, 2))
        >>> for t in range(1, 300):
        ...     g = 1.0 / (1.0 + np.exp(-5.0 * y[t - 1, 0]))
        ...     a = (1.0 - g) * 0.6 + g * -0.3
        ...     y[t] = a * y[t - 1] + rng.standard_normal(2)
        >>> res = STVAR(y, order=1, transition_variable="y1", delay=1).fit()
        >>> 0.0 < res.upper_fraction < 1.0
        True
        >>> res.forecast(3).shape
        (3, 2)
        >>> res.irf(4, regime="upper").shape, res.is_regimewise_stationary
        ((5, 2, 2), True)
        >>> STVAR(y, order=1, transition_variable="y1", transition="exponential").fit().transition
        'exponential'
        >>> STVAR(y, order=1, transition_variable="y1", transition="tanh")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: transition must be one of ('logistic', ...
    """

    __slots__ = ()

    def fit(self) -> STVARResult:
        """Estimate the transition by concentrated maximum likelihood.

        Conditional on ``(gamma, c)`` the model is linear, so the regime
        coefficients come from one weighted least-squares solve; the pair
        is searched on the log-determinant of the residual covariance from
        several starts around the best hard split, and the covariance,
        likelihood and parameter count are read off at the optimum.

        Returns:
            The fitted :class:`STVARResult`.

        Raises:
            NumericalError: If the transition variable has zero variance or
                every start lands on a singular residual covariance.

        Example:
            The smooth fit and the hard-split fit on one sample rank
            through the shared comparison surface:

            >>> import numpy as np
            >>> from cultivars.multivariate.nonlinear.threshold import TVAR
            >>> rng = np.random.default_rng(1)
            >>> y = np.zeros((400, 2))
            >>> for t in range(1, 400):
            ...     g = 1.0 / (1.0 + np.exp(-2.0 * y[t - 1, 0]))
            ...     y[t] = ((1.0 - g) * 0.6 - g * 0.3) * y[t - 1] + rng.standard_normal(2)
            >>> smooth = STVAR(y, order=1, transition_variable="y1").fit()
            >>> hard = TVAR(y, order=1, transition_variable="y1").fit()
            >>> smooth.n_params, hard.n_params
            (17.0, 19.0)
            >>> [row[0] for row in smooth.compare(hard, criterion="bic").rows]
            ['STVAR(1, logistic, d=1)', 'TVAR(1, d=1)']
        """
        return STVARResult._from_fit(self._fit_regimes(), self)
