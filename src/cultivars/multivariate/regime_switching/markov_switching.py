# filepath: /src/cultivars/multivariate/regime_switching/markov_switching.py
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
r"""The Markov-switching VAR: latent regimes, and a posterior instead of a date.

An ``M``-regime vector autoregression

.. math::

   y_t = D_{S_t} d_t + \sum_{i=1}^{p} A_{i, S_t}\, y_{t-i} + u_t,
   \qquad u_t \mid S_t = m \sim N(0, \Sigma_m),
   \qquad \Pr(S_t = j \mid S_{t-1} = i) = P_{ij},

whose deterministic block, lag coefficients and innovation covariance may
each switch with a latent first-order Markov chain (Krolzig 1997; Sims and
Zha 2006). Where the threshold and smooth-transition families compute their
regime from an observable, here the regime is never observed at all: what
comes back is a posterior over states at every date, and every reported
quantity -- fitted values, residuals, even the regime a date "is in" -- is
an expectation under that posterior rather than a fact about the data.
Estimation is EM on the same Hamilton filter and Kim smoother that power the
univariate family: the E-step filters and smooths per-regime multivariate
Gaussian densities, and the M-step updates the transition matrix from
expected transition counts and all regimes' coefficient blocks in one
probability-weighted GLS system, so a non-switching block is estimated
jointly across regimes rather than per regime and then averaged. The
likelihood is multimodal, so ``fit`` screens several starts and refines the
best -- which is why it takes a ``seed`` and reports convergence.

Two commitments shape the surface. First, the two facts about mixtures that
a reader would otherwise have to remember are enforced. The likelihood is
invariant to permuting regime labels, so a sorting convention is imposed and
``label_ordering`` names it on every result; without it two runs of the same
model could not be compared coefficient by coefficient. And the number of
regimes cannot be tested by a likelihood ratio -- under the null of fewer
regimes the extra regime's parameters are unidentified and its transition
probabilities sit on the boundary -- so that specific comparison is refused
while tests holding ``M`` fixed remain available. Second, the regime is the
composition hook. Conditional on the chain sitting in regime ``m`` the model
*is* a linear VAR, and :meth:`MarkovSwitchingVARResult.regime` returns that
system dressed as a closed reduced form, identity-stable across calls, which
every identification model in :mod:`~cultivars.multivariate.structural`
accepts directly. :class:`MarkovSwitchingSVAR` builds on exactly that: one
declaration, applied per regime, with the responses held frozen in the
regime they were computed in and read against its expected duration.

Layout. :class:`MarkovSwitchingVAR` is a thin ``fit()`` over
``_MarkovSwitchingVectorAutoRegressionModel`` in ``_internals``, which
extends the linear VAR model with the regime count and switching flags,
builds per-regime selection matrices for the joint M-step, runs
``hamilton_filter`` and ``kim_smoother`` from ``_internals`` inside
``_run_em``, and screens starts in ``_fit_markov`` before applying the
label permutation. :class:`MarkovSwitchingVARResult` carries the summary and
comparison mixins and builds one ``_RegimeSystemResult`` per regime, the
closed-system view the structural layer and :class:`MarkovSwitchingSVAR`
consume; its ``_likelihood_ratio_obstacle`` is where the regime-count test
is refused. :class:`MarkovSwitchingDFM` is a separate estimator on the same
substrate: the Kim-Nelson one-factor model with a switching intercept,
written as a ``_RegimeSwitchingLinearStateSpace`` and maximized through the
continuous-state Kim filter, returning :class:`MarkovSwitchingDFMResult`
with the smoothed low-state probability as its chronology. The scalar
family is :mod:`~cultivars.univariate.regime_switching`; the observed-regime
alternatives are :mod:`~cultivars.multivariate.nonlinear.threshold` and
:mod:`~cultivars.multivariate.nonlinear.smooth_transition`.

References:
    Hamilton, J. D. (1990). Analysis of time series subject to changes in
    regime. *Journal of Econometrics*, 45(1-2), 39-70.

    Kim, C.-J. (1994). Dynamic linear models with Markov-switching.
    *Journal of Econometrics*, 60(1-2), 1-22.

    Krolzig, H.-M. (1997). *Markov-Switching Vector Autoregressions:
    Modelling, Statistical Inference, and Application to Business Cycle
    Analysis*. Springer. Chapters 6 and 9.

    Kim, C.-J., & Nelson, C. R. (1998). Business cycle turning points, a new
    coincident index, and tests of duration dependence based on a dynamic
    factor model with regime switching. *Review of Economics and
    Statistics*, 80(2), 188-201.

    Sims, C. A., & Zha, T. (2006). Were there regime switches in U.S.
    monetary policy? *American Economic Review*, 96(1), 54-81.

    Sims, C. A., Waggoner, D. F., & Zha, T. (2008). Methods for inference in
    large multiple-equation Markov-switching models. *Journal of
    Econometrics*, 146(2), 255-274.

Example:
    Two regimes that differ in mean and scale around a common lag block. The
    fit dates the regimes, the comparison against the one-regime VAR is
    refused on principle while the comparison against a nested switching
    specification goes through, and a recursive identification applied per
    regime gives one impact matrix for each:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> rng = np.random.default_rng(0)
    >>> s = np.repeat([0, 1, 0, 1], 100)
    >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
    >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2, names=("a", "b")).fit(seed=0)
    >>> res.specification, bool((res.most_likely_regime == s[1:]).mean() > 0.98)
    ('MSIH(2)-VAR(1)', True)
    >>> res.likelihood_ratio_test(VAR(y, order=1).fit())  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    cultivars.exceptions.SpecificationError: a chi-squared likelihood-ratio test cannot compare ...
    >>> nested = MarkovSwitchingVAR(y, order=1, n_regimes=2, switching_variance=False)
    >>> nested.fit(seed=0).likelihood_ratio_test(res).df
    3
    >>> svar = MarkovSwitchingSVAR(res).identify()
    >>> svar.shock_names, svar.impact(0).shape, svar.irf(4, regime=1).shape
    (('a', 'b'), (2, 2), (5, 2, 2))
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from ..._core import (
    InformationCriteria,
    ProbabilityType,
    StructuralResult,
    SummaryTable,
    validate_choice,
    validate_endog_matrix,
)
from ..._internals import (
    _ComparisonMixin,
    _MarkovSwitchingVectorAutoRegressionModel,
    _RegimeSwitchingLinearStateSpace,
    _RegimeSystemResult,
    _SummaryMixin,
    _VectorMarkovSwitchingFit,
)
from ...exceptions import DimensionError, NumericalError, SpecificationError
from ..structural.zero_restrictions import RecursiveSVAR

__all__ = [
    "MarkovSwitchingDFM",
    "MarkovSwitchingDFMResult",
    "MarkovSwitchingSVAR",
    "MarkovSwitchingSVARResult",
    "MarkovSwitchingVAR",
    "MarkovSwitchingVARResult",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MarkovSwitchingVARResult(_SummaryMixin, _ComparisonMixin):
    r"""A fitted Markov-switching vector autoregression.

    Conditional on a latent first-order Markov chain :math:`S_t \in
    \{0, \dots, M-1\}` with transition matrix :math:`P`, the panel follows

    .. math::

       y_t = D_{S_t} d_t + \sum_{i=1}^{p} A_{i, S_t}\, y_{t-i} + u_t,
       \qquad u_t \mid S_t = m \sim N(0, \Sigma_m),

    with each of the three blocks either switching with the chain or held
    common across regimes. The chain is never observed: the record carries
    its filtered, predicted and smoothed posteriors, and every quantity that
    depends on the regime -- fitted values, residuals, the regime a date
    "is in" -- is an expectation under the smoothed posterior.

    Note:
        Two consequences of the mixture form are enforced here rather than
        left to the reader. The likelihood is invariant to relabelling the
        regimes, so a sorting convention is imposed and
        :attr:`label_ordering` names it; without it two runs of the same
        model could not be compared coefficient by coefficient. And the
        number of regimes cannot be tested by a likelihood ratio: under the
        null of fewer regimes the extra regime's parameters are unidentified
        and its transition probabilities lie on the boundary, so
        :meth:`likelihood_ratio_test` refuses any comparison that changes
        ``M`` while allowing those that hold it fixed.

    Attributes:
        endog: The observed ``(n, k)`` panel.
        names: Variable labels, in column order.
        order: Autoregressive order ``p`` within each regime.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        n_regimes: Number of latent regimes ``M``.
        switching_mean: Whether the deterministic block switches.
        switching_ar: Whether the lag coefficients switch.
        switching_variance: Whether the innovation covariance switches.
        transition: Row-stochastic ``(M, M)`` matrix, ``transition[i, j] =
            Pr(S_t = j | S_{t-1} = i)``.
        regimes: One closed-system view per regime, in label order; the
            composition hook the structural layer consumes.
        filtered_prob: ``Pr(S_t = m | y_1..t)``, shape ``(nobs, M)``.
        predicted_prob: ``Pr(S_t = m | y_1..t-1)``, shape ``(nobs, M)``.
        smoothed_prob: ``Pr(S_t = m | y_1..T)``, shape ``(nobs, M)``.
        ergodic_prob: Stationary distribution of ``transition``.
        expected_durations: ``1 / (1 - P_mm)``, in periods.
        fittedvalues: ``(nobs, k)`` posterior-weighted one-step means.
        resid: ``(nobs, k)`` residuals against those weighted means.
        llf: Log-likelihood from the final filter pass.
        nobs: Effective sample size ``n - p``.
        n_params: Free parameters: ``M (M - 1)`` transitions, the
            coefficient blocks counted once per regime that switches them,
            and the covariance block(s).
        label_ordering: Which quantity regimes were sorted by, ascending.
        n_iter: EM iterations used by the refining run.
        converged: Whether the refining run met its tolerance.

    See Also:
        * :class:`MarkovSwitchingVAR` -- the model whose ``fit()`` returns
          this record.
        * :class:`MarkovSwitchingSVAR` -- one identification scheme applied
          to every :meth:`regime` view.
        * :class:`~cultivars.univariate.regime_switching.MarkovSwitchingAR`
          -- the scalar family on the same filter and smoother.
        * :class:`~cultivars.multivariate.nonlinear.threshold.TVAR` -- the
          regime computed from an observable instead of inferred.

    References:
        Hamilton, J. D. (1990). Analysis of time series subject to changes
        in regime. *Journal of Econometrics*, 45(1-2), 39-70.

        Kim, C.-J. (1994). Dynamic linear models with Markov-switching.
        *Journal of Econometrics*, 60(1-2), 1-22.

        Krolzig, H.-M. (1997). *Markov-Switching Vector Autoregressions*.
        Springer. Chapters 6 and 9.

    Example:
        Two regimes that differ in their intercepts and scales around a
        common lag block. EM recovers the means, the transition matrix and
        the regime path, and the fitted values are the smoothed-posterior
        mixture of the two regimes' conditional means:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p = np.array([[0.97, 0.03], [0.05, 0.95]])
        >>> mu = np.array([[-1.0, -0.5], [1.5, 1.0]])
        >>> s, y = np.zeros(600, dtype=int), np.zeros((600, 2))
        >>> for t in range(1, 600):
        ...     s[t] = np.searchsorted(np.cumsum(p[s[t - 1]]), rng.random())
        ...     y[t] = mu[s[t]] + 0.3 * y[t - 1] + 0.4 * rng.standard_normal(2)
        >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0, n_init=3)
        >>> res.specification, res.converged, res.nobs, res.n_params
        ('MSIH(2)-VAR(1)', True, 599, 16.0)
        >>> np.vstack([view.deterministic[0] for view in res.regimes]).round(1)
        array([[-1. , -0.5],
               [ 1.5,  1. ]])
        >>> bool(np.abs(res.transition - p).max() < 0.03)
        True
        >>> bool((res.most_likely_regime == s[1:]).mean() > 0.98)
        True
        >>> mixture = sum(
        ...     res.smoothed_prob[:, [m]] * (v.deterministic[0] + y[:-1] @ v.coefficients[0].T)
        ...     for m, v in enumerate(res.regimes)
        ... )
        >>> bool(np.allclose(mixture, res.fittedvalues))
        True
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    """The observed ``(n, k)`` panel as supplied. Kept out of the repr."""

    names: tuple[str, ...]
    """Variable labels, in column order."""

    order: int
    """Autoregressive order within each regime."""

    trend: str
    """The deterministic specification shared by every regime's design."""

    n_regimes: int
    """Number of latent regimes ``M``."""

    switching_mean: bool
    """Whether the deterministic block switches with the chain."""

    switching_ar: bool
    """Whether the lag coefficients switch with the chain."""

    switching_variance: bool
    """Whether the innovation covariance switches with the chain."""

    transition: npt.NDArray[np.float64] = field(repr=False)
    """The row-stochastic ``(M, M)`` transition matrix. Kept out of the repr."""

    regimes: tuple[_RegimeSystemResult, ...] = field(repr=False)
    """One closed linear system per regime, in label order. Kept out of the repr."""

    filtered_prob: npt.NDArray[np.float64] = field(repr=False)
    """``Pr(S_t = m | y_1..t)``, shape ``(nobs, M)``. Kept out of the repr."""

    predicted_prob: npt.NDArray[np.float64] = field(repr=False)
    """``Pr(S_t = m | y_1..t-1)``, shape ``(nobs, M)``. Kept out of the repr."""

    smoothed_prob: npt.NDArray[np.float64] = field(repr=False)
    """``Pr(S_t = m | y_1..T)``, shape ``(nobs, M)``. Kept out of the repr."""

    ergodic_prob: npt.NDArray[np.float64] = field(repr=False)
    """The stationary distribution of :attr:`transition`, shape ``(M,)``. Kept out of the repr."""

    expected_durations: npt.NDArray[np.float64] = field(repr=False)
    """``1 / (1 - P_mm)`` per regime, in periods. Kept out of the repr."""

    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs, k)`` smoothed-posterior mixture of the regimes' one-step means.

    Kept out of the repr.
    """

    resid: npt.NDArray[np.float64] = field(repr=False)
    """``endog[p:] - fittedvalues``, shape ``(nobs, k)``. Kept out of the repr."""

    llf: float
    """Log-likelihood from the final filter pass."""

    nobs: int
    """Effective sample size ``n - p``."""

    n_params: float
    """Free parameters: transitions, the switching and common blocks, the covariance(s)."""

    label_ordering: str
    """The quantity regimes were sorted by, ascending."""

    n_iter: int
    """EM iterations used by the refining run."""

    converged: bool
    """Whether the refining run met its tolerance."""

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorMarkovSwitchingFit,
        model: _MarkovSwitchingVectorAutoRegressionModel[MarkovSwitchingVARResult],
    ) -> MarkovSwitchingVARResult:
        """Assemble the public result, building one stable view per regime.

        Each regime view carries the regime's own conditional-mean residuals
        rescaled by the square root of its smoothed weight (normalized to
        the sample size), so that the view's ``resid.T @ resid / nobs`` is
        the regime's probability-weighted residual covariance.

        Args:
            fit: The EM output: transition matrix, per-regime coefficient
                blocks and covariances, the three posteriors, the mixture
                fitted values and residuals.
            model: The model that produced it, read for the sample, the
                specification and the ordering convention.

        Returns:
            A populated result whose :attr:`regimes` are identity-stable.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> model = MarkovSwitchingVAR(y, order=1, n_regimes=2)
            >>> fit = model._fit_markov(max_iter=500, tol=1e-6, n_init=10, screen_iter=15, seed=0)
            >>> res = MarkovSwitchingVARResult._from_fit(fit, model)
            >>> view = res.regimes[0]
            >>> weighted = view.resid.T @ view.resid / res.nobs
            >>> round(view.nobs), bool(np.allclose(weighted, view.sigma_u, atol=1e-6))
            (199, True)
        """
        target = model.endog[model.order :]
        views: list[_RegimeSystemResult] = []
        for m in range(model.n_regimes):
            weight = fit.smoothed_prob[:, m]
            stack = fit.coefficients[m]
            deterministic = fit.deterministics[m]
            resid_m = target - cls._conditional_mean(model, stack, deterministic)
            total = max(float(weight.sum()), 1e-12)
            scaled = resid_m * np.sqrt(weight * (target.shape[0] / total))[:, None]
            views.append(
                _RegimeSystemResult(
                    index=m,
                    names=model.names,
                    order=model.order,
                    trend=model.trend,
                    coefficients=stack,
                    deterministic=deterministic,
                    sigma_u=fit.sigmas[m],
                    resid=scaled,
                    weight=weight,
                    nobs=float(weight.sum()),
                )
            )
        return cls(
            endog=model.endog,
            names=model.names,
            order=model.order,
            trend=model.trend,
            n_regimes=model.n_regimes,
            switching_mean=model.switching_mean,
            switching_ar=model.switching_ar,
            switching_variance=model.switching_variance,
            transition=fit.transition,
            regimes=tuple(views),
            filtered_prob=fit.filtered_prob,
            predicted_prob=fit.predicted_prob,
            smoothed_prob=fit.smoothed_prob,
            ergodic_prob=fit.ergodic_prob,
            expected_durations=fit.expected_durations,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            label_ordering=model.label_ordering,
            n_iter=fit.n_iter,
            converged=fit.converged,
        )

    @staticmethod
    def _conditional_mean(
        model: _MarkovSwitchingVectorAutoRegressionModel[MarkovSwitchingVARResult],
        stack: npt.NDArray[np.float64],
        deterministic: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """One regime's one-step conditional means over the effective sample.

        Lays the regime's blocks out against the model's ``[deterministic |
        lags]`` design and multiplies through.

        Args:
            model: The model, read for its design matrix.
            stack: The regime's ``(p, k, k)`` lag coefficients.
            deterministic: The regime's ``(d, k)`` deterministic block.

        Returns:
            A ``(nobs, k)`` array of conditional means as if the regime held
            at every date.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> model = MarkovSwitchingVAR(y, order=1, n_regimes=2)
            >>> res = model.fit(seed=0)
            >>> view = res.regimes[1]
            >>> mean = MarkovSwitchingVARResult._conditional_mean(
            ...     model, view.coefficients, view.deterministic
            ... )
            >>> hand = view.deterministic[0] + y[:-1] @ view.coefficients[0].T
            >>> mean.shape, bool(np.allclose(mean, hand))
            ((399, 2), True)
        """
        _, design, _ = model._design()
        d = deterministic.shape[0]
        k, p = len(model.names), model.order
        coef = np.zeros((design.shape[1], k), dtype=np.float64)
        coef[:d] = deterministic
        for i in range(p):
            coef[d + i * k : d + (i + 1) * k] = stack[i].T
        return np.asarray(design @ coef, dtype=np.float64)

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0).k_endog
            2
        """
        return len(self.names)

    @property
    def specification(self) -> str:
        """Krolzig code for which blocks switch, e.g. ``"MSIH(2)-VAR(1)"``.

        ``I`` for a switching intercept, ``A`` for switching lag
        coefficients, ``H`` for a switching covariance, in that order.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0).specification
            'MSIH(2)-VAR(1)'
            >>> model = MarkovSwitchingVAR(y, order=2, n_regimes=2, switching_variance=False)
            >>> model.fit(seed=0).specification
            'MSI(2)-VAR(2)'
        """
        letters = "".join(
            letter
            for letter, switching in (
                ("I", self.switching_mean),
                ("A", self.switching_ar),
                ("H", self.switching_variance),
            )
            if switching
        )
        return f"MS{letters}({self.n_regimes})-VAR({self.order})"

    def regime(self, index: int) -> _RegimeSystemResult:
        """One regime's closed-system view, identity-stable across calls.

        Args:
            index: Regime label in ``0..M-1``, under the fit's ordering
                convention.

        Returns:
            The :class:`_RegimeSystemResult`. The same object every call, so an
            identification result built from it remembers its source by
            identity.

        Raises:
            SpecificationError: If ``index`` is out of range.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> view = res.regime(1)
            >>> view is res.regime(1), view.index, view.is_stable, view.ma_representation(2).shape
            (True, 1, True, (3, 2, 2))
            >>> res.regime(2)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: regime must be in 0..1; got 2.
        """
        if not 0 <= index < self.n_regimes:
            raise SpecificationError(f"regime must be in 0..{self.n_regimes - 1}; got {index}.")
        return self.regimes[index]

    def probabilities(self, kind: ProbabilityType = "smoothed") -> npt.NDArray[np.float64]:
        """Return one of the three regime posteriors.

        Args:
            kind: ``"smoothed"`` conditions on the whole sample and dates
                regimes after the fact; ``"filtered"`` is the real-time view;
                ``"predicted"`` is the one-step-ahead forecast of the state.

        Returns:
            An array of shape ``(nobs, M)`` whose rows sum to one.

        Raises:
            SpecificationError: If ``kind`` is not one of the three.

        Example:
            The smoothed posterior is at least as sharp as the filtered one,
            which is at least as sharp as the predicted one:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> kinds = ("smoothed", "filtered", "predicted")
            >>> sharp = [res.probabilities(kind).max(axis=1).mean() for kind in kinds]
            >>> bool(sharp[0] >= sharp[1] >= sharp[2])
            True
            >>> bool(np.allclose(res.probabilities().sum(axis=1), 1.0))
            True
            >>> res.probabilities("viterbi")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: kind must be one of ('smoothed', ...
        """
        validate_choice(kind, ProbabilityType, "kind")
        return {
            "smoothed": self.smoothed_prob,
            "filtered": self.filtered_prob,
            "predicted": self.predicted_prob,
        }[kind]

    @property
    def most_likely_regime(self) -> npt.NDArray[np.int64]:
        """Pointwise most probable regime under the smoothed posterior.

        The *marginal* MAP state at each date, not the Viterbi path: the
        sequence can contain a one-period switch whose transition probability
        is near zero. Use it to date regimes, not to reason about the
        sequence of switches.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> path = res.most_likely_regime
            >>> path.shape, path.dtype.name, bool((path == s[1:]).mean() > 0.98)
            ((399,), 'int64', True)
        """
        return np.argmax(self.smoothed_prob, axis=1).astype(np.int64)

    @property
    def regime_shares(self) -> npt.NDArray[np.float64]:
        """Average smoothed probability of each regime over the sample.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> res.regime_shares.round(2), bool(np.isclose(res.regime_shares.sum(), 1.0))
            (array([0.5, 0.5]), True)
        """
        return np.asarray(self.smoothed_prob.mean(axis=0), dtype=np.float64)

    @property
    def regime_uncertainty(self) -> float:
        """Mean posterior entropy, normalized so ``0`` is sharp and ``1`` flat.

        Near zero, the regimes are cleanly separated; near one, the data
        barely distinguish them at all -- which no coefficient table shows.

        Example:
            Well-separated means give a sharp posterior:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> bool(res.regime_uncertainty < 0.05)
            True
        """
        p = np.clip(self.smoothed_prob, 1e-300, None)
        entropy = -(p * np.log(p)).sum(axis=1)
        return float(entropy.mean() / np.log(self.n_regimes))

    @property
    def is_regimewise_stationary(self) -> bool:
        """Whether every regime is stable read as a linear VAR.

        Sufficient for stationarity of the switching process but not
        necessary: one explosive regime is admissible if it is visited rarely
        and exited fast, a joint condition on the roots and the chain. A
        ``False`` here is a prompt to check that regime's expected duration.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> res.is_regimewise_stationary, [view.is_stable for view in res.regimes]
            (True, [True, True])
        """
        return all(view.is_stable for view in self.regimes)

    def _likelihood_ratio_obstacle(self, counterpart: _ComparisonMixin) -> str | None:
        """Block only the comparison that changes the number of regimes.

        A test holding ``M`` fixed -- dropping a switching block, say -- is
        perfectly valid; testing ``M`` itself is not, because under the
        smaller model the extra regime's parameters are unidentified and its
        transition probabilities sit on the boundary. A non-switching result
        counts as the one-regime case, which is exactly the classic invalid
        test.

        Args:
            counterpart: The other result in the comparison.

        Returns:
            ``None`` when the test may proceed, else the refusal message.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> full = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> model = MarkovSwitchingVAR(y, order=1, n_regimes=2, switching_variance=False)
            >>> same_m = model.fit(seed=0)
            >>> full._likelihood_ratio_obstacle(same_m) is None
            True
            >>> full._likelihood_ratio_obstacle(VAR(y, order=1).fit())[:60]
            'a chi-squared likelihood-ratio test cannot compare 2 regimes'
            >>> same_m.likelihood_ratio_test(full).df
            3
        """
        other = counterpart.n_regimes if isinstance(counterpart, MarkovSwitchingVARResult) else 1
        if other == self.n_regimes:
            return None
        return (
            f"a chi-squared likelihood-ratio test cannot compare {self.n_regimes} "
            f"regimes against {other}: under the smaller model the extra "
            "regime's parameters are unidentified and its transition "
            "probabilities lie on the boundary, so the statistic has no "
            "chi-squared limit. Use a parametric bootstrap instead. Tests "
            "that hold the number of regimes fixed are still available."
        )

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> res._comparison_label() == res.specification
            True
        """
        return self.specification

    def regime_table(self) -> SummaryTable:
        """One row per regime: scale, persistence, and how much sample it owns.

        Returns:
            A :class:`~cultivars.summary.SummaryTable` with columns for the
            covariance log-determinant, the largest companion root, the
            ergodic probability, the expected duration and the realized
            smoothed share.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> table = res.regime_table()
            >>> table.title, table.columns[:3], len(table.rows)
            ('MSIH(2)-VAR(1) regimes', ('regime', 'log|Sigma|', 'max |root|'), 2)
            >>> dict(table.metadata)["Ordering"]
            'ascending first-variable intercept'
        """
        shares = self.regime_shares
        return SummaryTable(
            title=f"{self.specification} regimes",
            metadata=(
                ("Ordering", f"ascending {self.label_ordering}"),
                ("Observations", f"{self.nobs}"),
            ),
            columns=("regime", "log|Sigma|", "max |root|", "ergodic", "duration", "share"),
            rows=tuple(
                (
                    f"{m}",
                    f"{np.linalg.slogdet(view.sigma_u)[1]:.4f}",
                    f"{view.stability_check().max_modulus:.4f}",
                    f"{self.ergodic_prob[m]:.4f}",
                    f"{self.expected_durations[m]:.2f}",
                    f"{shares[m]:.4f}",
                )
                for m, view in enumerate(self.regimes)
            ),
            notes=(
                "'ergodic' is the stationary probability implied by the "
                "transition matrix; 'share' is the average smoothed "
                "probability actually realized in this sample.",
            ),
        )

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        The transition matrix as the one coefficient row, the criteria and
        convergence in the metadata, and notes on regime-wise stability, the
        ordering convention, posterior sharpness, the per-regime composition
        hook and the refused regime-count test; a non-convergence warning
        goes first when the refining run did not meet its tolerance.
        Per-regime coefficients are in :meth:`regime_table` and on each
        :meth:`regime` view.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> table = res._summary_table()
            >>> table.title, table.rows[0][0]
            ('MSIH(2)-VAR(1) Results', 'transition')
            >>> table.columns
            ('', 'p[0->0]', 'p[0->1]', 'p[1->0]', 'p[1->1]')
            >>> dict(table.metadata)["Regimes"], len(table.notes), table.notes[0][:25]
            ('2', 5, 'Regime-wise stable: True;')
        """
        ic: InformationCriteria = self.information_criteria
        notes = [
            f"Regime-wise stable: {self.is_regimewise_stationary}; per-regime "
            "roots are in regime_table(). Regime-wise stability is sufficient "
            "but not necessary for stationarity of the switching process.",
            f"Regimes ordered by ascending {self.label_ordering}; the mixture "
            "likelihood is invariant to relabelling, so the convention is "
            "what makes two runs comparable.",
            f"Posterior sharpness: normalized regime entropy = "
            f"{self.regime_uncertainty:.3f} (0 sharp, 1 flat). Every fitted "
            "value and residual is an expectation under the smoothed "
            "posterior, not a fact about the data.",
            "Each regime is a closed linear system conditional on the chain: "
            "regime(m) is accepted by every identification model in "
            "cultivars.multivariate.structural, and MarkovSwitchingSVAR "
            "applies one scheme across all regimes.",
            "The number of regimes is not testable by a likelihood ratio; "
            "comparisons that change M are refused.",
        ]
        if not self.converged:
            notes.insert(
                0,
                f"EM did NOT converge in {self.n_iter} iterations; treat every "
                "number here as provisional.",
            )
        return SummaryTable(
            title=f"{self.specification} Results",
            metadata=(
                ("Model", self.specification),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Variables", f"{self.k_endog}"),
                ("AIC", f"{ic.aic:.3f}"),
                ("Regimes", f"{self.n_regimes}"),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
                ("Converged", f"{self.converged} ({self.n_iter} iter)"),
                ("Trend", self.trend),
            ),
            columns=(
                "",
                *(f"p[{i}->{j}]" for i in range(self.n_regimes) for j in range(self.n_regimes)),
            ),
            rows=(
                (
                    "transition",
                    *(
                        f"{self.transition[i, j]:.4f}"
                        for i in range(self.n_regimes)
                        for j in range(self.n_regimes)
                    ),
                ),
            ),
            notes=tuple(notes),
        )


class MarkovSwitchingVAR(_MarkovSwitchingVectorAutoRegressionModel[MarkovSwitchingVARResult]):
    """Markov-switching vector autoregression with ``M`` latent regimes.

    The constructor is inherited from
    ``_MarkovSwitchingVectorAutoRegressionModel``: it validates the panel
    and the VAR specification as the linear model does, then the regime
    count and the switching flags. At least one block must switch, and a
    switching mean needs a deterministic block to switch.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order within each regime.
        n_regimes: Number of latent regimes ``M``, at least two.
        switching_mean: Whether the deterministic block switches.
        switching_variance: Whether the innovation covariance switches.
        switching_ar: Whether the lag coefficients switch. Off by default:
            switching the lag block multiplies the parameter count by ``M``
            and is rarely what identifies the regimes.
        trend: Deterministic terms per regime.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Raises:
        SpecificationError: If no block switches, if ``switching_mean`` is
            requested with ``trend="n"``, or if the order, trend or names
            are malformed.
        DimensionError: If the sample cannot support ``M`` regimes of the
            specification.

    Attributes:
        _endog: The validated ``(nobs, k)`` panel.
        _names: The resolved variable labels.
        _order: The autoregressive order.
        _trend: The validated trend code.
        _prior: Always ``_NoPrior``; the EM has no shrinkage path.
        _m: The number of regimes.
        _sw_mean: Whether the deterministic block switches.
        _sw_ar: Whether the lag block switches.
        _sw_var: Whether the covariance switches.

    See Also:
        * :class:`MarkovSwitchingVARResult` -- the record ``fit()`` returns.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
          -- the one-regime model, and the first screening start.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p = np.array([[0.97, 0.03], [0.05, 0.95]])
        >>> mu = np.array([[-1.0, -0.5], [1.5, 1.0]])
        >>> s = np.zeros(600, dtype=int)
        >>> y = np.zeros((600, 2))
        >>> for t in range(1, 600):
        ...     s[t] = np.searchsorted(np.cumsum(p[s[t - 1]]), rng.random())
        ...     y[t] = mu[s[t]] + 0.3 * y[t - 1] + 0.4 * rng.standard_normal(2)
        >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0, n_init=3)
        >>> bool(res.regimes[0].deterministic[0, 0] < res.regimes[1].deterministic[0, 0])
        True
        >>> MarkovSwitchingVAR(
        ...     y, order=1, switching_mean=False, switching_variance=False
        ... )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: at least one of switching_mean, ...
    """

    __slots__ = ()

    def fit(
        self,
        *,
        max_iter: int = 500,
        tol: float = 1e-6,
        n_init: int = 10,
        screen_iter: int = 15,
        seed: int | np.random.Generator | None = None,
    ) -> MarkovSwitchingVARResult:
        """Estimate by EM with multi-start screening.

        Unlike the other multivariate families, ``fit`` takes arguments,
        because the likelihood is multimodal and the answer genuinely depends
        on where the search starts. The default screens ``n_init`` starts for
        ``screen_iter`` iterations each and refines only the best; pass a
        ``seed`` when the fit must be reproducible.

        Args:
            max_iter: Iteration cap for the refining run.
            tol: Convergence tolerance on the log-likelihood increment.
            n_init: Starts to screen, the first being the linear fit split
                along whichever block switches.
            screen_iter: Iterations used to score each screening start.
            seed: Seed or generator for the random starts.

        Returns:
            The fitted :class:`MarkovSwitchingVARResult`, regimes ordered by the
            convention :attr:`MarkovSwitchingVARResult.label_ordering` names.

        Raises:
            NumericalError: If every start fails to produce a finite
                likelihood.

        Example:
            The same seed gives the same fit. Screening matters even on a
            cleanly separated problem: the deterministic first start alone
            converges to a local optimum in which one persistent regime
            absorbs the mean shift, more than a hundred log-points below
            the screened fit:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> model = MarkovSwitchingVAR(y, order=1, n_regimes=2)
            >>> one, two = model.fit(seed=0), model.fit(seed=0)
            >>> one.llf == two.llf, one.converged, one.regime_shares.round(2)
            (True, True, array([0.5, 0.5]))
            >>> single = model.fit(seed=0, n_init=1)
            >>> single.converged, bool(single.llf < one.llf - 100), single.regime_shares.round(2)
            (True, True, array([0.92, 0.08]))
        """
        return MarkovSwitchingVARResult._from_fit(
            self._fit_markov(
                max_iter=max_iter,
                tol=tol,
                n_init=n_init,
                screen_iter=screen_iter,
                seed=seed,
            ),
            self,
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MarkovSwitchingSVARResult(_SummaryMixin):
    r"""An identified Markov-switching system, one structural result per regime.

    Conditional on regime :math:`m` the reduced form is a linear VAR with
    innovation covariance :math:`\Sigma_m`, so an identification is a
    factorization :math:`\Sigma_m = B_m B_m'` under restrictions that hold
    in every regime:

    .. math::

       u_t = B_{S_t}\, \varepsilon_t, \qquad \varepsilon_t \sim (0, I).

    Composition on both sides: :attr:`msvar` is the reduced form with its
    chain, posteriors and regime views, and :attr:`structurals` holds each
    regime's identification with its whole surface -- impact matrix,
    responses, diagnostics. This object owns the join: the guarantee of a
    common identifying declaration, and per-regime answers under one roof.

    Note:
        Every structural quantity here is *within-regime*. :meth:`irf`
        holds the chain frozen in the chosen regime, which is the same
        linearization the observed-regime families use; a response over a
        horizon longer than that regime's expected duration is an
        extrapolation, and the summary says so. Responses that integrate
        over the chain's future path are not offered.

    Attributes:
        msvar: The fitted Markov-switching reduced form.
        structurals: One structural result per regime, in label order.

    See Also:
        * :class:`MarkovSwitchingSVAR` -- the model whose ``identify()``
          returns this record.
        * :class:`MarkovSwitchingVARResult` -- the reduced form, with
          ``regime_table()`` for the durations to read responses against.
        * :class:`~cultivars.multivariate.structural.zero_restrictions.RecursiveSVAR`
          -- the default per-regime scheme.

    References:
        Sims, C. A., Waggoner, D. F., & Zha, T. (2008). Methods for
        inference in large multiple-equation Markov-switching models.
        *Journal of Econometrics*, 146(2), 255-274.

    Example:
        A recursive identification applied to both regimes of a fitted
        MS-VAR. Each regime's impact matrix factorizes its own covariance,
        and the shock labels are shared:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> s = np.repeat([0, 1, 0, 1], 100)
        >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
        >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2, names=("a", "b")).fit(seed=0)
        >>> svar = MarkovSwitchingSVAR(res).identify()
        >>> svar.n_regimes, svar.names, svar.shock_names
        (2, ('a', 'b'), ('a', 'b'))
        >>> all(
        ...     np.allclose(svar.impact(m) @ svar.impact(m).T, res.regime(m).sigma_u)
        ...     for m in range(2)
        ... )
        True
        >>> svar.irf(4, regime=1).shape, svar.structural(0).source is res.regime(0)
        ((5, 2, 2), True)
    """

    msvar: MarkovSwitchingVARResult = field(repr=False)
    """The fitted reduced form whose regime views were identified. Kept out of the repr."""

    structurals: tuple[StructuralResult, ...] = field(repr=False)
    """One structural result per regime, in the reduced form's label order. Kept out of the repr."""

    @property
    def n_regimes(self) -> int:
        """Number of regimes.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> MarkovSwitchingSVAR(res).identify().n_regimes
            2
        """
        return self.msvar.n_regimes

    @property
    def names(self) -> tuple[str, ...]:
        """Variable labels, in column order.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2, names=("a", "b")).fit(seed=0)
            >>> MarkovSwitchingSVAR(res).identify().names
            ('a', 'b')
        """
        return self.msvar.names

    @property
    def shock_names(self) -> tuple[str, ...]:
        """Labels of the identified shocks, common across regimes.

        Read from the first regime's result when its scheme names its
        shocks; ``shock1 ... shockk`` otherwise.

        Example:
            Under the recursive scheme the shocks carry the ordering's
            names:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2, names=("a", "b")).fit(seed=0)
            >>> MarkovSwitchingSVAR(res, order=("b", "a")).identify().shock_names
            ('b', 'a')
        """
        names = getattr(self.structurals[0], "shock_names", None)
        if names is None:
            return tuple(f"shock{j + 1}" for j in range(len(self.names)))
        return tuple(names)

    def structural(self, regime: int) -> StructuralResult:
        """One regime's structural result, with its full scheme surface.

        Args:
            regime: Regime label in ``0..M-1``.

        Returns:
            The :class:`StructuralResult` identified from that regime.

        Raises:
            SpecificationError: If ``regime`` is out of range.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> svar = MarkovSwitchingSVAR(res).identify()
            >>> one = svar.structural(1)
            >>> type(one).__name__, one.scheme, one.fevd(4).shape
            ('SVARResult', 'recursive', (5, 2, 2))
            >>> svar.structural(2)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: regime must be in 0..1; got 2.
        """
        if not 0 <= regime < self.n_regimes:
            raise SpecificationError(f"regime must be in 0..{self.n_regimes - 1}; got {regime}.")
        return self.structurals[regime]

    def impact(self, regime: int) -> npt.NDArray[np.float64]:
        """One regime's structural impact matrix.

        Args:
            regime: Regime label in ``0..M-1``.

        Returns:
            The ``(k, s)`` impact matrix of that regime's identification.

        Raises:
            SpecificationError: If ``regime`` is out of range.

        Example:
            The recursive impact is lower-triangular in every regime:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> svar = MarkovSwitchingSVAR(res).identify()
            >>> [float(svar.impact(m)[0, 1]) for m in range(2)]
            [0.0, 0.0]
        """
        return self.structural(regime).irf(0)[0]

    def irf(
        self, horizon: int = 20, *, regime: int, cumulative: bool = False
    ) -> npt.NDArray[np.float64]:
        """Structural impulse responses within one regime, held frozen.

        Freezing the regime is the same linearization it is for the
        observed-regime families, with the latent twist stated plainly: the
        response assumes the chain *stays* in this regime over the horizon,
        so read it against that regime's expected duration -- a 20-period
        response in a regime that lasts 5 is an extrapolation.

        Args:
            horizon: Largest lead to return.
            regime: Which regime's identified system to propagate.
            cumulative: Return running sums.

        Returns:
            An array of shape ``(horizon + 1, k, s)``; entry ``[h, i, j]`` is
            the response of variable ``i`` at lead ``h`` to identified shock
            ``j``, in that regime.

        Raises:
            SpecificationError: If ``regime`` is out of range.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> svar = MarkovSwitchingSVAR(res).identify()
            >>> plain, running = svar.irf(3, regime=0), svar.irf(3, regime=0, cumulative=True)
            >>> plain.shape, bool(np.allclose(plain[0], svar.impact(0)))
            ((4, 2, 2), True)
            >>> bool(np.allclose(running, np.cumsum(plain, axis=0)))
            True
        """
        return self.structural(regime).irf(horizon, cumulative=cumulative)

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        One row per regime holding the absolute impact diagonal of each
        identified shock, so the table shows how the shocks' sizes move with
        the regime; the scheme's own restriction note first, then the
        common-restriction, frozen-regime and label-ordering notes.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2, names=("a", "b")).fit(seed=0)
            >>> table = MarkovSwitchingSVAR(res).identify()._summary_table()
            >>> table.title, table.columns
            ('MS-SVAR (recursive) Results', ('impact |diag| of', 'a', 'b'))
            >>> [row[0] for row in table.rows]
            ['regime 0', 'regime 1']
            >>> dict(table.metadata)["Reduced form"], len(table.notes)
            ('MSIH(2)-VAR(1)', 4)
        """
        scheme = getattr(self.structurals[0], "scheme", "supplied")
        restriction = getattr(self.structurals[0], "restriction", "")
        shocks = self.shock_names
        rows = tuple(
            (
                f"regime {m}",
                *(f"{np.abs(np.diag(self.impact(m)))[j]:.4f}" for j in range(len(shocks))),
            )
            for m in range(self.n_regimes)
        )
        notes = [
            restriction,
            "The identifying restriction is common across regimes; the "
            "parameters it is applied to switch. Per-regime impact diagonals "
            "above show how the identified shocks' sizes move with the "
            "regime.",
            "irf(regime=m) holds the chain frozen in regime m, so read it "
            "against that regime's expected duration (regime_table() on the "
            "reduced form).",
            "Regime labels follow the reduced form's ordering convention "
            f"({self.msvar.label_ordering}); the likelihood is invariant to "
            "relabelling.",
        ]
        return SummaryTable(
            title=f"MS-SVAR ({scheme}) Results",
            metadata=(
                ("Scheme", str(scheme)),
                ("Regimes", f"{self.n_regimes}"),
                ("Identified shocks", f"{len(shocks)}"),
                ("Reduced form", self.msvar.specification),
                ("Observations", f"{self.msvar.nobs}"),
                ("Converged", f"{self.msvar.converged}"),
            ),
            columns=("impact |diag| of", *shocks),
            rows=rows,
            notes=tuple(note for note in notes if note),
        )


class MarkovSwitchingSVAR:
    """Per-regime structural identification of an MS-VAR, Sims-Waggoner-Zha.

    Constructs with a fitted :class:`MarkovSwitchingVAR` result. Not an
    ``_IdentificationModel``: that contract is one closed system, and an
    MS-VAR has ``M`` of them -- the regime views, each of which any scheme
    in this package already accepts directly. What this model adds is the
    discipline of one declaration for all regimes, and a packaged
    per-regime answer. Two routes: pass nothing (or ``order``) and every
    regime is factorized recursively under one ordering; or identify each
    ``msvar.regime(m)`` yourself with any point-identified scheme and pass
    the results, which are checked by identity against the views they must
    have come from.

    Attributes:
        _msvar: The fitted reduced form.
        _structurals: The supplied per-regime identifications, or ``None``
            for the recursive route.
        _order: The recursive ordering, or ``None`` for ``names`` order.

    See Also:
        * :class:`MarkovSwitchingSVARResult` -- the record ``identify()``
          returns.
        * :class:`~cultivars.multivariate.structural.zero_restrictions.RecursiveSVAR`
          -- the scheme applied per regime by default.

    References:
        Sims, C. A., Waggoner, D. F., & Zha, T. (2008). Methods for
        inference in large multiple-equation Markov-switching models.
        *Journal of Econometrics*, 146(2), 255-274.

    Example:
        >>> import numpy as np
        >>> from cultivars.multivariate.regime_switching.markov_switching import MarkovSwitchingVAR
        >>> rng = np.random.default_rng(0)
        >>> p = np.array([[0.97, 0.03], [0.05, 0.95]])
        >>> mu = np.array([[-1.0, -0.5], [1.5, 1.0]])
        >>> s = np.zeros(600, dtype=int)
        >>> y = np.zeros((600, 2))
        >>> for t in range(1, 600):
        ...     s[t] = np.searchsorted(np.cumsum(p[s[t - 1]]), rng.random())
        ...     y[t] = mu[s[t]] + 0.3 * y[t - 1] + 0.4 * rng.standard_normal(2)
        >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0, n_init=3)
        >>> svar = MarkovSwitchingSVAR(res).identify()
        >>> svar.impact(0).shape
        (2, 2)

        The packaging route takes identifications built from this fit's
        own regime views, and refuses any other:

        >>> from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR
        >>> mine = [RecursiveSVAR(res.regime(m)).identify() for m in range(2)]
        >>> bool(np.allclose(MarkovSwitchingSVAR(res, mine).identify().impact(1), svar.impact(1)))
        True
        >>> other = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=1, n_init=3)
        >>> theirs = [RecursiveSVAR(other.regime(m)).identify() for m in range(2)]
        >>> MarkovSwitchingSVAR(res, theirs)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: structurals[0] was not built from this MS-VAR's ...
    """

    __slots__ = ("_msvar", "_order", "_structurals")

    def __init__(
        self,
        msvar: MarkovSwitchingVARResult,
        structurals: Sequence[StructuralResult] | None = None,
        *,
        order: Sequence[str] | None = None,
    ) -> None:
        """Validate the reduced form and the identification route.

        Args:
            msvar: The fitted Markov-switching reduced form.
            structurals: Optional per-regime identifications to package, one
                per regime in label order, each produced by a point-identified
                model applied to the corresponding ``msvar.regime(m)``. When
                given, ``order`` must be omitted.
            order: The recursive ordering for the default scheme, applied
                identically in every regime. ``None`` orders the system as
                ``names`` stands.

        Raises:
            SpecificationError: If ``msvar`` is not a fitted MS-VAR result,
                both ``structurals`` and ``order`` are given, the count is
                wrong, or a supplied identification was not built from this
                MS-VAR's own regime view.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> MarkovSwitchingSVAR(res.regime(0))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: MarkovSwitchingSVAR constructs with a ...
            >>> from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR
            >>> mine = [RecursiveSVAR(res.regime(m)).identify() for m in range(2)]
            >>> MarkovSwitchingSVAR(res, mine[:1])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: structurals must supply one identification ...
            >>> MarkovSwitchingSVAR(res, mine, order=("y2", "y1"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: pass either per-regime identifications to ...
        """
        if not isinstance(msvar, MarkovSwitchingVARResult):
            raise SpecificationError(
                "MarkovSwitchingSVAR constructs with a fitted MarkovSwitchingVAR result; "
                f"got {type(msvar).__name__}. Fit "
                "cultivars.multivariate.regime_switching.MarkovSwitchingVAR first."
            )
        self._msvar = msvar
        if structurals is not None:
            if order is not None:
                raise SpecificationError(
                    "pass either per-regime identifications to package or an "
                    "ordering for the default recursive scheme, not both."
                )
            resolved = tuple(structurals)
            if len(resolved) != msvar.n_regimes:
                raise SpecificationError(
                    f"structurals must supply one identification per regime "
                    f"({msvar.n_regimes}); got {len(resolved)}."
                )
            for m, structural in enumerate(resolved):
                if structural.source is not msvar.regimes[m]:
                    raise SpecificationError(
                        f"structurals[{m}] was not built from this MS-VAR's "
                        f"own regime {m} view; identify msvar.regime({m}) and "
                        "pass what that returns, in label order."
                    )
            self._structurals: tuple[StructuralResult, ...] | None = resolved
        else:
            self._structurals = None
        self._order = None if order is None else tuple(str(name) for name in order)

    def identify(self) -> MarkovSwitchingSVARResult:
        """Identify every regime under one declaration and package the answers.

        Returns:
            The per-regime structural result. When no identifications were
            supplied, each regime is factorized recursively under the same
            ordering -- the restriction pattern held fixed while the
            parameters switch.

        Raises:
            SpecificationError: If a declared ordering is not a permutation
                of ``names``.
            NumericalError: If a regime's innovation covariance is not
                positive definite.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = np.repeat([0, 1, 0, 1], 100)
            >>> y = np.array([[-1.0, -0.5], [1.5, 1.0]])[s] + 0.4 * rng.standard_normal((400, 2))
            >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2, names=("a", "b")).fit(seed=0)
            >>> svar = MarkovSwitchingSVAR(res, order=("b", "a")).identify()
            >>> svar.shock_names, [float(svar.impact(m)[1, 1]) for m in range(2)]
            (('b', 'a'), [0.0, 0.0])
            >>> MarkovSwitchingSVAR(res, order=("b", "c")).identify()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: order must be a permutation of the variable ...
        """
        structurals: tuple[StructuralResult, ...] = (
            self._structurals
            if self._structurals is not None
            else tuple(
                RecursiveSVAR(view, order=self._order).identify() for view in self._msvar.regimes
            )
        )
        return MarkovSwitchingSVARResult(msvar=self._msvar, structurals=structurals)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MarkovSwitchingDFMResult(_SummaryMixin):
    r"""A fitted switching factor model: the factor, and the regime chronology.

    The Kim-Nelson model on a standardized panel :math:`x_t`,

    .. math::

       x_{it} = \lambda_i f_t + e_{it},
       \qquad
       f_t = \mu_{S_t} + \phi f_{t-1} + v_t,
       \qquad
       e_{it} \sim N(0, \psi_i), \quad v_t \sim N(0, \sigma_v^2),

    with a two-state chain on the factor's intercept and the first loading
    fixed at one. The record carries the parameters on the standardized
    scale, the filtered factor, both regime posteriors, and the
    standardization that was removed, so the panel can be reconstructed.

    Note:
        :attr:`llf` is the Kim (1994) collapse approximation to the exact
        likelihood -- after each update the regime-conditional state
        distributions are collapsed to one Gaussian per regime -- and it
        was maximized as such. It is comparable across fits of this model
        on the same panel and not against an exact likelihood. The factor
        path is the *filtered* one; the regime probabilities are
        available both filtered and smoothed.

    Attributes:
        panel: The observed ``(nobs, n_series)`` panel.
        series_names: One label per series.
        loadings: ``(n_series,)`` factor loadings on the standardized
            panel; the first is one by the identification convention.
        idiosyncratic_var: ``(n_series,)`` idiosyncratic variances on the
            standardized scale.
        phi: Factor persistence.
        sigma_v: Factor innovation standard deviation.
        mu: ``(2,)`` regime intercepts of the factor, low state first.
        regime_transition: ``(2, 2)`` row-stochastic chain matrix.
        factor: ``(nobs,)`` filtered factor estimate.
        filtered_prob: ``(nobs, 2)`` filtered regime probabilities.
        smoothed_prob: ``(nobs, 2)`` smoothed regime probabilities.
        means: ``(n_series,)`` series means removed before estimation.
        scales: ``(n_series,)`` series standard deviations divided out.
        llf: The maximized Kim-approximate log-likelihood -- the collapse
            approximation, reported under that name.
        n_params: Free parameters the likelihood was maximized over.
        converged: Whether the optimizer reported convergence.

    See Also:
        * :class:`MarkovSwitchingDFM` -- the model whose ``fit()`` returns
          this record.
        * :class:`~cultivars.multivariate.large_dim.dynamic_factor.DFM`
          -- the factor model without the switch.

    References:
        Kim, C.-J., & Nelson, C. R. (1998). Business cycle turning points, a
        new coincident index, and tests of duration dependence based on a
        dynamic factor model with regime switching. *Review of Economics
        and Statistics*, 80(2), 188-201.

        Kim, C.-J. (1994). Dynamic linear models with Markov-switching.
        *Journal of Econometrics*, 60(1-2), 1-22.

    Example:
        A hand-constructed record on a flat panel shows the accessors
        without a fit; the fitted example is on :class:`MarkovSwitchingDFM`:

        >>> import numpy as np
        >>> res = MarkovSwitchingDFMResult(
        ...     panel=np.zeros((80, 3)), series_names=("ip", "emp", "sales"),
        ...     loadings=np.array([1.0, 0.8, 1.2]), idiosyncratic_var=np.full(3, 0.3),
        ...     phi=0.6, sigma_v=0.5, mu=np.array([-1.5, 0.5]),
        ...     regime_transition=np.array([[0.8, 0.2], [0.05, 0.95]]),
        ...     factor=np.zeros(80), filtered_prob=np.full((80, 2), 0.5),
        ...     smoothed_prob=np.tile([0.1, 0.9], (80, 1)), means=np.zeros(3),
        ...     scales=np.ones(3), llf=-100.0, n_params=11, converged=True,
        ... )
        >>> res.n_series, res.nobs, res.expected_durations().round(1)
        (3, 80, array([ 5., 20.]))
        >>> res.recession_probabilities().shape, float(res.recession_probabilities()[0])
        ((80,), 0.1)
    """

    panel: npt.NDArray[np.float64] = field(repr=False)
    """The observed ``(nobs, n_series)`` panel as supplied, unstandardized. Kept out of the repr."""

    series_names: tuple[str, ...]
    """One label per series, in column order."""

    loadings: npt.NDArray[np.float64] = field(repr=False)
    """``(n_series,)`` loadings on the standardized panel, the first fixed at one.

    Kept out of the repr.
    """

    idiosyncratic_var: npt.NDArray[np.float64] = field(repr=False)
    """``(n_series,)`` idiosyncratic variances on the standardized scale. Kept out of the repr."""

    phi: float
    """Factor persistence."""

    sigma_v: float
    """Factor innovation standard deviation."""

    mu: npt.NDArray[np.float64]
    """``(2,)`` regime intercepts of the factor, low state first."""

    regime_transition: npt.NDArray[np.float64] = field(repr=False)
    """``(2, 2)`` row-stochastic transition matrix, low state first. Kept out of the repr."""

    factor: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs,)`` filtered factor estimate. Kept out of the repr."""

    filtered_prob: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, 2)`` filtered regime probabilities. Kept out of the repr."""

    smoothed_prob: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, 2)`` smoothed regime probabilities. Kept out of the repr."""

    means: npt.NDArray[np.float64] = field(repr=False)
    """``(n_series,)`` series means removed before estimation. Kept out of the repr."""

    scales: npt.NDArray[np.float64] = field(repr=False)
    """``(n_series,)`` series standard deviations divided out before estimation.

    Kept out of the repr.
    """

    llf: float
    """The maximized Kim-approximate log-likelihood."""

    n_params: int
    """Free parameters the likelihood was maximized over: ``2 n_series + 5``."""

    converged: bool
    """Whether the optimizer reported convergence."""

    @property
    def n_series(self) -> int:
        """Number of series in the panel.

        Example:
            >>> import numpy as np
            >>> res = MarkovSwitchingDFMResult(
            ...     panel=np.zeros((80, 3)), series_names=("ip", "emp", "sales"),
            ...     loadings=np.array([1.0, 0.8, 1.2]), idiosyncratic_var=np.full(3, 0.3),
            ...     phi=0.6, sigma_v=0.5, mu=np.array([-1.5, 0.5]),
            ...     regime_transition=np.array([[0.8, 0.2], [0.05, 0.95]]),
            ...     factor=np.zeros(80), filtered_prob=np.full((80, 2), 0.5),
            ...     smoothed_prob=np.tile([0.1, 0.9], (80, 1)), means=np.zeros(3),
            ...     scales=np.ones(3), llf=-100.0, n_params=11, converged=True,
            ... )
            >>> res.n_series
            3
        """
        return len(self.series_names)

    @property
    def nobs(self) -> int:
        """Panel length.

        Example:
            >>> import numpy as np
            >>> res = MarkovSwitchingDFMResult(
            ...     panel=np.zeros((80, 3)), series_names=("ip", "emp", "sales"),
            ...     loadings=np.array([1.0, 0.8, 1.2]), idiosyncratic_var=np.full(3, 0.3),
            ...     phi=0.6, sigma_v=0.5, mu=np.array([-1.5, 0.5]),
            ...     regime_transition=np.array([[0.8, 0.2], [0.05, 0.95]]),
            ...     factor=np.zeros(80), filtered_prob=np.full((80, 2), 0.5),
            ...     smoothed_prob=np.tile([0.1, 0.9], (80, 1)), means=np.zeros(3),
            ...     scales=np.ones(3), llf=-100.0, n_params=11, converged=True,
            ... )
            >>> res.nobs
            80
        """
        return int(self.panel.shape[0])

    def recession_probabilities(self) -> npt.NDArray[np.float64]:
        """Smoothed probability of the low-mean regime, per period.

        The business-cycle chronology: with the factor built from
        pro-cyclical indicators, the low-mean state is the contraction.

        Returns:
            ``smoothed_prob[:, 0]``, shape ``(nobs,)``.

        Example:
            >>> import numpy as np
            >>> res = MarkovSwitchingDFMResult(
            ...     panel=np.zeros((80, 3)), series_names=("ip", "emp", "sales"),
            ...     loadings=np.array([1.0, 0.8, 1.2]), idiosyncratic_var=np.full(3, 0.3),
            ...     phi=0.6, sigma_v=0.5, mu=np.array([-1.5, 0.5]),
            ...     regime_transition=np.array([[0.8, 0.2], [0.05, 0.95]]),
            ...     factor=np.zeros(80), filtered_prob=np.full((80, 2), 0.5),
            ...     smoothed_prob=np.tile([0.1, 0.9], (80, 1)), means=np.zeros(3),
            ...     scales=np.ones(3), llf=-100.0, n_params=11, converged=True,
            ... )
            >>> bool(np.allclose(res.recession_probabilities(), 0.1))
            True
        """
        return self.smoothed_prob[:, 0]

    def expected_durations(self) -> npt.NDArray[np.float64]:
        """Expected regime durations ``1 / (1 - p_jj)``, low state first.

        Returns:
            A ``(2,)`` array in periods.

        Example:
            >>> import numpy as np
            >>> res = MarkovSwitchingDFMResult(
            ...     panel=np.zeros((80, 3)), series_names=("ip", "emp", "sales"),
            ...     loadings=np.array([1.0, 0.8, 1.2]), idiosyncratic_var=np.full(3, 0.3),
            ...     phi=0.6, sigma_v=0.5, mu=np.array([-1.5, 0.5]),
            ...     regime_transition=np.array([[0.8, 0.2], [0.05, 0.95]]),
            ...     factor=np.zeros(80), filtered_prob=np.full((80, 2), 0.5),
            ...     smoothed_prob=np.tile([0.1, 0.9], (80, 1)), means=np.zeros(3),
            ...     scales=np.ones(3), llf=-100.0, n_params=11, converged=True,
            ... )
            >>> res.expected_durations().round(1)
            array([ 5., 20.])
        """
        staying = np.diag(self.regime_transition)
        return np.asarray(1.0 / np.maximum(1.0 - staying, 1e-12), dtype=np.float64)

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        One row per series with its loading and idiosyncratic variance;
        notes with the factor dynamics and regime intercepts, the expected
        durations and the low-state share, the likelihood approximation,
        the identification convention and the chronology accessor.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> res = MarkovSwitchingDFMResult(
            ...     panel=np.zeros((80, 3)), series_names=("ip", "emp", "sales"),
            ...     loadings=np.array([1.0, 0.8, 1.2]), idiosyncratic_var=np.full(3, 0.3),
            ...     phi=0.6, sigma_v=0.5, mu=np.array([-1.5, 0.5]),
            ...     regime_transition=np.array([[0.8, 0.2], [0.05, 0.95]]),
            ...     factor=np.zeros(80), filtered_prob=np.full((80, 2), 0.5),
            ...     smoothed_prob=np.tile([0.1, 0.9], (80, 1)), means=np.zeros(3),
            ...     scales=np.ones(3), llf=-100.0, n_params=11, converged=True,
            ... )
            >>> table = res._summary_table()
            >>> table.title, table.rows[1], dict(table.metadata)["Parameters"]
            ('Markov-Switching Dynamic Factor Results', ('emp', '0.8000', '0.3000'), '11')
            >>> table.notes[1][:48]
            'Expected durations: 5.0 periods (low), 20.0 (hig'
        """
        durations = self.expected_durations()
        share = float(self.smoothed_prob[:, 0].mean())
        rows = tuple(
            (
                name,
                f"{float(self.loadings[index]):.4f}",
                f"{float(self.idiosyncratic_var[index]):.4f}",
            )
            for index, name in enumerate(self.series_names)
        )
        notes = [
            f"Factor: phi = {self.phi:.4f}, sigma_v = {self.sigma_v:.4f}; "
            f"regime intercepts mu = ({self.mu[0]:+.4f}, {self.mu[1]:+.4f}), "
            "low state first by the ordering convention.",
            f"Expected durations: {durations[0]:.1f} periods (low), "
            f"{durations[1]:.1f} (high); the sample spends "
            f"{100.0 * share:.1f}% of its mass in the low state.",
            "The likelihood is the Kim (1994) collapse approximation, "
            "maximized as such and reported under that name.",
            "Identification: the first series loads with weight one, and "
            "loadings and variances are on the standardized panel's scale.",
            "recession_probabilities() is the smoothed low-state "
            "chronology -- the model's reason to exist.",
        ]
        return SummaryTable(
            title="Markov-Switching Dynamic Factor Results",
            metadata=(
                ("Model", "MS-DFM(1)"),
                ("Series", f"{self.n_series}"),
                ("Observations", f"{self.nobs}"),
                ("log L (approx)", f"{self.llf:.3f}"),
                ("Parameters", f"{self.n_params}"),
                ("Converged", f"{self.converged}"),
            ),
            columns=("series", "loading", "idiosyncratic var"),
            rows=rows,
            notes=tuple(notes),
        )


class MarkovSwitchingDFM:
    """Markov-switching dynamic factor model, Kim-Nelson.

    One factor, AR(1) with a two-state switching intercept; estimation by
    maximum likelihood through the continuous-state Kim filter. A fit
    costs tens of seconds rather than seconds -- the filter runs four
    Kalman updates per period inside the optimizer -- which is the known
    price of the model and is stated rather than hidden. The panel is
    standardized internally and every reported parameter is on that scale.

    Attributes:
        _panel: The validated ``(nobs, n_series)`` panel, unstandardized.
        _series_names: The resolved series labels.

    See Also:
        * :class:`MarkovSwitchingDFMResult` -- the record ``fit()`` returns.
        * :class:`MarkovSwitchingVAR` -- the switching model without the
          factor structure, for a handful of series.

    References:
        Kim, C.-J., & Nelson, C. R. (1998). Business cycle turning points, a
        new coincident index, and tests of duration dependence based on a
        dynamic factor model with regime switching. *Review of Economics
        and Statistics*, 80(2), 188-201.

        Chauvet, M. (1998). An econometric characterization of business
        cycle dynamics with factor structure and regime switching.
        *International Economic Review*, 39(4), 969-996.

    Example:
        Three coincident indicators loading on one factor whose intercept
        drops in a short-lived low state. The fit recovers the factor, the
        chronology and the asymmetric durations; about twenty-five seconds
        on this panel:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p = np.array([[0.8, 0.2], [0.05, 0.95]])
        >>> s, f = np.ones(120, dtype=int), np.zeros(120)
        >>> for t in range(1, 120):
        ...     s[t] = np.searchsorted(np.cumsum(p[s[t - 1]]), rng.random())
        ...     f[t] = [-1.5, 0.5][s[t]] + 0.6 * f[t - 1] + 0.5 * rng.standard_normal()
        >>> x = np.outer(f, [1.0, 0.8, 1.2]) + rng.standard_normal((120, 3)) * [0.5, 0.7, 0.6]
        >>> res = MarkovSwitchingDFM(x, series_names=("ip", "emp", "sales")).fit()
        >>> res.converged, res.n_params, bool(res.mu[0] < res.mu[1])
        (True, 11, True)
        >>> bool(np.corrcoef(res.factor, f)[0, 1] > 0.95)
        True
        >>> bool(((res.recession_probabilities() > 0.5) == (s == 0)).mean() > 0.95)
        True
        >>> bool(res.expected_durations()[0] < res.expected_durations()[1])
        True
    """

    __slots__ = ("_panel", "_series_names")

    def __init__(
        self,
        panel: npt.ArrayLike,
        *,
        series_names: Sequence[str] | None = None,
    ) -> None:
        """Validate the panel.

        Args:
            panel: The observed ``(nobs, n_series)`` panel of coincident
                indicators; each series is standardized internally.
            series_names: One label per series. Defaults to ``x1 ... xN``.

        Raises:
            DimensionError: If the panel has fewer than two series or fewer
                than sixty observations.
            SpecificationError: If ``series_names`` has the wrong length.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> MarkovSwitchingDFM(rng.standard_normal((80, 3)))._series_names
            ('x1', 'x2', 'x3')
            >>> MarkovSwitchingDFM(rng.standard_normal((50, 3)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.DimensionError: a switching factor model needs at least 60 ...
            >>> MarkovSwitchingDFM(rng.standard_normal((80, 1)))
            Traceback (most recent call last):
            cultivars.exceptions.DimensionError: a factor model needs at least 2 series; got 1.
        """
        self._panel = validate_endog_matrix(panel)
        nobs, n_series = self._panel.shape
        if n_series < 2:
            raise DimensionError(f"a factor model needs at least 2 series; got {n_series}.")
        if nobs < 60:
            raise DimensionError(
                f"a switching factor model needs at least 60 observations "
                f"to see both regimes; got {nobs}."
            )
        if series_names is None:
            self._series_names = tuple(f"x{i + 1}" for i in range(n_series))
        else:
            resolved = tuple(str(name) for name in series_names)
            if len(resolved) != n_series:
                raise SpecificationError(
                    f"series_names must have one entry per series "
                    f"({n_series}); got {len(resolved)}."
                )
            self._series_names = resolved

    @staticmethod
    def _build(theta: npt.NDArray[np.float64], n_series: int) -> _RegimeSwitchingLinearStateSpace:
        """The state space a parameter vector names.

        Layout of ``theta``: loadings 2..N, log idiosyncratic variances,
        arctanh of the factor persistence, log factor innovation sd, the
        low intercept, the log intercept gap, and the two staying-logits.
        The gap is logged so the low state is low by construction, which
        is the label-ordering convention.

        Args:
            theta: The ``(2 n_series + 5,)`` unconstrained parameter vector.
            n_series: Series in the panel.

        Returns:
            A two-regime linear state space with a scalar state.

        Example:
            At ``theta = 0`` the loadings are one, the variances one, the
            persistence zero and the chain symmetric:

            >>> import numpy as np
            >>> space = MarkovSwitchingDFM._build(np.zeros(2 * 3 + 5), 3)
            >>> type(space).__name__, space.regime_transition.tolist()
            ('_RegimeSwitchingLinearStateSpace', [[0.5, 0.5], [0.5, 0.5]])
        """
        loadings = np.concatenate([[1.0], theta[: n_series - 1]])
        psi = np.exp(theta[n_series - 1 : 2 * n_series - 1])
        phi = float(np.tanh(theta[2 * n_series - 1]))
        sigma_v = float(np.exp(theta[2 * n_series]))
        mu_low = float(theta[2 * n_series + 1])
        mu_high = mu_low + float(np.exp(theta[2 * n_series + 2]))
        p00 = 1.0 / (1.0 + np.exp(-theta[2 * n_series + 3]))
        p11 = 1.0 / (1.0 + np.exp(-theta[2 * n_series + 4]))
        design = np.tile(loadings.reshape(n_series, 1), (2, 1, 1))
        return _RegimeSwitchingLinearStateSpace(
            design=design,
            obs_cov=np.tile(np.diag(psi), (2, 1, 1)),
            transition=np.full((2, 1, 1), phi),
            state_cov=np.full((2, 1, 1), sigma_v**2),
            state_intercept=np.array([[mu_low], [mu_high]]),
            regime_transition=np.array([[p00, 1.0 - p00], [1.0 - p11, p11]]),
            initial_state=np.zeros(1),
            initial_state_cov=np.array([[10.0]]),
        )

    def fit(self, *, max_iter: int = 300) -> MarkovSwitchingDFMResult:
        """Maximize the Kim-approximate likelihood from data-driven starts.

        Starting values come from a principal-component pass -- the factor
        space is cheap to locate; the switch is what the optimizer earns
        -- and the parameters travel in unconstrained transforms, so the
        optimizer never sees a boundary. The search is L-BFGS-B with
        finite-difference gradients, which is where the running time goes.

        Args:
            max_iter: Optimizer iteration cap.

        Returns:
            The fitted :class:`MarkovSwitchingDFMResult`.

        Raises:
            NumericalError: If the likelihood cannot be evaluated at the
                starting point.

        Example:
            A capped run returns with ``converged`` false and a lower
            likelihood than the full fit on the class Example; the fit is
            deterministic, so two calls agree exactly:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> f = np.zeros(100)
            >>> for t in range(1, 100):
            ...     low = t % 25 < 5
            ...     f[t] = (-1.5 if low else 0.5) + 0.6 * f[t - 1] + 0.5 * rng.standard_normal()
            >>> x = np.outer(f, [1.0, 0.8, 1.2]) + 0.6 * rng.standard_normal((100, 3))
            >>> model = MarkovSwitchingDFM(x)
            >>> capped = model.fit(max_iter=5)
            >>> capped.converged, capped.n_params, capped.factor.shape
            (False, 11, (100,))
            >>> capped.llf == model.fit(max_iter=5).llf
            True
        """
        _, n_series = self._panel.shape
        means = self._panel.mean(axis=0)
        scales = self._panel.std(axis=0, ddof=0)
        scales = np.where(scales > 0.0, scales, 1.0)
        standardized = (self._panel - means) / scales
        _, _, vt = np.linalg.svd(standardized, full_matrices=False)
        pilot = standardized @ vt[0]
        if float(np.corrcoef(pilot, standardized[:, 0])[0, 1]) < 0.0:
            pilot = -pilot
        raw_loadings = standardized.T @ pilot / float(pilot @ pilot)
        pilot = pilot * raw_loadings[0]
        raw_loadings = raw_loadings / raw_loadings[0]
        residual = standardized - np.outer(pilot, raw_loadings)
        psi0 = np.maximum(residual.var(axis=0), 1e-3)
        phi0 = float(
            np.clip(
                (pilot[1:] @ pilot[:-1]) / max(float(pilot[:-1] @ pilot[:-1]), 1e-12),
                -0.95,
                0.95,
            )
        )
        innovation = pilot[1:] - phi0 * pilot[:-1]
        sigma0 = float(np.sqrt(max(np.var(innovation), 1e-4)))
        spread = float(pilot.std())
        theta0 = np.concatenate(
            [
                raw_loadings[1:],
                np.log(psi0),
                [np.arctanh(np.clip(phi0, -0.97, 0.97))],
                [np.log(sigma0)],
                [-0.8 * spread],
                [np.log(max(1.2 * spread, 1e-3))],
                [np.log(0.9 / 0.1), np.log(0.9 / 0.1)],
            ]
        )

        def negative(theta: npt.NDArray[np.float64]) -> float:
            try:
                value = self._build(theta, n_series).loglikelihood(standardized)
            except NumericalError:
                return 1e12
            return -value if np.isfinite(value) else 1e12

        start_value = negative(theta0)
        if start_value >= 1e12:
            raise NumericalError(
                "the switching factor likelihood cannot be evaluated at the "
                "principal-component starting point; the panel is degenerate."
            )
        search = minimize(negative, theta0, method="L-BFGS-B", options={"maxiter": max_iter})
        theta = np.asarray(search.x, dtype=np.float64)
        space = self._build(theta, n_series)
        outcome = space.filter(standardized)
        transition = space.regime_transition
        smoothed = space.smooth(standardized).smoothed_prob
        loadings = np.concatenate([[1.0], theta[: n_series - 1]])
        psi = np.exp(theta[n_series - 1 : 2 * n_series - 1])
        mu_low = float(theta[2 * n_series + 1])
        mu_high = mu_low + float(np.exp(theta[2 * n_series + 2]))
        return MarkovSwitchingDFMResult(
            panel=self._panel,
            series_names=self._series_names,
            loadings=loadings,
            idiosyncratic_var=psi,
            phi=float(np.tanh(theta[2 * n_series - 1])),
            sigma_v=float(np.exp(theta[2 * n_series])),
            mu=np.array([mu_low, mu_high]),
            regime_transition=transition,
            factor=outcome.filtered_state[:, 0],
            filtered_prob=outcome.filtered_prob,
            smoothed_prob=smoothed,
            means=means,
            scales=scales,
            llf=float(outcome.loglikelihood),
            n_params=int(theta.shape[0]),
            converged=bool(search.success),
        )
