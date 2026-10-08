# filepath: /src/cultivars/univariate/markov_switching.py
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
r"""Markov-switching autoregression: the latent-regime counterpart to SETAR.

A :math:`K`-regime autoregression whose intercept, autoregressive
coefficients, and innovation variance may each switch with a latent
first-order Markov chain :math:`S_t \in \{0, \ldots, K-1\}`,

.. math::

   y_t = c_{S_t} + \sum_{i=1}^{p} \phi_{i,S_t}\, y_{t-i} + \varepsilon_t,
   \qquad \varepsilon_t \sim N(0, \sigma^2_{S_t}),
   \qquad P_{ij} = \Pr(S_t = j \mid S_{t-1} = i).

Where :mod:`cultivars.univariate.threshold` computes the regime from
something observable, here the regime is never observed at all: what
comes back is a posterior over states at every date, and every reported
quantity -- fitted values, residuals, even the regime a date "is in" --
is an expectation under that posterior rather than a fact about the
data. The Hamilton filter produces the predicted and filtered posteriors
and the likelihood,

.. math::

   \Pr(S_t = j \mid y_{1:t-1}) = \sum_i P_{ij}\,\Pr(S_{t-1} = i \mid y_{1:t-1}),
   \qquad
   \Pr(S_t = j \mid y_{1:t}) \propto \Pr(S_t = j \mid y_{1:t-1})\,
   f(y_t \mid S_t = j, y_{1:t-1}),

   \log L = \sum_t \log \sum_j \Pr(S_t = j \mid y_{1:t-1})\,
   f(y_t \mid S_t = j, y_{1:t-1}),

and Kim's backward pass turns them into the smoothed posterior
:math:`\Pr(S_t = j \mid y_{1:T})`. Estimation is EM (Hamilton 1990): the
E-step is that filter and smoother, the M-step updates the transition
matrix from expected transition counts and the coefficients by
responsibility-weighted least squares. The likelihood is multimodal, so
:meth:`MarkovSwitchingAR.fit` screens several random starts and refines
the best -- which is why ``fit`` takes a ``seed`` and why the result
reports whether it converged.

Two commitments shape the surface. First, this is the
*intercept*-switching parameterization, and one class spans the Krolzig
(1997) taxonomy. The regime enters contemporaneously through
:math:`c_{S_t}`, so conditional on the observed lags each regime is
linear and inference is a plain :math:`K`-state chain -- which is what
lets estimation ride on a filter with :math:`K` states rather than the
:math:`K^{p+1}` that Hamilton's original *mean*-switching form requires;
the two coincide at :math:`p = 0` and differ in transient dynamics
otherwise. Which blocks switch is chosen at construction and
:attr:`MarkovSwitchingARResult.specification` reports which member was
fitted, ``I`` for a switching intercept, ``A`` for switching
autoregressive coefficients, ``H`` for a switching variance; Hamilton's
recession-dating model is ``MSIH(2)-AR(4)``,
``MarkovSwitchingAR(y, order=4, n_regimes=2)`` here. Second, two
identification facts are made explicit rather than left implicit. The
likelihood is invariant to permuting regime labels, so a sorting
convention is imposed and :attr:`MarkovSwitchingARResult.label_ordering`
names it. And the number of regimes cannot be tested by a likelihood
ratio: under the null of :math:`K` regimes, the parameters of the
:math:`(K+1)`-th and the transition probabilities into it are
unidentified, so the statistic has no chi-squared limit.
:meth:`MarkovSwitchingARResult.likelihood_ratio_test` therefore refuses
that specific comparison while still permitting tests that hold
:math:`K` fixed.

Layout. :class:`MarkovSwitchingAR` validates ``order`` and
``n_regimes`` through ``validate_order`` and the switching flags on the
``_MarkovSwitchingModel`` base in ``_internals``; ``_fit_family`` draws
starts through ``initial_transition``, runs each through ``_run_em`` for
``screen_iter`` iterations, and refines the best to ``max_iter``. Each
EM step builds a ``_MarkovSwitchingStateSpace`` in
``_internals._substrates`` -- the same engine
:mod:`cultivars.state_space.regime_switching` exposes -- whose
``filter`` and ``smooth`` are ``hamilton_filter`` and ``kim_smoother``
from ``_internals``; the M-step state lives in
``_ExpectationMaximizationState``. The packed ``_MarkovSwitchingFit``
is assembled by :meth:`MarkovSwitchingARResult._from_fit`, the result's
``state_space`` rebuilds the substrate for out-of-sample filtering, and
``simulate`` runs ``_simulate_markov_switching``. The vector model is
:mod:`cultivars.multivariate.regime_switching.markov_switching`; a
switching chain over a general state-space measurement is
:class:`~cultivars.state_space.regime_switching.MarkovSwitchingSSM`.

References:
    Hamilton, J. D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. *Econometrica*,
    57(2), 357-384.

    Hamilton, J. D. (1990). Analysis of time series subject to changes
    in regime. *Journal of Econometrics*, 45(1-2), 39-70.

    Kim, C.-J. (1994). Dynamic linear models with Markov-switching.
    *Journal of Econometrics*, 60(1-2), 1-22.

    Hansen, B. E. (1992). The likelihood ratio test under nonstandard
    conditions: Testing the Markov switching model of GNP. *Journal of
    Applied Econometrics*, 7(S1), S61-S82.

    Krolzig, H.-M. (1997). *Markov-Switching Vector Autoregressions:
    Modelling, Statistical Inference, and Application to Business Cycle
    Analysis*. Springer.

Example:
    A latent level shift that no observed-threshold model can see,
    dated in real time and after the fact:

    >>> import numpy as np
    >>> from cultivars.univariate.threshold import SETAR
    >>> rng = np.random.default_rng(0)
    >>> p = np.array([[0.97, 0.03], [0.05, 0.95]])
    >>> s = np.zeros(1500, dtype=int)
    >>> for t in range(1, 1500):
    ...     s[t] = np.searchsorted(np.cumsum(p[s[t - 1]]), rng.random())
    >>> y = np.array([-1.0, 1.5])[s] + 0.5 * rng.standard_normal(1500)
    >>> latent = MarkovSwitchingAR(y, order=1, n_regimes=2).fit(seed=0)
    >>> observed = SETAR(y, order=1).fit()
    >>> bool(latent.information_criteria.bic < observed.information_criteria.bic - 100)
    True
    >>> real_time = latent.probabilities("filtered").argmax(axis=1)
    >>> after_the_fact = latent.most_likely_regime
    >>> bool(np.mean(real_time == s[1:]) > 0.99), bool(np.mean(after_the_fact == s[1:]) > 0.99)
    (True, True)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import (
    _SIMULATION_BURN,
    InformationCriteria,
    ProbabilityType,
    SummaryTable,
    validate_choice,
)
from .._internals import (
    _ComparisonMixin,
    _MarkovSwitchingFit,
    _MarkovSwitchingModel,
    _MarkovSwitchingStateSpace,
    _SeriesMixin,
    _simulate_markov_switching,
    _StabilityAssessment,
    _SummaryMixin,
)
from ..exceptions import SpecificationError

__all__ = ["MarkovSwitchingAR", "MarkovSwitchingARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MarkovSwitchingARResult(_SummaryMixin, _SeriesMixin, _ComparisonMixin):
    r"""A fitted Markov-switching autoregression.

    The model

    .. math::

        y_t = c_{S_t} + \sum_{i=1}^{p} \phi_{i,S_t}\, y_{t-i} + \varepsilon_t,
        \qquad \varepsilon_t \sim N(0, \sigma^2_{S_t}),
        \qquad \Pr(S_t = j \mid S_{t-1} = i) = P_{ij},

    with :math:`S_t \in \{0, \ldots, K-1\}` a latent first-order Markov
    chain and whichever of the intercept, autoregressive block and
    variance were declared switching taking regime-specific values. The
    regime is never observed, so every per-date quantity here is an
    expectation under a posterior over states: the fitted value is
    :math:`\sum_j \Pr(S_t = j \mid \cdot)\, m_{j,t}`, the residual is
    taken against it, and the three posteriors -- predicted, filtered,
    smoothed -- differ only in what they condition on. The parameters
    are the EM maximizer from the best of several screened starts, with
    the regimes relabelled by the convention :attr:`label_ordering`
    names.

    Attributes:
        endog: The full input series.
        fittedvalues: Posterior-weighted one-step means, ``sum_j p_j m_j``.
        resid: Residuals against those weighted means.
        llf: Log-likelihood from the final Hamilton filter pass.
        nobs: Effective sample, ``len(endog) - order``.
        n_params: Free parameters: ``K(K-1)`` transition probabilities plus the
            switching and non-switching coefficient blocks and the variances.
        order: Autoregressive order within each regime.
        n_regimes: Number of regimes ``K``.
        switching_mean: Whether the intercept switches.
        switching_ar: Whether the autoregressive coefficients switch.
        switching_variance: Whether the innovation variance switches.
        transition: Row-stochastic ``(K, K)`` matrix.
        intercepts: Per-regime intercepts, shape ``(K,)``.
        ar_params: Per-regime autoregressive coefficients, shape ``(K, p)``.
        variances: Per-regime innovation variances, shape ``(K,)``.
        filtered_prob: ``Pr(S_t = j | y_1..t)``, shape ``(nobs, K)``.
        predicted_prob: ``Pr(S_t = j | y_1..t-1)``, shape ``(nobs, K)``.
        smoothed_prob: ``Pr(S_t = j | y_1..T)``, shape ``(nobs, K)``.
        ergodic_prob: Stationary distribution of ``transition``, shape ``(K,)``.
        expected_durations: ``1 / (1 - P_jj)``, in periods, shape ``(K,)``.
        n_iter: EM iterations used by the refining run.
        converged: Whether the refining run met its tolerance.

    Note:
        Non-switching blocks are stored at full ``(K, ...)`` shape with
        identical rows, so ``ar_params[j]`` is always the block regime
        ``j`` uses whether or not it switches; ``n_params`` counts each
        shared block once. ``fittedvalues``, ``resid`` and the three
        posteriors are aligned with the last ``nobs`` observations of
        ``endog``. Standard errors are not yet reported by this
        estimator.

    See Also:
        * :class:`MarkovSwitchingAR` -- the specification that produces
          this.
        * :class:`~cultivars.univariate.threshold.SETARResult` -- regimes
          computed from an observed variable instead of inferred.
        * :class:`~cultivars.state_space.regime_switching.MarkovSwitchingSSM`
          -- the same chain over a general state-space measurement.

    References:
        Hamilton, J. D. (1989). A new approach to the economic analysis
        of nonstationary time series and the business cycle.
        *Econometrica*, 57(2), 357-384.

        Kim, C.-J. (1994). Dynamic linear models with Markov-switching.
        *Journal of Econometrics*, 60(1-2), 1-22.

        Krolzig, H.-M. (1997). *Markov-Switching Vector Autoregressions*.
        Springer.

    Example:
        A two-regime shift in level, with the chain and the posterior
        dating compared against the truth:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p = np.array([[0.97, 0.03], [0.05, 0.95]])
        >>> mu = np.array([-1.0, 1.5])
        >>> s = np.zeros(1500, dtype=int)
        >>> for t in range(1, 1500):
        ...     s[t] = np.searchsorted(np.cumsum(p[s[t - 1]]), rng.random())
        >>> y = mu[s] + 0.5 * rng.standard_normal(1500)
        >>> res = MarkovSwitchingAR(y, order=1, n_regimes=2).fit(seed=0)
        >>> res.specification, res.converged, res.nobs, res.n_params
        ('MSIH(2)-AR(1)', True, 1499, 7)
        >>> bool(np.allclose(res.transition, p, atol=0.03))
        True
        >>> bool(np.allclose(res.intercepts, mu, atol=0.1))
        True
        >>> bool(np.mean(res.most_likely_regime == s[1:]) > 0.98)
        True
    """

    endog: npt.NDArray[np.float64]
    """``(n,)`` observed series, as given."""

    fittedvalues: npt.NDArray[np.float64]
    """``(nobs,)`` smoothed-posterior-weighted one-step means."""

    resid: npt.NDArray[np.float64]
    """``(nobs,)`` residuals against :attr:`fittedvalues`."""

    llf: float
    """Log-likelihood from the Hamilton filter at the final parameters."""

    nobs: int
    """Effective sample, ``len(endog) - order``."""

    n_params: float
    """Free parameters, counting each non-switching block once."""

    order: int
    """Autoregressive order within each regime."""

    n_regimes: int
    """Number of regimes :math:`K`."""

    switching_mean: bool
    """Whether the intercept takes a regime-specific value."""

    switching_ar: bool
    """Whether the autoregressive block takes regime-specific values."""

    switching_variance: bool
    """Whether the innovation variance takes a regime-specific value."""

    transition: npt.NDArray[np.float64]
    """``(K, K)`` row-stochastic matrix, ``[i, j]`` from ``i`` to ``j``."""

    intercepts: npt.NDArray[np.float64]
    """``(K,)`` intercepts, ascending when they switch."""

    ar_params: npt.NDArray[np.float64]
    """``(K, p)`` autoregressive coefficients, rows identical when shared."""

    variances: npt.NDArray[np.float64]
    """``(K,)`` innovation variances, entries identical when shared."""

    filtered_prob: npt.NDArray[np.float64]
    """``(nobs, K)`` real-time posterior ``Pr(S_t | y_1..t)``."""

    predicted_prob: npt.NDArray[np.float64]
    """``(nobs, K)`` one-step-ahead posterior ``Pr(S_t | y_1..t-1)``."""

    smoothed_prob: npt.NDArray[np.float64]
    """``(nobs, K)`` full-sample posterior ``Pr(S_t | y_1..T)``."""

    ergodic_prob: npt.NDArray[np.float64]
    """``(K,)`` stationary distribution of :attr:`transition`."""

    expected_durations: npt.NDArray[np.float64]
    """``(K,)`` mean regime spell lengths ``1 / (1 - P_jj)``."""

    n_iter: int
    """EM iterations used by the refining run."""

    converged: bool
    """Whether the refining run met its tolerance within ``max_iter``."""

    @classmethod
    def _from_fit(
        cls, fit: _MarkovSwitchingFit, model: _MarkovSwitchingModel[MarkovSwitchingARResult]
    ) -> MarkovSwitchingARResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The raw estimator output.
            model: The specification, read for the series, the orders and
                the switching flags.

        Returns:
            The assembled result.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> model = MarkovSwitchingAR(rng.standard_normal(300), order=2, n_regimes=2)
            >>> fit = model._fit_family(max_iter=50, tol=1e-6, n_init=2, screen_iter=5, seed=0)
            >>> res = MarkovSwitchingARResult._from_fit(fit, model)
            >>> res.order, res.n_regimes, res.ar_params.shape, res.nobs
            (2, 2, (2, 2), 298)
        """
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            order=model.order,
            n_regimes=model.n_regimes,
            switching_mean=model._sw_mean,
            switching_ar=model._sw_ar,
            switching_variance=model._sw_var,
            transition=fit.transition,
            intercepts=fit.intercepts,
            ar_params=fit.ar_params,
            variances=fit.variances,
            filtered_prob=fit.filtered_prob,
            predicted_prob=fit.predicted_prob,
            smoothed_prob=fit.smoothed_prob,
            ergodic_prob=fit.ergodic_prob,
            expected_durations=fit.expected_durations,
            n_iter=fit.n_iter,
            converged=fit.converged,
        )

    @property
    def specification(self) -> str:
        """Krolzig code for which blocks switch, e.g. ``"MSIH(2)-AR(4)"``.

        ``I`` marks a switching intercept, ``A`` switching autoregressive
        coefficients, ``H`` a switching variance -- so the code names the model
        you actually fitted rather than the family it came from. The
        constructor guarantees at least one letter is present.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(300)
            >>> model = MarkovSwitchingAR(y, order=1, n_regimes=3, switching_variance=False)
            >>> model.fit(seed=0, n_init=2).specification
            'MSI(3)-AR(1)'
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
        return f"MS{letters}({self.n_regimes})-AR({self.order})"

    @property
    def label_ordering(self) -> str:
        """Which quantity regimes were sorted by, ascending.

        A mixture likelihood is invariant to relabelling regimes, so the fit
        imposes an ordering to make two runs of the same specification
        comparable. The intercept is used when it switches; otherwise the
        variance; otherwise the first autoregressive coefficient. Reported
        because the meaning of "regime 0" depends on it: under a
        variance-switching-only model, regime 0 is the *quiet* regime, not a
        low-mean one.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(300)
            >>> model = MarkovSwitchingAR(y, order=1, n_regimes=2, switching_mean=False)
            >>> res = model.fit(seed=0, n_init=2)
            >>> res.label_ordering, bool(res.variances[0] <= res.variances[1])
            ('variance', True)
        """
        if self.switching_mean:
            return "intercept"
        if self.switching_variance:
            return "variance"
        return "first AR coefficient"

    def probabilities(self, kind: ProbabilityType = "smoothed") -> npt.NDArray[np.float64]:
        """Return one of the three regime posteriors.

        Args:
            kind: ``"smoothed"`` conditions on the whole sample and is what you
                want for dating regimes after the fact; ``"filtered"``
                conditions on data through ``t`` and is the real-time view;
                ``"predicted"`` conditions through ``t - 1`` and is the
                one-step-ahead forecast of the state.

        Returns:
            An array of shape ``(nobs, K)`` whose rows sum to one.

        Raises:
            SpecificationError: If ``kind`` is not one of the three.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = MarkovSwitchingAR(rng.standard_normal(300), order=1).fit(seed=0, n_init=2)
            >>> res.probabilities("filtered").shape, res.probabilities().sum(axis=1).round(6).max()
            ((299, 2), np.float64(1.0))
            >>> res.probabilities("viterbi")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
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

        This is the *marginal* MAP state at each date, taken independently, not
        the Viterbi path. The distinction matters: the sequence returned here
        can contain a one-period switch whose transition probability is
        near-zero, so as a *path* it may have essentially no posterior mass even
        though every element of it is individually most likely. Use it to date
        regimes, not to reason about the sequence of switches.

        Returns:
            An integer array of length ``nobs`` with values in ``0..K-1``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = MarkovSwitchingAR(rng.standard_normal(300), order=1).fit(seed=0, n_init=2)
            >>> regime = res.most_likely_regime
            >>> regime.dtype, regime.shape, bool(set(regime) <= {0, 1})
            (dtype('int64'), (299,), True)
        """
        return np.argmax(self.smoothed_prob, axis=1).astype(np.int64)

    @property
    def regime_shares(self) -> npt.NDArray[np.float64]:
        """Average smoothed probability of each regime over the sample.

        The posterior counterpart to a frequency count, and the honest one:
        counting :attr:`most_likely_regime` throws away every date the model was
        genuinely uncertain about. Compare against :attr:`ergodic_prob` -- a
        large gap says the sample is not representative of the fitted chain.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = MarkovSwitchingAR(rng.standard_normal(300), order=1).fit(seed=0, n_init=2)
            >>> res.regime_shares.shape, round(float(res.regime_shares.sum()), 6)
            ((2,), 1.0)
        """
        return np.asarray(self.smoothed_prob.mean(axis=0), dtype=np.float64)

    @property
    def regime_uncertainty(self) -> float:
        """Mean posterior entropy, normalized so ``0`` is sharp and ``1`` is flat.

        Averages the Shannon entropy of each date's smoothed distribution and
        divides by ``log K``. A value near zero means the regimes are cleanly
        separated and the model is effectively dating a deterministic
        partition; a value near one means the data barely distinguish the
        regimes at all, which no coefficient table will tell you.

        Example:
            Separated regimes against white noise forced into two:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = (np.arange(600) // 150) % 2
            >>> y = np.where(s == 1, 2.0, -2.0) + 0.5 * rng.standard_normal(600)
            >>> sharp = MarkovSwitchingAR(y, order=1).fit(seed=0, n_init=2)
            >>> flat = MarkovSwitchingAR(rng.standard_normal(600), order=1).fit(seed=0, n_init=2)
            >>> bool(sharp.regime_uncertainty < 0.01), bool(flat.regime_uncertainty > 0.05)
            (True, True)
        """
        p = np.clip(self.smoothed_prob, 1e-300, None)
        entropy = -(p * np.log(p)).sum(axis=1)
        return float(entropy.mean() / np.log(self.n_regimes))

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """Aligned per-observation output, widened by the regime posterior.

        Uses the two-argument ``super`` deliberately. ``@dataclass(slots=True)``
        builds a *new* class object and rebinds the name to it, so the
        ``__class__`` cell that zero-argument ``super()`` closes over still
        points at the original, pre-slots class -- which no subclass inherits
        from.

        Returns:
            The observed/fitted/residual triple, one smoothed-probability
            column per regime, and the pointwise most likely regime. The
            filtered and predicted posteriors are reachable through
            :meth:`probabilities` rather than widening this by ``2K`` columns.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = MarkovSwitchingAR(rng.standard_normal(300), order=1).fit(seed=0, n_init=2)
            >>> list(res._series())
            ['observed', 'fitted', 'resid', 'smoothed_prob_0', 'smoothed_prob_1', 'regime']
        """
        base = super(MarkovSwitchingARResult, self)._series()
        for j in range(self.n_regimes):
            base[f"smoothed_prob_{j}"] = self.smoothed_prob[:, j]
        base["regime"] = self.most_likely_regime.astype(np.float64)
        return base

    def regime_stability(self, regime: int) -> _StabilityAssessment:
        """Companion-eigenvalue verdict for one regime's autoregressive block.

        Args:
            regime: Regime index in ``0..K-1``.

        Returns:
            The :class:`_StabilityAssessment` for that regime read as a linear
            autoregression.

        Raises:
            SpecificationError: If ``regime`` is out of range.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = MarkovSwitchingAR(rng.standard_normal(300), order=1).fit(seed=0, n_init=2)
            >>> res.regime_stability(0).is_stable
            True
            >>> res.regime_stability(2)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: regime must be in 0..1; got 2.
        """
        if not 0 <= regime < self.n_regimes:
            raise SpecificationError(f"regime must be in 0..{self.n_regimes - 1}; got {regime}.")
        return _StabilityAssessment.assess_stability(self.ar_params[regime])

    @property
    def is_regimewise_stationary(self) -> bool:
        """Whether every regime is stationary read as a linear autoregression.

        Sufficient for stationarity of the switching process but not necessary.
        A Markov-switching model with one explosive regime is stationary
        provided that regime is visited rarely enough and exited fast enough --
        the condition involves the transition matrix and the explosive root
        jointly, not the roots alone. A ``False`` here is a prompt to look at
        that regime's expected duration, not a verdict that the process
        explodes.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = MarkovSwitchingAR(rng.standard_normal(300), order=1).fit(seed=0, n_init=2)
            >>> res.is_regimewise_stationary
            True
        """
        return all(self.regime_stability(j).is_stable for j in range(self.n_regimes))

    @property
    def unconditional_mean(self) -> float | None:
        """Ergodic mean of the process, or ``None`` if it is not defined.

        Computed as the ergodic-probability-weighted average of each regime's
        own unconditional mean ``c_j / (1 - sum phi_j)``. Returns ``None`` when
        any regime is non-stationary, since that regime has no unconditional
        mean to average in, and when the autoregressive sum sits at one.

        Example:
            On a draw from a stationary chain the ergodic mean sits near
            the sample mean:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> p = np.array([[0.97, 0.03], [0.05, 0.95]])
            >>> s = np.zeros(1500, dtype=int)
            >>> for t in range(1, 1500):
            ...     s[t] = np.searchsorted(np.cumsum(p[s[t - 1]]), rng.random())
            >>> y = np.array([-1.0, 1.5])[s] + 0.5 * rng.standard_normal(1500)
            >>> res = MarkovSwitchingAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> bool(abs(res.unconditional_mean - y.mean()) < 0.15)
            True
        """
        if not self.is_regimewise_stationary:
            return None
        denominators = 1.0 - self.ar_params.sum(axis=1)
        if np.any(np.abs(denominators) < 1e-12):
            return None
        return float(np.sum(self.ergodic_prob * (self.intercepts / denominators)))

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order.

        Regime-indexed names carry the regime in brackets, and transition
        entries read ``p[i->j]``. All ``K**2`` transition entries appear even
        though only ``K(K-1)`` are free, because a row displayed without its
        final element is harder to read than one whose redundancy is stated in
        the notes.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = MarkovSwitchingAR(rng.standard_normal(300), order=1).fit(seed=0, n_init=2)
            >>> list(res.params)[:4]
            ['const[0]', 'ar.L1[0]', 'const[1]', 'ar.L1[1]']
            >>> list(res.params)[-4:]
            ['p[0->0]', 'p[0->1]', 'p[1->0]', 'p[1->1]']
        """
        out: dict[str, float] = {}
        for j in range(self.n_regimes):
            out[f"const[{j}]"] = float(self.intercepts[j])
            for i in range(self.order):
                out[f"ar.L{i + 1}[{j}]"] = float(self.ar_params[j, i])
        for j in range(self.n_regimes):
            out[f"sigma2[{j}]"] = float(self.variances[j])
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                out[f"p[{i}->{j}]"] = float(self.transition[i, j])
        return out

    def regime_table(self) -> SummaryTable:
        """One row per regime: level, scale, and how long it lasts.

        The complement to the coefficient table, which reads down parameters;
        this reads across regimes, which is the comparison the model exists to
        support. Autoregressive coefficients are left to
        :attr:`params` -- there are ``p`` of them per regime and they do not fit
        a fixed-width row.

        Returns:
            A :class:`SummaryTable` with one row per regime.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = MarkovSwitchingAR(rng.standard_normal(300), order=1).fit(seed=0, n_init=2)
            >>> table = res.regime_table()
            >>> table.columns, len(table.rows)
            (('regime', 'const', 'sigma2', 'ergodic', 'duration', 'share'), 2)
            >>> table.metadata[0]
            ('Ordering', 'ascending intercept')
        """
        shares = self.regime_shares
        return SummaryTable(
            title=f"{self.specification} regimes",
            metadata=(
                ("Ordering", f"ascending {self.label_ordering}"),
                ("Observations", f"{self.nobs}"),
            ),
            columns=("regime", "const", "sigma2", "ergodic", "duration", "share"),
            rows=tuple(
                (
                    f"{j}",
                    f"{self.intercepts[j]:.4f}",
                    f"{self.variances[j]:.4f}",
                    f"{self.ergodic_prob[j]:.4f}",
                    f"{self.expected_durations[j]:.2f}",
                    f"{shares[j]:.4f}",
                )
                for j in range(self.n_regimes)
            ),
            notes=(
                "'ergodic' is the stationary probability implied by the transition "
                "matrix; 'share' is the average smoothed probability actually "
                "observed. 'duration' is 1 / (1 - P_jj), in periods.",
            ),
        )

    def transition_table(self) -> SummaryTable:
        r"""The estimated transition matrix, rows indexed by the origin regime.

        Returns:
            A :class:`SummaryTable` whose entry ``(i, j)`` is
            ``Pr(S_t = j | S_{t-1} = i)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = MarkovSwitchingAR(rng.standard_normal(300), order=1).fit(seed=0, n_init=2)
            >>> table = res.transition_table()
            >>> table.columns, [row[0] for row in table.rows]
            (('from \\ to', '0', '1'), ['0', '1'])
        """
        return SummaryTable(
            title=f"{self.specification} transition matrix",
            metadata=(("Regimes", f"{self.n_regimes}"),),
            columns=("from \\ to", *(f"{j}" for j in range(self.n_regimes))),
            rows=tuple(
                (f"{i}", *(f"{self.transition[i, j]:.4f}" for j in range(self.n_regimes)))
                for i in range(self.n_regimes)
            ),
            notes=("Rows sum to one, so K(K-1) of the K**2 entries are free.",),
        )

    def simulate(
        self,
        n: int = 200,
        *,
        seed: int | np.random.Generator | None = None,
        burn: int = _SIMULATION_BURN,
    ) -> npt.NDArray[np.float64]:
        """A fresh sample path at the fitted parameters.

        The regime chain is started from its ergodic distribution and
        stepped through the fitted transition matrix; within each regime
        the series follows that regime's intercept, autoregression and
        innovation variance. ``burn`` initial periods are discarded. Use
        :meth:`simulate_regimes` to keep the regime path as well.

        Args:
            n: Observations kept.
            seed: Seed or generator.
            burn: Periods discarded from the start.

        Returns:
            An array of shape ``(n,)``.

        Raises:
            SpecificationError: If the counts are not usable.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = MarkovSwitchingAR(rng.standard_normal(300), order=1).fit(seed=0, n_init=2)
            >>> res.simulate(50, seed=1).shape
            (50,)
            >>> res.simulate(0)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: n must be positive and burn non-negative; ...
        """
        return self.simulate_regimes(n, seed=seed, burn=burn)[0]

    def simulate_regimes(
        self,
        n: int = 200,
        *,
        seed: int | np.random.Generator | None = None,
        burn: int = _SIMULATION_BURN,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """A fresh sample path together with the regimes that generated it.

        Args:
            n: Observations kept.
            seed: Seed or generator.
            burn: Periods discarded from the start.

        Returns:
            ``(y, regimes)`` of shapes ``(n,)`` and ``(n,)``, the regime
            labels ``0 .. K - 1`` in the order of :attr:`intercepts`.

        Raises:
            SpecificationError: If the counts are not usable.

        Example:
            The simulated chain spends time in each regime at its ergodic
            frequency:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> p = np.array([[0.97, 0.03], [0.05, 0.95]])
            >>> s = np.zeros(1500, dtype=int)
            >>> for t in range(1, 1500):
            ...     s[t] = np.searchsorted(np.cumsum(p[s[t - 1]]), rng.random())
            >>> y = np.array([-1.0, 1.5])[s] + 0.5 * rng.standard_normal(1500)
            >>> res = MarkovSwitchingAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> path, regimes = res.simulate_regimes(5000, seed=1)
            >>> shares = np.bincount(regimes, minlength=2) / 5000
            >>> bool(np.allclose(shares, res.ergodic_prob, atol=0.05))
            True
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        return _simulate_markov_switching(
            n,
            transition=self.transition,
            intercepts=self.intercepts,
            ar_params=self.ar_params,
            variances=self.variances,
            rng=rng,
            burn=burn,
        )

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = MarkovSwitchingAR(rng.standard_normal(300), order=2).fit(seed=0, n_init=2)
            >>> res._comparison_label()
            'MSIH(2)-AR(2)'
        """
        return self.specification

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        Metadata carries the specification, regime count, EM iteration
        count and convergence flag, the sample size, the likelihood with
        its criteria, and the posterior entropy; the coefficient table is
        :attr:`params` in order; the notes lead with a non-convergence
        warning when it applies, then the labelling convention, the
        separation and durations, the stationarity caveat, the redundancy
        of the transition rows, and the untestability of ``K``.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> model = MarkovSwitchingAR(rng.standard_normal(300), order=1)
            >>> table = model.fit(seed=0, n_init=2)._summary_table()
            >>> table.title, len(table.notes), table.metadata[6]
            ('MSIH(2)-AR(1) Results', 6, ('Converged', 'True'))
            >>> model.fit(seed=0, n_init=2, max_iter=1)._summary_table().notes[0][:21]
            'EM did NOT converge i'
        """
        ic: InformationCriteria = self.information_criteria
        notes: list[str] = []
        if not self.converged:
            notes.append(
                f"EM did NOT converge in {self.n_iter} iterations. Treat every "
                f"estimate below as provisional; raise max_iter, or refit with a "
                f"different seed, before reading anything into them."
            )
        notes.append(
            f"Regimes ordered by ascending {self.label_ordering}; the likelihood "
            f"is invariant to relabelling, so that convention is what makes "
            f"'regime 0' mean anything."
        )
        notes.append(
            f"Mean posterior entropy {self.regime_uncertainty:.3f} of 1 "
            f"({'well' if self.regime_uncertainty < 0.25 else 'poorly'} separated "
            f"regimes); expected durations "
            f"{np.array2string(self.expected_durations, precision=1)} periods."
        )
        notes.append(
            f"Regime-wise stationary: {self.is_regimewise_stationary}. This is "
            f"sufficient but not necessary: a rarely visited explosive regime "
            f"still leaves the switching process stationary."
        )
        notes.append(
            "Transition rows sum to one, so only K(K-1) of the reported p[i->j] "
            "are free; the parameter count reflects that."
        )
        notes.append(
            "The number of regimes cannot be tested by a likelihood ratio; "
            "see likelihood_ratio_test."
        )
        notes.append("Standard errors are not yet available for this estimator.")
        return SummaryTable(
            title=f"{self.specification} Results",
            metadata=(
                ("Model", self.specification),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Regimes", f"{self.n_regimes}"),
                ("AIC", f"{ic.aic:.3f}"),
                ("EM iterations", f"{self.n_iter}"),
                ("BIC", f"{ic.bic:.3f}"),
                ("Converged", f"{self.converged}"),
                ("HQIC", f"{ic.hqic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("Entropy", f"{self.regime_uncertainty:.3f}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=tuple(notes),
        )

    def _likelihood_ratio_obstacle(self, counterpart: _ComparisonMixin) -> str | None:
        """Block only the comparison that changes the number of regimes.

        Unlike a threshold model, a Markov-switching model is perfectly
        testable against a nested specification that holds ``K`` fixed --
        dropping a switching block, say -- because every parameter remains
        identified under that null. What is not testable is ``K`` itself: under
        the null of fewer regimes, the extra regime's coefficients are free to
        take any value and the transition probabilities into it sit on the
        boundary at zero, so the statistic is neither chi-squared nor even
        pivotal. A result that is not a Markov-switching fit at all counts as
        the one-regime case, which is exactly the classic invalid test.

        Args:
            counterpart: The other result in the proposed test.

        Returns:
            A message when the regime counts differ, otherwise ``None``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(300)
            >>> two = MarkovSwitchingAR(y, order=1, n_regimes=2).fit(seed=0, n_init=2)
            >>> three = MarkovSwitchingAR(y, order=1, n_regimes=3).fit(seed=0, n_init=2)
            >>> two._likelihood_ratio_obstacle(three)[:62]
            'a chi-squared likelihood-ratio test cannot compare 2 regimes a'
            >>> same_k = MarkovSwitchingAR(y, order=1, switching_variance=False)
            >>> two._likelihood_ratio_obstacle(same_k.fit(seed=0, n_init=2)) is None
            True
        """
        other = counterpart.n_regimes if isinstance(counterpart, MarkovSwitchingARResult) else 1
        if other == self.n_regimes:
            return None
        return (
            f"a chi-squared likelihood-ratio test cannot compare {self.n_regimes} "
            f"regimes against {other}: under the smaller model the extra regime's "
            f"parameters are unidentified and its transition probabilities lie on "
            f"the boundary, so the statistic has no chi-squared limit. Use the "
            f"Hansen (1992) bound or a parametric bootstrap instead. Tests that "
            f"hold the number of regimes fixed are still available."
        )

    @property
    def state_space(self) -> _MarkovSwitchingStateSpace:
        """The fitted system, re-applicable to data it was not estimated on.

        Filtering a new series through this reports what the estimated regimes
        say about observations the fit never saw -- the natural out-of-sample
        check for a regime model, and one the fitted arrays alone cannot give.

        Example:
            The fitted chain dates a fresh draw from the same process:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> p = np.array([[0.97, 0.03], [0.05, 0.95]])
            >>> def draw(n):
            ...     s = np.zeros(n, dtype=int)
            ...     for t in range(1, n):
            ...         s[t] = np.searchsorted(np.cumsum(p[s[t - 1]]), rng.random())
            ...     return s, np.array([-1.0, 1.5])[s] + 0.5 * rng.standard_normal(n)
            >>> _, y = draw(1500)
            >>> res = MarkovSwitchingAR(y, order=1, n_regimes=2).fit(seed=0)
            >>> abs(res.state_space.loglikelihood(y) - res.llf) < 1e-8
            True
            >>> s_new, y_new = draw(300)
            >>> dated = res.state_space.filter(y_new).filtered_prob.argmax(axis=1)
            >>> bool(np.mean(dated == s_new[1:]) > 0.95)
            True
        """
        return _MarkovSwitchingStateSpace(
            self.transition, self.intercepts, self.ar_params, self.variances
        )


class MarkovSwitchingAR(_MarkovSwitchingModel[MarkovSwitchingARResult]):
    r"""Markov-switching autoregression with ``K`` latent regimes.

    The intercept-switching parameterization: the regime enters
    contemporaneously through :math:`c_{S_t}`, so conditional on the
    observed lags each regime is a linear autoregression and inference
    runs on a :math:`K`-state Hamilton filter rather than the
    :math:`K^{p+1}` states a mean-switching form needs. Which blocks
    switch is chosen here and reported by the result's ``specification``
    in Krolzig's ``MS{I,A,H}(K)-AR(p)`` code; at least one block must
    switch, since otherwise no regime is identified. Hamilton's
    recession-dating model is ``MSIH(2)-AR(4)``,
    ``MarkovSwitchingAR(y, order=4, n_regimes=2)``. ``order=0`` is
    allowed and gives a hidden Markov model with Gaussian emissions.

    Attributes:
        _endog: The validated series.
        _order: Autoregressive order within each regime.
        _k: Number of regimes.
        _sw_mean: Whether the intercept switches.
        _sw_var: Whether the variance switches.
        _sw_ar: Whether the autoregressive block switches.

    Args:
        endog: The series.
        order: Autoregressive order ``p`` within each regime.
        n_regimes: Number of regimes ``K``, at least two.
        switching_mean: Whether the intercept switches.
        switching_variance: Whether the innovation variance switches.
        switching_ar: Whether the autoregressive coefficients switch. Off by
            default: switching the AR block multiplies the parameter count by
            ``K`` and is rarely what identifies the regimes.

    Raises:
        SpecificationError: If ``order`` is not a non-negative integer,
            ``n_regimes`` is below two, or no block switches.
        DimensionError: If ``endog`` is not one-dimensional or has fewer
            than ``K(p + 2)`` observations.
        NumericalError: If ``endog`` contains non-finite values.

    See Also:
        * :class:`MarkovSwitchingARResult` -- what :meth:`fit` returns.
        * :class:`~cultivars.univariate.threshold.SETAR` -- regimes
          switched by an observed variable.
        * :class:`~cultivars.multivariate.regime_switching.markov_switching.MarkovSwitchingVAR`
          -- the vector counterpart.

    References:
        Hamilton, J. D. (1989). A new approach to the economic analysis
        of nonstationary time series and the business cycle.
        *Econometrica*, 57(2), 357-384.

        Hamilton, J. D. (1990). Analysis of time series subject to
        changes in regime. *Journal of Econometrics*, 45(1-2), 39-70.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p = np.array([[0.97, 0.03], [0.05, 0.95]])
        >>> mu = np.array([-1.0, 1.5])
        >>> s = np.zeros(1500, dtype=int)
        >>> for t in range(1, 1500):
        ...     s[t] = np.searchsorted(np.cumsum(p[s[t - 1]]), rng.random())
        >>> y = mu[s] + 0.5 * rng.standard_normal(1500)
        >>> res = MarkovSwitchingAR(y, order=1, n_regimes=2).fit(seed=0)
        >>> bool(res.intercepts[0] < res.intercepts[1])
        True
        >>> bool(np.allclose(res.expected_durations, [1 / 0.03, 1 / 0.05], rtol=0.25))
        True

        The switching blocks are the specification, and a model with none
        is refused:

        >>> MarkovSwitchingAR(y, order=1, switching_variance=False).fit(seed=0).specification
        'MSI(2)-AR(1)'
        >>> MarkovSwitchingAR(y, order=1, switching_mean=False, switching_variance=False)
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
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
    ) -> MarkovSwitchingARResult:
        """Estimate by EM with multi-start screening.

        Unlike the other families, this ``fit`` takes arguments, because the
        likelihood is multimodal and the answer genuinely depends on where the
        search starts. The default is to screen ``n_init`` starts for
        ``screen_iter`` iterations each, then refine only the best; pass a
        ``seed`` when you need the fit to be reproducible. The first start
        is deterministic -- intercepts at the within-regime quantiles of
        the series with a 0.9 diagonal -- and the rest draw intercepts from
        the sample and a diagonal in ``(0.8, 0.95)``, so ``n_init=1`` is
        the same fit for every seed.

        Args:
            max_iter: Iteration cap for the refining run.
            tol: Convergence tolerance on the log-likelihood increment.
            n_init: Random starts to screen.
            screen_iter: Iterations used to score each screening start.
            seed: Seed or generator for the random starts.

        Returns:
            The fitted :class:`MarkovSwitchingARResult`, regimes ordered by
            the convention :attr:`MarkovSwitchingARResult.label_ordering`
            names.

        Raises:
            NumericalError: If every start fails to produce a finite likelihood.

        Example:
            Two seeds reach the same optimum on well-separated regimes:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> s = (np.arange(600) // 100) % 2
            >>> y = np.where(s == 1, 2.0, -2.0) + 0.5 * rng.standard_normal(600)
            >>> model = MarkovSwitchingAR(y, order=1)
            >>> a, b = model.fit(seed=0, n_init=3), model.fit(seed=7, n_init=3)
            >>> bool(abs(a.llf - b.llf) < 1e-4), a.converged and b.converged
            (True, True)
        """
        return MarkovSwitchingARResult._from_fit(
            self._fit_family(
                max_iter=max_iter,
                tol=tol,
                n_init=n_init,
                screen_iter=screen_iter,
                seed=seed,
            ),
            self,
        )
