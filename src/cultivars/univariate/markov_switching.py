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

"""Markov-switching autoregression: the latent-regime counterpart to SETAR.

A ``K``-regime autoregression whose intercept, autoregressive coefficients, and
innovation variance may each switch with a latent first-order Markov chain
``S_t in {0, ..., K-1}``::

    y_t = c_{S_t} + phi_{1,S_t} y_{t-1} + ... + phi_{p,S_t} y_{t-p} + eps_t,
    eps_t ~ N(0, sigma2_{S_t}),

with row-stochastic transitions ``P[i, j] = Pr(S_t = j | S_{t-1} = i)``.

Where :mod:`cultivars.univariate.threshold` computes the regime from something
observable, here the regime is never observed at all: what comes back is a
posterior over states at every date, and every reported quantity -- fitted
values, residuals, even the regime a date "is in" -- is an expectation under
that posterior rather than a fact about the data. The results are shaped to
make that distinction hard to lose track of.

Which blocks switch is chosen at construction, so one class spans the Krolzig
(1997) taxonomy and :meth:`MSARResult.specification` reports which member you
actually fitted: ``I`` for a switching intercept, ``A`` for switching
autoregressive coefficients, ``H`` for a switching variance. Hamilton's
recession-dating model is ``MSIH(2)-AR(4)``, or
``MSAR(y, order=4, n_regimes=2)`` here.

This is the *intercept*-switching parameterization. The regime enters
contemporaneously through ``c_{S_t}``, so conditional on the observed lags each
regime is linear and inference is a plain ``K``-state chain -- which is what
lets estimation ride on a Hamilton filter with ``K`` states rather than the
``K**(p+1)`` states that Hamilton's original *mean*-switching form requires.
The two coincide at ``p = 0`` and differ in transient dynamics otherwise.

Estimation is EM (Hamilton 1990): the E-step is a filter followed by a Kim
smoother, and the M-step updates the transition matrix from expected transition
counts and the coefficients by responsibility-weighted least squares. The
likelihood is multimodal, so :meth:`MSAR.fit` screens several random starts and
refines the best -- which is why ``fit`` takes a ``seed`` and why the result
reports whether it converged.

Two identification facts shape the public surface. First, the likelihood is
invariant to permuting regime labels, so a sorting convention is imposed and
:attr:`MSARResult.label_ordering` names it rather than leaving it implicit.
Second, the number of regimes cannot be tested by a likelihood ratio: under the
null of ``K`` regimes, the parameters of the ``K + 1``-th and the transition
probabilities into it are unidentified, so the statistic has no chi-squared
limit. :meth:`MSARResult.likelihood_ratio_test` therefore refuses that specific
comparison while still permitting tests that hold ``K`` fixed.

References:
    Hamilton, J. D. (1989). A new approach to the economic analysis of
    nonstationary time series and the business cycle. *Econometrica*, 57(2).
    Hamilton, J. D. (1990). Analysis of time series subject to changes in
    regime. *Journal of Econometrics*, 45(1-2).
    Kim, C.-J. (1994). Dynamic linear models with Markov-switching. *Journal of
    Econometrics*, 60(1-2).
    Hansen, B. E. (1992). The likelihood ratio test under nonstandard
    conditions: testing the Markov switching model of GNP. *Journal of Applied
    Econometrics*, 7(S1).
    Krolzig, H.-M. (1997). *Markov-Switching Vector Autoregressions*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import InformationCriteria, ProbabilityType, SummaryTable, validate_choice
from .._internals import (
    _ComparisonMixin,
    _MarkovSwitchingFit,
    _MarkovSwitchingModel,
    _SeriesMixin,
    _StabilityResult,
    _SummaryMixin,
)
from ..exceptions import SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MSARResult(_SummaryMixin, _SeriesMixin, _ComparisonMixin):
    """A fitted Markov-switching autoregression.

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
    """

    endog: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int
    order: int
    n_regimes: int
    switching_mean: bool
    switching_ar: bool
    switching_variance: bool
    transition: npt.NDArray[np.float64]
    intercepts: npt.NDArray[np.float64]
    ar_params: npt.NDArray[np.float64]
    variances: npt.NDArray[np.float64]
    filtered_prob: npt.NDArray[np.float64]
    predicted_prob: npt.NDArray[np.float64]
    smoothed_prob: npt.NDArray[np.float64]
    ergodic_prob: npt.NDArray[np.float64]
    expected_durations: npt.NDArray[np.float64]
    n_iter: int
    converged: bool

    @classmethod
    def _from_fit(
        cls, fit: _MarkovSwitchingFit, model: _MarkovSwitchingModel[MSARResult]
    ) -> MSARResult:
        """Assemble the public result from a raw fit and its specification."""
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

    # -- specification -----------------------------------------------------

    @property
    def specification(self) -> str:
        """Krolzig code for which blocks switch, e.g. ``"MSIH(2)-AR(4)"``.

        ``I`` marks a switching intercept, ``A`` switching autoregressive
        coefficients, ``H`` a switching variance -- so the code names the model
        you actually fitted rather than the family it came from. The
        constructor guarantees at least one letter is present.
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
        """
        if self.switching_mean:
            return "intercept"
        if self.switching_variance:
            return "variance"
        return "first AR coefficient"

    # -- regime inference --------------------------------------------------

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
        """
        return np.argmax(self.smoothed_prob, axis=1).astype(np.int64)

    @property
    def regime_shares(self) -> npt.NDArray[np.float64]:
        """Average smoothed probability of each regime over the sample.

        The posterior counterpart to a frequency count, and the honest one:
        counting :attr:`most_likely_regime` throws away every date the model was
        genuinely uncertain about. Compare against :attr:`ergodic_prob` -- a
        large gap says the sample is not representative of the fitted chain.
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
        """
        base = super(MSARResult, self)._series()
        for j in range(self.n_regimes):
            base[f"smoothed_prob_{j}"] = self.smoothed_prob[:, j]
        base["regime"] = self.most_likely_regime.astype(np.float64)
        return base

    # -- per-regime dynamics -----------------------------------------------

    def regime_stability(self, regime: int) -> _StabilityResult:
        """Companion-eigenvalue verdict for one regime's autoregressive block.

        Args:
            regime: Regime index in ``0..K-1``.

        Returns:
            The :class:`_StabilityResult` for that regime read as a linear
            autoregression.

        Raises:
            SpecificationError: If ``regime`` is out of range.
        """
        if not 0 <= regime < self.n_regimes:
            raise SpecificationError(f"regime must be in 0..{self.n_regimes - 1}; got {regime}.")
        return _StabilityResult.assess_stability(self.ar_params[regime])

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
        """
        return all(self.regime_stability(j).is_stable for j in range(self.n_regimes))

    @property
    def unconditional_mean(self) -> float | None:
        """Ergodic mean of the process, or ``None`` if it is not defined.

        Computed as the ergodic-probability-weighted average of each regime's
        own unconditional mean ``c_j / (1 - sum phi_j)``. Returns ``None`` when
        any regime is non-stationary, since that regime has no unconditional
        mean to average in, and when the autoregressive sum sits at one.
        """
        if not self.is_regimewise_stationary:
            return None
        denominators = 1.0 - self.ar_params.sum(axis=1)
        if np.any(np.abs(denominators) < 1e-12):
            return None
        return float(np.sum(self.ergodic_prob * (self.intercepts / denominators)))

    # -- tables ------------------------------------------------------------

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order.

        Regime-indexed names carry the regime in brackets, and transition
        entries read ``p[i->j]``. All ``K**2`` transition entries appear even
        though only ``K(K-1)`` are free, because a row displayed without its
        final element is harder to read than one whose redundancy is stated in
        the notes.
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
        """The estimated transition matrix, rows indexed by the origin regime.

        Returns:
            A :class:`SummaryTable` whose entry ``(i, j)`` is
            ``Pr(S_t = j | S_{t-1} = i)``.
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

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        return self.specification

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
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
        """
        other = counterpart.n_regimes if isinstance(counterpart, MSARResult) else 1
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


class MSAR(_MarkovSwitchingModel[MSARResult]):
    """Markov-switching autoregression with ``K`` latent regimes.

    Args:
        endog: The series.
        order: Autoregressive order ``p`` within each regime.
        n_regimes: Number of regimes ``K``, at least two.
        switching_mean: Whether the intercept switches.
        switching_variance: Whether the innovation variance switches.
        switching_ar: Whether the autoregressive coefficients switch. Off by
            default: switching the AR block multiplies the parameter count by
            ``K`` and is rarely what identifies the regimes.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> p = np.array([[0.97, 0.03], [0.05, 0.95]])
        >>> mu = np.array([-1.0, 1.5])
        >>> s = np.zeros(1500, dtype=int)
        >>> for t in range(1, 1500):
        ...     s[t] = np.searchsorted(np.cumsum(p[s[t - 1]]), rng.random())
        >>> y = mu[s] + 0.5 * rng.standard_normal(1500)
        >>> res = MSAR(y, order=1, n_regimes=2).fit(seed=0)
        >>> bool(res.intercepts[0] < res.intercepts[1])
        True
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
    ) -> MSARResult:
        """Estimate by EM with multi-start screening.

        Unlike the other families, this ``fit`` takes arguments, because the
        likelihood is multimodal and the answer genuinely depends on where the
        search starts. The default is to screen ``n_init`` starts for
        ``screen_iter`` iterations each, then refine only the best; pass a
        ``seed`` when you need the fit to be reproducible.

        Args:
            max_iter: Iteration cap for the refining run.
            tol: Convergence tolerance on the log-likelihood increment.
            n_init: Random starts to screen.
            screen_iter: Iterations used to score each screening start.
            seed: Seed or generator for the random starts.

        Returns:
            The fitted :class:`MSARResult`, regimes ordered by the convention
            :attr:`MSARResult.label_ordering` names.

        Raises:
            NumericalError: If every start fails to produce a finite likelihood.
        """
        return MSARResult._from_fit(
            self._fit_family(
                max_iter=max_iter,
                tol=tol,
                n_init=n_init,
                screen_iter=screen_iter,
                seed=seed,
            ),
            self,
        )
