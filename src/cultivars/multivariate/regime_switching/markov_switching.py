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

"""The Markov-switching VAR: latent regimes, and a posterior instead of a date.

An ``M``-regime vector autoregression whose deterministic block, lag
coefficients, and innovation covariance may each switch with a latent
first-order Markov chain (Krolzig 1997; Sims & Zha 2006). Where the threshold
and smooth-transition families compute their regime from an observable, here
the regime is never observed at all: what comes back is a posterior over
states at every date, and every reported quantity -- fitted values,
residuals, even the regime a date "is in" -- is an expectation under that
posterior rather than a fact about the data.

Estimation is EM on the same Hamilton filter and Kim smoother that power the
univariate family: the E-step filters and smooths per-regime multivariate
Gaussian densities, and the M-step updates the transition matrix from
expected transition counts and all regimes' coefficient blocks in one
probability-weighted GLS system, so a non-switching block is estimated
jointly across regimes rather than per regime and then averaged. The
likelihood is multimodal, so ``fit`` screens several starts and refines the
best -- which is why it takes a ``seed`` and reports convergence.

Two facts about mixtures shape the surface. The likelihood is invariant to
permuting regime labels, so a sorting convention is imposed and
:attr:`MSVARResult.label_ordering` names it. And the number of regimes cannot
be tested by a likelihood ratio -- under the null of fewer regimes the extra
regime's parameters are unidentified and its transition probabilities sit on
the boundary -- so that specific comparison is refused while tests holding
``M`` fixed remain available.

The composition hook is :meth:`MSVARResult.regime`. Conditional on the chain
sitting in regime ``m``, the model *is* a linear VAR, and the regime view is
that system dressed as a closed reduced form: every identification model in
:mod:`cultivars.multivariate.structural` accepts it directly, which is what
:class:`~cultivars.multivariate.structural.MarkovSwitchingSVAR` builds on.

References:
    Krolzig, H.-M. (1997). *Markov-Switching Vector Autoregressions*.
        Springer.
    Hamilton, J. D. (1990). Analysis of time series subject to changes in
        regime. *Journal of Econometrics*, 45(1-2), 39-70.
    Kim, C.-J. (1994). Dynamic linear models with Markov-switching.
        *Journal of Econometrics*, 60(1-2), 1-22.
    Sims, C. A., & Zha, T. (2006). Were there regime switches in U.S.
        monetary policy? *American Economic Review*, 96(1), 54-81.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import (
    InformationCriteria,
    ProbabilityType,
    SummaryTable,
    companion_matrix,
    validate_choice,
)
from ..._internals import (
    _ComparisonMixin,
    _MarkovSwitchingVectorAutoRegressionModel,
    _StabilityTest,
    _SummaryMixin,
    _VectorMarkovSwitchingFit,
)
from ...exceptions import SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MSVARRegime:
    """One regime of a fitted MS-VAR, dressed as a closed linear system.

    Conditional on the chain sitting in this regime the model is a linear
    VAR, and this view exposes exactly the closed-system surface an
    identification model reads: labels, coefficients, an innovation
    covariance, residuals, and a moving-average representation. It therefore
    passes anywhere a fitted VAR result passes as an identification source --
    ``RecursiveSVAR(msvar.regime(1))`` works verbatim.

    Two honesty notes on the statistical members. ``nobs`` is the *expected*
    number of periods spent in this regime -- the sum of its smoothed
    probabilities, a float, because no date belongs to a latent regime with
    certainty. ``resid`` is the full-sample residuals under this regime's
    parameters, weighted by the square root of the smoothed probabilities and
    scaled so their second moment matches :attr:`sigma_u`; a scheme that only
    reads the covariance (recursive, long-run, short-run) is unaffected,
    while a scheme that reads residual moments directly is consuming a
    posterior-weighted object and should be interpreted accordingly.

    Attributes:
        index: This regime's label under the fit's ordering convention.
        names: Variable labels, in column order.
        order: Autoregressive order.
        trend: Deterministic specification.
        coefficients: ``(p, k, k)`` lag stack of this regime.
        deterministic: This regime's deterministic coefficients.
        sigma_u: This regime's innovation covariance.
        resid: Probability-weighted residuals, aligned with the effective
            sample.
        weight: This regime's smoothed probabilities, one per effective row.
        nobs: Expected periods spent in this regime.
    """

    index: int
    names: tuple[str, ...]
    order: int
    trend: str
    coefficients: npt.NDArray[np.float64] = field(repr=False)
    deterministic: npt.NDArray[np.float64] = field(repr=False)
    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    resid: npt.NDArray[np.float64] = field(repr=False)
    weight: npt.NDArray[np.float64] = field(repr=False)
    nobs: float

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables."""
        return len(self.names)

    def stability_check(self) -> _StabilityTest:
        """Companion-eigenvalue verdict for this regime's lag stack."""
        return _StabilityTest.assess_stability(self.coefficients)

    @property
    def is_stable(self) -> bool:
        """Whether this regime, read as a linear VAR, is stable."""
        return self.stability_check().is_stable

    def ma_representation(self, horizon: int = 20) -> npt.NDArray[np.float64]:
        """Moving-average matrices of this regime, held frozen.

        Args:
            horizon: Largest lead to return.

        Returns:
            An array of shape ``(horizon + 1, k, k)`` with ``Psi_0 = I``.

        Raises:
            SpecificationError: If ``horizon`` is negative.
        """
        if horizon < 0:
            raise SpecificationError(f"horizon must be non-negative; got {horizon}.")
        k, p = self.k_endog, self.order
        out = np.empty((horizon + 1, k, k), dtype=np.float64)
        if p == 0:
            out[:] = 0.0
            out[0] = np.eye(k)
            return out
        selector = np.zeros((k, k * p), dtype=np.float64)
        selector[:, :k] = np.eye(k)
        power = np.eye(k * p, dtype=np.float64)
        companion = companion_matrix(self.coefficients)
        for h in range(horizon + 1):
            out[h] = selector @ power @ selector.T
            power = power @ companion
        return out


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MSVARResult(_SummaryMixin, _ComparisonMixin):
    """A fitted Markov-switching vector autoregression.

    Attributes:
        endog: The observed panel.
        names: Variable labels, in column order.
        order: Autoregressive order within each regime.
        trend: Deterministic specification.
        n_regimes: Number of latent regimes ``M``.
        switching_mean: Whether the deterministic block switches.
        switching_ar: Whether the lag coefficients switch.
        switching_variance: Whether the innovation covariance switches.
        transition: Row-stochastic ``(M, M)`` matrix.
        regimes: One closed-system view per regime, in label order; the
            composition hook the structural layer consumes.
        filtered_prob: ``Pr(S_t = m | y_1..t)``, shape ``(nobs, M)``.
        predicted_prob: ``Pr(S_t = m | y_1..t-1)``, shape ``(nobs, M)``.
        smoothed_prob: ``Pr(S_t = m | y_1..T)``, shape ``(nobs, M)``.
        ergodic_prob: Stationary distribution of ``transition``.
        expected_durations: ``1 / (1 - P_mm)``, in periods.
        fittedvalues: Posterior-weighted one-step means.
        resid: Residuals against those weighted means.
        llf: Log-likelihood from the final filter pass.
        nobs: Effective sample size.
        n_params: Free parameters: transitions, coefficient slabs, and the
            covariance block(s).
        label_ordering: Which quantity regimes were sorted by, ascending.
        n_iter: EM iterations used by the refining run.
        converged: Whether the refining run met its tolerance.
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    names: tuple[str, ...]
    order: int
    trend: str
    n_regimes: int
    switching_mean: bool
    switching_ar: bool
    switching_variance: bool
    transition: npt.NDArray[np.float64] = field(repr=False)
    regimes: tuple[MSVARRegime, ...] = field(repr=False)
    filtered_prob: npt.NDArray[np.float64] = field(repr=False)
    predicted_prob: npt.NDArray[np.float64] = field(repr=False)
    smoothed_prob: npt.NDArray[np.float64] = field(repr=False)
    ergodic_prob: npt.NDArray[np.float64] = field(repr=False)
    expected_durations: npt.NDArray[np.float64] = field(repr=False)
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    resid: npt.NDArray[np.float64] = field(repr=False)
    llf: float
    nobs: int
    n_params: float
    label_ordering: str
    n_iter: int
    converged: bool

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorMarkovSwitchingFit,
        model: _MarkovSwitchingVectorAutoRegressionModel[MSVARResult],
    ) -> MSVARResult:
        """Assemble the public result, building one stable view per regime."""
        target = model.endog[model.order :]
        views: list[MSVARRegime] = []
        for m in range(model.n_regimes):
            weight = fit.smoothed_prob[:, m]
            stack = fit.coefficients[m]
            deterministic = fit.deterministics[m]
            resid_m = target - cls._conditional_mean(model, stack, deterministic)
            total = max(float(weight.sum()), 1e-12)
            scaled = resid_m * np.sqrt(weight * (target.shape[0] / total))[:, None]
            views.append(
                MSVARRegime(
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
        model: _MarkovSwitchingVectorAutoRegressionModel[MSVARResult],
        stack: npt.NDArray[np.float64],
        deterministic: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """One regime's one-step conditional means over the effective sample."""
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
        """Number of endogenous variables."""
        return len(self.names)

    @property
    def specification(self) -> str:
        """Krolzig code for which blocks switch, e.g. ``"MSIH(2)-VAR(1)"``."""
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

    def regime(self, index: int) -> MSVARRegime:
        """One regime's closed-system view, identity-stable across calls.

        Args:
            index: Regime label in ``0..M-1``, under the fit's ordering
                convention.

        Returns:
            The :class:`MSVARRegime`. The same object every call, so an
            identification result built from it remembers its source by
            identity.

        Raises:
            SpecificationError: If ``index`` is out of range.
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
        """
        return np.argmax(self.smoothed_prob, axis=1).astype(np.int64)

    @property
    def regime_shares(self) -> npt.NDArray[np.float64]:
        """Average smoothed probability of each regime over the sample."""
        return np.asarray(self.smoothed_prob.mean(axis=0), dtype=np.float64)

    @property
    def regime_uncertainty(self) -> float:
        """Mean posterior entropy, normalized so ``0`` is sharp and ``1`` flat.

        Near zero, the regimes are cleanly separated; near one, the data
        barely distinguish them at all -- which no coefficient table shows.
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
        """
        other = counterpart.n_regimes if isinstance(counterpart, MSVARResult) else 1
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
        """Short specification label for a ranking table."""
        return self.specification

    def regime_table(self) -> SummaryTable:
        """One row per regime: scale, persistence, and how much sample it owns.

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
        """Build the structured summary."""
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


class MSVAR(_MarkovSwitchingVectorAutoRegressionModel[MSVARResult]):
    """Markov-switching vector autoregression with ``M`` latent regimes.

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

    Example:
        >>> rng = np.random.default_rng(0)
        >>> p = np.array([[0.97, 0.03], [0.05, 0.95]])
        >>> mu = np.array([[-1.0, -0.5], [1.5, 1.0]])
        >>> s = np.zeros(600, dtype=int)
        >>> y = np.zeros((600, 2))
        >>> for t in range(1, 600):
        ...     s[t] = np.searchsorted(np.cumsum(p[s[t - 1]]), rng.random())
        ...     y[t] = mu[s[t]] + 0.3 * y[t - 1] + 0.4 * rng.standard_normal(2)
        >>> res = MSVAR(y, order=1, n_regimes=2).fit(seed=0, n_init=3)
        >>> bool(res.regimes[0].deterministic[0, 0] < res.regimes[1].deterministic[0, 0])
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
    ) -> MSVARResult:
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
            The fitted :class:`MSVARResult`, regimes ordered by the
            convention :attr:`MSVARResult.label_ordering` names.

        Raises:
            NumericalError: If every start fails to produce a finite
                likelihood.
        """
        return MSVARResult._from_fit(
            self._fit_markov(
                max_iter=max_iter,
                tol=tol,
                n_init=n_init,
                screen_iter=screen_iter,
                seed=seed,
            ),
            self,
        )
