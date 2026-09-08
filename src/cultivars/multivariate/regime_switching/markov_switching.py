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
:attr:`MarkovSwitchingVARResult.label_ordering` names it. And the number of regimes cannot
be tested by a likelihood ratio -- under the null of fewer regimes the extra
regime's parameters are unidentified and its transition probabilities sit on
the boundary -- so that specific comparison is refused while tests holding
``M`` fixed remain available.

The composition hook is :meth:`MarkovSwitchingVARResult.regime`. Conditional on the chain
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
from ..structural import RecursiveSVAR

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
    regimes: tuple[_RegimeSystemResult, ...] = field(repr=False)
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
        model: _MarkovSwitchingVectorAutoRegressionModel[MarkovSwitchingVARResult],
    ) -> MarkovSwitchingVARResult:
        """Assemble the public result, building one stable view per regime."""
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


class MarkovSwitchingVAR(_MarkovSwitchingVectorAutoRegressionModel[MarkovSwitchingVARResult]):
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
        >>> res = MarkovSwitchingVAR(y, order=1, n_regimes=2).fit(seed=0, n_init=3)
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
    """An identified Markov-switching system, one structural result per regime.

    Composition on both sides: :attr:`msvar` is the reduced form with its
    chain, posteriors, and regime views, and :attr:`structurals` holds each
    regime's identification with its whole surface -- impact matrix,
    responses, diagnostics. This object owns the join: the guarantee of a
    common identifying declaration, and per-regime answers under one roof.

    Attributes:
        msvar: The fitted Markov-switching reduced form.
        structurals: One structural result per regime, in label order.
    """

    msvar: MarkovSwitchingVARResult = field(repr=False)
    structurals: tuple[StructuralResult, ...] = field(repr=False)

    @property
    def n_regimes(self) -> int:
        """Number of regimes."""
        return self.msvar.n_regimes

    @property
    def names(self) -> tuple[str, ...]:
        """Variable labels, in column order."""
        return self.msvar.names

    @property
    def shock_names(self) -> tuple[str, ...]:
        """Labels of the identified shocks, common across regimes."""
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
        """
        return self.structural(regime).irf(horizon, cumulative=cumulative)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
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

    Constructs with a fitted :class:`~cultivars.multivariate.regime_switching.MarkovSwitchingVAR`
    result. Not an ``_IdentificationModel``: that contract is one closed
    system, and an MS-VAR has ``M`` of them -- the regime views, each of
    which any scheme in this package already accepts directly. What this
    model adds is the discipline of one declaration for all regimes, and a
    packaged per-regime answer.

    Args:
        msvar: The fitted Markov-switching reduced form.
        structurals: Optional per-regime identifications to package, one per
            regime in label order, each produced by a point-identified model
            applied to the corresponding ``msvar.regime(m)``. When given,
            ``order`` must be omitted.
        order: The recursive ordering for the default scheme, applied
            identically in every regime. ``None`` orders the system as
            ``names`` stands.

    Raises:
        SpecificationError: If ``msvar`` is not a fitted MS-VAR result, both
            ``structurals`` and ``order`` are given, the count is wrong, or a
            supplied identification was not built from this MS-VAR's own
            regime view.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> from cultivars.multivariate.regime_switching import MarkovSwitchingVAR
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
    """

    __slots__ = ("_msvar", "_order", "_structurals")

    def __init__(
        self,
        msvar: MarkovSwitchingVARResult,
        structurals: Sequence[StructuralResult] | None = None,
        *,
        order: Sequence[str] | None = None,
    ) -> None:
        """Validate the reduced form and the identification route."""
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
    """A fitted switching factor model: the factor, and the regime chronology.

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
    """

    panel: npt.NDArray[np.float64] = field(repr=False)
    series_names: tuple[str, ...]
    loadings: npt.NDArray[np.float64] = field(repr=False)
    idiosyncratic_var: npt.NDArray[np.float64] = field(repr=False)
    phi: float
    sigma_v: float
    mu: npt.NDArray[np.float64]
    regime_transition: npt.NDArray[np.float64] = field(repr=False)
    factor: npt.NDArray[np.float64] = field(repr=False)
    filtered_prob: npt.NDArray[np.float64] = field(repr=False)
    smoothed_prob: npt.NDArray[np.float64] = field(repr=False)
    means: npt.NDArray[np.float64] = field(repr=False)
    scales: npt.NDArray[np.float64] = field(repr=False)
    llf: float
    n_params: int
    converged: bool

    @property
    def n_series(self) -> int:
        """Number of series in the panel."""
        return len(self.series_names)

    @property
    def nobs(self) -> int:
        """Panel length."""
        return int(self.panel.shape[0])

    def recession_probabilities(self) -> npt.NDArray[np.float64]:
        """Smoothed probability of the low-mean regime, per period.

        The business-cycle chronology: with the factor built from
        pro-cyclical indicators, the low-mean state is the contraction.
        """
        return self.smoothed_prob[:, 0]

    def expected_durations(self) -> npt.NDArray[np.float64]:
        """Expected regime durations ``1 / (1 - p_jj)``, low state first."""
        staying = np.diag(self.regime_transition)
        return np.asarray(1.0 / np.maximum(1.0 - staying, 1e-12), dtype=np.float64)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
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
    costs minutes rather than seconds -- the filter runs four Kalman
    updates per period inside the optimizer -- which is the known price of
    the model and is stated rather than hidden.

    Args:
        panel: The observed ``(nobs, n_series)`` panel of coincident
            indicators; each series is standardized internally.
        series_names: One label per series. Defaults to ``x1 ... xN``.

    Raises:
        DimensionError: If the panel cannot support the model.

    Example:
        Fitting on simulated data with a recession regime takes a minute
        or two; see the build's verification for the recovery numbers.
    """

    __slots__ = ("_panel", "_series_names")

    def __init__(
        self,
        panel: npt.ArrayLike,
        *,
        series_names: Sequence[str] | None = None,
    ) -> None:
        """Validate the panel."""
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
        optimizer never sees a boundary.

        Args:
            max_iter: Optimizer iteration cap.

        Returns:
            The fitted :class:`MarkovSwitchingDFMResult`.

        Raises:
            NumericalError: If the likelihood cannot be evaluated at the
                starting point.
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
