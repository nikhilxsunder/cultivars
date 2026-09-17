from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt

from .._core import (
    _CRITICAL_LEVELS,
    _DEFAULT_ALPHA,
    _EXTREME_PVALUE,
    _MIN_CHAIN_DRAWS,
    _MIN_ESS_PER_CHAIN,
    _RHAT_TOL,
    SummaryTable,
    _ess_bulk,
    _ess_tail,
    _geweke,
    _mcse_mean,
    _predictive_pvalues,
    _rhat,
)
from ..exceptions import DimensionError, SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _JohansenRankTest:
    """Sequential trace and maximum-eigenvalue tests for the cointegrating rank.

    Both sequences read down from ``r = 0``. The trace statistic tests
    ``rank <= r`` against ``rank = k``; the maximum-eigenvalue statistic tests
    ``rank = r`` against ``rank = r + 1``. The conventional reading stops at the
    first ``r`` that is not rejected, which :meth:`selected_rank` implements and
    which is a sequential procedure whose overall size is not the nominal level
    of any single step -- a caveat the table states rather than hides.

    Attributes:
        eigenvalues: The squared canonical correlations, descending. One per
            modelled equation, since a conditional system cannot support more
            cointegrating relations than it has equations.
        trace_statistic: One entry per null ``rank <= r``, ``r = 0 .. k - 1``.
        max_eigenvalue_statistic: One entry per null ``rank = r``.
        trace_pvalue: Empirical p-values from the simulated null.
        max_eigenvalue_pvalue: Empirical p-values from the simulated null.
        nobs: Effective sample the statistics were computed on.
        deterministic: The Johansen case the null was simulated under.
        k_exog: Weakly exogenous integrated regressors the null accounted for.
        simulations: Replications behind each p-value, so its resolution is
            visible rather than implied.
        small_sample: Whether the Reinsel-Ahn degrees-of-freedom scaling was
            applied to the statistics.
    """

    eigenvalues: npt.NDArray[np.float64]
    trace_statistic: npt.NDArray[np.float64]
    max_eigenvalue_statistic: npt.NDArray[np.float64]
    trace_pvalue: npt.NDArray[np.float64]
    max_eigenvalue_pvalue: npt.NDArray[np.float64]
    nobs: int
    deterministic: str
    k_exog: int
    simulations: int
    small_sample: bool

    @property
    def k_endog(self) -> int:
        """Number of modelled equations, which bounds the rank."""
        return int(self.eigenvalues.shape[0])

    @property
    def ranks(self) -> tuple[int, ...]:
        """The null ranks tested, in order."""
        return tuple(range(self.k_endog))

    def selected_rank(self, *, alpha: float = 0.05, statistic: str = "trace") -> int:
        """First rank the sequence fails to reject.

        Args:
            alpha: Level for each step of the sequence.
            statistic: ``"trace"`` or ``"max_eigenvalue"``.

        Returns:
            The chosen rank, or ``k_endog`` if every null is rejected, which is
            the finding that the system is stationary in levels and does not
            want a vector error-correction representation at all.

        Raises:
            SpecificationError: If the level or the statistic is unrecognized.
        """
        if not 0.0 < alpha < 1.0:
            raise SpecificationError(f"alpha must lie strictly in (0, 1); got {alpha}.")
        if statistic == "trace":
            pvalues = self.trace_pvalue
        elif statistic == "max_eigenvalue":
            pvalues = self.max_eigenvalue_pvalue
        else:
            raise SpecificationError(
                f"statistic must be 'trace' or 'max_eigenvalue'; got {statistic!r}."
            )
        for rank, pvalue in enumerate(pvalues):
            if pvalue >= alpha:
                return rank
        return self.k_endog

    def to_table(self, *, alpha: float = 0.05) -> SummaryTable:
        """Render both sequences with the rank each one selects."""
        chosen = self.selected_rank(alpha=alpha)
        notes = [
            f"Read down from r = 0 and stop at the first row not rejected: trace selects "
            f"r = {chosen}, max-eigenvalue selects "
            f"r = {self.selected_rank(alpha=alpha, statistic='max_eigenvalue')}.",
            "This is a sequential procedure, so the overall probability of an incorrect "
            f"rank is not the {alpha:.0%} attached to any single row.",
            f"P-values are empirical over {self.simulations:,} draws from the simulated "
            f"asymptotic null for the '{self.deterministic}' case, so the smallest "
            f"resolvable value is {1 / self.simulations:.0e}.",
        ]
        if self.k_exog:
            notes.append(
                f"The null accounts for {self.k_exog} weakly exogenous integrated "
                "regressor(s); its critical values are materially larger than the "
                "unconditional Johansen ones and the two are not interchangeable."
            )
        if self.small_sample:
            notes.append(
                "Statistics carry the Reinsel-Ahn degrees-of-freedom scaling, which pulls "
                "the well-known upward size distortion of the asymptotic test back toward "
                "its nominal level in short samples."
            )
        return SummaryTable(
            title="Johansen cointegration rank test",
            metadata=(
                ("Observations", f"{self.nobs}"),
                ("Variables", f"{self.k_endog}"),
                ("Deterministic", self.deterministic),
                ("Exogenous I(1)", f"{self.k_exog}"),
                ("Selected rank", f"{chosen}"),
            ),
            columns=("H0", "eigenvalue", "trace", "P>trace", "max-eig", "P>max-eig"),
            rows=tuple(
                (
                    f"r <= {rank}",
                    f"{self.eigenvalues[rank]:.4f}",
                    f"{self.trace_statistic[rank]:.3f}",
                    f"{self.trace_pvalue[rank]:.4f}",
                    f"{self.max_eigenvalue_statistic[rank]:.3f}",
                    f"{self.max_eigenvalue_pvalue[rank]:.4f}",
                )
                for rank in self.ranks
            ),
            notes=tuple(notes),
        )

    def __repr__(self) -> str:
        """One line with the rank each sequence selects."""
        return (
            f"JohansenRankTest(k={self.k_endog}, nobs={self.nobs}, "
            f"deterministic={self.deterministic!r}, "
            f"trace_rank={self.selected_rank()}, "
            f"max_eigenvalue_rank={self.selected_rank(statistic='max_eigenvalue')})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _HypothesisTest:
    """A scalar test statistic, its p-value, and a verdict at a level.

    Every hypothesis-test record in the package descends from this one:
    a study that collects verdicts wants one type with ``reject`` and
    ``summary``, not a union of seven. Subclasses add the fields their
    construction needs and describe their own table through
    :meth:`_summary_table`; the renderers -- ``summary()``, ``str()``,
    the notebook HTML -- are derived here once. ``__repr__`` stays a
    one-line verdict, because a test in a REPL should read as a sentence
    and not as a table.

    Attributes:
        statistic: The test statistic.
        pvalue: Its p-value, or ``None`` for a test known only through
            tabulated critical values.
    """

    statistic: float
    pvalue: float | None

    def reject(self, *, alpha: float = _DEFAULT_ALPHA) -> bool:
        """Whether the null is rejected at level ``alpha``.

        Raises:
            SpecificationError: If the level is not inside ``(0, 1)`` or
                the test carries no p-value.
        """
        if not 0.0 < alpha < 1.0:
            raise SpecificationError(f"alpha must lie strictly inside (0, 1); got {alpha}.")
        if self.pvalue is None:
            raise SpecificationError(
                f"{type(self).__name__} carries no p-value; read its critical values."
            )
        return self.pvalue < alpha

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary for this test.

        Raises:
            NotImplementedError: If the concrete test does not supply one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _summary_table() to be displayable."
        )

    def summary(self) -> SummaryTable:
        """The test as a table, renderable as text, HTML, or a dataframe."""
        return self._summary_table()

    def __str__(self) -> str:
        """Render the summary."""
        return self._summary_table().to_text()

    def _repr_html_(self) -> str:
        """Render the summary as HTML for Jupyter."""
        return self._summary_table()._repr_html_()

    def _pvalue_text(self) -> str:
        """The p-value for a one-line repr."""
        return "None" if self.pvalue is None else f"{self.pvalue:.4g}"

    def __repr__(self) -> str:
        """One-line verdict."""
        name = type(self).__name__.lstrip("_")
        return f"{name}(statistic={self.statistic:.4f}, pvalue={self._pvalue_text()})"


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _ChiSquaredTest(_HypothesisTest):
    """A test referred to a chi-squared law with ``df`` degrees of freedom.

    The likelihood-ratio and Wald records are siblings here rather than
    parent and child: they carry the same three numbers, but a
    likelihood ratio is named by its construction and a Wald statistic
    by the null it was aimed at, and the one extra field is the
    difference between them.

    Attributes:
        statistic: The chi-squared statistic.
        df: Degrees of freedom.
        pvalue: Upper-tail probability under the chi-squared null.
    """

    df: int
    pvalue: float

    def _label(self) -> str:
        """The row label naming what was tested."""
        return "restriction"

    def _title(self) -> str:
        """The table title."""
        return "Chi-Squared Test"

    def _summary_table(self) -> SummaryTable:
        """One row: the statistic, its degrees of freedom, and the p-value."""
        verdict = "reject" if self.reject() else "keep"
        return SummaryTable(
            title=self._title(),
            metadata=(("Verdict at 5%", verdict),),
            columns=("null", "statistic", "df", "p-value"),
            rows=((self._label(), f"{self.statistic:.4f}", str(self.df), f"{self.pvalue:.4f}"),),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _ForecastComparisonTest(_HypothesisTest):
    """A test on a series of aligned forecast errors from a rolling evaluation.

    The three members -- Clark-West, Mincer-Zarnowitz, encompassing --
    share the harness that produced their inputs and therefore the same
    header: how many origins, at what horizon, and the verdict.

    Attributes:
        statistic: The test statistic.
        pvalue: Its p-value.
        horizon: Forecast horizon behind the series.
        nobs: Evaluation origins.
    """

    pvalue: float
    horizon: int
    nobs: int

    def _frame(
        self,
        *,
        title: str,
        verdict: str,
        columns: tuple[str, ...],
        rows: tuple[tuple[str, ...], ...],
        notes: tuple[str, ...],
    ) -> SummaryTable:
        """Assemble the family's table around its common header."""
        return SummaryTable(
            title=title,
            metadata=(
                ("Origins", str(self.nobs)),
                ("Horizon", str(self.horizon)),
                ("Verdict", verdict),
            ),
            columns=columns,
            rows=rows,
            notes=notes,
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _TabulatedTest(_HypothesisTest):
    """A test read against tabulated critical values, with or without a p-value.

    Unit-root and structural-break tests have non-standard limit laws.
    Where the literature supplies a response surface or the limit can be
    simulated, ``pvalue`` is set and ``reject`` reads it; where only
    three critical values exist, ``pvalue`` is ``None`` and ``reject``
    reads the table in the direction ``lower_tail`` says. Several
    statistics computed on the same sample by the same test -- the four
    Ng-Perron statistics, the sup-, exp- and ave-Wald functionals --
    travel as ``companions``, each a full record, so a table can show
    them together and each can still be read alone.

    Attributes:
        name: The test.
        statistic: The test statistic.
        pvalue: Its p-value, or ``None`` when only critical values exist.
        critical_values: ``{"1%", "5%", "10%"}`` critical values.
        null: The hypothesis under test.
        lower_tail: Whether rejection lies in the lower tail.
        nobs: Observations the statistic was computed on.
        companions: Further statistics of the same test on the same
            sample.
    """

    name: str
    critical_values: dict[str, float]
    null: str
    nobs: int
    lower_tail: bool = False
    companions: tuple[_TabulatedTest, ...] = ()

    def reject(self, *, alpha: float = _DEFAULT_ALPHA) -> bool:
        """Whether the null is rejected at level ``alpha``.

        By p-value when one exists; otherwise by the tabulated critical
        value, which requires ``alpha`` to be a tabulated level.

        Raises:
            SpecificationError: If the level is not inside ``(0, 1)``, or
                the test carries no p-value and ``alpha`` is not a
                tabulated level.
        """
        if self.pvalue is not None:
            return _HypothesisTest.reject(self, alpha=alpha)
        if not 0.0 < alpha < 1.0:
            raise SpecificationError(f"alpha must lie strictly inside (0, 1); got {alpha}.")
        label = f"{alpha * 100:g}%"
        if label not in self.critical_values:
            raise SpecificationError(
                f"{self.name} carries no p-value; alpha must be one of "
                f"{tuple(self.critical_values)} to use the tabulated critical values, got "
                f"{alpha}."
            )
        threshold = self.critical_values[label]
        return self.statistic < threshold if self.lower_tail else self.statistic > threshold

    def _row(self) -> tuple[str, ...]:
        """One table row: name, statistic, p-value, three critical values."""
        p = "" if self.pvalue is None else f"{self.pvalue:.4f}"
        cv = tuple(f"{self.critical_values[k]:.3f}" for k in _CRITICAL_LEVELS)
        return (self.name, f"{self.statistic:.4f}", p, *cv)

    def _verdict(self) -> str:
        """The verdict at the default level, or a pointer to the table."""
        try:
            rejected = self.reject()
        except SpecificationError:
            return "see critical values"
        return f"reject {self.null}" if rejected else f"keep {self.null}"

    def _title(self) -> str:
        """The table title."""
        return f"{self.name} Test"

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        """Header lines before the verdict; subclasses add their own."""
        return (("Null", self.null), ("Observations", str(self.nobs)))

    def _notes(self) -> tuple[str, ...]:
        """Closing lines."""
        tail = "lower" if self.lower_tail else "upper"
        return (f"Rejection lies in the {tail} tail.",)

    def _summary_table(self) -> SummaryTable:
        """The statistic and its companions against the critical values."""
        return SummaryTable(
            title=self._title(),
            metadata=(*self._metadata(), ("Verdict at 5%", self._verdict())),
            columns=("statistic", "value", "p-value", *_CRITICAL_LEVELS),
            rows=(self._row(), *(c._row() for c in self.companions)),
            notes=self._notes(),
        )

    def _repr_fields(self) -> tuple[str, ...]:
        """Extra ``key=value`` pieces for the one-line repr."""
        return ()

    def __repr__(self) -> str:
        """One-line verdict."""
        pieces = (
            f"name={self.name!r}",
            f"statistic={self.statistic:.4f}",
            f"pvalue={self._pvalue_text()}",
            *self._repr_fields(),
            f"nobs={self.nobs}",
        )
        return f"{type(self).__name__.lstrip('_')}({', '.join(pieces)})"


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _LikelihoodRatioTest(_ChiSquaredTest):
    """Verdict of a likelihood-ratio test between two nested fits.

    Attributes:
        statistic: ``2 * (llf_unrestricted - llf_restricted)``.
        df: Degrees of freedom, the difference in free parameter counts.
        pvalue: Upper-tail probability under a chi-squared null.
    """

    def _label(self) -> str:
        return "nested restriction"

    def _title(self) -> str:
        return "Likelihood-Ratio Test"

    def __repr__(self) -> str:
        """One-line verdict."""
        return (
            f"LikelihoodRatioTest(statistic={self.statistic:.4f}, df={self.df}, "
            f"pvalue={self.pvalue:.4g})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _WaldTest(_ChiSquaredTest):
    """Verdict of a chi-squared restriction test on a fitted model.

    Carries the same three numbers as :class:`_LikelihoodRatioResult` and one
    more, and that one is the reason they are separate classes. A
    likelihood-ratio test is named by its construction: two nested fits, one
    statistic, and the restriction is whatever distinguishes them, so the
    object needs no label. A Wald statistic is a *form* rather than a
    hypothesis -- the same quadratic in the same estimated covariance answers
    Granger causality, residual autocorrelation, normality, and conditional
    heteroskedasticity -- so a result that did not carry its own null would be
    four unrelated verdicts wearing one type and no way to tell them apart in
    a table.

    The distribution is asymptotic in every case. Wald statistics are also
    famously sensitive to how a nonlinear restriction is algebraically
    arranged, but every use here restricts coefficients to zero, which is
    linear and therefore invariant.

    Attributes:
        statistic: The chi-squared statistic.
        df: Degrees of freedom, the number of restrictions imposed.
        pvalue: Upper-tail probability under the chi-squared null.
        null: The hypothesis being tested, phrased so it reads as a sentence
            in a diagnostics table -- ``"gdp does not Granger-cause infl"``,
            not ``"granger"``.
    """

    statistic: float
    null: str

    def _label(self) -> str:
        return self.null

    def _title(self) -> str:
        return "Wald Test"

    def __repr__(self) -> str:
        """One-line verdict at the default level, which the text names."""
        verdict = "reject" if self.reject() else "keep"
        return (
            f"WaldTest(statistic={self.statistic:.4f}, df={self.df}, "
            f"pvalue={self.pvalue:.4g}, {verdict} at {_DEFAULT_ALPHA:.0%}: {self.null!r})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _ConvergenceTest:
    """Per-parameter convergence diagnostics of a Markov chain Monte Carlo run.

    One row per scalar quantity: its posterior mean and spread, the Monte
    Carlo standard error of that mean, rank-normalized bulk and tail
    effective sample sizes, rank-normalized split-R-hat with folding, and
    the largest-magnitude Geweke score across chains. The verdict applies
    the Vehtari et al. (2021) thresholds the report carries, so two reports
    are comparable only when their thresholds are.

    A parameter whose chain never moved has ``nan`` in every column and is
    listed under :attr:`degenerate` rather than counted against the verdict:
    the report cannot tell a parameter fixed by construction from a sampler
    that is stuck, and refuses to guess. Read the list.

    With a single chain, split-R-hat compares the two halves of that chain,
    which detects drift but not a chain stuck in one of several modes; the
    summary says so. The between-chain reading needs a second run with a
    different seed passed alongside the first. Two half-chains also make the
    statistic noisy when they are short: an independent sequence of 200 draws
    exceeds 1.01 about a fifth of the time and one of 500 draws about a
    fiftieth, so a flag close to the threshold on a short single chain is a
    reason to run longer, not a verdict.

    Attributes:
        names: Label of every scalar quantity, in row order.
        mean: ``(d,)`` posterior means.
        sd: ``(d,)`` posterior standard deviations.
        mcse: ``(d,)`` Monte Carlo standard errors of the means.
        ess_bulk: ``(d,)`` bulk effective sample sizes.
        ess_tail: ``(d,)`` tail effective sample sizes.
        rhat: ``(d,)`` split-R-hat statistics.
        geweke: ``(d,)`` signed Geweke scores, the one of largest magnitude
            across chains; ``nan`` when the chains are too short for the
            default segments.
        n_chains: Independent chains the diagnostics pooled.
        n_draws: Kept draws per chain.
        rhat_tol: R-hat above which a parameter is flagged.
        min_ess: Effective draws per chain below which a parameter is
            flagged, so the floor applied is ``min_ess * n_chains``.
        source: What was assessed, for the summary title.
    """

    names: tuple[str, ...]
    mean: npt.NDArray[np.float64] = field(repr=False)
    sd: npt.NDArray[np.float64] = field(repr=False)
    mcse: npt.NDArray[np.float64] = field(repr=False)
    ess_bulk: npt.NDArray[np.float64] = field(repr=False)
    ess_tail: npt.NDArray[np.float64] = field(repr=False)
    rhat: npt.NDArray[np.float64] = field(repr=False)
    geweke: npt.NDArray[np.float64] = field(repr=False)
    n_chains: int
    n_draws: int
    rhat_tol: float
    min_ess: float
    source: str

    @classmethod
    def _assess(
        cls,
        draws: npt.NDArray[np.float64],
        *,
        names: tuple[str, ...],
        rhat_tol: float = _RHAT_TOL,
        min_ess: float = _MIN_ESS_PER_CHAIN,
        source: str = "draws",
    ) -> Self:
        """Compute every diagnostic from ``(C, N, d)`` draws.

        Args:
            draws: ``C`` chains of ``N`` kept draws of ``d`` scalar quantities.
            names: ``d`` labels.
            rhat_tol: Flagging threshold for R-hat.
            min_ess: Flagging floor for effective draws, per chain.
            source: Label for the summary title.

        Raises:
            DimensionError: If ``draws`` is not three-dimensional or ``names``
                does not match its last axis.
            SpecificationError: If the thresholds are not positive or a chain
                is shorter than the minimum.
        """
        if draws.ndim != 3:
            raise DimensionError(f"draws must be (chains, draws, d); got shape {draws.shape}.")
        n_chains, n_draws, d = draws.shape
        if len(names) != d:
            raise DimensionError(f"Got {len(names)} names for {d} quantities.")
        if rhat_tol <= 1.0:
            raise SpecificationError(f"rhat_tol must exceed 1; got {rhat_tol}.")
        if min_ess <= 0.0:
            raise SpecificationError(f"min_ess must be positive; got {min_ess}.")
        if n_draws < _MIN_CHAIN_DRAWS:
            raise SpecificationError(
                f"Each chain needs at least {_MIN_CHAIN_DRAWS} kept draws; got {n_draws}."
            )
        columns = {
            key: np.full(d, np.nan) for key in ("mcse", "ess_bulk", "ess_tail", "rhat", "geweke")
        }
        flat = draws.reshape(n_chains * n_draws, d)
        mean = flat.mean(axis=0)
        sd = flat.std(axis=0, ddof=1) if flat.shape[0] > 1 else np.zeros(d)
        for j in range(d):
            chains = draws[:, :, j]
            columns["rhat"][j] = _rhat(chains)
            columns["ess_bulk"][j] = _ess_bulk(chains)
            columns["ess_tail"][j] = _ess_tail(chains)
            columns["mcse"][j] = _mcse_mean(chains)
            try:
                scores = _geweke(chains)
            except SpecificationError:
                continue
            if np.all(np.isnan(scores)):
                continue
            columns["geweke"][j] = scores[np.nanargmax(np.abs(scores))]
        return cls(
            names=tuple(names),
            mean=mean,
            sd=sd,
            mcse=columns["mcse"],
            ess_bulk=columns["ess_bulk"],
            ess_tail=columns["ess_tail"],
            rhat=columns["rhat"],
            geweke=columns["geweke"],
            n_chains=n_chains,
            n_draws=n_draws,
            rhat_tol=rhat_tol,
            min_ess=min_ess,
            source=source,
        )

    @property
    def ess_floor(self) -> float:
        """Total effective draws below which a parameter is flagged."""
        return self.min_ess * self.n_chains

    @property
    def degenerate(self) -> tuple[str, ...]:
        """Quantities whose chain never moved, excluded from the verdict."""
        return tuple(name for name, r in zip(self.names, self.rhat, strict=True) if np.isnan(r))

    @property
    def _failing(self) -> npt.NDArray[np.bool_]:
        assessed = ~np.isnan(self.rhat)
        bad = (
            (self.rhat > self.rhat_tol)
            | (self.ess_bulk < self.ess_floor)
            | (self.ess_tail < self.ess_floor)
        )
        return np.asarray(assessed & bad)

    @property
    def flagged(self) -> tuple[str, ...]:
        """Quantities failing the R-hat or either effective-sample-size test."""
        return tuple(np.asarray(self.names)[self._failing].tolist())

    @property
    def converged(self) -> bool:
        """Whether no assessed quantity is flagged.

        ``True`` with a non-empty :attr:`degenerate` list is a verdict on the
        quantities that moved, not on the run.
        """
        return not bool(self._failing.any())

    def worst(self, n: int = 5) -> tuple[str, ...]:
        """The ``n`` quantities with the largest R-hat, degenerate ones last."""
        order = np.argsort(np.nan_to_num(self.rhat, nan=-np.inf))[::-1]
        return tuple(np.asarray(self.names)[order[:n]].tolist())

    def summary(self, *, top: int | None = 20) -> SummaryTable:
        """Render as a table sorted by R-hat, descending.

        Args:
            top: Rows to show; ``None`` shows every quantity. The metadata
                always reports the count that was assessed.
        """
        order = np.argsort(np.nan_to_num(self.rhat, nan=-np.inf))[::-1]
        shown = order if top is None else order[:top]
        verdict = "converged" if self.converged else f"{len(self.flagged)} flagged"
        metadata = (
            ("Chains", str(self.n_chains)),
            ("Draws per chain", str(self.n_draws)),
            ("Quantities", str(len(self.names))),
            ("Verdict", verdict),
            ("R-hat threshold", f"{self.rhat_tol:g}"),
            ("ESS floor", f"{self.ess_floor:g}"),
        )
        rows = tuple(
            (
                self.names[j],
                f"{self.mean[j]:.4g}",
                f"{self.sd[j]:.4g}",
                f"{self.mcse[j]:.3g}",
                f"{self.ess_bulk[j]:.0f}",
                f"{self.ess_tail[j]:.0f}",
                f"{self.rhat[j]:.4f}",
                f"{self.geweke[j]:.2f}",
                "*" if self._failing[j] else "",
            )
            for j in shown
        )
        notes: list[str] = []
        if self.n_chains == 1:
            notes.append(
                "Single chain: R-hat compares the two halves of the run and detects drift, not "
                "a chain confined to one of several modes. Pass a second run with a different "
                "seed for the between-chain reading."
            )
            if self.n_draws < 1000:
                notes.append(
                    f"With {self.n_draws} draws the two half-chains are short and R-hat is "
                    "noisy: an independent sequence of 500 draws exceeds 1.01 in about one "
                    "run in fifty. Flags near the threshold call for a longer run."
                )
        if self.flagged:
            notes.append(
                f"* R-hat > {self.rhat_tol:g} or fewer than {self.ess_floor:g} effective draws "
                f"in the bulk or a tail: {', '.join(self.flagged[:8])}"
                + (" ..." if len(self.flagged) > 8 else "")
                + "."
            )
        if self.degenerate:
            notes.append(
                f"Never moved (fixed by construction or a stuck sampler; not assessed): "
                f"{', '.join(self.degenerate[:8])}"
                + (" ..." if len(self.degenerate) > 8 else "")
                + "."
            )
        if top is not None and len(self.names) > top:
            notes.append(
                f"Showing the {top} largest R-hat of {len(self.names)}; "
                "summary(top=None) shows all."
            )
        notes.append(
            "Rank-normalized split-R-hat with folding, bulk and tail ESS by Geyer's initial "
            "monotone sequence (Vehtari et al., 2021); Geweke z on the first 10% versus last 50%."
        )
        return SummaryTable(
            title=f"Convergence: {self.source}",
            metadata=metadata,
            columns=(
                "quantity",
                "mean",
                "sd",
                "mcse",
                "ess_bulk",
                "ess_tail",
                "rhat",
                "geweke",
                "",
            ),
            rows=rows,
            notes=tuple(notes),
        )

    def __repr__(self) -> str:
        assessed = self.rhat[~np.isnan(self.rhat)]
        worst = f"{assessed.max():.4f}" if assessed.size else "nan"
        bulk = self.ess_bulk[~np.isnan(self.ess_bulk)]
        least = f"{bulk.min():.0f}" if bulk.size else "nan"
        verdict = "converged" if self.converged else f"{len(self.flagged)} flagged"
        return (
            f"ConvergenceReport({len(self.names)} quantities, {self.n_chains} x {self.n_draws} "
            f"draws, max rhat={worst}, min ess_bulk={least}, {verdict})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _PredictiveCheckTest:
    """A prior or posterior predictive check: the data against replicated data.

    Each replication is a data set simulated from one draw of the
    parameters -- from the posterior, to ask whether the fitted model
    reproduces the features of the sample it was fitted to; from the
    prior, to ask what the prior says on the scale of the data before any
    sample touches it. A discrepancy statistic ``T`` is evaluated on the
    data and on every replication, and the tail probability ``P(T(y_rep)
    >= T(y))`` locates the data in the replicated distribution: near zero
    or one, the model does not produce data like these in that respect.

    The posterior version is not a frequentist test. Under a correctly
    specified model its p-value is concentrated near one half rather than
    uniform (Meng, 1994), because the same data fit the parameters and
    judge the fit, so a p-value in a tail understates the evidence of
    misspecification rather than overstating it. The prior version is
    not a test at all -- the data are one draw the prior may or may not
    cover -- and its table is read for the range of the replicated
    statistics as much as for the tail probability.

    Attributes:
        kind: ``"posterior"`` or ``"prior"``.
        statistics: Names of the discrepancy statistics, in row order.
        names: Variable labels, in column order.
        observed: ``(m, k)`` statistics of the data.
        replicated: ``(R, m, k)`` statistics of the replications.
        n_replications: Replicated data sets ``R``.
        nobs: Rows in the data set replicated -- the effective sample.
        source: What was checked, for the summary title.
        notes: Remarks from the replicating result or model.
    """

    kind: str
    statistics: tuple[str, ...]
    names: tuple[str, ...]
    observed: npt.NDArray[np.float64] = field(repr=False)
    replicated: npt.NDArray[np.float64] = field(repr=False)
    n_replications: int
    nobs: int
    source: str
    notes: tuple[str, ...] = ()

    @property
    def pvalues(self) -> npt.NDArray[np.float64]:
        """``(m, k)`` tail probabilities ``P(T(y_rep) >= T(y))``, ties counted as one half."""
        return _predictive_pvalues(self.observed, self.replicated)

    def pvalue(self, statistic: str, name: str | None = None) -> npt.NDArray[np.float64] | float:
        """One statistic's tail probabilities, for every variable or for one.

        Args:
            statistic: A name from :attr:`statistics`.
            name: A variable label; ``None`` returns the ``(k,)`` row.

        Raises:
            SpecificationError: If the statistic or the variable is unknown.
        """
        if statistic not in self.statistics:
            raise SpecificationError(
                f"unknown statistic {statistic!r}; this check computed {self.statistics}."
            )
        row = self.pvalues[self.statistics.index(statistic)]
        if name is None:
            return row
        if name not in self.names:
            raise SpecificationError(f"unknown variable {name!r}; expected one of {self.names}.")
        return float(row[self.names.index(name)])

    def replicated_quantiles(
        self, levels: tuple[float, ...] = (0.05, 0.5, 0.95)
    ) -> npt.NDArray[np.float64]:
        """Quantiles of each replicated statistic, ``(len(levels), m, k)``."""
        return np.asarray(np.quantile(self.replicated, levels, axis=0), dtype=np.float64)

    def _flags(self, level: float) -> npt.NDArray[np.bool_]:
        """Which ``(m, k)`` cells sit in a tail of the replicated distribution."""
        p = self.pvalues
        with np.errstate(invalid="ignore"):
            return np.asarray((p < level) | (p > 1.0 - level))

    def extreme(self, *, level: float = _EXTREME_PVALUE) -> tuple[str, ...]:
        """Labels ``statistic[variable]`` with a tail probability outside ``(level, 1 - level)``.

        Args:
            level: The tail size on each side.

        Raises:
            SpecificationError: If the level is not inside ``(0, 0.5)``.
        """
        if not 0.0 < level < 0.5:
            raise SpecificationError(f"level must lie strictly inside (0, 0.5); got {level}.")
        flags = self._flags(level)
        return tuple(
            f"{self.statistics[i]}[{self.names[j]}]"
            for i in range(len(self.statistics))
            for j in range(len(self.names))
            if flags[i, j]
        )

    def adequate(self, *, level: float = _EXTREME_PVALUE) -> bool:
        """Whether no statistic sits in a tail of its replicated distribution."""
        return not self.extreme(level=level)

    def summary(self, *, level: float = _EXTREME_PVALUE) -> SummaryTable:
        """Render as a table: one row per statistic and variable.

        Args:
            level: The tail size on each side that marks a row.

        Raises:
            SpecificationError: If the level is not inside ``(0, 0.5)``.
        """
        if not 0.0 < level < 0.5:
            raise SpecificationError(f"level must lie strictly inside (0, 0.5); got {level}.")
        p = self.pvalues
        bands = self.replicated_quantiles((0.05, 0.5, 0.95))
        flags = self._flags(level)
        rows = tuple(
            (
                self.statistics[i],
                self.names[j],
                f"{self.observed[i, j]:.4g}",
                f"{bands[0, i, j]:.4g}",
                f"{bands[1, i, j]:.4g}",
                f"{bands[2, i, j]:.4g}",
                "nan" if np.isnan(p[i, j]) else f"{p[i, j]:.3f}",
                "*" if flags[i, j] else "",
            )
            for i in range(len(self.statistics))
            for j in range(len(self.names))
        )
        extreme = self.extreme(level=level)
        verdict = "no statistic in a tail" if not extreme else f"{len(extreme)} in a tail"
        metadata = (
            ("Kind", f"{self.kind} predictive"),
            ("Replications", str(self.n_replications)),
            ("Observations", str(self.nobs)),
            ("Statistics", str(len(self.statistics))),
            ("Variables", str(len(self.names))),
            ("Verdict", verdict),
        )
        notes: list[str] = list(self.notes)
        if extreme:
            notes.append(
                f"* p-value below {level:g} or above {1.0 - level:g}: {', '.join(extreme[:8])}"
                + (" ..." if len(extreme) > 8 else "")
                + "."
            )
        if self.kind == "posterior":
            notes.append(
                "Posterior predictive p-values concentrate near 0.5 under a correctly "
                "specified model (Meng, 1994); a tail reading understates misspecification."
            )
        else:
            notes.append(
                "A prior predictive check locates the data in what the prior generates; the "
                "replicated quantiles say what the prior deems plausible on the data's scale."
            )
        if self.n_replications < 200:
            notes.append(
                f"{self.n_replications} replications resolve a tail probability to about "
                f"{1.0 / self.n_replications:.3f}; more replications sharpen the p-values."
            )
        return SummaryTable(
            title=f"{self.kind.capitalize()} predictive check: {self.source}",
            metadata=metadata,
            columns=("statistic", "variable", "observed", "q05", "median", "q95", "p", ""),
            rows=rows,
            notes=tuple(notes),
        )

    def __repr__(self) -> str:
        extreme = self.extreme()
        verdict = "no statistic in a tail" if not extreme else f"{len(extreme)} in a tail"
        return (
            f"PredictiveCheckTest({self.kind}, {self.n_replications} replications, "
            f"{len(self.statistics)} statistics x {len(self.names)} variables, {verdict})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _ClarkWestTest(_ForecastComparisonTest):
    """Verdict of the Clark-West (2007) test that a nesting model forecasts better.

    The one-sided alternative is that the larger model improves on the
    smaller; under the null the smaller model is true and the larger one's
    extra parameters only add estimation noise, which the statistic
    removes before testing. A rejection says the added structure has
    predictive content; a non-rejection says it has not shown any on
    this window.

    Attributes:
        statistic: The adjusted-MSPE t-type statistic.
        pvalue: Its upper-tail standard normal p-value.
        adjusted_differential: Mean of the adjusted loss differential,
            ``e_r**2 - e_u**2 + (f_r - f_u)**2``; positive favors the
            larger model.
        mspe_restricted: Mean squared prediction error of the smaller model.
        mspe_unrestricted: Mean squared prediction error of the larger model.
        horizon: Forecast horizon behind the series.
        nobs: Evaluation origins.
    """

    adjusted_differential: float
    mspe_restricted: float
    mspe_unrestricted: float

    def _summary_table(self) -> SummaryTable:
        """Render as a table."""
        verdict = "larger model improves" if self.reject() else "no improvement shown"
        rows = (("Clark-West adjusted MSPE", f"{self.statistic:.4f}", f"{self.pvalue:.4f}"),)
        notes = (
            f"Adjusted loss differential {self.adjusted_differential:+.5f} (positive favors "
            "the larger model); raw MSPE "
            f"{self.mspe_restricted:.5f} restricted versus {self.mspe_unrestricted:.5f} "
            "unrestricted.",
            "One-sided standard normal reference, Bartlett long-run variance through "
            f"horizon - 1 = {self.horizon - 1} lags. The adjustment removes the estimation "
            "noise the larger model carries under the null, which is what makes Diebold-"
            "Mariano invalid for nested forecasts.",
        )
        return self._frame(
            title="Clark-West Nested Forecast Comparison",
            verdict=verdict,
            columns=("test", "statistic", "p-value"),
            rows=rows,
            notes=notes,
        )

    def __repr__(self) -> str:
        return (
            f"ClarkWestTest(statistic={self.statistic:.4f}, pvalue={self.pvalue:.4g}, "
            f"horizon={self.horizon}, nobs={self.nobs})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _MincerZarnowitzTest(_ForecastComparisonTest):
    """Verdict of the Mincer-Zarnowitz efficiency regression ``y = a + b f``.

    An unbiased, efficient point forecast has intercept zero and slope
    one. A slope below one says the forecast overreacts -- shrinking it
    toward its mean would help -- and above one that it underreacts; an
    intercept away from zero is bias. The joint restriction is tested
    with a HAC Wald statistic, since multi-step errors overlap.

    Attributes:
        intercept: Estimated ``a``.
        slope: Estimated ``b``.
        statistic: Wald statistic for ``(a, b) = (0, 1)``, chi-squared scale.
        pvalue: Upper-tail p-value of ``statistic / 2`` under ``F(2, T - 2)``.
        r_squared: Fit of the regression, the forecast's explanatory
            share of the outcome.
        horizon: Forecast horizon behind the series.
        nobs: Evaluation origins.
    """

    intercept: float
    slope: float
    r_squared: float

    def _summary_table(self) -> SummaryTable:
        """Render as a table."""
        verdict = "efficiency rejected" if self.reject() else "efficiency not rejected"
        rows = (
            ("intercept (0 under H0)", f"{self.intercept:.4f}", ""),
            ("slope (1 under H0)", f"{self.slope:.4f}", ""),
            ("Wald, F(2, T - 2) reference", f"{self.statistic:.4f}", f"{self.pvalue:.4f}"),
        )
        reading = (
            "a slope below one says the forecast overreacts and would gain from shrinkage "
            "toward its mean; above one, that it underreacts."
        )
        notes = (
            f"R-squared {self.r_squared:.3f}. Under the null the forecast is unbiased and "
            "efficient with respect to itself; " + reading,
            "HAC covariance with Bartlett weights through "
            f"horizon - 1 = {self.horizon - 1} lags and T / (T - 2) scaling, W / 2 referred "
            "to F(2, T - 2) (Mincer & Zarnowitz, 1969)."
            + (
                " At multi-step horizons the truncated kernel understates the long-run "
                "variance and the test over-rejects -- roughly 13% at nominal 5% for a "
                "four-step horizon on 100 origins -- so read a marginal rejection with care."
                if self.horizon > 1
                else ""
            ),
        )
        return self._frame(
            title="Mincer-Zarnowitz Efficiency Regression",
            verdict=verdict,
            columns=("quantity", "estimate", "p-value"),
            rows=rows,
            notes=notes,
        )

    def __repr__(self) -> str:
        return (
            f"MincerZarnowitzTest(intercept={self.intercept:.4f}, slope={self.slope:.4f}, "
            f"statistic={self.statistic:.4f}, pvalue={self.pvalue:.4g}, nobs={self.nobs})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _EncompassingTest(_ForecastComparisonTest):
    """Verdict of the forecast encompassing test: does A already contain what B knows?

    Forecast A encompasses B when the optimal linear combination of the
    two puts zero weight on B. The one-sided alternative is that B carries
    information A lacks; a rejection says combining would help, and the
    reported weight is the least-squares share B would get.

    Attributes:
        statistic: The HLN-corrected statistic on ``e_A (e_A - e_B)``.
        pvalue: Its upper-tail ``t`` p-value.
        weight: Least-squares combination weight on B; zero under the null.
        horizon: Forecast horizon behind the series.
        nobs: Evaluation origins.
    """

    weight: float

    def _summary_table(self) -> SummaryTable:
        """Render as a table."""
        verdict = "B adds information" if self.reject() else "A encompasses B"
        rows = (("HLN encompassing", f"{self.statistic:.4f}", f"{self.pvalue:.4f}"),)
        notes = (
            f"Least-squares combination weight on B: {self.weight:+.3f} (zero under the null "
            "that A encompasses B; one would say B encompasses A).",
            "Diebold-Mariano machinery on e_A (e_A - e_B) with the Harvey-Leybourne-Newbold "
            f"correction, Bartlett long-run variance through horizon - 1 = {self.horizon - 1} "
            "lags, one-sided t reference (Harvey, Leybourne & Newbold, 1998).",
        )
        return self._frame(
            title="Forecast Encompassing",
            verdict=verdict,
            columns=("test", "statistic", "p-value"),
            rows=rows,
            notes=notes,
        )

    def __repr__(self) -> str:
        return (
            f"EncompassingTest(statistic={self.statistic:.4f}, pvalue={self.pvalue:.4g}, "
            f"weight={self.weight:.3f}, nobs={self.nobs})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _UnitRootTest(_TabulatedTest):
    """Verdict of a unit-root or stationarity test.

    The family is not one hypothesis but two. The Dickey-Fuller line --
    ADF, Phillips-Perron, DF-GLS, Ng-Perron, Zivot-Andrews -- puts the
    unit root under the null and rejects in the lower tail; KPSS puts
    stationarity under the null and rejects in the upper tail. The record
    carries which, so that ``reject`` reads correctly for both and a
    table of several tests can be read side by side: a series the ADF
    cannot reject a unit root for and the KPSS cannot reject stationarity
    for is one the sample does not decide, and that is a finding.

    P-values are exact where the literature supplies a response surface
    (MacKinnon for the Dickey-Fuller law) or a table dense enough to
    interpolate (KPSS); the tests known only through asymptotic critical
    values at three levels report ``pvalue=None`` and ``reject`` reads
    the table.

    Attributes:
        trend: Deterministic specification the test was run under.
        lags: Augmentation lags or kernel bandwidth, as the test uses.
        method: How ``lags`` was chosen or the kernel used.
        break_index: For a break-allowing test, the first observation of
            the new regime; ``None`` otherwise.
    """

    trend: str
    lags: int
    method: str
    break_index: int | None = None

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        out = [
            ("Null", self.null),
            ("Trend", self.trend),
            ("Lags", str(self.lags)),
            ("Method", self.method),
            ("Observations", str(self.nobs)),
        ]
        if self.break_index is not None:
            out.append(("Break at", str(self.break_index)))
        return tuple(out)

    def _repr_fields(self) -> tuple[str, ...]:
        return (f"null={self.null!r}", f"trend={self.trend!r}", f"lags={self.lags}")


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _BreakTest(_TabulatedTest):
    """Verdict of a test for one structural break at an unknown date.

    Andrews' (1993) sup-Wald and Andrews and Ploberger's (1994) exp- and
    ave-Wald share one limit process, a ``q``-dimensional Brownian bridge
    over the trimmed window, and the p-values here come from that limit
    simulated at the trimming actually used; the CUSUM tests of Brown,
    Durbin and Evans (1975) compare a recursive-residual path with a
    boundary, and the record carries the path so the crossing can be
    seen. Rejection always lies in the upper tail.

    Attributes:
        break_index: First observation of the new regime at the sup, or
            the first boundary crossing; ``None`` when nothing is located.
        n_restrictions: Coefficients allowed to change, ``q``.
        trimming: Fraction of the sample excluded at each end, or
            ``None`` for a boundary test.
        path: The statistic path over candidate dates, or the CUSUM path.
        bounds: The boundary at the tabulated levels, ``(3, n)``, for a
            CUSUM test; ``None`` otherwise.
    """

    break_index: int | None
    n_restrictions: int
    trimming: float | None
    path: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    bounds: npt.NDArray[np.float64] | None = field(default=None, repr=False)

    def _verdict(self) -> str:
        try:
            rejected = self.reject()
        except SpecificationError:
            return "see critical values"
        return "break" if rejected else "no break"

    def _title(self) -> str:
        return f"{self.name} Structural Break Test"

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        out = [
            ("Null", self.null),
            ("Restrictions", str(self.n_restrictions)),
            ("Observations", str(self.nobs)),
        ]
        if self.trimming is not None:
            out.append(("Trimming", f"{self.trimming:.2f}"))
        if self.break_index is not None:
            out.append(("Break at", str(self.break_index)))
        return tuple(out)

    def _notes(self) -> tuple[str, ...]:
        return ()

    def _repr_fields(self) -> tuple[str, ...]:
        return (f"break_index={self.break_index}",)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class _MultipleBreakTest:
    """Bai and Perron's (1998, 2003) multiple-break analysis of a linear regression.

    For every number of breaks up to ``max_breaks`` the record carries
    the global least-squares partition; the selected number comes from
    the information criterion asked for, or from the sequential
    ``sup F(l + 1 | l)`` procedure whose p-values follow from the
    single-break limit raised to the power ``l + 1``.

    Attributes:
        break_indices: First observation of each new regime under the
            selected number of breaks.
        n_breaks: The selected number.
        criterion: ``"bic"``, ``"lwz"`` or ``"sequential"``.
        ssr: Sum of squared residuals for ``0 .. max_breaks`` breaks.
        bic: Bayesian information criterion for each count.
        lwz: Liu-Wu-Zidek criterion for each count.
        partitions: The global partition for each count.
        sequential: The ``sup F(l + 1 | l)`` tests, one per step taken.
        n_restrictions: Regression coefficients, ``q``.
        trimming: Minimum segment length as a fraction of the sample.
        nobs: Observations.
    """

    break_indices: tuple[int, ...]
    n_breaks: int
    criterion: str
    ssr: npt.NDArray[np.float64] = field(repr=False)
    bic: npt.NDArray[np.float64] = field(repr=False)
    lwz: npt.NDArray[np.float64] = field(repr=False)
    partitions: tuple[tuple[int, ...], ...] = field(repr=False)
    sequential: tuple[_BreakTest, ...] = field(repr=False)
    n_restrictions: int
    trimming: float
    nobs: int

    @property
    def max_breaks(self) -> int:
        """Largest number of breaks considered."""
        return int(self.ssr.shape[0] - 1)

    def summary(self) -> SummaryTable:
        """Render as a table over the number of breaks."""
        rows = []
        for m in range(self.max_breaks + 1):
            dates = ", ".join(str(i) for i in self.partitions[m]) or "-"
            rows.append(
                (str(m), f"{self.ssr[m]:.4f}", f"{self.bic[m]:.4f}", f"{self.lwz[m]:.4f}", dates)
            )
        notes = tuple(
            f"sup F({step + 1} | {step}) = {t.statistic:.3f}"
            + ("" if t.pvalue is None else f", p = {t.pvalue:.4f}")
            for step, t in enumerate(self.sequential)
        )
        return SummaryTable(
            title="Bai-Perron Multiple Breaks",
            metadata=(
                ("Selected breaks", str(self.n_breaks)),
                ("Criterion", self.criterion),
                ("Break dates", ", ".join(str(i) for i in self.break_indices) or "none"),
                ("Trimming", f"{self.trimming:.2f}"),
                ("Observations", str(self.nobs)),
            ),
            columns=("breaks", "SSR", "BIC", "LWZ", "dates"),
            rows=tuple(rows),
            notes=notes,
        )

    def __repr__(self) -> str:
        return (
            f"MultipleBreakTest(n_breaks={self.n_breaks}, break_indices={self.break_indices}, "
            f"criterion={self.criterion!r}, nobs={self.nobs})"
        )
