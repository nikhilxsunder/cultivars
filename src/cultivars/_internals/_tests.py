from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt

from .._core import (
    _CRITICAL_LEVELS,
    _DEFAULT_ALPHA,
    _MIN_CHAIN_DRAWS,
    _MIN_ESS_PER_CHAIN,
    _RHAT_TOL,
    SummaryTable,
    _ess_bulk,
    _ess_tail,
    _geweke,
    _mcse_mean,
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
