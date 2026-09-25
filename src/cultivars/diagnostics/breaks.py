# filepath: /src/cultivars/diagnostics/breaks.py
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
r"""Structural-break tests: whether one regression held over the whole sample.

A break at a *known* date is a Chow test and needs nothing here. The
tests in this module are for the case that matters in practice, a break
at a date the data must locate, where the break date is a nuisance
parameter present only under the alternative and the usual chi-squared
limit does not apply. :func:`sup_wald` scans the trimmed window
:math:`[\pi_0 T, (1 - \pi_0) T]`, computes at each candidate date the
Wald statistic for equality of the coefficients on either side, and
reports the largest (Andrews 1993) together with the exponential and
average functionals of Andrews and Ploberger (1994), all three of which
converge under the null to functionals of

.. math::

   \frac{BB_q(\pi)^\top BB_q(\pi)}{\pi (1 - \pi)},
   \qquad \pi \in [\pi_0, 1 - \pi_0],

with :math:`BB_q` a :math:`q`-dimensional Brownian bridge. That limit is
simulated here at the trimming actually used, so the p-values are the
asymptotic ones for the window rather than an interpolation of a printed
table. :func:`bai_perron` extends the search to several breaks with the
dynamic-programming global minimizer of Bai and Perron (2003), which
tables the residual sum of every admissible segment once and finds the
best partition for each count from that table; it selects the number of
breaks by BIC, by the Liu-Wu-Zidek criterion, or by the sequential
:math:`\sup F(l + 1 \mid l)` procedure whose p-values follow from the
single-break limit as :math:`1 - G(x)^{l + 1}`, and reports every
partition considered. :func:`cusum` is the recursive-residual diagnostic
of Brown, Durbin and Evans (1975): the standardized one-step forecast
errors of the regression re-estimated as each observation arrives are
cumulated and compared with a boundary, and a path that crosses says the
coefficients drifted, without a date to name; the CUSUM of squares does
the same for the variance, with its finite-sample boundary simulated
exactly.

Two commitments shape the surface. First, the record shows its evidence.
A :class:`BreakTest` carries the Wald path over candidate dates or the
CUSUM path with its boundary, and a :class:`MultipleBreakTest` carries
the partition, residual sum, and both criteria for every count together
with each sequential test taken, so a verdict can be read against the
shape that produced it and a selection against the alternatives it beat.
Second, the null distribution is simulated, not tabulated. Every p-value
in the module comes from draws of the limit process, or of the exact
finite-sample path for the CUSUM of squares, at the trimming and sample
size in hand; the one exception is the CUSUM proper, whose boundary
multipliers are Brown, Durbin and Evans' and whose record therefore
carries critical values without a p-value.

Each function takes a target and, optionally, regressors; with none, the
regression is on the deterministic terms alone, so the test is for a
break in the mean or trend. Every coefficient is allowed to break at
once, and serially correlated or heteroskedastic errors are not
corrected for: include the lags the dynamics call for in ``exog``, or
read the result as a diagnostic rather than a test.

Layout. :class:`BreakTest` is the single-break record, a
:class:`~cultivars._internals._TabulatedTest` that reads its statistic
against simulated or tabulated critical values and carries the path;
:class:`MultipleBreakTest` is the Bai-Perron record with its own summary
table over counts. :func:`sup_wald`, :func:`bai_perron`, and
:func:`cusum` are the producers. The numerics live in ``_core``:
``_bridge_functionals`` simulates the sup, exp, and ave functionals of
the Brownian-bridge limit and ``_simulated_critical_values`` turns the
draws into a p-value and critical values; ``_segment_ssr`` tables the
segment residual sums and ``_bai_perron_partition`` runs the dynamic
program over them; ``_recursive_residuals`` produces the standardized
forecast errors, ``_cusum_squares_quantiles`` simulates the
finite-sample boundary, and ``_CUSUM_BOUNDARY`` holds the tabulated
multipliers; ``_validate_regression`` assembles the design from the
deterministic terms and the regressors.

References:
    Andrews, D. W. K. (1993). Tests for parameter instability and
    structural change with unknown change point. *Econometrica*,
    61(4), 821-856.

    Andrews, D. W. K., & Ploberger, W. (1994). Optimal tests when a
    nuisance parameter is present only under the alternative.
    *Econometrica*, 62(6), 1383-1414.

    Bai, J., & Perron, P. (1998). Estimating and testing linear models
    with multiple structural changes. *Econometrica*, 66(1), 47-78.

    Bai, J., & Perron, P. (2003). Computation and analysis of multiple
    structural change models. *Journal of Applied Econometrics*,
    18(1), 1-22.

    Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for
    testing the constancy of regression relationships over time.
    *Journal of the Royal Statistical Society B*, 37(2), 149-192.

    Perron, P. (2006). Dealing with structural breaks. In *Palgrave
    Handbook of Econometrics, Volume 1: Econometric Theory*
    (pp. 278-352). Palgrave Macmillan.

Example:
    A mean shift at observation 100: the sup-Wald dates it, the
    Bai-Perron selection counts it, and the CUSUM path crosses its band
    after it:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = np.concatenate([rng.standard_normal(100), 2.0 + rng.standard_normal(100)])
    >>> test = sup_wald(y, seed=0)
    >>> test.reject(), test.break_index
    (True, 100)
    >>> bai_perron(y, max_breaks=3, seed=0).break_indices
    (100,)
    >>> cusum(y).reject(alpha=0.05)
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from .._core import (
    _CRITICAL_ALPHAS,
    _CRITICAL_LEVELS,
    _CUSUM_BOUNDARY,
    SummaryTable,
    _bai_perron_partition,
    _bridge_functionals,
    _cusum_squares_quantiles,
    _recursive_residuals,
    _segment_ssr,
    _simulated_critical_values,
    _validate_regression,
)
from .._internals import _TabulatedTest
from ..exceptions import SpecificationError

__all__ = ["BreakTest", "MultipleBreakTest", "bai_perron", "cusum", "sup_wald"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class BreakTest(_TabulatedTest):
    r"""Verdict of a test for one structural break at an unknown date.

    Two families share this record. Andrews' (1993) sup-Wald and Andrews
    and Ploberger's (1994) exp- and ave-Wald compute the Wald statistic
    :math:`W(\pi)` for a split at every fraction :math:`\pi` of the
    trimmed window :math:`[\pi_0, 1 - \pi_0]` and take a functional of
    the path,

    .. math::

       \sup_{\pi} W(\pi), \qquad
       \log \int \exp\bigl(W(\pi) / 2\bigr)\, d\pi, \qquad
       \int W(\pi)\, d\pi,

    whose common limit is the same functional of
    :math:`BB_q(\pi)^\top BB_q(\pi) / (\pi (1 - \pi))` with :math:`BB_q`
    a :math:`q`-dimensional Brownian bridge. The p-values and critical
    values are that limit simulated at the trimming actually used, so
    they are exact for the window rather than read from a table computed
    at another one. The CUSUM tests of Brown, Durbin and Evans (1975)
    instead cumulate the recursive residuals and compare the path with a
    boundary; the record carries the path and the boundary at the three
    tabulated levels so the crossing can be seen, and the CUSUM proper
    carries no p-value, only the boundary multipliers, while the
    CUSUM of squares has its finite-sample boundary simulated and so has
    one. Rejection always lies in the upper tail.

    Attributes:
        break_index: First observation of the new regime at the sup, or
            the first boundary crossing; ``None`` when nothing is located.
        n_restrictions: Coefficients allowed to change, ``q``.
        trimming: Fraction of the sample excluded at each end, or
            ``None`` for a boundary test.
        path: The statistic path over candidate dates, or the CUSUM path.
        bounds: The boundary at the tabulated levels, ``(3, n)``, for a
            CUSUM test; ``None`` otherwise.

    Note:
        For the sup-Wald ``break_index`` is the arg-max of the Wald path
        whether or not the null is rejected: the date is where a break
        would be, and the verdict says whether one is. For a CUSUM test
        it is the first date the path leaves the 5% boundary, and
        ``None`` when the path stays inside. The exp- and ave-Wald
        companions locate nothing and carry no path; they are
        functionals of the whole window.

    See Also:
        * :func:`sup_wald` -- the Andrews family, returning this record
          with the exp- and ave-Wald as ``companions``.
        * :func:`cusum` -- the Brown-Durbin-Evans family, returning this
          record with ``path`` and ``bounds``.
        * :class:`MultipleBreakTest` -- the Bai-Perron record for more
          than one break.
        * :class:`~cultivars.diagnostics.hypothesis.TabulatedTest` -- the
          base that reads a statistic against tabulated critical values.

    References:
        Andrews, D. W. K. (1993). Tests for parameter instability and
        structural change with unknown change point. *Econometrica*,
        61(4), 821-856.

        Andrews, D. W. K., & Ploberger, W. (1994). Optimal tests when a
        nuisance parameter is present only under the alternative.
        *Econometrica*, 62(6), 1383-1414.

        Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for
        testing the constancy of regression relationships over time.
        *Journal of the Royal Statistical Society B*, 37(2), 149-192.

    Example:
        A mean shift at observation 100 is found by the sup-Wald at the
        right date, and the companions agree:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.concatenate([rng.standard_normal(100), 2.0 + rng.standard_normal(100)])
        >>> test = sup_wald(y, seed=0)
        >>> test
        BreakTest(name='sup-Wald', statistic=187.9089, pvalue=0, break_index=100, nobs=200)
        >>> [c.name for c in test.companions], test.n_restrictions, test.trimming
        (['exp-Wald', 'ave-Wald'], 1, 0.15)
        >>> test.path.shape, int(np.isnan(test.path).sum())
        ((200,), 59)

        The CUSUM carries no p-value, so its verdict comes from the
        boundary, and only at a tabulated level:

        >>> test = cusum(y)
        >>> test.pvalue, test.break_index, test.bounds.shape
        (None, 116, (3, 199))
        >>> test.reject(alpha=0.05)
        True
        >>> test.reject(alpha=0.03)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: CUSUM carries no p-value; ...
    """

    break_index: int | None
    """First observation of the new regime, or ``None`` when nothing is located.

    For the sup-Wald, the index at which the Wald path peaks, so the
    regime after the break starts here; reported whether or not the
    null is rejected. For a CUSUM test, the observation at which the
    path first leaves the 5% boundary, offset by the ``q`` observations
    the recursion consumes; ``None`` when it never does. ``None`` on
    the exp- and ave-Wald companions.
    """
    n_restrictions: int
    """Coefficients allowed to change across the break, :math:`q`.

    The dimension of the Brownian bridge in the Andrews limit, and the
    width of the regression whose recursive residuals a CUSUM test
    cumulates; deterministic terms count.
    """
    trimming: float | None
    """Fraction of the sample excluded at each end, or ``None`` for a boundary test.

    The Andrews window is :math:`[\\pi_0 T, (1 - \\pi_0) T]`, and the
    critical values are simulated at this exact :math:`\\pi_0`. A CUSUM
    test has no window and reports ``None``.
    """
    path: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    """The statistic path, ``(T,)`` for the sup-Wald or ``(T - q,)`` for a CUSUM test.

    For the sup-Wald, the Wald statistic at each candidate date, ``nan``
    outside the trimmed window, so ``path[break_index]`` is the
    statistic. For a CUSUM test, the cumulated recursive residuals
    (scaled by their standard deviation) or the cumulated share of
    squared residuals, one entry per recursive residual. ``None`` on
    the companions. Kept out of the repr.
    """
    bounds: npt.NDArray[np.float64] | None = field(default=None, repr=False)
    """The boundary at the 1%, 5%, and 10% levels, ``(3, T - q)``, for a CUSUM test.

    Row order follows the tabulated levels; the path breaks the
    boundary where its absolute value (CUSUM) or its deviation from the
    diagonal (CUSUM of squares) exceeds the corresponding row.
    ``None`` for the Andrews family. Kept out of the repr.
    """

    def _verdict(self) -> str:
        """The verdict at the default level, in this test's vocabulary.

        ``"break"`` or ``"no break"`` rather than the base's ``"reject"``
        and ``"keep"`` phrasing, and ``"see critical values"`` when the
        record carries no p-value and the default level is not tabulated,
        which for the tabulated levels here never happens.

        Returns:
            The verdict string for the summary header.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> sup_wald(rng.standard_normal(200), seed=0)._verdict()
            'no break'
        """
        try:
            rejected = self.reject()
        except SpecificationError:
            return "see critical values"
        return "break" if rejected else "no break"

    def _title(self) -> str:
        """The summary title, naming the statistic.

        Returns:
            ``"<name> Structural Break Test"``.

        Example:
            >>> import numpy as np
            >>> cusum(np.random.default_rng(0).standard_normal(100))._title()
            'CUSUM Structural Break Test'
        """
        return f"{self.name} Structural Break Test"

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        """Header lines: null, restrictions, observations, and what applies.

        ``Trimming`` is added when the record has a window and
        ``Break at`` when a date was located, so the CUSUM header omits
        the first and a quiet CUSUM omits both.

        Returns:
            Label-value pairs in display order.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.concatenate([rng.standard_normal(100), 2.0 + rng.standard_normal(100)])
            >>> sup_wald(y, seed=0)._metadata()[3:]
            (('Trimming', '0.15'), ('Break at', '100'))
        """
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
        """No closing lines.

        The base class notes which tail rejection lies in; every break
        test rejects in the upper tail, so the line would say the same
        thing on every record and is dropped.

        Returns:
            The empty tuple.

        Example:
            >>> import numpy as np
            >>> cusum(np.random.default_rng(0).standard_normal(100))._notes()
            ()
        """
        return ()

    def _repr_fields(self) -> tuple[str, ...]:
        """The one extra piece of the one-line repr.

        Returns:
            ``("break_index=<value>",)``.

        Example:
            >>> import numpy as np
            >>> cusum(np.random.default_rng(0).standard_normal(100))._repr_fields()
            ('break_index=None',)
        """
        return (f"break_index={self.break_index}",)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MultipleBreakTest:
    r"""Bai and Perron's (1998, 2003) multiple-break analysis of a linear regression.

    For each number of breaks :math:`m = 0, \ldots, m_{\max}` the record
    carries the global least-squares partition, the :math:`m` dates
    :math:`(T_1, \ldots, T_m)` that minimize the total sum of squared
    residuals over segments of at least ``trimming`` times the sample,
    found by dynamic programming over a table of segment sums. Three
    ways of choosing :math:`m` are computed on every call and the one
    asked for is reported. Two are information criteria over the
    partition sequence, with :math:`p_m = (m + 1) q + m` parameters,

    .. math::

       \mathrm{BIC}(m) = \ln\frac{S_m}{T} + \frac{p_m \ln T}{T},
       \qquad
       \mathrm{LWZ}(m) = \ln\frac{S_m}{T - p_m}
       + \frac{0.299\, p_m (\ln T)^{2.1}}{T},

    the second Liu, Wu and Zidek's (1997) modification that Bai and
    Perron found less prone to over-fitting. The third is sequential:
    from :math:`l` breaks, the largest single-break sup-F over the
    current segments tests :math:`l` against :math:`l + 1`, its p-value
    following from the single-break limit :math:`G` as
    :math:`1 - G(x)^{l + 1}`, and the procedure stops at the first
    non-rejection. The sequential tests are kept on the record whichever
    criterion selected, so the three can be read against each other.

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

    Note:
        The partitions are not nested: the best two-break partition need
        not contain the best single break, and in the example below it
        does not. ``sequential`` may be shorter than ``max_breaks``, since
        the procedure stops at the first non-rejection or when no segment
        is long enough to split; it is computed and kept under the
        information criteria as well. This record is not a hypothesis
        test and has no ``reject``; the sequential entries are
        :class:`BreakTest` records and do.

    See Also:
        * :func:`bai_perron` -- the producer.
        * :class:`BreakTest` -- the single-break record, and the type of
          each entry in ``sequential``.
        * :func:`sup_wald` -- the single-break test the sequential step
          applies segment by segment.

    References:
        Bai, J., & Perron, P. (1998). Estimating and testing linear models
        with multiple structural changes. *Econometrica*, 66(1), 47-78.

        Bai, J., & Perron, P. (2003). Computation and analysis of multiple
        structural change models. *Journal of Applied Econometrics*,
        18(1), 1-22.

        Liu, J., Wu, S., & Zidek, J. V. (1997). On segmented multivariate
        regression. *Statistica Sinica*, 7(2), 497-525.

    Example:
        Two mean shifts, at 100 and 200, in three hundred observations.
        BIC selects two breaks at the right dates; the third sequential
        test does not reject, and the single-break partition puts its
        one date at 200, not inside the two-break partition's:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.concatenate([
        ...     rng.standard_normal(100),
        ...     2.0 + rng.standard_normal(100),
        ...     -1.0 + rng.standard_normal(100),
        ... ])
        >>> test = bai_perron(y, max_breaks=3, seed=0)
        >>> test
        MultipleBreakTest(n_breaks=2, break_indices=(100, 200), criterion='bic', nobs=300)
        >>> test.max_breaks, test.partitions
        (3, ((), (200,), (100, 200), (100, 145, 200)))
        >>> [(t.name, round(t.pvalue, 3)) for t in test.sequential]
        [('sup F(1 | 0)', 0.0), ('sup F(2 | 1)', 0.0), ('sup F(3 | 2)', 0.999)]
        >>> bai_perron(y, max_breaks=3, criterion="sequential", seed=0).n_breaks
        2
    """

    break_indices: tuple[int, ...]
    """First observation of each new regime, under the selected number of breaks.

    ``partitions[n_breaks]``: ascending, each at least the minimum
    segment length from its neighbours and from the sample ends; empty
    when no break is selected.
    """
    n_breaks: int
    """The number of breaks the criterion selected, from ``0`` to ``max_breaks``."""
    criterion: str
    """Which rule selected ``n_breaks``: ``"bic"``, ``"lwz"``, or ``"sequential"``."""
    ssr: npt.NDArray[np.float64] = field(repr=False)
    """Sum of squared residuals of the global partition, ``(max_breaks + 1,)``.

    Entry ``m`` is :math:`S_m`, non-increasing in ``m`` because each
    further break can only fit better. Kept out of the repr.
    """
    bic: npt.NDArray[np.float64] = field(repr=False)
    """Bayesian information criterion per count, ``(max_breaks + 1,)``.

    :math:`\\ln(S_m / T) + p_m \\ln T / T`; the arg-min is the BIC
    selection. Kept out of the repr.
    """
    lwz: npt.NDArray[np.float64] = field(repr=False)
    """Liu-Wu-Zidek criterion per count, ``(max_breaks + 1,)``.

    :math:`\\ln(S_m / (T - p_m)) + 0.299\\, p_m (\\ln T)^{2.1} / T`, a
    heavier penalty than BIC's; the arg-min is the LWZ selection. Kept
    out of the repr.
    """
    partitions: tuple[tuple[int, ...], ...] = field(repr=False)
    """The global least-squares partition for each count, ``max_breaks + 1`` tuples.

    Entry ``m`` holds ``m`` ascending break dates; entry ``0`` is empty.
    Not nested across ``m``. Kept out of the repr.
    """
    sequential: tuple[BreakTest, ...] = field(repr=False)
    """The ``sup F(l + 1 | l)`` tests, in order, one per step the procedure took.

    Each is a :class:`BreakTest` whose ``break_index`` is the date the
    step would add and whose p-value and critical values come from the
    single-break limit raised to the power ``l + 1``. The tuple ends at
    the first non-rejection, or earlier when no segment can be split, so
    its length is at most ``max_breaks``. Kept out of the repr.
    """
    n_restrictions: int
    """Regression coefficients per segment, :math:`q`, deterministic terms included."""
    trimming: float
    """Minimum segment length as a fraction of the sample.

    Each segment holds at least ``floor(trimming * nobs)`` observations,
    and never fewer than ``q + 1``; the sequential limit is simulated at
    this trimming.
    """
    nobs: int
    """Observations the regression was fitted on."""

    @property
    def max_breaks(self) -> int:
        """Largest number of breaks considered.

        Recovered from the length of ``ssr`` rather than stored, so the
        arrays and the count cannot disagree.

        Returns:
            ``len(ssr) - 1``.

        Example:
            >>> import numpy as np
            >>> bai_perron(np.random.default_rng(0).standard_normal(200), max_breaks=2).max_breaks
            2
        """
        return int(self.ssr.shape[0] - 1)

    def summary(self) -> SummaryTable:
        """Render as a table over the number of breaks.

        One row per count with its SSR, both criteria, and the partition's
        dates, under a header naming the selection; the sequential tests
        follow as notes, one line each with statistic and p-value.

        Returns:
            The :class:`~cultivars._core.SummaryTable`, rendered by its
            ``to_text()`` and ``_repr_html_()``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.concatenate([rng.standard_normal(100), 2.0 + rng.standard_normal(100)])
            >>> table = bai_perron(y, max_breaks=2, seed=0).summary()
            >>> table.columns
            ('breaks', 'SSR', 'BIC', 'LWZ', 'dates')
            >>> table.rows[1][0], table.rows[1][-1]
            ('1', '100')
            >>> table.notes[0][:12]
            'sup F(1 | 0)'
        """
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
        """One-line selection: count, dates, criterion, observations.

        Returns:
            The repr string.

        Example:
            >>> import numpy as np
            >>> bai_perron(np.random.default_rng(0).standard_normal(200), max_breaks=2)
            MultipleBreakTest(n_breaks=0, break_indices=(), criterion='bic', nobs=200)
        """
        return (
            f"MultipleBreakTest(n_breaks={self.n_breaks}, break_indices={self.break_indices}, "
            f"criterion={self.criterion!r}, nobs={self.nobs})"
        )


def sup_wald(
    endog: npt.ArrayLike,
    exog: npt.ArrayLike | None = None,
    *,
    trend: str = "c",
    trimming: float = 0.15,
    n_draws: int = 5000,
    grid: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> BreakTest:
    r"""Andrews' sup-Wald test for one break at an unknown date, with exp- and ave-Wald.

    The regression :math:`y_t = x_t^\top \beta + u_t` is split at every
    date :math:`T_b` in the trimmed window
    :math:`[\pi_0 T, (1 - \pi_0) T]`, and the Wald statistic for
    equality of the two coefficient vectors, computed under a common
    error variance, is

    .. math::

       W(T_b) = \frac{S_0 - S_1(T_b)}{S_1(T_b) / (T - 2q)},

    with :math:`S_0` the full-sample and :math:`S_1(T_b)` the split
    residual sum of squares. Three functionals of the path are taken:
    the sup, which is Andrews' (1993) test and locates the break at its
    arg-max; and Andrews and Ploberger's (1994) exp-Wald,
    :math:`\log \operatorname{mean} \exp(W / 2)`, and ave-Wald,
    :math:`\operatorname{mean} W`, which are the optimal tests against
    alternatives distant from and close to the null respectively and
    give up the location. Under the null all three converge to the
    same functionals of :math:`BB_q(\pi)^\top BB_q(\pi) / (\pi(1 - \pi))`
    with :math:`BB_q` a :math:`q`-dimensional Brownian bridge, and the
    p-values and critical values come from ``n_draws`` simulated paths
    of that process on a grid of ``grid`` points at the trimming
    actually used, rather than from a table at another trimming. The
    sup's 5% value at :math:`q = 1`, :math:`\pi_0 = 0.15` reproduces
    Andrews' corrected 8.68 to within Monte Carlo error.

    Args:
        endog: The target, ``(T,)``.
        exog: Regressors ``(T, k)`` or ``(T,)``, or ``None``.
        trend: Deterministic terms, ``"n"``, ``"c"`` or ``"ct"``, placed
            before the regressors in the design; with ``exog=None`` the
            test is for a break in the mean or trend.
        trimming: Fraction of the sample excluded at each end, in
            ``(0, 0.5)``; the window also excludes the first and last
            ``q`` observations so that both segments are estimable.
        n_draws: Limit-process paths for the p-values, at least ``500``.
        grid: Points on which each path is simulated, at least ``100``.
        seed: Seed or generator for the limit simulation.

    Returns:
        The sup-Wald :class:`BreakTest`, with ``break_index`` at the
        arg-max, exp- and ave-Wald as ``companions``, and the Wald path
        over candidate dates in ``path`` (``nan`` outside the window).

    Raises:
        SpecificationError: If the trend is unknown, ``trimming`` is
            outside ``(0, 0.5)``, ``n_draws`` or ``grid`` is too small,
            or the window leaves no candidate date.
        DimensionError: If the regressors do not align with the target.
        NumericalError: If a value is not finite or the design is rank
            deficient.

    Note:
        Every coefficient is allowed to break at once; a partial break
        in a subset of coefficients is not offered. The common error
        variance means the test is for the coefficients, not the
        variance, and serially correlated or heteroskedastic errors are
        not corrected for: include the lags the dynamics call for in
        ``exog``, or read the result as a diagnostic. The three
        statistics share one limit simulation, so their p-values are
        computed from the same draws and are not independent evidence.

    See Also:
        * :class:`BreakTest` -- the record, and the ``reject`` that reads
          the simulated p-value.
        * :func:`bai_perron` -- when more than one break is possible, and
          for a selection between counts.
        * :func:`cusum` -- a boundary test without a limit simulation.
        * :func:`~cultivars.diagnostics.unit_roots.zivot_andrews` -- a
          unit-root test that allows one break under the alternative.

    References:
        Andrews, D. W. K. (1993). Tests for parameter instability and
        structural change with unknown change point. *Econometrica*,
        61(4), 821-856.

        Andrews, D. W. K. (2003). Tests for parameter instability and
        structural change with unknown change point: A corrigendum.
        *Econometrica*, 71(1), 395-397.

        Andrews, D. W. K., & Ploberger, W. (1994). Optimal tests when a
        nuisance parameter is present only under the alternative.
        *Econometrica*, 62(6), 1383-1414.

        Hansen, B. E. (1997). Approximate asymptotic p values for
        structural-change tests. *Journal of Business & Economic
        Statistics*, 15(1), 60-67.

    Example:
        A mean shift at observation 100 is dated exactly, and the 5%
        critical value is Andrews' to Monte Carlo error:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.concatenate([rng.standard_normal(100), 2.0 + rng.standard_normal(100)])
        >>> test = sup_wald(y, seed=0)
        >>> test
        BreakTest(name='sup-Wald', statistic=187.9089, pvalue=0, break_index=100, nobs=200)
        >>> round(test.critical_values["5%"], 1)
        8.6
        >>> [(c.name, c.pvalue) for c in test.companions]
        [('exp-Wald', 0.0), ('ave-Wald', 0.0)]

        A break in a slope, with the intercept and slope both allowed
        to change:

        >>> x = rng.standard_normal(200)
        >>> y = 0.5 * x + rng.standard_normal(200)
        >>> y[100:] += 1.5 * x[100:]
        >>> test = sup_wald(y, exog=x, seed=0)
        >>> test.n_restrictions, test.reject(), 95 <= test.break_index <= 105
        (2, True, True)
    """
    if not 0.0 < trimming < 0.5:
        raise SpecificationError(f"trimming must lie in (0, 0.5); got {trimming}.")
    if n_draws < 500 or grid < 100:
        raise SpecificationError(
            f"n_draws must be at least 500 and grid at least 100; got {n_draws}, {grid}."
        )
    y, design = _validate_regression(endog, exog, trend)
    nobs, q = design.shape
    first = max(int(np.floor(trimming * nobs)), q + 1)
    last = nobs - first
    if last <= first:
        raise SpecificationError("the trimmed window leaves no candidate break date.")
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    ssr_full = float(np.sum((y - design @ coef) ** 2))
    path = np.full(nobs, np.nan)
    for tb in range(first, last + 1):
        before, after = design[:tb], design[tb:]
        b0, _, _, _ = np.linalg.lstsq(before, y[:tb], rcond=None)
        b1, _, _, _ = np.linalg.lstsq(after, y[tb:], rcond=None)
        ssr_split = float(np.sum((y[:tb] - before @ b0) ** 2) + np.sum((y[tb:] - after @ b1) ** 2))
        path[tb] = (ssr_full - ssr_split) / (ssr_split / (nobs - 2 * q))
    window = path[first : last + 1]
    sup = float(window.max())
    break_index = first + int(np.argmax(window))
    exp_w = float(np.log(np.mean(np.exp(0.5 * window))))
    ave = float(window.mean())
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    sup_draws, exp_draws, ave_draws = _bridge_functionals(
        q, trimming, n_draws=n_draws, grid=grid, rng=rng
    )
    null = "no break in the coefficients"

    def record(name: str, statistic: float, draws: npt.NDArray[np.float64]) -> BreakTest:
        """Build one record from a statistic and its simulated null draws.

        The p-value is the share of draws at or above the statistic and
        the critical values are the draws' upper quantiles at the
        tabulated levels; only the sup-Wald record carries the located
        date.

        Args:
            name: ``"sup-Wald"``, ``"exp-Wald"`` or ``"ave-Wald"``.
            statistic: The functional's value on the sample.
            draws: ``(n_draws,)`` values of the same functional on the
                simulated limit paths.

        Returns:
            The :class:`BreakTest` without companions or path.
        """
        pvalue, cv = _simulated_critical_values(draws, statistic)
        return BreakTest(
            name=name,
            statistic=statistic,
            pvalue=pvalue,
            critical_values=cv,
            null=null,
            break_index=break_index if name == "sup-Wald" else None,
            n_restrictions=q,
            trimming=trimming,
            nobs=nobs,
        )

    companions = (record("exp-Wald", exp_w, exp_draws), record("ave-Wald", ave, ave_draws))
    pvalue, cv = _simulated_critical_values(sup_draws, sup)
    return BreakTest(
        name="sup-Wald",
        statistic=sup,
        pvalue=pvalue,
        critical_values=cv,
        null=null,
        break_index=break_index,
        n_restrictions=q,
        trimming=trimming,
        nobs=nobs,
        companions=companions,
        path=path,
    )


def bai_perron(
    endog: npt.ArrayLike,
    exog: npt.ArrayLike | None = None,
    *,
    trend: str = "c",
    max_breaks: int = 5,
    trimming: float = 0.15,
    criterion: str = "bic",
    alpha: float = 0.05,
    n_draws: int = 5000,
    grid: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> MultipleBreakTest:
    r"""Bai-Perron estimation of multiple breaks in a linear regression.

    The regression :math:`y_t = x_t^\top \beta_j + u_t` is allowed a
    different :math:`\beta_j` on each of :math:`m + 1` segments, every
    coefficient breaking at once. The sum of squared residuals of every
    admissible segment, one of at least ``trimming`` times the sample and
    never fewer than :math:`q + 1` observations, is tabled once, and the
    global least-squares partition for each :math:`m` up to
    ``max_breaks`` follows by dynamic programming over that table,
    which is what makes the search over :math:`\binom{T}{m}` partitions
    affordable. The number of breaks is then chosen three ways, all
    kept on the record, one reported. BIC and the Liu-Wu-Zidek criterion
    penalize the partition sequence, with :math:`p_m = (m + 1) q + m`
    parameters; LWZ's penalty grows as :math:`(\ln T)^{2.1}` rather than
    :math:`\ln T`, which is why the authors found it less prone to
    over-fitting. The sequential procedure starts from no break and,
    given the :math:`l`-break partition, computes on each segment the
    largest single-break F statistic,

    .. math::

       \sup F(l + 1 \mid l) = \max_{\text{segments}} \max_{\tau}
       \frac{S_{\text{seg}} - S_{\text{split}}(\tau)}
            {S_{\text{split}}(\tau) / (n_{\text{seg}} - 2q)},

    tests :math:`l` against :math:`l + 1` with it, and stops at the first
    non-rejection at ``alpha``. Because the maximum is over
    :math:`l + 1` independent segments, its p-value follows from the
    single-break sup-Wald limit :math:`G` simulated at the trimming as
    :math:`1 - G(x)^{l + 1}`, and its critical values are the quantiles
    of :math:`G` at :math:`(1 - \alpha)^{1 / (l + 1)}`.

    Args:
        endog: The target, ``(T,)``.
        exog: Regressors ``(T, k)`` or ``(T,)``, or ``None``.
        trend: Deterministic terms, ``"n"``, ``"c"`` or ``"ct"``, placed
            before the regressors in the design.
        max_breaks: Largest number of breaks considered; at least one.
        trimming: Minimum segment length as a fraction of the sample, in
            ``(0, 0.5)``; ``0.15`` is Bai and Perron's default.
        criterion: Which selection to report: ``"bic"``, ``"lwz"`` or
            ``"sequential"``.
        alpha: Level of the sequential tests, in ``(0, 1)``.
        n_draws: Limit-process paths for the sequential p-values, at
            least ``500``.
        grid: Points on which each path is simulated, at least ``100``.
        seed: Seed or generator for the limit simulation.

    Returns:
        The :class:`MultipleBreakTest`, with the partition, SSR, BIC and
        LWZ for every count and the sequential tests taken.

    Raises:
        SpecificationError: If the criterion, trend, trimming, level, or
            counts are unusable, or if ``max_breaks + 1`` segments of the
            minimum length do not fit the sample.
        DimensionError: If the regressors do not align with the target.
        NumericalError: If a value is not finite or the design is rank
            deficient.

    Note:
        The sequential F statistic here normalizes by the split
        segment's own residual sum, with ``n_seg - 2q`` degrees of
        freedom, rather than by the full-sample residual sum of the
        :math:`l + 1`-break model as in Bai and Perron's Section 4; the
        two agree asymptotically and the segment form keeps each step a
        self-contained single-break test. Serial correlation and
        heteroskedasticity in :math:`u_t` are not corrected for: include
        the lags the dynamics call for in ``exog``, or read the
        selection as descriptive. The sequential tests are run under
        every criterion, so ``criterion="bic"`` still pays for the limit
        simulation; ``n_draws`` is the lever if that matters.

    See Also:
        * :class:`MultipleBreakTest` -- the record, with the summary table
          over counts.
        * :func:`sup_wald` -- the single-break test with the same limit,
          when one break is the hypothesis.
        * :func:`cusum` -- a boundary test that needs no count of
          breaks.

    References:
        Bai, J., & Perron, P. (1998). Estimating and testing linear models
        with multiple structural changes. *Econometrica*, 66(1), 47-78.

        Bai, J., & Perron, P. (2003). Computation and analysis of multiple
        structural change models. *Journal of Applied Econometrics*,
        18(1), 1-22.

        Liu, J., Wu, S., & Zidek, J. V. (1997). On segmented multivariate
        regression. *Statistica Sinica*, 7(2), 497-525.

    Example:
        One mean shift at observation 100. BIC, LWZ, and the sequential
        procedure agree on one break at the right date:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.concatenate([rng.standard_normal(100), 2.0 + rng.standard_normal(100)])
        >>> bai_perron(y, max_breaks=3, seed=0)
        MultipleBreakTest(n_breaks=1, break_indices=(100,), criterion='bic', nobs=200)
        >>> bai_perron(y, max_breaks=3, criterion="lwz", seed=0).n_breaks
        1
        >>> test = bai_perron(y, max_breaks=2, criterion="sequential", seed=0)
        >>> test.n_breaks, [t.name for t in test.sequential]
        (1, ['sup F(1 | 0)', 'sup F(2 | 1)'])
        >>> test.sequential[0].pvalue, bool(test.sequential[1].pvalue > 0.05)
        (0.0, True)

        Too many breaks for the sample at the requested trimming is
        refused rather than silently capped:

        >>> bai_perron(y, max_breaks=10)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: 10 breaks with segments of at least 30 ...
    """
    if criterion not in ("bic", "lwz", "sequential"):
        raise SpecificationError(
            f"criterion must be 'bic', 'lwz' or 'sequential'; got {criterion!r}."
        )
    if not 0.0 < trimming < 0.5:
        raise SpecificationError(f"trimming must lie in (0, 0.5); got {trimming}.")
    if max_breaks < 1:
        raise SpecificationError(f"max_breaks must be at least 1; got {max_breaks}.")
    if not 0.0 < alpha < 1.0:
        raise SpecificationError(f"alpha must lie in (0, 1); got {alpha}.")
    y, design = _validate_regression(endog, exog, trend)
    nobs, q = design.shape
    min_size = max(int(np.floor(trimming * nobs)), q + 1)
    if (max_breaks + 1) * min_size > nobs:
        raise SpecificationError(
            f"{max_breaks} breaks with segments of at least {min_size} observations do not "
            f"fit {nobs}; lower max_breaks or trimming."
        )
    table = _segment_ssr(y, design, min_size)
    ssr = np.empty(max_breaks + 1)
    partitions: list[tuple[int, ...]] = []
    for m in range(max_breaks + 1):
        value, breaks = _bai_perron_partition(table, m, min_size)
        ssr[m] = value
        partitions.append(breaks)
    counts = np.arange(max_breaks + 1)
    n_params = (counts + 1) * q + counts
    bic = np.log(ssr / nobs) + n_params * np.log(nobs) / nobs
    lwz = np.log(ssr / (nobs - n_params)) + n_params * 0.299 * np.log(nobs) ** 2.1 / nobs

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    sup_draws, _, _ = _bridge_functionals(q, trimming, n_draws=n_draws, grid=grid, rng=rng)
    sequential: list[BreakTest] = []
    selected_sequential = 0
    for step in range(max_breaks):
        segments = (0, *partitions[step], nobs)
        best_stat, best_date = -np.inf, None
        for start, end in pairwise(segments):
            length = end - start
            if length < 2 * min_size:
                continue
            base = table[start, end]
            for tb in range(start + min_size, end - min_size + 1):
                split = table[start, tb] + table[tb, end]
                stat = (base - split) / (split / (length - 2 * q))
                if stat > best_stat:
                    best_stat, best_date = stat, tb
        if best_date is None:
            break
        survival = float(np.mean(sup_draws >= best_stat))
        pvalue = 1.0 - (1.0 - survival) ** (step + 1)
        cv = {
            label: float(np.quantile(sup_draws, (1.0 - level) ** (1.0 / (step + 1))))
            for label, level in zip(_CRITICAL_LEVELS, _CRITICAL_ALPHAS, strict=True)
        }
        test = BreakTest(
            name=f"sup F({step + 1} | {step})",
            statistic=float(best_stat),
            pvalue=pvalue,
            critical_values=cv,
            null=f"{step} breaks against {step + 1}",
            break_index=best_date,
            n_restrictions=q,
            trimming=trimming,
            nobs=nobs,
        )
        sequential.append(test)
        if pvalue >= alpha:
            break
        selected_sequential = step + 1
    if criterion == "bic":
        n_breaks = int(np.argmin(bic))
    elif criterion == "lwz":
        n_breaks = int(np.argmin(lwz))
    else:
        n_breaks = selected_sequential
    return MultipleBreakTest(
        break_indices=partitions[n_breaks],
        n_breaks=n_breaks,
        criterion=criterion,
        ssr=ssr,
        bic=np.asarray(bic, dtype=np.float64),
        lwz=np.asarray(lwz, dtype=np.float64),
        partitions=tuple(partitions),
        sequential=tuple(sequential),
        n_restrictions=q,
        trimming=trimming,
        nobs=nobs,
    )


def cusum(
    endog: npt.ArrayLike,
    exog: npt.ArrayLike | None = None,
    *,
    trend: str = "c",
    squares: bool = False,
    n_draws: int = 5000,
    seed: int | np.random.Generator | None = None,
) -> BreakTest:
    r"""CUSUM and CUSUM-of-squares tests of coefficient constancy.

    The recursive residuals of Brown, Durbin and Evans (1975) are the
    one-step forecast errors of the regression re-estimated as each
    observation arrives, standardized by their forecast variance,

    .. math::

       w_t = \frac{y_t - x_t^\top \hat\beta_{t-1}}
                  {\sqrt{1 + x_t^\top (X_{t-1}^\top X_{t-1})^{-1} x_t}},
       \qquad t = q + 1, \ldots, T,

    so that under constant coefficients and Gaussian errors they are
    i.i.d. :math:`\mathcal{N}(0, \sigma^2)`. The CUSUM cumulates them,
    :math:`W_r = \sum_{t \le r} w_t / \hat\sigma` with :math:`\hat\sigma`
    the residuals' standard deviation about their mean, and compares
    the path with the linear boundary
    :math:`\pm a \sqrt{n}\,(1 + 2r / n)` over the :math:`n = T - q`
    steps; the statistic is the largest ratio :math:`|W_r|` to the
    boundary shape, and it reads directly against the tabulated
    multipliers :math:`a = 1.143, 0.948, 0.850` at 1%, 5%, and 10%. A
    drift in the coefficients makes the forecast errors one-signed and
    the path leave the band. The CUSUM of squares cumulates the share
    :math:`s_r = \sum_{t \le r} w_t^2 / \sum_t w_t^2` against its
    expectation :math:`r / n` and takes the largest deviation; it is the
    sharper test for a change in variance, and since under the null the
    path is a function of i.i.d. Gaussian draws alone its finite-sample
    boundary is simulated exactly from ``n_draws`` such paths, which
    gives it a p-value the CUSUM proper does not have.

    Args:
        endog: The target, ``(T,)``.
        exog: Regressors ``(T, k)`` or ``(T,)``, or ``None``.
        trend: Deterministic terms, ``"n"``, ``"c"`` or ``"ct"``.
        squares: Whether to run the CUSUM-of-squares test instead of the
            CUSUM.
        n_draws: Simulated paths for the CUSUM-of-squares boundary, at
            least ``500``; unused by the CUSUM.
        seed: Seed or generator for that simulation.

    Returns:
        The :class:`BreakTest` with the ``(T - q,)`` path and the
        ``(3, T - q)`` boundary at the tabulated levels; the CUSUM test
        carries no p-value and reads against its critical values, the
        squared one carries both.

    Raises:
        SpecificationError: If the trend is unknown or ``n_draws`` is too
            small.
        DimensionError: If the regressors do not align with the target.
        NumericalError: If a value is not finite, the design is rank
            deficient, or the first ``q`` observations do not identify
            the regression.

    Note:
        ``break_index`` is the observation at which the path first
        leaves the 5% band, not an estimate of the break date: the CUSUM
        crossing lags a break, since the cumulated errors take time to
        drift out, and the CUSUM-of-squares crossing can precede one,
        since the share is normalized by the whole-sample sum and a late
        rise in variance depresses the early path. Use :func:`sup_wald`
        or :func:`bai_perron` to date a break the CUSUM has flagged. The
        CUSUM has low power against a break in a coefficient whose
        regressor averages zero over the sample, which is why the two
        tests are offered together.

    See Also:
        * :class:`BreakTest` -- the record, whose ``path`` and ``bounds``
          draw the familiar CUSUM plot.
        * :func:`sup_wald` -- the Andrews test, which dates the break and
          carries a p-value.
        * :func:`bai_perron` -- when more than one break is possible.

    References:
        Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for
        testing the constancy of regression relationships over time.
        *Journal of the Royal Statistical Society B*, 37(2), 149-192.

        Ploberger, W., & Krämer, W. (1992). The CUSUM test with OLS
        residuals. *Econometrica*, 60(2), 271-285.

    Example:
        A mean shift at 100 is caught by the CUSUM; the crossing comes
        after the break:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.concatenate([rng.standard_normal(100), 2.0 + rng.standard_normal(100)])
        >>> test = cusum(y)
        >>> test
        BreakTest(name='CUSUM', statistic=2.7283, pvalue=None, break_index=116, nobs=200)
        >>> test.critical_values
        {'1%': 1.143, '5%': 0.948, '10%': 0.85}
        >>> test.reject(alpha=0.01)
        True

        A tripling of the error standard deviation at 150 leaves the
        CUSUM inside its band and is caught by the CUSUM of squares:

        >>> y = np.concatenate([rng.standard_normal(150), 3.0 * rng.standard_normal(150)])
        >>> cusum(y).reject(alpha=0.05), cusum(y, squares=True, seed=0).reject()
        (False, True)
    """
    if n_draws < 500:
        raise SpecificationError(f"n_draws must be at least 500; got {n_draws}.")
    y, design = _validate_regression(endog, exog, trend)
    nobs, q = design.shape
    w = _recursive_residuals(y, design)
    count = w.shape[0]
    steps = np.arange(1, count + 1, dtype=np.float64)
    if squares:
        path = np.cumsum(w**2) / float(w @ w)
        deviation = np.abs(path - steps / count)
        statistic = float(deviation.max())
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        quantiles, draws = _cusum_squares_quantiles(
            count, (0.01, 0.05, 0.10), n_draws=n_draws, rng=rng
        )
        cv = {
            label: quantiles[level]
            for label, level in zip(_CRITICAL_LEVELS, _CRITICAL_ALPHAS, strict=True)
        }
        pvalue = float(np.mean(draws >= statistic))
        bounds = np.vstack([steps / count + cv[label] for label in _CRITICAL_LEVELS])
        crossing = np.flatnonzero(deviation > cv["5%"])
        name, null = "CUSUM of squares", "constant coefficients and variance"
    else:
        sigma = float(np.sqrt(np.sum((w - w.mean()) ** 2) / (count - 1)))
        path = np.cumsum(w) / sigma
        shape = np.sqrt(count) * (1.0 + 2.0 * steps / count)
        ratio = np.abs(path) / shape
        statistic = float(ratio.max())
        cv = dict(_CUSUM_BOUNDARY)
        pvalue = None
        bounds = np.vstack([shape * cv[label] for label in _CRITICAL_LEVELS])
        crossing = np.flatnonzero(ratio > cv["5%"])
        name, null = "CUSUM", "constant coefficients"
    break_index = int(q + crossing[0]) if crossing.size else None
    return BreakTest(
        name=name,
        statistic=statistic,
        pvalue=pvalue,
        critical_values=cv,
        null=null,
        break_index=break_index,
        n_restrictions=q,
        trimming=None,
        nobs=nobs,
        path=np.asarray(path, dtype=np.float64),
        bounds=np.asarray(bounds, dtype=np.float64),
    )
