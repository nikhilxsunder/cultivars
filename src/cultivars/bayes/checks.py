# filepath: /src/cultivars/bayes/checks.py
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
r"""Prior and posterior predictive checks: the model as a data-generating machine.

A Bayesian model is a machine for generating data, and the two checks in
this module run the machine and compare what comes out with what went in.
A *posterior* predictive check simulates replicated data sets from the
retained posterior draws and asks whether the fitted model reproduces the
features of the sample it was fitted to (Rubin 1984; Gelman, Meng & Stern
1996). A *prior* predictive check simulates them from the prior instead
and shows what the prior deems plausible on the scale of the data before
any sample is touched, which is the step of a Bayesian workflow at which a
Minnesota tightness or a volatility-of-volatility scale is chosen with the
eyes open (Gabry et al. 2019). Both compare *discrepancy statistics*,
per-variable features of a data set, and report where the data sit in the
replicated distribution of each as the tail probability

.. math::

   p_T = \Pr\bigl(T(y^{\text{rep}}) \ge T(y)\bigr),

with ties counted as one half. The statistics offered are ``mean``,
``sd``, ``min``, ``max``, ``skewness``, ``kurtosis`` (excess), ``acf1``
(lag-one autocorrelation), and ``arch1`` (lag-one autocorrelation of the
squared demeaned series); the default five -- ``mean``, ``sd``, ``acf1``,
``kurtosis``, ``arch1`` -- are the ones whose failure names the
misspecification. A Gaussian innovation cannot match a fat-tailed
``kurtosis``; a constant covariance cannot match the volatility clustering
in ``arch1``; a mis-specified deterministic term shows in ``mean``.

Two commitments shape the surface. First, the posterior check is presented
as a diagnostic and not as a test: under a correctly specified model the
posterior predictive p-value is concentrated near one half rather than
uniform, because the same data fit the parameters and judge the fit (Meng
1994), so the record flags tails, names the cell, and applies no
multiplicity correction across its ``m * k`` cells. Second, the record
keeps statistics and not panels: a check on two hundred replications of a
hundred-variable system is a few kilobytes, whatever the sample length,
and the replicated panels are freed as soon as their statistics are taken.

Layout. :class:`PredictiveCheckTest` is the frozen record, with the
p-values, the flagged cells, the replicated quantile bands, and the
summary table. :func:`replication_check` builds it from an observed panel
and replications of any provenance, and is the whole comparison;
:func:`posterior_predictive_check` and :func:`prior_predictive_check` are
that function preceded by the simulation and the labelling. The
simulation belongs to the models: a result that can replicate its sample
satisfies :class:`~cultivars._core.ReplicatingResult` by exposing
``observed`` and ``posterior_replications()`` -- the conjugate, Gibbs,
Student-t, and stochastic-volatility BVARs through
:class:`~cultivars._internals._ReplicationMixin`, and the
stochastic-volatility posterior on its own -- and a model that can draw
from its prior satisfies :class:`~cultivars._core.ReplicatingModel` by
exposing ``prior_replications()``: the conjugate and Student-t BVARs with
any dummy observations folded into the prior, the Gibbs BVAR under a
static prior without dummies, and the stochastic-volatility model with
the sampler's prior hyperparameters. A result may also attach
``_replication_notes()``, remarks on how it replicates, which the
posterior check carries onto the record. The numerics live in ``_core``:
``_discrepancy_statistics`` evaluates the registry of statistics
column-wise, ``_predictive_pvalues`` applies the tie-corrected tail rule,
and ``_validate_replications`` and ``_validate_statistics`` are the
gatekeepers.

References:
    Rubin, D. B. (1984). Bayesianly justifiable and relevant frequency
    calculations for the applied statistician. *Annals of Statistics*,
    12(4), 1151-1172.

    Meng, X.-L. (1994). Posterior predictive p-values. *Annals of
    Statistics*, 22(3), 1142-1160.

    Gelman, A., Meng, X.-L., & Stern, H. (1996). Posterior predictive
    assessment of model fitness via realized discrepancies. *Statistica
    Sinica*, 6(4), 733-760.

    Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A.
    (2019). Visualization in Bayesian workflow. *Journal of the Royal
    Statistical Society A*, 182(2), 389-402.

    Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B.,
    Yao, Y., Kennedy, L., Gabry, J., Bürkner, P.-C., & Modrák, M. (2020).
    Bayesian workflow. arXiv:2011.01808.

Example:
    Fit a conjugate BVAR to a stable system and check that it reproduces
    the sample's features:

    >>> import numpy as np
    >>> from cultivars.multivariate.large_dim.bayesian import BVAR
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((160, 2))
    >>> for t in range(1, 160):
    ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(2)
    >>> res = BVAR(y, order=1).fit(n_draws=300, seed=0)
    >>> check = posterior_predictive_check(res, seed=0)
    >>> check.kind, check.observed.shape, check.replicated.shape
    ('posterior', (5, 2), (200, 5, 2))
    >>> check.adequate()
    True

    The same model's prior, before fitting, on the two features the
    Minnesota prior is about:

    >>> prior = prior_predictive_check(BVAR(y, order=1), statistics=["sd", "acf1"], seed=0)
    >>> prior.kind, prior.n_replications
    ('prior', 200)
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from .._core import (
    _DISCREPANCY_NAMES,
    _DISCREPANCY_STATISTICS,
    _EXTREME_PVALUE,
    _MIN_REPLICATIONS,
    ReplicatingModel,
    ReplicatingResult,
    SummaryTable,
    _discrepancy_statistics,
    _predictive_pvalues,
    _source_label,
    _validate_replications,
    _validate_statistics,
    _variable_names,
)
from ..exceptions import SpecificationError

__all__ = [
    "PredictiveCheckTest",
    "posterior_predictive_check",
    "prior_predictive_check",
    "replication_check",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class PredictiveCheckTest:
    r"""A prior or posterior predictive check: the data against replicated data.

    The object :func:`posterior_predictive_check`,
    :func:`prior_predictive_check`, and :func:`replication_check` return.
    Each replication :math:`y^{\text{rep},(r)}` is a data set of the same
    shape as the observed panel, simulated from one draw
    :math:`\theta^{(r)}` of the parameters. From the posterior, the
    question is whether the fitted model reproduces the features of the
    sample it was fitted to; from the prior, what the prior deems
    plausible on the scale of the data before any sample touches it. A
    discrepancy statistic :math:`T` is evaluated per variable on the data
    and on every replication, and the tail probability

    .. math::

       p_T = \Pr\bigl(T(y^{\text{rep}}) \ge T(y)\bigr)
           \approx \frac{1}{R} \sum_{r=1}^{R}
             \mathbb{1}\bigl\{T(y^{\text{rep},(r)}) \ge T(y)\bigr\},

    locates the data in the replicated distribution, with ties counted as
    one half so that a statistic the model reproduces exactly reads
    :math:`0.5` rather than :math:`1`. Near zero or one, the model does not
    produce data like these in that respect, and the statistic's name says
    which respect: a Gaussian innovation cannot match a fat-tailed
    ``kurtosis``, a constant covariance cannot match the volatility
    clustering in ``arch1``, a misplaced constant shows in ``mean``.

    The posterior version is not a frequentist test. Under a correctly
    specified model :math:`p_T` is concentrated near one half rather than
    uniform (Meng, 1994), because the same data fit the parameters and
    judge the fit, so a value in a tail understates the evidence of
    misspecification rather than overstating it. The prior version is not
    a test at all, since the data are one draw the prior may or may not
    cover, and its table is read for the range of the replicated
    statistics as much as for the tail probability (Gabry et al., 2019):
    a prior whose 90% band for ``sd`` runs from a thousandth to a thousand
    times the data's scale is uninformative in a way no p-value shows.

    The record is frozen and holds only the statistics, never the
    replicated panels themselves, so it is small whatever :math:`R` and
    :math:`T` were; every derived quantity is recomputed from
    :attr:`observed` and :attr:`replicated` on access.

    Shapes:
        R: Replicated data sets, :attr:`n_replications`.
        T: Rows of the data set replicated, :attr:`nobs`.
        k: Variables, ``len(names)``.
        m: Discrepancy statistics, ``len(statistics)``.

    Attributes:
        kind: ``"posterior"`` or ``"prior"``, which sets how the summary
            tells the reader to interpret the tail probabilities.
        statistics: Names of the discrepancy statistics, in row order.
        names: Variable labels, in column order.
        observed: ``(m, k)`` statistics of the data.
        replicated: ``(R, m, k)`` statistics of the replications.
        n_replications: Replicated data sets :math:`R`.
        nobs: Rows in the data set replicated, the effective sample.
        source: What was checked, for the summary title.
        notes: Remarks from the replicating result or model, prepended to
            the summary's notes.

    Note:
        The tail probabilities are resolved to :math:`1 / R`, so with the
        default 200 replications the smallest non-zero value is
        :math:`0.005`, and a p-value of exactly zero means no replication
        reached the data, not that the probability is zero. The summary
        says so below 200 replications.

    Example:
        Replications that match the data in every default respect:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = rng.standard_normal((200, 2))
        >>> rep = rng.standard_normal((300, 200, 2))
        >>> check = replication_check(y, rep, names=("a", "b"))
        >>> check.statistics
        ('mean', 'sd', 'acf1', 'kurtosis', 'arch1')
        >>> check.pvalues.shape, check.adequate()
        ((5, 2), True)

        Heavy-tailed data against Gaussian replications: the scale and the
        mean are matched, the kurtosis is not, and the name of the failing
        statistic is the diagnosis:

        >>> heavy = rng.standard_t(3, size=(200, 2))
        >>> heavy = (heavy - heavy.mean(axis=0)) / heavy.std(axis=0)
        >>> check = replication_check(heavy, rep, names=("a", "b"))
        >>> check.extreme()
        ('kurtosis[a]', 'kurtosis[b]')
        >>> check.pvalue("kurtosis", "a")
        0.0
        >>> check
        PredictiveCheckTest(posterior, 300 replications, 5 statistics x 2 variables, 2 in a tail)

    See Also:
        * :func:`posterior_predictive_check` -- replicates from a fitted
          result and builds this record.
        * :func:`prior_predictive_check` -- replicates from a model's prior.
        * :func:`replication_check` -- builds this record from replications
          the caller simulated.
        * :class:`~cultivars.bayes.chains.ConvergenceTest` -- the other
          per-quantity report on a sampled result, on the draws rather than
          on what they generate.

    References:
        Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A.
        (2019). Visualization in Bayesian workflow. *Journal of the Royal
        Statistical Society A*, 182(2), 389-402.

        Gelman, A., Meng, X.-L., & Stern, H. (1996). Posterior predictive
        assessment of model fitness via realized discrepancies. *Statistica
        Sinica*, 6(4), 733-760.

        Meng, X.-L. (1994). Posterior predictive p-values. *Annals of
        Statistics*, 22(3), 1142-1160.

        Rubin, D. B. (1984). Bayesianly justifiable and relevant frequency
        calculations for the applied statistician. *Annals of Statistics*,
        12(4), 1151-1172.
    """

    kind: str
    """``"posterior"`` or ``"prior"``: which distribution the replications came from.

    Chooses the interpretive note in :meth:`summary` -- the Meng (1994)
    caveat for a posterior check, the read-the-band instruction for a prior
    check -- and the title. Nothing numeric depends on it.
    """

    statistics: tuple[str, ...]
    """Names of the discrepancy statistics, one per row of :attr:`observed`.

    Drawn from ``mean``, ``sd``, ``min``, ``max``, ``skewness``,
    ``kurtosis`` (excess), ``acf1`` (lag-one autocorrelation), and
    ``arch1`` (lag-one autocorrelation of the squared demeaned series), in
    the order the check was asked for them; the default is ``("mean",
    "sd", "acf1", "kurtosis", "arch1")``. These are the keys
    :meth:`pvalue` accepts and the first half of every label
    :meth:`extreme` returns.
    """

    names: tuple[str, ...]
    """Variable labels, one per column of :attr:`observed`, in column order.

    Taken from the replicating result when it carries names, else
    ``y1 ... yk``, or ``y`` for a single series. These are the keys
    :meth:`pvalue` accepts for its ``name`` argument and the bracketed
    half of every label :meth:`extreme` returns.
    """

    observed: npt.NDArray[np.float64] = field(repr=False)
    r""":math:`T(y)`: ``(m, k)`` discrepancy statistics of the observed panel.

    Row ``i`` is ``statistics[i]`` evaluated on each column of the data;
    the ``observed`` column of :meth:`summary`. Computed once, on the same
    effective sample the replications imitate, so a statistic with a
    sample-size dependence -- the lag-one autocorrelation's small-sample
    bias, say -- is biased identically in :attr:`observed` and in
    :attr:`replicated` and the comparison remains fair.
    """

    replicated: npt.NDArray[np.float64] = field(repr=False)
    r""":math:`T(y^{\text{rep},(r)})`: ``(R, m, k)`` statistics of every replication.

    ``replicated[r, i, j]`` is ``statistics[i]`` on variable ``j`` of the
    ``r``-th replicated panel. This axis is what :attr:`pvalues` counts
    over and :meth:`replicated_quantiles` takes quantiles over. A
    statistic undefined on some replication -- ``acf1`` of a constant
    series -- is ``nan`` there, and that cell's p-value is ``nan`` in turn.
    The replicated panels themselves are not kept.
    """

    n_replications: int
    r"""Replicated data sets :math:`R`, the leading axis of :attr:`replicated`.

    Also the resolution of every tail probability, :math:`1 / R`. For a
    posterior check it is at most the number of retained draws, one
    replication per draw; for a prior check it is however many prior draws
    were requested.
    """

    nobs: int
    """Rows in the data set that was replicated, the effective sample.

    For a result fitted with ``p`` presample rows this is ``T - p``, the
    length of the panel the statistics were evaluated on, and each
    replication has the same number of rows. Recorded so the summary can
    say what sample size the tail probabilities refer to.
    """

    source: str
    """What was checked, as it appears in the summary title.

    The replicating result's own label -- ``"BVAR(2)"``, ``"SV(KSC)"`` --
    for the two convenience functions, or whatever :func:`replication_check`
    was given, ``"replications"`` by default.
    """

    notes: tuple[str, ...] = ()
    """Remarks the replicating result or model attached, prepended to the summary's notes.

    A result whose replication rule has a caveat -- a stochastic-volatility
    posterior replicating with the smoothed rather than the filtered
    variance path, a prior with dummy observations folded in -- states it
    here, so the caveat travels with the check. Empty for
    :func:`replication_check` unless the caller supplies some.
    """

    @property
    def pvalues(self) -> npt.NDArray[np.float64]:
        r"""``(m, k)`` tail probabilities :math:`\Pr(T(y^{\text{rep}}) \ge T(y))`.

        The share of replications whose statistic exceeds the observed one,
        plus half the share that equals it, so a statistic the model
        reproduces exactly reads :math:`0.5` and a discrete statistic --
        ``min`` or ``max`` on integer-valued data -- is not pushed to one by
        ties. A cell is ``nan`` when the statistic is undefined on any
        replication. Rows follow :attr:`statistics` and columns
        :attr:`names`. Recomputed on each access from :attr:`observed` and
        :attr:`replicated`; the cost is one comparison per cell and
        replication.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal((100, 2))
            >>> check = replication_check(y, rng.standard_normal((200, 100, 2)))
            >>> p = check.pvalues
            >>> p.shape, bool(np.all((0.0 <= p) & (p <= 1.0)))
            ((5, 2), True)
        """
        return _predictive_pvalues(self.observed, self.replicated)

    def pvalue(self, statistic: str, name: str | None = None) -> npt.NDArray[np.float64] | float:
        """One statistic's tail probabilities, for every variable or for one.

        Args:
            statistic: A name from :attr:`statistics`.
            name: A variable label from :attr:`names`; ``None`` returns the
                whole ``(k,)`` row.

        Returns:
            The ``(k,)`` row of :attr:`pvalues` for the statistic, or the
            single float for the named variable.

        Raises:
            SpecificationError: If the statistic or the variable is unknown.
                The message lists what the check does carry.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(1)
            >>> y = rng.standard_normal((100, 2))
            >>> check = replication_check(y, rng.standard_normal((200, 100, 2)))
            >>> check.pvalue("mean").shape
            (2,)
            >>> 0.0 <= check.pvalue("mean", "y1") <= 1.0
            True
            >>> check.pvalue("median")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: unknown statistic 'median'; ...
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
        """Quantiles of each replicated statistic across the replications.

        For a prior check this is the primary reading: the band says what
        the prior considers plausible for each feature of the data, on the
        data's own scale, and a band that spans orders of magnitude is the
        signature of a prior that has not been thought about. For a
        posterior check it is the band the observed statistic is being
        located in; the ``q05``, ``median``, and ``q95`` columns of
        :meth:`summary` are this method at its default levels.

        Args:
            levels: Probability levels in ``(0, 1)``, in the order the
                first axis of the result should carry them.

        Returns:
            ``(len(levels), m, k)``, rows following :attr:`statistics` and
            columns :attr:`names`. ``nan`` wherever a replication's
            statistic is undefined, since :func:`numpy.quantile` propagates
            it.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(2)
            >>> y = rng.standard_normal((100, 3))
            >>> check = replication_check(y, rng.standard_normal((200, 100, 3)))
            >>> bands = check.replicated_quantiles((0.1, 0.9))
            >>> bands.shape, bool(np.all(bands[0] <= bands[1]))
            ((2, 5, 3), True)
        """
        return np.asarray(np.quantile(self.replicated, levels, axis=0), dtype=np.float64)

    def _flags(self, level: float) -> npt.NDArray[np.bool_]:
        """Mark the ``(m, k)`` cells whose tail probability lies outside ``(level, 1 - level)``.

        The one place the two-sided rule is evaluated; :meth:`extreme`,
        :meth:`adequate`, and the marker column of :meth:`summary` all read
        it, so the three can never disagree about which cells are in a
        tail. The comparison is two-sided because a p-value near one is as
        much a failure as one near zero: the observed statistic then sits
        below every replication.

        ``nan`` p-values compare false on both sides, so an undefined cell
        is never flagged; the ``invalid`` floating-point warning that the
        comparison would otherwise raise on ``nan`` is suppressed rather
        than allowed to reach the caller as noise.

        Args:
            level: The tail size on each side, already validated by the
                caller to lie strictly inside ``(0, 0.5)``; this method does
                not check it.

        Returns:
            ``(m, k)`` booleans, ``True`` where the cell is in a tail.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(3)
            >>> y = 2.0 + rng.standard_normal((100, 1))
            >>> check = replication_check(y, rng.standard_normal((200, 100, 1)))
            >>> check._flags(0.05)[:, 0].tolist()
            [True, False, False, False, False]
        """
        p = self.pvalues
        with np.errstate(invalid="ignore"):
            return np.asarray((p < level) | (p > 1.0 - level))

    def extreme(self, *, level: float = _EXTREME_PVALUE) -> tuple[str, ...]:
        """Labels ``statistic[variable]`` with a tail probability outside ``(level, 1 - level)``.

        Args:
            level: The tail size on each side; the default marks a cell
                whose p-value is below 0.05 or above 0.95.

        Returns:
            The labels in row-major order over statistics and variables;
            empty when nothing sits in a tail. A ``nan`` p-value is never
            flagged.

        Raises:
            SpecificationError: If the level is not inside ``(0, 0.5)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(3)
            >>> y = 2.0 + rng.standard_normal((100, 1))
            >>> check = replication_check(y, rng.standard_normal((200, 100, 1)))
            >>> check.extreme()
            ('mean[y]',)
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
        r"""Whether no statistic sits in a tail of its replicated distribution.

        The complement of :meth:`extreme` being non-empty. For a posterior
        check, ``True`` says the model reproduces every checked feature of
        its own sample at the resolution the replications afford, which is
        a necessary condition for adequacy and not a sufficient one; the
        statistics were chosen, and a feature not among them is not
        checked. For a prior check it says only that the data are not in
        the prior's tails, which a very diffuse prior satisfies trivially.

        The verdict is over ``m * k`` cells at once with no multiplicity
        adjustment, and that is deliberate: the check is a diagnostic that
        names what to look at, not a test with a size. Under a correct
        model with independent statistics the chance that at least one of
        ten cells lands in a 5% tail on each side is :math:`1 - 0.9^{10}
        \approx 0.65`, and Meng's concentration of posterior p-values near
        one half is what keeps the realized rate well below that. A single
        flagged cell near the boundary is a prompt to look at its row in
        :meth:`summary`, not a rejection.

        Args:
            level: The tail size on each side, as for :meth:`extreme`.

        Returns:
            ``True`` when :meth:`extreme` is empty at this level.

        Raises:
            SpecificationError: If the level is not inside ``(0, 0.5)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(12)
            >>> y = rng.standard_normal((100, 2))
            >>> check = replication_check(y, rng.standard_normal((200, 100, 2)))
            >>> check.adequate(), check.adequate(level=0.4)
            (True, False)
        """
        return not self.extreme(level=level)

    def summary(self, *, level: float = _EXTREME_PVALUE) -> SummaryTable:
        """Render as a table: one row per statistic and variable.

        Each row carries the observed statistic, the 5%, 50%, and 95%
        quantiles of its replicated distribution, the tail probability to
        three decimals, and a ``*`` when that probability is outside
        ``(level, 1 - level)``; the metadata block records the kind, the
        replication and observation counts, the table's dimensions, and
        the verdict. The notes carry, in order, whatever the replicating
        result attached (:attr:`notes`), the list of flagged cells when
        there are any (the first eight, then an ellipsis), the reading rule
        for the check's kind, and below 200 replications the resolution of
        the p-values.

        Args:
            level: The tail size on each side that marks a row.

        Returns:
            The :class:`~cultivars._core.SummaryTable`, titled
            ``"{Kind} predictive check: {source}"``; ``print`` it, display
            it in a notebook, or call its ``to_pandas()``.

        Raises:
            SpecificationError: If the level is not inside ``(0, 0.5)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(5)
            >>> y = rng.standard_normal((100, 1))
            >>> check = replication_check(y, rng.standard_normal((200, 100, 1)))
            >>> table = check.summary()
            >>> table.columns
            ('statistic', 'variable', 'observed', 'q05', 'median', 'q95', 'p', '')
            >>> len(table.rows), table.rows[0][:2]
            (5, ('mean', 'y'))
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
        """One line: the kind, the replication count, the table's shape, and the verdict.

        The dataclass ``repr`` is disabled (``repr=False``) because the
        arrays would dominate it; this one is what a bare ``check`` shows
        at a prompt and what appears inside containers. The verdict is
        evaluated at the default level, so the line matches
        :meth:`adequate` and :meth:`extreme` called without arguments.
        The record has no ``__str__`` of its own, so ``print(check)`` shows
        this line too; ``print(check.summary())`` renders the table.

        Returns:
            ``PredictiveCheckTest(<kind>, <R> replications, <m> statistics x
            <k> variables, <verdict>)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(6)
            >>> y = rng.standard_normal((100, 2))
            >>> rep = rng.standard_normal((200, 100, 2))
            >>> replication_check(y, rep, kind="prior")  # doctest: +ELLIPSIS
            PredictiveCheckTest(prior, 200 replications, 5 statistics x 2 variables, ...)
        """
        extreme = self.extreme()
        verdict = "no statistic in a tail" if not extreme else f"{len(extreme)} in a tail"
        return (
            f"PredictiveCheckTest({self.kind}, {self.n_replications} replications, "
            f"{len(self.statistics)} statistics x {len(self.names)} variables, {verdict})"
        )


def replication_check(
    observed: npt.ArrayLike,
    replicated: npt.ArrayLike,
    *,
    kind: str = "posterior",
    statistics: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
    source: str = "replications",
    notes: Sequence[str] = (),
) -> PredictiveCheckTest:
    r"""A predictive check on replications that came from elsewhere.

    The comparison :func:`posterior_predictive_check` and
    :func:`prior_predictive_check` run, exposed for replications the caller
    simulated: from a sampler outside the package, from a result whose
    replication rule is their own, or from a package result whose
    ``posterior_replications`` output they have transformed first (a check
    on log returns of a model fitted to levels, say). Each discrepancy
    statistic is evaluated column by column on the observed panel and on
    every replicated panel, and the record holds those statistics; the
    panels themselves are discarded. The two convenience functions are
    this function plus the simulation and the labelling.

    Args:
        observed: The data, ``(n,)`` for one series or ``(n, k)`` for a
            panel; anything :func:`numpy.asarray` accepts. A one-dimensional
            series is promoted to a single column.
        replicated: The replicated data sets, ``(R, n)`` or ``(R, n, k)``
            to match, every replication the same shape as ``observed``. At
            least 20 are required, since fewer cannot resolve a tail
            probability to better than :math:`0.05`.
        kind: ``"posterior"`` or ``"prior"``. It changes no number, only
            how :meth:`~PredictiveCheckTest.summary` tells the reader to
            interpret the tail probabilities.
        statistics: Discrepancy statistics to compute, from ``mean``,
            ``sd``, ``min``, ``max``, ``skewness``, ``kurtosis`` (excess),
            ``acf1`` (lag-one autocorrelation), and ``arch1`` (lag-one
            autocorrelation of the squared demeaned series), in the order
            the rows should carry them; duplicates are dropped. ``None``
            selects ``mean``, ``sd``, ``acf1``, ``kurtosis``, ``arch1``, which
            between them detect a misplaced level, a wrong scale, missed
            persistence, fat tails, and volatility clustering.
        names: One label per column of ``observed``. Defaults to
            ``y1 ... yk``, or ``y`` for a single series.
        source: What was checked, for the summary title.
        notes: Remarks to carry on the record, prepended to the summary's
            notes; the place to state a caveat about how the replications
            were made.

    Returns:
        The :class:`PredictiveCheckTest`, with ``observed`` of shape
        ``(m, k)`` and ``replicated`` of shape ``(R, m, k)`` for ``m``
        statistics.

    Raises:
        SpecificationError: If ``kind`` is neither ``"posterior"`` nor
            ``"prior"``, if a statistic name is unknown or the request is
            empty, or if ``names`` does not have one label per column.
        DimensionError: If ``observed`` is not one- or two-dimensional, if
            ``replicated`` is not two- or three-dimensional or its
            replications are not the shape of ``observed``, or if there are
            fewer than 20 replications.
        NumericalError: If any replication contains a non-finite value.

    Note:
        A statistic that is undefined on some replication -- ``acf1`` of a
        constant series -- is ``nan`` there and gives that cell a ``nan``
        tail probability, which is reported but never flagged. Finite but
        pathological replications (an explosive draw that reaches
        :math:`10^{300}`) are accepted; they show up as a replicated
        quantile band that dwarfs the observed statistic, which for a prior
        check is exactly the finding.

    See Also:
        * :func:`posterior_predictive_check` -- simulates the replications
          from a fitted result and calls this.
        * :func:`prior_predictive_check` -- simulates them from a model's
          prior and calls this.
        * :class:`PredictiveCheckTest` -- the record returned, with the
          reading rules for each kind.

    References:
        Gelman, A., Meng, X.-L., & Stern, H. (1996). Posterior predictive
        assessment of model fitness via realized discrepancies. *Statistica
        Sinica*, 6(4), 733-760.

    Example:
        Replications from a model the caller wrote -- here an AR(1) whose
        coefficient is drawn from a posterior-like distribution -- checked
        against the series:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = rng.standard_normal(100)
        >>> rep = rng.standard_normal((100, 100))
        >>> check = replication_check(y, rep, statistics=["mean", "sd"])
        >>> check.observed.shape, check.pvalues.shape
        ((2, 1), (2, 1))
        >>> check.names, check.source
        (('y',), 'replications')

        Too few replications are refused, since the resulting p-values
        would be meaningless:

        >>> replication_check(y, rep[:10])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.DimensionError: 10 replications; at least 20 are needed ...
    """
    if kind not in ("posterior", "prior"):
        raise SpecificationError(f"kind must be 'posterior' or 'prior'; got {kind!r}.")
    chosen = _validate_statistics(
        statistics, known=_DISCREPANCY_NAMES, default=_DISCREPANCY_STATISTICS
    )
    data, reps = _validate_replications(observed, replicated, minimum=_MIN_REPLICATIONS)
    k = data.shape[1]
    labels = (
        (tuple(f"y{i + 1}" for i in range(k)) if k > 1 else ("y",))
        if names is None
        else tuple(str(name) for name in names)
    )
    if len(labels) != k:
        raise SpecificationError(f"{len(labels)} names for {k} variables.")
    observed_stats = _discrepancy_statistics(data, chosen)
    replicated_stats = np.stack([_discrepancy_statistics(panel, chosen) for panel in reps])
    return PredictiveCheckTest(
        kind=kind,
        statistics=chosen,
        names=labels,
        observed=observed_stats,
        replicated=replicated_stats,
        n_replications=int(reps.shape[0]),
        nobs=int(data.shape[0]),
        source=source,
        notes=tuple(notes),
    )


def posterior_predictive_check(
    result: ReplicatingResult,
    *,
    statistics: Sequence[str] | None = None,
    n_replications: int = 200,
    seed: int | np.random.Generator | None = None,
) -> PredictiveCheckTest:
    r"""Replicate the sample from the posterior and compare its features with the data.

    For each of ``n_replications`` retained posterior draws
    :math:`\theta^{(r)}`, the result simulates a data set
    :math:`y^{\text{rep},(r)} \sim p(y \mid \theta^{(r)})` of the same shape
    as the effective sample it was fitted to, with fresh innovations and,
    for a conditional model, the same presample. The draws are taken
    evenly across the kept sequence rather than from its first stretch, so
    the replications span the posterior; asking for more replications than
    draws cycles through the draws with new innovations. The observed and
    replicated panels then go to :func:`replication_check`, and the result's
    own remarks on how it replicates -- a stochastic-volatility posterior
    replicating under the smoothed variance path, say -- ride along as the
    record's :attr:`~PredictiveCheckTest.notes`.

    This is the check of Gelman, Meng, and Stern (1996): the same data fit
    the parameters and judge the fit, so the tail probabilities are
    conservative in the sense the :class:`PredictiveCheckTest` docstring
    describes, and a value in a tail is a stronger finding than its size
    suggests.

    Args:
        result: A fitted result that can replicate its sample, that is,
            one satisfying :class:`~cultivars._core.ReplicatingResult` by
            exposing ``observed`` and ``posterior_replications()``: the
            Bayesian VAR family
            (:class:`~cultivars.multivariate.large_dim.bayesian.BVAR`,
            :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR`,
            :class:`~cultivars.multivariate.large_dim.student.StudentBVAR`,
            :class:`~cultivars.multivariate.large_dim.volatility.BVARSV`)
            and the stochastic-volatility posterior of
            :class:`~cultivars.univariate.stochastic_volatility.SV`.
        statistics: Discrepancy statistics, as for
            :func:`replication_check`; ``None`` selects ``mean``, ``sd``,
            ``acf1``, ``kurtosis``, ``arch1``.
        n_replications: Replicated data sets, one per posterior draw used.
            At least 20; 200 resolves a tail probability to
            :math:`0.005`.
        seed: Seed or generator for the replications' innovations. An
            integer gives a reproducible check; a
            :class:`numpy.random.Generator` is used in place and advanced.

    Returns:
        The :class:`PredictiveCheckTest` with ``kind="posterior"``, the
        result's variable names, its label as ``source``, and its
        replication remarks as ``notes``.

    Raises:
        SpecificationError: If the result cannot replicate its sample --
            a point estimate has no posterior to replicate from, and neither
            has a result whose draws were not retained -- or if a statistic
            name is unknown.
        DimensionError: If fewer than 20 replications are asked for.

    See Also:
        * :func:`prior_predictive_check` -- the same comparison against what
          the prior generates, before fitting.
        * :func:`replication_check` -- the comparison on replications made
          elsewhere.
        * :func:`~cultivars.bayes.chains.convergence` -- whether the draws
          the replications came from have converged, which this check
          presumes.

    References:
        Gelman, A., Meng, X.-L., & Stern, H. (1996). Posterior predictive
        assessment of model fitness via realized discrepancies. *Statistica
        Sinica*, 6(4), 733-760.

        Meng, X.-L. (1994). Posterior predictive p-values. *Annals of
        Statistics*, 22(3), 1142-1160.

    Example:
        A stochastic-volatility posterior checked on the two features that
        distinguish it from constant variance -- the scale and the
        volatility clustering:

        >>> import numpy as np
        >>> from cultivars.univariate.stochastic_volatility import SV
        >>> rng = np.random.default_rng(1)
        >>> h = np.zeros(300)
        >>> for t in range(1, 300):
        ...     h[t] = 0.95 * h[t - 1] + 0.3 * rng.standard_normal()
        >>> y = np.exp(0.5 * h) * rng.standard_normal(300)
        >>> post = SV(y, mean="zero").sample(n_draws=600, n_burn=200, seed=0)
        >>> check = posterior_predictive_check(post, statistics=["sd", "arch1"], seed=0)
        >>> check.statistics, check.names, check.kind
        (('sd', 'arch1'), ('y',), 'posterior')
        >>> check.n_replications, check.nobs
        (200, 300)

        A point estimate is refused:

        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> ols = VAR(rng.standard_normal((80, 2)), order=1).fit()
        >>> posterior_predictive_check(ols)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: VARResult cannot replicate its sample: ...
    """
    if not isinstance(result, ReplicatingResult):
        raise SpecificationError(
            f"{type(result).__name__} cannot replicate its sample: a posterior predictive "
            "check needs a result exposing observed and posterior_replications(), which the "
            "Bayesian VAR family (BVAR, GibbsBVAR, StudentBVAR, BVARSV) and the "
            "stochastic-volatility posterior provide. A point estimate has no posterior to "
            "replicate from."
        )
    replicated = result.posterior_replications(n_replications, seed=seed)
    observed = np.asarray(result.observed, dtype=np.float64)
    k = 1 if observed.ndim == 1 else observed.shape[1]
    remarks = getattr(result, "_replication_notes", None)
    notes = tuple(remarks()) if callable(remarks) else ()
    return replication_check(
        observed,
        replicated,
        kind="posterior",
        statistics=statistics,
        names=_variable_names(result, k),
        source=_source_label(result),
        notes=notes,
    )


def prior_predictive_check(
    model: ReplicatingModel,
    *,
    statistics: Sequence[str] | None = None,
    n_replications: int = 200,
    seed: int | np.random.Generator | None = None,
    **hyperparameters: object,
) -> PredictiveCheckTest:
    r"""Replicate the sample from the prior and locate the data in what it generates.

    For each of ``n_replications`` draws :math:`\theta^{(r)} \sim p(\theta)`
    from the model's prior, a data set :math:`y^{\text{rep},(r)} \sim p(y
    \mid \theta^{(r)})` of the shape of the effective sample is simulated,
    and the observed panel is located in the replicated distribution of
    each discrepancy statistic by :func:`replication_check`. No fitting
    happens: the check is what Gabry et al. (2019) call the prior
    predictive step of a Bayesian workflow, and its reading is the
    replicated quantile band of each statistic on the data's own scale --
    whether the prior deems the data's scale, persistence, and tail weight
    plausible, or whether it puts most of its mass on data that look
    nothing like these. The tail probability is reported but is not a
    test; the data are one draw the prior may or may not cover.

    A Minnesota-type prior centred on the unit root generates explosive
    systems at some rate by construction. The check measures that rate --
    the share of replications reaching beyond a thousand times the largest
    observation -- and records it as a note when it is non-zero, so the
    reader can decide whether that mass is intended.

    Args:
        model: An unfitted model that can draw from its prior, that is,
            one satisfying :class:`~cultivars._core.ReplicatingModel` by
            exposing ``observed`` and ``prior_replications()``:
            :class:`~cultivars.multivariate.large_dim.bayesian.BVAR`,
            :class:`~cultivars.multivariate.large_dim.student.StudentBVAR`,
            :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR` under
            a static prior without dummy observations, and
            :class:`~cultivars.univariate.stochastic_volatility.SV`. The
            stochastic-volatility BVAR is excluded: its random-walk
            volatility prior has no stationary distribution to replicate
            from.
        statistics: Discrepancy statistics, as for
            :func:`replication_check`; ``None`` selects ``mean``, ``sd``,
            ``acf1``, ``kurtosis``, ``arch1``.
        n_replications: Prior draws, and so replicated data sets. At least
            20.
        seed: Seed or generator for the prior draws and the innovations.
        **hyperparameters: Prior settings the model's ``prior_replications``
            accepts by name, passed through unchanged: ``df`` for
            ``StudentBVAR``; ``prior_mu``, ``prior_phi``, ``prior_sigma2``
            for ``SV``, exactly as its ``sample()`` takes them. A name the
            model does not accept is refused before anything is simulated.

    Returns:
        The :class:`PredictiveCheckTest` with ``kind="prior"``, the model's
        variable names, ``source`` naming the model and its prior label
        when the model carries one, and the explosive-share remark in
        ``notes`` when it applies.

    Raises:
        SpecificationError: If the model cannot draw from its prior, if a
            statistic name is unknown, or if a keyword in
            ``hyperparameters`` is not one the model's ``prior_replications``
            takes; the message lists the names it does take.
        DimensionError: If fewer than 20 replications are asked for.

    Note:
        Prior replications of an autoregressive model are simulated from
        the observed presample, so a prior check on a series in levels
        starts every replication where the data start; the ``mean``
        statistic then reads the drift the prior admits over the sample
        length, not a prior on the level itself.

    See Also:
        * :func:`posterior_predictive_check` -- the same comparison after
          fitting.
        * :func:`replication_check` -- the comparison on replications made
          elsewhere.
        * :class:`~cultivars.bayes.priors.NormalInverseWishartPrior` -- the
          conjugate prior whose tightness the check is usually used to
          calibrate.

    References:
        Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A.
        (2019). Visualization in Bayesian workflow. *Journal of the Royal
        Statistical Society A*, 182(2), 389-402.

        Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter,
        B., Yao, Y., Kennedy, L., Gabry, J., Bürkner, P.-C., & Modrák, M.
        (2020). Bayesian workflow. arXiv:2011.01808.

    Example:
        What the default Minnesota prior says a random-walk panel could
        look like, on the two features that prior is about:

        >>> import numpy as np
        >>> from cultivars.multivariate.large_dim.bayesian import BVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.cumsum(rng.standard_normal((120, 2)), axis=0)
        >>> check = prior_predictive_check(BVAR(y, order=2), statistics=["sd", "acf1"], seed=0)
        >>> check.kind, check.replicated.shape
        ('prior', (200, 2, 2))
        >>> check.source
        'BVAR under niw(l1=0.2, l3=1, l4=100)'

        A hyperparameter the model does not take is refused by name:

        >>> prior_predictive_check(BVAR(y, order=2), df=5)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: BVAR.prior_replications does not take ['df']; ...
    """
    if not isinstance(model, ReplicatingModel):
        raise SpecificationError(
            f"{type(model).__name__} cannot draw from its prior: a prior predictive check "
            "needs a model exposing observed and prior_replications(), which BVAR, "
            "StudentBVAR, GibbsBVAR (static prior, no dummy observations), and SV provide. "
            "The stochastic-volatility BVAR's random-walk volatility prior has no stationary "
            "distribution to replicate from; check it on the posterior side."
        )
    accepted = inspect.signature(model.prior_replications).parameters
    unknown = sorted(name for name in hyperparameters if name not in accepted)
    if unknown:
        raise SpecificationError(
            f"{type(model).__name__}.prior_replications does not take {unknown}; it takes "
            f"{[name for name in accepted if name not in ('n_replications', 'seed')]}."
        )
    replicated = cast(Any, model).prior_replications(n_replications, seed=seed, **hyperparameters)
    observed = np.asarray(model.observed, dtype=np.float64)
    k = 1 if observed.ndim == 1 else observed.shape[1]
    reach = np.abs(replicated).reshape(replicated.shape[0], -1).max(axis=1)
    wild = float((reach > 1e3 * np.abs(observed).max()).mean())
    notes = (
        (
            f"{wild:.0%} of the prior replications reach beyond a thousand times the largest "
            "observation: the prior admits explosive systems at that rate. A Minnesota prior "
            "centered on the unit root does so by construction; tighten the lag-one "
            "variance if that mass is unwanted.",
        )
        if wild > 0.0
        else ()
    )
    label = getattr(model, "prior", None)
    prior_label = getattr(label, "_label", None)
    source = (
        f"{type(model).__name__} under {prior_label()}"
        if callable(prior_label)
        else type(model).__name__
    )
    return replication_check(
        observed,
        replicated,
        kind="prior",
        statistics=statistics,
        names=_variable_names(model, k),
        source=source,
        notes=notes,
    )
