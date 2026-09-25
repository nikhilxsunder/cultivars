# filepath: /src/cultivars/diagnostics/nonlinearity.py
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
r"""Nonlinearity gateways: the tests that justify a regime model before it is fit.

A threshold or smooth-transition model is a claim that a linear
autoregression is wrong in a specific way, and the claim should be
tested on the linear model's own terms before the nonlinear one is
estimated. The difficulty every such test meets is that the parameters
of the alternative, the threshold or the transition's location and
speed, are not identified under the null, so a likelihood ratio has no
chi-squared limit and each test finds its own way around that. Four do
so from different directions. Teräsvirta's LM test replaces the
transition function by its Taylor expansion, so the alternative becomes
the cubic auxiliary regression

.. math::

   \hat u_t = \beta_0^\top w_t + \sum_{i = 1}^{p} \sum_{j = 1}^{3}
   \beta_{ji}\, y_{t-i}\, y_{t-d}^{\,j} + e_t,

whose exclusion restriction is an ordinary :math:`F` test, and it
carries a device for choosing between logistic and exponential
transitions. Tsay's arranged autoregression sorts the sample by the
threshold variable and asks whether the recursive predictive residuals
drift as the recursion crosses a threshold; it names no functional
form. Hansen's sup-F is the likelihood-ratio test of a SETAR against
its linear restriction, the sup taken over candidate thresholds, whose
p-value is simulated from fixed-regressor draws because the sup has no
tabulated law. Ramsey's RESET is the general-purpose check that the
linear fit has left a smooth function of itself in the residual. The
BDS statistic is the odd one out: applied to residuals, it tests
independence against any dependence at all, linear or not, and rejects
for neglected conditional heteroskedasticity as readily as for a
threshold, which is why it is read after an ARCH test and not instead
of one.

Two commitments shape the surface. First, a rejection says what it
rejects toward. Every :class:`LinearityTest` names its ``alternative``,
the Teräsvirta record carries the escalation sequence that chooses the
transition, and the Hansen record carries the threshold it found, so a
reader who gets four verdicts on one series can tell a threshold from
a smooth transition from a misspecification with no regime in it.
Second, the reference distribution is the one the test needs and not a
convenient stand-in: the :math:`F` form where the chi-squared over-
rejects at the sample sizes these tests meet, a simulated sup where no
table exists, and, for BDS, a size warning on the record itself below
the sample length at which the normal reference holds.

All five take a series, not residuals from a fit the caller has already
made: the linear autoregression of the stated ``order`` is fitted
inside each test, so the null is always the AR(:math:`p`) and the
degrees of freedom are always right. The BDS test is the exception and
takes whatever it is given, usually standardized residuals.

Layout. :class:`LinearityTest` is the record for the four
model-aimed tests and :class:`BDSTest` for the independence test, both
:class:`~cultivars.diagnostics.hypothesis.HypothesisTest` records with
``reject`` and a summary. :func:`terasvirta`, :func:`tsay`,
:func:`ramsey_reset`, :func:`hansen_threshold`, and :func:`bds` are the
producers, sharing ``_validate_specification`` for the sample-length
check. The numerics live in ``_core``: ``_terasvirta_lm`` runs the
cubic auxiliary regression and the escalation steps, ``_tsay_arranged``
the sorted recursion, ``_reset_test`` the augmented regression,
``_hansen_threshold`` the grid search and fixed-regressor simulation,
``_bds`` the correlation integrals, and ``_simulated_critical_values``
turns Hansen's draws into critical values; the trimming, grid,
replication, and radius defaults are the module's named constants.

References:
    Teräsvirta, T. (1994). Specification, estimation, and evaluation of
    smooth transition autoregressive models. *Journal of the
    American Statistical Association*, 89(425), 208-218.

    Tsay, R. S. (1989). Testing and modeling threshold autoregressive
    processes. *Journal of the American Statistical Association*,
    84(405), 231-240.

    Hansen, B. E. (1996). Inference when a nuisance parameter is not
    identified under the null hypothesis. *Econometrica*, 64(2),
    413-430.

    Ramsey, J. B. (1969). Tests for specification errors in classical
    linear least-squares regression analysis. *Journal of the Royal
    Statistical Society B*, 31(2), 350-371.

    Brock, W. A., Dechert, W. D., Scheinkman, J. A., & LeBaron, B.
    (1996). A test for independence based on the correlation
    dimension. *Econometric Reviews*, 15(3), 197-235.

    Davies, R. B. (1987). Hypothesis testing when a nuisance parameter
    is present only under the alternative. *Biometrika*, 74(1), 33-43.

    Teräsvirta, T., Tjøstheim, D., & Granger, C. W. J. (2010).
    *Modelling Nonlinear Economic Time Series*. Oxford University
    Press.

Example:
    A two-regime threshold autoregression: the model-aimed tests reject
    linearity and Hansen locates the threshold. An ARCH series, which
    has no regime in its mean, is invisible to Tsay and caught by BDS,
    the division of labour the module is built around:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros(400)
    >>> for t in range(1, 400):
    ...     slope = 0.8 if y[t - 1] < 0.0 else -0.5
    ...     y[t] = slope * y[t - 1] + rng.standard_normal()
    >>> tsay(y, order=1).reject(), terasvirta(y, order=1).reject()
    (True, True)
    >>> verdict = hansen_threshold(y, order=1, replications=500, seed=0)
    >>> verdict.reject(), bool(abs(verdict.threshold) < 0.5)
    (True, True)
    >>> e = np.zeros(400)
    >>> for t in range(1, 400):
    ...     e[t] = np.sqrt(0.2 + 0.7 * e[t - 1] ** 2) * rng.standard_normal()
    >>> tsay(e, order=1).reject(), bds(e).reject()
    (False, True)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core import (
    _BDS_RADIUS,
    _CRITICAL_LEVELS,
    _DEFAULT_GRID,
    _DEFAULT_TRIM,
    _HANSEN_REPLICATIONS,
    _NULL,
    SummaryTable,
    _bds,
    _hansen_threshold,
    _reset_test,
    _simulated_critical_values,
    _terasvirta_lm,
    _tsay_arranged,
    _validate_specification,
    validate_endog,
    validate_open_interval,
    validate_order,
)
from .._internals import _HypothesisTest
from ..exceptions import SpecificationError

__all__ = [
    "BDSTest",
    "LinearityTest",
    "bds",
    "hansen_threshold",
    "ramsey_reset",
    "terasvirta",
    "tsay",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class LinearityTest(_HypothesisTest):
    r"""Verdict of a test of a linear autoregression against a nonlinear alternative.

    The four members share a null, that the AR(:math:`p`) is correctly
    specified, and differ in the alternative they aim at, which the
    record names. Three are :math:`F` tests on an auxiliary regression
    of the linear model's residuals :math:`\hat u_t` on its regressors
    and a set of added terms,

    .. math::

       F = \frac{(SSR_0 - SSR_1) / q}{SSR_1 / (n - k - q)}
       \;\sim\; F(q,\; n - k - q),

    with :math:`q` added terms: cubic products of the lags and the
    transition variable for Teräsvirta, powers of the fitted values for
    RESET, and, for Tsay, the arranged autoregression's predictive
    residuals regressed on the lags. Hansen's test alone is a sup
    statistic, the largest :math:`F` over candidate thresholds, whose
    law has no table because the threshold is unidentified under the
    null; its p-value and critical values are simulated, it locates
    the threshold it found, and its ``df`` is ``None``. A rejection by
    Tsay or Hansen says "threshold"; a rejection by Teräsvirta says
    "smooth transition" and, through the escalation sequence carried as
    ``companions``, which kind; a rejection by RESET says only that
    something smooth was left in the residual.

    Attributes:
        name: The test.
        statistic: The test statistic, ``F`` for the regression tests
            and sup-``F`` for Hansen.
        pvalue: Its p-value.
        df: ``(numerator, denominator)`` degrees of freedom of an
            ``F`` reference; ``None`` for a simulated law.
        null: The hypothesis under test.
        alternative: What a rejection points to.
        order: Autoregressive order of the linear null.
        delay: Delay of the transition variable, or ``None`` where the
            test has none.
        nobs: Observations in the auxiliary regression.
        threshold: Hansen's estimated threshold; ``None`` otherwise.
        critical_values: Simulated ``{"1%", "5%", "10%"}`` critical
            values for Hansen; ``None`` otherwise.
        suggestion: Teräsvirta's model-selection reading, ``"LSTAR"``
            or ``"ESTAR"``; ``None`` otherwise.
        companions: The steps of Teräsvirta's escalation sequence.

    Note:
        The Teräsvirta record always carries a ``suggestion``, whether
        or not linearity is rejected: the rule compares the three
        escalation p-values and one of them is always smallest. Read it
        only after :meth:`reject` says there is a transition to choose
        between. The tests are for the conditional mean; neglected
        conditional heteroskedasticity inflates all four, so run an
        ARCH test on the linear residuals first and read a rejection
        here in its light.

    See Also:
        * :func:`terasvirta`, :func:`tsay`, :func:`reset`,
          :func:`hansen_threshold` -- the four producers.
        * :class:`BDSTest` -- the independence test that catches any
          dependence, linear or not.
        * :class:`~cultivars.univariate.threshold.SETAR`,
          :class:`~cultivars.univariate.smooth_transition.LSTAR`, and
          :class:`~cultivars.univariate.smooth_transition.ESTAR` -- the
          models a rejection licenses.

    References:
        Teräsvirta, T. (1994). Specification, estimation, and evaluation
        of smooth transition autoregressive models. *Journal of the
        American Statistical Association*, 89(425), 208-218.

        Tsay, R. S. (1989). Testing and modeling threshold autoregressive
        processes. *Journal of the American Statistical Association*,
        84(405), 231-240.

        Hansen, B. E. (1996). Inference when a nuisance parameter is not
        identified under the null hypothesis. *Econometrica*, 64(2),
        413-430.

    Example:
        A two-regime threshold autoregression is caught by all four,
        each naming its own alternative:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(400)
        >>> for t in range(1, 400):
        ...     slope = 0.8 if y[t - 1] < 0.0 else -0.5
        ...     y[t] = slope * y[t - 1] + rng.standard_normal()
        >>> for test in (terasvirta(y, order=1), tsay(y, order=1), reset(y, order=1)):
        ...     print(f"{test.name:<11} {test.alternative:<24} {test.df}  {test.reject()}")
        Teräsvirta  smooth transition        (3, 394)  True
        Tsay        threshold                (2, 336)  True
        RESET       smooth misspecification  (2, 395)  True
        >>> test = hansen_threshold(y, order=1, replications=500, seed=0)
        >>> test.df, test.reject(), bool(abs(test.threshold) < 0.5)
        (None, True, True)
    """

    name: str
    """The test: ``"Teräsvirta"``, ``"Tsay"``, ``"RESET"``, ``"Hansen sup-F"``, or a step name.

    An escalation-step companion is named ``"H04"``, ``"H03"``, or ``"H02"``.
    """
    pvalue: float
    """Upper-tail p-value: from the ``F`` reference, or simulated for Hansen."""
    df: tuple[int, int] | None
    """``(numerator, denominator)`` degrees of freedom of the ``F`` reference, or ``None``.

    ``None`` on the Hansen record, whose sup statistic is read against
    simulated critical values rather than an ``F`` law.
    """
    null: str
    """The hypothesis under test, ``"linear AR(p) is correctly specified"`` with ``p`` filled in.

    An escalation-step companion carries its own null instead, the
    restriction that step imposes on the cubic auxiliary regression.
    """
    alternative: str
    """What a rejection points to.

    ``"threshold"`` for Tsay and Hansen, ``"smooth transition"`` for
    Teräsvirta, ``"smooth misspecification"`` for RESET.
    """
    order: int
    """Autoregressive order :math:`p` of the linear null."""
    delay: int | None
    """Delay :math:`d` of the transition variable :math:`y_{t-d}`, or ``None`` for RESET.

    The value given for Teräsvirta and Tsay; the delay Hansen's search
    selected when it was left to choose.
    """
    nobs: int
    """Observations in the auxiliary regression, ``T - max(order, delay)``."""
    threshold: float | None = None
    """Hansen's least-squares threshold estimate; ``None`` otherwise.

    On the scale of the transition variable :math:`y_{t-d}`.
    """
    critical_values: dict[str, float] | None = None
    """Simulated ``{"1%", "5%", "10%"}`` critical values of Hansen's sup-``F``.

    ``None`` on the three ``F`` tests, whose reference is tabulated.
    """
    suggestion: str | None = None
    """Teräsvirta's reading of the escalation sequence; ``None`` off that record.

    ``"ESTAR"`` when the middle step ``H03`` rejects most strongly,
    ``"LSTAR"`` otherwise; always set on a Teräsvirta record, and
    meaningful only when linearity is rejected.
    """
    companions: tuple[LinearityTest, ...] = field(default=(), repr=False)
    """The escalation steps ``H04``, ``H03``, ``H02`` of a Teräsvirta record, each a full record.

    Empty on the other three tests and on the steps themselves. Kept
    out of the repr.
    """

    def _row(self) -> tuple[str, ...]:
        """One table row: name, statistic, degrees of freedom, p-value.

        The degrees-of-freedom cell is blank for a simulated law.

        Returns:
            Four strings.

        Example:
            >>> import numpy as np
            >>> reset(np.random.default_rng(0).standard_normal(200), order=1)._row()[:1]
            ('RESET',)
        """
        df = "" if self.df is None else f"({self.df[0]}, {self.df[1]})"
        return (self.name, f"{self.statistic:.4f}", df, f"{self.pvalue:.4f}")

    def _summary_table(self) -> SummaryTable:
        """Render as a table, companions included.

        The header names the null, the alternative, the order, the
        delay when the test has one, the sample, the verdict, and
        Hansen's threshold when there is one; the rows are this record
        and its companions; the notes carry Hansen's simulated critical
        values and Teräsvirta's model-selection reading.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``str()``
            and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> test = terasvirta(np.random.default_rng(0).standard_normal(200), order=1)
            >>> table = test._summary_table()
            >>> [row[0] for row in table.rows]
            ['Teräsvirta', 'H04', 'H03', 'H02']
        """
        verdict = "reject linearity" if self.reject() else "keep linearity"
        metadata = [
            ("Null", self.null),
            ("Alternative", self.alternative),
            ("Order", str(self.order)),
            ("Observations", str(self.nobs)),
            ("Verdict at 5%", verdict),
        ]
        if self.delay is not None:
            metadata.insert(3, ("Delay", str(self.delay)))
        if self.threshold is not None:
            metadata.append(("Threshold", f"{self.threshold:.4f}"))
        notes: list[str] = []
        if self.critical_values is not None:
            cv = ", ".join(f"{k}: {v:.3f}" for k, v in self.critical_values.items())
            notes.append(f"Simulated critical values {cv}.")
        if self.suggestion is not None:
            notes.append(
                f"Escalation sequence H04, H03, H02 read by Teräsvirta's rule: the "
                f"strongest rejection points to {self.suggestion}."
            )
        return SummaryTable(
            title=f"{self.name} Linearity Test",
            metadata=tuple(metadata),
            columns=("test", "statistic", "df", "p-value"),
            rows=(self._row(), *(c._row() for c in self.companions)),
            notes=tuple(notes),
        )

    def __repr__(self) -> str:
        """One-line verdict: name, statistic, p-value, order, observations.

        Returns:
            The repr string.

        Example:
            >>> import numpy as np
            >>> tsay(np.random.default_rng(0).standard_normal(200), order=1)
            LinearityTest(name='Tsay', statistic=0.8350, pvalue=0.4358, order=1, nobs=199)
        """
        return (
            f"LinearityTest(name={self.name!r}, statistic={self.statistic:.4f}, "
            f"pvalue={self.pvalue:.4g}, order={self.order}, nobs={self.nobs})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class BDSTest(_HypothesisTest):
    r"""Verdict of the BDS test of independence at one embedding dimension.

    The correlation integral :math:`C_m(\varepsilon)` is the fraction of
    pairs of :math:`m`-histories :math:`(y_t, \ldots, y_{t + m - 1})`
    lying within :math:`\varepsilon` of each other in the sup norm.
    Under independence the histories are close exactly when every
    coordinate is, so :math:`C_m(\varepsilon) = C_1(\varepsilon)^m`, and
    the standardized gap

    .. math::

       W_m = \sqrt{T}\,
       \frac{C_m(\varepsilon) - C_1(\varepsilon)^m}{\hat\sigma_m(\varepsilon)}
       \;\xrightarrow{d}\; \mathcal{N}(0, 1)

    is the statistic (Brock, Dechert, Scheinkman & LeBaron 1996), with
    :math:`\hat\sigma_m` the closed-form asymptotic standard deviation.
    The record is the statistic at the largest dimension asked for,
    with the lower dimensions as ``companions``. A dependence that
    shows at every dimension is the usual signature of neglected
    conditional heteroskedasticity; one that appears only at high
    dimension is rarer and worth a look. The asymptotic normal
    reference is reliable from about 500 observations and over-rejects
    below that, roughly 8% at nominal 5% on 250 Gaussian observations
    and 15% on 100, which the record says.

    Attributes:
        statistic: The standardized statistic ``W_m``.
        pvalue: Its two-sided standard normal p-value.
        dimension: The embedding dimension ``m``.
        epsilon: The radius, in the units of the series.
        nobs: Observations.
        companions: The lower dimensions ``2 .. m - 1``.

    Note:
        The test is against *any* departure from i.i.d., so on raw data
        it rejects for linear autocorrelation as readily as for a
        threshold, and on residuals it rejects for an ARCH effect as
        readily as for a missed regime. It earns its place after the
        linear and ARCH structure has been removed, as the check that
        nothing is left. The summary lists the dimensions in ascending
        order, this record last.

    See Also:
        * :func:`bds` -- the producer.
        * :class:`LinearityTest` -- the tests that name a specific
          nonlinear alternative.

    References:
        Brock, W. A., Dechert, W. D., Scheinkman, J. A., & LeBaron, B.
        (1996). A test for independence based on the correlation
        dimension. *Econometric Reviews*, 15(3), 197-235.

        Kanzler, L. (1999). Very fast and correctly sized estimation of
        the BDS statistic. Christ Church, University of Oxford, Working
        Paper.

    Example:
        Gaussian noise keeps independence at every dimension; an ARCH
        process rejects it at every dimension, which is the pattern the
        Note describes:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> test = bds(rng.standard_normal(1000))
        >>> test.dimension, test.reject(), [c.dimension for c in test.companions]
        (3, False, [2])
        >>> e = np.zeros(1000)
        >>> for t in range(1, 1000):
        ...     e[t] = np.sqrt(0.2 + 0.7 * e[t - 1] ** 2) * rng.standard_normal()
        >>> test = bds(e, dimension=4)
        >>> [round(c.pvalue, 4) for c in test.companions], round(test.pvalue, 4)
        ([0.0, 0.0], 0.0)
    """

    pvalue: float
    """Two-sided standard normal p-value of ``statistic``."""
    dimension: int
    """The embedding dimension :math:`m`, the length of the histories compared."""
    epsilon: float
    r"""The radius :math:`\varepsilon` in the units of the series.

    ``1.5`` standard deviations of the series by default, the choice
    with the best size in the Brock et al. tables; the same radius on
    every companion, so the dimensions are comparable.
    """
    nobs: int
    """Observations the correlation integrals were computed on."""
    companions: tuple[BDSTest, ...] = field(default=(), repr=False)
    """The records at dimensions ``2 .. m - 1``, ascending, each with the same radius.

    Empty when ``dimension`` is ``2`` and on the companions themselves.
    Kept out of the repr.
    """

    def _row(self) -> tuple[str, ...]:
        """One table row: dimension, statistic, p-value.

        Returns:
            Three strings.

        Example:
            >>> import numpy as np
            >>> bds(np.random.default_rng(0).standard_normal(300))._row()[0]
            '3'
        """
        return (str(self.dimension), f"{self.statistic:.4f}", f"{self.pvalue:.4f}")

    def _summary_table(self) -> SummaryTable:
        """Render as a table over the embedding dimensions.

        Companions first and this record last, so the dimensions read
        upward; the header carries the radius, the sample, and the
        verdict, and the notes carry the reading rule for residuals and,
        below 500 observations, the size warning.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``str()``
            and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> test = bds(np.random.default_rng(0).standard_normal(300), dimension=4)
            >>> table = test._summary_table()
            >>> [row[0] for row in table.rows], len(table.notes)
            (['2', '3', '4'], 2)
        """
        verdict = "reject independence" if self.reject() else "keep independence"
        rows = (*(c._row() for c in self.companions), self._row())
        notes = [
            "Two-sided standard normal reference. On residuals, a rejection at every "
            "dimension usually means neglected conditional heteroskedasticity; run an "
            "ARCH test before reading it as a threshold.",
        ]
        if self.nobs < 500:
            notes.append(
                f"With {self.nobs} observations the asymptotic reference over-rejects; "
                "treat p-values near the level as inconclusive."
            )
        return SummaryTable(
            title="BDS Independence Test",
            metadata=(
                ("Radius", f"{self.epsilon:.4f}"),
                ("Observations", str(self.nobs)),
                ("Verdict at 5%", verdict),
            ),
            columns=("dimension", "statistic", "p-value"),
            rows=rows,
            notes=tuple(notes),
        )

    def __repr__(self) -> str:
        """One-line verdict: statistic, p-value, dimension, observations.

        Returns:
            The repr string.

        Example:
            >>> import numpy as np
            >>> bds(np.random.default_rng(0).standard_normal(1000))
            BDSTest(statistic=-0.8186, pvalue=0.413, dimension=3, nobs=1000)
        """
        return (
            f"BDSTest(statistic={self.statistic:.4f}, pvalue={self.pvalue:.4g}, "
            f"dimension={self.dimension}, nobs={self.nobs})"
        )


def terasvirta(endog: npt.ArrayLike, *, order: int, delay: int = 1) -> LinearityTest:
    r"""Teräsvirta's (1994) LM-type test of linearity against smooth transition.

    A STAR model is a linear AR(:math:`p`) whose coefficients move with
    a transition function :math:`G(y_{t-d})`, and under the null
    :math:`G` is constant. Replacing :math:`G` by its third-order Taylor
    expansion in the transition variable turns the alternative into the
    auxiliary regression

    .. math::

       \hat u_t = \beta_0^\top w_t
       + \sum_{i = 1}^{p} \beta_{1i}\, y_{t-i} y_{t-d}
       + \sum_{i = 1}^{p} \beta_{2i}\, y_{t-i} y_{t-d}^2
       + \sum_{i = 1}^{p} \beta_{3i}\, y_{t-i} y_{t-d}^3 + e_t,

    with :math:`w_t` the linear regressors and :math:`\hat u_t` the AR
    residual, and the null of linearity into the exclusion of the
    :math:`3p` interaction terms, referred to :math:`F(3p, n - 4p - 1)`
    because the chi-squared form over-rejects at the sample sizes the
    test meets. The escalation sequence ``H04``, ``H03``, ``H02`` in
    ``companions`` tests the cubic terms, then the quadratic given the
    cubic, then the linear given both, and is the paper's rule for
    choosing the transition: when the quadratic step rejects most
    strongly the transition is even in :math:`y_{t-d}` and hence
    exponential (ESTAR), otherwise logistic (LSTAR). At order 1 the test
    holds size at 4% on 100 observations and rejects a two-regime SETAR
    or LSTAR with 250 observations essentially always.

    Args:
        endog: The series, ``(T,)``.
        order: Autoregressive order :math:`p` of the linear null, at
            least 1.
        delay: Delay :math:`d` of the transition variable, at least 1.

    Returns:
        The :class:`LinearityTest` named ``"Teräsvirta"`` with
        ``suggestion`` set and the three escalation steps as
        ``companions``.

    Raises:
        SpecificationError: If the order or delay is not positive or the
            series is too short for the auxiliary regression.
        NumericalError: If a value is not finite or the auxiliary
            regressions are degenerate.

    Note:
        The transition variable is a lag of the series itself; a STAR
        with an exogenous transition variable is not covered. The
        ``suggestion`` is always set, since one of the three steps is
        always the strongest; read it only when the test rejects.

    See Also:
        * :func:`tsay` -- the threshold test that assumes no functional
          form for the transition.
        * :func:`hansen_threshold` -- the sup-F test that also locates
          the threshold.
        * :class:`~cultivars.univariate.smooth_transition.LSTAR` and
          :class:`~cultivars.univariate.smooth_transition.ESTAR` -- the
          models the suggestion points to.

    References:
        Teräsvirta, T. (1994). Specification, estimation, and evaluation
        of smooth transition autoregressive models. *Journal of the
        American Statistical Association*, 89(425), 208-218.

        Luukkonen, R., Saikkonen, P., & Teräsvirta, T. (1988). Testing
        linearity against smooth transition autoregressive models.
        *Biometrika*, 75(3), 491-499.

    Example:
        A logistic transition is rejected and the escalation sequence
        reads it as logistic: the quadratic step is not the strongest:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(300)
        >>> for t in range(1, 300):
        ...     g = 1.0 / (1.0 + np.exp(-5.0 * y[t - 1]))
        ...     y[t] = 0.8 * y[t - 1] - 1.4 * y[t - 1] * g + rng.standard_normal()
        >>> verdict = terasvirta(y, order=1)
        >>> verdict.reject(), verdict.suggestion
        (True, 'LSTAR')
        >>> [(c.name, round(c.pvalue, 4)) for c in verdict.companions]
        [('H04', 0.1403), ('H03', 0.0002), ('H02', 0.0)]
    """
    y = validate_endog(endog)
    order = validate_order(order, "order", minimum=1)
    delay = validate_order(delay, "delay", minimum=1)
    _validate_specification(y, order, delay)
    (statistic, pvalue, df1, df2), steps = _terasvirta_lm(y, order, delay)
    nobs = y.shape[0] - max(order, delay)
    null = _NULL.format(order=order)
    companions = tuple(
        LinearityTest(
            name=label,
            statistic=s,
            pvalue=p,
            df=(a, b),
            null=hypothesis,
            alternative="smooth transition",
            order=order,
            delay=delay,
            nobs=nobs,
        )
        for label, hypothesis, (s, p, a, b) in zip(
            ("H04", "H03", "H02"),
            ("cubic terms vanish", "quadratic terms vanish given cubic", "linear terms vanish"),
            steps,
            strict=True,
        )
    )
    suggestion = "ESTAR" if steps[1][1] < min(steps[0][1], steps[2][1]) else "LSTAR"
    return LinearityTest(
        name="Teräsvirta",
        statistic=statistic,
        pvalue=pvalue,
        df=(df1, df2),
        null=null,
        alternative="smooth transition",
        order=order,
        delay=delay,
        nobs=nobs,
        suggestion=suggestion,
        companions=companions,
    )


def tsay(endog: npt.ArrayLike, *, order: int, delay: int = 1) -> LinearityTest:
    r"""Tsay's (1989) arranged-autoregression test of linearity against a threshold.

    The AR(:math:`p`) rows are sorted by the threshold variable
    :math:`y_{t-d}` and the regression fitted recursively down the
    sorted sample from :math:`3\sqrt{T} + p` rows on, each step
    producing a standardized one-step predictive residual
    :math:`\hat e_t`. Under linearity the sorting is irrelevant, the
    predictive residuals are white, and they are orthogonal to the
    regressors; under a threshold the recursion crosses it at some
    point in the sorted sample, after which the residuals shift in mean
    with the regressors. The statistic is the :math:`F` test of the
    predictive residuals on the regressors,

    .. math::

       \hat e_t = \gamma^\top w_t + v_t,
       \qquad
       F = \frac{(SSR_0 - SSR_1) / (p + 1)}{SSR_1 / (n - 2p - 1)},

    with :math:`n` the rows after the recursion's start. It assumes no
    functional form for the transition, only that the threshold
    variable is a lag of the series, and is the natural first look
    before a SETAR; at order 1 it holds 5% size on 100 observations.

    Args:
        endog: The series, ``(T,)``.
        order: Autoregressive order :math:`p` of the linear null, at
            least 1.
        delay: Delay :math:`d` of the threshold variable, at least 1.

    Returns:
        The :class:`LinearityTest` named ``"Tsay"``, with
        ``alternative="threshold"`` and no companions.

    Raises:
        SpecificationError: If the order or delay is not positive or the
            series is too short for the auxiliary regression.
        NumericalError: If a value is not finite, the recursion cannot
            start, or the final regression is degenerate.

    Note:
        The test does not locate the threshold; :func:`hansen_threshold`
        does. Its power is against a shift in the conditional mean
        across the sorted sample, so a threshold in the variance alone
        is not what it catches.

    See Also:
        * :func:`hansen_threshold` -- the sup-F test for the same
          alternative, with a threshold estimate.
        * :func:`terasvirta` -- the test for a smooth rather than
          abrupt transition.
        * :class:`~cultivars.univariate.threshold.SETAR` -- the model a
          rejection licenses.

    References:
        Tsay, R. S. (1989). Testing and modeling threshold autoregressive
        processes. *Journal of the American Statistical Association*,
        84(405), 231-240.

        Tong, H. (1990). *Non-linear Time Series: A Dynamical System
        Approach*. Oxford University Press.

    Example:
        A two-regime SETAR is rejected; white noise is not:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(300)
        >>> for t in range(1, 300):
        ...     slope = 0.8 if y[t - 1] <= 0.0 else -0.5
        ...     y[t] = slope * y[t - 1] + rng.standard_normal()
        >>> tsay(y, order=1).reject()
        True
        >>> tsay(rng.standard_normal(300), order=1).reject()
        False
    """
    y = validate_endog(endog)
    order = validate_order(order, "order", minimum=1)
    delay = validate_order(delay, "delay", minimum=1)
    _validate_specification(y, order, delay)
    statistic, pvalue, df1, df2 = _tsay_arranged(y, order, delay)
    return LinearityTest(
        name="Tsay",
        statistic=statistic,
        pvalue=pvalue,
        df=(df1, df2),
        null=_NULL.format(order=order),
        alternative="threshold",
        order=order,
        delay=delay,
        nobs=y.shape[0] - max(order, delay),
    )


def ramsey_reset(endog: npt.ArrayLike, *, order: int, powers: int = 3) -> LinearityTest:
    r"""Ramsey's RESET on an autoregression: do powers of the fitted value explain the residual?

    The AR(:math:`p`) is fitted, its fitted values :math:`\hat y_t`
    raised to the powers :math:`2, \ldots, k`, and those powers added
    to the regression,

    .. math::

       y_t = \phi^\top w_t + \sum_{j = 2}^{k} \gamma_j\, \hat y_t^{\,j} + e_t,

    with the joint exclusion :math:`\gamma_2 = \cdots = \gamma_k = 0`
    tested by :math:`F(k - 1,\; n - p - k)` with :math:`n` the rows of
    the autoregression. The test names no alternative: any smooth
    misspecification of the conditional mean, a missing regime, a
    nonlinear lag, an omitted regressor correlated with the fitted
    value, raises the powers of the fitted value, so it is the
    general-purpose check to run when the regime tests disagree, and a
    rejection here without one from :func:`tsay` or :func:`terasvirta`
    points away from a regime model.

    Args:
        endog: The series, ``(T,)``.
        order: Autoregressive order :math:`p` of the linear null, at
            least 1.
        powers: Highest power :math:`k` of the fitted value added, at
            least 2; ``2`` tests the square alone.

    Returns:
        The :class:`LinearityTest` named ``"RESET"``, with
        ``alternative="smooth misspecification"``, ``delay=None``, and
        no companions.

    Raises:
        SpecificationError: If the order is not positive, ``powers`` is
            below 2, or the series is too short.
        NumericalError: If a value is not finite or the augmented
            regression is degenerate, which a series whose fitted values
            are nearly constant produces.

    Note:
        High powers of the fitted value are nearly collinear with each
        other, so ``powers`` above 4 buys little and costs conditioning;
        the default of 3 is Ramsey's.

    See Also:
        * :func:`terasvirta` and :func:`tsay` -- the tests that name an
          alternative.
        * :class:`LinearityTest` -- the record.

    References:
        Ramsey, J. B. (1969). Tests for specification errors in classical
        linear least-squares regression analysis. *Journal of the Royal
        Statistical Society B*, 31(2), 350-371.

        Thursby, J. G., & Schmidt, P. (1977). Some properties of tests for
        specification error in a linear regression model. *Journal of the
        American Statistical Association*, 72(359), 635-641.

    Example:
        A linear AR(1) keeps the null; the two-regime series of the
        module examples rejects it:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(300)
        >>> for t in range(1, 300):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal()
        >>> verdict = ramsey_reset(y, order=1)
        >>> verdict.reject(), verdict.df, verdict.delay
        (False, (2, 295), None)
        >>> z = np.zeros(300)
        >>> for t in range(1, 300):
        ...     z[t] = (0.8 if z[t - 1] <= 0.0 else -0.5) * z[t - 1] + rng.standard_normal()
        >>> ramsey_reset(z, order=1).reject()
        True
    """
    y = validate_endog(endog)
    order = validate_order(order, "order", minimum=1)
    powers = validate_order(powers, "powers", minimum=2)
    _validate_specification(y, order, 1)
    statistic, pvalue, df1, df2 = _reset_test(y, order, powers)
    return LinearityTest(
        name="RESET",
        statistic=statistic,
        pvalue=pvalue,
        df=(df1, df2),
        null=_NULL.format(order=order),
        alternative="smooth misspecification",
        order=order,
        delay=None,
        nobs=y.shape[0] - order,
    )


def hansen_threshold(
    endog: npt.ArrayLike,
    *,
    order: int,
    delay: int | None = None,
    trim: float = _DEFAULT_TRIM,
    n_grid: int = _DEFAULT_GRID,
    replications: int = _HANSEN_REPLICATIONS,
    seed: int | np.random.Generator | None = None,
) -> LinearityTest:
    r"""Hansen's (1996) sup-F test of a SETAR against its linear restriction.

    At every candidate threshold :math:`c` on the trimmed quantile grid
    of :math:`y_{t-d}` the two-regime regression

    .. math::

       y_t = \phi_1^\top w_t\, \mathbb{1}\{y_{t-d} \le c\}
           + \phi_2^\top w_t\, \mathbb{1}\{y_{t-d} > c\} + e_t

    is fitted and the :math:`F` statistic for :math:`\phi_1 = \phi_2`
    computed; the sup over the grid is the statistic and its maximizer
    the threshold estimate. Because :math:`c` is unidentified under the
    null the sup has no tabulated law, and the p-value comes from
    ``replications`` draws in which the response is replaced by
    standard normal noise with the regressors and splits held fixed,
    Hansen's fixed-regressor bootstrap, which reproduces the null
    distribution of the sup conditional on the design. Leaving
    ``delay`` as ``None`` searches :math:`d = 1, \ldots, p` jointly,
    and the simulated sup ranges over the same delays so the search is
    paid for in the p-value. Homoskedastic errors are assumed; on 100
    observations at order 1 the size is 5.5%, on 250 it is 3.5%.

    Args:
        endog: The series, ``(T,)``.
        order: Autoregressive order :math:`p` per regime, at least 1.
        delay: Delay :math:`d` of the threshold variable; ``None``
            searches ``1 .. order``.
        trim: Fraction of the sorted threshold variable excluded at
            each end of the grid, in ``(0, 0.5)``; ``0.15`` by default.
        n_grid: Candidate thresholds per delay, at least 1; ``300`` by
            default.
        replications: Simulated sup statistics behind the p-value, at
            least ``200``; ``1000`` by default.
        seed: Seed or generator for the simulation.

    Returns:
        The :class:`LinearityTest` named ``"Hansen sup-F"`` with
        ``threshold``, the selected ``delay``, and simulated
        ``critical_values`` set, and ``df=None``.

    Raises:
        SpecificationError: If the order, delay, trim, grid, or
            replication count is unusable or the series is too short.
        NumericalError: If a value is not finite or no admissible split
            exists.

    Note:
        The threshold estimate is the least-squares one and is
        super-consistent under the alternative, but the record carries
        no interval for it; fit the
        :class:`~cultivars.univariate.threshold.SETAR` for inference on
        the threshold. Under conditional heteroskedasticity the
        fixed-regressor draws should be scaled by the residuals, which
        is not done here, so read a rejection on ARCH residuals with
        that in mind.

    See Also:
        * :func:`tsay` -- the threshold test without a simulation and
          without a threshold estimate.
        * :func:`~cultivars.diagnostics.breaks.sup_wald` -- the same
          sup-F construction over time rather than over a threshold
          variable.
        * :class:`~cultivars.univariate.threshold.SETAR` -- the model a
          rejection licenses.

    References:
        Hansen, B. E. (1996). Inference when a nuisance parameter is not
        identified under the null hypothesis. *Econometrica*, 64(2),
        413-430.

        Hansen, B. E. (1997). Inference in TAR models. *Studies in
        Nonlinear Dynamics & Econometrics*, 2(1), 1-14.

        Chan, K. S. (1993). Consistency and limiting distribution of the
        least squares estimator of a threshold autoregressive model.
        *Annals of Statistics*, 21(1), 520-533.

    Example:
        A two-regime SETAR with threshold zero is rejected and the
        threshold located near zero; pinning the wrong delay loses the
        rejection:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(300)
        >>> for t in range(1, 300):
        ...     slope = 0.8 if y[t - 1] <= 0.0 else -0.5
        ...     y[t] = slope * y[t - 1] + rng.standard_normal()
        >>> verdict = hansen_threshold(y, order=1, replications=500, seed=0)
        >>> verdict.reject(), bool(abs(verdict.threshold) < 0.5), verdict.delay
        (True, True, 1)
        >>> hansen_threshold(y, order=2, delay=2, replications=500, seed=0).reject()
        False
    """
    y = validate_endog(endog)
    order = validate_order(order, "order", minimum=1)
    trim = validate_open_interval(trim, "trim", low=0.0, high=0.5)
    n_grid = validate_order(n_grid, "n_grid", minimum=1)
    if replications < 200:
        raise SpecificationError(f"replications must be at least 200; got {replications}.")
    delays = (
        tuple(range(1, order + 1))
        if delay is None
        else (validate_order(delay, "delay", minimum=1),)
    )
    _validate_specification(y, order, max(delays))
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    statistic, pvalue, threshold, chosen, simulated = _hansen_threshold(
        y, order, delays, trim=trim, n_grid=n_grid, replications=replications, rng=rng
    )
    _p, critical_values = _simulated_critical_values(simulated, statistic)
    return LinearityTest(
        name="Hansen sup-F",
        statistic=statistic,
        pvalue=pvalue,
        df=None,
        null=_NULL.format(order=order),
        alternative="threshold",
        order=order,
        delay=chosen,
        nobs=y.shape[0] - max(order, max(delays)),
        threshold=threshold,
        critical_values={k: critical_values[k] for k in _CRITICAL_LEVELS},
    )


def bds(endog: npt.ArrayLike, *, dimension: int = 3, epsilon: float | None = None) -> BDSTest:
    r"""The BDS test of independence, usually on residuals.

    The correlation integral of the :math:`m`-histories at radius
    :math:`\varepsilon`,

    .. math::

       C_m(\varepsilon) = \binom{T_m}{2}^{-1}
       \sum_{s < t} \mathbb{1}\bigl\{\|y^m_s - y^m_t\|_\infty < \varepsilon\bigr\},

    is compared with the :math:`m`-th power of the one-point integral,
    which is what independence implies; the standardized gap is
    asymptotically standard normal, and the test is run at every
    dimension from 2 to ``dimension`` with the same radius. The radius
    defaults to ``1.5`` standard deviations of the series, the choice
    with the best size in the Brock et al. tables, and the reference is
    trustworthy from about 500 observations. On GARCH residuals with
    500 observations the test rejects about 90% of the time, which is
    its most common use: what the ARCH test says with a form, BDS says
    without one.

    Args:
        endog: The series, typically standardized residuals, at least
            ``10 * dimension + 50`` observations.
        dimension: Largest embedding dimension :math:`m`, at least 2;
            ``2`` and every dimension up to it are reported.
        epsilon: Radius in the units of the series; ``None`` takes
            ``1.5`` standard deviations.

    Returns:
        The :class:`BDSTest` at ``dimension``, lower dimensions as
        ``companions``.

    Raises:
        SpecificationError: If the dimension is below 2, the radius is
            not positive, or the series is too short.
        NumericalError: If a value is not finite, or the radius leaves
            no pairs close or every pair close.

    Note:
        Applied to residuals from an estimated model the statistic's
        limit is unchanged for linear models, by Brock's theorem, but
        not for GARCH-standardized residuals, where it is mildly
        conservative; the 90% power figure is on the raw GARCH series.
        The cost is :math:`O(T^2)` in memory and time through the
        pairwise distances.

    See Also:
        * :class:`BDSTest` -- the record, with the reading rule for
          residuals in its summary.
        * :func:`tsay`, :func:`terasvirta`, :func:`hansen_threshold` --
          the tests that name an alternative.

    References:
        Brock, W. A., Dechert, W. D., Scheinkman, J. A., & LeBaron, B.
        (1996). A test for independence based on the correlation
        dimension. *Econometric Reviews*, 15(3), 197-235.

        Brock, W. A., Hsieh, D. A., & LeBaron, B. (1991). *Nonlinear
        Dynamics, Chaos, and Instability: Statistical Theory and
        Economic Evidence*. MIT Press.

    Example:
        Gaussian noise keeps independence; a radius far below the
        series' scale leaves no close pairs and is refused:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = rng.standard_normal(600)
        >>> bool(bds(y).pvalue > 0.01)
        True
        >>> bds(y, epsilon=1e-9)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.NumericalError: a radius of 1e-09 leaves the correlation integral ...
    """
    y = validate_endog(endog)
    dimension = validate_order(dimension, "dimension", minimum=2)
    if y.shape[0] < 10 * dimension + 50:
        raise SpecificationError(
            f"the BDS test at dimension {dimension} needs at least {10 * dimension + 50} "
            f"observations; got {y.shape[0]}."
        )
    radius = _BDS_RADIUS * float(y.std()) if epsilon is None else float(epsilon)
    if not radius > 0.0:
        raise SpecificationError(f"epsilon must be positive; got {radius}.")
    statistics, pvalues = _bds(y, dimension, radius)
    nobs = y.shape[0]
    companions = tuple(
        BDSTest(statistic=float(s), pvalue=float(p), dimension=m, epsilon=radius, nobs=nobs)
        for m, s, p in zip(range(2, dimension), statistics[:-1], pvalues[:-1], strict=True)
    )
    return BDSTest(
        statistic=float(statistics[-1]),
        pvalue=float(pvalues[-1]),
        dimension=dimension,
        epsilon=radius,
        nobs=nobs,
        companions=companions,
    )
