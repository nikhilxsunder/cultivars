# filepath: /src/cultivars/diagnostics/unit_roots.py
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
r"""Unit-root and stationarity tests: the question every model in the package asks first.

A level VAR, a cointegrating system, and a differenced ARMA are answers
to one question, whether the series carries a stochastic trend, and the
tests here are how that question is put to the data. The difficulty is
that under a unit root the least-squares :math:`t` ratio on the level
does not converge to a normal but to a functional of Brownian motion,

.. math::

   t_{\gamma} \;\xrightarrow{d}\;
   \frac{\int_0^1 W(r)\, dW(r)}{\bigl(\int_0^1 W(r)^2\, dr\bigr)^{1/2}},

whose quantiles depend on the deterministic terms and sit far to the
left of the normal's, and that the alternative of interest, a root just
inside the circle, is where every test in the family has least power.
The tests come in two families. The Dickey-Fuller line puts the unit
root under the null: :func:`adf` with its augmentation chosen on a
common sample, :func:`phillips_perron` with a kernel correction instead
of augmentation, :func:`dfgls` on a GLS-detrended series (Elliott,
Rothenberg & Stock 1996), which recovers most of the power the OLS
detrending loses near the null, and :func:`ng_perron`, whose :math:`M`
statistics keep size under a large negative moving-average root that
inflates the others. :func:`kpss` puts stationarity under the null
instead, so that a series both families fail to reject is one the
sample cannot classify, which the record says rather than hides.
:func:`zivot_andrews` allows one break at an unknown date, because a
broken trend is the classic way a stationary series is mistaken for a
unit root (Perron 1989). :func:`long_run_variance` is the kernel
estimator the corrected tests divide by, exposed on its own.

Two commitments shape the surface. First, the two nulls share one
record. A :class:`UnitRootTest` carries ``lower_tail`` so that
``reject`` reads correctly whichever null the test put up, and a table
of an ADF and a KPSS on the same series reads side by side without the
reader translating tails. Second, a p-value is reported only where one
exists. The Dickey-Fuller law has MacKinnon's response surface and the
KPSS table is dense enough to interpolate, and those tests carry
p-values; DF-GLS, Ng-Perron, and Zivot-Andrews are known through
asymptotic critical values at three levels and carry ``pvalue=None``,
so their ``reject`` reads the table and refuses a level it does not
have, rather than interpolating a number that would look more precise
than it is.

Every ``trend`` argument takes ``"n"`` (nothing), ``"c"`` (constant) or
``"ct"`` (constant and linear trend); KPSS, DF-GLS and Ng-Perron admit
only the last two, and Zivot-Andrews always carries both and takes a
``model`` for the break instead. The trend must include every
deterministic term the alternative would need, since a trending series
tested against a constant alone has no power.

Layout. :class:`UnitRootTest` is the record, a
:class:`~cultivars.diagnostics.hypothesis.TabulatedTest` with the trend,
lags, method, and break fields; the seven functions are the producers.
The numerics live in ``_core``: ``_dickey_fuller_regression`` runs the
augmented regression and ``_resolve_dickey_fuller_lags`` chooses the
augmentation by AIC, BIC, the t-statistic rule, or the modified AIC on
a common sample, with ``_schwert_max_lags`` as the ceiling;
``_mackinnon_pvalue`` and ``_mackinnon_critical_values`` are the
response surface; ``_gls_detrend`` is the Elliott-Rothenberg-Stock
detrending with the local-to-unity constants in ``_GLS_DETREND_C``;
``_phillips_perron``, ``_kpss_statistic``, and ``_ng_perron_statistics``
compute their statistics, ``_long_run_variance`` and
``_newey_west_bandwidth`` the kernel estimate beneath the first two;
``_zivot_andrews`` runs the search over dates; and the KPSS, DF-GLS,
Ng-Perron, and Zivot-Andrews tables are the module's named constants,
read through ``_critical_value_table``.

References:
    Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators
    for autoregressive time series with a unit root. *Journal of the
    American Statistical Association*, 74(366), 427-431.

    Said, S. E., & Dickey, D. A. (1984). Testing for unit roots in
    autoregressive-moving average models of unknown order.
    *Biometrika*, 71(3), 599-607.

    Phillips, P. C. B., & Perron, P. (1988). Testing for a unit root in
    time series regression. *Biometrika*, 75(2), 335-346.

    Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992).
    Testing the null hypothesis of stationarity against the
    alternative of a unit root. *Journal of Econometrics*, 54(1-3),
    159-178.

    Elliott, G., Rothenberg, T. J., & Stock, J. H. (1996). Efficient tests
    for an autoregressive unit root. *Econometrica*, 64(4), 813-836.

    Ng, S., & Perron, P. (2001). Lag length selection and the construction
    of unit root tests with good size and power. *Econometrica*,
    69(6), 1519-1554.

    Perron, P., & Qu, Z. (2007). A simple modification to improve the
    finite sample properties of Ng and Perron's unit root tests.
    *Economics Letters*, 94(1), 12-19.

    Zivot, E., & Andrews, D. W. K. (1992). Further evidence on the great
    crash, the oil-price shock, and the unit-root hypothesis.
    *Journal of Business & Economic Statistics*, 10(3), 251-270.

    Perron, P. (1989). The great crash, the oil price shock, and the unit
    root hypothesis. *Econometrica*, 57(6), 1361-1401.

    MacKinnon, J. G. (1994). Approximate asymptotic distribution functions
    for unit-root and cointegration tests. *Journal of Business &
    Economic Statistics*, 12(2), 167-176.

    MacKinnon, J. G. (1996). Numerical distribution functions for unit
    root and cointegration tests. *Journal of Applied Econometrics*,
    11(6), 601-618.

    Stock, J. H. (1994). Unit roots, structural breaks and trends. In
    *Handbook of Econometrics, Volume 4* (pp. 2739-2841). North-Holland.

Example:
    The two families agree on a random walk and on white noise, from
    opposite tails; a test known only through its table carries no
    p-value and reads the table:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> walk = np.cumsum(rng.standard_normal(300))
    >>> adf(walk).reject(), kpss(walk).reject()
    (False, True)
    >>> noise = rng.standard_normal(300)
    >>> adf(noise).reject(), kpss(noise).reject()
    (True, False)
    >>> dfgls(walk).pvalue, dfgls(walk).reject(alpha=0.05)
    (None, False)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import (
    _DFGLS_CRITICAL,
    _GLS_DETREND_C,
    _KPSS_CRITICAL,
    _NG_PERRON_CRITICAL,
    _ZIVOT_ANDREWS_CRITICAL,
    _critical_value_table,
    _dickey_fuller_regression,
    _gls_detrend,
    _kpss_pvalue,
    _kpss_statistic,
    _long_run_variance,
    _mackinnon_critical_values,
    _mackinnon_pvalue,
    _newey_west_bandwidth,
    _ng_perron_statistics,
    _phillips_perron,
    _resolve_dickey_fuller_lags,
    _schwert_max_lags,
    _zivot_andrews,
    validate_choice,
    validate_endog,
)
from .._internals import _TabulatedTest
from ..exceptions import SpecificationError

__all__ = [
    "UnitRootTest",
    "adf",
    "dfgls",
    "kpss",
    "long_run_variance",
    "ng_perron",
    "phillips_perron",
    "zivot_andrews",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class UnitRootTest(_TabulatedTest):
    r"""Verdict of a unit-root or stationarity test.

    The family is not one hypothesis but two. The Dickey-Fuller line,
    ADF, Phillips-Perron, DF-GLS, Ng-Perron, Zivot-Andrews, puts the
    unit root under the null and rejects in the lower tail: under
    :math:`\rho = 1` in :math:`\Delta y_t = (\rho - 1) y_{t-1} + \cdots`
    the :math:`t` ratio converges not to a normal but to the
    Dickey-Fuller functional

    .. math::

       \frac{\int_0^1 W(r)\, dW(r)}{\bigl(\int_0^1 W(r)^2\, dr\bigr)^{1/2}},

    whose quantiles depend on the deterministic terms and sit well to
    the left of the normal's. KPSS puts stationarity under the null and
    rejects in the upper tail, its statistic a scaled integral of a
    squared Brownian bridge. The record carries which, so that
    :meth:`reject` reads correctly for both and a table of several
    tests can be read side by side: a series the ADF cannot reject a
    unit root for and the KPSS cannot reject stationarity for is one the
    sample does not decide, and that is a finding.

    P-values are exact where the literature supplies a response surface
    (MacKinnon for the Dickey-Fuller law) or a table dense enough to
    interpolate (KPSS, whose p-value is then clipped to the table's
    range); the tests known only through asymptotic critical values at
    three levels report ``pvalue=None`` and :meth:`reject` reads the
    table, refusing a level it does not carry.

    Attributes:
        trend: Deterministic specification the test was run under.
        lags: Augmentation lags or kernel bandwidth, as the test uses.
        method: How ``lags`` was chosen or the kernel used.
        break_index: For a break-allowing test, the first observation of
            the new regime; ``None`` otherwise.

    Note:
        ``lags`` means two different things across the family, and
        ``method`` disambiguates: an augmentation count for ADF, DF-GLS,
        Ng-Perron, and Zivot-Andrews, where ``method`` names the
        selection rule and its ceiling; a kernel bandwidth for
        Phillips-Perron and KPSS, where ``method`` names the kernel.
        Ng-Perron's four statistics travel as ``companions`` of the
        :math:`MZ_t` record, since they are one test on one sample.

    See Also:
        * :func:`adf`, :func:`phillips_perron`, :func:`dfgls`,
          :func:`ng_perron`, :func:`zivot_andrews` -- the unit-root-null
          producers.
        * :func:`kpss` -- the stationarity-null producer.
        * :class:`~cultivars.diagnostics.seasonality.SeasonalUnitRootTest`
          -- the same record at the seasonal frequencies.
        * :class:`~cultivars.diagnostics.hypothesis.TabulatedTest` -- the
          base that reads a statistic against critical values.

    References:
        Dickey, D. A., & Fuller, W. A. (1979). Distribution of the
        estimators for autoregressive time series with a unit root.
        *Journal of the American Statistical Association*, 74(366),
        427-431.

        MacKinnon, J. G. (1996). Numerical distribution functions for
        unit root and cointegration tests. *Journal of Applied
        Econometrics*, 11(6), 601-618.

        Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y.
        (1992). Testing the null hypothesis of stationarity against the
        alternative of a unit root. *Journal of Econometrics*, 54(1-3),
        159-178.

    Example:
        A random walk keeps the ADF null and rejects the KPSS null; white
        noise does the reverse. The two records reject in opposite
        tails and agree on both series:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> walk = np.cumsum(rng.standard_normal(300))
        >>> noise = rng.standard_normal(300)
        >>> adf(walk).reject(), kpss(walk).reject()
        (False, True)
        >>> adf(noise).reject(), kpss(noise).reject()
        (True, False)
        >>> test = adf(walk)
        >>> test.lower_tail, test.trend, test.method, test.break_index
        (True, 'c', 'aic (max 15)', None)

        A test known only through its table has no p-value and reads
        the level it is asked for from the table:

        >>> test = dfgls(walk)
        >>> test.pvalue, test.reject(alpha=0.05)
        (None, False)
    """

    trend: str
    """Deterministic specification: ``"n"``, ``"c"``, or ``"ct"``, as the producer accepted it.

    The seasonal tests append ``"+s"`` when dummies were included; the
    break-allowing tests name the break form the same way the producer
    did.
    """
    lags: int
    """Augmentation lags of the regression, or the kernel bandwidth, whichever the test uses.

    An integer either way; ``method`` says which.
    """
    method: str
    """How ``lags`` was chosen, or which kernel was used.

    ``"aic (max 15)"`` or ``"fixed"`` for an augmentation count;
    ``"bartlett kernel"`` for a bandwidth.
    """
    break_index: int | None = None
    """First observation of the new regime for a break-allowing test; ``None`` otherwise.

    Set by :func:`zivot_andrews` at the date that minimizes the
    statistic, whether or not the null is rejected.
    """

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        """Header lines: null, trend, lags, method, observations, and the break when there is one.

        Returns:
            Label-value pairs in display order.

        Example:
            >>> import numpy as np
            >>> adf(np.random.default_rng(0).standard_normal(200), lags=2)._metadata()[1:4]
            (('Trend', 'c'), ('Lags', '2'), ('Method', 'fixed'))
        """
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
        """The extra pieces of the one-line repr: null, trend, lags.

        Returns:
            Three ``key=value`` strings.

        Example:
            >>> import numpy as np
            >>> kpss(np.random.default_rng(0).standard_normal(200))._repr_fields()
            ("null='stationarity'", "trend='c'", 'lags=4')
        """
        return (f"null={self.null!r}", f"trend={self.trend!r}", f"lags={self.lags}")


def adf(
    endog: npt.ArrayLike,
    *,
    trend: str = "c",
    lags: int | None = None,
    max_lags: int | None = None,
    method: str = "aic",
) -> UnitRootTest:
    r"""Augmented Dickey-Fuller test of a unit root.

    The regression

    .. math::

       \Delta y_t = d_t + \gamma\, y_{t-1} + \sum_{j = 1}^{p} \phi_j\, \Delta y_{t-j} + e_t

    tests :math:`\gamma = 0` by the :math:`t` ratio on :math:`y_{t-1}`,
    read against the Dickey-Fuller law rather than the normal. The
    augmentation :math:`p` absorbs serial correlation in the
    differences, so that under an ARMA short run the limit still holds
    (Said & Dickey 1984). When ``lags`` is not given it is chosen on
    the common sample that the largest candidate leaves, so the
    criteria compare like with like; the test itself is then run on
    the longest sample the chosen augmentation allows. The p-value is
    MacKinnon's (1996) response surface, and the critical values its
    finite-sample ones at the effective sample size.

    Args:
        endog: The series, ``(T,)``.
        trend: ``"n"``, ``"c"`` or ``"ct"``; the law depends on it.
        lags: Augmentation lags, at least 0, or ``None`` to select.
        max_lags: Largest augmentation under selection; default
            Schwert's :math:`\lfloor 12 (T / 100)^{1/4} \rfloor`.
        method: ``"aic"``, ``"bic"``, ``"t-stat"`` (general-to-specific
            at 10%) or ``"maic"`` (Ng-Perron modified AIC).

    Returns:
        The :class:`UnitRootTest` named ``"ADF"`` with MacKinnon's
        response-surface p-value and finite-sample critical values;
        ``lags`` is the augmentation and ``method`` how it was chosen.

    Raises:
        SpecificationError: If the trend, method, or counts are unusable
            or the sample is too short.
        DimensionError: If the series is not one-dimensional.
        NumericalError: If the series is not finite or the design is
            singular.

    Note:
        BIC under-augments against moving-average errors and the test
        then over-rejects, which is why AIC is the default; the
        modified AIC is built for the neighbourhood of the null and
        pays for that with long augmentations on a plainly stationary
        series. The trend must include every deterministic term the
        alternative would need: a trending series tested with
        ``trend="c"`` has no power.

    See Also:
        * :func:`kpss` -- the complementary test with stationarity under
          the null.
        * :func:`phillips_perron` -- the same law with a nonparametric
          correction in place of the augmentation.
        * :func:`dfgls` -- the more powerful GLS-detrended version.
        * :func:`zivot_andrews` -- when the alternative allows a break.

    References:
        Dickey, D. A., & Fuller, W. A. (1979). Distribution of the
        estimators for autoregressive time series with a unit root.
        *Journal of the American Statistical Association*, 74(366),
        427-431.

        Said, S. E., & Dickey, D. A. (1984). Testing for unit roots in
        autoregressive-moving average models of unknown order.
        *Biometrika*, 71(3), 599-607.

        MacKinnon, J. G. (1996). Numerical distribution functions for
        unit root and cointegration tests. *Journal of Applied
        Econometrics*, 11(6), 601-618.

        Schwert, G. W. (1989). Tests for unit roots: A Monte Carlo
        investigation. *Journal of Business & Economic Statistics*,
        7(2), 147-159.

    Example:
        A random walk keeps the null; a stationary AR(1) with a root of
        0.9 rejects it at 300 observations:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> walk = np.cumsum(rng.standard_normal(300))
        >>> test = adf(walk)
        >>> test.reject(), round(test.pvalue, 3), test.trend, test.lags
        (False, 0.841, 'c', 0)
        >>> y = np.zeros(300)
        >>> for t in range(1, 300):
        ...     y[t] = 0.9 * y[t - 1] + rng.standard_normal()
        >>> test = adf(y)
        >>> test.reject(), test.method
        (True, 'aic (max 15)')
    """
    validate_choice(trend, ("n", "c", "ct"), "trend")
    y = validate_endog(endog)
    chosen, label = _resolve_dickey_fuller_lags(y, lags, max_lags, trend, method)
    tau, _, _, _, n_eff = _dickey_fuller_regression(y, chosen, trend)
    return UnitRootTest(
        name="ADF",
        statistic=tau,
        pvalue=_mackinnon_pvalue(tau, trend),
        critical_values=_mackinnon_critical_values(trend, n_eff),
        null="unit root",
        lower_tail=True,
        trend=trend,
        lags=chosen,
        nobs=n_eff,
        method=label,
    )


def phillips_perron(
    endog: npt.ArrayLike,
    *,
    trend: str = "c",
    kernel: str = "bartlett",
    bandwidth: float | None = None,
) -> UnitRootTest:
    r"""Phillips-Perron :math:`Z_t` test of a unit root.

    The un-augmented Dickey-Fuller :math:`t` is corrected with a kernel
    estimate :math:`\hat\lambda^2` of the long-run variance of the
    residuals,

    .. math::

       Z_t = \sqrt{\frac{\hat\sigma^2}{\hat\lambda^2}}\; t_{\gamma}
       - \frac{\hat\lambda^2 - \hat\sigma^2}{2\hat\lambda}
         \Bigl(T^{-2} \sum_t \tilde y_{t-1}^2\Bigr)^{-1/2},

    so serial correlation is handled nonparametrically rather than by
    adding lags. The corrected statistic follows the same Dickey-Fuller
    law, and MacKinnon's p-value applies. The correction is exact in
    the limit but is known to over-reject badly under a large negative
    moving-average root at any practical sample size, which is the
    case the ADF augmentation and the Ng-Perron tests handle better.

    Args:
        endog: The series, ``(T,)``.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        kernel: ``"bartlett"`` or ``"quadratic-spectral"``.
        bandwidth: Kernel bandwidth; ``None`` for Newey-West's rule
            :math:`\lfloor 4 (T / 100)^{2/9} \rfloor` (Bartlett) or
            Andrews' plug-in (quadratic spectral).

    Returns:
        The :class:`UnitRootTest` named ``"Phillips-Perron"``; ``lags``
        reports the bandwidth used and ``method`` the kernel.

    Raises:
        SpecificationError: If the trend or kernel is unusable.
        DimensionError: If the series is not one-dimensional.
        NumericalError: If the series is not finite or the design is
            singular.

    See Also:
        * :func:`adf` -- the parametric correction, more robust to
          moving-average errors.
        * :func:`ng_perron` -- the same idea with an autoregressive
          spectral estimate and GLS detrending, which fixes the size
          problem.
        * :func:`long_run_variance` -- the kernel estimator on its own.

    References:
        Phillips, P. C. B., & Perron, P. (1988). Testing for a unit root
        in time series regression. *Biometrika*, 75(2), 335-346.

        Newey, W. K., & West, K. D. (1994). Automatic lag selection in
        covariance matrix estimation. *Review of Economic Studies*,
        61(4), 631-653.

    Example:
        The same verdicts as ADF on the same series, with the bandwidth
        where the augmentation would be:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> walk = np.cumsum(rng.standard_normal(300))
        >>> test = phillips_perron(walk)
        >>> test.reject(), test.lags, test.method
        (False, 5, 'bartlett kernel')
        >>> phillips_perron(walk, bandwidth=3).lags
        3
    """
    validate_choice(trend, ("n", "c", "ct"), "trend")
    y = validate_endog(endog)
    z_t, _, n_eff = _phillips_perron(y, trend, kernel=kernel, bandwidth=bandwidth)
    width = _newey_west_bandwidth(n_eff) if bandwidth is None else int(np.floor(bandwidth))
    return UnitRootTest(
        name="Phillips-Perron",
        statistic=z_t,
        pvalue=_mackinnon_pvalue(z_t, trend),
        critical_values=_mackinnon_critical_values(trend, n_eff),
        null="unit root",
        lower_tail=True,
        trend=trend,
        lags=width,
        nobs=n_eff,
        method=f"{kernel} kernel",
    )


def kpss(
    endog: npt.ArrayLike,
    *,
    trend: str = "c",
    kernel: str = "bartlett",
    bandwidth: float | None = None,
) -> UnitRootTest:
    r"""KPSS test with stationarity under the null.

    The series is regressed on the deterministic terms, and the
    residuals' partial sums :math:`S_t = \sum_{i \le t} \hat e_i` are
    scaled by a kernel estimate of their long-run variance,

    .. math::

       \eta = \frac{T^{-2} \sum_{t = 1}^{T} S_t^2}{\hat\lambda^2},

    which under stationarity converges to the integral of a squared
    Brownian bridge (a second-level bridge for ``"ct"``) and under a
    unit root diverges, so rejection lies in the upper tail. This is
    the Lagrange multiplier test of a zero variance for a random-walk
    component, and it is the complement of the Dickey-Fuller family:
    the two nulls are the two sides of the question, and a series that
    keeps both is one the sample cannot decide. The p-value is
    interpolated in the authors' table and clipped to ``[0.01, 0.10]``
    outside it, so ``0.01`` reads as "at most 1%" and ``0.10`` as "at
    least 10%".

    Args:
        endog: The series, ``(T,)``.
        trend: ``"c"`` for level stationarity or ``"ct"`` for trend
            stationarity.
        kernel: ``"bartlett"`` or ``"quadratic-spectral"``.
        bandwidth: Kernel bandwidth; ``None`` for the automatic rule.

    Returns:
        The :class:`UnitRootTest` named ``"KPSS"`` with
        ``null="stationarity"`` and ``lower_tail=False``; ``lags``
        reports the bandwidth.

    Raises:
        SpecificationError: If the trend or kernel is unusable.
        DimensionError: If the series is not one-dimensional.
        NumericalError: If the series is not finite.

    Note:
        The long-run variance is estimated under the null, so a
        bandwidth too short for the residual autocorrelation
        over-rejects and one too long loses power; the test is known to
        over-reject on a highly persistent stationary series at any
        bandwidth, which is the mirror of the ADF's low power there.
        The p-value's clipping means two strong rejections cannot be
        ranked by it; rank by the statistic against its critical values.

    See Also:
        * :func:`adf` -- the complementary test with the unit root under
          the null.
        * :func:`~cultivars.diagnostics.seasonality.canova_hansen` --
          the same construction at the seasonal frequencies.
        * :func:`long_run_variance` -- the kernel estimator on its own.

    References:
        Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y.
        (1992). Testing the null hypothesis of stationarity against the
        alternative of a unit root. *Journal of Econometrics*, 54(1-3),
        159-178.

        Nyblom, J. (1989). Testing for the constancy of parameters over
        time. *Journal of the American Statistical Association*,
        84(405), 223-230.

    Example:
        A random walk rejects level stationarity at the table's floor;
        white noise keeps it at its ceiling:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> walk = np.cumsum(rng.standard_normal(300))
        >>> test = kpss(walk)
        >>> test.reject(), test.pvalue, test.critical_values["5%"]
        (True, 0.01, 0.463)
        >>> kpss(rng.standard_normal(300)).pvalue
        0.1
    """
    validate_choice(trend, ("c", "ct"), "trend")
    y = validate_endog(endog)
    statistic, _ = _kpss_statistic(y, trend, kernel=kernel, bandwidth=bandwidth)
    table = dict(_KPSS_CRITICAL[trend])
    width = _newey_west_bandwidth(y.shape[0]) if bandwidth is None else int(np.floor(bandwidth))
    return UnitRootTest(
        name="KPSS",
        statistic=statistic,
        pvalue=_kpss_pvalue(statistic, trend),
        critical_values=_critical_value_table((table[0.01], table[0.05], table[0.10])),
        null="stationarity",
        lower_tail=False,
        trend=trend,
        lags=width,
        nobs=int(y.shape[0]),
        method=f"{kernel} kernel",
    )


def dfgls(
    endog: npt.ArrayLike,
    *,
    trend: str = "c",
    lags: int | None = None,
    max_lags: int | None = None,
    method: str = "maic",
) -> UnitRootTest:
    r"""Elliott-Rothenberg-Stock DF-GLS test of a unit root.

    The deterministic terms are removed by generalized least squares
    under the local alternative :math:`\bar\alpha = 1 + \bar c / T`,
    with :math:`\bar c = -7` for a constant and :math:`-13.5` for a
    trend: the quasi-differenced series :math:`y_t - \bar\alpha y_{t-1}`
    is regressed on the quasi-differenced deterministic terms, and the
    detrended series is :math:`y_t - \hat\beta^\top d_t`. The
    Dickey-Fuller regression is then run on the detrended series
    without deterministic terms. The result is close to the
    point-optimal test and far more powerful than ADF near the null,
    where the power of the Dickey-Fuller family collapses. The
    augmentation is chosen on the OLS-detrended series, as Perron and
    Qu (2007) recommend: on GLS-detrended data the modified AIC
    over-selects badly whenever the series is far from the null. Even
    so, on a series close to white noise the modified AIC still selects
    long augmentations and the statistic loses power; ``method="aic"``
    or a fixed ``lags`` is the remedy there. Critical values are the
    authors' asymptotic ones; no response surface exists, so ``pvalue``
    is ``None``.

    Args:
        endog: The series, ``(T,)``.
        trend: ``"c"`` or ``"ct"``.
        lags: Augmentation lags, at least 0, or ``None`` to select.
        max_lags: Largest augmentation under selection; default
            Schwert's rule.
        method: Selection rule, as :func:`adf`; the modified AIC is the
            authors' companion recommendation (Ng & Perron 2001).

    Returns:
        The :class:`UnitRootTest` named ``"DF-GLS"`` with
        ``pvalue=None`` and the asymptotic critical values.

    Raises:
        SpecificationError: If the trend, method, or counts are unusable.
        DimensionError: If the series is not one-dimensional.
        NumericalError: If the series is not finite or the design is
            singular.

    Note:
        With ``pvalue=None``, :meth:`~UnitRootTest.reject` accepts only
        the tabulated levels 1%, 5%, and 10%. The asymptotic critical
        values are slightly liberal in short samples; at 100
        observations the 5% size is nearer 6%.

    See Also:
        * :func:`ng_perron` -- the same detrending with the
          :math:`M` statistics, better sized under moving-average
          errors.
        * :func:`adf` -- the OLS-detrended original, with a p-value.

    References:
        Elliott, G., Rothenberg, T. J., & Stock, J. H. (1996). Efficient
        tests for an autoregressive unit root. *Econometrica*, 64(4),
        813-836.

        Ng, S., & Perron, P. (2001). Lag length selection and the
        construction of unit root tests with good size and power.
        *Econometrica*, 69(6), 1519-1554.

        Perron, P., & Qu, Z. (2007). A simple modification to improve the
        finite sample properties of Ng and Perron's unit root tests.
        *Economics Letters*, 94(1), 12-19.

    Example:
        A near-unit-root AR(1) that ADF rejects only narrowly is rejected
        comfortably here; the modified AIC's long augmentation on white
        noise, and the remedy:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(300)
        >>> for t in range(1, 300):
        ...     y[t] = 0.9 * y[t - 1] + rng.standard_normal()
        >>> test = dfgls(y)
        >>> test.pvalue, test.reject(), test.critical_values["5%"]
        (None, True, -1.95)
        >>> noise = rng.standard_normal(300)
        >>> dfgls(noise).lags, dfgls(noise, method="aic").lags
        (9, 0)
    """
    validate_choice(trend, ("c", "ct"), "trend")
    y = validate_endog(endog)
    detrended = _gls_detrend(y, trend, _GLS_DETREND_C[trend])
    chosen, label = _resolve_dickey_fuller_lags(y, lags, max_lags, trend, method)
    tau, _, _, _, n_eff = _dickey_fuller_regression(detrended, chosen, "n")
    return UnitRootTest(
        name="DF-GLS",
        statistic=tau,
        pvalue=None,
        critical_values=_critical_value_table(_DFGLS_CRITICAL[trend]),
        null="unit root",
        lower_tail=True,
        trend=trend,
        lags=chosen,
        nobs=n_eff,
        method=label,
    )


def ng_perron(
    endog: npt.ArrayLike,
    *,
    trend: str = "c",
    lags: int | None = None,
    max_lags: int | None = None,
    method: str = "maic",
) -> UnitRootTest:
    r"""Ng-Perron :math:`M` tests of a unit root on a GLS-detrended series.

    With :math:`\tilde y_t` the GLS-detrended series and :math:`s^2_{AR}`
    the autoregressive spectral density estimate at frequency zero,
    the four statistics are

    .. math::

       MZ_\alpha = \frac{T^{-1} \tilde y_T^2 - s^2_{AR}}
                        {2 T^{-2} \sum_{t} \tilde y_{t-1}^2},
       \qquad
       MSB = \Bigl(\frac{T^{-2} \sum_t \tilde y_{t-1}^2}{s^2_{AR}}\Bigr)^{1/2},
       \qquad
       MZ_t = MZ_\alpha \times MSB,

    and :math:`MP_T` the modified point-optimal statistic. :math:`MZ_t`
    is the primary record; the other three travel as ``companions``.
    The autoregressive spectral estimate replaces the kernel estimate
    the Phillips-Perron test uses, and with the modified AIC choosing
    the augmentation the tests keep their size under a large negative
    moving-average root, where ADF and Phillips-Perron over-reject
    badly. The augmentation is chosen on the OLS-detrended series
    (Perron & Qu 2007) and the statistics computed on the GLS-detrended
    one. Critical values are asymptotic; ``pvalue`` is ``None``.

    The modified AIC is built for the neighbourhood of the null. Far
    from it, on a series close to white noise, its penalty is dominated
    by the level coefficient and it selects long augmentations that
    cost the :math:`M` statistics most of their power; ``method="aic"``
    or a fixed ``lags`` is the remedy when the series is plainly
    stationary, at the price of size under a negative moving-average
    root.

    Args:
        endog: The series, ``(T,)``.
        trend: ``"c"`` or ``"ct"``.
        lags: Augmentation for the spectral estimate, at least 0, or
            ``None`` to select.
        max_lags: Largest augmentation under selection; default
            Schwert's rule.
        method: Selection rule, as :func:`adf`; default the modified
            AIC.

    Returns:
        The :math:`MZ_t` :class:`UnitRootTest` named
        ``"Ng-Perron MZ_t"`` with :math:`MZ_\alpha`, :math:`MSB`, and
        :math:`MP_T` as ``companions``; every record has ``pvalue=None``
        and its own asymptotic critical values, all four rejecting in
        the lower tail.

    Raises:
        SpecificationError: If the trend, method, or counts are unusable.
        DimensionError: If the series is not one-dimensional.
        NumericalError: If the series is not finite or the design is
            singular.

    Note:
        :math:`MSB` and :math:`MP_T` are positive statistics that reject
        when *small*, which the lower-tail flag carries; their critical
        values are read in the same direction as the negative ones.

    See Also:
        * :func:`dfgls` -- the :math:`t` statistic on the same
          detrended series.
        * :func:`phillips_perron` -- the kernel-corrected original the
          :math:`M` tests improve on.

    References:
        Ng, S., & Perron, P. (2001). Lag length selection and the
        construction of unit root tests with good size and power.
        *Econometrica*, 69(6), 1519-1554.

        Perron, P., & Ng, S. (1996). Useful modifications to some unit
        root tests with dependent errors and their local asymptotic
        properties. *Review of Economic Studies*, 63(3), 435-463.

        Perron, P., & Qu, Z. (2007). A simple modification to improve the
        finite sample properties of Ng and Perron's unit root tests.
        *Economics Letters*, 94(1), 12-19.

    Example:
        The four statistics on a stationary AR(1), all rejecting:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(300)
        >>> for t in range(1, 300):
        ...     y[t] = 0.9 * y[t - 1] + rng.standard_normal()
        >>> test = ng_perron(y)
        >>> test.name, test.reject(), test.lags
        ('Ng-Perron MZ_t', True, 0)
        >>> [(c.name, c.reject()) for c in test.companions]
        [('MZ_a', True), ('MSB', True), ('MP_T', True)]
    """
    validate_choice(trend, ("c", "ct"), "trend")
    y = validate_endog(endog)
    detrended = _gls_detrend(y, trend, _GLS_DETREND_C[trend])
    chosen, label = _resolve_dickey_fuller_lags(y, lags, max_lags, trend, method)
    stats = _ng_perron_statistics(detrended, chosen, trend)
    table = _NG_PERRON_CRITICAL[trend]
    n_eff = int(y.shape[0] - 1 - chosen)

    def record(key: str, display: str, companions: tuple[UnitRootTest, ...]) -> UnitRootTest:
        """Build one of the four records from the shared statistics and table.

        Args:
            key: The statistic's key in the table, ``"MZt"``, ``"MZa"``,
                ``"MSB"``, or ``"MPT"``.
            display: The record's name.
            companions: The other records, for the primary one only.

        Returns:
            The :class:`UnitRootTest` with the shared trend, lags,
            sample, and method.
        """
        return UnitRootTest(
            name=display,
            statistic=stats[key],
            pvalue=None,
            critical_values=_critical_value_table(table[key]),
            null="unit root",
            lower_tail=True,
            trend=trend,
            lags=chosen,
            nobs=n_eff,
            method=label,
            companions=companions,
        )

    companions = (record("MZa", "MZ_a", ()), record("MSB", "MSB", ()), record("MPT", "MP_T", ()))
    return record("MZt", "Ng-Perron MZ_t", companions)


def zivot_andrews(
    endog: npt.ArrayLike,
    *,
    model: str = "c",
    lags: int | None = None,
    max_lags: int | None = None,
    trimming: float = 0.15,
) -> UnitRootTest:
    r"""Zivot-Andrews test of a unit root against a one-time break at an unknown date.

    Perron (1989) showed that a stationary series with a broken trend
    looks like a unit root to the Dickey-Fuller test, and Zivot and
    Andrews made the break date endogenous. For every candidate date
    :math:`T_b` inside the trimmed window the regression

    .. math::

       \Delta y_t = \mu + \beta t + \theta\, DU_t(T_b) + \gamma\, DT_t(T_b)
       + \alpha\, y_{t-1} + \sum_{j = 1}^{p} \phi_j\, \Delta y_{t-j} + e_t

    is run with the break dummies of the chosen model, :math:`DU` a
    level shift and :math:`DT` a slope change from :math:`T_b` on; the
    statistic is the smallest :math:`t` on :math:`y_{t-1}` over the
    window, and the date that produces it is the estimated break. The
    null is a unit root *without* a break, so a rejection says
    "stationary around a broken trend", not "broken unit root". Critical
    values are the authors' asymptotic ones for the minimized statistic,
    well to the left of the plain Dickey-Fuller values because the
    minimum over dates is taken; ``pvalue`` is ``None``.

    Args:
        endog: The series, ``(T,)``.
        model: ``"c"`` for a break in the intercept, ``"t"`` in the trend
            slope, ``"ct"`` in both.
        lags: Fixed augmentation, at least 0, or ``None`` to select once
            by the t-statistic rule on the no-break regression.
        max_lags: Largest augmentation under selection; default
            Schwert's rule.
        trimming: Fraction of the sample excluded at each end, in
            ``(0, 0.5)``.

    Returns:
        The :class:`UnitRootTest` named ``"Zivot-Andrews"`` with
        ``break_index`` set and ``trend="ct"``, since the regression
        always carries a constant and a trend.

    Raises:
        SpecificationError: If the model, trimming, or counts are unusable.
        DimensionError: If the series is not one-dimensional.
        NumericalError: If the series is not finite or every candidate
            is singular.

    Note:
        ``break_index`` is the arg-min of the statistic and is reported
        whether or not the null is rejected; under the null it is not an
        estimate of anything. The augmentation is selected once on the
        no-break regression and held fixed across dates, which is the
        authors' procedure and keeps the search over dates comparable.
        The test has one break; for several, or for a break under the
        null as well, see the structural-break module.

    See Also:
        * :func:`adf` -- the test without a break, which this one nests.
        * :func:`~cultivars.diagnostics.breaks.sup_wald` -- the break
          test on a stationary regression.
        * :func:`~cultivars.diagnostics.breaks.bai_perron` -- several
          breaks.

    References:
        Zivot, E., & Andrews, D. W. K. (1992). Further evidence on the
        great crash, the oil-price shock, and the unit-root hypothesis.
        *Journal of Business & Economic Statistics*, 10(3), 251-270.

        Perron, P. (1989). The great crash, the oil price shock, and the
        unit root hypothesis. *Econometrica*, 57(6), 1361-1401.

    Example:
        A trend-stationary series with a level shift at 150 looks like a
        unit root to ADF and is rejected here, with the break dated:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> t = np.arange(300)
        >>> y = 0.05 * t + 5.0 * (t >= 150) + rng.standard_normal(300)
        >>> test = zivot_andrews(y)
        >>> test.reject(), test.break_index, test.trend
        (True, 150, 'ct')
        >>> adf(y, trend="ct").reject()
        False
    """
    validate_choice(model, ("c", "t", "ct"), "model")
    if not 0.0 < trimming < 0.5:
        raise SpecificationError(f"trimming must lie in (0, 0.5); got {trimming}.")
    y = validate_endog(endog)
    ceiling = _schwert_max_lags(y.shape[0]) if max_lags is None else int(max_lags)
    statistic, break_index, used = _zivot_andrews(
        y, model=model, lags=lags, max_lags=ceiling, trimming=trimming
    )
    return UnitRootTest(
        name="Zivot-Andrews",
        statistic=statistic,
        pvalue=None,
        critical_values=_critical_value_table(_ZIVOT_ANDREWS_CRITICAL[model]),
        null="unit root",
        lower_tail=True,
        trend="ct",
        lags=used,
        nobs=int(y.shape[0] - 1 - used),
        method="fixed" if lags is not None else f"t-stat (max {ceiling})",
        break_index=break_index,
    )


def long_run_variance(
    endog: npt.ArrayLike, *, kernel: str = "bartlett", bandwidth: float | None = None
) -> float:
    r"""Kernel estimate of a series' long-run variance about its mean.

    The long-run variance :math:`\lambda^2 = \sum_{j = -\infty}^{\infty}
    \gamma_j = 2\pi f(0)` is the variance of the normalized partial sum
    in the limit, and the quantity every kernel-corrected test in this
    module divides by. The estimator is

    .. math::

       \hat\lambda^2 = \hat\gamma_0 + 2 \sum_{j = 1}^{T - 1}
       k\!\left(\frac{j}{b}\right) \hat\gamma_j,

    with :math:`k` the Bartlett or quadratic-spectral kernel and
    :math:`b` the bandwidth. The Bartlett kernel is the Newey-West
    estimator, always positive and with bandwidth
    :math:`\lfloor 4 (T / 100)^{2/9} \rfloor` by default; the quadratic
    spectral kernel is Andrews' (1991) optimal choice among positive
    definite kernels, with his AR(1) plug-in bandwidth by default, and
    converges faster at the cost of weight on distant lags.

    Args:
        endog: The series, ``(T,)``.
        kernel: ``"bartlett"`` or ``"quadratic-spectral"``.
        bandwidth: Kernel bandwidth; ``None`` for the automatic rule of
            the chosen kernel.

    Returns:
        The estimate, a positive float.

    Raises:
        SpecificationError: If the kernel or bandwidth is unusable.
        DimensionError: If the series is not one-dimensional.
        NumericalError: If the series is not finite.

    See Also:
        * :func:`kpss` and :func:`phillips_perron` -- the tests that
          divide by this estimate.
        * :func:`~cultivars.diagnostics.seasonality.canova_hansen` --
          its multivariate form for the seasonal scores.

    References:
        Newey, W. K., & West, K. D. (1987). A simple, positive
        semi-definite, heteroskedasticity and autocorrelation consistent
        covariance matrix. *Econometrica*, 55(3), 703-708.

        Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation
        consistent covariance matrix estimation. *Econometrica*, 59(3),
        817-858.

    Example:
        White noise has long-run variance one; an AR(1) with root 0.9
        has :math:`1 / (1 - 0.9)^2 = 100`, which the kernel estimate
        approaches from below because its bandwidth is short relative
        to the persistence:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> e = rng.standard_normal(2000)
        >>> bool(0.8 < long_run_variance(e) < 1.2)
        True
        >>> y = np.zeros(2000)
        >>> for t in range(1, 2000):
        ...     y[t] = 0.9 * y[t - 1] + rng.standard_normal()
        >>> bool(long_run_variance(y) < long_run_variance(y, bandwidth=50) < 100)
        True
    """
    y = validate_endog(endog)
    return _long_run_variance(y - y.mean(), kernel=kernel, bandwidth=bandwidth)
