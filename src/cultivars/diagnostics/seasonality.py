# filepath: /src/cultivars/diagnostics/seasonality.py
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
r"""Seasonal unit roots: is the seasonal pattern deterministic, or does it wander?

Seasonal differencing :math:`1 - L^s` removes :math:`s` unit roots at
once, one at every frequency :math:`\omega_k = 2\pi k / s`,

.. math::

   1 - L^s = (1 - L)(1 + L) \prod_{k = 1}^{s/2 - 1}
   \bigl(1 - 2\cos\omega_k\, L + L^2\bigr),

and a series rarely needs all of them: a stable seasonal pattern is
handled by dummies, a unit root at the zero frequency by
:math:`1 - L`, and only a pattern whose summers slowly turn into winters
by the seasonal difference. Over-differencing a stable pattern plants a
moving-average unit root the model then has to undo, so which factors
are needed is a question to test, not assume. HEGY (1990) tests each
root separately in one regression, with the seasonal random walk as
the null, so the augmented Dickey-Fuller reading carries over frequency
by frequency: a :math:`t` that cannot reject at the zero frequency and
:math:`F` statistics that reject at every seasonal one say "difference
once, use dummies". Canova and Hansen (1995) turn the null around, as
KPSS does for the zero frequency: stationary seasonality against a
unit root at the frequency tested, so that a series HEGY cannot reject
a seasonal root for and Canova-Hansen cannot reject stationarity for
is one the sample does not decide, and that is a finding.

Two commitments shape the surface. First, both tests are read against
simulated laws rather than the published tables. HEGY's finite-sample
null is drawn at the sample size and deterministic set in hand for any
even period, so a monthly test with a trend and dummies is not read
against a quarterly table, and the Canova-Hansen limit is the
generalized von Mises law, drawn as the integral of a squared Brownian
bridge of the tested dimension. Second, every frequency is a row and
every row is a full record: the headline record carries the others as
``companions``, each with its own tail, p-value, and critical values,
so a single frequency can be read alone, the joint statistic can be
read as the headline, and a mixed table in which :math:`t` rows reject
low and :math:`F` rows reject high still reads correctly through
``reject``.

Layout. :class:`SeasonalUnitRootTest` is the record, a
:class:`~cultivars.diagnostics.unit_roots.UnitRootTest` with the family
and period added and the title and reading notes a mixed-tail table
needs; :func:`hegy` and :func:`canova_hansen` are the producers. The
numerics live in ``_core``: ``_validate_seasonal`` checks the period
and the sample, ``_hegy_statistics`` builds the filtered regressors and
the regression, ``_select_hegy_lags`` chooses the augmentation on a
common sample, ``_hegy_null_draws`` simulates the seasonal random walk
through the same regression, ``_canova_hansen`` computes the LM
statistics and their dimensions, ``_von_mises_draws`` simulates the
bridge integrals, ``_frequency_labels`` names the frequencies, and
``_simulated_critical_values`` turns either set of draws into a
p-value and critical values in the row's tail; the Schwert and
Newey-West bandwidth rules and the replication default are the
module's named constants.

References:
    Hylleberg, S., Engle, R. F., Granger, C. W. J., & Yoo, B. S. (1990).
    Seasonal integration and cointegration. *Journal of
    Econometrics*, 44(1-2), 215-238.

    Beaulieu, J. J., & Miron, J. A. (1993). Seasonal unit roots in
    aggregate U.S. data. *Journal of Econometrics*, 55(1-2), 305-328.

    Canova, F., & Hansen, B. E. (1995). Are seasonal patterns constant
    over time? A test for seasonal stability. *Journal of Business &
    Economic Statistics*, 13(3), 237-252.

    Ghysels, E., & Osborn, D. R. (2001). *The Econometric Analysis of
    Seasonal Time Series*. Cambridge University Press.

Example:
    A random walk with a fixed quarterly pattern: HEGY keeps the root at
    frequency zero and rejects every seasonal one, Canova-Hansen keeps
    stationary seasonality, and together they say "difference once and
    model the season with dummies":

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = np.cumsum(rng.standard_normal(200)) + np.tile([1.0, -0.5, 2.0, -2.5], 50)
    >>> root = hegy(y, period=4, replications=500, seed=0)
    >>> root.reject(), [c.reject() for c in root.companions]
    (False, [True, True, True, True])
    >>> canova_hansen(y, period=4, seed=0).reject()
    False
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from .._core import (
    _HEGY_REPLICATIONS,
    _canova_hansen,
    _frequency_labels,
    _hegy_null_draws,
    _hegy_statistics,
    _newey_west_bandwidth,
    _schwert_max_lags,
    _select_hegy_lags,
    _simulated_critical_values,
    _validate_seasonal,
    _von_mises_draws,
    validate_choice,
    validate_order,
)
from ..exceptions import SpecificationError
from .unit_roots import UnitRootTest

__all__ = ["SeasonalUnitRootTest", "canova_hansen", "hegy"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SeasonalUnitRootTest(UnitRootTest):
    r"""Verdict of a seasonal unit-root or seasonal-stability test, one frequency per row.

    Seasonal differencing :math:`1 - L^s` factors over the :math:`s`
    roots of unity,

    .. math::

       1 - L^s = (1 - L)(1 + L) \prod_{k = 1}^{s/2 - 1}
       \bigl(1 - 2\cos(2\pi k / s)\, L + L^2\bigr),

    one factor per frequency :math:`\omega_k = 2\pi k / s`, and a
    seasonal test is a question asked at each factor. This record is a
    :class:`UnitRootTest` whose ``companions`` are the other frequencies
    of the same regression, and whose rows may reject in different
    tails: HEGY's :math:`t` statistics at the two real roots,
    :math:`\omega = 0` and :math:`\omega = \pi`, reject in the lower
    tail, its :math:`F` statistics at the complex pairs and the joint
    restrictions in the upper; every Canova-Hansen row, whose null is
    stationarity, in the upper. The headline record is HEGY's
    :math:`t` at frequency zero, or Canova-Hansen's joint statistic,
    and ``lower_tail`` is set per row so :meth:`reject` reads each
    correctly. The record adds nothing to the fields but the family
    and the period; what it adds is the title and the reading notes a
    mixed-tail table needs.

    Attributes:
        family: ``"HEGY"`` or ``"Canova-Hansen"``.
        period: Seasonal period ``s``.

    Note:
        The two families answer complementary questions and are read
        together, as ADF and KPSS are at the zero frequency: a
        frequency HEGY cannot reject a root at and Canova-Hansen cannot
        reject stationarity at is one the sample does not decide. The
        Canova-Hansen ``lags`` field is the kernel bandwidth, not an
        augmentation lag count, and its ``method`` says so.

    See Also:
        * :func:`hegy` and :func:`canova_hansen` -- the producers.
        * :class:`UnitRootTest` -- the base, with the zero-frequency
          tests.
        * :func:`~cultivars.diagnostics.unit_roots.kpss` -- the
          zero-frequency counterpart of Canova-Hansen.

    References:
        Hylleberg, S., Engle, R. F., Granger, C. W. J., & Yoo, B. S.
        (1990). Seasonal integration and cointegration. *Journal of
        Econometrics*, 44(1-2), 215-238.

        Canova, F., & Hansen, B. E. (1995). Are seasonal patterns
        constant over time? A test for seasonal stability. *Journal of
        Business & Economic Statistics*, 13(3), 237-252.

    Example:
        A random walk with a fixed quarterly pattern: HEGY keeps the
        root at frequency zero and rejects every seasonal one, which
        reads "difference once and use dummies", and the rows reject in
        different tails:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> season = np.tile([1.0, -0.5, 2.0, -2.5], 50)
        >>> y = np.cumsum(rng.standard_normal(200)) + season
        >>> test = hegy(y, period=4, seed=0)
        >>> test.family, test.period, test.lower_tail, test.reject()
        ('HEGY', 4, True, False)
        >>> [(c.lower_tail, c.reject()) for c in test.companions]
        [(True, True), (False, True), (False, True), (False, True)]

        Canova-Hansen on the same series keeps stationary seasonality
        at every frequency:

        >>> test = canova_hansen(y, period=4, seed=0)
        >>> test.family, test.lower_tail, test.reject()
        ('Canova-Hansen', False, False)
    """

    family: str
    """Which test: ``"HEGY"``, unit root under the null, or ``"Canova-Hansen"``, stationarity."""

    period: int
    """Seasonal period :math:`s`; even, since the Nyquist root :math:`\\omega = \\pi` is tested."""

    def _title(self) -> str:
        """The summary title, by family.

        Returns:
            ``"HEGY Seasonal Unit Root Test"`` or
            ``"Canova-Hansen Seasonal Stability Test"``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal(120)
            >>> canova_hansen(y, period=4, seed=0)._title()
            'Canova-Hansen Seasonal Stability Test'
        """
        return (
            "HEGY Seasonal Unit Root Test"
            if self.family == "HEGY"
            else "Canova-Hansen Seasonal Stability Test"
        )

    def _metadata(self) -> tuple[tuple[str, str], ...]:
        """The base header with the period in front.

        Returns:
            ``Period`` followed by the :class:`UnitRootTest` lines: null,
            trend, lags, method, observations.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal(120)
            >>> hegy(y, period=4, seed=0)._metadata()[:2]
            (('Period', '4'), ('Null', 'unit root at frequency 0'))
        """
        return (("Period", str(self.period)), *UnitRootTest._metadata(self))

    def _notes(self) -> tuple[str, ...]:
        """Closing lines: which tail each row rejects in, and how to read the table.

        Replaces the base's single tail note, which cannot describe a
        table whose rows reject in both tails.

        Returns:
            Two lines for either family.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal(120)
            >>> hegy(y, period=4, seed=0)._notes()[0][:42]
            't statistics reject in the lower tail, F s'
        """
        if self.family == "HEGY":
            return (
                "t statistics reject in the lower tail, F statistics in the upper; every row "
                "is read against its own simulated finite-sample law.",
                "Read: keep at frequency 0 and reject at every seasonal frequency means "
                "difference once and model the season with dummies; keep at a seasonal "
                "frequency means that root needs its factor of the seasonal difference.",
            )
        return (
            "Rejection lies in the upper tail; the joint row is the headline and the "
            "per-frequency rows locate the instability.",
            "The null is stationary seasonality, so this test and HEGY are read together "
            "as KPSS and ADF are.",
        )


def hegy(
    endog: npt.ArrayLike,
    *,
    period: int,
    trend: str = "c",
    seasonal: bool = True,
    lags: int | None = None,
    max_lags: int | None = None,
    method: str = "aic",
    replications: int = _HEGY_REPLICATIONS,
    seed: int | np.random.Generator | None = None,
) -> SeasonalUnitRootTest:
    r"""HEGY test of a unit root at each seasonal frequency, and at the zero frequency.

    Hylleberg, Engle, Granger and Yoo's regression tests every factor of
    :math:`1 - L^s` in one equation,

    .. math::

       (1 - L^s)\, y_t = d_t + \pi_0\, x_{0,t-1} + \pi_{s/2}\, x_{s/2,t-1}
       + \sum_{k = 1}^{s/2 - 1} \bigl(\pi_k^a\, x^a_{k,t-1} + \pi_k^b\, x^b_{k,t-1}\bigr)
       + \sum_{j = 1}^{p} \phi_j\, (1 - L^s)\, y_{t-j} + e_t,

    where each :math:`x` is the series with every root of
    :math:`1 - L^s` but one filtered out, so that :math:`\pi_0 = 0` is a
    unit root at frequency zero, :math:`\pi_{s/2} = 0` one at the Nyquist
    frequency :math:`\pi`, and :math:`\pi_k^a = \pi_k^b = 0` the complex
    pair at :math:`2\pi k / s`. The zero and Nyquist roots are tested by
    :math:`t` in the lower tail, each harmonic pair by :math:`F`, and two
    joint :math:`F` statistics, every seasonal root and every root, close
    the family. Under the null of a seasonal random walk none of these
    has a tabulated law that fits every period and deterministic set, so
    ``replications`` seasonal random walks of the same length are run
    through the same regression and the p-values and critical values
    read off them; at :math:`s = 4`, :math:`T = 100` with intercept and
    dummies the simulated 5% points reproduce Hylleberg et al.'s Table 1
    to within Monte Carlo error. The augmentation absorbs serial
    correlation in the seasonal difference and is chosen by ``method``
    on a common sample when ``lags`` is not given.

    Args:
        endog: The series, ``(T,)``, at least a few full cycles long.
        period: Seasonal period :math:`s`, even and at least 2.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        seasonal: Whether seasonal dummies enter the regression, which
            is what makes the test robust to a deterministic seasonal
            pattern under the alternative; ``True`` by default, and the
            record's ``trend`` reads ``"c+s"`` when they do.
        lags: Augmentation lags of the seasonal difference, or ``None``
            to select by ``method``.
        max_lags: Largest augmentation under selection; default
            Schwert's :math:`\lfloor 12 (T / 100)^{1/4} \rfloor`.
        method: ``"aic"``, ``"bic"`` or ``"t-stat"`` for the selection.
            BIC under-augments against moving-average errors, with an
            MA(1) coefficient of 0.5 it picks one or two lags where four
            are needed and the tests over-reject at 10-15%, so AIC is
            the default, as for :func:`~cultivars.diagnostics.unit_roots.adf`.
        replications: Seasonal random walks behind the p-values, at
            least ``500``.
        seed: Seed or generator for the simulation.

    Returns:
        The zero-frequency :class:`SeasonalUnitRootTest`, with the Nyquist
        :math:`t`, each harmonic :math:`F`, the joint seasonal :math:`F`
        and the joint :math:`F` as ``companions``, in that order.

    Raises:
        SpecificationError: If the period is odd or below 2, the trend,
            method, lags, or replication count is unusable, or the series
            is too short.
        NumericalError: If a value is not finite or the regression is
            degenerate.

    Note:
        The simulated null is the pure seasonal random walk with i.i.d.
        Gaussian innovations and the same deterministic set; the
        augmentation is fixed at the selected ``lags`` in the draws, so
        the p-values are exact for the regression as run rather than
        for the selection procedure. Each row rejects in its own tail,
        which the record's ``lower_tail`` carries, and the table's notes
        say how to read the pattern of rejections.

    See Also:
        * :func:`canova_hansen` -- the same frequencies with stationarity
          under the null.
        * :func:`~cultivars.diagnostics.unit_roots.adf` -- the
          zero-frequency test whose reading carries over.
        * :class:`SeasonalUnitRootTest` -- the record.

    References:
        Hylleberg, S., Engle, R. F., Granger, C. W. J., & Yoo, B. S.
        (1990). Seasonal integration and cointegration. *Journal of
        Econometrics*, 44(1-2), 215-238.

        Beaulieu, J. J., & Miron, J. A. (1993). Seasonal unit roots in
        aggregate U.S. data. *Journal of Econometrics*, 55(1-2), 305-328.

        Ghysels, E., Lee, H. S., & Noh, J. (1994). Testing for unit roots
        in seasonal time series: Some theoretical extensions and a Monte
        Carlo investigation. *Journal of Econometrics*, 62(2), 415-442.

    Example:
        A seasonal random walk, :math:`y_t = y_{t-4} + e_t`, has a unit
        root at every frequency, and no row rejects:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = rng.standard_normal(200)
        >>> for t in range(4, 200):
        ...     y[t] = y[t - 4] + rng.standard_normal()
        >>> verdict = hegy(y, period=4, replications=500, seed=0)
        >>> verdict.reject(), all(not c.reject() for c in verdict.companions)
        (False, True)
        >>> verdict.lags, verdict.method
        (2, 'aic over 0..14')

        A random walk with a fixed quarterly pattern keeps only the
        zero-frequency root:

        >>> z = np.cumsum(rng.standard_normal(200)) + np.tile([1.0, -0.5, 2.0, -2.5], 50)
        >>> verdict = hegy(z, period=4, replications=500, seed=0)
        >>> verdict.reject(), [c.reject() for c in verdict.companions]
        (False, [True, True, True, True])
    """
    y = _validate_seasonal(endog, period)
    trend = validate_choice(trend, ("n", "c", "ct"), "trend")
    if replications < 500:
        raise SpecificationError(f"replications must be at least 500; got {replications}.")
    if lags is None:
        ceiling = _schwert_max_lags(y.shape[0]) if max_lags is None else int(max_lags)
        ceiling = validate_order(ceiling, "max_lags", minimum=0)
        method = validate_choice(method, ("aic", "bic", "t-stat"), "method")
        chosen = _select_hegy_lags(y, period, trend, ceiling, method, seasonal=seasonal)
        label = f"{method} over 0..{ceiling}"
    else:
        chosen = validate_order(lags, "lags", minimum=0)
        label = "fixed"
    statistics, nobs = _hegy_statistics(y, period, trend, chosen, seasonal=seasonal)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    draws = _hegy_null_draws(
        period, y.shape[0], trend, seasonal=seasonal, replications=replications, rng=rng
    )
    spec = f"{trend}+s" if seasonal else trend
    labels = _frequency_labels(period)
    names = [
        "HEGY t, frequency 0",
        f"HEGY t, frequency {labels[0]}",
        *(f"HEGY F, frequency {label}" for label in labels[1:]),
        "HEGY F, all seasonal frequencies",
        "HEGY F, all frequencies",
    ]
    nulls = [
        "unit root at frequency 0",
        f"unit root at frequency {labels[0]}",
        *(f"unit root at frequency {label}" for label in labels[1:]),
        "unit roots at every seasonal frequency",
        "unit roots at every frequency",
    ]
    lower = [True, True] + [False] * (len(names) - 2)
    records = []
    for i, (name, null, tail) in enumerate(zip(names, nulls, lower, strict=True)):
        pvalue, cv = _simulated_critical_values(draws[:, i], float(statistics[i]), lower_tail=tail)
        records.append(
            SeasonalUnitRootTest(
                name=name,
                statistic=float(statistics[i]),
                pvalue=pvalue,
                critical_values=cv,
                null=null,
                lower_tail=tail,
                trend=spec,
                lags=chosen,
                nobs=nobs,
                method=label,
                family="HEGY",
                period=period,
            )
        )
    return replace(records[0], companions=tuple(records[1:]))


def canova_hansen(
    endog: npt.ArrayLike,
    *,
    period: int,
    trend: str = "c",
    lags: int = 0,
    bandwidth: int | None = None,
    n_draws: int = 5000,
    grid: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> SeasonalUnitRootTest:
    r"""Canova-Hansen test of stationary seasonality against a seasonal unit root.

    The series is regressed on the trend terms, :math:`s - 1`
    trigonometric seasonal regressors, and ``lags`` lags of itself; under
    the null the seasonal coefficients are constant, under the
    alternative those at the frequency tested follow a random walk, and
    the Lagrange multiplier statistic for that alternative is

    .. math::

       L = T^{-2} \sum_{t = 1}^{T} \hat F_t^\top\, \hat\Omega^{-1}\, \hat F_t,
       \qquad
       \hat F_t = \sum_{i \le t} f_i\, \hat e_i,

    with :math:`f_i` the seasonal columns of the frequency under test,
    :math:`\hat e_i` the residuals, and :math:`\hat\Omega` the Bartlett
    long-run covariance of the scores. Its limit is the generalized von
    Mises law, the integral of a squared :math:`q`-dimensional Brownian
    bridge, with :math:`q` the number of columns tested: one at the
    Nyquist frequency, two at each harmonic, :math:`s - 1` jointly. The
    p-values come from ``n_draws`` simulated bridges, whose 5% point at
    :math:`q = 1` is KPSS's 0.463. Rejection lies in the upper tail. The
    joint test is the headline and the per-frequency tests say where
    the instability sits.

    Args:
        endog: The series, ``(T,)``, at least a few full cycles long.
        period: Seasonal period :math:`s`, even and at least 2.
        trend: ``"n"``, ``"c"`` or ``"ct"``.
        lags: Lags of the series among the regressors, at least 0, which
            absorb short-run dynamics that would otherwise inflate the
            statistic.
        bandwidth: Bartlett lags of the long-run covariance, at least 0;
            default Newey-West's :math:`\lfloor 4 (T / 100)^{2/9} \rfloor`.
        n_draws: Simulated bridges behind each p-value, at least ``500``.
        grid: Points on which each bridge is simulated, at least ``100``.
        seed: Seed or generator for the simulation.

    Returns:
        The joint :class:`SeasonalUnitRootTest`, with the Nyquist and each
        harmonic frequency as ``companions``, in that order; its ``lags``
        field is the kernel bandwidth and its ``method`` names the
        kernel and the lag count.

    Raises:
        SpecificationError: If the period is odd or below 2, the trend,
            lags, bandwidth, or simulation sizes are unusable, or the
            series is too short.
        NumericalError: If a value is not finite or a long-run
            covariance is singular.

    Note:
        The test is the seasonal analogue of KPSS and inherits its
        habits: the long-run variance is estimated under the null, so a
        bandwidth too short for the residual autocorrelation
        over-rejects and one too long loses power, and a series with a
        unit root at frequency zero should be differenced first, since
        the statistic is for the seasonal frequencies alone. Lags of the
        series are the paper's remedy for short-run dynamics and are
        preferred to a wider bandwidth.

    See Also:
        * :func:`hegy` -- the same frequencies with a unit root under the
          null; read the two together.
        * :func:`~cultivars.diagnostics.unit_roots.kpss` -- the
          zero-frequency test this one generalizes.
        * :class:`SeasonalUnitRootTest` -- the record.

    References:
        Canova, F., & Hansen, B. E. (1995). Are seasonal patterns
        constant over time? A test for seasonal stability. *Journal of
        Business & Economic Statistics*, 13(3), 237-252.

        Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y.
        (1992). Testing the null hypothesis of stationarity against the
        alternative of a unit root. *Journal of Econometrics*, 54(1-3),
        159-178.

        Busetti, F., & Harvey, A. (2003). Seasonality tests. *Journal of
        Business & Economic Statistics*, 21(3), 420-436.

    Example:
        A fixed quarterly cycle keeps stationary seasonality; a seasonal
        random walk rejects it, jointly and at each frequency:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> t = np.arange(300)
        >>> stable = np.cos(np.pi * t / 2) + rng.standard_normal(300)
        >>> verdict = canova_hansen(stable, period=4, seed=0)
        >>> verdict.reject(), verdict.method
        (False, 'Bartlett kernel, 0 lag(s) of y')
        >>> y = rng.standard_normal(200)
        >>> for s in range(4, 200):
        ...     y[s] = y[s - 4] + rng.standard_normal()
        >>> verdict = canova_hansen(y, period=4, seed=0)
        >>> verdict.reject(), [c.reject() for c in verdict.companions]
        (True, [True, True])
    """
    y = _validate_seasonal(endog, period)
    trend = validate_choice(trend, ("n", "c", "ct"), "trend")
    lags = validate_order(lags, "lags", minimum=0)
    if n_draws < 500 or grid < 100:
        raise SpecificationError(
            f"n_draws must be at least 500 and grid at least 100; got {n_draws}, {grid}."
        )
    width = _newey_west_bandwidth(y.shape[0]) if bandwidth is None else int(np.floor(bandwidth))
    width = validate_order(width, "bandwidth", minimum=0)
    statistics, degrees, nobs = _canova_hansen(y, period, trend, lags, bandwidth=width)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    draws = {
        int(q): _von_mises_draws(int(q), n_draws=n_draws, grid=grid, rng=rng) for q in set(degrees)
    }
    labels = _frequency_labels(period)
    names = [*(f"Canova-Hansen, frequency {label}" for label in labels), "Canova-Hansen, joint"]
    nulls = [
        *(f"stationary seasonality at frequency {label}" for label in labels),
        "stationary seasonality at every frequency",
    ]
    records = []
    for i, (name, null) in enumerate(zip(names, nulls, strict=True)):
        pvalue, cv = _simulated_critical_values(draws[int(degrees[i])], float(statistics[i]))
        records.append(
            SeasonalUnitRootTest(
                name=name,
                statistic=float(statistics[i]),
                pvalue=pvalue,
                critical_values=cv,
                null=null,
                lower_tail=False,
                trend=trend,
                lags=width,
                nobs=nobs,
                method=f"Bartlett kernel, {lags} lag(s) of y",
                family="Canova-Hansen",
                period=period,
            )
        )
    return replace(records[-1], companions=tuple(records[:-1]))
