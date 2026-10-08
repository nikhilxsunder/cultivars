# filepath: /src/cultivars/diagnostics/long_memory.py
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
r"""Semiparametric long memory: the differencing parameter without a short-run model.

A fractionally integrated process :math:`(1 - L)^d y_t = u_t` with
:math:`u_t` short-memory has a spectrum that behaves near the origin as

.. math::

   f(\lambda) \sim c\,\lambda^{-2d}, \qquad \lambda \to 0^+,

whatever the short-run structure of :math:`u_t`, and the three
estimators here read :math:`d` off that behaviour at the :math:`m`
lowest Fourier frequencies alone. An ARFIMA fit commits to an ARMA
short-run model and estimates :math:`d` jointly with it; when the ARMA
order is wrong, :math:`d` absorbs the error. These need no short-run
model, and their bandwidth :math:`m` is the only choice, trading bias
from the short-run spectrum, which grows with :math:`m`, against
variance, which shrinks with it. Geweke-Porter-Hudak is the
log-periodogram regression, the oldest and least efficient, with
standard error :math:`\pi / \sqrt{24 m}`; local Whittle is Robinson's
Gaussian semiparametric estimator, standard error :math:`1 / (2\sqrt{m})`
and valid for :math:`d < 3/4`; exact local Whittle fractionally
differences the series inside the objective and is valid for any
:math:`d`, which makes it the one to run when the answer might be "this
is a unit root". The unit-root question is exactly where they earn
their place: :math:`d = 1` and :math:`d = 0.7` reject the same ADF null
and mean different things, the first a permanent shock and the second a
shock that decays, however slowly.

Two commitments shape the surface. First, a record is an estimate that
carries its test, not a test that happens to expose an estimate: every
producer returns a :class:`LongMemoryEstimate` with ``d`` and ``se``
first, and the natural nulls, :math:`d = 0` for no long memory,
:math:`d = 1/2` for the stationarity boundary, :math:`d = 1` for a unit
root, are one :meth:`LongMemoryEstimate.test` away without
re-estimating, the summary table showing all three at once. Second,
the record says what its standard error is worth. The asymptotic
standard error is a function of the bandwidth alone and does not see
the short-run dynamics the estimator ignored, so the record carries
``m``, the table names it, and the local Whittle record warns when the
estimate has left the range its limit theory covers rather than
printing a p-value from a law that no longer holds.

Layout. :class:`LongMemoryEstimate` is the record, a
:class:`~cultivars.diagnostics.hypothesis.HypothesisTest` with
:meth:`~LongMemoryEstimate.test`, :meth:`~LongMemoryEstimate.confidence_interval`,
and :attr:`~LongMemoryEstimate.is_stationary`; :func:`gph`,
:func:`local_whittle`, and :func:`exact_local_whittle` are the
producers, which validate, estimate, and assemble the record through
one shared constructor. The numerics live in ``_core``:
``_validate_semiparametric`` checks the series and applies the
bandwidth rule, ``_gph`` runs the log-periodogram regression,
``local_whittle_d`` minimizes the concentrated Whittle objective and is
shared with the ARFIMA model's starting value, ``_exact_local_whittle``
applies the fractional difference and mean correction inside that
objective, and the bandwidth exponents, search bounds, and minimum
sample length are the module's named defaults.

References:
    Geweke, J., & Porter-Hudak, S. (1983). The estimation and
    application of long memory time series models. *Journal of Time
    Series Analysis*, 4(4), 221-238.

    Robinson, P. M. (1995). Gaussian semiparametric estimation of long
    range dependence. *Annals of Statistics*, 23(5), 1630-1661.

    Shimotsu, K., & Phillips, P. C. B. (2005). Exact local Whittle
    estimation of fractional integration. *Annals of Statistics*,
    33(4), 1890-1933.

    Shimotsu, K. (2010). Exact local Whittle estimation of fractional
    integration with unknown mean and time trend. *Econometric
    Theory*, 26(2), 501-540.

    Baillie, R. T. (1996). Long memory processes and fractional
    integration in econometrics. *Journal of Econometrics*, 73(1),
    5-59.

Example:
    A fractionally integrated series with :math:`d = 0.3` is stationary
    with long memory: every estimator puts :math:`d` between zero and
    one half, and the Whittle pair does so with the smaller standard
    error:

    >>> import numpy as np
    >>> from cultivars._core import fractional_difference_weights
    >>> rng = np.random.default_rng(0)
    >>> weights = fractional_difference_weights(-0.3, 1000)
    >>> y = np.convolve(rng.standard_normal(1000), weights)[:1000]
    >>> for estimate in (gph(y), local_whittle(y), exact_local_whittle(y)):
    ...     print(f"{estimate.method:<20} d={estimate.d:.2f}  se={estimate.se:.3f}")
    gph                  d=0.36  se=0.137
    local_whittle        d=0.31  se=0.053
    exact_local_whittle  d=0.32  se=0.053
    >>> estimate.reject(), estimate.test(0.5).reject(), estimate.is_stationary
    (True, True, True)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from .._core import (
    _ELW_BOUNDS,
    _GPH_EXPONENT,
    _LW_BOUNDS,
    _METHOD_LABELS,
    _MIN_LONG_MEMORY_OBS,
    _WHITTLE_EXPONENT,
    SummaryTable,
    _exact_local_whittle,
    _gph,
    _validate_semiparametric,
    local_whittle_d,
)
from .._internals import _HypothesisTest
from ..exceptions import SpecificationError

__all__ = ["LongMemoryEstimate", "exact_local_whittle", "gph", "local_whittle"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class LongMemoryEstimate(_HypothesisTest):
    r"""A semiparametric estimate of the fractional differencing parameter, with a test.

    The record is an estimate first, ``d`` and ``se``, and a test second.
    Each of the three estimators is asymptotically normal in the
    bandwidth :math:`m`,

    .. math::

       \sqrt{m}\,(\hat d - d) \;\xrightarrow{d}\;
       \mathcal{N}\!\left(0,\; \tfrac{1}{4}\right)
       \quad\text{(local and exact local Whittle)},
       \qquad
       \mathcal{N}\!\left(0,\; \tfrac{\pi^2}{24}\right)
       \quad\text{(GPH)},

    so ``statistic`` is :math:`(\hat d - d_0) / \widehat{se}` and
    ``pvalue`` its two-sided standard normal p-value, with
    :math:`d_0 = 0`, no long memory, as built; :meth:`test` re-aims it
    at any other null, :math:`1/2` for the stationarity boundary or
    :math:`1` for a unit root, without re-estimating. Standard errors
    are asymptotic in the bandwidth and do not see the short-run
    dynamics the estimator ignores, so the reader should expect them to
    be mildly optimistic: on 500 observations the nominal 95% interval
    covers about 90 to 94% of the time, dipping to the low 80s for exact
    local Whittle near :math:`d = 1/2`, where its mean correction
    switches regime.

    Attributes:
        d: The estimate.
        se: Its asymptotic standard error.
        statistic: ``(d - null_d) / se``.
        pvalue: Two-sided standard normal p-value.
        null_d: The value of ``d`` under test.
        method: ``"gph"``, ``"local_whittle"`` or ``"exact_local_whittle"``.
        bandwidth: Fourier frequencies used, ``m``.
        nobs: Observations.

    Note:
        The three canonical nulls sit in the summary table together, so
        ``print(est)`` answers "no long memory", "stationary", and "unit
        root" at once. Because ``statistic`` and ``pvalue`` are always
        against ``null_d``, :meth:`reject` on a freshly built record is
        the test of :math:`d = 0`; call ``est.test(1.0).reject()`` for
        the unit-root question. The local Whittle estimator's limit
        theory holds only for :math:`d < 3/4` and its optimizer is
        bounded above at one, so an estimate at the bound is a
        signal to switch to exact local Whittle, which the table says.

    See Also:
        * :func:`gph`, :func:`local_whittle`, :func:`exact_local_whittle`
          -- the three producers.
        * :func:`~cultivars.diagnostics.unit_roots.adf` -- the integer
          alternative, which cannot tell :math:`d = 1` from
          :math:`d = 0.7`.
        * :class:`~cultivars.univariate.fractional.ARFIMA` -- the
          parametric estimate of :math:`d` jointly with a short-run model.

    References:
        Geweke, J., & Porter-Hudak, S. (1983). The estimation and
        application of long memory time series models. *Journal of Time
        Series Analysis*, 4(4), 221-238.

        Robinson, P. M. (1995). Gaussian semiparametric estimation of long
        range dependence. *Annals of Statistics*, 23(5), 1630-1661.

        Shimotsu, K., & Phillips, P. C. B. (2005). Exact local Whittle
        estimation of fractional integration. *Annals of Statistics*,
        33(4), 1890-1933.

    Example:
        A random walk is :math:`d = 1`: the estimate rejects no long
        memory and keeps a unit root, and the interval covers one:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> est = exact_local_whittle(np.cumsum(rng.standard_normal(500)))
        >>> est.method, round(est.d, 4), round(est.se, 4), est.bandwidth
        ('exact_local_whittle', 1.059, 0.0668, 56)
        >>> est.reject(), est.test(1.0).reject(), est.is_stationary
        (True, False, False)
        >>> low, high = est.confidence_interval()
        >>> bool(low < 1.0 < high)
        True
    """

    d: float
    """The point estimate of the fractional differencing parameter."""
    se: float
    r"""Asymptotic standard error of ``d``.

    :math:`1 / (2\sqrt{m})` for the two Whittle estimators and
    :math:`\pi / \sqrt{24 m}` for GPH; a function of the bandwidth
    alone, and therefore blind to the short-run dynamics.
    """
    pvalue: float
    """Two-sided standard normal p-value of ``statistic`` against ``null_d``."""
    null_d: float
    """The value of ``d`` the ``statistic`` and ``pvalue`` are against; ``0.0`` as built."""
    method: str
    """Which estimator: ``"gph"``, ``"local_whittle"`` or ``"exact_local_whittle"``."""
    bandwidth: int
    r"""Fourier frequencies used, :math:`m`, the one tuning choice.

    Set by the producer as :math:`\lfloor T^{\alpha} \rfloor` with the
    method's default exponent unless given; the standard error shrinks
    with it and the bias from short-run dynamics grows with it.
    """
    nobs: int
    """Observations the periodogram was computed on."""

    def test(self, null: float) -> LongMemoryEstimate:
        """The same estimate re-aimed at ``d = null``.

        Only ``statistic``, ``pvalue``, and ``null_d`` change; the
        estimate, its standard error, and the bandwidth are the same,
        because a semiparametric estimate does not depend on the
        hypothesis.

        Args:
            null: The value of ``d`` under the null.

        Returns:
            A record with ``statistic`` and ``pvalue`` against ``null``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> est = exact_local_whittle(np.cumsum(rng.standard_normal(500)))
            >>> unit_root = est.test(1.0)
            >>> unit_root.null_d, round(unit_root.statistic, 3), bool(unit_root.pvalue > 0.05)
            (1.0, 0.882, True)
        """
        statistic = (self.d - float(null)) / self.se
        return replace(
            self,
            statistic=statistic,
            pvalue=2.0 * float(norm.sf(abs(statistic))),
            null_d=float(null),
        )

    def confidence_interval(self, *, alpha: float = 0.05) -> tuple[float, float]:
        r"""Asymptotic ``1 - alpha`` interval for ``d``.

        :math:`\hat d \pm z_{1 - \alpha/2}\, \widehat{se}`, symmetric
        because the limit is normal.

        Args:
            alpha: One minus the coverage, in ``(0, 1)``.

        Returns:
            ``(low, high)``.

        Raises:
            SpecificationError: If ``alpha`` is not inside ``(0, 1)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> est = exact_local_whittle(np.cumsum(rng.standard_normal(500)))
            >>> tuple(round(v, 3) for v in est.confidence_interval())
            (0.928, 1.19)
            >>> tuple(round(v, 3) for v in est.confidence_interval(alpha=0.32))
            (0.993, 1.125)
        """
        if not 0.0 < alpha < 1.0:
            raise SpecificationError(f"alpha must lie strictly inside (0, 1); got {alpha}.")
        half = float(norm.ppf(1.0 - alpha / 2.0)) * self.se
        return self.d - half, self.d + half

    @property
    def is_stationary(self) -> bool:
        r"""Whether the point estimate lies in the stationary range :math:`d < 1/2`.

        A reading of the estimate, not a test; the boundary test is
        ``test(0.5)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> exact_local_whittle(rng.standard_normal(500)).is_stationary
            True
        """
        return self.d < 0.5

    def _summary_table(self) -> SummaryTable:
        """Render as a table with the three canonical nulls.

        One row each for :math:`d = 0`, :math:`d = 1/2`, and
        :math:`d = 1` with the z statistic and its p-value, under a
        header carrying the estimate, its standard error, the bandwidth,
        and the stationarity reading; the 95% interval is a note, and
        the local Whittle record adds a warning above :math:`d = 0.7`
        while the GPH record notes its efficiency loss.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``str()``
            and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> table = exact_local_whittle(np.cumsum(rng.standard_normal(500)))._summary_table()
            >>> table.columns, [row[0] for row in table.rows]
            (('null', 'z', 'p-value'), ['d = 0', 'd = 1/2', 'd = 1'])
        """
        rows = []
        for label, null in (("d = 0", 0.0), ("d = 1/2", 0.5), ("d = 1", 1.0)):
            z = (self.d - null) / self.se
            rows.append((label, f"{z:.4f}", f"{2.0 * norm.sf(abs(z)):.4f}"))
        low, high = self.confidence_interval()
        notes = [
            f"95% interval for d: ({low:.4f}, {high:.4f}); the standard error is asymptotic "
            f"in the bandwidth m = {self.bandwidth}.",
        ]
        if self.method == "local_whittle" and self.d > 0.7:
            notes.append(
                "The local Whittle limit theory holds for d < 3/4; at this estimate prefer "
                "exact_local_whittle, which is valid for any d."
            )
        if self.method == "gph":
            notes.append(
                "GPH is the least efficient of the three at a given bandwidth; its standard "
                "error is pi / sqrt(24 m) against 1 / (2 sqrt(m)) for the Whittle estimators."
            )
        return SummaryTable(
            title=f"{_METHOD_LABELS.get(self.method, self.method)} Long Memory Estimate",
            metadata=(
                ("d", f"{self.d:.4f}"),
                ("Std. error", f"{self.se:.4f}"),
                ("Bandwidth", str(self.bandwidth)),
                ("Observations", str(self.nobs)),
                ("Stationary", "yes" if self.is_stationary else "no"),
            ),
            columns=("null", "z", "p-value"),
            rows=tuple(rows),
            notes=tuple(notes),
        )

    def __repr__(self) -> str:
        """One-line estimate: method, ``d``, standard error, bandwidth, observations.

        The test against ``null_d`` is left to the table, since the
        record is an estimate first.

        Returns:
            The repr string.

        Example:
            >>> import numpy as np
            >>> gph(np.random.default_rng(0).standard_normal(500))
            LongMemoryEstimate(method='gph', d=-0.0135, se=0.1704, bandwidth=22, nobs=500)
        """
        return (
            f"LongMemoryEstimate(method={self.method!r}, d={self.d:.4f}, se={self.se:.4f}, "
            f"bandwidth={self.bandwidth}, nobs={self.nobs})"
        )

    @classmethod
    def _record(cls, d: float, se: float, *, method: str, m: int, nobs: int) -> LongMemoryEstimate:
        """Assemble the record with the ``d = 0`` test.

        The one constructor the three producers share, so the default
        null and its p-value are computed in one place.

        Args:
            d: The estimate.
            se: Its asymptotic standard error.
            method: The producer's name.
            m: The bandwidth used.
            nobs: Observations.

        Returns:
            The record, with ``statistic = d / se`` and ``null_d = 0.0``.

        Example:
            >>> est = LongMemoryEstimate._record(0.3, 0.1, method="gph", m=20, nobs=400)
            >>> round(est.statistic, 6), est.null_d, round(est.pvalue, 4)
            (3.0, 0.0, 0.0027)
        """
        statistic = d / se
        return cls(
            d=d,
            se=se,
            statistic=statistic,
            pvalue=2.0 * float(norm.sf(abs(statistic))),
            null_d=0.0,
            method=method,
            bandwidth=m,
            nobs=nobs,
        )


def gph(
    endog: npt.ArrayLike, *, f_bandwidth: int | None = None, exponent: float = _GPH_EXPONENT
) -> LongMemoryEstimate:
    r"""Geweke-Porter-Hudak log-periodogram estimate of :math:`d`.

    Near the origin the spectrum of a long-memory process behaves as
    :math:`f(\lambda) \sim c\,\lambda^{-2d}`, so the log periodogram at
    the first :math:`m` Fourier frequencies :math:`\lambda_j = 2\pi j / T`
    is a regression,

    .. math::

       \log I(\lambda_j) = a - d \log\bigl(4 \sin^2(\lambda_j / 2)\bigr) + e_j,
       \qquad j = 1, \ldots, m,

    and :math:`\hat d` is the least-squares slope, with standard error
    :math:`\pi / \sqrt{24 m}` from the known variance of the
    log-periodogram errors. The default bandwidth
    :math:`m = \lfloor T^{1/2} \rfloor` is the paper's; it trades bias
    from the short-run spectrum, which grows with :math:`m`, against
    variance, which shrinks with it, and the standard error is about
    0.14 at :math:`T = 1000`, so the test has little power against
    :math:`d` near zero at that length. The estimator is the oldest of
    the three and the least efficient at a given bandwidth; its
    justification is its simplicity, and the regression residuals are
    there to be looked at. Valid for :math:`d < 1/2`; on a series that
    may be integrated, difference first or use
    :func:`exact_local_whittle`.

    Args:
        endog: The series, ``(T,)``, at least 64 observations.
        f_bandwidth: Fourier frequencies used, at least ``2`` and at most
            ``T // 2``; ``None`` applies the rule.
        exponent: Exponent of the rule ``m = floor(T ** exponent)``, in
            ``(0, 1)``; ``0.5`` by default.

    Returns:
        The :class:`LongMemoryEstimate`, with ``method="gph"`` and the
        test against :math:`d = 0`.

    Raises:
        SpecificationError: If the series is shorter than 64, the
            bandwidth is below 2 or above the available frequencies, or
            the exponent is outside ``(0, 1)``.
        NumericalError: If a value is not finite or a periodogram
            ordinate is zero, which a constant series produces.

    Note:
        A bandwidth of a handful of frequencies is accepted and gives a
        standard error above one; the estimate is then noise, and the
        record's ``se`` says so rather than the call refusing.

    See Also:
        * :func:`local_whittle` -- the same bandwidth idea with a
          Gaussian objective, and a standard error smaller by a factor
          of :math:`\pi / \sqrt{6}`.
        * :func:`exact_local_whittle` -- when :math:`d` may be at or
          above one half.

    References:
        Geweke, J., & Porter-Hudak, S. (1983). The estimation and
        application of long memory time series models. *Journal of Time
        Series Analysis*, 4(4), 221-238.

        Robinson, P. M. (1995). Log-periodogram regression of time series
        with long range dependence. *Annals of Statistics*, 23(3),
        1048-1072.

        Hurvich, C. M., Deo, R., & Brodsky, J. (1998). The mean squared
        error of Geweke and Porter-Hudak's estimator of the memory
        parameter of a long-memory time series. *Journal of Time Series
        Analysis*, 19(1), 19-46.

    Example:
        White noise is :math:`d = 0`, and the estimate is within three
        standard errors of it at the default bandwidth:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> est = gph(rng.standard_normal(1000))
        >>> bool(abs(est.d) < 3 * est.se), est.bandwidth
        (True, 31)

        A wider bandwidth tightens the standard error at the price of
        short-run bias:

        >>> gph(rng.standard_normal(1000), f_bandwidth=100).se < est.se
        True
    """
    y, m = _validate_semiparametric(endog, f_bandwidth, exponent, minimum=_MIN_LONG_MEMORY_OBS)
    d, se = _gph(y, m)
    return LongMemoryEstimate._record(d, se, method="gph", m=m, nobs=y.shape[0])


def local_whittle(
    endog: npt.ArrayLike, *, f_bandwidth: int | None = None, exponent: float = _WHITTLE_EXPONENT
) -> LongMemoryEstimate:
    r"""Robinson's (1995) local Whittle estimate of :math:`d`.

    The Gaussian semiparametric estimator minimizes the concentrated
    local Whittle objective over the first :math:`m` Fourier
    frequencies,

    .. math::

       R(d) = \log\Bigl(\frac{1}{m} \sum_{j = 1}^{m} \lambda_j^{2d} I(\lambda_j)\Bigr)
       - \frac{2d}{m} \sum_{j = 1}^{m} \log \lambda_j,

    which is the Whittle likelihood of the local model
    :math:`f(\lambda) = c\,\lambda^{-2d}` with :math:`c` concentrated
    out. Its limit is :math:`\sqrt{m}(\hat d - d) \to \mathcal{N}(0, 1/4)`
    for :math:`-1/2 < d < 3/4`, more efficient than GPH at the same
    bandwidth by a factor of :math:`\pi^2 / 6` in variance; it stays
    consistent up to :math:`d < 1` but loses its limit law beyond
    :math:`3/4`, and the record says so when the estimate lands there.
    The search runs over :math:`(-1/2, 1)`, so a unit-root series is
    reported at the upper boundary with a pointer to
    :func:`exact_local_whittle`. The default bandwidth
    :math:`m = \lfloor T^{0.65} \rfloor` is wider than GPH's because the
    efficiency gain makes the variance side of the trade cheaper.

    Args:
        endog: The series, ``(T,)``, at least 64 observations.
        f_bandwidth: Fourier frequencies used, at least ``2`` and at most
            ``T // 2``; ``None`` applies the rule.
        exponent: Exponent of the rule ``m = floor(T ** exponent)``, in
            ``(0, 1)``; ``0.65`` by default.

    Returns:
        The :class:`LongMemoryEstimate`, with ``method="local_whittle"``,
        standard error :math:`1 / (2\sqrt{m})`, and the test against
        :math:`d = 0`.

    Raises:
        SpecificationError: If the series is shorter than 64, the
            bandwidth is below 2 or above the available frequencies, or
            the exponent is outside ``(0, 1)``.
        NumericalError: If a value is not finite.

    Note:
        An estimate at ``1.0`` to display precision is the search bound,
        not a finding: the objective is still decreasing there and the
        series is at least a unit root. Switch to :func:`exact_local_whittle`,
        whose search extends to two, for the value.

    See Also:
        * :func:`exact_local_whittle` -- the same objective on the
          fractionally differenced series, valid for any :math:`d`.
        * :func:`gph` -- the log-periodogram regression, less efficient
          and simpler.

    References:
        Robinson, P. M. (1995). Gaussian semiparametric estimation of long
        range dependence. *Annals of Statistics*, 23(5), 1630-1661.

        Künsch, H. R. (1987). Statistical aspects of self-similar
        processes. In *Proceedings of the First World Congress of the
        Bernoulli Society* (Vol. 1, pp. 67-74). VNU Science Press.

        Phillips, P. C. B., & Shimotsu, K. (2004). Local Whittle
        estimation in nonstationary and unit root cases. *Annals of
        Statistics*, 32(2), 656-692.

    Example:
        White noise is :math:`d = 0`; the default bandwidth is wider than
        GPH's and the standard error correspondingly smaller:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> est = local_whittle(rng.standard_normal(1000))
        >>> bool(abs(est.d) < 3 * est.se), est.bandwidth, round(float(est.se), 3)
        (True, 89, 0.053)

        A random walk lands on the search bound:

        >>> round(local_whittle(np.cumsum(rng.standard_normal(500))).d, 3)
        1.0
    """
    y, m = _validate_semiparametric(endog, f_bandwidth, exponent, minimum=_MIN_LONG_MEMORY_OBS)
    d, m_eff = local_whittle_d(y, m, exponent, bounds=_LW_BOUNDS)
    se = 1.0 / (2.0 * np.sqrt(m_eff))
    return LongMemoryEstimate._record(d, se, method="local_whittle", m=m_eff, nobs=y.shape[0])


def exact_local_whittle(
    endog: npt.ArrayLike, *, f_bandwidth: int | None = None, exponent: float = _WHITTLE_EXPONENT
) -> LongMemoryEstimate:
    r"""Shimotsu-Phillips exact local Whittle estimate of :math:`d`, valid for any :math:`d`.

    The local Whittle objective is evaluated not on the periodogram of
    the series but on that of its fractional difference,

    .. math::

       R(d) = \log\Bigl(\frac{1}{m} \sum_{j = 1}^{m}
       I_{\Delta^d (y - \hat\mu(d))}(\lambda_j)\Bigr)
       - \frac{2d}{m} \sum_{j = 1}^{m} \log \lambda_j,

    where :math:`\Delta^d = (1 - L)^d` is applied inside the objective
    for each trial :math:`d`. Differencing the series to short memory
    before reading its periodogram is what removes the :math:`d < 3/4`
    restriction of :func:`local_whittle`: the limit
    :math:`\sqrt{m}(\hat d - d) \to \mathcal{N}(0, 1/4)` then holds across
    the stationary and nonstationary ranges alike (Shimotsu & Phillips
    2005). The mean correction :math:`\hat\mu(d)` is Shimotsu's (2010)
    weighted estimate, the sample mean for :math:`d \le 1/2` and the
    first observation for :math:`d \ge 3/4` with a smooth blend
    between, so a random walk is anchored at its start rather than at a
    mean it does not have. The search runs over :math:`(-1/2, 2)`. This
    is the estimator for the question the ADF test cannot answer:
    whether a series that rejects nothing is :math:`d = 1` or merely
    :math:`d = 0.7`.

    Args:
        endog: The series, ``(T,)``, at least 64 observations.
        f_bandwidth: Fourier frequencies used, at least ``2`` and at most
            ``T // 2``; ``None`` applies the rule.
        exponent: Exponent of the rule ``m = floor(T ** exponent)``, in
            ``(0, 1)``; ``0.65`` by default.

    Returns:
        The :class:`LongMemoryEstimate`, with
        ``method="exact_local_whittle"``, standard error
        :math:`1 / (2\sqrt{m})`, and the test against :math:`d = 0`.

    Raises:
        SpecificationError: If the series is shorter than 64, the
            bandwidth is below 2 or above the available frequencies, or
            the exponent is outside ``(0, 1)``.
        NumericalError: If a value is not finite.

    Note:
        The mean correction switches regime between :math:`1/2` and
        :math:`3/4`, and the finite-sample coverage of the asymptotic
        interval is weakest there, in the low 80s at 500 observations
        against a nominal 95. Each trial :math:`d` fractionally
        differences the whole series, so the call is slower than
        :func:`local_whittle` by the cost of the optimizer's evaluations.

    See Also:
        * :func:`local_whittle` -- the same objective on the raw
          periodogram, faster, valid for :math:`d < 3/4`.
        * :func:`~cultivars.diagnostics.unit_roots.adf` -- the integer
          question this estimator refines.
        * :class:`~cultivars.univariate.fractional_integration.ARFIMA` --
          the parametric alternative once :math:`d` is in the
            stationary range.

    References:
        Shimotsu, K., & Phillips, P. C. B. (2005). Exact local Whittle
        estimation of fractional integration. *Annals of Statistics*,
        33(4), 1890-1933.

        Shimotsu, K. (2010). Exact local Whittle estimation of fractional
        integration with unknown mean and time trend. *Econometric
        Theory*, 26(2), 501-540.

    Example:
        A random walk is :math:`d = 1`, and the estimate keeps that null
        while rejecting no long memory:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> walk = np.cumsum(rng.standard_normal(500))
        >>> est = exact_local_whittle(walk)
        >>> bool(abs(est.d - 1.0) < 3 * est.se), est.test(1.0).reject(), est.reject()
        (True, False, True)

        Where :func:`local_whittle` stops at its bound, this one reads
        past it:

        >>> round(local_whittle(walk).d, 3), bool(est.d > 1.0)
        (1.0, True)
    """
    y, m = _validate_semiparametric(endog, f_bandwidth, exponent, minimum=_MIN_LONG_MEMORY_OBS)
    d, se = _exact_local_whittle(y, m, bounds=_ELW_BOUNDS)
    return LongMemoryEstimate._record(d, se, method="exact_local_whittle", m=m, nobs=y.shape[0])
