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

"""Semiparametric long memory: the differencing parameter without a short-run model.

An ARFIMA fit commits to an ARMA short-run structure and estimates ``d``
jointly with it; when the ARMA order is wrong, ``d`` absorbs the error.
The three estimators here read ``d`` off the periodogram's behaviour at
the ``m`` lowest Fourier frequencies alone, so they need no short-run
model and their bandwidth is the only choice. Geweke-Porter-Hudak is
the log-periodogram regression, the oldest and least efficient; local
Whittle is the Gaussian semiparametric estimator, more efficient at the
same bandwidth but only valid for ``d < 3/4``; exact local Whittle
fractionally differences the series inside the objective and is valid
for any ``d``, which makes it the one to run when the answer might be
"this is a unit root". The unit-root question is exactly where they
earn their place: ``d = 1`` and ``d = 0.7`` reject the same ADF null and
mean different things.

Each record carries a standard error, so the natural questions --
``d = 0``, no long memory; ``d = 1/2``, the stationarity boundary;
``d = 1``, a unit root -- are one :meth:`LongMemoryEstimate.test` away,
and the record's own ``statistic`` and ``pvalue`` address the first.

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
    """A semiparametric estimate of the fractional differencing parameter, with a test.

    The record is an estimate first -- ``d`` and ``se`` -- and a test
    second: ``statistic`` is ``(d - null_d) / se`` and ``pvalue`` its
    two-sided normal p-value, with ``null_d = 0`` (no long memory) as
    built; :meth:`test` re-aims it at any other null, ``1/2`` for the
    stationarity boundary or ``1`` for a unit root, without
    re-estimating. Standard errors are asymptotic in the bandwidth, so
    the record carries ``m`` and the reader should expect them to be
    mildly optimistic: on 500 observations the nominal 95% interval
    covers about 90-94% of the time, dipping to the low 80s for exact
    local Whittle near ``d = 1/2``, where its mean correction switches
    regime.

    Attributes:
        d: The estimate.
        se: Its asymptotic standard error.
        statistic: ``(d - null_d) / se``.
        pvalue: Two-sided standard normal p-value.
        null_d: The value of ``d`` under test.
        method: ``"gph"``, ``"local_whittle"`` or ``"exact_local_whittle"``.
        bandwidth: Fourier frequencies used, ``m``.
        nobs: Observations.
    """

    d: float
    se: float
    pvalue: float
    null_d: float
    method: str
    bandwidth: int
    nobs: int

    def test(self, null: float) -> LongMemoryEstimate:
        """The same estimate re-aimed at ``d = null``.

        Args:
            null: The value of ``d`` under the null.

        Returns:
            A record with ``statistic`` and ``pvalue`` against ``null``.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> est = exact_local_whittle(np.cumsum(rng.standard_normal(500)))
            >>> bool(est.test(1.0).pvalue > 0.05), bool(est.reject())
            (True, True)
        """
        statistic = (self.d - float(null)) / self.se
        return replace(
            self,
            statistic=statistic,
            pvalue=2.0 * float(norm.sf(abs(statistic))),
            null_d=float(null),
        )

    def confidence_interval(self, *, alpha: float = 0.05) -> tuple[float, float]:
        """Asymptotic ``1 - alpha`` interval for ``d``.

        Raises:
            SpecificationError: If ``alpha`` is not inside ``(0, 1)``.
        """
        if not 0.0 < alpha < 1.0:
            raise SpecificationError(f"alpha must lie strictly inside (0, 1); got {alpha}.")
        half = float(norm.ppf(1.0 - alpha / 2.0)) * self.se
        return self.d - half, self.d + half

    @property
    def is_stationary(self) -> bool:
        """Whether the point estimate lies in the stationary range ``d < 1/2``."""
        return self.d < 0.5

    def _summary_table(self) -> SummaryTable:
        """Render as a table with the three canonical nulls."""
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
        """One-line estimate."""
        return (
            f"LongMemoryEstimate(method={self.method!r}, d={self.d:.4f}, se={self.se:.4f}, "
            f"bandwidth={self.bandwidth}, nobs={self.nobs})"
        )

    @classmethod
    def _record(cls, d: float, se: float, *, method: str, m: int, nobs: int) -> LongMemoryEstimate:
        """Assemble the record with the ``d = 0`` test."""
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
    """Geweke-Porter-Hudak log-periodogram estimate of ``d``.

    Regress the log periodogram on ``-log(4 sin^2(lambda / 2))`` over the
    first ``m`` Fourier frequencies; the slope is ``d``. The default
    bandwidth ``m = floor(T^0.5)`` is the paper's; it trades bias from
    the short-run spectrum against variance, and the standard error
    ``pi / sqrt(24 m)`` is about 0.15 at ``T = 1000``, so the test has
    little power against ``d`` near zero at that length. Valid for
    ``d < 1/2``; on a series that may be integrated, difference first
    or use :func:`exact_local_whittle`.

    Args:
        endog: The series.
        f_bandwidth: Fourier frequencies used; ``None`` applies the rule.
        exponent: Exponent of the rule ``m = floor(T ** exponent)``.

    Returns:
        The :class:`LongMemoryEstimate`.

    Raises:
        SpecificationError: If the series is too short or the bandwidth
            is unusable.
        NumericalError: If the periodogram has a zero ordinate.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> est = gph(rng.standard_normal(1000))
        >>> bool(abs(est.d) < 3 * est.se), est.bandwidth
        (True, 31)
    """
    y, m = _validate_semiparametric(endog, f_bandwidth, exponent, minimum=_MIN_LONG_MEMORY_OBS)
    d, se = _gph(y, m)
    return LongMemoryEstimate._record(d, se, method="gph", m=m, nobs=y.shape[0])


def local_whittle(
    endog: npt.ArrayLike, *, f_bandwidth: int | None = None, exponent: float = _WHITTLE_EXPONENT
) -> LongMemoryEstimate:
    """Robinson's (1995) local Whittle estimate of ``d``.

    Minimize the Gaussian semiparametric objective ``log(mean(lambda_j^{2d}
    I_j)) - 2d mean(log lambda_j)`` over the first ``m`` frequencies. The
    estimator is ``N(0, 1/4)`` in ``sqrt(m)`` units for ``-1/2 < d
    3/4``, more efficient than GPH at the same bandwidth; it stays
    consistent up to ``d < 1`` but loses its limit law beyond ``3/4``,
    and the record says so when the estimate lands there. The search
    runs over ``(-1/2, 1)``, so a unit-root series is reported at the
    upper boundary with a pointer to :func:`exact_local_whittle`.

    Args:
        endog: The series.
        f_bandwidth: Fourier frequencies used; ``None`` applies the rule.
        exponent: Exponent of the rule ``m = floor(T ** exponent)``.

    Returns:
        The :class:`LongMemoryEstimate`.

    Raises:
        SpecificationError: If the series is too short or the bandwidth
            is unusable.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> est = local_whittle(rng.standard_normal(1000))
        >>> bool(abs(est.d) < 3 * est.se), est.bandwidth
        (True, 89)
    """
    y, m = _validate_semiparametric(endog, f_bandwidth, exponent, minimum=_MIN_LONG_MEMORY_OBS)
    d, m_eff = local_whittle_d(y, m, exponent, bounds=_LW_BOUNDS)
    se = 1.0 / (2.0 * np.sqrt(m_eff))
    return LongMemoryEstimate._record(d, se, method="local_whittle", m=m_eff, nobs=y.shape[0])


def exact_local_whittle(
    endog: npt.ArrayLike, *, f_bandwidth: int | None = None, exponent: float = _WHITTLE_EXPONENT
) -> LongMemoryEstimate:
    """Shimotsu-Phillips exact local Whittle estimate of ``d``, valid for any ``d``.

    The objective is evaluated on the periodogram of the fractionally
    differenced, mean-corrected series ``(1 - L)^d (y - mu(d))``, which
    removes the ``d < 3/4`` restriction of :func:`local_whittle` and
    keeps the ``N(0, 1/4)`` limit across the stationary and
    nonstationary ranges alike. The mean correction is Shimotsu's
    (2010) weighted estimate, the sample mean for ``d <= 1/2`` and the
    first observation for ``d >= 3/4``, so a random walk is anchored at
    its start. The search runs over ``(-1/2, 2)``. This is the estimator
    for the question ADF cannot answer: whether a series that rejects
    nothing is ``d = 1`` or merely ``d = 0.7``.

    Args:
        endog: The series.
        f_bandwidth: Fourier frequencies used; ``None`` applies the rule.
        exponent: Exponent of the rule ``m = floor(T ** exponent)``.

    Returns:
        The :class:`LongMemoryEstimate`.

    Raises:
        SpecificationError: If the series is too short or the bandwidth
            is unusable.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> walk = np.cumsum(rng.standard_normal(500))
        >>> est = exact_local_whittle(walk)
        >>> bool(abs(est.d - 1.0) < 3 * est.se), bool(est.test(1.0).pvalue > 0.05)
        (True, True)
    """
    y, m = _validate_semiparametric(endog, f_bandwidth, exponent, minimum=_MIN_LONG_MEMORY_OBS)
    d, se = _exact_local_whittle(y, m, bounds=_ELW_BOUNDS)
    return LongMemoryEstimate._record(d, se, method="exact_local_whittle", m=m, nobs=y.shape[0])
