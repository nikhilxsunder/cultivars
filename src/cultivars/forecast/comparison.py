# filepath: /src/cultivars/forecast/comparison.py
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
r"""Comparing forecasters: is the loss gap real, and does it persist?

Diebold-Mariano (1995) asks the plain question -- is the average loss
differential :math:`d_t = L_{A,t} - L_{B,t}` between two forecast series
distinguishable from zero --

.. math::

   \mathrm{DM} = \frac{\bar d}{\sqrt{\hat\sigma^2_{LR} / T}},
   \qquad
   \hat\sigma^2_{LR} = \hat\gamma_0 + 2 \sum_{j=1}^{h-1}
   \Bigl(1 - \frac{j}{h}\Bigr)\hat\gamma_j,

with a variance that respects the serial correlation multi-step forecast
errors mechanically carry, and the Harvey-Leybourne-Newbold small-sample
correction always applied alongside, because evaluation windows in
macroeconomics are short. Giacomini-White (2006) sharpens the question to
a conditional one: not "who was better on average" but "was the gap
predictable from what we knew at the time," which is the version that
matters for choosing a forecaster going forward. Around those two sit
the tests that read a forecast rather than a loss: Clark-West for the
nested case Diebold-Mariano cannot handle, Mincer-Zarnowitz for whether
a forecast could be recalibrated against itself, and encompassing for
whether one forecast already holds what another knows.

Two commitments shape the surface. First, the inputs are aligned
*series*, one value per evaluation origin -- losses for the comparison
tests, point forecasts and outcomes for the forecast tests -- and
producing them is deliberately not this module's job. The rolling-origin
loop belongs in :class:`~cultivars.forecast.backtest.Backtest`, whose
record hands over exactly these series through
:meth:`~cultivars.forecast.backtest.BacktestResult.losses` and its
``point`` and ``realized`` arrays, or in the caller's own driver code;
conflating model selection with model evaluation inside a test is how
look-ahead bias gets laundered, so the alignment is trusted here, never
constructed. Second, two standard abuses are warned against rather than
silently permitted. The comparison tests are about *forecasts*, not
models: comparing nested models' recursive forecasts breaks the
Diebold-Mariano asymptotics (Clark-McCracken territory, named and not
implemented), and :func:`clark_west` is the approximately normal
substitute. And the Giacomini-White framework wants rolling-window
estimation, whose finite memory is what makes its null well-posed.

Layout. :class:`ForecastComparison` runs Diebold-Mariano, its HLN
correction, and Giacomini-White on the default instruments together and
returns a :class:`ForecastComparisonResult`; :func:`giacomini_white`
opens the instrument set and returns a :class:`GiacominiWhiteTest`;
:func:`clark_west`, :func:`mincer_zarnowitz`, and :func:`encompassing`
return their own records, all four of which share the
``_ForecastComparisonTest`` base of ``_internals`` for the origin count,
horizon, and verdict header. The numerics live in ``_core``:
``_validate_aligned_series`` checks the alignment, horizon, and the
``_MIN_COMPARISON_ORIGINS`` floor of eight; ``_bartlett_long_run_variance``
is the HAC variance every statistic divides by; ``_conditional_instruments``
builds the Giacomini-White design with the horizon-aware lag and
``_giacomini_white`` its Wald statistic; and ``_clark_west``,
``_mincer_zarnowitz``, and ``_forecast_encompassing`` are the three
forecast tests.

References:
    Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive
    accuracy. *Journal of Business & Economic Statistics*, 13(3),
    253-263.

    Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality
    of prediction mean squared errors. *International Journal of
    Forecasting*, 13(2), 281-291.

    Giacomini, R., & White, H. (2006). Tests of conditional predictive
    ability. *Econometrica*, 74(6), 1545-1578.

    Clark, T. E., & West, K. D. (2007). Approximately normal tests for
    equal predictive accuracy in nested models. *Journal of
    Econometrics*, 138(1), 291-311.

    West, K. D. (2006). Forecast evaluation. In G. Elliott, C. W. J.
    Granger, & A. Timmermann (Eds.), *Handbook of Economic Forecasting*
    (Vol. 1, pp. 99-134). Elsevier.

Example:
    Two VAR orders backtested on one schedule, compared on their
    one-step squared losses and, since the orders nest, by Clark-West:

    >>> import numpy as np
    >>> from cultivars.forecast.backtest import Backtest
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> y = np.random.default_rng(1).standard_normal((160, 2))
    >>> small = Backtest(y, lambda w: VAR(w, order=1).fit(), start=100).run()
    >>> large = Backtest(y, lambda w: VAR(w, order=3).fit(), start=100).run()
    >>> verdict = ForecastComparison(small.losses(name="y1"), large.losses(name="y1")).compute()
    >>> verdict.nobs, bool(verdict.mean_differential < 0.0)
    (60, True)
    >>> nested = clark_west(small.realized[:, 0, 0], small.point[:, 0, 0], large.point[:, 0, 0])
    >>> nested.nobs, bool(nested.reject())
    (60, False)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import scipy.stats as sst

from .._core import (
    _MIN_COMPARISON_ORIGINS,
    SummaryTable,
    _bartlett_long_run_variance,
    _clark_west,
    _conditional_instruments,
    _forecast_encompassing,
    _giacomini_white,
    _mincer_zarnowitz,
    _validate_aligned_series,
)
from .._internals import _ForecastComparisonTest as ForecastComparisonTest
from .._internals import _SummaryMixin
from ..exceptions import DimensionError, NumericalError, SpecificationError

__all__ = [
    "ClarkWestTest",
    "EncompassingTest",
    "ForecastComparison",
    "ForecastComparisonResult",
    "GiacominiWhiteTest",
    "MincerZarnowitzTest",
    "clark_west",
    "encompassing",
    "giacomini_white",
    "mincer_zarnowitz",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ClarkWestTest(ForecastComparisonTest):
    r"""Verdict of the Clark-West (2007) test that a nesting model forecasts better.

    The one-sided alternative is that the larger model improves on the
    smaller; under the null the smaller model is true and the larger one's
    extra parameters only add estimation noise, which the statistic
    removes before testing. With :math:`e_{r,t}` and :math:`e_{u,t}` the
    restricted and unrestricted forecast errors and :math:`\hat y_{r,t}`,
    :math:`\hat y_{u,t}` the forecasts, the adjusted loss differential

    .. math::

       \hat f_t = e_{r,t}^2 - \bigl[e_{u,t}^2 - (\hat y_{r,t} - \hat y_{u,t})^2\bigr]

    adds back the squared forecast gap that the larger model's noise
    costs it under the null, and the statistic is the :math:`t`-ratio of
    its mean against a Bartlett long-run variance through :math:`h - 1`
    lags, read against the upper tail of the standard normal. A rejection
    says the added structure has predictive content; a non-rejection says
    it has not shown any on this window.

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

    Note:
        The raw MSPE difference can favor the *smaller* model while the
        test still rejects, and often does: under the null the larger
        model's MSPE is inflated by parameter noise, so an unadjusted
        Diebold-Mariano comparison of nested forecasts is biased toward
        the nested model and its limit distribution is not normal
        (Clark & McCracken, 2001). The adjustment is what makes the
        normal reference usable.

    See Also:
        * :func:`clark_west` -- the producer.
        * :class:`ForecastComparison` -- the Diebold-Mariano test, for
          non-nested forecasts.
        * :class:`EncompassingTest` -- whether one forecast already
          contains the other's information.

    References:
        Clark, T. E., & West, K. D. (2007). Approximately normal tests
        for equal predictive accuracy in nested models. *Journal of
        Econometrics*, 138(1), 291-311.

        Clark, T. E., & McCracken, M. W. (2001). Tests of equal forecast
        accuracy and encompassing for nested models. *Journal of
        Econometrics*, 105(1), 85-110.

    Example:
        A predictor with real content against the no-change forecast it
        nests:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal(200)
        >>> y = 0.5 * x + rng.standard_normal(200)
        >>> verdict = clark_west(y, np.zeros(200), 0.5 * x)
        >>> verdict
        ClarkWestTest(statistic=5.0456, pvalue=2.26e-07, horizon=1, nobs=200)
        >>> bool(verdict.reject()), round(verdict.adjusted_differential, 3)
        (True, 0.397)
        >>> round(verdict.mspe_restricted, 3), round(verdict.mspe_unrestricted, 3)
        (1.225, 1.059)
    """

    adjusted_differential: float
    """Mean of :math:`\\hat f_t`, the MSPE gap with the larger model's noise added back.

    Positive favors the larger model.
    """
    mspe_restricted: float
    """Mean squared prediction error of the nested (smaller) model."""
    mspe_unrestricted: float
    """Mean squared prediction error of the nesting (larger) model."""

    def _summary_table(self) -> SummaryTable:
        """Render as a table.

        One row with the statistic and p-value under the family header,
        and notes giving the adjusted differential, the raw MSPEs, and
        the reference distribution.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``summary()``,
            ``str()``, and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal(200)
            >>> y = 0.5 * x + rng.standard_normal(200)
            >>> table = clark_west(y, np.zeros(200), 0.5 * x)._summary_table()
            >>> table.metadata[2], table.rows[0][0]
            (('Verdict', 'larger model improves'), 'Clark-West adjusted MSPE')
        """
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
        """Represent the test as a string.

        The statistic, p-value, horizon, and origin count; the MSPEs
        and the differential are left to ``summary()``.

        Returns:
            The one-line representation.
        """
        return (
            f"ClarkWestTest(statistic={self.statistic:.4f}, pvalue={self.pvalue:.4g}, "
            f"horizon={self.horizon}, nobs={self.nobs})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class MincerZarnowitzTest(ForecastComparisonTest):
    r"""Verdict of the Mincer-Zarnowitz efficiency regression ``y = a + b f``.

    An unbiased, efficient point forecast has intercept zero and slope
    one in

    .. math::

       y_t = a + b\, f_t + u_t,
       \qquad H_0 : (a, b) = (0, 1),

    because a forecast that is the conditional mean cannot be improved
    by any linear transformation of itself. A slope below one says the
    forecast overreacts -- shrinking it toward its mean would help -- and
    above one that it underreacts; an intercept away from zero is bias.
    The joint restriction is tested with a Wald statistic on a HAC
    covariance, since multi-step errors overlap, and :math:`W / 2` is
    referred to :math:`F(2, T - 2)` for the small-sample reading.

    Attributes:
        intercept: Estimated ``a``.
        slope: Estimated ``b``.
        statistic: Wald statistic for ``(a, b) = (0, 1)``, chi-squared scale.
        pvalue: Upper-tail p-value of ``statistic / 2`` under ``F(2, T - 2)``.
        r_squared: Fit of the regression, the forecast's explanatory
            share of the outcome.
        horizon: Forecast horizon behind the series.
        nobs: Evaluation origins.

    Note:
        The regression is a test of the forecast against itself, not
        against a rival: it can reject for a forecast that beats every
        alternative and fail to reject for a useless one whose errors
        happen to be uncorrelated with it. Read it beside a comparison.
        At multi-step horizons the truncated Bartlett kernel understates
        the long-run variance and the test over-rejects, roughly 13% at
        nominal 5% for a four-step horizon on 100 origins; the summary
        says so whenever ``horizon > 1``.

    See Also:
        * :func:`mincer_zarnowitz` -- the producer.
        * :class:`~cultivars.forecast.calibration.Calibration` -- the
          density-forecast analogue: are the outcomes where the forecasts
          said?
        * :class:`EncompassingTest` -- the combination weight a rival
          would earn, the other reading of "could be improved".

    References:
        Mincer, J. A., & Zarnowitz, V. (1969). The evaluation of economic
        forecasts. In J. A. Mincer (Ed.), *Economic Forecasts and
        Expectations* (pp. 3-46). NBER.

    Example:
        A forecast that is the conditional mean passes; the same
        forecast scaled by three overreacts and fails on its slope:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal(200)
        >>> y = 0.5 * x + rng.standard_normal(200)
        >>> efficient = mincer_zarnowitz(y, 0.5 * x)
        >>> round(efficient.slope, 2), bool(efficient.reject())
        (0.86, False)
        >>> overreacting = mincer_zarnowitz(y, 1.5 * x)
        >>> round(overreacting.slope, 2), bool(overreacting.reject())
        (0.29, True)
    """

    intercept: float
    """Estimated :math:`a`; zero under the null, otherwise the forecast's bias."""
    slope: float
    """Estimated :math:`b`; one under the null, below one for an overreacting forecast."""
    r_squared: float
    """Coefficient of determination of the regression, the forecast's share of the outcome."""

    def _summary_table(self) -> SummaryTable:
        """Render as a table.

        The two estimates and the Wald row under the family header; the
        notes give the fit, the reading of the slope, the HAC details,
        and the multi-step over-rejection warning when it applies.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``summary()``,
            ``str()``, and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal(200)
            >>> y = 0.5 * x + rng.standard_normal(200)
            >>> table = mincer_zarnowitz(y, 0.5 * x)._summary_table()
            >>> table.metadata[2]
            ('Verdict', 'efficiency not rejected')
            >>> [row[0] for row in table.rows]
            ['intercept (0 under H0)', 'slope (1 under H0)', 'Wald, F(2, T - 2) reference']
        """
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
        """Represent the test as a string.

        The two estimates, the Wald statistic, its p-value, and the
        origin count.

        Returns:
            The one-line representation.
        """
        return (
            f"MincerZarnowitzTest(intercept={self.intercept:.4f}, slope={self.slope:.4f}, "
            f"statistic={self.statistic:.4f}, pvalue={self.pvalue:.4g}, nobs={self.nobs})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class EncompassingTest(ForecastComparisonTest):
    r"""Verdict of the forecast encompassing test: does A already contain what B knows?

    Forecast A encompasses B when the optimal linear combination of the
    two,

    .. math::

       y_t = (1 - \lambda)\, f_{A,t} + \lambda\, f_{B,t} + u_t,

    puts zero weight on B. Since :math:`\lambda` is the least-squares
    coefficient of :math:`e_{A,t}` on :math:`e_{A,t} - e_{B,t}`, the null
    :math:`\lambda = 0` is :math:`\mathbb{E}[e_{A,t}(e_{A,t} - e_{B,t})] =
    0`, and the test is Diebold-Mariano machinery on the series
    :math:`d_t = e_{A,t}(e_{A,t} - e_{B,t})` with the
    Harvey-Leybourne-Newbold correction and a one-sided :math:`t`
    reference. The one-sided alternative is that B carries information A
    lacks; a rejection says combining would help, and the reported
    weight is the least-squares share B would get.

    Attributes:
        statistic: The HLN-corrected statistic on ``e_A (e_A - e_B)``.
        pvalue: Its upper-tail ``t`` p-value.
        weight: Least-squares combination weight on B; zero under the null.
        horizon: Forecast horizon behind the series.
        nobs: Evaluation origins.

    Note:
        The test is directional: "A encompasses B" and "B encompasses A"
        are two calls with the arguments swapped, and both can fail to
        reject on a short window, which says the window cannot tell,
        not that the forecasts are equivalent. The weight is reported
        unclipped, so a value above one or below zero is possible and
        says the least-squares combination extrapolates beyond the two
        forecasts.

    See Also:
        * :func:`encompassing` -- the producer.
        * :class:`ForecastComparison` -- whether the losses differ at all.
        * :func:`~cultivars.bayes.combination.stacking` -- the weights a
          pool of density forecasters earns, the same idea across many.

    References:
        Harvey, D. I., Leybourne, S. J., & Newbold, P. (1998). Tests for
        forecast encompassing. *Journal of Business & Economic
        Statistics*, 16(2), 254-259.

        Chong, Y. Y., & Hendry, D. F. (1986). Econometric evaluation of
        linear macro-economic models. *Review of Economic Studies*, 53(4),
        671-690.

    Example:
        A zero forecast does not encompass an informative one; swapped,
        the informative forecast encompasses the zero:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal(200)
        >>> y = 0.5 * x + rng.standard_normal(200)
        >>> forward = encompassing(y, np.zeros(200), 0.5 * x)
        >>> forward
        EncompassingTest(statistic=5.0330, pvalue=5.387e-07, weight=0.859, nobs=200)
        >>> reverse = encompassing(y, 0.5 * x, np.zeros(200))
        >>> bool(forward.reject()), bool(reverse.reject()), round(reverse.weight, 3)
        (True, False, 0.141)
    """

    weight: float
    """Least-squares weight :math:`\\lambda` on forecast B; zero under the null.

    Unclipped, so it can lie outside ``[0, 1]``.
    """

    def _summary_table(self) -> SummaryTable:
        """Render as a table.

        One row with the statistic and p-value under the family header,
        and notes giving the combination weight and the reference.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``summary()``,
            ``str()``, and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> x = rng.standard_normal(200)
            >>> y = 0.5 * x + rng.standard_normal(200)
            >>> table = encompassing(y, np.zeros(200), 0.5 * x)._summary_table()
            >>> table.metadata[2], table.rows[0][0]
            (('Verdict', 'B adds information'), 'HLN encompassing')
        """
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
        """Represent the test as a string.

        The statistic, p-value, combination weight, and origin count.

        Returns:
            The one-line representation.
        """
        return (
            f"EncompassingTest(statistic={self.statistic:.4f}, pvalue={self.pvalue:.4g}, "
            f"weight={self.weight:.3f}, nobs={self.nobs})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class GiacominiWhiteTest(ForecastComparisonTest):
    r"""Verdict of the Giacomini-White (2006) test of conditional predictive ability.

    The null is not "the two forecasters lose the same on average" but
    "nothing known at the origin predicts which will lose more,"

    .. math::

       H_0 : \mathbb{E}\bigl[d_t \mid \mathcal{F}_{t-h}\bigr] = 0
       \quad\Longrightarrow\quad
       \mathbb{E}\bigl[h_{t}\, d_t\bigr] = 0,

    with :math:`d_t = L_{A,t} - L_{B,t}` the loss differential and
    :math:`h_t` a vector of instruments measurable at the origin. The
    statistic is the Wald form :math:`T\, \bar z^\top \hat\Omega^{-1}
    \bar z` on :math:`z_t = h_t d_t`, chi-squared on the number of
    instruments, so a rejection is a statement about a usable rule: the
    regression of the differential on the instruments, whose fitted sign
    at a fresh origin says which forecaster to run next. The constant-only
    case is the unconditional test; with the differential lagged by the
    horizon as the default instrument, a rejection with an insignificant
    mean says the gap is there but switches sides, which the average
    hides.

    Attributes:
        statistic: The Wald statistic ``T zbar' Omega^{-1} zbar``.
        pvalue: Its upper-tail chi-squared p-value on ``df`` degrees of
            freedom.
        df: Number of instruments, constant included.
        coefficients: OLS coefficients of the differential on the
            instruments, in the order ``constant, lagged differentials,
            caller instruments``; the decision rule.
        instruments: What the instruments are, in words.
        share_a: Fraction of retained origins at which the fitted rule
            favors the first forecaster (fitted differential negative).
        mean_differential: Average of ``losses_a - losses_b`` over the
            retained origins; negative favors the first forecaster.
        horizon: Forecast horizon behind the series.
        nobs: Origins retained after the instruments' lags.

    Note:
        The framework is built for rolling-window estimation: with a
        window of fixed length the forecasting *method* is what the null
        describes, and its finite memory keeps that null well-posed. Under
        recursive estimation the parameters converge and the null drifts
        as the window grows. ``nobs`` is smaller than the series length
        because the lagged-differential instruments consume the first
        ``horizon + lags - 1`` origins. At multi-step horizons the
        truncated Bartlett kernel understates the moment covariance and
        the test over-rejects mildly, about 7-9% at nominal 5% for a
        four-step horizon.

    See Also:
        * :func:`giacomini_white` -- the producer, with the open
          instrument set.
        * :class:`ForecastComparison` -- the unconditional question, with
          the small-sample correction, and this test on the default
          instruments as its third row.

    References:
        Giacomini, R., & White, H. (2006). Tests of conditional predictive
        ability. *Econometrica*, 74(6), 1545-1578.

    Example:
        A loss gap that flips sign every 25 origins averages to nearly
        nothing and the unconditional test misses it; a regime indicator
        the forecaster could have known makes it predictable:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> regime = np.where((np.arange(200) // 25) % 2 == 0, 1.0, -1.0)
        >>> losses_a = rng.standard_normal(200) ** 2 + 0.4 * regime
        >>> losses_b = rng.standard_normal(200) ** 2
        >>> bool(ForecastComparison(losses_a, losses_b).compute().hln_pvalue > 0.3)
        True
        >>> verdict = giacomini_white(losses_a, losses_b, instruments=regime)
        >>> verdict
        GiacominiWhiteTest(statistic=8.9204, pvalue=0.03037, df=3, horizon=1, nobs=199)
        >>> verdict.instruments, verdict.coefficients.round(2), bool(verdict.reject())
        ('constant, d[t-1], z1', array([-0.15, -0.09,  0.34]), True)
    """

    df: int
    """Instruments in the test, constant included; the chi-squared degrees of freedom."""
    coefficients: npt.NDArray[np.float64] = field(repr=False)
    """OLS coefficients of the differential on the instruments, the decision rule.

    Ordered ``constant, lagged differentials, caller instruments``. Kept
    out of the repr.
    """
    instruments: str
    """The instrument set in words, such as ``"constant, d[t-1], z1"``."""
    share_a: float
    """Share of retained origins at which the fitted rule favors forecaster A."""
    mean_differential: float
    """Average of ``losses_a - losses_b`` over the retained origins; negative favors A."""

    def _summary_table(self) -> SummaryTable:
        """Render as a table.

        One row with the Wald statistic, its degrees of freedom, and the
        p-value under the family header; the notes give the instruments,
        the decision rule, the share of origins favoring A, the mean
        differential, and the covariance and estimation-scheme caveats.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``summary()``,
            ``str()``, and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> losses_a, losses_b = rng.standard_normal(150) ** 2, rng.standard_normal(150) ** 2
            >>> table = giacomini_white(losses_a, losses_b)._summary_table()
            >>> table.columns, table.rows[0][0], table.rows[0][2]
            (('test', 'statistic', 'df', 'p-value'), 'Giacomini-White Wald', '2')
        """
        verdict = "gap is predictable" if self.reject() else "no conditional predictability shown"
        rows = (
            (
                "Giacomini-White Wald",
                f"{self.statistic:.4f}",
                str(self.df),
                f"{self.pvalue:.4f}",
            ),
        )
        rule = ", ".join(f"{c:+.4f}" for c in self.coefficients)
        notes = (
            f"Instruments: {self.instruments}. Decision rule coefficients ({rule}); the "
            f"fitted differential favors forecaster A at {100 * self.share_a:.1f}% of "
            f"origins. Mean differential {self.mean_differential:+.5f} (negative favors A).",
            "Chi-squared reference; the moment covariance is the uncentered sample "
            "covariance at one step and Bartlett HAC through horizon - 1 = "
            f"{self.horizon - 1} lags beyond, which over-rejects mildly at multi-step "
            "horizons. The framework wants rolling-window estimation behind the losses, "
            "whose finite memory keeps the null well-posed.",
        )
        return self._frame(
            title="Giacomini-White Conditional Predictive Ability",
            verdict=verdict,
            columns=("test", "statistic", "df", "p-value"),
            rows=rows,
            notes=notes,
        )

    def __repr__(self) -> str:
        """Represent the test as a string.

        The statistic, p-value, degrees of freedom, horizon, and retained
        origin count; the rule and instruments are left to ``summary()``.

        Returns:
            The one-line representation.
        """
        return (
            f"GiacominiWhiteTest(statistic={self.statistic:.4f}, pvalue={self.pvalue:.4g}, "
            f"df={self.df}, horizon={self.horizon}, nobs={self.nobs})"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ForecastComparisonResult(_SummaryMixin):
    r"""The verdict on a loss differential, unconditional and conditional.

    Losses are negatively oriented, so a *negative* mean differential
    favors the first forecaster. Three readings of the same series
    :math:`d_t = L_{A,t} - L_{B,t}` travel together. The Diebold-Mariano
    statistic

    .. math::

       \mathrm{DM} = \frac{\bar d}{\sqrt{\hat\sigma^2_{LR} / T}},
       \qquad
       \hat\sigma^2_{LR} = \hat\gamma_0 + 2 \sum_{j=1}^{h-1}
       \Bigl(1 - \frac{j}{h}\Bigr) \hat\gamma_j,

    with the Bartlett long-run variance through :math:`h - 1` lags, is
    read against the standard normal; the Harvey-Leybourne-Newbold
    version scales it by :math:`\sqrt{(T + 1 - 2h + h(h-1)/T)/T}` and
    reads it against :math:`t_{T-1}`, which corrects the over-rejection
    of the normal reference in short windows; and the Giacomini-White
    statistic asks whether :math:`d_t` was predictable from a constant
    and :math:`d_{t-h}`, the conditional question.

    Attributes:
        mean_differential: Average of ``losses_a - losses_b``.
        dm_statistic: Diebold-Mariano statistic under the standard normal
            null.
        dm_pvalue: Its two-sided p-value.
        hln_statistic: The Harvey-Leybourne-Newbold corrected statistic,
            referred to a ``t`` distribution -- the number to trust in the
            short evaluation windows macroeconomics actually has.
        hln_pvalue: Its two-sided p-value.
        gw_statistic: Giacomini-White conditional statistic with a
            constant and the differential lagged by the horizon -- the
            most recent one known at the origin -- as instruments,
            chi-squared with two degrees of freedom under the null of no
            conditional predictability; :func:`giacomini_white` opens
            the instrument set.
        gw_pvalue: Its upper-tail p-value.
        horizon: The forecast horizon the losses were produced at, which
            sets the variance's serial-correlation window.
        nobs: Evaluation origins.

    Note:
        These are tests about forecasts, not models. Two nested models'
        recursive forecasts break the asymptotics -- the larger model's
        extra parameters inflate its loss under the null and the
        differential's limit is not normal -- which is Clark-McCracken
        territory, named and not implemented; :func:`clark_west` is the
        approximately normal substitute for that case. The two-sided
        p-values answer "do the forecasters differ"; the sign of
        ``mean_differential`` says which way.

    See Also:
        * :class:`ForecastComparison` -- the producer.
        * :class:`GiacominiWhiteTest` -- the conditional test with the
          full instrument set and the decision rule.
        * :class:`ClarkWestTest` -- for nested forecasts.
        * :func:`~cultivars.forecast.confidence_set.model_confidence_set`
          -- the same question for more than two forecasters.

    References:
        Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive
        accuracy. *Journal of Business & Economic Statistics*, 13(3),
        253-263.

        Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the
        equality of prediction mean squared errors. *International
        Journal of Forecasting*, 13(2), 281-291.

        Giacomini, R., & White, H. (2006). Tests of conditional predictive
        ability. *Econometrica*, 74(6), 1545-1578.

    Example:
        Two loss series sharing a common component, the second worse by
        0.4 on average:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> shared = rng.standard_normal(120) ** 2
        >>> a = shared + 0.05 * rng.standard_normal(120)
        >>> b = shared + 0.4 + 0.05 * rng.standard_normal(120)
        >>> verdict = ForecastComparison(a, b).compute()
        >>> round(verdict.mean_differential, 3), verdict.nobs, verdict.horizon
        (-0.404, 120, 1)
        >>> round(verdict.dm_statistic, 1), round(verdict.hln_statistic, 1)
        (-58.0, -57.7)
        >>> bool(verdict.hln_pvalue < 1e-6), bool(verdict.gw_pvalue < 1e-6)
        (True, True)
    """

    mean_differential: float
    """Average of ``losses_a - losses_b``; negative favors the first forecaster."""
    dm_statistic: float
    """Diebold-Mariano statistic, the mean differential over its long-run standard error."""
    dm_pvalue: float
    """Two-sided standard normal p-value of ``dm_statistic``."""
    hln_statistic: float
    """``dm_statistic`` scaled by the Harvey-Leybourne-Newbold small-sample factor."""
    hln_pvalue: float
    """Two-sided :math:`t_{T-1}` p-value of ``hln_statistic``; the one to read in short windows."""
    gw_statistic: float
    """Giacomini-White Wald statistic on a constant and :math:`d_{t-h}`; :math:`\\chi^2_2`."""
    gw_pvalue: float
    """Upper-tail chi-squared p-value of ``gw_statistic``."""
    horizon: int
    """Forecast horizon behind the losses, :math:`h`; the long-run variance uses ``h - 1`` lags."""
    nobs: int
    """Evaluation origins, :math:`T`."""

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary: the three tests as rows.

        Under a header with the origin count, horizon, and mean
        differential; the notes read the sign, say which row to trust
        in short windows, state what the conditional row asks, and
        restate the nested-model caveat.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``summary()``,
            ``str()``, and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> a, b = rng.standard_normal(100) ** 2, rng.standard_normal(100) ** 2
            >>> table = ForecastComparison(a, b).compute()._summary_table()
            >>> [row[0] for row in table.rows]
            ['Diebold-Mariano', 'Harvey-Leybourne-Newbold', 'Giacomini-White (conditional)']
        """
        favored = "A" if self.mean_differential < 0.0 else "B"
        rows = (
            ("Diebold-Mariano", f"{self.dm_statistic:.4f}", f"{self.dm_pvalue:.4f}"),
            (
                "Harvey-Leybourne-Newbold",
                f"{self.hln_statistic:.4f}",
                f"{self.hln_pvalue:.4f}",
            ),
            (
                "Giacomini-White (conditional)",
                f"{self.gw_statistic:.4f}",
                f"{self.gw_pvalue:.4f}",
            ),
        )
        notes = [
            f"Mean loss differential {self.mean_differential:+.5f}: the "
            f"sign favors forecaster {favored} (losses are negatively "
            "oriented).",
            "The HLN row is the DM statistic with the small-sample "
            "correction, referred to a t distribution; in short evaluation "
            "windows it is the number to trust.",
            "Giacomini-White asks whether the gap was *predictable* from a "
            "constant and the differential lagged by the horizon; its "
            "framework wants rolling-window estimation behind the losses.",
            "These are tests about forecasts, not models: nested-model "
            "comparisons need the Clark-McCracken corrections, which are "
            "deliberately not implemented here.",
        ]
        return SummaryTable(
            title="Forecast Comparison",
            metadata=(
                ("Origins", f"{self.nobs}"),
                ("Horizon", f"{self.horizon}"),
                ("Mean differential", f"{self.mean_differential:+.5f}"),
            ),
            columns=("test", "statistic", "p-value"),
            rows=rows,
            notes=tuple(notes),
        )


class ForecastComparison:
    r"""Test a loss differential between two forecasters.

    Takes two aligned loss series, one value per evaluation origin, and
    answers the unconditional question -- is the average of
    :math:`d_t = L_{A,t} - L_{B,t}` distinguishable from zero, with a
    variance that respects the serial correlation multi-step errors
    carry -- twice, as Diebold-Mariano and with the
    Harvey-Leybourne-Newbold correction, and the conditional question
    once, as Giacomini-White on the default instruments. The losses can
    be anything negatively oriented: squared errors, CRPS, log scores,
    pinball losses, so long as both series were produced by the same
    loss on the same origins.

    Args:
        losses_a: ``(T,)`` losses of the first forecaster, one per
            evaluation origin, negatively oriented.
        losses_b: ``(T,)`` losses of the second, aligned origin by origin
            -- an alignment this object cannot verify and wholly depends
            on.

    Attributes:
        _losses_a: The validated ``(T,)`` first loss series.
        _losses_b: The validated ``(T,)`` second loss series.

    Raises:
        DimensionError: If the series' shapes disagree.
        SpecificationError: If the window is too short, or the series are
            numerically identical and there is no differential to test.
        NumericalError: If the losses are not finite.

    Note:
        Producing the loss series is deliberately the caller's job: the
        rolling-origin loop belongs in driver code or in
        :class:`~cultivars.forecast.backtest.Backtest`, whose record's
        :meth:`~cultivars.forecast.backtest.BacktestResult.losses` gives
        exactly the aligned series this class takes, because conflating
        model selection with model evaluation inside a test is how
        look-ahead bias gets laundered. Fewer than eight origins are
        refused: a differential's long-run variance on so short a window
        has neither power nor reliable size.

    See Also:
        * :class:`ForecastComparisonResult` -- the record
          :meth:`compute` returns.
        * :func:`giacomini_white` -- the conditional test with an open
          instrument set.
        * :func:`clark_west` -- for nested forecasts.
        * :func:`encompassing` -- whether one forecast already contains
          the other's information.

    References:
        Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive
        accuracy. *Journal of Business & Economic Statistics*, 13(3),
        253-263.

        Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the
        equality of prediction mean squared errors. *International
        Journal of Forecasting*, 13(2), 281-291.

    Example:
        Synthetic loss series, then the squared-loss columns of two VAR
        backtests on one schedule:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> shared = rng.standard_normal(120) ** 2
        >>> a = shared + 0.05 * rng.standard_normal(120)
        >>> b = shared + 0.4 + 0.05 * rng.standard_normal(120)
        >>> verdict = ForecastComparison(a, b).compute()
        >>> bool(verdict.mean_differential < 0.0)
        True
        >>> bool(verdict.hln_pvalue < 0.01)
        True
        >>> from cultivars.forecast.backtest import Backtest
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> y = np.random.default_rng(1).standard_normal((160, 2))
        >>> one = Backtest(y, lambda w: VAR(w, order=1).fit(), horizons=2, start=100).run()
        >>> two = Backtest(y, lambda w: VAR(w, order=2).fit(), horizons=2, start=100).run()
        >>> first, second = one.losses(horizon=2, name="y1"), two.losses(horizon=2, name="y1")
        >>> ForecastComparison(first, second).compute(horizon=2).nobs
        59
    """

    __slots__ = ("_losses_a", "_losses_b")

    def __init__(self, losses_a: npt.ArrayLike, losses_b: npt.ArrayLike) -> None:
        """Validate and align the two loss series.

        Both are coerced to ``float64`` and flattened, then checked to
        share a length of at least ``_MIN_COMPARISON_ORIGINS`` and to be
        finite. Whether they differ at all is checked in :meth:`compute`,
        where the differential is formed.

        Args:
            losses_a: The first loss series.
            losses_b: The second, aligned origin by origin.

        Raises:
            DimensionError: If the lengths disagree.
            SpecificationError: If fewer than eight origins are given.
            NumericalError: If a loss is not finite.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> short = rng.standard_normal(5), rng.standard_normal(5)
            >>> ForecastComparison(*short)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: a comparison over 5 origins has no power ...
        """
        first = np.asarray(losses_a, dtype=np.float64).ravel()
        second = np.asarray(losses_b, dtype=np.float64).ravel()
        if first.shape != second.shape:
            raise DimensionError(
                f"the loss series must align origin by origin; got shapes "
                f"{first.shape} and {second.shape}."
            )
        if first.shape[0] < _MIN_COMPARISON_ORIGINS:
            raise SpecificationError(
                f"a comparison over {first.shape[0]} origins has no power "
                f"and unreliable size; provide at least {_MIN_COMPARISON_ORIGINS}."
            )
        if not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
            raise NumericalError("losses must be finite.")
        self._losses_a = first
        self._losses_b = second

    def compute(self, *, horizon: int = 1) -> ForecastComparisonResult:
        r"""Run the unconditional and conditional tests.

        Forms :math:`d_t`, refuses a numerically constant differential,
        and computes the Bartlett long-run variance through ``horizon -
        1`` lags for the Diebold-Mariano ratio; scales that ratio by the
        HLN factor :math:`\sqrt{(T + 1 - 2h + h(h-1)/T)/T}` for the
        :math:`t_{T-1}` reading; and builds the Giacomini-White
        instruments -- a constant and :math:`d_{t-h}` -- for the
        conditional Wald statistic. Deterministic: nothing here draws.

        Args:
            horizon: The forecast horizon behind the losses; the
                differential's variance sums autocovariances through
                ``horizon - 1`` lags with Bartlett weights, since an
                ``h``-step error is mechanically an MA(``h - 1``).

        Returns:
            The :class:`ForecastComparisonResult`.

        Raises:
            SpecificationError: If the horizon is not positive or exceeds
                what the window can support, or the two series are
                numerically identical.
            NumericalError: If the conditional moment matrix degenerates.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> a, b = rng.standard_normal(100) ** 2, rng.standard_normal(100) ** 2
            >>> verdict = ForecastComparison(a, b).compute(horizon=4)
            >>> verdict.horizon, verdict.nobs
            (4, 100)
            >>> bool(abs(verdict.hln_statistic) < abs(verdict.dm_statistic))
            True
            >>> ForecastComparison(a, a).compute()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: the two loss series are numerically ...
        """
        if horizon < 1:
            raise SpecificationError(f"horizon must be at least 1; got {horizon}.")
        differential = self._losses_a - self._losses_b
        count = differential.shape[0]
        if horizon >= count // 2:
            raise SpecificationError(
                f"a horizon of {horizon} needs more than {2 * horizon} "
                f"evaluation origins; got {count}."
            )
        centered = differential - differential.mean()
        if float(np.abs(centered).max()) <= 1e-14 * max(float(np.abs(differential).max()), 1.0):
            raise SpecificationError(
                "the two loss series are numerically identical; there is no differential to test."
            )
        variance = _bartlett_long_run_variance(centered, horizon)
        mean = float(differential.mean())
        spread = float(np.sqrt(variance / count))
        dm = mean / spread
        dm_pvalue = 2.0 * float(sst.norm.sf(abs(dm)))
        adjust = float(np.sqrt((count + 1 - 2 * horizon + horizon * (horizon - 1) / count) / count))
        hln = dm * adjust
        hln_pvalue = 2.0 * float(sst.t.sf(abs(hln), count - 1))
        target, design = _conditional_instruments(differential, None, horizon=horizon, lags=1)
        gw, gw_pvalue, _ = _giacomini_white(target, design, horizon=horizon)
        return ForecastComparisonResult(
            mean_differential=mean,
            dm_statistic=dm,
            dm_pvalue=dm_pvalue,
            hln_statistic=hln,
            hln_pvalue=hln_pvalue,
            gw_statistic=gw,
            gw_pvalue=gw_pvalue,
            horizon=int(horizon),
            nobs=count,
        )


def clark_west(
    realized: npt.ArrayLike,
    restricted: npt.ArrayLike,
    unrestricted: npt.ArrayLike,
    *,
    horizon: int = 1,
) -> ClarkWestTest:
    r"""Clark-West (2007) test that a nesting model forecasts better than the model it nests.

    Under the null that the smaller model is true, the larger one's extra
    parameters are estimated noise, so its mean squared prediction error
    is *expected* to exceed the smaller one's and Diebold-Mariano is biased
    against it. Clark and West subtract the noise term ``(f_r - f_u)**2``
    from the loss differential,

    .. math::

       \hat f_t = e_{r,t}^2 - e_{u,t}^2 + (\hat y_{r,t} - \hat y_{u,t})^2,
       \qquad
       \mathrm{CW} = \frac{\bar{\hat f}}{\sqrt{\hat\sigma^2_{LR}(\hat f) / T}},

    and test what remains with a one-sided normal reference, which is
    approximately correctly sized in the recursive and rolling schemes a
    backtest runs.

    Args:
        realized: ``(T,)`` outcomes.
        restricted: ``(T,)`` point forecasts of the nested model.
        unrestricted: ``(T,)`` point forecasts of the nesting model,
            aligned origin by origin -- ``BacktestResult.point[:, h - 1,
            j]`` from two backtests on the same schedule.
        horizon: The forecast horizon behind the series; sets the
            long-run variance's Bartlett window.

    Returns:
        The :class:`ClarkWestTest`.

    Raises:
        DimensionError: If the series do not align.
        SpecificationError: If there are too few origins or the horizon
            is not positive or too long for the window.
        NumericalError: If a series is not finite.

    Note:
        Nesting is the caller's claim; the function cannot check that the
        restricted model is a special case of the unrestricted one, and
        on two non-nested forecasts the adjustment simply rewards the
        forecast that differs more from the other. For non-nested
        forecasts use :class:`ForecastComparison`. The test takes point
        forecasts and outcomes rather than losses because the adjustment
        needs the forecasts themselves.

    See Also:
        * :class:`ClarkWestTest` -- the record, with the raw MSPEs beside
          the adjusted differential.
        * :class:`ForecastComparison` -- Diebold-Mariano, for non-nested
          forecasts.
        * :class:`~cultivars.forecast.backtest.BacktestResult` -- supplies
          aligned ``point`` and ``realized`` arrays.

    References:
        Clark, T. E., & West, K. D. (2007). Approximately normal tests
        for equal predictive accuracy in nested models. *Journal of
        Econometrics*, 138(1), 291-311.

    Example:
        A predictor with real content against the no-change forecast it
        nests; the raw MSPEs favor it too, but on a null predictor they
        would not, and the adjustment is what keeps the test honest:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal(200)
        >>> y = 0.5 * x + rng.standard_normal(200)
        >>> verdict = clark_west(y, np.zeros(200), 0.5 * x)
        >>> bool(verdict.pvalue < 0.01)
        True
        >>> verdict.nobs, verdict.horizon
        (200, 1)
        >>> bool(verdict.mspe_unrestricted < verdict.mspe_restricted)
        True
        >>> clark_west(y, np.zeros(200), 0.5 * x, horizon=150)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: a horizon of 150 needs more than 300 ...
    """
    series, count = _validate_aligned_series(
        realized,
        restricted,
        unrestricted,
        horizon=horizon,
        minimum=_MIN_COMPARISON_ORIGINS,
        labels="realized, restricted, and unrestricted",
    )
    statistic, pvalue, adjusted = _clark_west(series[0], series[1], series[2], horizon=horizon)
    return ClarkWestTest(
        statistic=statistic,
        pvalue=pvalue,
        adjusted_differential=adjusted,
        mspe_restricted=float(((series[0] - series[1]) ** 2).mean()),
        mspe_unrestricted=float(((series[0] - series[2]) ** 2).mean()),
        horizon=int(horizon),
        nobs=int(count),
    )


def mincer_zarnowitz(
    realized: npt.ArrayLike, forecast: npt.ArrayLike, *, horizon: int = 1
) -> MincerZarnowitzTest:
    r"""Mincer-Zarnowitz efficiency regression of outcomes on point forecasts.

    .. math::

       y_t = a + b\, f_t + u_t,
       \qquad H_0 : (a, b) = (0, 1),

    the forecast is unbiased and cannot be improved by a linear
    recalibration of itself. Multi-step errors overlap, so the Wald
    covariance is HAC with Bartlett weights through ``horizon - 1`` lags;
    at those horizons the test over-rejects in short windows (about 13%
    at nominal 5% for four steps on 100 origins), and the record says so.

    Args:
        realized: ``(T,)`` outcomes.
        forecast: ``(T,)`` point forecasts, aligned origin by origin --
            ``BacktestResult.point[:, h - 1, j]`` against
            ``BacktestResult.realized[:, h - 1, j]``.
        horizon: The forecast horizon behind the series.

    Returns:
        The :class:`MincerZarnowitzTest`.

    Raises:
        DimensionError: If the series do not align.
        SpecificationError: If there are too few origins or the horizon
            is unusable.
        NumericalError: If a series is not finite or the forecasts are
            constant.

    Note:
        A constant forecast has no slope to estimate and is refused as
        singular rather than reported as inefficient. The regression
        says nothing about a rival; a forecast can pass it and still
        lose every comparison, and fail it while winning them all, so
        read the verdict as "could this forecast be recalibrated" and not
        as "is this forecast good".

    See Also:
        * :class:`MincerZarnowitzTest` -- the record, with the estimates
          and the fit.
        * :func:`encompassing` -- whether a rival would earn weight in a
          combination, the other reading of "could be improved".
        * :class:`~cultivars.forecast.calibration.Calibration` -- the
          density-forecast analogue.

    References:
        Mincer, J. A., & Zarnowitz, V. (1969). The evaluation of economic
        forecasts. In J. A. Mincer (Ed.), *Economic Forecasts and
        Expectations* (pp. 3-46). NBER.

    Example:
        A forecast equal to the conditional mean passes; the same
        forecast reported at twice its scale overreacts and fails:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> f = rng.standard_normal(200)
        >>> verdict = mincer_zarnowitz(f + 0.5 * rng.standard_normal(200), f)
        >>> round(verdict.slope, 1), bool(verdict.pvalue > 0.05)
        (1.0, True)
        >>> overreacting = mincer_zarnowitz(0.5 * f + 0.5 * rng.standard_normal(200), f)
        >>> round(overreacting.slope, 1), bool(overreacting.pvalue < 0.01)
        (0.5, True)
        >>> mincer_zarnowitz(f, np.ones(200))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.NumericalError: the forecasts are constant; the efficiency ...
    """
    (y, f), count = _validate_aligned_series(
        realized,
        forecast,
        horizon=horizon,
        minimum=_MIN_COMPARISON_ORIGINS,
        labels="realized and forecast",
    )
    intercept, slope, statistic, pvalue, r_squared = _mincer_zarnowitz(y, f, horizon=horizon)
    return MincerZarnowitzTest(
        intercept=intercept,
        slope=slope,
        statistic=statistic,
        pvalue=pvalue,
        r_squared=r_squared,
        horizon=int(horizon),
        nobs=int(count),
    )


def encompassing(
    realized: npt.ArrayLike,
    forecast_a: npt.ArrayLike,
    forecast_b: npt.ArrayLike,
    *,
    horizon: int = 1,
) -> EncompassingTest:
    r"""Test whether forecast A encompasses forecast B (Harvey, Leybourne & Newbold, 1998).

    A encompasses B when the optimal combination of the two,
    :math:`(1 - \lambda) f_A + \lambda f_B`, puts no weight on B, so
    combining them would not help. The test asks whether

    .. math::

       d_t = e_{A,t}\,(e_{A,t} - e_{B,t})

    has a positive mean, one-sided, with the Diebold-Mariano long-run
    variance and the Harvey-Leybourne-Newbold correction; a rejection
    says B carries information A lacks, and the reported weight is the
    share :math:`\lambda` B would get in a least-squares combination. Run
    it both ways to learn whether either forecast is redundant.

    Args:
        realized: ``(T,)`` outcomes.
        forecast_a: ``(T,)`` point forecasts claimed to encompass.
        forecast_b: ``(T,)`` point forecasts claimed to be encompassed,
            aligned origin by origin.
        horizon: The forecast horizon behind the series.

    Returns:
        The :class:`EncompassingTest`.

    Raises:
        DimensionError: If the series do not align.
        SpecificationError: If there are too few origins, the horizon is
            unusable, or the two forecasts are identical.
        NumericalError: If a series is not finite.

    Note:
        Encompassing is not the same question as accuracy. A worse
        forecast can still carry information the better one lacks, in
        which case neither encompasses the other and a combination beats
        both; and the better forecast can encompass the worse one, in
        which case the comparison is settled and combining is pointless.
        The two directions together, with a
        :class:`ForecastComparison`, are the full reading.

    See Also:
        * :class:`EncompassingTest` -- the record, with the weight.
        * :class:`ForecastComparison` -- whether the losses differ at all.
        * :func:`~cultivars.bayes.combination.stacking` -- combination
          weights for a pool of density forecasters.

    References:
        Harvey, D. I., Leybourne, S. J., & Newbold, P. (1998). Tests for
        forecast encompassing. *Journal of Business & Economic
        Statistics*, 16(2), 254-259.

    Example:
        Two predictors that each explain part of the outcome: the one
        missing ``z`` does not encompass the one that has it, and the
        one that has both encompasses the one that lacks ``z``:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> x, z = rng.standard_normal(200), rng.standard_normal(200)
        >>> y = x + z + 0.5 * rng.standard_normal(200)
        >>> forward = encompassing(y, x, x + z)
        >>> bool(forward.pvalue < 0.01), bool(abs(forward.weight - 1.0) < 0.1)  # x lacks z
        (True, True)
        >>> reverse = encompassing(y, x + z, x)
        >>> bool(reverse.pvalue > 0.05), bool(abs(reverse.weight) < 0.1)  # x + z has it
        (True, True)
        >>> encompassing(y, x, x)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: the two forecasts are numerically identical; ...
    """
    (y, a, b), count = _validate_aligned_series(
        realized,
        forecast_a,
        forecast_b,
        horizon=horizon,
        minimum=_MIN_COMPARISON_ORIGINS,
        labels="realized and both forecasts",
    )
    statistic, pvalue, weight = _forecast_encompassing(y - a, y - b, horizon=horizon)
    return EncompassingTest(
        statistic=statistic, pvalue=pvalue, weight=weight, horizon=int(horizon), nobs=int(count)
    )


def giacomini_white(
    losses_a: npt.ArrayLike,
    losses_b: npt.ArrayLike,
    *,
    horizon: int = 1,
    lags: int = 1,
    instruments: npt.ArrayLike | None = None,
) -> GiacominiWhiteTest:
    r"""Giacomini-White (2006) test that a loss gap was predictable at the origin.

    The conditional null :math:`\mathbb{E}[d_t \mid \mathcal{F}_{t-h}] =
    0` is tested through the instruments :math:`h_t`: a constant,
    ``lags`` lagged differentials, and whatever the caller adds, via the
    Wald statistic

    .. math::

       \mathrm{GW} = T\, \bar z^\top \hat\Omega^{-1} \bar z,
       \qquad z_t = h_t\, d_t,
       \qquad \mathrm{GW} \sim \chi^2_{\dim h}

    under the null, with :math:`\hat\Omega` the uncentered sample
    covariance of :math:`z_t` at one step and Bartlett HAC beyond.
    Because a loss indexed by its origin is only realized ``horizon``
    steps later, the lagged differentials start at :math:`d_{t - h}` --
    the most recent one actually known -- rather than :math:`d_{t-1}`,
    and the sample drops its first ``horizon + lags - 1`` origins.
    ``lags=0`` with no instruments is the unconditional test, a
    chi-squared cousin of Diebold-Mariano.

    The test is built for rolling-window estimation, where parameter
    estimates never converge and the forecasting *method* -- model,
    window, estimator -- is what the null is about; under a recursive
    scheme the null drifts as the window grows. Use
    :class:`ForecastComparison` for the unconditional question with the
    small-sample correction. At one step the size is close to nominal
    from 100 origins on; at multi-step horizons the truncated Bartlett
    kernel understates the moment covariance and the test over-rejects
    mildly (about 7-9% at nominal 5% for a four-step horizon), which the
    record's notes say.

    Args:
        losses_a: ``(T,)`` losses of the first forecaster, one per
            evaluation origin, negatively oriented.
        losses_b: ``(T,)`` losses of the second, aligned origin by origin.
        horizon: The forecast horizon behind the losses; sets both the
            instrument lag and the HAC window.
        lags: Lagged differentials among the instruments.
        instruments: Optional ``(T,)`` or ``(T, m)`` further instruments,
            each row known at its origin -- a regime indicator, a
            volatility proxy, the sign of the last error. The caller
            vouches for measurability; the function cannot.

    Returns:
        The :class:`GiacominiWhiteTest`.

    Raises:
        DimensionError: If the series or instruments do not align.
        SpecificationError: If there are too few origins, the horizon or
            ``lags`` is invalid, or the series are numerically identical.
        NumericalError: If a series is not finite or the instruments are
            collinear.

    Note:
        Caller instruments are aligned to the full ``(T,)`` series and
        trimmed with it, so ``instruments[t]`` must be known at origin
        ``t``, before the loss at ``t`` is realized; an instrument that
        peeks at the outcome makes the gap predictable by construction.
        A constant caller instrument duplicates the built-in constant and
        is refused as collinear. The retained sample must hold at least
        eight origins, ``2 * horizon + 1``, and twice the instrument
        count, whichever is largest.

    See Also:
        * :class:`GiacominiWhiteTest` -- the record, with the decision
          rule.
        * :class:`ForecastComparison` -- this test on the default
          instruments beside the unconditional ones.

    References:
        Giacomini, R., & White, H. (2006). Tests of conditional predictive
        ability. *Econometrica*, 74(6), 1545-1578.

    Example:
        Two forecasters whose loss gap reverses with a regime; the
        regime as instrument makes the reversal predictable, and the
        constant-only test, which is the unconditional one, sees only
        the near-zero average:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> regime = np.repeat([1.0, -1.0], 100)
        >>> a = rng.standard_normal(200) ** 2 + 0.6 * regime
        >>> b = rng.standard_normal(200) ** 2 - 0.6 * regime
        >>> verdict = giacomini_white(a, b, instruments=regime)
        >>> bool(verdict.pvalue < 0.01), verdict.instruments, verdict.df, verdict.nobs
        (True, 'constant, d[t-1], z1', 3, 199)
        >>> unconditional = giacomini_white(a, b, lags=0)
        >>> bool(unconditional.pvalue > 0.1), unconditional.instruments, unconditional.nobs
        (True, 'constant', 200)
        >>> giacomini_white(a, b, horizon=2, lags=2).instruments
        'constant, d[t-2], d[t-3]'
    """
    series, count = _validate_aligned_series(
        losses_a, losses_b, horizon=horizon, minimum=_MIN_COMPARISON_ORIGINS, labels="losses"
    )
    if lags < 0:
        raise SpecificationError(f"lags must be non-negative; got {lags}.")
    differential = series[0] - series[1]
    if float(np.abs(differential - differential.mean()).max()) <= 1e-14 * max(
        float(np.abs(differential).max()), 1.0
    ):
        raise SpecificationError(
            "the two loss series are numerically identical; there is no differential to test."
        )
    extra: npt.NDArray[np.float64] | None = None
    labels = ["constant", *(f"d[t-{horizon + j}]" for j in range(lags))]
    if instruments is not None:
        extra = np.asarray(instruments, dtype=np.float64)
        if extra.ndim == 1:
            extra = extra[:, None]
        if extra.ndim != 2 or extra.shape[0] != count:
            raise DimensionError(
                f"instruments must be (T,) or (T, m) with T = {count} origins; got shape "
                f"{extra.shape}."
            )
        if not np.all(np.isfinite(extra)):
            raise NumericalError("instruments must be finite.")
        labels.extend(f"z{k + 1}" for k in range(extra.shape[1]))
    drop = horizon + lags - 1 if lags > 0 else 0
    retained = count - drop
    if retained < max(_MIN_COMPARISON_ORIGINS, 2 * horizon + 1, 2 * len(labels)):
        raise SpecificationError(
            f"{retained} origins remain after the instruments' lags, too few for "
            f"{len(labels)} instruments at horizon {horizon}; provide more origins or fewer lags."
        )
    target, design = _conditional_instruments(differential, extra, horizon=horizon, lags=lags)
    statistic, pvalue, coefficients = _giacomini_white(target, design, horizon=horizon)
    fitted = design @ coefficients
    return GiacominiWhiteTest(
        statistic=statistic,
        pvalue=pvalue,
        df=int(design.shape[1]),
        coefficients=coefficients,
        instruments=", ".join(labels),
        share_a=float(np.mean(fitted < 0.0)),
        mean_differential=float(target.mean()),
        horizon=int(horizon),
        nobs=int(target.shape[0]),
    )
