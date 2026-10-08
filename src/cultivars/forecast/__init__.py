# filepath: /src/cultivars/forecast/__init__.py
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
"""Forecast evaluation: scoring, comparison, and calibration of densities.

A fitted model's forecast is a claim, and this package is where claims
are graded. The workflow runs in one direction. A
:class:`~cultivars.forecast.backtest.Backtest` re-estimates a model at
every origin of a rolling or expanding schedule and records forecasts
beside outcomes, aligned by construction; from that record, or from raw
arrays a rival forecaster produced, the other modules read. Scoring
grades one origin's density with strictly proper rules; calibration asks
across many origins whether reality landed where the densities said;
comparison tests whether one forecaster's losses differ from another's,
unconditionally and conditionally, and whether a forecast is efficient,
nested-better, or encompassed; the model confidence set answers the
many-model version of the same question; conditional forecasts hold
part of a system on a stated path and ask what the rest does; and the
fan chart hands over the quantile bands of any of it.

The package has one rule that every module keeps: everything is
reported per origin, per horizon, and per series before anything is
averaged, and every loss is negatively oriented. A comparison test needs
the loss *series* and its serial correlation, a calibration record needs
the whole stack of transforms, and a summary that averaged first would
have thrown away exactly what those tests use; the ``(H, k)`` means and
the summary tables are conveniences on top of arrays that are the
result. Alignment is the other half of the rule. The tests trust that
origin :math:`t` of one series is origin :math:`t` of the other and
cannot verify it, so the backtest is where that promise is made and the
tests are where it is spent.

Each module is a leaf: names are imported from the module, not from
this package, and every producer takes arrays -- predictive paths,
outcomes, losses -- rather than models, so a forecaster from outside the
package is graded on the same terms as one from inside it.

Layout. :mod:`~cultivars.forecast.backtest` is the harness,
:class:`~cultivars.forecast.backtest.Backtest`, and its record,
:class:`~cultivars.forecast.backtest.BacktestResult`, which every
other module can read from. :mod:`~cultivars.forecast.scoring` is
:class:`~cultivars.forecast.scoring.DensityScore` and
:func:`~cultivars.forecast.scoring.pinball_loss`;
:mod:`~cultivars.forecast.calibration` is
:class:`~cultivars.forecast.calibration.Calibration` with the PIT series
and the Berkowitz test; :mod:`~cultivars.forecast.comparison` is
:class:`~cultivars.forecast.comparison.ForecastComparison` for
Diebold-Mariano with its corrections and Giacomini-White, and
:func:`~cultivars.forecast.comparison.clark_west`,
:func:`~cultivars.forecast.comparison.mincer_zarnowitz`,
:func:`~cultivars.forecast.comparison.encompassing`, and
:func:`~cultivars.forecast.comparison.giacomini_white` for the forecast
tests; :mod:`~cultivars.forecast.confidence_set` is
:func:`~cultivars.forecast.confidence_set.model_confidence_set`;
:mod:`~cultivars.forecast.conditional` is
:func:`~cultivars.forecast.conditional.conditional_forecast`; and
:mod:`~cultivars.forecast.fan` is
:func:`~cultivars.forecast.fan.fan_chart`. Model combination, which
pools the density forecasters these tools rank, lives with the Bayesian
workflow in :mod:`~cultivars.bayes.combination`.

Example:
    Backtest two forecasters on one schedule, compare their losses,
    and read the density forecaster's calibration at horizon one:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> from cultivars.multivariate.large_dim.bayesian import BVAR
    >>> y = np.random.default_rng(0).standard_normal((160, 2))
    >>> point = backtest.Backtest(y, lambda w: VAR(w, order=1).fit(), start=120).run()
    >>> fit = lambda w: BVAR(w, order=1).fit(n_draws=100, seed=0)
    >>> density = backtest.Backtest(y, fit, start=120).run(seed=0)
    >>> point.n_origins, density.n_draws
    (40, 100)
    >>> first, second = point.losses(name="y1"), density.losses(name="y1")
    >>> comparison.ForecastComparison(first, second).compute().nobs
    40
    >>> calibration.Calibration(*density.record(1)).compute().pit.shape
    (40, 2)
"""

from . import backtest, calibration, comparison, conditional, confidence_set, fan, scoring

__all__ = [
    "backtest",
    "calibration",
    "comparison",
    "conditional",
    "confidence_set",
    "fan",
    "scoring",
]
