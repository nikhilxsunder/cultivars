# filepath: /src/cultivars/forecast/scoring.py
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
r"""Proper scores for density forecasts: what a predictive sample earned.

A density forecast is a claim about a distribution, and only a *strictly
proper* scoring rule -- one a forecaster cannot improve by hedging away
from their honest belief -- deserves to grade it (Gneiting & Raftery
2007). That is the admission criterion here. For a predictive
distribution :math:`F` and an outcome :math:`y`, a score :math:`S(F, y)`
is strictly proper when :math:`\mathbb{E}_{Y \sim G}\, S(F, Y)` is
minimized over :math:`F` uniquely at :math:`F = G`; each score below is
computed from the predictive sample :math:`x^{(1)}, \dots, x^{(S)}` that
stands in for :math:`F`. The CRPS grades each series exactly from the
sample,

.. math::

   \mathrm{CRPS}(F, y) = \mathbb{E}|X - y| - \tfrac{1}{2}\, \mathbb{E}|X - X'|,

the energy score is its multivariate generalization with the Euclidean
norm in place of the absolute value, grading the joint claim; the log
score :math:`-\log f(y)` is the classical alternative, and the one number
in this module that is *estimated* (by a kernel density on the draws)
rather than exact, which its docstring says plainly. The point losses
ride along as degenerate cases because backtests report them, not
because they suffice: a point forecast is a density forecast that has
thrown its uncertainty away. The pinball loss is the proper score for
the one point forecast that keeps a distributional claim, a quantile.

Two commitments shape the surface. First, everything scores raw arrays
-- ``forecast_paths`` output, or any rival forecaster's simulations --
so the comparison is never restricted to this package's own models.
This module carries a load the design placed deliberately: the Gibbs
BVAR family reports no marginal likelihood, so out-of-sample density
scores are the honest instrument that ranks a conjugate fit against a
horseshoe fit against anything else that produces predictive paths.
Second, all scores are negatively oriented (smaller is better), and
everything is reported per horizon and per series before any averaging,
because aggregation is a modelling choice the user should make in
daylight; the summary averages for display and says so.

Layout. :class:`DensityScore` scores one origin and returns a
:class:`DensityScoreResult` with the per-cell arrays; :func:`pinball_loss`
scores a quantile forecast elementwise. The numerics live in ``_core``:
``crps_from_draws`` is the exact sample CRPS, ``energy_score`` the exact
sample energy score by chunked pairwise norms, ``_kernel_log_score`` the
Gaussian-kernel log density with Silverman's bandwidth, and
``_pinball_loss`` the check function. Across many origins the same
scores are collected by
:class:`~cultivars.forecast.backtest.BacktestResult`.

References:
    Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
    prediction, and estimation. *Journal of the American Statistical
    Association*, 102(477), 359-378.

    Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic
    forecasts, calibration and sharpness. *Journal of the Royal
    Statistical Society: Series B*, 69(2), 243-268.

    Gneiting, T. (2011). Making and evaluating point forecasts. *Journal
    of the American Statistical Association*, 106(494), 746-762.

Example:
    A conjugate and a horseshoe BVAR fitted to the same 120 observations
    and scored on the four that follow, the comparison the module exists
    for:

    >>> import numpy as np
    >>> from cultivars.bayes.priors import HorseshoePrior
    >>> from cultivars.multivariate.large_dim.bayesian import BVAR
    >>> from cultivars.multivariate.large_dim.gibbs import GibbsBVAR
    >>> y = np.random.default_rng(0).standard_normal((124, 2))
    >>> conjugate = BVAR(y[:120], order=1).fit(n_draws=200, seed=0)
    >>> sampler = GibbsBVAR(y[:120], order=1, prior=HorseshoePrior())
    >>> horseshoe = sampler.fit(n_draws=400, n_burn=200, seed=0)
    >>> first = DensityScore(conjugate.forecast_paths(4, seed=0), y[120:]).compute()
    >>> second = DensityScore(horseshoe.forecast_paths(4, seed=0), y[120:]).compute()
    >>> first.crps.shape, second.crps.shape
    ((4, 2), (4, 2))
    >>> bool(np.all(first.crps > 0.0)) and bool(np.all(second.energy > 0.0))
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..engine._core import (
    SummaryTable,
    _kernel_log_score,
    _pinball_loss,
    crps_from_draws,
    energy_score,
)
from ..engine._internals import _SummaryMixin
from ..exceptions import DimensionError, NumericalError

__all__ = ["DensityScore", "DensityScoreResult", "pinball_loss"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class DensityScoreResult(_SummaryMixin):
    r"""One forecast origin's scores, per horizon and per series.

    Every score is negatively oriented -- smaller is better -- and nothing
    is averaged: aggregation across horizons or series is the caller's
    explicit act, made in daylight. For a predictive sample
    :math:`x^{(1)}, \dots, x^{(S)}` and an outcome :math:`y`, the scores
    per cell are the continuous ranked probability score

    .. math::

       \mathrm{CRPS} = \frac{1}{S} \sum_{s} |x^{(s)} - y|
       - \frac{1}{2 S^2} \sum_{s, s'} |x^{(s)} - x^{(s')}|,

    the negative log predictive density :math:`-\log \hat f(y)` with
    :math:`\hat f` a Gaussian kernel estimate on the draws, and the
    squared and absolute errors of the predictive mean and median; per
    horizon, the energy score

    .. math::

       \mathrm{ES} = \frac{1}{S} \sum_{s} \|x^{(s)} - y\|
       - \frac{1}{2 S^2} \sum_{s, s'} \|x^{(s)} - x^{(s')}\|

    in the Euclidean norm over all :math:`k` series at once, the
    multivariate CRPS. The CRPS and energy score are strictly proper and
    exact functionals of the sample; the log score is strictly proper but
    estimated, and is the one most sensitive to tail misfit.

    Attributes:
        names: One label per series.
        crps: ``(h, k)`` continuous ranked probability scores, exact from
            the predictive sample, in each variable's own units.
        log_score: ``(h, k)`` negative log predictive density at the
            outcome, the density estimated by a Gaussian kernel on the
            draws with Silverman's bandwidth -- the module's one estimated
            quantity, and the score most sensitive to tail misfit.
        energy: ``(h,)`` energy scores of the joint predictive, the
            multivariate CRPS.
        squared_error: ``(h, k)`` squared errors of the predictive mean.
        absolute_error: ``(h, k)`` absolute errors of the predictive
            median.
        n_draws: Predictive draws scored against.

    Note:
        The point losses ride along because backtests report them, not
        because they suffice: the squared error scores the mean and the
        absolute error the median, each the point forecast that is
        optimal under its loss, and both discard the spread that the
        proper scores grade. A score is a statement about one realized
        outcome; a ranking of forecasters needs the scores of many
        origins, which :class:`~cultivars.forecast.backtest.BacktestResult`
        collects, and none of it is a posterior probability of model
        truth.

    See Also:
        * :class:`DensityScore` -- the producer.
        * :func:`pinball_loss` -- the proper score for a quantile
          forecast, which this record does not carry.
        * :meth:`~cultivars.forecast.backtest.BacktestResult.losses` --
          the same CRPS and log score, one value per origin over a
          backtest.

    References:
        Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring
        rules, prediction, and estimation. *Journal of the American
        Statistical Association*, 102(477), 359-378.

        Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007).
        Probabilistic forecasts, calibration and sharpness. *Journal of
        the Royal Statistical Society: Series B*, 69(2), 243-268.

    Example:
        Standard-normal draws scored against the outcome zero recover
        the closed forms :math:`\mathrm{CRPS} = 2\phi(0) - 1/\sqrt{\pi}
        \approx 0.234` and :math:`-\log \phi(0) \approx 0.919`:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> exact = DensityScore(rng.standard_normal((20000, 1)), np.zeros(1)).compute()
        >>> exact.n_horizons, exact.n_draws, exact.crps.shape
        (1, 20000, (1, 1))
        >>> bool(abs(exact.crps[0, 0] - 0.234) < 0.01)
        True
        >>> bool(abs(exact.log_score[0, 0] - 0.919) < 0.1)
        True

        A forecaster whose densities are too narrow is penalized by every
        proper score, the log score most of all:

        >>> outcome = rng.standard_normal((4, 2))
        >>> honest = DensityScore(rng.standard_normal((2000, 4, 2)), outcome).compute()
        >>> narrow = DensityScore(0.3 * rng.standard_normal((2000, 4, 2)), outcome).compute()
        >>> honest.crps.shape, honest.energy.shape
        ((4, 2), (4,))
        >>> bool(honest.crps.mean() < narrow.crps.mean())
        True
        >>> bool(narrow.log_score.mean() / honest.log_score.mean() > 3)
        True
    """

    names: tuple[str, ...]
    """Series labels, in column order; ``"y1"``, ``"y2"``, ... unless the producer was given names.

    A single series is ``"y1"``, not ``"y"``.
    """
    crps: npt.NDArray[np.float64] = field(repr=False)
    """``(h, k)`` continuous ranked probability scores, exact from the sample.

    In each series' own units, so not comparable across series of
    different scale. Kept out of the repr.
    """
    log_score: npt.NDArray[np.float64] = field(repr=False)
    """``(h, k)`` negative log predictive densities at the outcome.

    Estimated by a Gaussian kernel on the draws with Silverman's
    bandwidth, floored so a degenerate column still scores; the one
    estimated score, and the one most sensitive to tail misfit. Kept out
    of the repr.
    """
    energy: npt.NDArray[np.float64] = field(repr=False)
    """``(h,)`` energy scores of the joint predictive across all series.

    The multivariate CRPS, equal to the CRPS exactly when ``k = 1``. Kept
    out of the repr.
    """
    squared_error: npt.NDArray[np.float64] = field(repr=False)
    """``(h, k)`` squared errors of the predictive mean. Kept out of the repr."""
    absolute_error: npt.NDArray[np.float64] = field(repr=False)
    """``(h, k)`` absolute errors of the predictive median. Kept out of the repr."""
    n_draws: int
    """Predictive draws scored against, :math:`S`."""

    @property
    def n_horizons(self) -> int:
        """Horizons scored, :math:`h`, the first axis of every score array."""
        return int(self.crps.shape[0])

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary: one row per series, averaged over horizons.

        Mean CRPS, mean log score, and the root mean squared error of
        the predictive mean per series, with the joint energy score in
        the notes; the notes also restate the orientation, which scores
        are exact, and that the averaging is for display only.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``summary()``,
            ``str()``, and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> paths, outcome = rng.standard_normal((2000, 4, 2)), rng.standard_normal((4, 2))
            >>> table = DensityScore(paths, outcome).compute()._summary_table()
            >>> table.columns
            ('series', 'mean CRPS', 'mean log score', 'RMSE')
            >>> table.rows[0]
            ('y1', '0.4078', '1.1493', '0.6777')
        """
        rows = tuple(
            (
                name,
                f"{float(self.crps[:, index].mean()):.4f}",
                f"{float(self.log_score[:, index].mean()):.4f}",
                f"{float(np.sqrt(self.squared_error[:, index].mean())):.4f}",
            )
            for index, name in enumerate(self.names)
        )
        notes = [
            "All scores are negatively oriented: smaller is better. The "
            "CRPS and energy score are exact functionals of the predictive "
            "sample; the log score is kernel-estimated on the draws and is "
            "the number most sensitive to tail misfit.",
            f"Joint energy score, averaged over horizons: {float(self.energy.mean()):.4f}.",
            "Rows average over horizons for display only; the per-horizon "
            "arrays are the result, and aggregation is the caller's "
            "explicit choice.",
            "Scores are statements about realized samples, not posterior "
            "probabilities of model truth; nothing here is a Bayes factor.",
        ]
        return SummaryTable(
            title="Density Forecast Scores",
            metadata=(
                ("Series", f"{len(self.names)}"),
                ("Horizons", f"{self.n_horizons}"),
                ("Draws", f"{self.n_draws}"),
            ),
            columns=("series", "mean CRPS", "mean log score", "RMSE"),
            rows=rows,
            notes=tuple(notes),
        )


class DensityScore:
    r"""Score one origin's density forecast against what happened.

    Constructs from raw arrays -- the ``forecast_paths`` output of any
    Bayesian result in the package, or any rival forecaster's simulated
    paths -- so the grading is model-agnostic by design. The predictive
    sample :math:`\{x^{(s)}_{h}\}_{s=1}^{S}` at each horizon :math:`h`
    stands in for the predictive distribution, and every score in
    :class:`DensityScoreResult` is a functional of that sample and the
    outcome :math:`y_h`: the CRPS and the energy score exactly, the log
    score through a kernel density estimate, the point losses through
    the sample mean and median.

    Args:
        paths: ``(n_draws, h, k)`` predictive paths; a single-horizon
            ``(n_draws, k)`` block is promoted to ``h = 1``.
        realized: ``(h, k)`` outcomes over the same horizons; ``(k,)`` is
            promoted to ``h = 1``.
        names: One label per series. Defaults to ``y1 ... yk``.

    Attributes:
        _paths: The validated ``(n_draws, h, k)`` predictive sample.
        _realized: The validated ``(h, k)`` outcomes.
        _names: The resolved series labels.

    Raises:
        DimensionError: If the arrays cannot be scored together.
        NumericalError: If any input is not finite.

    Note:
        One origin at a time. The class scores a single predictive
        sample against a single outcome path; a ranking of forecasters
        needs many origins, which is
        :class:`~cultivars.forecast.backtest.Backtest`'s job, and its
        record computes the same CRPS and log score per origin through
        :meth:`~cultivars.forecast.backtest.BacktestResult.losses`. A
        single series must arrive as a column, ``(n_draws, 1)`` or
        ``(n_draws, h, 1)``; a one-dimensional ``(n_draws,)`` sample is
        not promoted, because it cannot be told from a single-horizon
        ``(n_draws,)`` sample of ``k`` series with the draws axis
        missing.

    See Also:
        * :class:`DensityScoreResult` -- the record :meth:`compute`
          returns.
        * :class:`~cultivars.forecast.calibration.Calibration` -- the
          calibration side of the same question, across many origins.
        * :func:`~cultivars.forecast.fan.fan_chart` -- the bands of the
          same paths.

    References:
        Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring
        rules, prediction, and estimation. *Journal of the American
        Statistical Association*, 102(477), 359-378.

    Example:
        Synthetic draws, then a BVAR's posterior predictive scored
        against the four observations it held out:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> paths = rng.standard_normal((2000, 4, 2))
        >>> outcome = rng.standard_normal((4, 2))
        >>> scores = DensityScore(paths, outcome).compute()
        >>> scores.crps.shape
        (4, 2)
        >>> bool(np.all(scores.crps > 0.0))
        True
        >>> from cultivars.multivariate.large_dim.bayesian import BVAR
        >>> y = np.random.default_rng(0).standard_normal((124, 2))
        >>> fit = BVAR(y[:120], order=1, names=("gdp", "cpi")).fit(n_draws=200, seed=0)
        >>> held_out = DensityScore(fit.forecast_paths(4, seed=0), y[120:], names=fit.names)
        >>> record = held_out.compute()
        >>> record.names, record.n_horizons, record.n_draws
        (('gdp', 'cpi'), 4, 200)
    """

    __slots__ = ("_names", "_paths", "_realized")

    def __init__(
        self,
        paths: npt.ArrayLike,
        realized: npt.ArrayLike,
        *,
        names: tuple[str, ...] | None = None,
    ) -> None:
        """Validate and align the sample and the outcomes.

        Both inputs are coerced to ``float64``, a single-horizon sample
        and outcome promoted to ``h = 1``, and the horizon and series
        axes checked to agree; every value must be finite. Labels default
        to ``y1 ... yk`` and are otherwise checked one per series.
        Nothing is scored until :meth:`compute`.

        Args:
            paths: ``(n_draws, h, k)`` or ``(n_draws, k)`` predictive
                sample.
            realized: ``(h, k)`` or ``(k,)`` outcomes.
            names: Series labels, or ``None`` for the default.

        Raises:
            DimensionError: If the shapes do not align or the labels do
                not match the series.
            NumericalError: If any input is not finite.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> scorer = DensityScore(rng.standard_normal((100, 2)), rng.standard_normal(2))
            >>> scorer._paths.shape, scorer._realized.shape, scorer._names
            ((100, 1, 2), (1, 2), ('y1', 'y2'))
            >>> paths, outcome = rng.standard_normal((100, 3, 2)), rng.standard_normal((2, 2))
            >>> DensityScore(paths, outcome)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.DimensionError: paths must be (n_draws, h, k) against (h, k) ...
        """
        block = np.asarray(paths, dtype=np.float64)
        if block.ndim == 2:
            block = block[:, None, :]
        outcome = np.asarray(realized, dtype=np.float64)
        if outcome.ndim == 1:
            outcome = outcome[None, :]
        if block.ndim != 3 or outcome.ndim != 2 or block.shape[1:] != outcome.shape:
            raise DimensionError(
                f"paths must be (n_draws, h, k) against (h, k) realizations; "
                f"got {np.asarray(paths).shape} against "
                f"{np.asarray(realized).shape}."
            )
        if not (np.all(np.isfinite(block)) and np.all(np.isfinite(outcome))):
            raise NumericalError("paths and realizations must be finite.")
        k = block.shape[2]
        if names is None:
            resolved = tuple(f"y{index + 1}" for index in range(k))
        else:
            resolved = tuple(str(name) for name in names)
            if len(resolved) != k:
                raise DimensionError(
                    f"names must have one entry per series ({k}); got {len(resolved)}."
                )
        self._paths = block
        self._realized = outcome
        self._names = resolved

    def _kernel_log_score(self) -> npt.NDArray[np.float64]:
        """Negative log predictive density by Gaussian kernel, per cell.

        One call of ``_core``'s ``_kernel_log_score`` per horizon, each
        estimating the density of every series' draws at that horizon
        with Silverman's bandwidth and evaluating it at the outcome;
        stacked to ``(h, k)``.

        Returns:
            The ``(h, k)`` negative log scores.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> scorer = DensityScore(rng.standard_normal((500, 3, 2)), rng.standard_normal((3, 2)))
            >>> scorer._kernel_log_score().shape
            (3, 2)
        """
        return np.stack(
            [
                _kernel_log_score(self._paths[:, h, :], self._realized[h])
                for h in range(self._paths.shape[1])
            ]
        )

    def compute(self) -> DensityScoreResult:
        """Evaluate every score.

        The CRPS is computed on every ``(horizon, series)`` column at
        once by flattening the sample to ``(n_draws, h * k)``; the energy
        score once per horizon on the ``(n_draws, k)`` joint sample; the
        log score through :meth:`_kernel_log_score`; and the point losses
        from the sample mean (squared error) and median (absolute error).
        Deterministic: nothing here draws.

        Returns:
            The :class:`DensityScoreResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> paths, outcome = rng.standard_normal((2000, 4, 2)), rng.standard_normal((4, 2))
            >>> record = DensityScore(paths, outcome, names=("gdp", "cpi")).compute()
            >>> record.names, record.crps.shape, record.energy.shape, record.n_draws
            (('gdp', 'cpi'), (4, 2), (4,), 2000)
            >>> bool(np.allclose(record.squared_error, (paths.mean(axis=0) - outcome) ** 2))
            True
        """
        draws, horizons, k = self._paths.shape
        flat = self._paths.reshape(draws, horizons * k)
        crps = crps_from_draws(flat, self._realized.ravel()).reshape(horizons, k)
        energy = np.array(
            [energy_score(self._paths[:, h, :], self._realized[h]) for h in range(horizons)]
        )
        mean_error = self._paths.mean(axis=0) - self._realized
        median = np.median(self._paths, axis=0)
        return DensityScoreResult(
            names=self._names,
            crps=crps,
            log_score=self._kernel_log_score(),
            energy=energy,
            squared_error=mean_error**2,
            absolute_error=np.abs(median - self._realized),
            n_draws=draws,
        )


def pinball_loss(
    realized: npt.ArrayLike, quantile: npt.ArrayLike, tau: float
) -> npt.NDArray[np.float64]:
    r"""Score a quantile forecast with the pinball loss, elementwise.

    The proper score for a :math:`\tau`-quantile forecast :math:`q` of an
    outcome :math:`y`,

    .. math::

       \ell_\tau(y, q) = \bigl(\tau - \mathbb{1}\{y < q\}\bigr)(y - q)
       = \begin{cases}
           \tau\,(y - q) & y \ge q, \\
           (1 - \tau)\,(q - y) & y < q,
         \end{cases}

    which a forecaster minimizes in expectation exactly by reporting the
    true conditional quantile, the way the squared error is minimized by
    the conditional mean. Under-prediction of a high quantile is
    penalized at rate :math:`\tau` and over-prediction at :math:`1 -
    \tau`, so the asymmetry of the level is the asymmetry of the loss. A
    quantile VAR's one-step forecasts, the :math:`\tau`-quantile of a
    predictive sample, or any rival's quantile forecasts score the same
    way; ``BacktestResult.losses("pinball", tau=...)`` applies it over a
    backtest.

    Args:
        realized: Outcomes, any shape.
        quantile: Quantile forecasts of the same shape.
        tau: The level forecast, strictly inside ``(0, 1)``.

    Returns:
        The losses, same shape, negatively oriented.

    Raises:
        DimensionError: If the shapes disagree.
        SpecificationError: If the level is outside ``(0, 1)``.
        NumericalError: If an input is not finite.

    Note:
        Elementwise and in the variable's units: nothing is averaged, and
        a ``(T, k)`` panel of forecasts scores to a ``(T, k)`` panel of
        losses. The loss at :math:`\tau = 0.5` is half the absolute
        error, so a median forecast's pinball loss and its absolute error
        rank forecasters identically.

    See Also:
        * :meth:`~cultivars.forecast.backtest.BacktestResult.losses` --
          the same loss per origin over a backtest, with ``kind="pinball"``.
        * :meth:`~cultivars.forecast.backtest.BacktestResult.quantile` --
          the quantile forecasts a density backtest supplies.
        * :class:`~cultivars.multivariate.nonlinear.quantile.QVAR` -- the
          quantile forecaster this scores natively.

    References:
        Koenker, R., & Bassett, G. (1978). Regression quantiles.
        *Econometrica*, 46(1), 33-50.

        Gneiting, T. (2011). Making and evaluating point forecasts.
        *Journal of the American Statistical Association*, 106(494),
        746-762.

    Example:
        Missing a 90% quantile from below costs nine times missing it
        from above by the same amount, and the median loss is half the
        absolute error:

        >>> import numpy as np
        >>> pinball_loss([1.0, -1.0], [0.0, 0.0], 0.9)
        array([0.9, 0.1])
        >>> pinball_loss([1.0, -1.0], [0.0, 0.0], 0.1)
        array([0.1, 0.9])
        >>> bool(np.allclose(pinball_loss([2.0, -3.0], [0.0, 0.0], 0.5), [1.0, 1.5]))
        True
        >>> pinball_loss([1.0], [0.0], 1.0)
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: tau must lie strictly inside (0, 1); got 1.0.
    """
    y = np.asarray(realized, dtype=np.float64)
    q = np.asarray(quantile, dtype=np.float64)
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(q))):
        raise NumericalError("realized and quantile forecasts must be finite.")
    return _pinball_loss(y, q, tau)
