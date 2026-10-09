# filepath: /src/cultivars/forecast/backtest.py
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
r"""Rolling-origin backtests: the harness that produces aligned forecasts and outcomes.

Every out-of-sample evaluation is the same loop: hold the sample at an
origin, estimate on what is available, forecast the next :math:`H`
periods, record what happened, advance. With :math:`o_t` the number of
observations the :math:`t`-th estimation saw, the loop visits

.. math::

   o_t = \texttt{start} + t \cdot \texttt{step},
   \qquad t = 0, \dots, T - 1,

and stores, for every horizon :math:`h \le H` and series :math:`j`, the
forecast :math:`\hat y_{\,o_t + h - 1 \mid o_t,\, j}` in the same cell as
the outcome :math:`y_{\,o_t + h - 1,\, j}`. :class:`Backtest` runs that
loop so that the alignment every downstream test depends on but cannot
check is made once, by construction, and the scoring, calibration,
comparison, and model-confidence-set tools all read from the one
record, :class:`BacktestResult`.

The model is supplied as a *fitting function* from a sample window to a
fitted result, so anything that can be fitted to a prefix of the data
can be backtested: ``lambda y: VAR(y, order=2).fit()`` or ``lambda y:
BVAR(y, order=4).fit(n_draws=500)``. A result exposing
``forecast_paths(steps, seed=...)`` is backtested as a density forecaster
and its draws are recorded; one exposing ``forecast(steps)`` as a point
forecaster. A ``predict`` callable overrides that convention for a
forecaster from outside the package.

Two commitments shape the surface. First, estimation is repeated at
every origin. A stale-parameter schedule -- re-estimating every
:math:`r` origins and forecasting from old parameters in between --
needs a result that can forecast from data it was not fitted to, a hook
the result family does not carry; the ``step`` argument spaces the
origins instead, which cuts cost the same way without a forecast ever
being made from a model that has not seen the sample up to its origin.
Second, a loss is a series, not a number. Every loss the record
computes -- squared, absolute, pinball, CRPS, log score -- comes back
one value per origin, because a comparison test needs the difference of
two such series and its serial correlation, and a summary statistic
would have thrown away exactly what the test uses; the ``(H, k)``
averages are conveniences on top.

Layout. :class:`Backtest` is the harness and :class:`BacktestResult`
the record it returns. The scores live in ``_core``: ``crps_from_draws``
is the exact sample CRPS, ``_kernel_log_score`` the negative log
predictive density by Gaussian kernel, and ``_pinball_loss`` the
quantile loss; ``PredictiveResult`` is the protocol that marks a result
as a density forecaster, and ``_validate_names`` / ``_variable_names``
resolve the series labels. :class:`~cultivars._internals._SummaryMixin`
gives the record its ``summary()``, ``str()``, and notebook rendering.

References:
    Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive
    accuracy. *Journal of Business & Economic Statistics*, 13(3),
    253-263.

    Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring
    rules, prediction, and estimation. *Journal of the American
    Statistical Association*, 102(477), 359-378.

    Tashman, L. J. (2000). Out-of-sample tests of forecasting
    accuracy: An analysis and review. *International Journal of
    Forecasting*, 16(4), 437-450.

    West, K. D. (2006). Forecast evaluation. In G. Elliott, C. W. J.
    Granger, & A. Timmermann (Eds.), *Handbook of Economic
    Forecasting* (Vol. 1, pp. 99-134). Elsevier.

Example:
    A VAR backtested from 100 observations, and the aligned loss series
    a comparison test would take:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((160, 2))
    >>> for t in range(1, 160):
    ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
    >>> record = Backtest(y, lambda w: VAR(w, order=1).fit(), horizons=2, start=100).run()
    >>> record.n_origins, record.realized.shape, record.is_density
    (59, (59, 2, 2), False)
    >>> record.losses("squared", horizon=1, name="y1").shape
    (59,)

    The same sample under a density forecaster, which unlocks the
    proper scoring rules and the calibration record:

    >>> from cultivars.multivariate.large_dim.bayesian import BVAR
    >>> fit = lambda w: BVAR(w, order=1).fit(n_draws=100, seed=0)
    >>> density = Backtest(y, fit, horizons=2, start=120, step=5).run(seed=0)
    >>> density.n_origins, density.n_draws, density.mean_crps.shape
    (8, 100, (2, 2))
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..engine._core import (
    PredictiveResult,
    SummaryTable,
    _kernel_log_score,
    _pinball_loss,
    _validate_names,
    _variable_names,
    crps_from_draws,
)
from ..engine._internals import _SummaryMixin
from ..exceptions import DimensionError, NumericalError, SpecificationError

__all__ = ["Backtest", "BacktestResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class BacktestResult(_SummaryMixin):
    r"""The aligned record of a rolling-origin forecasting exercise.

    One row per evaluation origin, one slab per horizon, one column per
    series: what was forecast, what happened, and, for a density
    forecaster, the predictive draws behind the forecast. Everything
    downstream, a density score, a calibration record, a Diebold-Mariano
    or Clark-West comparison, a model confidence set, reads from these
    arrays, so the alignment the comparison tests cannot verify is
    guaranteed here by construction: origin :math:`t`'s :math:`h`-step
    forecast and the outcome at ``origins[t] + h - 1`` are stored in the
    same cell,

    .. math::

       \texttt{point}[t, h - 1, j] = \hat y_{\,o_t + h - 1 \mid o_t,\, j},
       \qquad
       \texttt{realized}[t, h - 1, j] = y_{\,o_t + h - 1,\, j},

    with :math:`o_t` the number of observations the :math:`t`-th
    estimation saw. Every loss the record computes is a function of
    those two arrays and, for the density losses, the draws, and is
    returned one value per origin so that a comparison test receives a
    series it can difference against another backtest's.

    Attributes:
        names: Series labels.
        horizons: Longest horizon; every horizon ``1 .. horizons`` is kept.
        origins: ``(T,)`` observations available at each origin, so the
            first forecast target is row ``origins[t]`` of the sample.
        realized: ``(T, H, k)`` outcomes.
        point: ``(T, H, k)`` point forecasts -- the predictive mean for a
            density forecaster.
        paths: ``(T, S, H, k)`` predictive draws, or ``None`` for a point
            forecaster.
        scheme: ``"expanding"`` or ``"rolling"``.
        window: Observations each estimation used under a rolling scheme;
            the first origin's count under an expanding one.
        step: Origins between successive evaluations.

    Note:
        The record is the same type for a point and a density
        forecaster; the difference is whether ``paths`` is ``None``, which
        :attr:`is_density` reads. Every method that needs draws says so
        in its error rather than returning a degenerate value, so a
        point backtest handed to a calibration tool fails at the call
        that needs the draws, with the remedy named. Losses are
        negatively oriented throughout: smaller is better, including the
        log score, which is the negative log predictive density.

    See Also:
        * :class:`Backtest` -- the harness that produces this record.
        * :class:`~cultivars.forecast.comparison.ForecastComparison` and
          :func:`~cultivars.forecast.comparison.clark_west` -- the
          comparison tests :meth:`losses` feeds.
        * :class:`~cultivars.forecast.calibration.Calibration` -- the
          calibration record :meth:`record` feeds.
        * :func:`~cultivars.forecast.confidence_set.model_confidence_set`
          -- the set of models the losses cannot separate.

    References:
        Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive
        accuracy. *Journal of Business & Economic Statistics*, 13(3),
        253-263.

        Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring
        rules, prediction, and estimation. *Journal of the American
        Statistical Association*, 102(477), 359-378.

        Tashman, L. J. (2000). Out-of-sample tests of forecasting
        accuracy: An analysis and review. *International Journal of
        Forecasting*, 16(4), 437-450.

    Example:
        A point backtest of a VAR, expanding from 100 observations, and
        the aligned loss series a comparison test would take:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((160, 2))
        >>> for t in range(1, 160):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
        >>> record = Backtest(y, lambda w: VAR(w, order=1).fit(), horizons=2, start=100).run()
        >>> record.n_origins, record.horizons, record.k_endog, record.is_density
        (59, 2, 2, False)
        >>> record.origins[:3], record.realized.shape, record.paths is None
        (array([100, 101, 102]), (59, 2, 2), True)
        >>> record.losses("squared", horizon=2, name="y1").shape
        (59,)
        >>> record.rmse.round(2)
        array([[1.06, 1.19],
               [1.21, 1.21]])

        A density backtest records draws and unlocks the density losses:

        >>> from cultivars.multivariate.large_dim.bayesian import BVAR
        >>> fit = lambda w: BVAR(w, order=1).fit(n_draws=100, seed=0)
        >>> density = Backtest(y, fit, horizons=2, start=120, step=5).run()
        >>> density.n_origins, density.n_draws, density.paths.shape
        (8, 100, (8, 100, 2, 2))
        >>> density.losses("crps", horizon=1).shape, density.record(2)[0].shape
        ((8, 2), (8, 100, 2))
    """

    names: tuple[str, ...]
    """Series labels, in column order; ``"y1"``, ``"y2"``, ... unless the sample carried names."""
    horizons: int
    """Longest horizon :math:`H`; every horizon ``1 .. H`` has a slab in the arrays."""
    origins: npt.NDArray[np.intp] = field(repr=False)
    """``(T,)`` observations available at each origin, ascending, ``step`` apart.

    Origin ``t`` was estimated on the first ``origins[t]`` rows of the
    sample (the last ``window`` of them under a rolling scheme) and
    forecast rows ``origins[t] .. origins[t] + H - 1``. Kept out of the
    repr.
    """
    realized: npt.NDArray[np.float64] = field(repr=False)
    """``(T, H, k)`` outcomes, ``realized[t, h - 1]`` the sample row ``origins[t] + h - 1``.

    Kept out of the repr.
    """
    point: npt.NDArray[np.float64] = field(repr=False)
    """``(T, H, k)`` point forecasts, aligned with ``realized`` cell for cell.

    The result's ``forecast`` for a point forecaster; the mean of the
    predictive draws for a density one. Kept out of the repr.
    """
    paths: npt.NDArray[np.float64] | None = field(repr=False)
    """``(T, S, H, k)`` predictive draws, or ``None`` for a point forecaster.

    ``paths[t, :, h - 1]`` are the :math:`S` draws of the :math:`h`-step
    predictive distribution from origin ``t``; ``paths[t].mean(axis=0)``
    is ``point[t]``. Kept out of the repr.
    """
    scheme: str
    """``"expanding"``, each estimation on every row up to its origin, or ``"rolling"``."""
    window: int
    """Rows each estimation used under a rolling scheme; the first origin's count when expanding."""
    step: int
    """Origins between successive evaluations; ``1`` evaluates every origin."""

    @property
    def n_origins(self) -> int:
        """Evaluation origins, :math:`T`, the length of every loss series.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((130, 2))
            >>> Backtest(y, lambda w: VAR(w, order=1).fit(), horizons=1, start=100).run().n_origins
            30
        """
        return int(self.origins.shape[0])

    @property
    def k_endog(self) -> int:
        """Series forecast, :math:`k`, the last axis of every array."""
        return len(self.names)

    @property
    def n_draws(self) -> int:
        """Predictive draws per origin, :math:`S`; zero for a point forecaster."""
        return 0 if self.paths is None else int(self.paths.shape[1])

    @property
    def is_density(self) -> bool:
        """Whether predictive draws were recorded, which the density losses and :meth:`record` need.

        ``paths is not None``.
        """
        return self.paths is not None

    @property
    def errors(self) -> npt.NDArray[np.float64]:
        """``(T, H, k)`` forecast errors ``point - realized``.

        The sign convention is forecast minus outcome, so a positive
        error is an over-forecast. Recomputed on each access from the
        two stored arrays.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((130, 2))
            >>> record = Backtest(y, lambda w: VAR(w, order=1).fit(), horizons=3, start=100).run()
            >>> record.errors.shape
            (28, 3, 2)
            >>> bool(np.allclose(record.errors, record.point - record.realized))
            True
        """
        return np.asarray(self.point - self.realized, dtype=np.float64)

    def _horizon_index(self, horizon: int) -> int:
        """The slab index of a one-based horizon, after checking it exists.

        Args:
            horizon: A horizon in ``1 .. horizons``.

        Returns:
            ``horizon - 1``.

        Raises:
            SpecificationError: If the horizon is outside the record.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((130, 2))
            >>> record = Backtest(y, lambda w: VAR(w, order=1).fit(), horizons=2, start=100).run()
            >>> record._horizon_index(2)
            1
            >>> record._horizon_index(3)
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: horizon must lie in 1 .. 2; got 3.
        """
        if not 1 <= horizon <= self.horizons:
            raise SpecificationError(f"horizon must lie in 1 .. {self.horizons}; got {horizon}.")
        return horizon - 1

    def _series_index(self, name: str) -> int:
        """The column index of a series label, after checking it exists.

        Args:
            name: One of ``names``.

        Returns:
            Its position.

        Raises:
            SpecificationError: If the label is unknown.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((130, 2))
            >>> record = Backtest(y, lambda w: VAR(w, order=1).fit(), horizons=1, start=100).run()
            >>> record._series_index("y2")
            1
        """
        if name not in self.names:
            raise SpecificationError(f"unknown series {name!r}; expected one of {self.names}.")
        return self.names.index(name)

    def _density_losses(self, kind: str) -> npt.NDArray[np.float64]:
        r"""``(T, H, k)`` CRPS or log scores, one origin at a time.

        Each cell scores the :math:`S` draws at that origin, horizon,
        and series against the outcome: the CRPS as the sample estimate

        .. math::

           \mathrm{CRPS} = \frac{1}{S} \sum_{s} |x_s - y|
           - \frac{1}{2 S^2} \sum_{s, s'} |x_s - x_{s'}|,

        and the log score as the negative log of a kernel density of
        the draws at the outcome. Both are computed for every horizon
        at once because the loop over origins dominates and the
        per-horizon slices are cheap.

        Args:
            kind: ``"crps"`` or ``"log"``.

        Returns:
            The ``(T, H, k)`` array, negatively oriented.

        Raises:
            SpecificationError: If the backtest recorded no draws.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.bayesian import BVAR
            >>> y = np.random.default_rng(0).standard_normal((140, 2))
            >>> fit = lambda w: BVAR(w, order=1).fit(n_draws=50, seed=0)
            >>> record = Backtest(y, fit, horizons=2, start=130, step=5).run()
            >>> record._density_losses("crps").shape
            (2, 2, 2)
        """
        if self.paths is None:
            raise SpecificationError(
                f"{kind} losses need predictive draws; this backtest recorded point forecasts "
                "only. Backtest a result exposing forecast_paths()."
            )
        scorer = crps_from_draws if kind == "crps" else _kernel_log_score
        out = np.empty(self.realized.shape)
        for t in range(self.n_origins):
            for h in range(self.horizons):
                out[t, h] = scorer(self.paths[t, :, h, :], self.realized[t, h])
        return out

    def quantile(self, tau: float, *, horizon: int = 1) -> npt.NDArray[np.float64]:
        r"""``(T, k)`` quantile forecasts at level :math:`\tau` for one horizon.

        From the predictive draws when they were recorded, as the
        empirical :math:`\tau`-quantile across the :math:`S` draws at each
        origin; for a point backtest the recorded forecasts *are* the
        quantile forecasts, the case of a quantile regression backtested
        through ``predict``, and are returned as given whatever
        :math:`\tau` is asked for.

        Args:
            tau: The level, strictly inside ``(0, 1)``.
            horizon: The horizon, in ``1 .. horizons``.

        Returns:
            The ``(T, k)`` quantile forecasts.

        Raises:
            SpecificationError: If the level or horizon is unusable.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.bayesian import BVAR
            >>> y = np.random.default_rng(0).standard_normal((140, 2))
            >>> fit = lambda w: BVAR(w, order=1).fit(n_draws=50, seed=0)
            >>> record = Backtest(y, fit, horizons=1, start=130, step=5).run()
            >>> low, high = record.quantile(0.1), record.quantile(0.9)
            >>> low.shape, bool(np.all(low < high))
            ((2, 2), True)
        """
        if not 0.0 < tau < 1.0:
            raise SpecificationError(f"tau must lie strictly inside (0, 1); got {tau}.")
        h = self._horizon_index(horizon)
        if self.paths is None:
            return np.asarray(self.point[:, h, :], dtype=np.float64)
        return np.asarray(np.quantile(self.paths[:, :, h, :], tau, axis=1), dtype=np.float64)

    def losses(
        self,
        kind: str = "squared",
        *,
        horizon: int = 1,
        name: str | None = None,
        tau: float | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""One loss series per origin, aligned for a comparison test.

        The five losses are the squared and absolute error of the point
        forecast, the CRPS and log score of the predictive draws, and
        the pinball loss of the :math:`\tau`-quantile forecast,

        .. math::

           \ell_\tau(y, q) = (y - q)\bigl(\tau - \mathbb{1}\{y < q\}\bigr),

        which is the loss a quantile forecast is optimal under. Each is
        returned per origin so that two backtests on the same origins
        can be differenced; that difference is what a
        :class:`~cultivars.forecast.comparison.ForecastComparison` tests.

        Args:
            kind: ``"squared"`` or ``"absolute"`` on the point forecast;
                ``"crps"`` or ``"log"`` on the predictive draws;
                ``"pinball"`` on the ``tau``-quantile forecast.
            horizon: The horizon to score, in ``1 .. horizons``.
            name: A series label for a ``(T,)`` series; ``None`` returns
                ``(T, k)``.
            tau: The quantile level, required by and only by ``"pinball"``.

        Returns:
            The losses, negatively oriented, ``(T,)`` or ``(T, k)``.

        Raises:
            SpecificationError: If the kind, horizon, name, or level is
                unusable, or a density loss is asked of a point backtest.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((130, 2))
            >>> record = Backtest(y, lambda w: VAR(w, order=1).fit(), horizons=2, start=100).run()
            >>> record.losses().shape, record.losses("absolute", horizon=2, name="y1").shape
            ((29, 2), (29,))
            >>> bool(np.allclose(record.losses("squared"), record.errors[:, 0, :] ** 2))
            True
            >>> record.losses("crps")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: crps losses need predictive draws; ...
        """
        if kind not in ("squared", "absolute", "crps", "log", "pinball"):
            raise SpecificationError(
                f"kind must be 'squared', 'absolute', 'crps', 'log', or 'pinball'; got {kind!r}."
            )
        level = 0.5
        if kind == "pinball":
            if tau is None:
                raise SpecificationError("the pinball loss needs the quantile level tau.")
            level = float(tau)
        elif tau is not None:
            raise SpecificationError(f"tau applies to the pinball loss only; got kind={kind!r}.")
        h = self._horizon_index(horizon)
        if kind == "squared":
            table = self.errors[:, h, :] ** 2
        elif kind == "absolute":
            table = np.abs(self.errors[:, h, :])
        elif kind == "pinball":
            table = _pinball_loss(
                self.realized[:, h, :], self.quantile(level, horizon=horizon), level
            )
        else:
            table = self._density_losses(kind)[:, h, :]
        if name is None:
            return np.asarray(table, dtype=np.float64)
        return np.asarray(table[:, self._series_index(name)], dtype=np.float64)

    def record(self, horizon: int = 1) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """One horizon's ``(paths, realized)`` as ``(T, S, k)`` and ``(T, k)``.

        The shapes a calibration record takes: the draws of the
        :math:`h`-step predictive distribution at each origin beside the
        outcome they were for, which is what a probability integral
        transform is computed from.

        Args:
            horizon: The horizon, in ``1 .. horizons``.

        Returns:
            ``(paths, realized)`` for that horizon.

        Raises:
            SpecificationError: If the horizon is unknown or no draws were
                recorded.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.bayesian import BVAR
            >>> y = np.random.default_rng(0).standard_normal((140, 2))
            >>> fit = lambda w: BVAR(w, order=1).fit(n_draws=50, seed=0)
            >>> record = Backtest(y, fit, horizons=2, start=130, step=5).run()
            >>> paths, realized = record.record(2)
            >>> paths.shape, realized.shape
            ((2, 50, 2), (2, 2))
        """
        h = self._horizon_index(horizon)
        if self.paths is None:
            raise SpecificationError(
                "a calibration record needs predictive draws; this backtest recorded point "
                "forecasts only."
            )
        return (
            np.asarray(self.paths[:, :, h, :], dtype=np.float64),
            np.asarray(self.realized[:, h, :], dtype=np.float64),
        )

    @property
    def rmse(self) -> npt.NDArray[np.float64]:
        """``(H, k)`` root mean squared errors, averaged over origins.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((130, 2))
            >>> record = Backtest(y, lambda w: VAR(w, order=1).fit(), horizons=3, start=100).run()
            >>> record.rmse.shape
            (3, 2)
        """
        return np.asarray(np.sqrt((self.errors**2).mean(axis=0)), dtype=np.float64)

    @property
    def mae(self) -> npt.NDArray[np.float64]:
        """``(H, k)`` mean absolute errors, averaged over origins.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((130, 2))
            >>> record = Backtest(y, lambda w: VAR(w, order=1).fit(), horizons=3, start=100).run()
            >>> bool(np.all(record.mae <= record.rmse))
            True
        """
        return np.asarray(np.abs(self.errors).mean(axis=0), dtype=np.float64)

    @property
    def mean_crps(self) -> npt.NDArray[np.float64]:
        """``(H, k)`` mean continuous ranked probability scores; needs draws.

        Raises:
            SpecificationError: If the backtest recorded no draws.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.bayesian import BVAR
            >>> y = np.random.default_rng(0).standard_normal((140, 2))
            >>> fit = lambda w: BVAR(w, order=1).fit(n_draws=50, seed=0)
            >>> Backtest(y, fit, horizons=2, start=130, step=5).run().mean_crps.shape
            (2, 2)
        """
        return np.asarray(self._density_losses("crps").mean(axis=0), dtype=np.float64)

    @property
    def mean_log_score(self) -> npt.NDArray[np.float64]:
        """``(H, k)`` mean negative log predictive densities; needs draws.

        Raises:
            SpecificationError: If the backtest recorded no draws.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.bayesian import BVAR
            >>> y = np.random.default_rng(0).standard_normal((140, 2))
            >>> fit = lambda w: BVAR(w, order=1).fit(n_draws=50, seed=0)
            >>> Backtest(y, fit, horizons=2, start=130, step=5).run().mean_log_score.shape
            (2, 2)
        """
        return np.asarray(self._density_losses("log").mean(axis=0), dtype=np.float64)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary: one row per horizon and series.

        RMSE, MAE, and, when draws were recorded, mean CRPS per cell,
        under a header with the counts and the scheme; the notes
        describe the scheme, restate the alignment rule, and say which
        tools the record can feed.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``summary()``,
            ``str()``, and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((130, 2))
            >>> record = Backtest(y, lambda w: VAR(w, order=1).fit(), horizons=2, start=100).run()
            >>> table = record._summary_table()
            >>> table.columns, len(table.rows), table.rows[0][:2]
            (('h', 'series', 'RMSE', 'MAE', 'mean CRPS'), 4, ('1', 'y1'))
        """
        rmse, mae = self.rmse, self.mae
        crps = self.mean_crps if self.paths is not None else None
        rows = tuple(
            (
                str(h + 1),
                name,
                f"{rmse[h, j]:.4f}",
                f"{mae[h, j]:.4f}",
                "-" if crps is None else f"{crps[h, j]:.4f}",
            )
            for h in range(self.horizons)
            for j, name in enumerate(self.names)
        )
        notes = [
            f"{self.scheme.capitalize()} scheme, {self.n_origins} origins "
            f"{self.step} apart, re-estimated at every origin; "
            + (
                f"rolling window of {self.window} observations."
                if self.scheme == "rolling"
                else f"first estimation on {self.window} observations."
            ),
            "Row t's h-step forecast is aligned with the outcome at origins[t] + h - 1; "
            "losses(kind, horizon=h, name=...) hands the aligned series to a comparison test.",
        ]
        if crps is None:
            notes.append(
                "Point forecasts only: CRPS, log scores, and calibration need a result "
                "exposing forecast_paths()."
            )
        else:
            notes.append(
                f"{self.n_draws} predictive draws per origin; record(h) feeds a calibration record."
            )
        return SummaryTable(
            title="Backtest",
            metadata=(
                ("Origins", str(self.n_origins)),
                ("Horizons", str(self.horizons)),
                ("Series", str(self.k_endog)),
                ("Scheme", self.scheme),
                ("Draws", str(self.n_draws)),
            ),
            columns=("h", "series", "RMSE", "MAE", "mean CRPS"),
            rows=rows,
            notes=tuple(notes),
        )


class Backtest:
    r"""Run a rolling- or expanding-origin forecasting exercise.

    The harness behind every out-of-sample evaluation in the package.
    Given the full sample and a *fitting function* from a sample window
    to a fitted result, it visits the origins

    .. math::

       o_t = \texttt{start} + t \cdot \texttt{step},
       \qquad t = 0, \dots, T - 1,
       \qquad
       T = \Bigl\lfloor \frac{n - H - \texttt{start}}{\texttt{step}} \Bigr\rfloor + 1,

    and at each one estimates on the observations available -- all
    :math:`o_t` of them under the expanding scheme, the last
    :math:`\texttt{window}` under the rolling one -- forecasts horizons
    :math:`1, \dots, H`, and stores the forecast beside the outcome it was
    for, :math:`y_{o_t + h - 1}`. The record that comes back,
    :class:`BacktestResult`, is therefore aligned by construction, which
    is the one property the comparison, calibration, and confidence-set
    tools depend on and cannot verify for themselves.

    How the fitted result is asked to forecast is a convention the harness
    applies for the package's own families, and a ``predict`` hook
    overrides it for any other. A result exposing
    ``forecast_paths(steps, seed=...)`` (every sampled result with a
    closed system) is treated as a density forecaster and its draws are
    kept; one exposing ``forecast(steps)`` as a point forecaster. The
    fitting function receives exactly what the model constructor would:
    an ``(m, k)`` window for a multivariate sample, an ``(m,)`` one for a
    single series.

    Args:
        endog: The full sample, ``(n,)`` or ``(n, k)``.
        fit: A function from a sample window ``(m, k)`` (``(m,)`` for a
            single series) to a fitted result, typically
            ``lambda w: VAR(w, order=2).fit()``.
        horizons: Longest horizon :math:`H`; every horizon ``1 .. horizons``
            is forecast and recorded.
        start: Observations in the first estimation window. The first
            origin forecasts rows ``start .. start + horizons - 1``.
        scheme: ``"expanding"`` grows the window from ``start``;
            ``"rolling"`` keeps it at ``window`` observations.
        window: Rolling window length; defaults to ``start``. Ignored under
            the expanding scheme.
        step: Origins between successive evaluations, at least one.
        predict: ``(result, steps, seed) -> forecast`` for a forecaster the
            convention does not cover: ``(n_draws, steps, k)`` draws for a
            density forecaster, ``(steps, k)`` for a point one. By default
            ``forecast_paths(steps, seed=seed)`` is used when the result
            offers it, else ``forecast(steps)``.
        names: Series labels; default ``y1 .. yk`` (``y`` for one series).

    Attributes:
        _endog: The validated sample as ``(n, k)`` ``float64``, a single
            series stored as one column.
        _fit: The fitting function, called once per origin.
        _horizons: :math:`H`.
        _start: Observations in the first estimation window.
        _scheme: ``"expanding"`` or ``"rolling"``.
        _window: The resolved rolling window, ``start`` when none was given;
            recorded on the result under either scheme.
        _step: Origins between successive evaluations.
        _predict: The override forecasting hook, or ``None`` for the
            convention.
        _names: The resolved series labels.
        _univariate: Whether the sample arrived one-dimensional, so that
            each window is handed to ``fit`` as ``(m,)``.

    Raises:
        SpecificationError: If the window, horizon, or step leaves no
            origin to evaluate, or the scheme is unknown.
        DimensionError: If the sample is not one- or two-dimensional, or
            the labels do not match its columns.
        NumericalError: If the sample is not finite.

    Note:
        Estimation is repeated at every origin; there is no
        stale-parameter schedule that re-estimates every :math:`r` origins
        and forecasts from old parameters in between, because that needs
        a result able to forecast from data it was not fitted to, a hook
        the result family does not carry. ``step`` spaces the origins
        instead, which cuts the cost the same way without a forecast ever
        being made from a model that has not seen the sample up to its
        origin. Nothing is estimated until :meth:`run` is called;
        constructing the harness only validates the schedule.

    Warning:
        A forecast made with ``seed=None`` draws fresh predictive shocks
        on every run, so two density backtests of the same sample will
        differ at display precision unless :meth:`run` is given a seed.
        Point backtests are deterministic regardless.

    See Also:
        * :class:`BacktestResult` -- the aligned record :meth:`run`
          returns, and the losses it computes.
        * :class:`~cultivars.forecast.comparison.ForecastComparison` --
          the test two records feed.
        * :func:`~cultivars.forecast.confidence_set.model_confidence_set`
          -- the set of forecasters a collection of records cannot
          separate.

    References:
        Tashman, L. J. (2000). Out-of-sample tests of forecasting
        accuracy: An analysis and review. *International Journal of
        Forecasting*, 16(4), 437-450.

        West, K. D. (2006). Forecast evaluation. In G. Elliott, C. W. J.
        Granger, & A. Timmermann (Eds.), *Handbook of Economic
        Forecasting* (Vol. 1, pp. 99-134). Elsevier.

    Example:
        A VAR backtested from 100 observations, then a rolling schedule
        that evaluates every tenth origin on a 60-observation window:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> y = np.random.default_rng(0).standard_normal((130, 2))
        >>> harness = Backtest(y, lambda w: VAR(w, order=1).fit(), horizons=2, start=100)
        >>> harness.origins[:3], harness.origins.shape
        (array([100, 101, 102]), (29,))
        >>> harness.run().point.shape
        (29, 2, 2)
        >>> rolling = Backtest(
        ...     y, lambda w: VAR(w, order=1).fit(), start=100, scheme="rolling", window=60, step=10
        ... )
        >>> rolling.origins, rolling.run().window
        (array([100, 110, 120]), 60)

        A forecaster from outside the package is backtested through
        ``predict``; here the historical mean, whose "result" is the
        window's mean and whose forecast repeats it at every horizon:

        >>> benchmark = Backtest(
        ...     y[:, 0],
        ...     lambda w: w.mean(),
        ...     horizons=2,
        ...     start=100,
        ...     predict=lambda mean, steps, seed: np.full(steps, mean),
        ... )
        >>> record = benchmark.run()
        >>> record.names, record.point.shape
        (('y',), (29, 2, 1))
        >>> bool(np.isclose(record.point[0, 0, 0], y[:100, 0].mean()))
        True
    """

    __slots__ = (
        "_endog",
        "_fit",
        "_horizons",
        "_names",
        "_predict",
        "_scheme",
        "_start",
        "_step",
        "_univariate",
        "_window",
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        fit: Callable[[npt.NDArray[np.float64]], object],
        *,
        horizons: int = 1,
        start: int,
        scheme: str = "expanding",
        window: int | None = None,
        step: int = 1,
        predict: Callable[[object, int, int | None], npt.ArrayLike] | None = None,
        names: Sequence[str] | None = None,
    ) -> None:
        """Validate the sample and the schedule; estimate nothing.

        The sample is coerced to ``float64`` and a single series stored as
        one column, with the fact remembered so that windows go back to
        ``fit`` in the shape they arrived. The schedule is checked to
        leave at least one origin: ``start`` must be at least one and at
        most ``n - horizons``, a rolling window at most ``start``.

        Args:
            endog: The full sample, ``(n,)`` or ``(n, k)``.
            fit: The fitting function, stored as given.
            horizons: Checked to be at least one.
            start: Checked to lie in ``1 .. n - horizons``.
            scheme: Checked to be ``"expanding"`` or ``"rolling"``.
            window: Resolved to ``start`` when ``None``; under the rolling
                scheme checked to lie in ``1 .. start``.
            step: Checked to be at least one.
            predict: Stored as given.
            names: Checked against the number of columns; defaulted to
                ``y1 .. yk``, or ``y`` for a single series.

        Raises:
            SpecificationError: If the scheme is unknown, or the horizon,
                step, start, or window is out of range.
            DimensionError: If the sample is not one- or two-dimensional,
                or the labels do not match its columns.
            NumericalError: If the sample is not finite.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((130, 2))
            >>> Backtest(y, lambda w: w, start=130)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: start must leave at least one origin ...
            >>> rolling = dict(start=100, scheme="rolling", window=120)
            >>> Backtest(y, lambda w: w, **rolling)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: a rolling window must lie in 1 .. start ...
        """
        data = np.asarray(endog, dtype=np.float64)
        univariate = data.ndim == 1
        if univariate:
            data = data[:, None]
        if data.ndim != 2:
            raise DimensionError(f"endog must be 1-D or 2-D; got {data.ndim}-D.")
        if not np.all(np.isfinite(data)):
            raise NumericalError("endog must be finite.")
        if scheme not in ("expanding", "rolling"):
            raise SpecificationError(f"scheme must be 'expanding' or 'rolling'; got {scheme!r}.")
        if horizons < 1:
            raise SpecificationError(f"horizons must be at least 1; got {horizons}.")
        if step < 1:
            raise SpecificationError(f"step must be at least 1; got {step}.")
        n = data.shape[0]
        if not 1 <= start <= n - horizons:
            raise SpecificationError(
                f"start must leave at least one origin with {horizons} horizons after it: "
                f"1 <= start <= {n - horizons}; got {start}."
            )
        resolved_window = start if window is None else int(window)
        if scheme == "rolling" and not 1 <= resolved_window <= start:
            raise SpecificationError(
                f"a rolling window must lie in 1 .. start ({start}); got {resolved_window}."
            )
        labels = _validate_names(names, _variable_names(None, data.shape[1]), label="series")
        if len(labels) != data.shape[1]:
            raise SpecificationError(f"{len(labels)} names for {data.shape[1]} series.")
        self._endog = data
        self._fit = fit
        self._horizons = int(horizons)
        self._start = int(start)
        self._scheme = scheme
        self._window = resolved_window
        self._step = int(step)
        self._predict = predict
        self._names = labels
        self._univariate = univariate

    @property
    def origins(self) -> npt.NDArray[np.intp]:
        r"""Observations available at each origin the schedule will evaluate.

        The arithmetic sequence ``start, start + step, ...`` up to the last
        value that still leaves ``horizons`` rows to forecast,
        :math:`n - H`; its length is the :math:`T` of the record
        :meth:`run` returns.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((130, 2))
            >>> Backtest(y, lambda w: w, horizons=2, start=100, step=7).origins
            array([100, 107, 114, 121, 128])
        """
        n = self._endog.shape[0]
        return np.arange(self._start, n - self._horizons + 1, self._step, dtype=np.intp)

    def _window_at(self, origin: int) -> npt.NDArray[np.float64]:
        """The estimation sample at one origin, in the shape ``fit`` expects.

        Rows ``0 .. origin - 1`` under the expanding scheme, the last
        ``window`` of them under the rolling one; a single series is
        handed back as ``(m,)``, the way it arrived. A view of the stored
        sample, not a copy.

        Args:
            origin: Observations available, one of :attr:`origins`.

        Returns:
            The ``(m, k)`` or ``(m,)`` window.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((130, 2))
            >>> Backtest(y, lambda w: w, start=100)._window_at(110).shape
            (110, 2)
            >>> rolling = Backtest(y, lambda w: w, start=100, scheme="rolling", window=60)
            >>> rolling._window_at(110).shape, bool(np.all(rolling._window_at(110) == y[50:110]))
            ((60, 2), True)
            >>> Backtest(y[:, 0], lambda w: w, start=100)._window_at(100).shape
            (100,)
        """
        first = origin - self._window if self._scheme == "rolling" else 0
        block = self._endog[first:origin]
        return block[:, 0] if self._univariate else block

    def _forecast(self, result: object, seed: int | None) -> npt.NDArray[np.float64]:
        """One origin's forecast as ``(n_draws, H, k)`` draws or ``(H, k)`` points.

        Applies the forecasting convention and normalizes what comes back.
        The ``predict`` hook is called when given; otherwise a result
        satisfying :class:`~cultivars._core.PredictiveResult` is asked for
        ``forecast_paths(H, seed=seed)`` and any other for ``forecast(H)``.
        The raw array is then read by shape, in this order: a
        ``(H, k, 3)`` low/mean/high summary, the ``forecast()`` of the
        sampled families, keeps its middle column; an ``(H,)`` vector for
        a single series becomes ``(H, 1)``; an ``(H, k)`` array is a point
        forecast; an ``(S, H)`` array for a single series becomes
        ``(S, H, 1)``; and an ``(S, H, k)`` array is a set of draws.

        Args:
            result: What the fitting function returned at this origin.
            seed: Passed to ``forecast_paths`` or the ``predict`` hook, so
                that each origin draws its own predictive shocks.

        Returns:
            The normalized forecast; three-dimensional for draws,
            two-dimensional for points.

        Raises:
            SpecificationError: If no hook was given and the result offers
                neither ``forecast_paths`` nor a callable ``forecast``.
            DimensionError: If the forecast has none of the recognized
                shapes.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> from cultivars.multivariate.large_dim.bayesian import BVAR
            >>> y = np.random.default_rng(0).standard_normal((130, 2))
            >>> harness = Backtest(y, lambda w: w, horizons=2, start=100)
            >>> harness._forecast(VAR(y, order=1).fit(), None).shape
            (2, 2)
            >>> harness._forecast(BVAR(y, order=1).fit(n_draws=20, seed=0), 0).shape
            (20, 2, 2)
            >>> harness._forecast(object(), None)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: object offers neither forecast_paths() ...
        """
        k, steps = self._endog.shape[1], self._horizons
        if self._predict is not None:
            raw = np.asarray(self._predict(result, steps, seed), dtype=np.float64)
        elif isinstance(result, PredictiveResult):
            raw = np.asarray(result.forecast_paths(steps, seed=seed), dtype=np.float64)
        else:
            method = getattr(result, "forecast", None)
            if not callable(method):
                raise SpecificationError(
                    f"{type(result).__name__} offers neither forecast_paths() nor forecast(); "
                    "pass predict= to say how it forecasts."
                )
            raw = np.asarray(method(steps), dtype=np.float64)
        if raw.ndim == 3 and raw.shape[-1] == 3 and raw.shape[:2] == (steps, k):
            raw = raw[:, :, 1]  # the family's (steps, k, 3) low/mean/high summary
        if raw.ndim == 1 and raw.shape == (steps,) and k == 1:
            raw = raw[:, None]
        if raw.ndim == 2 and raw.shape == (steps, k):
            return raw
        if raw.ndim == 2 and k == 1 and raw.shape[1] == steps:
            raw = raw[:, :, None]
        if raw.ndim == 3 and raw.shape[1:] == (steps, k):
            return raw
        raise DimensionError(
            f"a forecast at horizon {steps} must be (n_draws, {steps}, {k}) draws or "
            f"({steps}, {k}) points; got shape {raw.shape} from {type(result).__name__}."
        )

    def run(self, *, seed: int | None = None) -> BacktestResult:
        """Estimate at every origin, forecast, and record.

        The loop the class exists for. At each origin the fitting function
        is called on the window, the result is asked to forecast through
        :meth:`_forecast`, and the forecast is stored beside the outcomes
        ``endog[origin : origin + horizons]``. Whether the exercise is a
        point or a density backtest is decided by the first origin's
        forecast and then enforced: every origin must produce the same
        kind, and every density origin the same number of draws, so that
        the record's arrays are rectangular. The point forecast recorded
        for a density origin is the mean of its draws.

        Args:
            seed: Base seed for the predictive draws; origin ``t`` uses
                ``seed + t`` so that repeated runs reproduce and origins
                do not share shocks. ``None`` leaves every origin's draws
                unseeded.

        Returns:
            The :class:`BacktestResult`.

        Raises:
            SpecificationError: If a forecaster cannot be read.
            DimensionError: If a forecast has an unexpected shape, or
                origins disagree on the number of draws or on whether
                they produce draws at all.
            NumericalError: If a forecast is not finite.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.large_dim.bayesian import BVAR
            >>> y = np.random.default_rng(0).standard_normal((130, 2))
            >>> fit = lambda w: BVAR(w, order=1).fit(n_draws=20, seed=0)
            >>> harness = Backtest(y, fit, horizons=2, start=120, step=5)
            >>> first, again = harness.run(seed=1), harness.run(seed=1)
            >>> first.paths.shape, bool(np.array_equal(first.paths, again.paths))
            ((2, 20, 2, 2), True)
            >>> bool(np.allclose(first.point, first.paths.mean(axis=1)))
            True

            A hook whose number of draws changes between origins is
            refused rather than padded:

            >>> draws = lambda result, h, seed: np.zeros((seed + 2, h, 2))
            >>> uneven = Backtest(y, lambda w: None, start=100, step=10, predict=draws)
            >>> uneven.run(seed=0)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.DimensionError: origin 110 produced 3 draws; earlier origins ...
        """
        origins = self.origins
        n_origins, k, steps = origins.shape[0], self._endog.shape[1], self._horizons
        realized = np.empty((n_origins, steps, k))
        point = np.empty((n_origins, steps, k))
        paths: npt.NDArray[np.float64] | None = None
        density: bool | None = None
        for t, origin in enumerate(origins):
            result = self._fit(self._window_at(int(origin)))
            raw = self._forecast(result, None if seed is None else seed + t)
            if not np.all(np.isfinite(raw)):
                raise NumericalError(f"the forecast at origin {origin} is not finite.")
            realized[t] = self._endog[origin : origin + steps]
            if raw.ndim == 3:
                if density is False:
                    raise DimensionError(
                        f"origin {origin} produced draws where earlier origins produced points."
                    )
                if paths is None:
                    paths = np.empty((n_origins, raw.shape[0], steps, k))
                elif paths.shape[1] != raw.shape[0]:
                    raise DimensionError(
                        f"origin {origin} produced {raw.shape[0]} draws; earlier origins "
                        f"produced {paths.shape[1]}."
                    )
                paths[t] = raw
                point[t] = raw.mean(axis=0)
                density = True
            else:
                if density:
                    raise DimensionError(
                        f"origin {origin} produced points where earlier origins produced draws."
                    )
                point[t] = raw
                density = False
        return BacktestResult(
            names=self._names,
            horizons=steps,
            origins=origins,
            realized=realized,
            point=point,
            paths=paths,
            scheme=self._scheme,
            window=self._window,
            step=self._step,
        )
