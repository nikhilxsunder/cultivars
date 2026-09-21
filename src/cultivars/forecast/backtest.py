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

"""Rolling-origin backtests: the harness that produces aligned forecasts and outcomes.

Every out-of-sample evaluation is the same loop: hold the sample at an
origin, estimate on what is available, forecast the next ``h`` periods,
record what happened, advance. :class:`Backtest` runs that loop so that
the alignment every downstream test depends on but cannot check -- origin
``t``'s ``h``-step forecast against the outcome at ``origin + h - 1`` --
is made once, by construction, and the scoring, calibration, comparison,
and model-confidence-set tools all read from the one record.

The model is supplied as a *fitting function* from a sample window to a
fitted result, so anything that can be fitted to a prefix of the data
can be backtested: ``lambda y: VAR(y, order=2).fit()`` or ``lambda y:
BVAR(y, order=4).fit(n_draws=500)``. A result exposing
``forecast_paths(steps, seed=...)`` is backtested as a density forecaster
and its draws are recorded; one exposing ``forecast(steps)`` as a point
forecaster. A ``predict`` callable overrides that convention for a
forecaster from outside the package.

Estimation is repeated at every origin. A stale-parameter schedule --
re-estimating every ``r`` origins and forecasting from old parameters in
between -- needs a result that can forecast from data it was not fitted
to, a hook the result family does not carry; the ``step`` argument spaces
the origins instead, which cuts cost the same way without a forecast ever
being made from a model that has not seen the sample up to its origin.

Example:
    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form import VAR
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((160, 2))
    >>> for t in range(1, 160):
    ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
    >>> record = Backtest(y, lambda w: VAR(w, order=1).fit(), horizons=2, start=100).run()
    >>> record.n_origins, record.realized.shape, record.is_density
    (59, (59, 2, 2), False)
    >>> record.losses("squared", horizon=1, name="y1").shape
    (59,)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .._core import (
    PredictiveResult,
    SummaryTable,
    _kernel_log_score,
    _pinball_loss,
    _validate_names,
    _variable_names,
    crps_from_draws,
)
from .._internals import _SummaryMixin
from ..exceptions import DimensionError, NumericalError, SpecificationError

__all__ = ["Backtest", "BacktestResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class BacktestResult(_SummaryMixin):
    """The aligned record of a rolling-origin forecasting exercise.

    One row per evaluation origin, one slab per horizon, one column per
    series: what was forecast, what happened, and -- for a density
    forecaster -- the predictive draws behind the forecast. Everything
    downstream (a density score, a calibration record, a Diebold-Mariano
    or Clark-West comparison, a model confidence set) reads from these
    arrays, so the alignment the comparison tests cannot verify is
    guaranteed here by construction: origin ``t``'s ``h``-step forecast
    and the outcome at ``origins[t] + h - 1`` are stored in the same cell.

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
    """

    names: tuple[str, ...]
    horizons: int
    origins: npt.NDArray[np.intp] = field(repr=False)
    realized: npt.NDArray[np.float64] = field(repr=False)
    point: npt.NDArray[np.float64] = field(repr=False)
    paths: npt.NDArray[np.float64] | None = field(repr=False)
    scheme: str
    window: int
    step: int

    @property
    def n_origins(self) -> int:
        """Evaluation origins."""
        return int(self.origins.shape[0])

    @property
    def k_endog(self) -> int:
        """Series forecast."""
        return len(self.names)

    @property
    def n_draws(self) -> int:
        """Predictive draws per origin, zero for a point forecaster."""
        return 0 if self.paths is None else int(self.paths.shape[1])

    @property
    def is_density(self) -> bool:
        """Whether predictive draws were recorded."""
        return self.paths is not None

    @property
    def errors(self) -> npt.NDArray[np.float64]:
        """``(T, H, k)`` forecast errors ``point - realized``."""
        return np.asarray(self.point - self.realized, dtype=np.float64)

    def _horizon_index(self, horizon: int) -> int:
        if not 1 <= horizon <= self.horizons:
            raise SpecificationError(f"horizon must lie in 1 .. {self.horizons}; got {horizon}.")
        return horizon - 1

    def _series_index(self, name: str) -> int:
        if name not in self.names:
            raise SpecificationError(f"unknown series {name!r}; expected one of {self.names}.")
        return self.names.index(name)

    def _density_losses(self, kind: str) -> npt.NDArray[np.float64]:
        """``(T, H, k)`` CRPS or log scores, one origin at a time."""
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
        """``(T, k)`` quantile forecasts at level ``tau`` for one horizon.

        From the predictive draws when they were recorded; for a point
        backtest the recorded forecasts *are* the quantile forecasts --
        the case of a quantile regression backtested through ``predict``
        -- and are returned as given.

        Raises:
            SpecificationError: If the level or horizon is unusable.
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
        """One loss series per origin, aligned for a comparison test.

        Args:
            kind: ``"squared"`` or ``"absolute"`` on the point forecast;
                ``"crps"`` or ``"log"`` on the predictive draws;
                ``"pinball"`` on the ``tau``-quantile forecast.
            horizon: The horizon to score.
            name: A series label for a ``(T,)`` series; ``None`` returns
                ``(T, k)``.
            tau: The quantile level, required by and only by ``"pinball"``.

        Returns:
            The losses, negatively oriented.

        Raises:
            SpecificationError: If the kind, horizon, name, or level is
                unusable, or a density loss is asked of a point backtest.
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

        The shapes a calibration record takes.

        Raises:
            SpecificationError: If the horizon is unknown or no draws were
                recorded.
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
        """``(H, k)`` root mean squared errors."""
        return np.asarray(np.sqrt((self.errors**2).mean(axis=0)), dtype=np.float64)

    @property
    def mae(self) -> npt.NDArray[np.float64]:
        """``(H, k)`` mean absolute errors."""
        return np.asarray(np.abs(self.errors).mean(axis=0), dtype=np.float64)

    @property
    def mean_crps(self) -> npt.NDArray[np.float64]:
        """``(H, k)`` mean continuous ranked probability scores; needs draws."""
        return np.asarray(self._density_losses("crps").mean(axis=0), dtype=np.float64)

    @property
    def mean_log_score(self) -> npt.NDArray[np.float64]:
        """``(H, k)`` mean negative log predictive densities; needs draws."""
        return np.asarray(self._density_losses("log").mean(axis=0), dtype=np.float64)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary: one row per horizon and series."""
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
    """Run a rolling- or expanding-origin forecasting exercise.

    Args:
        endog: The full sample, ``(n,)`` or ``(n, k)``.
        fit: A function from a sample window ``(m, k)`` (``(m,)`` for a
            single series) to a fitted result.
        horizons: Longest horizon; every horizon ``1 .. horizons`` is
            forecast and recorded.
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

    Raises:
        SpecificationError: If the window, horizon, or step leaves no
            origin to evaluate, or the scheme is unknown.
        DimensionError: If the sample is not one- or two-dimensional.
        NumericalError: If the sample is not finite.
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
        """Validate the schedule."""
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
        """Observations available at each origin the schedule will evaluate."""
        n = self._endog.shape[0]
        return np.arange(self._start, n - self._horizons + 1, self._step, dtype=np.intp)

    def _window_at(self, origin: int) -> npt.NDArray[np.float64]:
        """The estimation sample at one origin."""
        first = origin - self._window if self._scheme == "rolling" else 0
        block = self._endog[first:origin]
        return block[:, 0] if self._univariate else block

    def _forecast(self, result: object, seed: int | None) -> npt.NDArray[np.float64]:
        """One origin's forecast as ``(n_draws, H, k)`` draws or ``(H, k)`` points."""
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

        Args:
            seed: Base seed for the predictive draws; origin ``t`` uses
                ``seed + t`` so that repeated runs reproduce and origins
                do not share shocks.

        Returns:
            The :class:`BacktestResult`.

        Raises:
            SpecificationError: If a forecaster cannot be read.
            DimensionError: If a forecast has an unexpected shape, or
                origins disagree on the number of draws.
            NumericalError: If a forecast is not finite.
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
