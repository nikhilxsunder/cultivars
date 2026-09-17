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

import numpy as np
import numpy.typing as npt

from .._core import PredictiveResult, _variable_names
from .._internals import _BacktestResult as BacktestResult
from ..exceptions import DimensionError, NumericalError, SpecificationError

__all__ = ["Backtest", "BacktestResult"]


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
        labels = (
            _variable_names(None, data.shape[1])
            if names is None
            else tuple(str(name) for name in names)
        )
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
