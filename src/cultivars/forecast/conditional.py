# filepath: /src/cultivars/forecast/conditional.py
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

"""Conditional forecasts: the system's future given stated paths for part of it.

A policy scenario is a conditional forecast: hold the policy rate on a
stated path, or oil prices, or a fiscal variable, and ask what the rest
of the system does. Waggoner and Zha (1999) make the question exact for
any closed linear system. Write the future in moving-average form,
``y_{T+h} = mu_h + sum_{j<h} Psi_j P eps_{T+h-j}``; each conditioned cell
is then one linear restriction on the stacked future shocks, and the
forecast is the law of the system given those restrictions -- a Gaussian
centered on the minimum-norm shock path that delivers the conditions,
spread over the shock directions the conditions leave free. Under a
sampled result every retained ``(B, Sigma)`` draw contributes its own
conditional paths, so parameter uncertainty is kept.

The conditions are *hard*: each conditioned cell is met exactly on every
path. The shocks that deliver them are reported, together with a modesty
reading (Leeper & Zha, 2003): a scenario that needs shocks far larger
than the model deems typical is one whose forecast the model itself does
not believe, and the summary says so rather than printing the bands as
though they meant the same thing.

References:
    Waggoner, D. F., & Zha, T. (1999). Conditional forecasts in dynamic
        multivariate models. *Review of Economics and Statistics*, 81(4),
        639-651.
    Leeper, E. M., & Zha, T. (2003). Modest policy interventions.
        *Journal of Monetary Economics*, 50(8), 1673-1700.

Example:
    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form import VAR
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((200, 2))
    >>> for t in range(1, 200):
    ...     y[t] = np.array([[0.5, 0.2], [0.1, 0.6]]) @ y[t - 1] + rng.standard_normal(2)
    >>> res = VAR(y, order=1, names=("gdp", "rate")).fit()
    >>> scenario = conditional_forecast(res, 4, {"rate": [1.0, 1.0, None, None]}, seed=0)
    >>> bool(np.allclose(scenario.paths[:, :2, 1], 1.0))
    True
    >>> scenario.n_conditions, scenario.paths.shape
    (2, (1000, 4, 2))
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from .._core import (
    _conditional_restrictions,
    _draw_conditional_shocks,
    _moving_average_from_stack,
    _moving_average_paths,
    _source_label,
)
from .._internals import _ConditionalForecastResult as ConditionalForecastResult
from .._internals import _simulate_vector_autoregression
from ..exceptions import DimensionError, NumericalError, SpecificationError

__all__ = ["ConditionalForecastResult", "conditional_forecast"]


def _condition_grid(
    conditions: Mapping[str, Sequence[float | None]], names: tuple[str, ...], steps: int
) -> npt.NDArray[np.float64]:
    """Lay the stated conditions on a ``(steps, k)`` grid, ``nan`` where free."""
    grid = np.full((steps, len(names)), np.nan)
    if not conditions:
        raise SpecificationError("at least one variable must be conditioned.")
    for name, path in conditions.items():
        if name not in names:
            raise SpecificationError(f"unknown variable {name!r}; expected one of {names}.")
        values = list(path)
        if len(values) > steps:
            raise DimensionError(
                f"the condition on {name!r} states {len(values)} horizons; the forecast has "
                f"{steps}."
            )
        for h, value in enumerate(values):
            if value is None:
                continue
            number = float(value)
            if not np.isfinite(number):
                continue
            grid[h, names.index(name)] = number
    if not np.any(np.isfinite(grid)):
        raise SpecificationError("every stated condition is None or nan; nothing to condition on.")
    return grid


def _impact(sigma: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Lower Cholesky factor of an innovation covariance."""
    try:
        return np.linalg.cholesky(0.5 * (sigma + sigma.T))
    except np.linalg.LinAlgError as error:
        raise NumericalError("the innovation covariance is not positive definite.") from error


def conditional_forecast(
    result: object,
    steps: int,
    conditions: Mapping[str, Sequence[float | None]],
    *,
    n_draws: int = 1000,
    seed: int | np.random.Generator | None = None,
) -> ConditionalForecastResult:
    """Forecast a closed system given stated future values of some of its variables.

    Args:
        result: A fitted closed system. A point result must expose
            ``forecast(steps)``, ``ma_representation(horizon)``,
            ``sigma_u`` and ``names`` (every VAR-family result does); the
            conditional paths then carry shock uncertainty at the fitted
            parameters. A result carrying ``beta_draws`` and
            ``sigma_draws`` -- the Bayesian VAR family -- is propagated
            draw by draw, and the paths carry parameter uncertainty too.
        steps: Horizons ahead.
        conditions: Variable name to its conditioned path, one entry per
            horizon from the first, ``None`` (or ``nan``) where the
            variable is free; a path shorter than ``steps`` leaves the
            remaining horizons free.
        n_draws: Conditional paths to simulate. Under a sampled result
            they are spread evenly over the retained parameter draws.
        seed: Seed or generator.

    Returns:
        The :class:`ConditionalForecastResult`.

    Raises:
        SpecificationError: If the horizon or draw count is not positive,
            a variable is unknown, nothing is conditioned, or the result
            cannot be propagated.
        DimensionError: If a condition path is longer than the horizon.
        NumericalError: If the conditions are linearly dependent or the
            covariance degenerates.
    """
    if steps < 1:
        raise SpecificationError(f"steps must be at least 1; got {steps}.")
    if n_draws < 1:
        raise SpecificationError(f"n_draws must be at least 1; got {n_draws}.")
    names = getattr(result, "names", None)
    if not isinstance(names, tuple) or not names:
        raise SpecificationError(
            f"{type(result).__name__} carries no variable names; a conditional forecast needs a "
            "closed multivariate system."
        )
    grid = _condition_grid(conditions, names, steps)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    k = len(names)
    beta_draws = getattr(result, "beta_draws", None)
    sigma_draws = getattr(result, "sigma_draws", None)
    sampled = isinstance(beta_draws, np.ndarray) and isinstance(sigma_draws, np.ndarray)

    if isinstance(beta_draws, np.ndarray) and isinstance(sigma_draws, np.ndarray):
        endog = np.asarray(getattr(result, "endog"), dtype=np.float64)  # noqa: B009
        order, trend = int(getattr(result, "order")), str(getattr(result, "trend"))  # noqa: B009
        stack_of = getattr(result, "_stack_of")  # noqa: B009
        n_kept = int(beta_draws.shape[0])
        picks = np.linspace(0, n_kept - 1, n_draws).round().astype(int)
        paths = np.empty((n_draws, steps, k))
        unconditional = np.zeros((steps, k))
        implied = np.zeros((steps, k))
        norm_sq = 0.0
        for s, draw in enumerate(picks):
            beta = beta_draws[draw]
            stack = stack_of(beta)
            mean = _simulate_vector_autoregression(
                beta,
                np.zeros((steps, k)),
                order=order,
                trend=trend,
                presample=endog[endog.shape[0] - order :],
                start=endog.shape[0] + 1,
            )
            psi = _moving_average_from_stack(stack, steps)
            impact = _impact(sigma_draws[draw])
            restriction, target = _conditional_restrictions(psi, impact, grid - mean)
            shocks, star = _draw_conditional_shocks(restriction, target, n_draws=1, rng=rng)
            paths[s] = _moving_average_paths(psi, impact, shocks.reshape(1, steps, k), mean)[0]
            unconditional += mean / n_draws
            implied += star.reshape(steps, k) / n_draws
            norm_sq += float(star @ star) / n_draws
    else:
        forecast = getattr(result, "forecast", None)
        ma = getattr(result, "ma_representation", None)
        sigma = getattr(result, "sigma_u", None)
        if not (callable(forecast) and callable(ma) and isinstance(sigma, np.ndarray)):
            raise SpecificationError(
                f"{type(result).__name__} cannot be propagated: a conditional forecast needs "
                "forecast(steps), ma_representation(horizon) and sigma_u on a point result, or "
                "beta_draws and sigma_draws on a sampled one."
            )
        mean = np.asarray(forecast(steps), dtype=np.float64)
        if mean.ndim == 3 and mean.shape[-1] == 3:
            mean = mean[:, :, 1]
        if mean.shape != (steps, k):
            raise DimensionError(
                f"forecast({steps}) returned shape {mean.shape}; expected ({steps}, {k})."
            )
        psi = np.asarray(ma(steps - 1), dtype=np.float64)
        if psi.shape != (steps, k, k):
            raise DimensionError(
                f"ma_representation({steps - 1}) returned shape {psi.shape}; expected "
                f"({steps}, {k}, {k})."
            )
        impact = _impact(np.asarray(sigma, dtype=np.float64))
        restriction, target = _conditional_restrictions(psi, impact, grid - mean)
        shocks, star = _draw_conditional_shocks(restriction, target, n_draws=n_draws, rng=rng)
        paths = _moving_average_paths(psi, impact, shocks.reshape(n_draws, steps, k), mean)
        unconditional = mean
        implied = star.reshape(steps, k)
        norm_sq = float(star @ star)

    n_conditions = int(np.isfinite(grid).sum())
    return ConditionalForecastResult(
        names=names,
        steps=int(steps),
        conditions=grid,
        unconditional=np.asarray(unconditional, dtype=np.float64),
        mean=np.asarray(paths.mean(axis=0), dtype=np.float64),
        paths=np.asarray(paths, dtype=np.float64),
        implied_shocks=np.asarray(implied, dtype=np.float64),
        modesty=norm_sq / n_conditions,
        modesty_pvalue=float(chi2.sf(norm_sq, n_conditions)),
        n_conditions=n_conditions,
        source=_source_label(result),
        parameter_uncertainty=sampled,
    )
