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
r"""Conditional forecasts: the system's future given stated paths for part of it.

A policy scenario is a conditional forecast: hold the policy rate on a
stated path, or oil prices, or a fiscal variable, and ask what the rest
of the system does. Waggoner and Zha (1999) make the question exact for
any closed linear system. Write the future in moving-average form,

.. math::

   y_{T+h} = \mu_h + \sum_{j=0}^{h-1} \Psi_j P\, \varepsilon_{T+h-j},
   \qquad \varepsilon_t \sim \mathcal{N}(0, I_k),

with :math:`P` the lower Cholesky factor of the innovation covariance;
each conditioned cell is then one linear restriction on the stacked
future shocks, :math:`R \varepsilon = r`, and the forecast is the law of
the system given those restrictions -- a Gaussian centered on the
minimum-norm shock path :math:`\varepsilon^\ast = R^\top (R R^\top)^{-1}
r` that delivers the conditions, spread over the shock directions the
conditions leave free with covariance :math:`I - R^\top (R R^\top)^{-1}
R`. Under a sampled result every retained :math:`(B, \Sigma)` draw
contributes its own conditional paths, so parameter uncertainty is kept.

Two commitments shape the surface. First, the conditions are *hard*:
each conditioned cell is met exactly on every path. Interval (soft)
conditions, which would let a cell wander inside a band, are not
implemented, and the record says so in its notes rather than leaving a
degenerate band to be misread. Second, the shocks that deliver the
scenario are reported, together with a modesty reading (Leeper & Zha,
2003): :math:`\|\varepsilon^\ast\|^2` is :math:`\chi^2` on the number
of conditions under the model, so a scenario that needs shocks far
larger than the model deems typical is one whose forecast the model
itself does not believe, and the summary says so rather than printing
the bands as though they meant the same thing as an unconditional
forecast's.

Layout. :func:`conditional_forecast` is the producer and
:class:`ConditionalForecastResult` the record, which carries the paths,
the implied shocks, and the modesty reading, and satisfies
:class:`~cultivars._core.PredictiveResult` so that a fan chart or a
density score reads it like any other density forecaster. The numerics
live in ``_core``: ``_validate_conditions`` turns the name-to-path
mapping into a ``(steps, k)`` grid with ``nan`` for free cells,
``_moving_average_from_stack`` recovers :math:`\Psi_j` from a lag stack,
``_conditional_restrictions`` builds :math:`R` and :math:`r`,
``_draw_conditional_shocks`` returns :math:`\varepsilon^\ast` and the
restricted Gaussian draws, and ``_moving_average_paths`` propagates them;
``_simulate_vector_autoregression`` in ``_internals`` supplies the
per-draw mean path under a sampled result.

References:
    Waggoner, D. F., & Zha, T. (1999). Conditional forecasts in dynamic
    multivariate models. *Review of Economics and Statistics*, 81(4),
    639-651.

    Leeper, E. M., & Zha, T. (2003). Modest policy interventions.
    *Journal of Monetary Economics*, 50(8), 1673-1700.

    Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and
    conditional projection using realistic prior distributions.
    *Econometric Reviews*, 3(1), 1-100.

Example:
    A two-variable system with the rate held at one for two quarters,
    and the modesty reading that says the model finds that ordinary:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
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
    >>> round(scenario.modesty, 2), bool(scenario.modesty_pvalue > 0.05)
    (1.38, True)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from ..engine._core import (
    SummaryTable,
    _conditional_restrictions,
    _draw_conditional_shocks,
    _lower_cholesky,
    _moving_average_from_stack,
    _moving_average_paths,
    _source_label,
    _validate_conditions,
)
from ..engine._internals import _simulate_vector_autoregression, _SummaryMixin
from ..exceptions import DimensionError, SpecificationError

__all__ = ["ConditionalForecastResult", "conditional_forecast"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ConditionalForecastResult(_SummaryMixin):
    r"""A forecast of the whole system given stated future paths of some of it.

    Waggoner and Zha (1999): the future is written in moving-average form,

    .. math::

       y_{T+h} = \mu_h + \sum_{j=0}^{h-1} \Psi_j P\, \varepsilon_{T+h-j},
       \qquad \varepsilon_t \sim \mathcal{N}(0, I_k),

    with :math:`P` the lower Cholesky factor of the innovation covariance,
    so that each conditioned cell is one linear restriction on the
    stacked future shocks, :math:`R \varepsilon = r`, and the forecast is
    the system's law given those restrictions: the minimum-norm shock
    path that delivers the conditions at the center,

    .. math::

       \varepsilon^\ast = R^\top (R R^\top)^{-1} r,

    and the conditional Gaussian of the remaining shock directions around
    it, :math:`\varepsilon \mid R\varepsilon = r \sim \mathcal{N}\bigl(
    \varepsilon^\ast,\ I - R^\top (R R^\top)^{-1} R\bigr)`. Every
    conditioned cell is met exactly on every path; the free cells spread
    over what the conditions leave undetermined. Under a sampled result
    every retained parameter draw contributes its own conditional paths,
    so the bands carry parameter uncertainty as well.

    The record also says whether the scenario is one the model believes.
    Under the model :math:`\|\varepsilon^\ast\|^2` is :math:`\chi^2` on
    the number of conditions :math:`q`, so :attr:`modesty`
    :math:`= \|\varepsilon^\ast\|^2 / q` is one when the scenario needs
    shocks of typical size and large when it needs shocks the model deems
    improbable (Leeper & Zha, 2003); the paths of a scenario like that
    describe a future agents would notice was engineered.

    Attributes:
        names: Series labels.
        steps: Horizons ahead.
        conditions: ``(steps, k)`` conditioned values, ``nan`` where free.
        unconditional: ``(steps, k)`` mean path with no conditions imposed
            (averaged over draws under a sampled result).
        mean: ``(steps, k)`` conditional mean path.
        paths: ``(S, steps, k)`` conditional predictive paths; conditioned
            cells equal their conditions on every path.
        implied_shocks: ``(steps, k)`` minimum-norm standardized shocks
            that deliver the conditions, in the orthogonalized (impact
            matrix) basis; averaged over draws under a sampled result.
        modesty: ``||eps*||**2`` of the minimum-norm shocks relative to
            the number of conditions -- one when the scenario needs shocks
            of typical size, large when it needs shocks the model deems
            improbable (Leeper & Zha, 2003).
        modesty_pvalue: Upper-tail chi-squared probability of
            ``||eps*||**2`` on ``n_conditions`` degrees of freedom.
        n_conditions: Cells conditioned.
        source: The result forecast.
        parameter_uncertainty: Whether the paths mix over parameter draws.

    Note:
        Conditions are *hard*: each conditioned cell is met exactly, so
        the 16-84% band of a conditioned cell is degenerate at the
        condition. Interval (soft) conditions, which would widen it, are
        not implemented. The record satisfies the
        :class:`~cultivars._core.PredictiveResult` protocol through
        :meth:`forecast_paths`, so a fan chart, a density score, or a
        model combination reads it like any other density forecaster,
        except that the horizon is fixed at construction.

    See Also:
        * :func:`conditional_forecast` -- the producer.
        * :func:`~cultivars.forecast.fan.fan_chart` -- the bands the paths
          give.
        * :class:`~cultivars.forecast.scoring.DensityScore` -- scores the
          paths against an outcome.

    References:
        Waggoner, D. F., & Zha, T. (1999). Conditional forecasts in
        dynamic multivariate models. *Review of Economics and Statistics*,
        81(4), 639-651.

        Leeper, E. M., & Zha, T. (2003). Modest policy interventions.
        *Journal of Monetary Economics*, 50(8), 1673-1700.

    Example:
        A two-variable VAR with the rate held at one for two quarters:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((200, 2))
        >>> for t in range(1, 200):
        ...     y[t] = np.array([[0.5, 0.2], [0.1, 0.6]]) @ y[t - 1] + rng.standard_normal(2)
        >>> res = VAR(y, order=1, names=("gdp", "rate")).fit()
        >>> scenario = conditional_forecast(res, 4, {"rate": [1.0, 1.0, None, None]}, seed=0)
        >>> scenario.n_conditions, scenario.paths.shape, scenario.parameter_uncertainty
        (2, (1000, 4, 2), False)
        >>> scenario.conditioned[:, 1]
        array([ True,  True, False, False])
        >>> bool(np.allclose(scenario.paths[:, :2, 1], 1.0))
        True
        >>> round(scenario.modesty, 2), round(scenario.modesty_pvalue, 3)
        (1.38, 0.251)

        The same model asked to hold the rate at eight, a scenario it
        finds implausible:

        >>> ambitious = conditional_forecast(res, 4, {"rate": [8.0] * 4}, seed=0)
        >>> round(ambitious.modesty, 1), bool(ambitious.modesty_pvalue < 1e-10)
        (25.8, True)
    """

    names: tuple[str, ...]
    """Series labels, in the result's column order."""
    steps: int
    """Horizons ahead, :math:`H`; the horizon the paths were simulated for."""
    conditions: npt.NDArray[np.float64] = field(repr=False)
    """``(steps, k)`` conditioned values, ``nan`` where a cell is free.

    The grid ``conditional_forecast`` built from the ``conditions``
    mapping, with short paths padded by ``nan``. Kept out of the repr.
    """
    unconditional: npt.NDArray[np.float64] = field(repr=False)
    """``(steps, k)`` mean path with no conditions imposed, :math:`\\mu_h`.

    The result's own point forecast for a point result; the mean of the
    per-draw point forecasts for a sampled one. Kept out of the repr.
    """
    mean: npt.NDArray[np.float64] = field(repr=False)
    """``(steps, k)`` conditional mean path, ``paths.mean(axis=0)``.

    Equals the condition in every conditioned cell. Kept out of the
    repr.
    """
    paths: npt.NDArray[np.float64] = field(repr=False)
    """``(S, steps, k)`` conditional predictive paths.

    Each is :math:`\\mu_h + \\sum_j \\Psi_j P \\varepsilon_{T+h-j}` for one
    draw of the restricted shocks, so the conditioned cells equal their
    conditions on every path; under a sampled result each path also
    uses its own parameter draw. Kept out of the repr.
    """
    implied_shocks: npt.NDArray[np.float64] = field(repr=False)
    """``(steps, k)`` minimum-norm standardized shocks :math:`\\varepsilon^\\ast`.

    In the orthogonalized basis of the impact matrix :math:`P`, so each
    entry is in units of that shock's standard deviation; zero at
    horizons beyond the last condition. Averaged over draws under a
    sampled result. Kept out of the repr.
    """
    modesty: float
    """:math:`\\|\\varepsilon^\\ast\\|^2 / q`, the shock norm per condition.

    One when the scenario needs shocks of typical size; large when it
    needs shocks the model deems improbable. Under a sampled result the
    norm is averaged over parameter draws before dividing.
    """
    modesty_pvalue: float
    """Upper-tail :math:`\\chi^2_q` probability of :math:`\\|\\varepsilon^\\ast\\|^2`.

    Small when the scenario is one the model finds improbable.
    """
    n_conditions: int
    """Cells conditioned, :math:`q`, the count of finite entries in ``conditions``."""
    source: str
    """The class name of the result forecast, for the summary title."""
    parameter_uncertainty: bool
    """Whether the paths mix over parameter draws (a sampled result) or hold them fixed."""

    @property
    def k_endog(self) -> int:
        """Series forecast, :math:`k`, the last axis of every array."""
        return len(self.names)

    @property
    def n_draws(self) -> int:
        """Conditional paths, :math:`S`, the first axis of ``paths``."""
        return int(self.paths.shape[0])

    @property
    def conditioned(self) -> npt.NDArray[np.bool_]:
        """``(steps, k)`` mask of conditioned cells, ``~isnan(conditions)``.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = VAR(y, order=1).fit()
            >>> scenario = conditional_forecast(res, 3, {"y1": [0.0, None, 0.0]}, n_draws=10)
            >>> scenario.conditioned
            array([[ True, False],
                   [False, False],
                   [ True, False]])
        """
        return np.asarray(~np.isnan(self.conditions))

    def forecast_paths(
        self, steps: int | None = None, *, seed: int | np.random.Generator | None = None
    ) -> npt.NDArray[np.float64]:
        """The conditional paths, in the shape every scorer and fan chart reads.

        The member that makes the record a
        :class:`~cultivars._core.PredictiveResult`. The paths were
        simulated once, at construction, so nothing is drawn here: the
        stored ``(S, steps, k)`` array is returned as is.

        Args:
            steps: Must equal :attr:`steps` when given; the paths were
                simulated once, at construction.
            seed: Ignored; kept for the predictive-result signature.

        Returns:
            The ``(S, steps, k)`` conditional paths.

        Raises:
            SpecificationError: If a different horizon is asked for.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = VAR(y, order=1).fit()
            >>> scenario = conditional_forecast(res, 3, {"y1": [0.0]}, n_draws=10, seed=0)
            >>> scenario.forecast_paths().shape, scenario.forecast_paths(3, seed=1).shape
            ((10, 3, 2), (10, 3, 2))
            >>> scenario.forecast_paths(6)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: the conditional forecast was simulated ...
        """
        if steps is not None and steps != self.steps:
            raise SpecificationError(
                f"the conditional forecast was simulated for {self.steps} steps; asked for "
                f"{steps}. Build a new one for another horizon."
            )
        return np.asarray(self.paths, dtype=np.float64)

    def quantiles(
        self, levels: tuple[float, ...] = (0.05, 0.16, 0.5, 0.84, 0.95)
    ) -> npt.NDArray[np.float64]:
        """``(len(levels), steps, k)`` pointwise quantiles of the conditional paths.

        Pointwise: each cell's quantile across the :math:`S` paths, not
        a quantile of whole paths, so a conditioned cell shows its
        condition at every level.

        Args:
            levels: Probability levels, each inside ``[0, 1]``.

        Returns:
            The ``(len(levels), steps, k)`` quantiles.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = VAR(y, order=1).fit()
            >>> scenario = conditional_forecast(res, 3, {"y1": [0.5]}, n_draws=50, seed=0)
            >>> bands = scenario.quantiles((0.16, 0.84))
            >>> bands.shape, bands[:, 0, 0]
            ((2, 3, 2), array([0.5, 0.5]))
        """
        return np.asarray(np.quantile(self.paths, levels, axis=0), dtype=np.float64)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary: mean paths, conditioned cells marked.

        One row per horizon and series with the unconditional mean, the
        conditional mean, the 16-84% band, and a star on conditioned
        cells; the notes read the modesty statistic, say whether the
        bands carry parameter uncertainty, and restate that conditions
        are hard.

        Returns:
            The :class:`~cultivars._core.SummaryTable` that ``summary()``,
            ``str()``, and the notebook renderer display.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = VAR(y, order=1).fit()
            >>> scenario = conditional_forecast(res, 2, {"y1": [0.5]}, n_draws=50, seed=0)
            >>> table = scenario._summary_table()
            >>> table.columns
            ('h', 'series', 'unconditional', 'conditional', '16-84%', '')
            >>> table.rows[0][1], table.rows[0][3], table.rows[0][5], table.rows[1][5]
            ('y1', '0.5000', '*', '')
        """
        low, high = np.quantile(self.paths, [0.16, 0.84], axis=0)
        mask = self.conditioned
        rows = tuple(
            (
                str(h + 1),
                name,
                f"{self.unconditional[h, j]:.4f}",
                f"{self.mean[h, j]:.4f}",
                f"[{low[h, j]:.4f}, {high[h, j]:.4f}]",
                "*" if mask[h, j] else "",
            )
            for h in range(self.steps)
            for j, name in enumerate(self.names)
        )
        notes = [
            f"* conditioned cell ({self.n_conditions} of {self.steps * self.k_endog}); the "
            "conditional path equals the condition there on every draw.",
            f"Modesty {self.modesty:.2f} (p = {self.modesty_pvalue:.3f}): the minimum-norm "
            "shocks that deliver the scenario have squared norm "
            f"{self.modesty * self.n_conditions:.2f} against {self.n_conditions} expected under "
            "the model. Well above one, the scenario is one the model finds improbable and "
            "agents would notice (Leeper & Zha, 2003).",
            (
                "Bands mix parameter draws and conditional shock draws."
                if self.parameter_uncertainty
                else "Bands carry conditional shock uncertainty at fixed parameters; the "
                "point result offers no parameter draws to propagate."
            ),
            "Hard conditions only: each conditioned cell is met exactly. Interval (soft) "
            "conditions are not implemented.",
        ]
        return SummaryTable(
            title=f"Conditional Forecast: {self.source}",
            metadata=(
                ("Steps", str(self.steps)),
                ("Series", str(self.k_endog)),
                ("Conditions", str(self.n_conditions)),
                ("Draws", str(self.n_draws)),
                ("Modesty", f"{self.modesty:.2f}"),
            ),
            columns=("h", "series", "unconditional", "conditional", "16-84%", ""),
            rows=rows,
            notes=tuple(notes),
        )


def conditional_forecast(
    result: object,
    steps: int,
    conditions: Mapping[str, Sequence[float | None]],
    *,
    n_draws: int = 1000,
    seed: int | np.random.Generator | None = None,
) -> ConditionalForecastResult:
    r"""Forecast a closed system given stated future values of some of its variables.

    The Waggoner-Zha (1999) construction on whichever kind of result is
    passed. For a point result the unconditional mean path :math:`\mu`
    is the result's own ``forecast(steps)``, the moving-average matrices
    :math:`\Psi_0, \dots, \Psi_{H-1}` its ``ma_representation``, and the
    impact matrix :math:`P` the lower Cholesky factor of ``sigma_u``;
    the conditions become the restriction system

    .. math::

       R \varepsilon = r, \qquad r_i = c_{h_i, j_i} - \mu_{h_i, j_i},

    one row per conditioned cell, with :math:`R` built from the rows of
    :math:`\Psi_j P` that reach that cell, and ``n_draws`` shock paths
    are drawn from :math:`\mathcal{N}(\varepsilon^\ast, I - R^\top (R
    R^\top)^{-1} R)` and propagated through the moving average. For a
    sampled result the same construction is repeated once per parameter
    draw: ``n_draws`` draws are picked evenly over the retained
    posterior, each supplies its own :math:`\mu`, :math:`\Psi`, and
    :math:`P` from ``beta_draws`` and ``sigma_draws``, and contributes
    one conditional path, so the paths mix parameter and shock
    uncertainty.

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

    Note:
        The result is read by duck typing so that a system from outside
        the package can be conditioned on: ``names`` is required in every
        case, and then either the sampled trio ``beta_draws``,
        ``sigma_draws``, ``_stack_of`` (with ``endog``, ``order``,
        ``trend``) or the point trio ``forecast``, ``ma_representation``,
        ``sigma_u``. A ``forecast`` that returns the sampled families'
        ``(steps, k, 3)`` low/mean/high summary is read at its middle
        column. Conditions are hard, so a scenario that pins every
        variable at every horizon has no free shock directions and every
        path is the same; the modesty reading is then the only content.
        Under a sampled result the modesty norm is the average of
        :math:`\|\varepsilon^\ast\|^2` over parameter draws, and its
        p-value is read against :math:`\chi^2_q` as if it were a single
        draw's norm.

    See Also:
        * :class:`ConditionalForecastResult` -- the record, with the
          modesty reading and the paths.
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VARResult`
          -- a point result whose ``forecast``, ``ma_representation``,
          and ``sigma_u`` are what the point branch reads.
        * :func:`~cultivars.forecast.fan.fan_chart` -- the bands the paths
          give.

    References:
        Waggoner, D. F., & Zha, T. (1999). Conditional forecasts in
        dynamic multivariate models. *Review of Economics and Statistics*,
        81(4), 639-651.

        Leeper, E. M., & Zha, T. (2003). Modest policy interventions.
        *Journal of Monetary Economics*, 50(8), 1673-1700.

    Example:
        A point VAR with one variable pinned for two horizons, then a
        Bayesian VAR whose paths also mix over parameter draws:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> from cultivars.multivariate.large_dim.bayesian import BVAR
        >>> y = np.random.default_rng(0).standard_normal((120, 2))
        >>> res = VAR(y, order=1).fit()
        >>> scenario = conditional_forecast(res, 3, {"y1": [0.5, 0.5]}, n_draws=200, seed=0)
        >>> scenario.n_conditions, scenario.paths.shape, scenario.parameter_uncertainty
        (2, (200, 3, 2), False)
        >>> bool(np.allclose(scenario.paths[:, :2, 0], 0.5))
        True
        >>> posterior = BVAR(y, order=1).fit(n_draws=200, seed=0)
        >>> sampled = conditional_forecast(posterior, 3, {"y1": [0.5, 0.5]}, n_draws=50, seed=0)
        >>> sampled.source, sampled.parameter_uncertainty, sampled.n_draws
        ('BVARResult', True, 50)

        A short path leaves later horizons free, and ``None`` or ``nan``
        frees a cell in the middle:

        >>> mixed = conditional_forecast(
        ...     res, 3, {"y1": [0.5, None], "y2": [float("nan"), 0.2]}, n_draws=20, seed=0
        ... )
        >>> mixed.conditions
        array([[0.5, nan],
               [nan, 0.2],
               [nan, nan]])
        >>> conditional_forecast(res, 3, {"y1": [None, None]})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: every stated condition is None or nan; ...
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
    grid = _validate_conditions(conditions, names, steps)
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
            impact = _lower_cholesky(sigma_draws[draw], "the innovationcovariance")
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
        impact = _lower_cholesky(np.asarray(sigma, dtype=np.float64), "the innovationcovariance")
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
