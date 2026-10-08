# filepath: /src/cultivars/forecast/fan.py
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
r"""Fan-chart extraction: quantile paths, and nothing else.

A fan chart is quantiles of the predictive paths at each horizon -- a data
product, not a figure. For a predictive sample of :math:`S` paths, each
band edge is the pointwise empirical quantile

.. math::

   q_{\tau}(h, j) = \hat F^{-1}_{h, j}(\tau),

one per horizon :math:`h`, series :math:`j`, and level :math:`\tau`,
and a fan is the stack of those edges at a chosen set of levels: the
median, a 68% band, a 90% band, whatever the reader is meant to see.
The package draws nothing; :func:`fan_chart` hands the arrays to
whatever plots them, with the quantile convention stated by the caller
rather than baked in.

Two commitments shape the surface. First, the bands are pointwise. A
90% band at horizon :math:`h` contains 90% of the draws *at that
horizon*; it does not say that 90% of whole paths lie inside the fan at
every horizon, which is a narrower joint statement no fan chart makes.
Second, the input is the raw path array and not a result. Any
``(n_draws, steps, k)`` block enters -- the ``forecast_paths`` of a
sampled result, the ``paths`` of a conditional forecast, a model
combination's pooled draws -- so the fan is computed the same way
whichever forecaster produced the paths.

Layout. :func:`fan_chart` is the whole module; it validates the shape
and the levels and defers the arithmetic to ``numpy.quantile``. There is
no record type because the quantile array is the product.

References:
    Britton, E., Fisher, P., & Whitley, J. (1998). The Inflation Report
    projections: Understanding the fan chart. *Bank of England Quarterly
    Bulletin*, 38(1), 30-37.

Example:
    The default fan of a BVAR's posterior predictive, and a custom set
    of levels:

    >>> import numpy as np
    >>> from cultivars.multivariate.large_dim.bayesian import BVAR
    >>> y = np.random.default_rng(0).standard_normal((120, 2))
    >>> paths = BVAR(y, order=1).fit(n_draws=200, seed=0).forecast_paths(8, seed=0)
    >>> fan_chart(paths).shape
    (5, 8, 2)
    >>> median, upper = fan_chart(paths, quantiles=(0.5, 0.9))
    >>> bool(np.all(upper > median))
    True
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError, SpecificationError

__all__ = ["fan_chart"]


def fan_chart(
    paths: npt.ArrayLike,
    *,
    quantiles: tuple[float, ...] = (0.05, 0.16, 0.5, 0.84, 0.95),
) -> npt.NDArray[np.float64]:
    r"""Quantile paths of a predictive sample, per horizon and series.

    For each horizon :math:`h` and series :math:`j`, the empirical
    quantiles of the :math:`S` draws at the stated levels,

    .. math::

       q_{\tau}(h, j) = \hat F^{-1}_{h, j}(\tau),
       \qquad \tau \in \texttt{quantiles},

    stacked level by level. The quantiles are *pointwise* -- each cell
    is computed on its own draws -- so the bands describe the marginal
    predictive distribution at each horizon, not a joint band that a
    stated fraction of whole paths stays inside. The default levels give
    the 90% and 68% bands around the median, the convention of the Bank
    of England's inflation fan.

    Args:
        paths: ``(n_draws, steps, k)`` predictive paths -- the
            ``forecast_paths`` output of any Bayesian result, or any
            forecaster's simulations.
        quantiles: Strictly increasing levels interior to ``(0, 1)``.

    Returns:
        An array of shape ``(len(quantiles), steps, k)``.

    Raises:
        DimensionError: If the paths are not three-dimensional.
        SpecificationError: If the quantile levels are malformed.

    Note:
        A data product, not a figure: the package draws nothing, and the
        levels are the caller's to state, so a plotting routine reads
        row :math:`i` of the result as the :math:`\tau_i` band edge.
        NumPy's default linear interpolation between order statistics
        is used, so with few draws the outer levels are interpolated
        from the sample's extremes and should be read as approximate.

    See Also:
        * :meth:`~cultivars.forecast.conditional.ConditionalForecastResult.quantiles`
          -- the same computation on a conditional forecast's paths.
        * :class:`~cultivars.forecast.scoring.DensityScore` -- scores the
          paths the fan summarizes.

    References:
        Britton, E., Fisher, P., & Whitley, J. (1998). The Inflation
        Report projections: Understanding the fan chart. *Bank of England
        Quarterly Bulletin*, 38(1), 30-37.

    Example:
        Standard-normal draws recover the normal quantiles at every
        horizon; a single level returns one band:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> fan = fan_chart(rng.standard_normal((4000, 6, 2)))
        >>> fan.shape
        (5, 6, 2)
        >>> fan[:, 0, 0].round(1)
        array([-1.6, -1. ,  0. ,  1. ,  1.7])
        >>> bool(np.all(np.diff(fan, axis=0) >= 0.0))
        True
        >>> fan_chart(rng.standard_normal((100, 3, 1)), quantiles=(0.5,)).shape
        (1, 3, 1)
        >>> fan_chart(rng.standard_normal((100, 3, 1)), quantiles=(0.5, 0.2))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: quantiles must be strictly increasing levels ...
    """
    block = np.asarray(paths, dtype=np.float64)
    if block.ndim != 3:
        raise DimensionError(f"paths must be (n_draws, steps, k); got shape {block.shape}.")
    levels = np.asarray(quantiles, dtype=np.float64)
    if (
        levels.ndim != 1
        or levels.shape[0] < 1
        or np.any(levels <= 0.0)
        or np.any(levels >= 1.0)
        or np.any(np.diff(levels) <= 0.0)
    ):
        raise SpecificationError(
            f"quantiles must be strictly increasing levels interior to (0, 1); got {quantiles}."
        )
    return np.asarray(np.quantile(block, levels, axis=0), dtype=np.float64)
