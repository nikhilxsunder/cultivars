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

"""Fan-chart extraction: quantile paths, and nothing else.

A fan chart is quantiles of the predictive paths at each horizon -- a data
product, not a figure. The package draws nothing; this function hands the
arrays to whatever plots them, with the quantile convention stated by the
caller rather than baked in.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..exceptions import DimensionError, SpecificationError


def fan_chart(
    paths: npt.ArrayLike,
    *,
    quantiles: tuple[float, ...] = (0.05, 0.16, 0.5, 0.84, 0.95),
) -> npt.NDArray[np.float64]:
    """Quantile paths of a predictive sample, per horizon and series.

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

    Example:
        >>> rng = np.random.default_rng(0)
        >>> fan = fan_chart(rng.standard_normal((4000, 6, 2)))
        >>> fan.shape
        (5, 6, 2)
        >>> bool(np.all(np.diff(fan, axis=0) >= 0.0))
        True
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
