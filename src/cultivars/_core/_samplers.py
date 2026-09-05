# filepath: /src/cultivars/_core/_samplers.py
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

"""Sampling primitives: single conditional draws with no recursion over data.

The membership test for this module is the same one that keeps the Hamilton
filter out of :mod:`cultivars._core`: a primitive may draw from a
distribution or vectorize arithmetic over a sample, but the moment a
function walks the time axis -- a filter pass, a smoother pass -- it is an
engine and belongs in the internals layer. The stochastic-volatility *path*
draw therefore lives in :mod:`cultivars._internals._samplers`, composing
these primitives with the state-space engine; what lives here is everything
it needs that has no memory: the Kim-Shephard-Chib mixture constants and
indicator draw, and the conjugate covariance draws.

References:
    Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
        Likelihood inference and comparison with ARCH models. *Review of
        Economic Studies*, 65(3), 361-393.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.stats as sst

from ..exceptions import NumericalError
from ._defaults import _KSC_MEAN, _KSC_PROB, _KSC_VAR


def _draw_inverse_wishart(
    scale: npt.NDArray[np.float64], df: float, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """One inverse-Wishart draw.

    Args:
        scale: Positive-definite scale matrix.
        df: Degrees of freedom, greater than ``dim - 1``.
        rng: Random generator.

    Returns:
        A positive-definite matrix of the scale's shape.

    Raises:
        NumericalError: If the scale has lost positive definiteness, which in
            a Gibbs loop means an upstream block has collapsed.
    """
    try:
        draw = np.asarray(
            sst.invwishart.rvs(df=df, scale=scale, random_state=rng),
            dtype=np.float64,
        )
    except np.linalg.LinAlgError as error:
        raise NumericalError(
            "an inverse-Wishart scale matrix lost positive definiteness during sampling."
        ) from error
    return np.atleast_2d(draw)


def _draw_inverse_gamma(shape: float, rate: float, rng: np.random.Generator) -> float:
    """One inverse-gamma draw under the shape/rate convention.

    Args:
        shape: Shape parameter ``a``.
        rate: Rate parameter ``b``, so the mean is ``b / (a - 1)`` for
            ``a > 1``.
        rng: Random generator.

    Returns:
        A positive scalar.
    """
    return float(rate / rng.gamma(shape, 1.0))


def _draw_mixture_indicators(
    log_squared: npt.NDArray[np.float64],
    log_variance: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> npt.NDArray[np.intp]:
    """Draw the KSC mixture component behind each observation.

    Args:
        log_squared: ``log(e_t**2 + offset)``, one per period.
        log_variance: The current log-variance path ``h_t``.
        rng: Random generator.

    Returns:
        Component indices in ``0..6``, one per period.
    """
    gap = log_squared[:, None] - log_variance[:, None] - _KSC_MEAN[None, :]
    log_kernel = (
        np.log(_KSC_PROB)[None, :]
        - 0.5 * np.log(_KSC_VAR)[None, :]
        - 0.5 * gap**2 / _KSC_VAR[None, :]
    )
    log_kernel -= log_kernel.max(axis=1, keepdims=True)
    prob = np.exp(log_kernel)
    prob /= prob.sum(axis=1, keepdims=True)
    uniform = np.asarray(rng.random(int(log_squared.shape[0])), dtype=np.float64)
    return np.asarray((prob.cumsum(axis=1) < uniform[:, None]).sum(axis=1), dtype=np.intp)
