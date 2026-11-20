# filepath: /src/cultivars/engine/_internals/_coefficients.py
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
"""Mean coefficients: a conditional-mean draw on its natural scale.

One record, :class:`_MeanCoefficients`. The mean layers in
:mod:`~cultivars.engine._internals._means` optimize in search coordinates that
need not be the coefficients themselves, and they need one shape in which to
hand the implied intercept, autoregressive, and moving-average coefficients to
the objectives, the fit records, and the summaries. This is that shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "_MeanCoefficients",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class _MeanCoefficients:
    """A conditional-mean draw, reported on its natural scale.

    What a mean layer hands back for display, as opposed to the search
    coordinates it optimizes in. The moving-average block is present and empty
    rather than absent for a pure autoregression, so consumers index it without
    branching.

    Attributes:
        const: Intercept, or ``None`` when no intercept is estimated.
        ar: Autoregressive coefficients; empty when the order is zero.
        ma: Moving-average coefficients; empty when the order is zero.

    See Also:
        * :meth:`._ARMAMean.unpack` -- returns an instance.
        * :meth:`._LinearMean.unpack` -- returns an instance.
        * :meth:`._MeanLayer.unpack` -- returns an instance.

    Example:
        >>> import numpy as np
        >>> mean = _MeanCoefficients(const=0.1, ar=np.array([0.5]), ma=np.empty(0))
        >>> mean.ar.size, mean.ma.size
        (1, 0)
    """

    const: float | None
    """Intercept of the mean equation, or ``None`` when none is estimated."""
    ar: npt.NDArray[np.float64]
    """Autoregressive coefficients, shape ``(p,)``; empty when ``p == 0``."""
    ma: npt.NDArray[np.float64]
    """Moving-average coefficients, shape ``(q,)``; empty when ``q == 0``.

    Present and empty for a pure autoregression, never ``None``.
    """
