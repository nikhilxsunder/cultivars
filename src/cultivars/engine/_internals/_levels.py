# filepath: /src/cultivars/engine/_internals/_levels.py
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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "_ConditionalLevels",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class _ConditionalLevels:
    """One unit's equations written in levels, ready to be linked.

    The common currency of a global system. Units may be estimated as a VARX in
    levels or as a conditional error-correction model in differences, and those
    are different objects with different parameters -- but both imply the same
    thing about how the unit's variables respond to their own past and to the
    foreign aggregates, and that implication is what the linkage consumes.
    Converting each unit to this record once means the global solve never learns
    which estimator produced which unit.

    Attributes:
        phi: ``(p, k, k)`` coefficients on the unit's own lagged levels.
        impact: ``(k, m)`` contemporaneous response to the foreign levels.
        exog_lags: ``(p, k, m)`` coefficients on lagged foreign levels, padded
            to the same depth as ``phi`` so the two stack row by row.
        deterministic: ``(d, k)`` deterministic coefficients.
        names: The unit's own variable labels.
        exog_names: The unit's foreign-aggregate labels.
    """

    phi: npt.NDArray[np.float64]
    impact: npt.NDArray[np.float64]
    exog_lags: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    names: tuple[str, ...]
    exog_names: tuple[str, ...]

    @property
    def order(self) -> int:
        """Lags of the unit's own levels."""
        return int(self.phi.shape[0])

    @property
    def k_endog(self) -> int:
        """Variables the unit models."""
        return int(self.phi.shape[1])

    @property
    def k_exog(self) -> int:
        """Foreign aggregates the unit reads."""
        return int(self.impact.shape[1])
