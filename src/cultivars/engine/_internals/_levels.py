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
"""Conditional levels: one unit of a global system, written in levels.

One record, :class:`_ConditionalLevels`, the form in which a unit model is
handed to the solve that links units into a global system. A VARX estimated
in levels and a conditional error-correction model estimated in differences
both convert to it, so the linkage reads one shape whatever estimated the
unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "_ConditionalLevels",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class _ConditionalLevels:
    r"""One unit's equations written in levels, ready to be linked.

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

    Read as an equation, with :math:`x` the unit's :math:`k` variables, :math:`f` its
    :math:`m` foreign aggregates, and ``d`` the deterministic terms::

        x_t = sum_l phi[l] x_{t-l} + impact f_t + sum_l exog_lags[l] f_{t-l}
              + deterministic' d_t + u_t

    with the lag index :math:`\ell` running from one to ``order``.

    See Also:
        * :class:`~cultivars.multivariate.reduced_form.vector_autoregression.ConditionalLevels`
          -- the public name of this class.
        * :meth:`.VECMXResult.to_varx` -- returns an instance.
        * :meth:`.VARXResult.to_varx` -- returns an instance.
    """

    phi: npt.NDArray[np.float64]
    """Coefficients on the unit's own lagged levels, ``(p, k, k)``, lag on the leading axis."""
    impact: npt.NDArray[np.float64]
    """Contemporaneous response to the foreign levels, ``(k, m)``."""
    exog_lags: npt.NDArray[np.float64]
    """Coefficients on the lagged foreign levels, ``(p, k, m)``.

    Padded with zeros to the depth of ``phi``, so the two stack lag by lag.
    """
    deterministic: npt.NDArray[np.float64]
    """Deterministic coefficients, ``(d, k)``, one row per deterministic term."""
    names: tuple[str, ...]
    """Labels of the unit's own variables, length :math:`k`."""
    exog_names: tuple[str, ...]
    """Labels of the unit's foreign aggregates, length :math:`m`."""

    @property
    def order(self) -> int:
        """Lags of the unit's own levels.

        Returns:
            The leading dimension ``p`` of ``phi``.
        """
        return int(self.phi.shape[0])

    @property
    def k_endog(self) -> int:
        """Variables the unit models.

        Returns:
            The dimension :math:`k` of the unit, read from ``phi``.
        """
        return int(self.phi.shape[1])

    @property
    def k_exog(self) -> int:
        """Foreign aggregates the unit reads.

        Returns:
            The number :math:`m` of foreign aggregates, read from ``impact``.
        """
        return int(self.impact.shape[1])
