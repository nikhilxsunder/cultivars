# filepath: /src/cultivars/engine/_internals/_layouts.py
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
"""Parameter layout: which design column serves which regime in a switching regression.

One record, :class:`_ParameterLayout`. The M-step of a Markov-switching
autoregression is a weighted least-squares regression on a design matrix
whose intercept block and autoregressive block may each be shared across
regimes or repeated once per regime. The layout says where each block
sits, so the code that builds the design and the code that reads the
coefficients back cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "_ParameterLayout",
]


@dataclass(frozen=True, slots=True)
class _ParameterLayout:
    """Column bookkeeping for the responsibility-weighted regression.

    Switching and non-switching blocks share one design matrix; this object
    is the single source of truth for which column belongs to which regime,
    so the M-step never has to recompute offsets inline.

    The intercept block comes first and the autoregressive block second.
    A switching block is regime-major: all of regime zero's columns, then
    all of regime one's.

    Note:
        Nothing is validated. A ``regime`` outside ``0 .. n_regimes - 1``
        passed to :meth:`intercept_col` or :meth:`ar_slice` returns an
        index beyond the design.

    See Also:
        * :meth:`._MarkovSwitchingModel._parameter_layout` -- returns an
          instance.

    Example:
        Switching intercepts over a shared AR(3) block:

        >>> layout = _ParameterLayout(
        ...     n_regimes=2, order=3, switching_mean=True, switching_ar=False
        ... )
        >>> layout.n_intercept, layout.n_ar, layout.width
        (2, 3, 5)
        >>> layout.intercept_col(1), layout.ar_slice(1)
        (1, slice(2, 5, None))
    """

    n_regimes: int
    """Number of regimes :math:`K`."""
    order: int
    """Autoregressive order ``p``."""
    switching_mean: bool
    """Whether each regime has its own intercept column; otherwise one column is shared."""
    switching_ar: bool
    """Whether each regime has its own block of ``order`` AR columns; otherwise one is shared."""

    @property
    def n_intercept(self) -> int:
        """Number of intercept columns.

        Returns:
            ``n_regimes`` when the mean switches, otherwise ``1``.
        """
        return self.n_regimes if self.switching_mean else 1

    @property
    def n_ar(self) -> int:
        """Number of autoregressive columns.

        Returns:
            ``n_regimes * order`` when the AR block switches, otherwise
            ``order``.
        """
        return self.n_regimes * self.order if self.switching_ar else self.order

    @property
    def width(self) -> int:
        """Total design width.

        Returns:
            ``n_intercept + n_ar``.
        """
        return self.n_intercept + self.n_ar

    def intercept_col(self, regime: int) -> int:
        """Index of the intercept column serving ``regime``.

        Args:
            regime: Regime index, counted from zero.

        Returns:
            ``regime`` when the mean switches, otherwise ``0``.
        """
        return regime if self.switching_mean else 0

    def ar_slice(self, regime: int) -> slice:
        """Column slice of the AR block serving ``regime``.

        Args:
            regime: Regime index, counted from zero.

        Returns:
            A slice of ``order`` columns: the regime's own block when the
            AR coefficients switch, otherwise the shared block.
        """
        base = self.n_intercept
        if self.switching_ar:
            return slice(base + regime * self.order, base + (regime + 1) * self.order)
        return slice(base, base + self.order)
