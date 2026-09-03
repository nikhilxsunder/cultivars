# filepath: /src/cultivars/_core/_types.py
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

"""Public type aliases for categorical specification arguments.

These are PEP 695 alias statements, so they are lazily evaluated and carry
no import-time cost. Every model constructor that takes a categorical
option annotates it with one of these rather than a bare ``str``, which is
what lets ``mypy`` reject a misspelled option at the call site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from scipy.optimize._minimize import _MinimizeOptions as OptimizerOptions
else:
    OptimizerOptions = dict
    """Keyword options accepted by :func:`scipy.optimize.minimize`.

    Aliased from the ``scipy-stubs`` ``TypedDict`` so that a mistyped option key or
    value is a type error rather than a silently ignored dictionary entry. The name
    is private to the stubs and does not exist at runtime, hence the guard; the
    runtime fallback keeps ``typing.get_type_hints`` working for Sphinx autodoc.
    """

type CointegrationTrend = Literal[
    "none", "restricted_constant", "constant", "restricted_trend", "trend"
]
"""Where a deterministic term sits relative to the cointegrating space.

Johansen's five cases. A constant or trend can enter inside the cointegrating
relations, outside them in the short-run equation, or not at all, and the three
say different things about the long run: a constant restricted to the
cointegrating space allows the relations a non-zero equilibrium level without
giving the levels a drift, while an unrestricted one does both. The asymptotic
distribution of the rank statistics depends on which of the five holds, so this
is not a display option.
"""

type OptimizerMethod = Literal["L-BFGS-B", "Nelder-Mead"]
"""Numerical minimizer an objective surface asks :func:`scipy.optimize.minimize` for.

Deliberately narrower than the full scipy list: an objective that needs a third
algorithm adds it here, which keeps the set of minimizers the package actually
exercises visible in one place.
"""

type Method = Literal["css", "exact"]
"""Estimator for a conditional-mean model: conditional sum of squares or exact ML."""

type Trend = Literal["n", "c", "ct"]
"""Deterministic terms: none, constant, or constant plus linear trend."""

type Mean = Literal["constant", "zero"]
"""Conditional-mean specification for a volatility model."""

type Vol = Literal["GARCH", "GJR", "EGARCH"]
"""Finite-order conditional-variance family.

The fractionally integrated family is deliberately absent: its order is fixed
at ``(1, d, 1)`` and it takes a truncation lag instead of ``p``, ``o``, ``q``,
so it is a different specification surface rather than another value of this
option.
"""

type Transition = Literal["logistic", "exponential"]
"""Smooth-transition function: logistic (LSTAR) or exponential (ESTAR)."""

type Activation = Literal["tanh", "relu"]
"""Hidden-layer activation for a neural mean function."""

type LongMemoryMethod = Literal["gph", "local_whittle"]
"""Semiparametric estimator for the fractional differencing parameter."""

type ProbabilityType = Literal["smoothed", "filtered", "predicted"]
"""The three posteriors a Hamilton filter and Kim smoother produce."""

type PanelEffects = Literal["none", "unit"]
"""Which intercepts a panel vector autoregression lets vary across units.

``"unit"`` estimates one intercept per unit as a dummy block, which is the
least-squares dummy-variable estimator and carries the Nickell bias. ``"none"``
pools the deterministic terms, which is only defensible when the units are
known to share a mean. Slopes are pooled under both -- unit-varying dynamics
are a different specification, not a value of this option.
"""

type Frequency = Literal["high", "stock", "flow"]
"""How often a variable is seen, and what a low-frequency reading measures.

``"high"`` is observed every sub-period. ``"stock"`` is a level read at the end
of the low-frequency period -- a survey balance, an end-of-quarter position --
so the reading is the last sub-period's value and nothing else. ``"flow"`` is
accumulated across the period -- output, spending, hours -- so the reading is a
weighted combination of every sub-period in it.

The distinction is not presentational. A stock reading pins down one latent
value; a flow reading pins down a weighted sum and leaves the individual
sub-periods free to move against each other. Treating a flow as a stock throws
away two thirds of a quarterly constraint and silently changes what the model
is fitted to.
"""

type FunctionalBasis = Literal["fpca", "nelson-siegel", "bspline"]

type Regime = Literal["lower", "upper"]
"""Which regime of an observed-regime model an operation reads."""
