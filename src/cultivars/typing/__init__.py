# filepath: /src/cultivars/typing/__init__.py
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
"""Public type aliases for the categorical options a specification takes.

A model specification in this package is a handful of numbers and a
handful of words: an order, a penalty weight, and then ``trend="ct"``,
``vol="GJR"``, ``frequency="flow"``. The numbers are validated by
shape; the words are validated against a closed list, and this package
is where those lists live as types. Each name is a PEP 695 ``type``
alias over a :data:`~typing.Literal`, so a static checker rejects a
misspelled option at the call site of any parameter annotated with it,
and the alias is also what the package's own runtime check reads:
``validate_choice(value, Trend, "trend")`` unwraps the alias to its
permitted values and raises :class:`~cultivars.exceptions.SpecificationError`
on anything else, so the type and the check cannot drift apart. The
one exception is :data:`OptimizerOptions`, an alias of the
``scipy-stubs`` ``TypedDict`` for :func:`scipy.optimize.minimize`'s
options, present so that a mistyped optimizer key is a type error and
not a silently ignored dictionary entry; at runtime it is ``dict``.

The package has one rule that every alias keeps: a name here is a
specification choice with statistical content, never a display option.
:data:`CointegrationTrend` selects one of Johansen's five deterministic
cases and with it the asymptotic distribution of the rank statistics;
:data:`Frequency` decides whether a low-frequency reading pins one
latent value (a stock) or a weighted sum (a flow); :data:`PanelEffects`
decides whether the estimator carries the Nickell bias. Where a family
is a different specification surface rather than another value -- the
fractionally integrated GARCH beside :data:`Vol`, unit-varying slopes
beside :data:`PanelEffects` -- it is deliberately not a value, and the
alias's docstring says so.

Each name is imported from this package: ``from cultivars.typing import
Trend``. Nothing here is a class to instantiate or a function to call;
the aliases are annotations and validator arguments, and their
docstrings are the reference for what each option means. They are
re-exported from ``_core._types``, where the private aliases the
internals use beside them also live.

Layout, by the surface each alias governs. Univariate conditional mean:
:data:`Trend` (``"n"``, ``"c"``, ``"ct"``), :data:`Method` (``"css"``,
``"exact"``), :data:`Transition` (LSTAR or ESTAR), :data:`Activation`
(neural mean functions), :data:`LongMemoryMethod` (``"gph"``,
``"local_whittle"``). Conditional variance: :data:`Mean` and
:data:`Vol`. Multivariate: :data:`CointegrationTrend` (Johansen's
cases), :data:`PanelEffects`, :data:`Penalty` (sparse-VAR families),
:data:`Frequency` (mixed-frequency observation types),
:data:`FunctionalBasis` (curve bases). Switching and observed-regime
models: :data:`ProbabilityType` (``"smoothed"``, ``"filtered"``,
``"predicted"``) and :data:`Regime` (``"lower"``, ``"upper"``).
Optimization: :data:`OptimizerMethod` and :data:`OptimizerOptions`.

Example:
    An alias annotates a variable, enumerates its own values, and
    validates at runtime through the same object:

    >>> from typing import get_args
    >>> from cultivars._core import validate_choice
    >>> trend: Trend = "ct"
    >>> get_args(Trend.__value__)
    ('n', 'c', 'ct')
    >>> validate_choice(trend, Trend, "trend")
    'ct'
    >>> validate_choice("quadratic", Trend, "trend")
    Traceback (most recent call last):
        ...
    cultivars.exceptions.SpecificationError: trend must be one of ('n', 'c', 'ct'); got 'quadratic'.
    >>> get_args(CointegrationTrend.__value__)
    ('none', 'restricted_constant', 'constant', 'restricted_trend', 'trend')
"""

from ..engine._core import (
    Activation,
    CointegrationTrend,
    Frequency,
    FunctionalBasis,
    LongMemoryMethod,
    Mean,
    Method,
    OptimizerMethod,
    OptimizerOptions,
    PanelEffects,
    Penalty,
    ProbabilityType,
    Regime,
    Transition,
    Trend,
    Vol,
)

__all__ = [
    "Activation",
    "CointegrationTrend",
    "Frequency",
    "FunctionalBasis",
    "LongMemoryMethod",
    "Mean",
    "Method",
    "OptimizerMethod",
    "OptimizerOptions",
    "PanelEffects",
    "Penalty",
    "ProbabilityType",
    "Regime",
    "Transition",
    "Trend",
    "Vol",
]
