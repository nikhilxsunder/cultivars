# filepath: /src/cultivars/typing.py
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
"""Type aliases for the categorical options a specification takes.

A model specification in this package is a handful of numbers and a handful
of words: an order, a penalty weight, and then ``trend="ct"``,
``vol="GJR"``, ``frequency="flow"``. The numbers are validated by shape;
the words are validated against a closed list, and this module is where
those lists are named. Each name is a PEP 695 ``type`` alias over a
:data:`~typing.Literal`, so a static checker rejects a misspelled option at
the call site of any parameter annotated with it, and the same alias is
what the runtime check reads before it raises
:class:`~cultivars.exceptions.SpecificationError`, so the type and the
check cannot drift apart. :data:`OptimizerOptions` is the one alias that is
not a closed list: it stands for the options dictionary of
:func:`scipy.optimize.minimize`.

The aliases are defined in ``cultivars_engine.typing`` and this module is
their public name; they are the same objects. Nothing here is a class to
instantiate or a function to call: the names are annotations, and their
docstrings are the reference for what each option means.

Layout, by the surface each alias governs. Univariate conditional mean:
:data:`Trend`, :data:`Method`, :data:`Transition`, :data:`Activation`,
:data:`LongMemoryMethod`. Conditional variance: :data:`Mean` and
:data:`Vol`. Multivariate: :data:`CointegrationTrend`,
:data:`PanelEffects`, :data:`Penalty`, :data:`Frequency`,
:data:`FunctionalBasis`. Switching and observed-regime models:
:data:`ProbabilityType` and :data:`Regime`. Optimization:
:data:`OptimizerMethod` and :data:`OptimizerOptions`. Structural and
testing protocols: :class:`Identification`, :data:`Residuals` and
:data:`FTest`.

Example:
    An alias annotates a variable and enumerates its own values:

    >>> from typing import get_args
    >>> from cultivars.typing import CointegrationTrend, Trend
    >>> trend: Trend = "ct"
    >>> get_args(Trend.__value__)
    ('n', 'c', 'ct')
    >>> get_args(CointegrationTrend.__value__)
    ('none', 'restricted_constant', 'constant', 'restricted_trend', 'trend')
"""

from __future__ import annotations

from cultivars_engine.typing import (
    Activation,
    CointegrationTrend,
    Frequency,
    FTest,
    FunctionalBasis,
    Identification,
    LongMemoryMethod,
    Mean,
    Method,
    OptimizerMethod,
    OptimizerOptions,
    PanelEffects,
    Penalty,
    ProbabilityType,
    Regime,
    Residuals,
    Transition,
    Trend,
    Vol,
)

__all__ = [
    "Activation",
    "CointegrationTrend",
    "FTest",
    "Frequency",
    "FunctionalBasis",
    "Identification",
    "LongMemoryMethod",
    "Mean",
    "Method",
    "OptimizerMethod",
    "OptimizerOptions",
    "PanelEffects",
    "Penalty",
    "ProbabilityType",
    "Regime",
    "Residuals",
    "Transition",
    "Trend",
    "Vol",
]
