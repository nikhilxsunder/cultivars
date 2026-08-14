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
