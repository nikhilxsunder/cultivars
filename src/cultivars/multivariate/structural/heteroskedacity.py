# filepath: /src/cultivars/multivariate/structural/heteroskedacity.py
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
r"""Deprecated alias of :mod:`~cultivars.multivariate.structural.heteroskedasticity`.

The module was published under a misspelled name. This alias keeps that
import path working for one release and emits a :class:`FutureWarning` on
import; it will be removed before 1.0.0b1. Import from
:mod:`~cultivars.multivariate.structural.heteroskedasticity` instead.

Example:
    The old path warns and resolves to the same class:

    >>> import importlib, sys, warnings
    >>> name = "cultivars.multivariate.structural.heteroskedacity"
    >>> _ = sys.modules.pop(name, None)
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     old = importlib.import_module(name)
    >>> [w.category.__name__ for w in caught]
    ['FutureWarning']
    >>> from cultivars.multivariate.structural import heteroskedasticity
    >>> old.HeteroskedasticSVAR is heteroskedasticity.HeteroskedasticSVAR
    True
"""

from __future__ import annotations

import warnings
from pathlib import Path

from .heteroskedasticity import HeteroskedasticSVAR

__all__ = ["HeteroskedasticSVAR"]

# Skip every frame inside this package, so the warning names the user's line
# whether the alias was imported by name or reached through the package's
# ``__getattr__``.
warnings.warn(
    "cultivars.multivariate.structural.heteroskedacity is deprecated and will be "
    "removed before 1.0.0b1; import from "
    "cultivars.multivariate.structural.heteroskedasticity instead.",
    FutureWarning,
    skip_file_prefixes=(str(Path(__file__).parent),),
)
