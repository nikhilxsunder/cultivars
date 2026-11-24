# filepath: /src/cultivars/exceptions.py
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
"""The exception and warning hierarchy every module raises from.

Every failure the package detects on purpose is one of three exceptions,
told apart by where the failure lies rather than by which module found it.
:class:`DimensionError` is the geometry of an input: a rank, a shape or a
length that does not conform. :class:`SpecificationError` is the content of
a choice: an option outside its closed list, an order below its floor, a
sample too short for the depth asked of it. :class:`NumericalError` is the
arithmetic: a non-finite value, a covariance that is not positive
semidefinite, a singular system. All three derive from
:class:`CultivarsError`, so ``except CultivarsError`` catches everything
the package raises deliberately, and each also derives from the builtin
that matches its failure mode (``ValueError`` for the first two,
``RuntimeError`` for the third), so code written against the builtins keeps
working. :class:`StabilityWarning` and :class:`ExperimentalWarning` are
``UserWarning`` subclasses for a fit that is usable but not what its name
promises, and for the unstable surface of :mod:`cultivars.experimental`.

The classes are defined in ``cultivars_engine.exceptions`` and this module
is their public name. They are the same objects, not subclasses: an
``except`` clause written against either module catches an error raised
through the other, and a traceback prints the defining module,
``cultivars_engine.exceptions``.

Example:
    The class says what kind of fix is needed, and the builtin ancestry
    keeps generic handlers working:

    >>> import numpy as np
    >>> from cultivars.exceptions import CultivarsError, SpecificationError
    >>> from cultivars.univariate.box_jenkins import ARIMA
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> attempts = (
    ...     lambda: VAR(np.zeros((50, 2, 2)), order=1),
    ...     lambda: ARIMA(np.arange(50.0), order=(1, 0, 0), trend="quadratic"),
    ...     lambda: ARIMA(np.r_[np.arange(49.0), np.nan], order=(1, 0, 0)),
    ... )
    >>> for attempt in attempts:
    ...     try:
    ...         attempt()
    ...     except CultivarsError as error:
    ...         print(type(error).__name__, isinstance(error, ValueError))
    DimensionError True
    SpecificationError True
    NumericalError False

    The public name and the engine's are one class:

    >>> import cultivars_engine.exceptions
    >>> SpecificationError is cultivars_engine.exceptions.SpecificationError
    True
"""

from __future__ import annotations

from cultivars_engine.exceptions import (
    CultivarsError,
    DimensionError,
    ExperimentalWarning,
    NumericalError,
    SpecificationError,
    StabilityWarning,
)

__all__ = [
    "CultivarsError",
    "DimensionError",
    "ExperimentalWarning",
    "NumericalError",
    "SpecificationError",
    "StabilityWarning",
]
