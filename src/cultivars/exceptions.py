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
"""Public exception and warning hierarchy for cultivars.

Every failure the package detects is raised as one of three
exceptions, and the three are distinguished by *where* the failure was
found rather than by which module found it. :class:`DimensionError` is
the geometry of an input: a rank, a shape, a length that does not
conform. :class:`SpecificationError` is the content of a choice: an
option outside its closed list, an order below its floor, a sample too
short for the depth asked of it, a result handed to a consumer that
does not expose the surface it reads. :class:`NumericalError` is the
arithmetic: a non-finite value, a covariance that is not positive
semidefinite, a singular system, an unstable transition where a
stationary solution was required. A user reading a traceback learns
from the class alone whether to reshape the data, change the
specification, or look at the numbers; the message then names the
argument and the permitted values.

Two commitments shape the surface. First, the hierarchy is complete: all
three derive from :class:`CultivarsError`, so ``except CultivarsError``
catches everything the package raises on purpose and nothing it raises
by accident -- an ``AssertionError`` or a library error escaping from
``numpy`` is a bug and propagates. Second, the hierarchy is compatible:
each exception also derives from the builtin that best matches its
failure mode, ``ValueError`` for the two that concern the argument and
``RuntimeError`` for the one that concerns the computation, so code
written against the builtins keeps working. :class:`StabilityWarning`
stands apart as a ``UserWarning`` because a non-stationary fit is not
a failure; it is a fit whose forecasts and impulse responses mean
something other than their names promise, and a warning is the honest
register for that.

Layout. The four classes are the whole module; there is no
``__init__`` logic, no error-code table, and no message formatting
helper, because the raise sites across ``_core`` and ``_internals``
compose their own messages in one shape -- what was expected, what was
got -- and a central formatter would only hide which check fired.
The validators that raise most of them are in ``_core``
(``validate_endog_matrix``, ``validate_order``, ``validate_choice``,
``validate_transition`` and their kin), and :mod:`~cultivars.typing`
holds the closed lists a :class:`SpecificationError` quotes.

Example:
    The class says what kind of fix is needed; the builtin ancestry
    keeps generic handlers working:

    >>> import numpy as np
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
"""

from __future__ import annotations

__all__ = [
    "CultivarsError",
    "DimensionError",
    "NumericalError",
    "SpecificationError",
    "StabilityWarning",
]


class CultivarsError(Exception):
    """Base class for every exception raised by cultivars.

    Catch this to handle any failure the package itself detects --
    a malformed input, an inconsistent specification, a computation
    that cannot proceed -- while letting genuine bugs (``AssertionError``,
    ``NotImplementedError``, library errors escaping from ``numpy`` or
    ``scipy``) propagate. Never raised directly; every raise site uses
    one of the three subclasses, each of which also derives from the
    builtin that matches its failure mode.

    See Also:
        * :class:`DimensionError` -- the geometry of an input is wrong.
        * :class:`SpecificationError` -- the content of a choice is wrong.
        * :class:`NumericalError` -- the arithmetic could not be carried out.

    Example:
        >>> import numpy as np
        >>> from cultivars.univariate.box_jenkins import ARIMA
        >>> try:
        ...     ARIMA(np.arange(50.0), order=(1, 0, 0), trend="quadratic")
        ... except CultivarsError as error:
        ...     print(type(error).__name__)
        SpecificationError
        >>> CultivarsError.__mro__[1:3]
        (<class 'Exception'>, <class 'BaseException'>)
    """


class DimensionError(CultivarsError, ValueError):
    """An array has the wrong shape, rank, or is non-conformable.

    The input was understood but its geometry does not fit: a series
    where a panel was expected, a coefficient block whose trailing shape
    disagrees with the state dimension, two series of different lengths
    handed to a bivariate test. Also a ``ValueError``, so code that
    already catches that keeps working. Raised almost entirely by the
    shape validators in ``_core`` (``validate_endog``,
    ``validate_endog_matrix``, ``validate_aligned``, ``validate_panel``)
    before any model logic runs.

    See Also:
        * :class:`SpecificationError` -- when the shape is fine and the
          choice is not.

    Example:
        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> VAR(np.zeros((50, 2, 2)), order=1)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.DimensionError: endog must be two-dimensional (nobs, k); got a ...
        >>> try:
        ...     VAR(np.zeros((50, 2, 2)), order=1)
        ... except ValueError as error:
        ...     print(isinstance(error, CultivarsError))
        True
    """


class SpecificationError(CultivarsError, ValueError):
    """A model, polynomial, or coefficient set is invalid or inconsistent.

    The shapes are fine but the choice is not: an unknown categorical
    option, an order below its minimum, a sample too short for the
    requested depth, a result handed to a consumer that does not expose
    the surface it reads, a transition matrix whose rows do not sum to
    one. The most common exception in the package, and the one a
    misspelled keyword lands on. Also a ``ValueError``. The closed lists
    a message quotes are the ``Literal`` aliases in
    :mod:`~cultivars.typing`, and the check that fires is usually
    ``validate_choice``, ``validate_order`` or ``validate_transition``
    in ``_core``.

    See Also:
        * :class:`DimensionError` -- when the choice is fine and the
          shape is not.
        * :mod:`~cultivars.typing` -- the permitted values quoted in the
          messages.

    Example:
        >>> import numpy as np
        >>> from cultivars.univariate.box_jenkins import ARIMA
        >>> ARIMA(np.arange(50.0), order=(1, 0, 0), trend="quadratic")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: trend must be one of ('n', 'c', 'ct'); got ...
        >>> ARIMA(np.arange(50.0), order=(-1, 0, 0))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: ...
    """


class NumericalError(CultivarsError, RuntimeError):
    """A computation failed numerically (non-finite input, singular system).

    The specification and shapes were valid but the arithmetic could
    not be carried out or would return nonsense: a ``nan`` or ``inf``
    in the data or a system matrix, a covariance that is not positive
    semidefinite, a lag polynomial exactly singular at some frequency,
    an unstable transition matrix where a stationary solution was asked
    for, an identification whose restrictions no factorization can
    honor. A ``RuntimeError`` rather than a ``ValueError`` because the
    failure is discovered in the computation, not in the argument, so a
    handler written for bad arguments does not swallow it.

    See Also:
        * :class:`StabilityWarning` -- the warning category for a fit
          that succeeded but is non-stationary, which is not an error.

    Example:
        >>> import numpy as np
        >>> from cultivars.state_space.linear_gaussian import LinearGaussianSSM
        >>> LinearGaussianSSM.stationary_covariance([[1.0]], [[1.0]], [[1.0]])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.NumericalError: stationary_covariance requires a stable transition ...
        >>> try:
        ...     LinearGaussianSSM.stationary_covariance([[1.0]], [[1.0]], [[1.0]])
        ... except ValueError:
        ...     print("argument handler")
        ... except RuntimeError:
        ...     print("computation handler")
        computation handler
    """


class StabilityWarning(UserWarning):
    """The warning category for a fitted model that is non-stationary.

    A fit that converged to a companion matrix with a root on or outside
    the unit circle is still a fit -- the estimates are reported, and the
    result's ``is_stable`` and ``stability_check()`` say what it found
    -- but forecasts, impulse responses and spectral densities from it
    are not the objects their names promise. This is the category a
    layer uses when it chooses to warn about that rather than refuse; a
    ``UserWarning`` rather than a :class:`CultivarsError` because
    nothing has failed. Filter it with :func:`warnings.filterwarnings`
    when the non-stationarity is the point of the exercise. Results
    also report instability in their summary notes, which are not
    affected by the filter.

    See Also:
        * :class:`NumericalError` -- raised, not warned, when a
          computation *requires* stationarity and does not have it.

    Example:
        The category can be silenced or promoted on its own, leaving
        other warnings alone:

        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     warnings.warn("root on the unit circle", StabilityWarning)
        >>> issubclass(caught[0].category, UserWarning), caught[0].category.__name__
        (True, 'StabilityWarning')
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     warnings.filterwarnings("ignore", category=StabilityWarning)
        ...     warnings.warn("root on the unit circle", StabilityWarning)
        ...     warnings.warn("something else", UserWarning)
        >>> [w.category.__name__ for w in caught]
        ['UserWarning']
    """
