# filepath: /src/cultivars/multivariate/structural/__init__.py
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
r"""Structural identification: what the data cannot supply, declared or tested.

A fitted closed system delivers its innovation covariance :math:`\Sigma_u`
and nothing more about the shocks behind it; every factorization
:math:`\Sigma_u = B B'` is a different economic story, and choosing one is
the whole of structural analysis. Each module here is one way of choosing,
and the package is organized by where the identifying information comes
from. Declared zeros are
:mod:`~cultivars.multivariate.structural.zero_restrictions` -- a causal
ordering on impact, an ordering of permanence at the long run, an AB
pattern anywhere, or zeros split across the two horizons -- and they
point-identify. Declared signs are
:mod:`~cultivars.multivariate.structural.sign_restrictions` and
:mod:`~cultivars.multivariate.structural.set_identification`: they
set-identify, and the two modules report the set honestly in two different
ways, sampled under a named prior or bounded exactly with none. Outside
series are :mod:`~cultivars.multivariate.structural.external_instruments`, a
proxy that identifies one column and reports its own strength. Statistical
properties of the shocks are
:mod:`~cultivars.multivariate.structural.heteroskedacity`,
:mod:`~cultivars.multivariate.structural.stochastic_volatility` and
:mod:`~cultivars.multivariate.structural.non_gaussian` -- declared variance
regimes, estimated variance paths, and independence with non-Gaussianity --
which identify without any economic restriction and return shocks labelled
by statistics, never by economics. A larger information set is
:mod:`~cultivars.multivariate.structural.factor_augmented`, the FAVAR, where
the structural shock propagates through a panel. And a model is
:mod:`~cultivars.multivariate.structural.perturbation`, where the structure
is a DSGE solved to first or second order and estimated by likelihood,
rather than restrictions laid over a reduced form.

Every scheme that point-identifies returns the one
:class:`~cultivars.multivariate.structural.zero_restrictions.SVARResult`,
with exactly as many shock columns as it identified and the identifying
assumption restated in its summary; every scheme that set-identifies returns
the set itself, as draws or as bounds; every scheme whose identification is
a statistical condition tests that condition on the data and prints the
verdict, because the posterior or the point estimate alone cannot certify
it. The package re-exports modules only: import a scheme from its module.

Example:
    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> from cultivars.multivariate.structural import zero_restrictions, sign_restrictions
    >>> rng = np.random.default_rng(0)
    >>> B = np.array([[1.0, 0.0], [0.5, 1.0]])
    >>> y = np.zeros((402, 2))
    >>> for t in range(1, 402):
    ...     y[t] = 0.4 * y[t - 1] + B @ rng.standard_normal(2)
    >>> res = VAR(y, order=1, names=("r", "x")).fit()
    >>> point = zero_restrictions.RecursiveSVAR(res).identify()
    >>> point.scheme, point.is_complete, point.impact.round(2).tolist()
    ('recursive', True, [[1.0, 0.0], [0.5, 1.0]])
    >>> signs = {"demand": {"r": "+", "x": "+"}}
    >>> sset = sign_restrictions.SignRestrictedSVAR(res, signs, draws=100, seed=0).identify()
    >>> sset.n_accepted, sset.shock_names
    (100, ('demand', 'unrestricted1'))
"""

from . import (
    external_instruments,
    factor_augmented,
    heteroskedacity,
    non_gaussian,
    perturbation,
    set_identification,
    sign_restrictions,
    stochastic_volatility,
    zero_restrictions,
)

__all__ = [
    "external_instruments",
    "factor_augmented",
    "heteroskedacity",
    "non_gaussian",
    "perturbation",
    "set_identification",
    "sign_restrictions",
    "stochastic_volatility",
    "zero_restrictions",
]
