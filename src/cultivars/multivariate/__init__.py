# filepath: /src/cultivars/multivariate/__init__.py
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
r"""Multivariate time series: one closed-system contract, five families on it.

Every model in this package estimates how a vector of series moves
together, and every one of them is organized around a single contract.
A fitted reduced form is a *closed system*: it exposes its variable
names, its coefficient stack, its innovation covariance
:math:`\Sigma_u`, its residuals and its moving-average representation
:math:`\Psi_h`, and everything downstream -- structural identification,
regime views, connectedness networks -- consumes that protocol rather
than a particular class. The families differ in what they assume about
the coefficients. :mod:`~cultivars.multivariate.reduced_form` is the
least-squares core: the VAR and VARX, the error-correction form for
cointegrated levels, the vector moving average, mixed frequencies,
panels, the closed global VAR, functional and term-structure designs,
each a different :math:`X` on one estimator.
:mod:`~cultivars.multivariate.large_dim` is what to do when :math:`p k^2`
coefficients outrun the sample: shrink them toward a prior, set most of
them to zero, or factor the panel. :mod:`~cultivars.multivariate.nonlinear`
lets the coefficients depend on an observable state, a quantile level or
the date. :mod:`~cultivars.multivariate.regime_switching` lets them
switch with a latent Markov chain, and returns a posterior over regimes
rather than a chronology. :mod:`~cultivars.multivariate.structural` is
where the shocks acquire names: declared zeros, declared signs, outside
instruments, or statistical properties of the innovations, applied to any
closed system, plus the DSGE whose structure is a solved model.

Two commitments run through every family. First, estimation and
identification are separate acts. A reduced form is fitted once and
reports only what the data determine; a structural scheme is constructed
*from* that result, declares what it adds, and restates the declaration
in its summary, so an impulse response can always be traced back to the
assumption that signed it. Second, a result is an immutable record that
renders the same way everywhere and refuses what it cannot support: a
partially identified scheme carries only the columns it identified, a
set-identified scheme returns the set, a mixture refuses the regime-count
test its likelihood cannot answer, and a long-run restriction refuses a
unit root. The packages re-export modules only; import a model from its
module.

Example:
    A reduced form fitted once, then read three ways -- its own
    orthogonalized responses, a declared recursive identification, and
    a Diebold-Yilmaz connectedness view -- all from the same record:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR
    >>> from cultivars.multivariate.large_dim.spillover import Spillover
    >>> rng = np.random.default_rng(0)
    >>> B = np.array([[1.0, 0.0], [0.5, 1.0]])
    >>> y = np.zeros((402, 2))
    >>> for t in range(1, 402):
    ...     y[t] = 0.4 * y[t - 1] + B @ rng.standard_normal(2)
    >>> res = VAR(y, order=1, names=("r", "x")).fit()
    >>> svar = RecursiveSVAR(res).identify()
    >>> bool(np.allclose(svar.irf(4), res.irf(4, orthogonalized=True)))
    True
    >>> network = Spillover(res, horizon=10).compute()
    >>> network.table.shape, bool(0.0 <= network.total <= 100.0)
    ((2, 2), True)
"""

from . import large_dim, nonlinear, reduced_form, regime_switching, structural

__all__ = [
    "large_dim",
    "nonlinear",
    "reduced_form",
    "regime_switching",
    "structural",
]
