# filepath: /src/cultivars/bayes/chains.py
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

"""Convergence diagnostics for Markov chain Monte Carlo output.

Every sampled result in the package -- the stochastic-volatility posteriors,
the Gibbs, Student-t, stochastic-volatility and hierarchical BVARs, the
time-varying-parameter models, the factor and structural volatility models,
the particle-chain DSGE posterior -- carries a ``convergence()`` method that
assesses its retained draws and returns a :class:`ConvergenceTest`. This
module is the same machinery for draws that did not come from the package:
a chain from another sampler, or a function of the package's draws (an
impulse response at a horizon, a variance ratio) whose own convergence is
the question.

The statistics are rank-normalized split-R-hat with folding, bulk and tail
effective sample sizes by Geyer's initial monotone sequence, the Monte Carlo
standard error of the mean, and Geweke's early-versus-late score, following
Vehtari, Gelman, Simpson, Carpenter, and Bürkner (2021). Each function takes
one or more chains of the same quantity: a chain is ``(S,)`` for a scalar or
``(S, d)`` for ``d`` quantities drawn jointly, and several chains are several
positional arguments from independent runs.

A single chain is assessed by splitting it in half, which detects drift but
cannot detect a chain confined to one of several posterior modes. A second
run from a different seed is what detects that, and passing it is the
recommended practice rather than an option.

Example:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chain_a = rng.standard_normal((2000, 2))
    >>> chain_b = rng.standard_normal((2000, 2))
    >>> report = convergence(chain_a, chain_b, names=("alpha", "beta"))
    >>> report.converged
    True
    >>> report.n_chains, report.n_draws
    (2, 2000)
    >>> stuck = np.stack([chain_a[:, 0], chain_b[:, 0] + 4.0])
    >>> rhat(stuck[0], stuck[1]) > 1.5
    True
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .._core import (
    _MIN_ESS_PER_CHAIN,
    _RHAT_TOL,
    _ess_bulk,
    _ess_tail,
    _geweke,
    _mcse_mean,
    _per_column,
    _rhat,
    _stack,
)
from .._internals import _ConvergenceTest as ConvergenceTest
from ..exceptions import DimensionError

__all__ = ["convergence", "ess_bulk", "ess_tail", "geweke", "mcse", "rhat"]


def rhat(*chains: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
    """Rank-normalized split-R-hat with folding.

    Args:
        *chains: One or more ``(S,)`` or ``(S, d)`` chains of the same
            quantity from independent runs.

    Returns:
        A float for scalar chains, else ``(d,)``. ``nan`` where a quantity
        never moved. Values above about 1.01 indicate the chains disagree.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> float(rhat(rng.standard_normal(1000), rng.standard_normal(1000))) < 1.02
        True
    """
    return _per_column(chains, _rhat)


def ess_bulk(*chains: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
    """Bulk effective sample size, for central posterior quantities.

    Args:
        *chains: One or more ``(S,)`` or ``(S, d)`` chains.

    Returns:
        A float for scalar chains, else ``(d,)``; the total across chains.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(2)
        >>> 1400 < float(ess_bulk(rng.standard_normal(2000))) < 2600
        True
    """
    return _per_column(chains, _ess_bulk)


def ess_tail(*chains: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
    """Tail effective sample size, for credible-interval endpoints.

    The smaller of the effective sizes of the 5% and 95% quantile
    indicators.

    Args:
        *chains: One or more ``(S,)`` or ``(S, d)`` chains.

    Returns:
        A float for scalar chains, else ``(d,)``; the total across chains.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(3)
        >>> float(ess_tail(rng.standard_normal(2000))) > 800
        True
    """
    return _per_column(chains, _ess_tail)


def mcse(*chains: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
    """Monte Carlo standard error of the posterior mean.

    Args:
        *chains: One or more ``(S,)`` or ``(S, d)`` chains.

    Returns:
        A float for scalar chains, else ``(d,)``.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(4)
        >>> round(float(mcse(rng.standard_normal(10_000))), 1)
        0.0
    """
    return _per_column(chains, _mcse_mean)


def geweke(
    *chains: npt.ArrayLike, first: float = 0.1, last: float = 0.5
) -> npt.NDArray[np.float64]:
    """Geweke's early-versus-late z-score, one per chain and quantity.

    Args:
        *chains: One or more ``(S,)`` or ``(S, d)`` chains.
        first: Fraction of each chain forming the early segment.
        last: Fraction forming the late segment.

    Returns:
        ``(C,)`` for scalar chains, else ``(C, d)``. Magnitudes above about
        2 indicate the early and late segments disagree in mean.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(5)
        >>> geweke(rng.standard_normal(1000), rng.standard_normal(1000)).shape
        (2,)
    """
    stack, scalar = _stack(chains)
    scores = np.stack(
        [_geweke(stack[:, :, j], first=first, last=last) for j in range(stack.shape[2])], axis=1
    )
    return scores[:, 0] if scalar else scores


def convergence(
    *chains: npt.ArrayLike,
    names: Sequence[str] | None = None,
    rhat_tol: float = _RHAT_TOL,
    min_ess: float = _MIN_ESS_PER_CHAIN,
    source: str = "draws",
) -> ConvergenceTest:
    """Full per-quantity report over one or more chains.

    Args:
        *chains: One or more ``(S,)`` or ``(S, d)`` chains from independent
            runs.
        names: ``d`` labels; defaults to ``x[0] .. x[d-1]``, or ``x`` for a
            scalar chain.
        rhat_tol: R-hat above which a quantity is flagged.
        min_ess: Effective draws per chain below which a quantity is
            flagged.
        source: Title for the report.

    Returns:
        The report; ``report.summary()`` renders it.

    Raises:
        DimensionError: If ``names`` does not match the number of quantities.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(6)
        >>> report = convergence(rng.standard_normal((2000, 3)))
        >>> report.names
        ('x[0]', 'x[1]', 'x[2]')
        >>> report.flagged
        ()
    """
    stack, scalar = _stack(chains)
    d = stack.shape[2]
    if names is None:
        labels: tuple[str, ...] = ("x",) if scalar else tuple(f"x[{j}]" for j in range(d))
    else:
        labels = tuple(names)
        if len(labels) != d:
            raise DimensionError(f"Got {len(labels)} names for {d} quantities.")
    return ConvergenceTest._assess(
        stack, names=labels, rhat_tol=rhat_tol, min_ess=min_ess, source=source
    )
