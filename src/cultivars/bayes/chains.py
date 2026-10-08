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
r"""Convergence diagnostics for Markov chain Monte Carlo output.

A sampler's draws are only a posterior once the chain has forgotten where
it started and has visited the distribution in proportion, and neither is
observable directly. The diagnostics here are the standard indirect
evidence, following Vehtari, Gelman, Simpson, Carpenter, and Bürkner
(2021): rank-normalized split-:math:`\widehat{R}` with folding, which
compares chains, or the two halves of one chain, on location and on
scale; bulk and tail effective sample sizes by Geyer's initial monotone
sequence, which say how many independent draws the correlated ones are
worth for a mean and for an interval endpoint; the Monte Carlo standard
error of the mean; and Geweke's early-versus-late score, which asks
whether the burn-in was long enough. For :math:`S` pooled draws with
integrated autocorrelation time :math:`\tau`, the effective size is
:math:`S / \tau`, and the working thresholds are :math:`\widehat{R} \le
1.01` and at least 100 effective draws per chain.

Every sampled result in the package -- the stochastic-volatility
posteriors, the Gibbs, Student-t, stochastic-volatility, and hierarchical
BVARs, the time-varying-parameter models, the factor and structural
volatility models, the particle-chain DSGE posterior -- carries a
``convergence()`` method that assesses its retained draws and returns a
:class:`ConvergenceTest`. This module is the same machinery for draws that
did not come from the package: a chain from another sampler, or a
function of the package's draws (an impulse response at a horizon, a
variance ratio, a ratio of two parameters) whose own convergence is the
question. Each function takes one or more chains of the same quantity: a
chain is ``(S,)`` for a scalar or ``(S, d)`` for ``d`` quantities drawn
jointly, and several chains are several positional arguments from
independent runs.

Two commitments shape the surface. First, a single chain is assessed by
splitting it in half, which detects drift but cannot detect a chain
confined to one of several posterior modes; a second run from a different
seed is what detects that, and passing one is the recommended practice
rather than an option, so every function takes ``*chains``. Second, a
quantity whose draws never moved is reported as degenerate and is neither
passed nor failed: the diagnostics cannot tell a parameter fixed by
construction from a sampler that is stuck, and the report says which
quantities it declined to judge rather than guessing.

Layout. The six public functions are thin: each stacks its chains through
``_core``'s ``_stack`` and applies a per-quantity statistic from
``_core._estimators`` -- ``_rhat``, ``_ess_bulk``, ``_ess_tail``,
``_mcse_mean``, ``_geweke`` -- which in turn share
``_potential_scale_reduction``, ``_effective_sample_size``, and the
FFT ``_autocovariance``; the rank normalization and the split are
``_core._transforms``. :func:`convergence` hands the stack to
:meth:`ConvergenceTest._assess`, the classmethod on the frozen record in
``_internals``, which computes every column and applies the thresholds;
the ``convergence()`` method of every sampled result reaches the same
classmethod through :class:`~cultivars._internals._ConvergenceMixin`, so
a report built here and one built by a result are the same object and
render the same table. The thresholds ``_RHAT_TOL`` and
``_MIN_ESS_PER_CHAIN``, and the eight-draw minimum chain length
``_MIN_CHAIN_DRAWS``, are ``_core`` defaults.

References:
    Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation
    using multiple sequences. *Statistical Science*, 7(4), 457-472.

    Geweke, J. (1992). Evaluating the accuracy of sampling-based approaches
    to the calculation of posterior moments. In J. M. Bernardo, J. O.
    Berger, A. P. Dawid, & A. F. M. Smith (Eds.), *Bayesian Statistics 4*
    (pp. 169-193). Oxford University Press.

    Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical
    Science*, 7(4), 473-483.

    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C.
    (2021). Rank-normalization, folding, and localization: An improved
    :math:`\widehat{R}` for assessing convergence of MCMC. *Bayesian
    Analysis*, 16(2), 667-718.

Example:
    Two runs of two quantities, assessed together:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chain_a = rng.standard_normal((2000, 2))
    >>> chain_b = rng.standard_normal((2000, 2))
    >>> report = convergence(chain_a, chain_b, names=("alpha", "beta"))
    >>> report.converged
    True
    >>> report.n_chains, report.n_draws
    (2, 2000)

    Two runs that settled in different places are caught by
    :math:`\widehat{R}`, which is what the second seed is for:

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

__all__ = ["ConvergenceTest", "convergence", "ess_bulk", "ess_tail", "geweke", "mcse", "rhat"]


def rhat(*chains: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
    r"""Rank-normalized split-:math:`\widehat{R}` with folding, per quantity.

    The potential scale reduction of Gelman and Rubin (1992) compares the
    spread of chain means with the spread within chains: for :math:`M`
    chains of :math:`N` draws with within-chain variance :math:`W` and
    between-chain variance :math:`B`,

    .. math::

       \widehat{R} = \sqrt{\frac{\widehat{\operatorname{var}}^{+}}{W}},
       \qquad
       \widehat{\operatorname{var}}^{+} = \frac{N - 1}{N}\, W + \frac{1}{N}\, B .

    Three modifications from Vehtari et al. (2021) make the statistic
    usable as a stopping rule. Each chain is split in half before the
    comparison, so :math:`M = 2C` for :math:`C` supplied chains and a single
    run still yields a between-chain term, which is what detects a chain
    that is still drifting. The draws are replaced by their normal scores,
    :math:`z = \Phi^{-1}\!\bigl((r - 3/8) / (S + 1/4)\bigr)` for the pooled
    rank :math:`r` among all :math:`S` draws, which makes the statistic
    invariant to monotone transformations and defined for quantities with
    no finite variance. And the statistic is computed a second time on the
    scores of :math:`|x - \operatorname{median}(x)|`; this folded version
    is sensitive to chains that agree in location but not in scale, which
    the plain version misses. The larger of the two is reported.

    Args:
        *chains: One or more chains of the same quantity from independent
            runs, each ``(S,)`` for a scalar or ``(S, d)`` for ``d``
            quantities drawn jointly. All chains must have the same shape,
            with ``S`` at least eight and finite throughout.

    Returns:
        A float for ``(S,)`` chains, else ``(d,)``. ``nan`` for a quantity
        whose draws never moved; the report built by :func:`convergence`
        lists such quantities as degenerate rather than counting them for
        or against the verdict. Values above about 1.01 indicate the chains,
        or the halves of a chain, disagree.

    Raises:
        SpecificationError: If no chain is given, or a chain is shorter than
            eight draws.
        DimensionError: If a chain is not one- or two-dimensional, or the
            chains differ in shape.
        NumericalError: If any draw is not finite.

    Note:
        With a single chain the between-chain term compares its two halves,
        which detects drift but not a chain confined to one of several
        posterior modes. The two half-chains are also short: an independent
        sequence of 200 draws exceeds 1.01 about a fifth of the time, one of
        500 about a fiftieth. A value near the threshold from one short
        chain is a reason to run longer or add a second seed, not a verdict.

    See Also:
        :func:`ess_bulk`: The companion size statistic on the same
            rank-normalized split chains.
        :func:`convergence`: The full report, which applies the threshold.

    References:
        Gelman, A., & Rubin, D. B. (1992). Inference from iterative
            simulation using multiple sequences. *Statistical Science*,
            7(4), 457-472.
        Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner,
            P.-C. (2021). Rank-normalization, folding, and localization: An
            improved :math:`\widehat{R}` for assessing convergence of MCMC.
            *Bayesian Analysis*, 16(2), 667-718.

    Example:
        Two runs that target the same distribution agree:

        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> float(rhat(rng.standard_normal(1000), rng.standard_normal(1000))) < 1.02
        True

        Two runs stuck in different places do not:

        >>> float(rhat(rng.standard_normal(1000), 3.0 + rng.standard_normal(1000))) > 1.5
        True

        Rank normalization makes the statistic indifferent to heavy tails,
        so a Cauchy quantity is assessed like any other:

        >>> float(rhat(rng.standard_cauchy(2000), rng.standard_cauchy(2000))) < 1.02
        True
    """
    return _per_column(chains, _rhat)


def ess_bulk(*chains: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
    r"""Bulk effective sample size, for central posterior quantities.

    The number of independent draws that would carry the same information
    about the centre of the posterior as the correlated draws supplied.
    With :math:`S` pooled draws and integrated autocorrelation time
    :math:`\tau`,

    .. math::

       \widehat{S}_{\text{eff}} = \frac{S}{\hat\tau},
       \qquad
       \hat\tau = -1 + 2 \sum_{t=0}^{K} \hat P_t,
       \qquad
       \hat P_t = \hat\rho_{2t} + \hat\rho_{2t+1},

    where :math:`\hat\rho_t` is the autocorrelation at lag :math:`t` pooled
    across the split chains through the between-chain variance, so that a
    chain which has not mixed shows as a slowly decaying correlation rather
    than as a small within-chain variance. The truncation :math:`K` is
    Geyer's (1992) initial monotone sequence: the paired sums are cut at
    the first non-positive pair and then forced non-increasing. The
    computation is on the rank-normalized split chains of :func:`rhat`, and
    the estimate is capped at :math:`S \log_{10} S`, since an antithetic
    chain can otherwise report more effective draws than the estimator's
    own precision supports.

    Args:
        *chains: One or more ``(S,)`` or ``(S, d)`` chains of the same
            shape, with ``S`` at least eight and finite throughout.

    Returns:
        A float for ``(S,)`` chains, else ``(d,)``; the total across chains,
        not per chain. ``nan`` for a quantity that never moved.

    Raises:
        SpecificationError: If no chain is given, or a chain is shorter than
            eight draws.
        DimensionError: If a chain is not one- or two-dimensional, or the
            chains differ in shape.
        NumericalError: If any draw is not finite.

    Note:
        Vehtari et al. (2021) recommend at least 100 effective draws per
        chain before any other diagnostic is read, because
        :math:`\widehat{R}` and the standard errors are themselves noisy
        below that. The threshold :func:`convergence` applies is
        ``min_ess`` times the number of chains.

    See Also:
        :func:`ess_tail`: The same estimator on quantile indicators, for
            interval endpoints.
        :func:`mcse`: The standard error that the untransformed version of
            this size implies for the posterior mean.

    References:
        Geyer, C. J. (1992). Practical Markov chain Monte Carlo.
            *Statistical Science*, 7(4), 473-483.
        Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner,
            P.-C. (2021). Rank-normalization, folding, and localization: An
            improved :math:`\widehat{R}` for assessing convergence of MCMC.
            *Bayesian Analysis*, 16(2), 667-718.

    Example:
        Independent draws have an effective size close to their count:

        >>> import numpy as np
        >>> rng = np.random.default_rng(2)
        >>> 1400 < float(ess_bulk(rng.standard_normal(2000))) < 2600
        True

        A first-order autoregression with coefficient :math:`\phi` has
        :math:`\tau = (1 + \phi) / (1 - \phi)`, so at :math:`\phi = 0.9`
        the 2000 draws are worth about a hundred:

        >>> x = np.empty(2000)
        >>> x[0] = 0.0
        >>> for t in range(1, 2000):
        ...     x[t] = 0.9 * x[t - 1] + rng.standard_normal()
        >>> 40 < float(ess_bulk(x)) < 250
        True
    """
    return _per_column(chains, _ess_bulk)


def ess_tail(*chains: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
    r"""Tail effective sample size, for credible-interval endpoints.

    A sampler that mixes well in the centre of a posterior can still mix
    poorly in its tails, and the bulk size cannot see it. The tail size is
    the smaller of the effective sample sizes of the two indicators

    .. math::

       \mathbb{1}\{x_s \le q_{0.05}\}, \qquad \mathbb{1}\{x_s \le q_{0.95}\},

    where :math:`q_{0.05}` and :math:`q_{0.95}` are the quantiles of the
    pooled draws, each computed on the split chains by the same Geyer
    estimator as :func:`ess_bulk`. The Monte Carlo error of a quantile is
    governed by the effective size of its indicator, so this is the size
    that says how well the endpoints of a 90% interval are determined.

    Args:
        *chains: One or more ``(S,)`` or ``(S, d)`` chains of the same
            shape, with ``S`` at least eight and finite throughout.

    Returns:
        A float for ``(S,)`` chains, else ``(d,)``; the total across chains.
        ``nan`` for a quantity that never moved.

    Raises:
        SpecificationError: If no chain is given, or a chain is shorter than
            eight draws.
        DimensionError: If a chain is not one- or two-dimensional, or the
            chains differ in shape.
        NumericalError: If any draw is not finite.

    See Also:
        :func:`ess_bulk`: The size for means and medians.

    References:
        Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner,
            P.-C. (2021). Rank-normalization, folding, and localization: An
            improved :math:`\widehat{R}` for assessing convergence of MCMC.
            *Bayesian Analysis*, 16(2), 667-718.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(3)
        >>> float(ess_tail(rng.standard_normal(2000))) > 800
        True
    """
    return _per_column(chains, _ess_tail)


def mcse(*chains: npt.ArrayLike) -> float | npt.NDArray[np.float64]:
    r"""Monte Carlo standard error of the posterior mean.

    .. math::

       \widehat{\operatorname{se}}(\bar x)
       = \frac{\hat\sigma}{\sqrt{\widehat{S}_{\text{eff}}}},

    with :math:`\hat\sigma` the standard deviation of the pooled draws and
    :math:`\widehat{S}_{\text{eff}}` the Geyer effective sample size of the
    raw split chains rather than the rank-normalized ones, since it is the
    untransformed draws whose average is being reported. The number says
    how far the reported mean is likely to sit from the mean of the
    distribution the sampler targets; it says nothing about how far that
    distribution sits from the truth, which is the posterior standard
    deviation's job.

    Args:
        *chains: One or more ``(S,)`` or ``(S, d)`` chains of the same
            shape, with ``S`` at least eight and finite throughout.

    Returns:
        A float for ``(S,)`` chains, else ``(d,)``, in the units of the
        quantity. ``nan`` for a quantity that never moved.

    Raises:
        SpecificationError: If no chain is given, or a chain is shorter than
            eight draws.
        DimensionError: If a chain is not one- or two-dimensional, or the
            chains differ in shape.
        NumericalError: If any draw is not finite.

    Note:
        A standard error a tenth of the posterior standard deviation is the
        usual working target: beyond that, more draws sharpen the reported
        mean to digits the posterior does not support.

    See Also:
        :func:`ess_bulk`: The rank-normalized size, which is the one to
            read for a convergence verdict.

    References:
        Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner,
            P.-C. (2021). Rank-normalization, folding, and localization: An
            improved :math:`\widehat{R}` for assessing convergence of MCMC.
            *Bayesian Analysis*, 16(2), 667-718.

    Example:
        Ten thousand independent standard normal draws pin the mean to
        about :math:`1 / \sqrt{10000} = 0.01`:

        >>> import numpy as np
        >>> rng = np.random.default_rng(4)
        >>> round(float(mcse(rng.standard_normal(10_000))), 1)
        0.0
    """
    return _per_column(chains, _mcse_mean)


def geweke(
    *chains: npt.ArrayLike, first: float = 0.1, last: float = 0.5
) -> npt.NDArray[np.float64]:
    r"""Geweke's early-versus-late z-score, one per chain and quantity.

    For a chain of :math:`N` draws, the first :math:`N_A = \lfloor
    \text{first} \cdot N \rfloor` and the last :math:`N_B = \lfloor
    \text{last} \cdot N \rfloor` draws are compared as two samples from
    what should be the same distribution:

    .. math::

       z = \frac{\bar x_A - \bar x_B}
                {\sqrt{\widehat{\operatorname{lrv}}_A / N_A
                     + \widehat{\operatorname{lrv}}_B / N_B}},

    where each long-run variance is the segment's sample variance times
    the ratio of its length to its effective sample size,
    :math:`\widehat{\operatorname{lrv}} = s^2 \, N / \widehat{S}_{\text{eff}}`.
    Geweke (1992) estimated the long-run variances with a spectral window;
    using the Geyer size instead means no window is tuned separately from
    the rest of this module. A large score says the chain was still moving
    when the early segment was drawn, that is, the burn-in was too short.
    Unlike the other statistics here, nothing is pooled: each chain is
    scored on its own.

    Args:
        *chains: One or more ``(S,)`` or ``(S, d)`` chains of the same
            shape, finite throughout.
        first: Fraction of each chain forming the early segment.
        last: Fraction of each chain forming the late segment. The
            segments must not overlap, so ``first + last <= 1``.

    Returns:
        ``(C,)`` for ``(S,)`` chains, else ``(C, d)``, for ``C`` chains.
        Under the null the scores are standard normal, so magnitudes above
        about 2 indicate the early and late segments disagree in mean;
        ``nan`` where a segment never moved.

    Raises:
        SpecificationError: If no chain is given; if the fractions are not
            positive or sum to more than one; or if either segment has
            fewer than eight draws, which at the defaults means a chain
            shorter than 80.
        DimensionError: If a chain is not one- or two-dimensional, or the
            chains differ in shape.
        NumericalError: If any draw is not finite.

    Note:
        The score is a diagnostic of burn-in, not of mixing: a chain that
        is stationary but slowly mixing passes it while failing
        :func:`ess_bulk`, and a chain confined to one mode passes it while
        failing :func:`rhat` against a second run. :func:`convergence`
        reports it for reading and leaves it out of the verdict.

    References:
        Geweke, J. (1992). Evaluating the accuracy of sampling-based
            approaches to the calculation of posterior moments. In J. M.
            Bernardo, J. O. Berger, A. P. Dawid, & A. F. M. Smith (Eds.),
            *Bayesian Statistics 4* (pp. 169-193). Oxford University Press.

    Example:
        One score per chain:

        >>> import numpy as np
        >>> rng = np.random.default_rng(5)
        >>> geweke(rng.standard_normal(1000), rng.standard_normal(1000)).shape
        (2,)

        A chain whose level is still moving is flagged:

        >>> drifting = np.linspace(0.0, 3.0, 1000) + rng.standard_normal(1000)
        >>> bool(abs(geweke(drifting)[0]) > 2.0)
        True
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
    r"""Full per-quantity report over one or more chains.

    One row per scalar quantity: the posterior mean and standard deviation,
    :func:`mcse`, :func:`ess_bulk`, :func:`ess_tail`, :func:`rhat`, and the
    :func:`geweke` score of largest magnitude across chains. The verdict
    applies the thresholds of Vehtari et al. (2021): a quantity is flagged
    when

    .. math::

       \widehat{R} > \texttt{rhat\_tol}
       \quad\text{or}\quad
       \min\!\bigl(\widehat{S}_{\text{bulk}}, \widehat{S}_{\text{tail}}\bigr)
       < \texttt{min\_ess} \cdot C,

    for :math:`C` chains, and the report is converged when nothing is
    flagged. A quantity whose draws never moved has ``nan`` in every
    column and is listed as degenerate rather than counted either way,
    because the report cannot tell a parameter fixed by construction from
    a sampler that is stuck; read the list. The Geweke score is reported
    for reading and is ``nan`` when the chains are too short for its
    default segments.

    This is the same machinery behind the ``convergence()`` method of every
    sampled result in the package, for draws that did not come from the
    package or for a function of the package's draws whose own convergence
    is the question.

    Args:
        *chains: One or more ``(S,)`` or ``(S, d)`` chains of the same shape
            from independent runs, with ``S`` at least eight and finite
            throughout. Passing two runs from different seeds is what lets
            :math:`\widehat{R}` detect a chain confined to one mode; a
            single run is assessed by its two halves, which detects drift
            only.
        names: ``d`` labels, one per column; defaults to ``x[0], ...,
            x[d-1]``, or ``x`` for ``(S,)`` chains.
        rhat_tol: :math:`\widehat{R}` above which a quantity is flagged.
            Must exceed one.
        min_ess: Effective draws per chain below which a quantity is
            flagged; the floor applied to the pooled size is ``min_ess``
            times the number of chains. Must be positive.
        source: What was assessed, for the summary title.

    Returns:
        The :class:`ConvergenceTest`: read
        :attr:`~ConvergenceTest.converged`,
        :attr:`~ConvergenceTest.flagged`, and
        :attr:`~ConvergenceTest.degenerate`; :meth:`~ConvergenceTest.worst`
        names the quantities with the largest :math:`\widehat{R}`, and
        :meth:`~ConvergenceTest.summary` renders the table in that order.

    Raises:
        SpecificationError: If no chain is given, a chain is shorter than
            eight draws, ``rhat_tol`` does not exceed one, or ``min_ess`` is
            not positive.
        DimensionError: If a chain is not one- or two-dimensional, the
            chains differ in shape, or ``names`` does not match the number
            of quantities.
        NumericalError: If any draw is not finite.

    See Also:
        :class:`ConvergenceTest`: The record and its summary.

    References:
        Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner,
            P.-C. (2021). Rank-normalization, folding, and localization: An
            improved :math:`\widehat{R}` for assessing convergence of MCMC.
            *Bayesian Analysis*, 16(2), 667-718.

    Example:
        Three quantities drawn jointly from one run:

        >>> import numpy as np
        >>> rng = np.random.default_rng(6)
        >>> report = convergence(rng.standard_normal((2000, 3)))
        >>> report.names
        ('x[0]', 'x[1]', 'x[2]')
        >>> report.flagged, report.converged
        ((), True)

        Two runs that disagree in location are flagged on
        :math:`\widehat{R}`:

        >>> stuck = convergence(rng.standard_normal(500), 4.0 + rng.standard_normal(500))
        >>> stuck.flagged, round(float(stuck.rhat[0]), 1) > 1.5
        (('x',), True)

        A column that never moved is reported, not judged:

        >>> fixed = np.column_stack([rng.standard_normal(2000), np.ones(2000)])
        >>> report = convergence(fixed, names=("free", "fixed"))
        >>> report.degenerate, report.converged
        (('fixed',), True)
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
