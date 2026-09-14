# filepath: /src/cultivars/bayes/evidence.py
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

"""Marginal likelihoods, Bayes factors, and posterior model probabilities.

The marginal likelihood ``m(y) = int f(y | theta) pi(theta) dtheta`` is the
Bayesian answer to "which model", and its logarithm is the one number a
posterior can report in place of a likelihood. Three routes to it live in
the package, each on the results that admit it:

* **Analytic**, for the conjugate :class:`~cultivars.multivariate.large_dim.BVAR`,
  whose Normal-inverse-Wishart posterior integrates in closed form.
* **Chib (1995)**, for the :class:`~cultivars.multivariate.large_dim.GibbsBVAR`
  under a static prior, from the two-block Gibbs output with no extra runs.
* **Modified harmonic mean** (Geweke, 1999), for the particle-chain
  :class:`~cultivars.multivariate.structural.PerturbationDSGE` posterior,
  where the likelihood itself is an estimate and the number inherits that
  noise.

:func:`marginal_likelihood` dispatches to whichever a result offers and
refuses the rest by name. :func:`modified_harmonic_mean` and
:func:`bridge_sampling` are the same estimators on draws that came from
elsewhere -- another sampler, or a kernel of the user's own -- and
:func:`compare` ranks any set of records fitted to the same sample.

Not implemented, deliberately: the plain harmonic mean (infinite variance),
and Chib's estimator for adaptive-prior or dummy-observation Gibbs BVARs
(reduced runs, and the augmented-sample subtlety, respectively). Both are
refused with the reason rather than approximated silently.

Example:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> z = rng.standard_normal((4000, 2))
    >>> record = modified_harmonic_mean(z, -0.5 * (z**2).sum(axis=1), source="N(0, I)")
    >>> round(record.log_value, 1)  # log(2 pi)
    1.8
    >>> record.method
    'modified harmonic mean'
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from .._core import _MHM_TAU, SummaryTable, _bridge_sampling, _modified_harmonic_mean
from .._internals import _MarginalLikelihoodSelection
from ..exceptions import SpecificationError

__all__ = [
    "bridge_sampling",
    "compare",
    "marginal_likelihood",
    "modified_harmonic_mean",
]


def marginal_likelihood(result: object) -> _MarginalLikelihoodSelection:
    """The log marginal likelihood of a fitted result, by whichever route it admits.

    Args:
        result: A result exposing ``marginal_likelihood()``: the conjugate
            BVAR (analytic), the Gibbs BVAR (Chib), or the particle-chain
            DSGE posterior (modified harmonic mean).

    Returns:
        The record.

    Raises:
        SpecificationError: If the result offers no marginal likelihood.
            Sampled results that do not -- the stochastic-volatility family,
            the Student-t and stochastic-volatility BVARs, the
            time-varying-parameter models -- need reduced-run or
            particle-based estimators that are not implemented; compare
            those by out-of-sample predictive score.
    """
    method = getattr(result, "marginal_likelihood", None)
    if not callable(method):
        raise SpecificationError(
            f"{type(result).__name__} offers no marginal likelihood. Supported: the conjugate "
            "BVAR (analytic), the Gibbs BVAR under a static prior (Chib), and the "
            "particle-chain DSGE posterior (modified harmonic mean). Other sampled results "
            "need estimators not implemented here; compare them by predictive score."
        )
    record = method()
    if not isinstance(record, _MarginalLikelihoodSelection):
        raise SpecificationError(
            f"{type(result).__name__}.marginal_likelihood() returned {type(record).__name__}, "
            "not a _MarginalLikelihoodSelection record."
        )
    return record


def modified_harmonic_mean(
    draws: npt.ArrayLike,
    log_kernel: npt.ArrayLike,
    *,
    tau: float = _MHM_TAU,
    source: str = "draws",
) -> _MarginalLikelihoodSelection:
    """Geweke's (1999) modified harmonic mean on posterior draws.

    Args:
        draws: ``(S,)`` or ``(S, d)`` posterior draws. Transform bounded
            parameters to the real line first: a Gaussian envelope that
            leaves the support biases the estimate, and the coverage note
            says when that has happened.
        log_kernel: ``(S,)`` unnormalized log posterior -- log likelihood
            plus log prior, in the same parameterization as ``draws`` -- at
            each draw.
        tau: Probability mass of the Gaussian envelope retained.
        source: Label for the record.

    Returns:
        The record, with the Monte Carlo standard error.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> z = rng.standard_normal((3000, 3))
        >>> record = modified_harmonic_mean(z, -0.5 * (z**2).sum(axis=1))
        >>> bool(abs(record.log_value - 1.5 * np.log(2 * np.pi)) < 0.05)
        True
    """
    log_ml, mcse, coverage = _modified_harmonic_mean(draws, log_kernel, tau=tau)
    notes = ["Modified harmonic mean (Geweke, 1999) with a truncated Gaussian envelope."]
    if abs(coverage - tau) > 0.1:
        notes.append(
            f"Only {coverage:.0%} of the draws fall inside the {tau:.0%} envelope: the "
            "posterior is far from Gaussian, or a bounded parameter was not transformed."
        )
    return _MarginalLikelihoodSelection(
        log_value=log_ml,
        mcse=mcse,
        method="modified harmonic mean",
        n_draws=int(np.asarray(draws).shape[0]),
        source=source,
        notes=tuple(notes),
    )


def bridge_sampling(
    draws: npt.ArrayLike,
    log_kernel: npt.ArrayLike,
    kernel: Callable[[npt.NDArray[np.float64]], float],
    *,
    seed: int | np.random.Generator | None = None,
    n_proposal: int | None = None,
    source: str = "draws",
) -> _MarginalLikelihoodSelection:
    """Meng-Wong bridge sampling with a Gaussian proposal fitted to the draws.

    Lower variance than the harmonic mean on a roughly elliptical
    posterior, at the price of evaluating the kernel at fresh proposal
    points -- so the kernel must be a callable, and the parameterization
    must give it finite values wherever a Gaussian around the draws can
    land (transform bounded parameters to the real line).

    Args:
        draws: ``(S,)`` or ``(S, d)`` posterior draws.
        log_kernel: ``(S,)`` unnormalized log posterior at each draw.
        kernel: The same function of one ``(d,)`` point.
        seed: Seed or generator for the proposal draws.
        n_proposal: Proposal draws; default the number of posterior draws
            entering the bridge (half of ``draws``).
        source: Label for the record.

    Returns:
        The record, with the Frühwirth-Schnatter (2004) error approximation.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(2)
        >>> z = rng.standard_normal((3000, 2))
        >>> record = bridge_sampling(
        ...     z, -0.5 * (z**2).sum(axis=1), lambda x: -0.5 * float(x @ x), seed=0
        ... )
        >>> bool(abs(record.log_value - np.log(2 * np.pi)) < 0.05)
        True
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    log_ml, mcse, n_iter = _bridge_sampling(
        draws, log_kernel, kernel, rng=rng, n_proposal=n_proposal
    )
    return _MarginalLikelihoodSelection(
        log_value=log_ml,
        mcse=mcse,
        method="bridge sampling",
        n_draws=int(np.asarray(draws).shape[0]),
        source=source,
        notes=(
            f"Bridge sampling (Meng & Wong, 1996), Gaussian proposal fitted to half the "
            f"draws, converged in {n_iter} iterations.",
        ),
    )


def compare(
    *records: _MarginalLikelihoodSelection,
    names: Sequence[str] | None = None,
    prior_probabilities: Sequence[float] | None = None,
) -> SummaryTable:
    """Rank models by marginal likelihood with posterior model probabilities.

    Args:
        *records: Two or more records computed on the same sample.
        names: Row labels; default each record's ``source``.
        prior_probabilities: Prior model probabilities; default equal.

    Returns:
        The comparison table, best model first.

    Raises:
        SpecificationError: If fewer than two records are given.
    """
    if len(records) < 2:
        raise SpecificationError("compare needs at least two records.")
    return records[0].compare(*records[1:], names=names, prior_probabilities=prior_probabilities)
