# filepath: /src/cultivars/_core/_convergence.py
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

"""Markov-chain convergence primitives on a ``(chains, draws)`` array.

Every function here takes one scalar quantity's draws laid out as ``(C, N)``
-- ``C`` independent chains of ``N`` kept draws each -- and returns a number.
A single chain is the ``C = 1`` case; nothing assumes more, and every
statistic degrades gracefully to its single-chain reading (split-R-hat
compares the two halves of the one chain, which detects a trend but not a
chain stuck in one of several modes).

The estimators are those of Vehtari, Gelman, Simpson, Carpenter, and Bürkner
(2021), which replace the classical potential scale reduction with a version
that works for heavy-tailed and asymmetric posteriors:

* Chains are *split* in half so that a within-chain trend registers as
  between-chain disagreement.
* Draws are *rank-normalized* -- replaced by the normal scores of their
  pooled ranks -- so that a parameter without finite moments still has a
  well-defined R-hat and effective sample size.
* R-hat is the larger of the rank-normalized statistic (sensitive to location
  differences) and the same statistic on the draws *folded* about their
  median (sensitive to scale differences).
* Bulk effective sample size is Geyer's (1992) initial monotone sequence
  estimator on the rank-normalized split chains; tail effective sample size
  is the smaller of the sizes for the 5% and 95% quantile indicators, which
  is what governs the reliability of an interval.

Geweke's (1992) score compares the mean of an early segment of a chain to
the mean of a late one, with the long-run variance of each segment estimated
through the same Geyer sequence rather than a separately tuned spectral
window.

A constant chain has no within-chain variance, and every statistic returns
``nan`` for it rather than a number that would be read as a verdict: a
parameter that never moved is either fixed by construction or a sampler
that is stuck, and the caller, not this module, knows which.

References:
    Geweke, J. (1992). Evaluating the accuracy of sampling-based approaches
        to the calculation of posterior moments. In *Bayesian Statistics 4*.
    Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical
        Science*, 7(4), 473-483.
    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C.
        (2021). Rank-normalization, folding, and localization: An improved
        R-hat for assessing convergence of MCMC. *Bayesian Analysis*, 16(2),
        667-718.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy.stats import norm, rankdata

from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._defaults import _MIN_CHAIN_DRAWS


def _stack(chains: tuple[npt.ArrayLike, ...]) -> tuple[npt.NDArray[np.float64], bool]:
    """Stack ``(S,)`` or ``(S, d)`` chains into ``(C, S, d)``.

    Returns:
        The stack and whether every chain was one-dimensional, so the caller
        can return a scalar rather than a length-one array.

    Raises:
        SpecificationError: If no chain is given.
        DimensionError: If a chain is not one- or two-dimensional, or the
            chains differ in shape.
    """
    if not chains:
        raise SpecificationError("At least one chain is required.")
    arrays = [np.asarray(chain, dtype=np.float64) for chain in chains]
    scalar = all(array.ndim == 1 for array in arrays)
    columns: list[npt.NDArray[np.float64]] = []
    for array in arrays:
        if array.ndim == 1:
            columns.append(array[:, None])
        elif array.ndim == 2:
            columns.append(array)
        else:
            raise DimensionError(f"A chain must be (S,) or (S, d); got shape {array.shape}.")
    shapes = {column.shape for column in columns}
    if len(shapes) != 1:
        raise DimensionError(f"Chains differ in shape: {sorted(shapes)}.")
    return np.stack(columns), scalar


def _per_column(
    chains: tuple[npt.ArrayLike, ...], statistic: Callable[[npt.NDArray[np.float64]], float]
) -> float | npt.NDArray[np.float64]:
    """Apply a ``(C, N) -> float`` statistic to every quantity in the stack."""
    stack, scalar = _stack(chains)
    values = np.array([statistic(stack[:, :, j]) for j in range(stack.shape[2])])
    return float(values[0]) if scalar else values


def _as_chains(draws: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce one quantity's draws to a validated ``(C, N)`` float array.

    Args:
        draws: ``(N,)`` for one chain or ``(C, N)`` for several.

    Returns:
        The ``(C, N)`` array.

    Raises:
        DimensionError: If ``draws`` is not one- or two-dimensional.
        SpecificationError: If a chain is shorter than the minimum.
        NumericalError: If any draw is not finite.

    Example:
        >>> _as_chains(np.arange(10.0)).shape
        (1, 10)
    """
    chains = np.asarray(draws, dtype=np.float64)
    if chains.ndim == 1:
        chains = chains[None, :]
    elif chains.ndim != 2:
        raise DimensionError(
            f"draws must be (N,) for one chain or (C, N) for several; got shape {chains.shape}."
        )
    if chains.shape[1] < _MIN_CHAIN_DRAWS:
        raise SpecificationError(
            f"Each chain needs at least {_MIN_CHAIN_DRAWS} kept draws for a convergence "
            f"diagnostic; got {chains.shape[1]}."
        )
    if not np.all(np.isfinite(chains)):
        raise NumericalError("Chain draws contain non-finite values.")
    return chains


def _split_chains(chains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Halve every chain, so ``(C, N)`` becomes ``(2C, N // 2)``.

    An odd trailing draw is dropped, so both halves have the same length.

    Example:
        >>> _split_chains(np.arange(9.0)[None, :]).shape
        (2, 4)
    """
    half = chains.shape[1] // 2
    return np.concatenate([chains[:, :half], chains[:, half : 2 * half]], axis=0)


def _rank_normalize(chains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Replace every draw by the normal score of its rank among all draws.

    Ranks are pooled across chains (ties averaged) and mapped through the
    Blom offset ``(r - 3/8) / (S + 1/4)`` before the normal quantile, which
    keeps the scores finite at both extremes.

    Example:
        >>> z = _rank_normalize(np.array([[1.0, 2.0, 3.0, 4.0]]))
        >>> bool(np.all(np.diff(z[0]) > 0))
        True
    """
    ranks = rankdata(chains, method="average", axis=None).reshape(chains.shape)
    size = chains.size
    return np.asarray(norm.ppf((ranks - 0.375) / (size + 0.25)), dtype=np.float64)


def _autocovariance(chains: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Biased sample autocovariance of every chain, all lags, via the FFT.

    Example:
        >>> acov = _autocovariance(np.array([[1.0, -1.0, 1.0, -1.0]]))
        >>> [round(float(v), 2) for v in acov[0]]
        [1.0, -0.75, 0.5, -0.25]
    """
    n_draws = chains.shape[1]
    n_fft = 1 << int(2 * n_draws - 1).bit_length()
    centered = chains - chains.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, n=n_fft, axis=1)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft, axis=1)[:, :n_draws]
    return np.asarray(acov / n_draws, dtype=np.float64)


def _potential_scale_reduction(chains: npt.NDArray[np.float64]) -> float:
    """Classical R-hat over already split (and possibly transformed) chains.

    ``sqrt(var_hat / W)`` with ``var_hat`` the weighted average of the
    within-chain variance ``W`` and the between-chain variance ``B``.
    Returns ``nan`` when the chains carry no within-chain variance.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> round(_potential_scale_reduction(rng.standard_normal((4, 500))), 1)
        1.0
    """
    n_chains, n_draws = chains.shape
    within = float(chains.var(axis=1, ddof=1).mean())
    if not within > 0.0:
        return float("nan")
    between = float(n_draws * chains.mean(axis=1).var(ddof=1)) if n_chains > 1 else 0.0
    var_hat = (n_draws - 1) / n_draws * within + between / n_draws
    return float(np.sqrt(var_hat / within))


def _effective_sample_size(chains: npt.NDArray[np.float64]) -> float:
    """Geyer initial-monotone-sequence effective sample size of ``(M, N)`` chains.

    The pooled autocorrelation at each lag is ``1 - (W - mean acov) /
    var_hat``; consecutive lags are summed in pairs, the sequence is
    truncated at the first non-positive pair (initial positive sequence)
    and then forced non-increasing (initial monotone sequence), and the
    integrated autocorrelation time is ``-1 + 2 * sum of the pairs``. The
    estimate is capped at ``M N log10(M N)``, since an antithetic chain can
    otherwise report more effective draws than a bound the estimator's
    precision supports. Returns ``nan`` for chains without variance.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> ess = _effective_sample_size(rng.standard_normal((2, 1000)))
        >>> 1400 < ess < 2600
        True
    """
    n_chains, n_draws = chains.shape
    acov = _autocovariance(chains)
    chain_var = acov[:, 0] * n_draws / (n_draws - 1)
    within = float(chain_var.mean())
    var_hat = within * (n_draws - 1) / n_draws
    if n_chains > 1:
        var_hat += float(chains.mean(axis=1).var(ddof=1))
    if not var_hat > 0.0:
        return float("nan")
    rho = 1.0 - (within - acov.mean(axis=0)) / var_hat
    n_pairs = n_draws // 2
    pairs = rho[: 2 * n_pairs].reshape(n_pairs, 2).sum(axis=1)
    negative = np.flatnonzero(pairs <= 0.0)
    cutoff = int(negative[0]) if negative.size else n_pairs
    tau = 1.0 if cutoff == 0 else -1.0 + 2.0 * float(np.minimum.accumulate(pairs[:cutoff]).sum())
    total = n_chains * n_draws
    ess = total / tau if tau > 0.0 else float(total)
    return float(min(ess, total * np.log10(total)))


def _rhat(draws: npt.ArrayLike) -> float:
    """Rank-normalized split-R-hat with folding.

    Args:
        draws: ``(N,)`` or ``(C, N)`` kept draws of one quantity.

    Returns:
        The larger of the rank-normalized R-hat and the folded rank-normalized
        R-hat; ``nan`` for a constant chain.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> mixed = rng.standard_normal((2, 400))
        >>> rhat(mixed) < 1.02
        True
        >>> apart = np.stack([mixed[0], mixed[1] + 5.0])
        >>> rhat(apart) > 1.5
        True
    """
    chains = _as_chains(draws)
    split = _split_chains(chains)
    plain = _potential_scale_reduction(_rank_normalize(split))
    folded = _potential_scale_reduction(_rank_normalize(np.abs(split - np.median(chains))))
    if np.isnan(plain) or np.isnan(folded):
        return float("nan")
    return max(plain, folded)


def _ess_bulk(draws: npt.ArrayLike) -> float:
    """Bulk effective sample size on the rank-normalized split chains.

    Args:
        draws: ``(N,)`` or ``(C, N)`` kept draws of one quantity.

    Returns:
        Effective draws for estimating central posterior quantities; ``nan``
        for a constant chain.

    Example:
        >>> rng = np.random.default_rng(1)
        >>> x = rng.standard_normal(2000)
        >>> 1400 < ess_bulk(x) < 2600
        True
        >>> sticky = np.repeat(x[:200], 10)
        >>> ess_bulk(sticky) < 400
        True
    """
    return _effective_sample_size(_rank_normalize(_split_chains(_as_chains(draws))))


def _ess_tail(draws: npt.ArrayLike) -> float:
    """Tail effective sample size: the lesser of the 5% and 95% quantile sizes.

    Each is the effective sample size of the indicator that a draw lies
    below the corresponding pooled quantile, which is the quantity whose
    Monte Carlo error governs a credible-interval endpoint.

    Args:
        draws: ``(N,)`` or ``(C, N)`` kept draws of one quantity.

    Returns:
        Effective draws for the interval endpoints; ``nan`` for a constant
        chain.

    Example:
        >>> rng = np.random.default_rng(2)
        >>> 900 < ess_tail(rng.standard_normal(2000)) < 2600
        True
    """
    chains = _as_chains(draws)
    lower, upper = np.quantile(chains, [0.05, 0.95])
    sizes = [
        _effective_sample_size(_split_chains((chains <= q).astype(np.float64)))
        for q in (lower, upper)
    ]
    if any(np.isnan(s) for s in sizes):
        return float("nan")
    return float(min(sizes))


def _ess_mean(draws: npt.ArrayLike) -> float:
    """Effective sample size for the posterior mean, on the raw split chains.

    This is the untransformed size that enters the Monte Carlo standard
    error of the mean; the rank-normalized bulk size is the one to read for
    a convergence verdict.

    Example:
        >>> rng = np.random.default_rng(3)
        >>> 1400 < ess_mean(rng.standard_normal(2000)) < 2600
        True
    """
    return _effective_sample_size(_split_chains(_as_chains(draws)))


def _mcse_mean(draws: npt.ArrayLike) -> float:
    """Monte Carlo standard error of the posterior mean.

    ``sd / sqrt(ess_mean)``, with the standard deviation pooled across
    chains.

    Example:
        >>> rng = np.random.default_rng(4)
        >>> round(mcse_mean(rng.standard_normal(10_000)), 1)
        0.0
    """
    chains = _as_chains(draws)
    ess = _ess_mean(chains)
    if np.isnan(ess):
        return float("nan")
    return float(chains.std(ddof=1) / np.sqrt(ess))


def _geweke(
    draws: npt.ArrayLike, *, first: float = 0.1, last: float = 0.5
) -> npt.NDArray[np.float64]:
    """Geweke's early-versus-late mean-difference score, one per chain.

    The long-run variance of each segment is its sample variance times the
    ratio of its length to its effective sample size, so no spectral window
    is tuned separately.

    Args:
        draws: ``(N,)`` or ``(C, N)`` kept draws of one quantity.
        first: Fraction of each chain forming the early segment.
        last: Fraction of each chain forming the late segment.

    Returns:
        ``(C,)`` z-scores, ``nan`` where a segment is constant.

    Raises:
        SpecificationError: If the fractions are not positive, overlap, or
            leave a segment too short to estimate an autocorrelation.

    Example:
        >>> rng = np.random.default_rng(5)
        >>> z = geweke(rng.standard_normal((3, 1000)))
        >>> z.shape
        (3,)
        >>> bool(np.all(np.abs(z) < 4.0))
        True
    """
    if not (first > 0.0 and last > 0.0 and first + last <= 1.0):
        raise SpecificationError(
            f"first and last must be positive fractions summing to at most 1; got {first}, {last}."
        )
    chains = _as_chains(draws)
    n_draws = chains.shape[1]
    n_first = int(first * n_draws)
    n_last = int(last * n_draws)
    if min(n_first, n_last) < _MIN_CHAIN_DRAWS:
        raise SpecificationError(
            f"Geweke segments need at least {_MIN_CHAIN_DRAWS} draws each; got {n_first} "
            f"and {n_last} from {n_draws} draws."
        )
    scores = np.empty(chains.shape[0], dtype=np.float64)
    for c, chain in enumerate(chains):
        early = chain[:n_first]
        late = chain[n_draws - n_last :]
        variances = []
        for segment in (early, late):
            ess = _effective_sample_size(segment[None, :])
            variances.append(float("nan") if np.isnan(ess) else segment.var(ddof=1) / ess)
        denominator = float(np.sqrt(variances[0] + variances[1]))
        scores[c] = (
            float("nan") if not denominator > 0.0 else (early.mean() - late.mean()) / denominator
        )
    return scores
