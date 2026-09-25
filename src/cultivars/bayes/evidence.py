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
r"""Marginal likelihoods, Bayes factors, and posterior model probabilities.

The marginal likelihood

.. math::

   m(y) = \int f(y \mid \theta)\, \pi(\theta)\, d\theta

is the probability a model assigned to the data before seeing them, and
it is the Bayesian answer to "which model": the ratio :math:`m_1(y) /
m_2(y)` is the Bayes factor, and with prior model probabilities it gives
posterior model probabilities. Its logarithm is the one number a
posterior can report in place of a log likelihood, and unlike a
likelihood it carries its own penalty for complexity, since a model that
spreads prior mass over parameters the data do not want pays for it in
:math:`m(y)`. Three routes to it live in the package, each on the results
that admit it. The conjugate
:class:`~cultivars.multivariate.large_dim.bayesian.BVAR` integrates its
Normal-inverse-Wishart posterior in closed form, so its number is exact.
The :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR` under a
static prior uses Chib's (1995) identity, :math:`\log m(y) = \log f(y
\mid \theta^*) + \log \pi(\theta^*) - \log \pi(\theta^* \mid y)`, with
the posterior ordinate at a high-density point estimated from the
two-block Gibbs output and no extra runs. The particle-chain
:class:`~cultivars.multivariate.structural.perturbation.PerturbationDSGE`
posterior uses Geweke's (1999) modified harmonic mean, and since its
likelihood is itself a particle-filter estimate, the number inherits that
noise.

Two commitments shape the surface. First, every route returns the same
record with its Monte Carlo standard error, zero when exact, so that
:func:`compare` can say when two models are not separated by the
estimates at hand, and so that
:func:`~cultivars.bayes.combination.bayesian_model_average` can warn that
a weight ratio is only as precise as the difference of two noisy
estimates. Second, what is not implemented is refused with the reason
rather than approximated silently: the plain harmonic mean, whose
variance is infinite because the prior's tails are heavier than the
posterior's; Chib's estimator for the Gibbs BVAR under an adaptive prior,
which needs reduced runs; and Chib's estimator with dummy observations,
where the augmented sample makes the ordinate's meaning subtle. Sampled
results without a route -- the stochastic-volatility family, the
Student-t and stochastic-volatility BVARs, the time-varying-parameter
models -- say so by name and point to out-of-sample predictive score.

Layout. :func:`marginal_likelihood` is the dispatcher: it calls a result's
own ``marginal_likelihood()`` and returns the record, which the three
results above define on themselves and which delegates to ``_core`` --
the closed-form update for the conjugate model, ``_core._estimators.
_chib_independent_normal_wishart`` for the Gibbs model, and
``_modified_harmonic_mean`` for the DSGE posterior.
:func:`modified_harmonic_mean` and :func:`bridge_sampling` expose the two
generic estimators on draws that came from elsewhere, another sampler or
a kernel of the user's own, through ``_core._estimators.
_modified_harmonic_mean`` and ``_bridge_sampling``, both of which fit
their Gaussian envelope with ``_gaussian_envelope`` and validate their
input with ``_validate_posterior_draws``. :func:`compare` ranks any set
of records by delegating to the record's own ``compare`` method in
``_internals._selections``, which builds the table and reads the
Kass-Raftery evidence scale through ``_core._converters._evidence_label``.
The record itself,
:class:`~cultivars.bayes.combination.MarginalLikelihoodSelection`, is the
``_internals`` dataclass every route constructs.

References:
    Chib, S. (1995). Marginal likelihood from the Gibbs output. *Journal
    of the American Statistical Association*, 90(432), 1313-1321.

    Geweke, J. (1999). Using simulation methods for Bayesian econometric
    models: Inference, development, and communication. *Econometric
    Reviews*, 18(1), 1-73.

    Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *Journal of the
    American Statistical Association*, 90(430), 773-795.

    Meng, X.-L., & Wong, W. H. (1996). Simulating ratios of normalizing
    constants via a simple identity: A theoretical exploration.
    *Statistica Sinica*, 6(4), 831-860.

    Newton, M. A., & Raftery, A. E. (1994). Approximate Bayesian inference
    with the weighted likelihood bootstrap. *Journal of the Royal
    Statistical Society B*, 56(1), 3-48.

Example:
    The evidence of a bivariate standard normal from its unnormalized
    kernel :math:`-\tfrac{1}{2}\|z\|^2` is the missing constant
    :math:`2\pi`:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> z = rng.standard_normal((4000, 2))
    >>> record = modified_harmonic_mean(z, -0.5 * (z**2).sum(axis=1), source="N(0, I)")
    >>> round(record.log_value, 1)  # log(2 pi)
    1.8
    >>> record.method
    'modified harmonic mean'

    A package result carries its own route, and the records rank:

    >>> from cultivars.multivariate.large_dim.bayesian import BVAR
    >>> y = np.zeros((120, 2))
    >>> for t in range(1, 120):
    ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
    >>> one = marginal_likelihood(BVAR(y, order=1).fit(n_draws=100, seed=0))
    >>> one.method, one.mcse
    ('analytic', 0.0)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from .._core import _MHM_TAU, SummaryTable, _bridge_sampling, _modified_harmonic_mean
from .._internals import _MarginalLikelihoodSelection as MarginalLikelihoodSelection
from ..exceptions import SpecificationError

__all__ = [
    "MarginalLikelihoodSelection",
    "bridge_sampling",
    "compare",
    "marginal_likelihood",
    "modified_harmonic_mean",
]


def marginal_likelihood(result: object) -> MarginalLikelihoodSelection:
    r"""The log marginal likelihood of a fitted result, by whichever route it admits.

    The marginal likelihood :math:`m(y) = \int f(y \mid \theta)\, \pi(\theta)\,
    d\theta` is the probability the model assigned to the data before
    seeing them, and its logarithm is the one number a posterior can report
    in place of a log likelihood. This function is the dispatcher: it calls
    the result's own ``marginal_likelihood()`` method, which knows the route
    its posterior admits, and hands back the record so that results of
    different kinds can be ranked by :func:`compare` or weighted by
    :func:`~cultivars.bayes.combination.bayesian_model_average` without the
    caller knowing which estimator produced each number.

    Three routes exist in the package. The conjugate
    :class:`~cultivars.multivariate.large_dim.bayesian.BVAR` integrates
    :math:`(B, \Sigma)` out in closed form, so its record is exact with a
    zero Monte Carlo error. The
    :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR` under a
    static prior uses Chib's (1995) identity :math:`\log m(y) = \log f(y
    \mid \theta^*) + \log \pi(\theta^*) - \log \pi(\theta^* \mid y)`, with
    the posterior ordinate at a high-density point estimated from the
    Gibbs output and no extra runs. The particle-chain
    :class:`~cultivars.multivariate.structural.perturbation.PerturbationDSGE`
    posterior uses the modified harmonic mean of :func:`modified_harmonic_mean`,
    and inherits the noise of its particle-filter likelihood.

    Args:
        result: A fitted result exposing a ``marginal_likelihood()`` method
            that returns a record: the three results above. Duck-typed, so
            a result from outside the package that returns the same record
            type is accepted.

    Returns:
        The :class:`~cultivars.bayes.combination.MarginalLikelihoodSelection`
        the result produced: ``log_value``, its Monte Carlo standard error
        ``mcse`` (zero when exact), the ``method``, the ``source`` label,
        and ``nobs``, the number of observations the likelihood scored,
        which :func:`compare` uses to refuse comparisons across different
        effective samples.

    Raises:
        SpecificationError: If the result offers no marginal likelihood, or
            if its method returns something other than the record. The
            sampled results that do not offer one -- the
            stochastic-volatility family, the Student-t and
            stochastic-volatility BVARs, the time-varying-parameter models,
            the Gibbs BVAR under an adaptive prior or with dummy
            observations -- need reduced-run or particle-based estimators
            that are not implemented; compare those by out-of-sample
            predictive score through
            :func:`~cultivars.bayes.combination.stacking` or a backtest.

    See Also:
        * :func:`compare` -- ranks records from this function with Bayes
          factors and posterior model probabilities.
        * :func:`modified_harmonic_mean` and :func:`bridge_sampling` -- the
          same estimators on draws that came from elsewhere.
        * :func:`~cultivars.bayes.combination.bayesian_model_average` --
          weights models by these records.

    References:
        Chib, S. (1995). Marginal likelihood from the Gibbs output. *Journal
        of the American Statistical Association*, 90(432), 1313-1321.

        Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *Journal of the
        American Statistical Association*, 90(430), 773-795.

    Example:
        The conjugate BVAR's record is exact:

        >>> import numpy as np
        >>> from cultivars.multivariate.large_dim.bayesian import BVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((120, 2))
        >>> for t in range(1, 120):
        ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
        >>> record = marginal_likelihood(BVAR(y, order=1).fit(n_draws=100, seed=0))
        >>> record.method, record.mcse, record.nobs
        ('analytic', 0.0, 119)
        >>> record.source
        'BVAR(1), niw(l1=0.2, l3=1, l4=100)'

        A point estimate has no marginal likelihood and is refused by name:

        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> marginal_likelihood(VAR(y, order=1).fit())  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: VARResult offers no marginal likelihood. ...
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
    if not isinstance(record, MarginalLikelihoodSelection):
        raise SpecificationError(
            f"{type(result).__name__}.marginal_likelihood() returned {type(record).__name__}, "
            "not a MarginalLikelihoodSelection record."
        )
    return record


def modified_harmonic_mean(
    draws: npt.ArrayLike,
    log_kernel: npt.ArrayLike,
    *,
    tau: float = _MHM_TAU,
    source: str = "draws",
) -> MarginalLikelihoodSelection:
    r"""Geweke's (1999) modified harmonic mean on posterior draws.

    The harmonic-mean identity :math:`m(y)^{-1} = \mathbb{E}_{\theta \mid
    y}\bigl[g(\theta) / (f(y \mid \theta)\, \pi(\theta))\bigr]` holds for
    any density :math:`g` on the parameter space, and the plain harmonic
    mean, :math:`g = \pi`, has infinite variance because the prior's tails
    are heavier than the posterior's. Geweke's modification takes :math:`g`
    to be a Gaussian fitted to the draws and truncated to its central
    :math:`\tau` ellipsoid, so that :math:`g` has lighter tails than the
    posterior by construction:

    .. math::

       \log \widehat{m}(y) = -\log \frac{1}{S} \sum_{s=1}^{S}
       \frac{g_\tau(\theta^{(s)})}{f(y \mid \theta^{(s)})\, \pi(\theta^{(s)})},
       \qquad
       g_\tau(\theta) = \frac{\phi(\theta;\, \hat\mu, \hat\Sigma)}{\tau}
       \, \mathbb{1}\{(\theta - \hat\mu)^\top \hat\Sigma^{-1}
       (\theta - \hat\mu) \le \chi^2_{d}(\tau)\}.

    The sum is formed on the log scale, and the Monte Carlo standard error
    of the log estimate is the standard deviation of the ratios divided by
    the square root of their Geyer effective sample size, so autocorrelated
    draws are charged for their autocorrelation. The share of draws that
    fall inside the ellipsoid is compared with :math:`\tau`: a share far
    from it says the posterior is not the Gaussian the envelope assumes,
    and the record notes it.

    Args:
        draws: ``(S,)`` or ``(S, d)`` posterior draws, finite throughout,
            with ``S`` greater than ``2d + 2`` so a Gaussian can be fitted.
            Transform bounded parameters to the real line first (a log for
            a variance, a logit for a probability) and include the
            Jacobian in ``log_kernel``; a Gaussian envelope that leaves the
            support biases the estimate.
        log_kernel: ``(S,)`` unnormalized log posterior, log likelihood plus
            log prior, evaluated at each draw in the same parameterization
            as ``draws``.
        tau: Probability mass of the Gaussian envelope retained, in
            ``(0, 1)``. Draws outside the ``tau`` ellipsoid receive zero
            weight; a smaller value is more robust to non-Gaussian tails
            and uses fewer draws.
        source: Label for the record's ``source`` field, which is the row
            label in :func:`compare`.

    Returns:
        The :class:`~cultivars.bayes.combination.MarginalLikelihoodSelection`
        with ``method`` set to ``"modified harmonic mean"``, the log
        estimate, its Monte Carlo standard error (``nan`` when fewer than
        eight draws fall inside the ellipsoid), ``n_draws`` the number of
        draws supplied, ``nobs`` zero since the function cannot know the
        sample, and a note on the envelope, plus a second note when the
        inside share differs from ``tau`` by more than a tenth.

    Raises:
        SpecificationError: If ``tau`` is not inside ``(0, 1)``, if there
            are too few draws to fit the envelope, or if no draw falls
            inside the truncation ellipsoid.
        DimensionError: If ``draws`` is not one- or two-dimensional, or
            ``log_kernel`` does not have one value per draw.
        NumericalError: If any draw or kernel value is not finite.

    Note:
        The estimator is biased upward when the posterior's tails are
        heavier than the envelope's inside the ellipsoid, and the coverage
        check catches only gross failures; on a skewed posterior the share
        can sit near ``tau`` while the estimate is off by a tenth of a nat.
        For a roughly elliptical posterior :func:`bridge_sampling` has
        lower variance and no such bias, at the price of a callable
        kernel. Records from this function carry ``nobs = 0``, so
        :func:`compare` cannot check that they score the same sample as
        the records they are ranked against; that is the caller's
        responsibility.

    See Also:
        * :func:`bridge_sampling` -- the lower-variance estimator when the
          kernel can be evaluated at new points.
        * :func:`marginal_likelihood` -- the same estimator applied by the
          particle-chain DSGE result to its own draws.
        * :func:`compare` -- ranks the records.

    References:
        Geweke, J. (1999). Using simulation methods for Bayesian
        econometric models: Inference, development, and communication.
        *Econometric Reviews*, 18(1), 1-73.

        Newton, M. A., & Raftery, A. E. (1994). Approximate Bayesian
        inference with the weighted likelihood bootstrap. *Journal of the
        Royal Statistical Society B*, 56(1), 3-48.

    Example:
        Draws from a standard normal in three dimensions with the
        unnormalized kernel :math:`-\tfrac{1}{2}\|z\|^2`; the evidence is
        the missing constant :math:`(2\pi)^{3/2}`:

        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> z = rng.standard_normal((3000, 3))
        >>> record = modified_harmonic_mean(z, -0.5 * (z**2).sum(axis=1))
        >>> bool(abs(record.log_value - 1.5 * np.log(2 * np.pi)) < 0.05)
        True
        >>> record.method, record.n_draws
        ('modified harmonic mean', 3000)

        A bounded parameter left on its own scale biases the estimate; the
        log transform with its Jacobian removes the bias. Here the draws
        are exponential, the kernel is :math:`-x`, and the true log
        evidence is zero:

        >>> rng = np.random.default_rng(3)
        >>> x = rng.exponential(size=4000)
        >>> raw = modified_harmonic_mean(x, -x)
        >>> u = np.log(x)
        >>> transformed = modified_harmonic_mean(u, -np.exp(u) + u)
        >>> round(raw.log_value, 2), round(transformed.log_value, 2)
        (0.12, 0.01)
    """
    log_ml, mcse, coverage = _modified_harmonic_mean(draws, log_kernel, tau=tau)
    notes = ["Modified harmonic mean (Geweke, 1999) with a truncated Gaussian envelope."]
    if abs(coverage - tau) > 0.1:
        notes.append(
            f"Only {coverage:.0%} of the draws fall inside the {tau:.0%} envelope: the "
            "posterior is far from Gaussian, or a bounded parameter was not transformed."
        )
    return MarginalLikelihoodSelection(
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
) -> MarginalLikelihoodSelection:
    r"""Meng-Wong bridge sampling with a Gaussian proposal fitted to the draws.

    Bridge sampling estimates the ratio of two normalizing constants from
    draws of both densities. With the unnormalized posterior :math:`p(\theta)
    = f(y \mid \theta)\, \pi(\theta)` and a normalized proposal
    :math:`q(\theta)`, for any bridge function :math:`\alpha`,

    .. math::

       m(y) = \frac{\mathbb{E}_{q}\bigl[p(\theta)\, \alpha(\theta)\bigr]}
                   {\mathbb{E}_{p/m}\bigl[q(\theta)\, \alpha(\theta)\bigr]},

    and Meng and Wong's optimal bridge, :math:`\alpha \propto 1 / (s_1 p
    + s_2\, m\, q)` with :math:`s_1, s_2` the proposal and posterior shares
    of the draws, depends on :math:`m` itself, so the estimate is the fixed
    point of an iteration started at the median log ratio. The draws are
    split in half: the first half fits the Gaussian proposal, the second
    half enters the bridge, so that the proposal is not fitted to the
    draws it is then compared with. Because the proposal's draws must be
    scored by the posterior, the kernel has to be a callable and not a
    vector of values, and it has to return a finite number wherever a
    Gaussian around the draws can land.

    On a roughly elliptical posterior the estimate has lower variance than
    :func:`modified_harmonic_mean` and none of its truncation bias. The
    reported error is the Frühwirth-Schnatter (2004) approximation, which
    treats the proposal draws as independent and charges the posterior
    half for its autocorrelation through its Geyer effective size.

    Args:
        draws: ``(S,)`` or ``(S, d)`` posterior draws, finite throughout,
            with ``S`` greater than ``2d + 2``. Bounded parameters must be
            transformed to the real line with the Jacobian in the kernel,
            as for :func:`modified_harmonic_mean`; here the requirement is
            strict rather than a matter of bias, because the kernel is
            evaluated at proposal points that may fall outside a bounded
            support.
        log_kernel: ``(S,)`` unnormalized log posterior at each draw.
        kernel: The same log posterior as a function of one ``(d,)`` point,
            returning a float. It is called once per proposal draw.
        seed: Seed or generator for the proposal draws. An integer gives a
            reproducible estimate; a :class:`numpy.random.Generator` is
            used in place and advanced.
        n_proposal: Proposal draws, at least eight. ``None`` uses as many
            as enter the bridge from the posterior side, ``S // 2``.
        source: Label for the record's ``source`` field.

    Returns:
        The :class:`~cultivars.bayes.combination.MarginalLikelihoodSelection`
        with ``method`` set to ``"bridge sampling"``, the log estimate, its
        approximate Monte Carlo standard error, ``n_draws`` the number of
        posterior draws supplied, ``nobs`` zero, and a note recording the
        proposal and the iterations to convergence.

    Raises:
        SpecificationError: If there are too few draws to fit the proposal,
            or ``n_proposal`` is below eight.
        DimensionError: If ``draws`` is not one- or two-dimensional, or
            ``log_kernel`` does not have one value per draw.
        NumericalError: If any draw or kernel value is not finite, if the
            kernel returns a non-finite value at a proposal point, or if
            the fixed-point iteration does not converge within a thousand
            steps.

    Note:
        The kernel is evaluated ``n_proposal`` times, once per proposal
        draw; for a model whose likelihood is a Kalman filter that is
        ``S // 2`` filter passes by default. Records from this function
        carry ``nobs = 0``, so :func:`compare` cannot check that they
        score the same sample as the records they are ranked against.

    See Also:
        * :func:`modified_harmonic_mean` -- when only kernel values at the
          draws are available.
        * :func:`compare` -- ranks the records.

    References:
        Meng, X.-L., & Wong, W. H. (1996). Simulating ratios of normalizing
        constants via a simple identity: A theoretical exploration.
        *Statistica Sinica*, 6(4), 831-860.

        Frühwirth-Schnatter, S. (2004). Estimating marginal likelihoods for
        mixture and Markov switching models using bridge sampling
        techniques. *Econometrics Journal*, 7(1), 143-167.

        Gronau, Q. F., Sarafoglou, A., Matzke, D., Ly, A., Boehm, U.,
        Marsman, M., Leslie, D. S., Forster, J. J., Wagenmakers, E.-J., &
        Steingroever, H. (2017). A tutorial on bridge sampling. *Journal of
        Mathematical Psychology*, 81, 80-97.

    Example:
        A bivariate standard normal, kernel :math:`-\tfrac{1}{2}\|z\|^2`,
        whose evidence is :math:`2\pi`:

        >>> import numpy as np
        >>> rng = np.random.default_rng(2)
        >>> z = rng.standard_normal((3000, 2))
        >>> record = bridge_sampling(
        ...     z, -0.5 * (z**2).sum(axis=1), lambda x: -0.5 * float(x @ x), seed=0
        ... )
        >>> bool(abs(record.log_value - np.log(2 * np.pi)) < 0.05)
        True
        >>> record.method, record.mcse < 0.01
        ('bridge sampling', True)
        >>> record.notes[0]  # doctest: +ELLIPSIS
        'Bridge sampling (Meng & Wong, 1996), ... converged in ... iterations.'
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    log_ml, mcse, n_iter = _bridge_sampling(
        draws, log_kernel, kernel, rng=rng, n_proposal=n_proposal
    )
    return MarginalLikelihoodSelection(
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
    *records: MarginalLikelihoodSelection,
    names: Sequence[str] | None = None,
    prior_probabilities: Sequence[float] | None = None,
) -> SummaryTable:
    r"""Rank models by marginal likelihood with posterior model probabilities.

    One row per record, best first. Each row carries the log marginal
    likelihood, its Monte Carlo standard error, the log Bayes factor
    against the best model :math:`\log B = \log m_i(y) - \max_j \log
    m_j(y)`, the Kass-Raftery reading of :math:`2 \log B` (from "not worth
    more than a bare mention" through "very strong"), the posterior model
    probability

    .. math::

       p(M_i \mid y) = \frac{\pi_i\, m_i(y)}{\sum_j \pi_j\, m_j(y)},

    and the method that produced the number. The notes warn when the
    records score different numbers of observations, since their
    differences are then not Bayes factors, and when a model sits within
    two Monte Carlo standard errors of the best, since the ranking is then
    not resolved by the estimates at hand.

    Args:
        *records: Two or more
            :class:`~cultivars.bayes.combination.MarginalLikelihoodSelection`
            records computed on the same sample, from
            :func:`marginal_likelihood`, :func:`modified_harmonic_mean`, or
            :func:`bridge_sampling`.
        names: Row labels, one per record, in order. ``None`` uses each
            record's ``source``.
        prior_probabilities: :math:`\pi_i`, one per record, non-negative
            and not all zero; normalized to sum to one. ``None`` gives
            every model the same prior probability.

    Returns:
        The :class:`~cultivars._core.SummaryTable` with columns ``model``,
        ``log ML``, ``mcse``, ``log BF vs best``, ``evidence``,
        ``post. prob.``, and ``method``; ``print`` it or call its
        ``to_pandas()``.

    Raises:
        SpecificationError: If fewer than two records are given, or a prior
            probability is negative or all are zero.
        DimensionError: If ``names`` or ``prior_probabilities`` do not have
            one entry per record.

    See Also:
        * :func:`~cultivars.bayes.combination.bayesian_model_average` --
          the same posterior model probabilities as weights on a mixture
          predictive, rather than as a ranking.
        * :func:`marginal_likelihood` -- produces the records for package
          results.

    References:
        Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *Journal of the
        American Statistical Association*, 90(430), 773-795.

    Example:
        Two conjugate BVARs differing in prior tightness, ranked; the six-nat
        gap reads as very strong evidence for the looser prior:

        >>> import numpy as np
        >>> from cultivars.bayes.priors import NormalInverseWishartPrior
        >>> from cultivars.multivariate.large_dim.bayesian import BVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((120, 2))
        >>> for t in range(1, 120):
        ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
        >>> loose = marginal_likelihood(BVAR(y, order=1).fit(n_draws=100, seed=0))
        >>> tight = BVAR(y, order=1, prior=NormalInverseWishartPrior(tightness=0.02))
        >>> tight = marginal_likelihood(tight.fit(n_draws=100, seed=0))
        >>> table = compare(loose, tight, names=("loose", "tight"))
        >>> table.columns
        ('model', 'log ML', 'mcse', 'log BF vs best', 'evidence', 'post. prob.', 'method')
        >>> table.rows[0]
        ('loose', '-357.408', '0.000', '0.000', 'best', '0.998', 'analytic')
        >>> table.rows[1][0], table.rows[1][3], table.rows[1][4]
        ('tight', '-6.140', 'very strong')
    """
    if len(records) < 2:
        raise SpecificationError("compare needs at least two records.")
    return records[0].compare(*records[1:], names=names, prior_probabilities=prior_probabilities)
