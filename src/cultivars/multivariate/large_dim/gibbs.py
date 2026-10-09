# filepath: /src/cultivars/multivariate/large_dim/gibbs.py
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
r"""The Gibbs-sampled BVAR: every prior the conjugate model must refuse.

The conjugate model buys its exact posterior by tying the coefficient
prior to the innovation covariance through a Kronecker product, and two
whole families of priors cannot pay that price. Litterman's original
prior -- cross-variable coefficients shrunk harder than own lags --
breaks the factorization by design. And the adaptive shrinkage
hierarchies -- the horseshoe, spike-and-slab selection,
Dirichlet-Laplace, Normal-Gamma -- have no fixed variance to factor at
all: each coefficient's variance is a latent state with its own full
conditional. This model is where both families estimate. The prior is
stated *independently* of the covariance, the independent
Normal-Wishart pairing of Koop and Korobilis (2010),

.. math::

   \beta \sim N(\underline\beta, \underline V),
   \qquad
   \Sigma \sim IW(\underline S, \underline\nu),

and the posterior is reached by Gibbs sampling: one joint
generalized-least-squares draw of all coefficients given the covariance,

.. math::

   \beta \mid \Sigma, y \sim N(\bar\beta, \bar V),
   \qquad
   \bar V^{-1} = \underline V^{-1} + \Sigma^{-1} \otimes X'X,

exact and ordering-free; an inverse-Wishart draw of the covariance given
the coefficients from the residual scatter; and, for an adaptive prior,
one sweep of its scale hierarchy's own exact conditionals, which rewrites
:math:`\underline V` before the next coefficient draw.

Two commitments shape the surface. First, what is given up is stated
rather than hidden. With the prior independent of the covariance the
evidence has no closed form; the marginal likelihood offered is Chib's
(1995) estimate from the Gibbs output, which the two-block sampler makes
exact for a static moment prior, and which is refused -- not
approximated -- under an adaptive prior, whose scale block would need
reduced runs, and under a dummy-observation prior, whose artificial rows
the sampler treats as data. Draws are a Markov chain, not independent,
so burn-in and thinning are real choices, and the result's
``convergence`` says whether they were enough. And the joint coefficient
draw factorizes nothing, so each sweep costs a Cholesky of a
:math:`(k^2 p + k n_{det})`-square precision: exact at any moderate
dimension, and deliberately not pretending to Banbura-scale systems,
which belong to the conjugate path. Second, the adaptive prior's latent
diagnostic is a first-class output, Rao-Blackwellized over sweeps: for a
selection prior :meth:`GibbsBVARResult.inclusion_probabilities` -- the
posterior probability that each lag coefficient is in the slab, which is
the George-Sun-Ni restriction search read as a result rather than a
procedure -- and for a global-local prior
:meth:`GibbsBVARResult.shrinkage_scales`, how much freedom each
standardized coefficient retained.

Layout. :class:`GibbsBVAR` validates the specification on
``_GibbsBayesianVectorAutoRegressionModel`` in ``_internals``, whose
``_fit_gibbs`` builds the design, lays the prior's moments and any dummy
rows out through ``_gibbs_static_inputs``, runs the sweep -- the GLS
coefficient draw inline, ``_draw_inverse_wishart`` from
``_core._samplers`` for the covariance, and the prior's own
``_draw_scales`` and ``_scale_variance`` for an adaptive hierarchy -- and
packs a ``_VectorGibbsFit``; its ``_marginal_likelihood`` evaluates the
Chib identity. :class:`GibbsBVARResult` extends
``_VectorPosteriorDrawsResult``, which supplies the credible intervals,
the posterior-band impulse responses, the posterior predictive, the
stability share, the chain diagnostics and the posterior replications.
The priors are in :mod:`~cultivars.bayes.priors`; the conjugate model is
:mod:`~cultivars.multivariate.large_dim.bayesian`; the same sampler with
stochastic volatility on the innovations is
:mod:`~cultivars.multivariate.large_dim.volatility`.

References:
    Koop, G., & Korobilis, D. (2010). Bayesian multivariate time series
    methods for empirical macroeconomics. *Foundations and Trends in
    Econometrics*, 3(4), 267-358.

    George, E. I., Sun, D., & Ni, S. (2008). Bayesian stochastic search
    for VAR model restrictions. *Journal of Econometrics*, 142(1),
    553-580.

    Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe
    estimator for sparse signals. *Biometrika*, 97(2), 465-480.

    Griffin, J. E., & Brown, P. J. (2010). Inference with normal-gamma
    prior distributions in regression problems. *Bayesian Analysis*,
    5(1), 171-188.

    Chib, S. (1995). Marginal likelihood from the Gibbs output. *Journal
    of the American Statistical Association*, 90(432), 1313-1321.

Example:
    A sparse VAR(1) -- own lags only -- under three priors: the
    selection prior finds the sparsity pattern, the horseshoe shrinks
    the cross lags, and Chib's evidence ranks the static priors:

    >>> import numpy as np
    >>> from cultivars.bayes.priors import (
    ...     HorseshoePrior,
    ...     IndependentNormalWishartPrior,
    ...     SpikeAndSlabPrior,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((200, 3))
    >>> for t in range(1, 200):
    ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(3)
    >>> selected = GibbsBVAR(y, order=1, prior=SpikeAndSlabPrior()).fit(
    ...     n_draws=300, n_burn=100, seed=0
    ... )
    >>> included = selected.inclusion_probabilities()[0] > 0.5
    >>> bool(np.array_equal(included, np.eye(3, dtype=bool)))
    True
    >>> shrunk = GibbsBVAR(y, order=1, prior=HorseshoePrior()).fit(
    ...     n_draws=300, n_burn=100, seed=0
    ... )
    >>> cross = ~np.eye(3, dtype=bool)
    >>> bool(np.abs(shrunk.coefficients[0][cross]).max() < 0.15)
    True
    >>> loose = GibbsBVAR(y, order=1).fit(n_draws=300, n_burn=100, seed=0)
    >>> tight = GibbsBVAR(
    ...     y, order=1, prior=IndependentNormalWishartPrior(tightness=0.05)
    ... ).fit(n_draws=300, n_burn=100, seed=0)
    >>> bool(loose.marginal_likelihood().log_value > tight.marginal_likelihood().log_value)
    True
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ...bayes.priors import IndependentNormalWishartPrior
from ...engine._core import SummaryTable, Trend
from ...engine._internals import (
    _GibbsBayesianVectorAutoRegressionModel,
    _MarginalLikelihoodSelection,
    _Prior,
    _VectorGibbsFit,
    _VectorPosteriorDrawsResult,
)
from ...exceptions import SpecificationError

__all__ = ["GibbsBVAR", "GibbsBVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class GibbsBVARResult(_VectorPosteriorDrawsResult):
    r"""A fitted Gibbs BVAR: a sampled posterior, and the prior's own diagnostics.

    The posterior of

    .. math::

       y_t = c + \sum_{i=1}^{p} A_i y_{t-i} + u_t,
       \qquad u_t \sim N(0, \Sigma),

    under a prior on the stacked coefficients :math:`\beta` stated
    independently of :math:`\Sigma`, reached by Gibbs sampling: a joint
    generalized-least-squares draw of :math:`\beta \mid \Sigma, y`, an
    inverse-Wishart draw of :math:`\Sigma \mid \beta, y`, and, under an
    adaptive prior, one sweep of the scale hierarchy's conditionals. The
    propagation surface -- credible intervals, impulse responses with
    posterior bands, the posterior predictive, the stability share, the
    chain diagnostics -- is the family's shared one from
    ``_VectorPosteriorDrawsResult``. What this result adds is the adaptive
    prior's averaged latent diagnostic -- inclusion probabilities under a
    selection prior, local shrinkage scales under a global-local one --
    and Chib's marginal likelihood where the two-block sampler makes it
    exact.

    Note:
        ``marginal_likelihood`` is available under a static moment prior
        and refused under an adaptive prior (whose scale block would need
        reduced runs) or a dummy-observation prior (whose artificial rows
        the sampler treats as data). ``shrinkage`` is the posterior mean
        of the latent diagnostic, Rao-Blackwellized over kept sweeps: a
        conditional inclusion probability, not a binary indicator, under
        a selection prior; a conditional prior standard deviation on the
        standardized scale under a global-local prior. The draws are a
        Markov chain -- burn-in and thinning are real choices and
        ``convergence`` says whether they were enough. The point
        summaries (``coefficients``, ``sigma_u``, ``resid``) are at the
        posterior mean and inherit its caveats.

    Attributes:
        prior_label: Short description of the prior estimated under.
        shrinkage: ``(w, k)`` posterior mean per-coefficient diagnostic from
            an adaptive prior, in design-row order; empty under a static
            prior. Read it through :meth:`inclusion_probabilities` or
            :meth:`shrinkage_scales`.
        shrinkage_label: What ``shrinkage`` is; empty under a static prior.
        n_dummy: Artificial rows the prior contributed.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.

    See Also:
        * :class:`GibbsBVAR` -- the model that produces this record.
        * :class:`~cultivars.multivariate.large_dim.bayesian.BVARResult`
          -- the conjugate posterior, exact and with a closed-form
          marginal likelihood.
        * :class:`~cultivars.bayes.priors.SpikeAndSlabPrior` -- the
          selection prior whose inclusion map this result reports.

    References:
        Koop, G., & Korobilis, D. (2010). Bayesian multivariate time
        series methods for empirical macroeconomics. *Foundations and
        Trends in Econometrics*, 3(4), 267-358.

        George, E. I., Sun, D., & Ni, S. (2008). Bayesian stochastic
        search for VAR model restrictions. *Journal of Econometrics*,
        142(1), 553-580.

        Chib, S. (1995). Marginal likelihood from the Gibbs output.
        *Journal of the American Statistical Association*, 90(432),
        1313-1321.

    Example:
        A diagonal VAR(1) under a selection prior: the own lags are
        included with probability near one and the cross lags are not:

        >>> import numpy as np
        >>> from cultivars.bayes.priors import SpikeAndSlabPrior
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((200, 3))
        >>> for t in range(1, 200):
        ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(3)
        >>> res = GibbsBVAR(y, order=1, prior=SpikeAndSlabPrior()).fit(
        ...     n_draws=300, n_burn=100, seed=0
        ... )
        >>> res.prior_label, res.shrinkage_label, res.n_kept, res.n_dummy
        ('ssvs(spike=0.1, slab=10, p=0.5)', 'posterior inclusion probability', 200, 0)
        >>> res.shrinkage.shape, res.beta_draws.shape, res.sigma_draws.shape
        ((4, 3), (200, 4, 3), (200, 3, 3))
        >>> included = res.inclusion_probabilities()[0]
        >>> off_diagonal = included[~np.eye(3, dtype=bool)]
        >>> bool(np.all(np.diag(included) > 0.9)), bool(off_diagonal.max() < 0.5)
        (True, True)
        >>> low, mid, high = res.credible_interval("y1", "y1.L1")
        >>> bool(0.3 < low < mid < high < 0.8)
        True
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    """The observed panel, shape ``(nobs_total, k)``. Kept out of the repr."""
    names: tuple[str, ...]
    """Variable labels, in column order."""
    order: int
    """The autoregressive order ``p``."""
    trend: str
    """The deterministic specification: ``"n"``, ``"c"`` or ``"ct"``."""
    prior_label: str
    """Short description of the prior estimated under."""
    coefficients: npt.NDArray[np.float64] = field(repr=False)
    """``(p, k, k)`` lag stack at the posterior mean. Kept out of the repr."""
    deterministic: npt.NDArray[np.float64] = field(repr=False)
    """``(n_det, k)`` deterministic block at the posterior mean. Kept out of the repr."""
    beta_mean: npt.NDArray[np.float64] = field(repr=False)
    """``(w, k)`` full posterior mean coefficient matrix. Kept out of the repr."""
    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    """``(k, k)`` posterior mean innovation covariance. Kept out of the repr."""
    beta_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, w, k)`` coefficient draws. Kept out of the repr."""
    sigma_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, k, k)`` covariance draws. Kept out of the repr."""
    shrinkage: npt.NDArray[np.float64] = field(repr=False)
    """``(w, k)`` averaged latent diagnostic of an adaptive prior; empty otherwise."""
    shrinkage_label: str
    """What ``shrinkage`` measures; empty under a static prior."""
    resid: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, k)`` residuals at the posterior mean. Kept out of the repr."""
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, k)`` one-step means at the posterior mean. Kept out of the repr."""
    nobs: int
    """Effective sample size after the ``p`` presample rows."""
    n_dummy: int
    """Artificial rows a dummy-observation prior contributed."""
    n_draws: int
    """Total sampler iterations, burn-in included."""
    n_burn: int
    """Iterations discarded from the start."""
    thin: int
    """Every ``thin``-th post-burn iteration is kept."""
    _engine: _GibbsBayesianVectorAutoRegressionModel[GibbsBVARResult] = field(repr=False)
    """The model, kept for the marginal-likelihood estimator. Kept out of the repr."""

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorGibbsFit,
        model: _GibbsBayesianVectorAutoRegressionModel[GibbsBVARResult],
    ) -> GibbsBVARResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The packed draws from ``_fit_gibbs``.
            model: The specification the draws were produced for; kept on
                the record so ``marginal_likelihood`` can evaluate the
                prior and likelihood ordinates.

        Returns:
            The public posterior record.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal((120, 2))
            >>> model = GibbsBVAR(y, order=1)
            >>> res = GibbsBVARResult._from_fit(
            ...     model._fit_gibbs(n_draws=40, n_burn=10, thin=1, seed=0), model
            ... )
            >>> res.names, res.n_kept, res.prior_label.startswith("inw")
            (('y1', 'y2'), 30, True)
        """
        return cls(
            endog=model.endog,
            names=model.names,
            order=model.order,
            trend=model.trend,
            prior_label=model.prior._label(),
            coefficients=fit.coefficient_stack,
            deterministic=fit.deterministic,
            beta_mean=fit.beta_mean,
            sigma_u=fit.sigma_u,
            beta_draws=fit.beta_draws,
            sigma_draws=fit.sigma_draws,
            shrinkage=fit.shrinkage,
            shrinkage_label=fit.shrinkage_label,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            nobs=fit.nobs,
            n_dummy=fit.n_dummy,
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
            _engine=model,
        )

    def marginal_likelihood(self) -> _MarginalLikelihoodSelection:
        r"""Chib's (1995) log marginal likelihood of the sample.

        The two-block identity

        .. math::

           \log m(y) = \log p(y \mid \beta^*, \Sigma^*) + \log p(\beta^*, \Sigma^*)
           - \log p(\Sigma^* \mid y) - \log p(\beta^* \mid \Sigma^*, y),

        at the posterior mean :math:`(\beta^*, \Sigma^*)`, with the
        covariance ordinate averaged over the retained coefficient draws
        and the coefficient ordinate evaluated exactly from its Gaussian
        conditional. The Monte-Carlo standard error of the average is on
        the record.

        Returns:
            The record; ``compare()`` on it ranks models fitted to the same
            sample.

        Raises:
            SpecificationError: Under an adaptive prior, whose extra Gibbs
                block would need reduced runs, or a dummy-observation prior,
                whose artificial rows the sampler treats as data.

        Example:
            Two static priors on one sample, ranked by Bayes factor:

            >>> import numpy as np
            >>> from cultivars.bayes.priors import IndependentNormalWishartPrior, SpikeAndSlabPrior
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((200, 3))
            >>> for t in range(1, 200):
            ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(3)
            >>> loose = GibbsBVAR(y, order=1).fit(n_draws=300, n_burn=100, seed=0)
            >>> tight = GibbsBVAR(
            ...     y, order=1, prior=IndependentNormalWishartPrior(tightness=0.05)
            ... ).fit(n_draws=300, n_burn=100, seed=0)
            >>> evidence = loose.marginal_likelihood()
            >>> evidence.method, bool(evidence.mcse < 0.1)
            ('chib', True)
            >>> bool(evidence.log_value > tight.marginal_likelihood().log_value)
            True
            >>> adaptive = GibbsBVAR(y, order=1, prior=SpikeAndSlabPrior()).fit(
            ...     n_draws=60, n_burn=20, seed=0
            ... )
            >>> adaptive.marginal_likelihood()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: Chib's estimator here covers the two-block ...
        """
        return self._engine._marginal_likelihood(
            self.beta_draws,
            self.sigma_draws,
            source=f"Gibbs BVAR({self.order}), {self.prior_label}",
        )

    def _lag_diagnostic(self) -> npt.NDArray[np.float64]:
        """The lag-block rows of ``shrinkage`` as a ``(p, k, k)`` stack.

        Skips the deterministic rows and transposes each lag block so
        that entry ``[lag, i, j]`` refers to equation ``i`` and regressor
        ``j``, matching ``coefficients``.

        Returns:
            A ``(p, k, k)`` stack.

        Example:
            >>> import numpy as np
            >>> from cultivars.bayes.priors import HorseshoePrior
            >>> y = np.random.default_rng(0).standard_normal((150, 2))
            >>> res = GibbsBVAR(y, order=2, prior=HorseshoePrior()).fit(
            ...     n_draws=60, n_burn=20, seed=0
            ... )
            >>> res.shrinkage.shape, res._lag_diagnostic().shape
            ((5, 2), (2, 2, 2))
            >>> bool(np.allclose(res._lag_diagnostic()[0], res.shrinkage[1:3].T))
            True
        """
        offset, k = self._n_deterministic, self.k_endog
        return np.stack(
            [
                self.shrinkage[offset + lag * k : offset + (lag + 1) * k, :].T
                for lag in range(self.order)
            ]
        )

    def inclusion_probabilities(self) -> npt.NDArray[np.float64]:
        """Posterior inclusion probability of each lag coefficient.

        Entry ``[lag, i, j]`` is the posterior probability that variable
        ``j``'s coefficient at that lag in equation ``i`` sits in the slab
        -- is included -- averaged over kept sweeps
        (Rao-Blackwellized: the averaged quantity is the exact conditional
        probability, not the binary indicator).

        Returns:
            A ``(p, k, k)`` stack aligned with :attr:`coefficients`.

        Raises:
            SpecificationError: If the prior was not a selection prior --
                only a spike-and-slab posterior defines inclusion.

        Example:
            >>> import numpy as np
            >>> from cultivars.bayes.priors import HorseshoePrior, SpikeAndSlabPrior
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((200, 3))
            >>> for t in range(1, 200):
            ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(3)
            >>> res = GibbsBVAR(y, order=1, prior=SpikeAndSlabPrior()).fit(
            ...     n_draws=300, n_burn=100, seed=0
            ... )
            >>> included = res.inclusion_probabilities()
            >>> included.shape, bool(np.all((0 <= included) & (included <= 1)))
            ((1, 3, 3), True)
            >>> bool(np.all(np.diag(included[0]) > 0.9))
            True
            >>> GibbsBVAR(y, order=1, prior=HorseshoePrior()).fit(
            ...     n_draws=60, n_burn=20, seed=0
            ... ).inclusion_probabilities()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: inclusion probabilities exist only under ...
        """
        if "inclusion" not in self.shrinkage_label:
            raise SpecificationError(
                "inclusion probabilities exist only under a selection prior; "
                f"this model was fit under {self.prior_label}. Global-local "
                "shrinkage reports shrinkage_scales() instead."
            )
        return self._lag_diagnostic()

    def shrinkage_scales(self) -> npt.NDArray[np.float64]:
        """Posterior mean local-global scale of each standardized lag coefficient.

        Entry ``[lag, i, j]`` is the averaged prior standard deviation the
        hierarchy assigned to that coefficient on the standardized
        (unit-free) scale: near zero means shrunk away, order one means
        left free. Comparable within a fit, not across priors.

        Returns:
            A ``(p, k, k)`` stack aligned with :attr:`coefficients`.

        Raises:
            SpecificationError: If the prior was not a global-local
                shrinkage prior.

        Example:
            Under the horseshoe the own lags of a diagonal VAR keep a
            larger scale than the cross lags:

            >>> import numpy as np
            >>> from cultivars.bayes.priors import HorseshoePrior
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((200, 3))
            >>> for t in range(1, 200):
            ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(3)
            >>> res = GibbsBVAR(y, order=1, prior=HorseshoePrior()).fit(
            ...     n_draws=300, n_burn=100, seed=0
            ... )
            >>> scale = res.shrinkage_scales()[0]
            >>> scale.shape, bool(np.diag(scale).min() > scale[~np.eye(3, dtype=bool)].max())
            ((3, 3), True)
            >>> static = GibbsBVAR(y, order=1).fit(n_draws=60, n_burn=20, seed=0)
            >>> static.shrinkage_scales()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: shrinkage scales exist only under a ...
        """
        if "scale" not in self.shrinkage_label:
            raise SpecificationError(
                "shrinkage scales exist only under a global-local prior; "
                f"this model was fit under {self.prior_label}."
            )
        return self._lag_diagnostic()

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Returns:
            One row per equation with the posterior mean and 68% interval
            of its own first lag; the metadata carries the prior, the draw
            bookkeeping, the dimensions and the dummy rows; the notes
            state the sampler, the marginal-likelihood scope, the
            stability share, the absence of criteria, the predictive
            semantics and -- under an adaptive prior -- the diagnostic
            that is available.

        Example:
            >>> import numpy as np
            >>> from cultivars.bayes.priors import SpikeAndSlabPrior
            >>> y = np.random.default_rng(0).standard_normal((150, 2))
            >>> table = GibbsBVAR(y, order=1).fit(n_draws=60, n_burn=20, seed=0)._summary_table()
            >>> table.title, table.columns, len(table.rows), len(table.notes)
            ('Gibbs BVAR(1) Results', ('equation', 'own L1 mean', '68% interval'), 2, 5)
            >>> table.metadata[2]
            ('Draws', '40 kept of 60')
            >>> selected = GibbsBVAR(y, order=1, prior=SpikeAndSlabPrior()).fit(
            ...     n_draws=60, n_burn=20, seed=0
            ... )
            >>> selected._summary_table().notes[0][:16]
            'Selection prior:'
        """
        rows = []
        for name in self.names:
            if self.order:
                low, mid, high = self.credible_interval(name, f"{name}.L1")
                rows.append((name, f"{mid:.4f}", f"[{low:.4f}, {high:.4f}]"))
            else:
                rows.append((name, "-", "-"))
        notes = [
            "The posterior is sampled by Gibbs (joint GLS coefficient draw, "
            "inverse-Wishart covariance), so draws are a Markov chain: "
            "burn-in and thinning matter, and independent-draw intuition "
            "does not apply.",
            "marginal_likelihood() gives Chib's estimate from the Gibbs output "
            "for static priors; adaptive and dummy-observation priors are refused.",
            self._stability_note(),
            "No llf, parameter count, or information criteria are reported: a posterior has none.",
            "forecast() is the full posterior predictive -- parameter and "
            "shock uncertainty jointly.",
        ]
        if self.shrinkage_label:
            if "inclusion" in self.shrinkage_label:
                included = int(np.sum(self._lag_diagnostic() > 0.5))
                total = self.order * self.k_endog**2
                notes.insert(
                    0,
                    f"Selection prior: {included} of {total} lag coefficients "
                    "carry posterior inclusion probability above one half; "
                    "inclusion_probabilities() has the full map.",
                )
            else:
                notes.insert(
                    0,
                    "Global-local prior: shrinkage_scales() maps how much "
                    "freedom each standardized lag coefficient retained.",
                )
        return SummaryTable(
            title=f"Gibbs BVAR({self.order}) Results",
            metadata=(
                ("Model", f"GibbsBVAR({self.order})"),
                ("Prior", self.prior_label),
                ("Draws", f"{self.n_kept} kept of {self.n_draws}"),
                ("Burn-in", f"{self.n_burn}"),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.nobs}"),
                ("Dummy rows", f"{self.n_dummy}"),
                ("Trend", self.trend),
            ),
            columns=("equation", "own L1 mean", "68% interval"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class GibbsBVAR(_GibbsBayesianVectorAutoRegressionModel[GibbsBVARResult]):
    r"""Non-conjugate Bayesian VAR: independent Normal-Wishart, sampled by Gibbs.

    One model, two families of priors the conjugate path refuses. Static
    moment priors with any diagonal variance -- the default
    :class:`~cultivars.bayes.priors.IndependentNormalWishartPrior`, which
    is Litterman's full prior with the cross-equation weight kept, or a
    :class:`~cultivars.bayes.priors.MinnesotaPrior` passed directly --
    and the adaptive shrinkage hierarchies:
    :class:`~cultivars.bayes.priors.HorseshoePrior`,
    :class:`~cultivars.bayes.priors.SpikeAndSlabPrior`,
    :class:`~cultivars.bayes.priors.DirichletLaplacePrior`,
    :class:`~cultivars.bayes.priors.NormalGammaPrior`. The prior is
    :math:`\beta \sim N(\underline\beta, \underline V)` independent of
    :math:`\Sigma \sim IW(\underline S, \underline\nu)`, so the
    coefficient conditional is the GLS posterior

    .. math::

       \bar V = \bigl(\underline V^{-1} + \Sigma^{-1} \otimes X'X\bigr)^{-1},
       \qquad
       \bar\beta = \bar V \bigl(\underline V^{-1} \underline\beta
       + (\Sigma^{-1} \otimes X')\, y\bigr),

    one :math:`(k^2 p + k n_{det})`-dimensional Gaussian per sweep, and
    the covariance conditional is inverse-Wishart on the residual
    scatter. Static moment priors may be composed with ``+`` (a
    :class:`~cultivars.bayes.priors.SumOfCoefficientsPrior` or
    :class:`~cultivars.bayes.priors.DummyInitialObservationPrior` adds
    artificial rows the sampler treats as data); adaptive priors stand
    alone.

    Attributes:
        _endog: The validated ``(nobs_total, k)`` panel.
        _order: The autoregressive order.
        _trend: The deterministic specification.
        _names: One label per variable.
        _prior: The validated prior.

    Args:
        endog: The observed panel, shape ``(nobs, k)``. A proper prior
            makes wide systems admissible, though each Gibbs sweep costs a
            Cholesky of the joint coefficient precision, so very large
            ``k`` belongs to the conjugate model.
        order: Autoregressive order, at least 1.
        prior: Any finite-variance prior, static or adaptive. Defaults to
            ``IndependentNormalWishartPrior()``. Adaptive priors do not
            compose with ``+``.
        trend: Deterministic terms, ``"n"``, ``"c"`` or ``"ct"``.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Raises:
        SpecificationError: If the prior is improper (``NoPrior``),
            ``trend`` is not one of the three names, or ``names`` does not
            have one entry per column. An adaptive prior inside a
            composition is accepted here and refused at ``fit``.
        DimensionError: If the panel is not a time-down-the-rows matrix or
            too short for the order.

    See Also:
        * :class:`GibbsBVARResult` -- the posterior record ``fit`` returns.
        * :class:`~cultivars.multivariate.large_dim.bayesian.BVAR` -- the
          conjugate model: exact posterior, closed-form evidence, any
          dimension, but a Kronecker-restricted prior.
        * :class:`~cultivars.multivariate.large_dim.volatility.BVARSV` --
          the same sampler with stochastic volatility on the innovations.

    References:
        Koop, G., & Korobilis, D. (2010). Bayesian multivariate time
        series methods for empirical macroeconomics. *Foundations and
        Trends in Econometrics*, 3(4), 267-358.

        George, E. I., Sun, D., & Ni, S. (2008). Bayesian stochastic
        search for VAR model restrictions. *Journal of Econometrics*,
        142(1), 553-580.

    Example:
        >>> import numpy as np
        >>> from cultivars.bayes.priors import SpikeAndSlabPrior
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((200, 3))
        >>> for t in range(1, 200):
        ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(3)
        >>> model = GibbsBVAR(y, order=1, prior=SpikeAndSlabPrior())
        >>> res = model.fit(n_draws=300, n_burn=100, seed=0)
        >>> res.inclusion_probabilities().shape
        (1, 3, 3)
        >>> bool(np.all(np.diag(res.inclusion_probabilities()[0]) > 0.5))
        True
        >>> from cultivars.bayes.priors import HorseshoePrior, SumOfCoefficientsPrior
        >>> composed = GibbsBVAR(y, order=1, prior=HorseshoePrior() + SumOfCoefficientsPrior())
        >>> composed.fit(n_draws=60, n_burn=20, seed=0)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: adaptive shrinkage priors do not compose: ...
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        prior: _Prior | None = None,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
    ) -> None:
        """Default the prior to the independent Normal-Wishart, then validate.

        Args:
            endog: The observed panel.
            order: Autoregressive order.
            prior: A proper prior, or ``None`` for the default.
            trend: Deterministic terms.
            names: One label per variable, or ``None`` for defaults.

        Example:
            >>> import numpy as np
            >>> model = GibbsBVAR(np.random.default_rng(0).standard_normal((100, 2)), order=2)
            >>> model.order, model.trend, model.names, model.prior._label()[:3]
            (2, 'c', ('y1', 'y2'), 'inw')
        """
        super().__init__(
            endog,
            order=order,
            trend=trend,
            names=names,
            prior=prior if prior is not None else IndependentNormalWishartPrior(),
        )

    def fit(
        self,
        *,
        n_draws: int = 2000,
        n_burn: int = 500,
        thin: int = 1,
        seed: int | np.random.Generator | None = None,
    ) -> GibbsBVARResult:
        """Run the Gibbs sampler and keep the post-burn draws.

        Each sweep draws all coefficients jointly given the covariance,
        the covariance given the coefficients, and -- under an adaptive
        prior -- the scale hierarchy; the first ``n_burn`` sweeps are
        discarded and every ``thin``-th of the rest is kept. The same
        ``seed`` reproduces the chain exactly.

        Args:
            n_draws: Total sampler iterations.
            n_burn: Burn-in iterations discarded, below ``n_draws``.
            thin: Keep every ``thin``-th post-burn draw, at least 1.
            seed: Seed or generator, for reproducibility.

        Returns:
            The fitted :class:`GibbsBVARResult`.

        Raises:
            SpecificationError: If the prior is improper, mixes adaptive
                components into a composition, or the draw bookkeeping is
                inconsistent.
            NumericalError: If a conditional draw collapses.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((200, 2))
            >>> for t in range(1, 200):
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
            >>> model = GibbsBVAR(y, order=1)
            >>> res = model.fit(n_draws=200, n_burn=50, thin=3, seed=0)
            >>> res.n_kept, res.n_draws, res.n_burn, res.thin
            (50, 200, 50, 3)
            >>> again = model.fit(n_draws=200, n_burn=50, thin=3, seed=0)
            >>> bool(np.array_equal(res.beta_draws, again.beta_draws))
            True
            >>> model.fit(n_draws=50, n_burn=50)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: n_draws (50) must exceed n_burn (50).
        """
        return GibbsBVARResult._from_fit(
            self._fit_gibbs(n_draws=n_draws, n_burn=n_burn, thin=thin, seed=seed),
            self,
        )
