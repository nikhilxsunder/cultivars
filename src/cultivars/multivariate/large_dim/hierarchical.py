# filepath: /src/cultivars/multivariate/large_dim/hierarchical.py
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
r"""Hierarchical shrinkage: the tightness is estimated, not asserted.

Every Bayesian VAR result depends on hyperparameters someone chose -- the
Minnesota tightness above all -- and the standard practice of fixing them
at folklore values means the reported bands condition on a number nobody
defends. Giannone, Lenza and Primiceri (2015) close that gap: treat the
hyperparameters :math:`\theta = (\lambda, \mu, \delta)` as unknowns with
their own hyperpriors and let the conjugate model's closed-form marginal
likelihood carry the data's opinion about them,

.. math::

   p(\theta \mid y) \propto p(y \mid \theta)\, p(\theta),
   \qquad
   p(B, \Sigma \mid y) = \int p(B, \Sigma \mid y, \theta)\,
   p(\theta \mid y)\, d\theta,

where :math:`p(y \mid \theta)` is exactly the number
:class:`~cultivars.multivariate.large_dim.bayesian.BVAR` already reports
and :math:`p(B, \Sigma \mid y, \theta)` is its Normal-inverse-Wishart
posterior. The hyperparameter posterior is one density of at most three
dimensions, and everything downstream follows from standard machinery.

Both of the paper's readings are offered, behind one surface. Empirical
Bayes (``method="empirical"``) maximizes the hyperparameter posterior and
conditions on the mode: fast, and what most applied work does. The full
hierarchy (``method="full"``, the default) runs an adaptive random-walk
Metropolis chain over :math:`\log\theta` from the mode and draws
:math:`(B, \Sigma)` from the exact conditional posterior at every kept
state, so the retained draws *marginalize* over the tightness and the
bands stop pretending it was known. On stubbornly informative samples the
two agree; when they disagree, the disagreement is the finding.

Two commitments shape the surface. First, the hierarchy is the paper's,
stated in its parameterization: the Minnesota tightness :math:`\lambda`
and, optionally, the sum-of-coefficients and dummy-initial-observation
loosenesses :math:`\mu` and :math:`\delta` (larger is looser), under
Gamma hyperpriors with mode 0.2 and standard deviation 0.4 for
:math:`\lambda` and mode 1 and standard deviation 1 for :math:`\mu` and
:math:`\delta`; the lag decay, the deterministic looseness and the
persistence prior mean are held fixed as the paper holds them. Second,
what the reported evidence means is said on the result: its
``log_marginal_likelihood`` is :math:`p(y \mid \hat\theta)` at the mode,
conditional on it, not the integrated evidence over the hyperprior; and a
hyperparameter the data cannot update -- :math:`\mu` when the presample
mean is zero and the sum-of-coefficients rows vanish -- comes back with
its hyperprior as its posterior rather than with a number that looks
estimated.

Layout. :class:`HierarchicalBVAR` stores the fixed prior settings and
builds the base model on ``_BayesianVectorAutoRegressionModel`` in
``_internals`` at the hyperprior modes; ``_prior_at`` maps a
hyperparameter vector to a composed
:class:`~cultivars.bayes.priors.NormalInverseWishartPrior` with the
dummy priors at reciprocal tightness, and ``fit`` turns the modes and
standard deviations into Gamma shapes and scales through
``_gamma_from_mode`` from ``_core._samplers``. The base's
``_fit_hierarchical`` finds the mode by Nelder-Mead in log space, runs
the Metropolis chain with a scalar step tuned toward a 0.3 acceptance
rate in blocks of fifty during burn-in and frozen afterwards, draws
``(B, Sigma)`` through ``_draw_conjugate`` at each kept state, and packs
a ``_VectorHierarchicalFit``. :class:`HierarchicalBVARResult` extends
:class:`~cultivars.multivariate.large_dim.bayesian.BVARResult`, so the
credible intervals, impulse responses, predictive and stability share
are the conjugate family's. The fixed-tightness model is
:mod:`~cultivars.multivariate.large_dim.bayesian`; the priors the
hierarchy ranges over are in :mod:`~cultivars.bayes.priors`.

References:
    Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection
    for vector autoregressions. *Review of Economics and Statistics*,
    97(2), 436-451.

    Banbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian
    vector auto regressions. *Journal of Applied Econometrics*, 25(1),
    71-92.

    Sims, C. A., & Zha, T. (1998). Bayesian methods for dynamic
    multivariate models. *International Economic Review*, 39(4),
    949-968.

Example:
    The tightness the data prefer, under both readings, on a bivariate
    VAR(1) with persistence well below the random-walk prior mean:

    >>> import numpy as np
    >>> from cultivars.multivariate.large_dim.bayesian import BVAR
    >>> from cultivars.bayes.priors import NormalInverseWishartPrior
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((150, 2))
    >>> for t in range(1, 150):
    ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(2)
    >>> y += 5.0
    >>> model = HierarchicalBVAR(y, order=1)
    >>> empirical = model.fit(method="empirical", n_draws=200, seed=0)
    >>> full = model.fit(method="full", n_draws=300, n_burn=300, seed=0)
    >>> bool(np.allclose(empirical.hyper_mode, full.hyper_mode))
    True
    >>> low, mean, high = full.hyperparameter_interval("lambda")
    >>> bool(low < empirical.hyper_mode[0] < high)
    True
    >>> fixed = BVAR(y, order=1, prior=NormalInverseWishartPrior(tightness=0.2)).fit(
    ...     n_draws=10, seed=0
    ... )
    >>> bool(empirical.log_marginal_likelihood >= fixed.log_marginal_likelihood)
    True
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable, Trend, _gamma_from_mode
from ..._internals import (
    _BayesianVectorAutoRegressionModel,
    _Prior,
)
from ...bayes.priors import (
    DummyInitialObservationPrior,
    NormalInverseWishartPrior,
    SumOfCoefficientsPrior,
)
from ...exceptions import SpecificationError
from .bayesian import BVARResult

__all__ = ["HierarchicalBVAR", "HierarchicalBVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class HierarchicalBVARResult(BVARResult):
    r"""A fitted hierarchical BVAR: the tightness carries a posterior too.

    Everything the conjugate result reports, with one change of meaning
    and one addition. The change: under the full method the retained
    ``(B, Sigma)`` draws marginalize over the hyperparameters
    :math:`\theta = (\lambda, \mu, \delta)`,

    .. math::

       p(B, \Sigma \mid y) = \int p(B, \Sigma \mid y, \theta)\,
       p(\theta \mid y)\, d\theta,
       \qquad
       p(\theta \mid y) \propto p(y \mid \theta)\, p(\theta),

    with :math:`p(y \mid \theta)` the conjugate model's closed-form
    marginal likelihood, so every band downstream -- coefficients,
    impulse responses, the predictive -- includes hyperparameter
    uncertainty; :attr:`log_marginal_likelihood` is the value *at the
    hyperparameter mode*, conditional on it. The addition is the
    hyperparameter layer itself: the posterior mode, the kept draws, and
    the chain's acceptance rate.

    Note:
        Under ``method="empirical"`` the ``(B, Sigma)`` draws condition on
        ``hyper_mode``, ``hyper_draws`` is empty and ``acceptance_rate``
        is ``nan``; ``hyperparameter_interval`` refuses. A hyperparameter
        whose dummies carry no information -- ``mu`` when the presample
        mean is zero, since the sum-of-coefficients rows are that mean
        divided by ``mu`` -- has a posterior equal to its hyperprior and a
        mode at the search's start; the marginal likelihood is flat in
        it, which is correct and worth noticing. The summary's
        acceptance-rate guidance is for the kept span; a short run after
        a long adaptive burn-in can sit above it without harm.

    Attributes:
        hyper_names: One label per hyperparameter, in draw-column order --
            ``"lambda"``, then ``"mu"`` and ``"delta"`` when active,
            parameterized as in Giannone-Lenza-Primiceri.
        hyper_mode: Posterior-mode hyperparameter vector.
        hyper_draws: ``(S, d)`` kept hyperparameter draws; empty under
            empirical Bayes.
        acceptance_rate: Metropolis acceptance over the kept span; ``nan``
            under empirical Bayes.
        method: ``"full"`` or ``"empirical"``.

    See Also:
        * :class:`HierarchicalBVAR` -- the model that produces this record.
        * :class:`~cultivars.multivariate.large_dim.bayesian.BVARResult`
          -- the conjugate record this one extends, with the tightness
          fixed.

    References:
        Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior
        selection for vector autoregressions. *Review of Economics and
        Statistics*, 97(2), 436-451.

    Example:
        The full hierarchy on a bivariate VAR(1): the tightness has a
        posterior, the mode agrees with the empirical-Bayes fit, and the
        coefficient bands marginalize over it:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((150, 2))
        >>> for t in range(1, 150):
        ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(2)
        >>> y += 5.0
        >>> model = HierarchicalBVAR(y, order=1)
        >>> full = model.fit(method="full", n_draws=300, n_burn=300, seed=0)
        >>> full.method, full.hyper_names, full.hyper_draws.shape, full.n_kept
        ('full', ('lambda', 'mu', 'delta'), (300, 3), 300)
        >>> low, mean, high = full.hyperparameter_interval("lambda")
        >>> bool(0.0 < low < mean < high < 1.0), bool(0.1 < full.acceptance_rate < 0.9)
        (True, True)
        >>> empirical = model.fit(method="empirical", n_draws=100, seed=0)
        >>> bool(np.allclose(empirical.hyper_mode, full.hyper_mode))
        True
        >>> bool(empirical.log_marginal_likelihood == full.log_marginal_likelihood)
        True
    """

    hyper_names: tuple[str, ...]
    """Hyperparameter labels in draw-column order: ``lambda``, then ``mu``, ``delta``."""
    hyper_mode: npt.NDArray[np.float64] = field(repr=False)
    """``(d,)`` posterior-mode hyperparameter vector. Kept out of the repr."""
    hyper_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, d)`` kept hyperparameter draws; empty under empirical Bayes. Kept out."""
    acceptance_rate: float
    """Metropolis acceptance over the kept span; ``nan`` under empirical Bayes."""
    method: str
    """``"full"`` (marginalized) or ``"empirical"`` (conditioned on the mode)."""

    def hyperparameter_interval(self, name: str) -> npt.NDArray[np.float64]:
        """One hyperparameter's posterior summary: 16th percentile, mean, 84th.

        Args:
            name: A label from :attr:`hyper_names`.

        Returns:
            A ``(3,)`` array ``(low, mean, high)``.

        Raises:
            SpecificationError: If the name is unknown, or the fit was
                empirical Bayes and holds no hyperparameter draws.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = 5.0 + np.cumsum(0.0 * rng.standard_normal((150, 2)), axis=0)
            >>> y += rng.standard_normal((150, 2))
            >>> model = HierarchicalBVAR(y, order=1)
            >>> full = model.fit(method="full", n_draws=200, n_burn=200, seed=0)
            >>> low, mean, high = full.hyperparameter_interval("delta")
            >>> bool(low < mean < high)
            True
            >>> full.hyperparameter_interval("rho")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: unknown hyperparameter 'rho'; ...
            >>> empirical = model.fit(method="empirical", n_draws=50, seed=0)
            >>> empirical.hyperparameter_interval("lambda")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: the empirical-Bayes fit conditions on the ...
        """
        if name not in self.hyper_names:
            raise SpecificationError(
                f"unknown hyperparameter {name!r}; expected one of {self.hyper_names}."
            )
        if not self.hyper_draws.shape[0]:
            raise SpecificationError(
                "the empirical-Bayes fit conditions on the hyperparameter "
                "mode and holds no hyperparameter draws; refit with "
                "method='full' for a hyperparameter posterior."
            )
        draws = self.hyper_draws[:, self.hyper_names.index(name)]
        return np.array(
            [float(np.quantile(draws, 0.16)), float(draws.mean()), float(np.quantile(draws, 0.84))]
        )

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Returns:
            One row per hyperparameter with its mode and, under the full
            method, its posterior mean and 68% interval; the metadata
            carries the method, the draw count, the dimensions and the
            log marginal likelihood at the mode; the notes state the
            hyperpriors and the exploration, the acceptance rate (full
            method), what the reported evidence conditions on, and the
            stability share.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = 5.0 + rng.standard_normal((150, 2))
            >>> empirical = HierarchicalBVAR(y, order=1).fit(
            ...     method="empirical", n_draws=50, seed=0
            ... )
            >>> table = empirical._summary_table()
            >>> table.title, table.columns[:2], table.metadata[1]
            ('Hierarchical BVAR(1) Results', ('hyperparameter', 'mode'), ('Method', 'empirical'))
            >>> table.rows[0][2:], len(table.notes)
            (('-', '-'), 3)
            >>> full = HierarchicalBVAR(y, order=1).fit(
            ...     method="full", n_draws=100, n_burn=100, seed=0
            ... )
            >>> len(full._summary_table().notes), full._summary_table().notes[1][:26]
            (4, 'Metropolis acceptance rate')
        """
        rows = []
        for i, name in enumerate(self.hyper_names):
            if self.hyper_draws.shape[0]:
                low, mid, high = self.hyperparameter_interval(name)
                rows.append(
                    (name, f"{self.hyper_mode[i]:.4f}", f"{mid:.4f}", f"[{low:.4f}, {high:.4f}]")
                )
            else:
                rows.append((name, f"{self.hyper_mode[i]:.4f}", "-", "-"))
        notes = [
            "Hyperparameters follow Giannone-Lenza-Primiceri: Gamma "
            "hyperpriors (lambda: mode 0.2, sd 0.4; mu, delta: mode 1, sd "
            "1), posterior explored "
            + (
                "by adaptive random-walk Metropolis over log "
                "hyperparameters; the (B, Sigma) draws marginalize over "
                "them."
                if self.method == "full"
                else "to its mode only (empirical Bayes); the (B, Sigma) "
                "draws condition on the mode."
            ),
            "log_marginal_likelihood is evaluated at the hyperparameter "
            "mode, conditional on it; the full model evidence would "
            "integrate over the hyperprior and is not reported.",
            f"Posterior probability of stability: {self.stable_share:.2f}.",
        ]
        if self.method == "full":
            notes.insert(
                1,
                f"Metropolis acceptance rate {self.acceptance_rate:.2f}; "
                "rates far outside 0.1-0.6 warrant a longer burn-in.",
            )
        return SummaryTable(
            title=f"Hierarchical BVAR({self.order}) Results",
            metadata=(
                ("Model", f"Hierarchical BVAR({self.order})"),
                ("Method", self.method),
                ("Draws", f"{self.n_kept}"),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.nobs}"),
                ("Trend", self.trend),
                ("log ML at mode", f"{self.log_marginal_likelihood:.3f}"),
            ),
            columns=("hyperparameter", "mode", "posterior mean", "68% interval"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class HierarchicalBVAR(_BayesianVectorAutoRegressionModel[HierarchicalBVARResult]):
    r"""Giannone-Lenza-Primiceri hierarchical shrinkage over the conjugate BVAR.

    The conjugate Normal-inverse-Wishart prior with Minnesota moments is
    indexed by the overall tightness :math:`\lambda` and, when the
    dummy-observation priors are included, by the sum-of-coefficients
    looseness :math:`\mu` and the dummy-initial-observation looseness
    :math:`\delta` (larger is looser, as in the paper). Each carries a
    Gamma hyperprior -- mode 0.2 and standard deviation 0.4 for
    :math:`\lambda`, mode 1 and standard deviation 1 for :math:`\mu` and
    :math:`\delta` -- and the conjugate model's closed-form marginal
    likelihood :math:`p(y \mid \theta)` is the likelihood of the
    hyperparameter posterior. ``fit`` finds its mode by direct
    optimization in log space and, under the full method, explores it by
    adaptive random-walk Metropolis, drawing one ``(B, Sigma)`` from the
    exact conditional posterior at every kept state. The lag decay, the
    deterministic looseness and the persistence prior mean are held
    fixed, as in the paper.

    Attributes:
        _endog: The validated ``(nobs_total, k)`` panel.
        _order: The autoregressive order.
        _trend: The deterministic specification.
        _names: One label per variable.
        _prior: The prior at the hyperprior modes, the search's start.
        _soc: Whether the sum-of-coefficients dummies are active.
        _dio: Whether the dummy-initial-observation row is active.
        _decay: The fixed Minnesota lag decay.
        _exogenous: The fixed looseness of the deterministic block.
        _persistence: The fixed prior mean of each own first lag.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order, at least 1.
        sum_of_coefficients: Include the sum-of-coefficients dummies, with
            their looseness ``mu`` as a hyperparameter.
        dummy_initial_observation: Include Sims' co-persistence dummy, with
            its looseness ``delta`` as a hyperparameter.
        decay: Minnesota lag decay, held fixed as in the paper.
        exogenous: Looseness of the deterministic block, held fixed.
        persistence: Prior mean of each variable's own first lag, one
            value for all or one per variable.
        trend: Deterministic terms, ``"n"``, ``"c"`` or ``"ct"``.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Raises:
        SpecificationError: If ``trend`` is not one of the three names,
            ``names`` does not have one entry per column, or the prior's
            own validation of ``decay``, ``exogenous`` or ``persistence``
            fails.
        DimensionError: If the panel is not a time-down-the-rows matrix or
            too short for the order.

    See Also:
        * :class:`HierarchicalBVARResult` -- the record ``fit`` returns.
        * :class:`~cultivars.multivariate.large_dim.bayesian.BVAR` -- the
          conjugate model at a fixed tightness.
        * :class:`~cultivars.bayes.priors.NormalInverseWishartPrior` -- the
          prior family the hierarchy ranges over.

    References:
        Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior
        selection for vector autoregressions. *Review of Economics and
        Statistics*, 97(2), 436-451.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((150, 2))
        >>> for t in range(1, 150):
        ...     y[t] = 0.6 * y[t - 1] + rng.standard_normal(2)
        >>> res = HierarchicalBVAR(y, order=1).fit(
        ...     method="empirical", n_draws=100, seed=0
        ... )
        >>> res.hyper_names
        ('lambda', 'mu', 'delta')
        >>> res.beta_draws.shape
        (100, 3, 2)
        >>> lambda_only = HierarchicalBVAR(
        ...     y, order=1, sum_of_coefficients=False, dummy_initial_observation=False
        ... ).fit(method="empirical", n_draws=50, seed=0)
        >>> lambda_only.hyper_names, lambda_only.prior_label, lambda_only.n_dummy
        (('lambda',), 'glp(lambda)', 0)
    """

    __slots__ = (
        "_decay",
        "_dio",
        "_exogenous",
        "_persistence",
        "_soc",
    )

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        sum_of_coefficients: bool = True,
        dummy_initial_observation: bool = True,
        decay: float = 1.0,
        exogenous: float = 100.0,
        persistence: float | Sequence[float] = 1.0,
        trend: Trend = "c",
        names: Sequence[str] | None = None,
    ) -> None:
        """Validate the specification; the prior itself is hyperparameterized.

        The fixed prior settings are stored first, then the base model is
        built at the hyperprior modes ``(0.2, 1, 1)`` so the panel and
        the prior family are validated once.

        Args:
            endog: The observed panel.
            order: Autoregressive order.
            sum_of_coefficients: Include the sum-of-coefficients dummies.
            dummy_initial_observation: Include the co-persistence dummy.
            decay: Minnesota lag decay.
            exogenous: Looseness of the deterministic block.
            persistence: Prior mean of each own first lag.
            trend: Deterministic terms.
            names: One label per variable, or ``None`` for defaults.

        Example:
            >>> import numpy as np
            >>> panel = np.random.default_rng(0).standard_normal((100, 2))
            >>> model = HierarchicalBVAR(panel, order=1)
            >>> model._soc, model._dio, model._decay, model._exogenous, model.prior._label()
            (True, True, 1.0, 100.0, 'niw(l1=0.2, l3=1, l4=100) + soc(1) + dio(1)')
        """
        self._soc = bool(sum_of_coefficients)
        self._dio = bool(dummy_initial_observation)
        self._decay = float(decay)
        self._exogenous = float(exogenous)
        self._persistence = persistence
        super().__init__(
            endog,
            order=order,
            trend=trend,
            names=names,
            prior=self._prior_at(np.array([0.2, 1.0, 1.0])),
        )

    def _prior_at(self, theta: npt.NDArray[np.float64]) -> _Prior:
        """The prior a hyperparameter vector names.

        The vector is always three long -- ``(lambda, mu, delta)`` in the
        paper's parameterization -- with inactive dummies simply not
        composed in. The dummy priors take a tightness, so ``mu`` and
        ``delta`` enter as their reciprocals.

        Args:
            theta: ``(lambda, mu, delta)``, all positive.

        Returns:
            The composed prior.

        Example:
            >>> import numpy as np
            >>> panel = np.random.default_rng(0).standard_normal((100, 2))
            >>> model = HierarchicalBVAR(panel, order=1)
            >>> model._prior_at(np.array([0.5, 2.0, 4.0]))._label()
            'niw(l1=0.5, l3=1, l4=100) + soc(0.5) + dio(0.25)'
        """
        prior: _Prior = NormalInverseWishartPrior(
            tightness=float(theta[0]),
            decay=self._decay,
            exogenous=self._exogenous,
            persistence=self._persistence,
        )
        if self._soc:
            prior = prior + SumOfCoefficientsPrior(tightness=1.0 / float(theta[1]))
        if self._dio:
            prior = prior + DummyInitialObservationPrior(tightness=1.0 / float(theta[2]))
        return prior

    def fit(
        self,
        *,
        method: str = "full",
        n_draws: int = 1000,
        n_burn: int = 500,
        seed: int | np.random.Generator | None = None,
    ) -> HierarchicalBVARResult:
        """Estimate the hyperparameter posterior, then the VAR under it.

        Builds the active hyperparameter set and its Gamma hyperpriors
        from the modes and standard deviations the paper uses, finds the
        posterior mode, and either conditions on it (``"empirical"``) or
        runs the Metropolis chain from it (``"full"``), drawing
        ``(B, Sigma)`` at every kept state. The same ``seed`` reproduces
        the chain exactly.

        Args:
            method: ``"full"`` marginalizes the ``(B, Sigma)`` draws over
                the hyperparameters by Metropolis; ``"empirical"``
                conditions on the hyperparameter posterior mode.
            n_draws: Kept ``(B, Sigma)`` draws, at least 1.
            n_burn: Burn-in Metropolis iterations (full method), at
                least 0.
            seed: Seed or generator, for reproducibility.

        Returns:
            The fitted :class:`HierarchicalBVARResult`.

        Raises:
            SpecificationError: If the method or counts are malformed.
            NumericalError: If the mode search fails.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = 5.0 + rng.standard_normal((150, 2))
            >>> model = HierarchicalBVAR(y, order=1)
            >>> full = model.fit(method="full", n_draws=100, n_burn=100, seed=0)
            >>> again = model.fit(method="full", n_draws=100, n_burn=100, seed=0)
            >>> bool(np.array_equal(full.hyper_draws, again.hyper_draws))
            True
            >>> model.fit(method="map")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: method must be 'full' or 'empirical'; ...
            >>> model.fit(n_draws=0)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: n_draws must be positive and n_burn ...
        """
        names = ["lambda"]
        modes = [0.2]
        sds = [0.4]
        if self._soc:
            names.append("mu")
            modes.append(1.0)
            sds.append(1.0)
        if self._dio:
            names.append("delta")
            modes.append(1.0)
            sds.append(1.0)
        shapes_scales = [_gamma_from_mode(m, s) for m, s in zip(modes, sds, strict=True)]
        active = tuple(names)

        def factory(theta: npt.NDArray[np.float64]) -> _Prior:
            full = np.ones(3)
            full[0] = theta[0]
            position = 1
            if self._soc:
                full[1] = theta[position]
                position += 1
            if self._dio:
                full[2] = theta[position]
            return self._prior_at(full)

        fit = self._fit_hierarchical(
            factory=factory,
            hyper_names=active,
            hyper_shapes=np.array([pair[0] for pair in shapes_scales]),
            hyper_scales=np.array([pair[1] for pair in shapes_scales]),
            start=np.asarray(modes, dtype=np.float64),
            method=method,
            n_draws=n_draws,
            n_burn=n_burn,
            seed=seed,
        )
        return HierarchicalBVARResult(
            endog=self.endog,
            names=self.names,
            order=self.order,
            trend=self.trend,
            prior_label=f"glp({', '.join(active)})",
            coefficients=fit.coefficient_stack,
            deterministic=fit.deterministic,
            beta_mean=fit.beta_mean,
            sigma_u=fit.sigma_u,
            beta_draws=fit.beta_draws,
            sigma_draws=fit.sigma_draws,
            log_marginal_likelihood=fit.log_marginal_likelihood,
            posterior_df=fit.posterior_df,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            nobs=fit.nobs,
            n_dummy=fit.n_dummy,
            hyper_names=fit.hyper_names,
            hyper_mode=fit.hyper_mode,
            hyper_draws=fit.hyper_draws,
            acceptance_rate=fit.acceptance,
            method=fit.method,
        )
