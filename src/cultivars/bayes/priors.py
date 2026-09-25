# filepath: /src/cultivars/bayes/priors.py
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
r"""Priors for the Bayesian vector autoregressions: moments, dummies, and hierarchies.

A prior here is a frozen record that answers three questions about a VAR
whose coefficients :math:`B` are ``(width, k)`` in design-column order:
what the coefficients are centred on, :meth:`coefficient_mean`; how far
the data may pull each one away, :meth:`coefficient_variance`; and which
pseudo-observations, if any, are stacked under the sample,
:meth:`dummy_observations`. Every model that takes ``prior=`` reads the
prior through those three hooks and a :class:`~cultivars._internals._PriorContext`
describing the sample, so a prior knows nothing about the estimator and an
estimator knows nothing about the prior beyond its moments and its rows.
The Minnesota family lives in that interface directly. Litterman's prior
sets the mean at independent random walks and the variance at

.. math::

   \operatorname{Var}(A_{l,ij}) = \Bigl(\frac{\lambda_1}{l^{\lambda_3}}\Bigr)^{2}
   \times \begin{cases} 1 & i = j \\ \lambda_2^2\, s_i^2 / s_j^2 & i \ne j, \end{cases}

and the sum-of-coefficients and dummy-initial-observation priors add
rows rather than moments, each a Theil-Goldberger pseudo-sample that
says the system is near a unit root or near its pre-sample mean. Priors
compose with ``+``: means and variances combine by precision weighting,
dummy blocks stack, and :class:`NoPrior` is the identity, so
``NormalInverseWishartPrior() + SumOfCoefficientsPrior() +
DummyInitialObservationPrior()`` is the prior of Sims and Zha (1998) and
Giannone, Lenza and Primiceri (2015) stated as a sum.

The adaptive priors do not have a variance to state. Under the horseshoe,
spike-and-slab, Dirichlet-Laplace, and Normal-Gamma hierarchies each
coefficient's variance is a product of latent scales with their own full
conditionals, so those classes keep the moments interface, reporting the
variance at their initial scales, and add three hooks the Gibbs sampler
drives: initialize the scales, redraw them given the current
standardized coefficients, and read the variance they imply. Shrinkage
acts on the coefficient divided by the Minnesota unit ratio
:math:`s_i / s_j`, so one latent scale means the same thing whatever the
variables' units, and only the endogenous lag block is shrunk, with the
deterministic and exogenous columns held at a fixed loose variance,
exactly where the Minnesota prior leaves them.

Two commitments shape the surface. First, the pairing with the residual
covariance is stated in the prior's name and nowhere else.
:class:`NormalInverseWishartPrior` is the conjugate pairing, with the
cross-equation weight pinned to one because the Kronecker structure
demands it and an exact posterior and marginal likelihood in return;
:class:`IndependentNormalWishartPrior` keeps the weight and pays with a
Gibbs sampler and a simulated Chib evidence; :class:`MinnesotaPrior` is
the same moments for the per-equation point path. The conjugate model
refuses a prior whose variance does not factor rather than quietly
pinning the weight itself. Second, hyperparameters are checked when they
are used, not at construction: a prior is a plain frozen dataclass, and
the ``SpecificationError`` for a negative tightness surfaces at the
model's fit, where the context that makes the check meaningful exists.

Layout. The nine classes defined here are the public priors:
:class:`MinnesotaPrior`, :class:`NormalInverseWishartPrior`,
:class:`SumOfCoefficientsPrior`, :class:`DummyInitialObservationPrior`,
and :class:`IndependentNormalWishartPrior` state moments and rows;
:class:`HorseshoePrior`, :class:`SpikeAndSlabPrior`,
:class:`DirichletLaplacePrior`, and :class:`NormalGammaPrior` are the
adaptive hierarchies. Three more are re-exported from ``_internals``
under public names: :class:`NoPrior`, the identity of composition, and
the two volatility-law priors :class:`VolatilityPrior` and
:class:`RandomWalkVolatilityPrior`, which the stochastic-volatility
models read and which do not enter coefficient composition at all. The
abstract base :class:`~cultivars._internals._Prior`, the adaptive base
:class:`~cultivars._internals._AdaptivePrior`, the composite built by
``+``, and the :class:`~cultivars._internals._PriorContext` record live in
``_internals``; the generalized-inverse-Gaussian sampler the
Dirichlet-Laplace and Normal-Gamma conditionals need lives in ``_core``.
The consumers are :class:`~cultivars.multivariate.large_dim.bayesian.BVAR`
and :class:`~cultivars.multivariate.large_dim.student.StudentBVAR` for
the conjugate path,
:class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR` for the
independent and adaptive paths, and
:class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
for the point-estimate path.

References:
    Litterman, R. B. (1986). Forecasting with Bayesian vector
    autoregressions: Five years of experience. *Journal of Business &
    Economic Statistics*, 4(1), 25-38.

    Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and
    conditional projection using realistic prior distributions.
    *Econometric Reviews*, 3(1), 1-100.

    Kadiyala, K. R., & Karlsson, S. (1997). Numerical methods for
    estimation and inference in Bayesian VAR-models. *Journal of Applied
    Econometrics*, 12(2), 99-132.

    Sims, C. A., & Zha, T. (1998). Bayesian methods for dynamic
    multivariate models. *International Economic Review*, 39(4),
    949-968.

    Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection
    for vector autoregressions. *Review of Economics and Statistics*,
    97(2), 436-451.

    Koop, G., & Korobilis, D. (2010). Bayesian multivariate time series
    methods for empirical macroeconomics. *Foundations and Trends in
    Econometrics*, 3(4), 267-358.

    Polson, N. G., & Scott, J. G. (2010). Shrink globally, act locally:
    Sparse Bayesian regularization and prediction. In *Bayesian
    Statistics 9* (pp. 501-538). Oxford University Press.

Example:
    Compose the Sims-Zha prior and fit the conjugate model; the label
    records the sum, and the three dummy rows are folded into the
    posterior exactly:

    >>> import numpy as np
    >>> from cultivars.multivariate.large_dim.bayesian import BVAR
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((120, 2))
    >>> for t in range(1, 120):
    ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
    >>> prior = (
    ...     NormalInverseWishartPrior(tightness=0.1)
    ...     + SumOfCoefficientsPrior(tightness=1.0)
    ...     + DummyInitialObservationPrior(tightness=1.0)
    ... )
    >>> res = BVAR(y, order=1, prior=prior).fit(n_draws=200, seed=0)
    >>> res.prior_label
    'niw(l1=0.1, l3=1, l4=100) + soc(1) + dio(1)'
    >>> res.n_dummy, res.marginal_likelihood().method
    (3, 'analytic')

    An adaptive prior goes to the Gibbs model alone, and reports the
    shrinkage it learned:

    >>> from cultivars.multivariate.large_dim.gibbs import GibbsBVAR
    >>> res = GibbsBVAR(y, order=1, prior=HorseshoePrior()).fit(
    ...     n_draws=600, n_burn=200, seed=0
    ... )
    >>> res.prior_label, res.shrinkage_scales().shape
    ('horseshoe', (1, 2, 2))
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import _draw_generalized_inverse_gaussian
from .._internals import _AdaptivePrior, _Prior, _PriorContext
from .._internals import _NoPrior as NoPrior
from .._internals import _RandomWalkVolatilityPrior as RandomWalkVolatilityPrior
from .._internals import _VolatilityPrior as VolatilityPrior
from ..exceptions import DimensionError, SpecificationError

__all__ = [
    "DirichletLaplacePrior",
    "DummyInitialObservationPrior",
    "HorseshoePrior",
    "IndependentNormalWishartPrior",
    "MinnesotaPrior",
    "NoPrior",
    "NormalGammaPrior",
    "NormalInverseWishartPrior",
    "RandomWalkVolatilityPrior",
    "SpikeAndSlabPrior",
    "SumOfCoefficientsPrior",
    "VolatilityPrior",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class MinnesotaPrior(_Prior):
    r"""Litterman's prior: shrink toward independent random walks.

    The prior of Litterman (1986) and Doan, Litterman, and Sims (1984) states
    that, absent evidence, each variable follows its own random walk: the
    coefficient on a variable's own first lag is centred at
    ``persistence`` (one, for levels) and every other coefficient at zero,
    with variances that say how far the data may pull each one away. For
    equation :math:`i` and the coefficient on variable :math:`j` at lag
    :math:`l`, with :math:`s_i` the residual scale of a univariate
    autoregression of variable :math:`i`,

    .. math::

       \operatorname{Var}(A_{l,ij}) =
       \begin{cases}
         \left(\dfrac{\lambda_1}{l^{\lambda_3}}\right)^{2} & i = j,\\[1.2em]
         \left(\dfrac{\lambda_1 \lambda_2}{l^{\lambda_3}}\,
               \dfrac{s_i}{s_j}\right)^{2} & i \ne j,
       \end{cases}
       \qquad
       \operatorname{Var}(c_i) = (\lambda_1 \lambda_4\, s_i)^{2},

    the last for the intercept, any trend, and any exogenous regressor. Five
    hyperparameters, and each answers a different question. ``tightness``
    :math:`\lambda_1` is how much the prior is believed at all; it scales
    every variance and is the one people tune. ``cross_equation``
    :math:`\lambda_2` says how much less a variable's dependence on *other*
    variables is believed than its dependence on itself, which is the
    prior's central economic claim -- most of what a series does is
    explained by its own past -- and is the hyperparameter a
    dummy-observation implementation cannot express. ``decay``
    :math:`\lambda_3` tightens longer lags toward zero, encoding that
    distant history matters less. ``exogenous`` :math:`\lambda_4` loosens
    the deterministic and exogenous block and defaults large: those
    coefficients carry levels and slopes that nobody means to shrink, and a
    tight default there is a silent and expensive mistake.
    ``sum_of_coefficients`` :math:`\lambda_5` is the only one that cannot
    be a variance: it restricts a *sum* of coefficients, so it enters as
    artificial rows through :meth:`dummy_observations`. The scale ratio
    :math:`s_i / s_j` is what makes variables measured in different units
    comparable, so that shrinkage is a statement about dynamics rather than
    about whether a series is quoted in percent or in levels.

    With :math:`\lambda_2 \ne 1` the prior variance does not factor as
    :math:`\Sigma \otimes \Omega`, so there is no Normal-inverse-Wishart
    closed form, and estimation runs equation by equation conditional on
    the scales. That is Litterman's original procedure and it is correct;
    it is also why this prior and
    :class:`NormalInverseWishartPrior` are different estimation paths
    rather than the same object with different numbers. This prior is
    accepted by :class:`~cultivars.multivariate.reduced_form.vector_autoregression.VAR`
    for a shrunk point estimate and by
    :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR` for a sampled
    posterior; the conjugate
    :class:`~cultivars.multivariate.large_dim.bayesian.BVAR` accepts it only
    with ``cross_equation=1``, where it coincides with the conjugate form.

    Attributes:
        tightness: Overall confidence, :math:`\lambda_1`. Values near 0.1
            to 0.3 are usual for macroeconomic data; large recovers least
            squares.
        cross_equation: How much harder cross-variable coefficients shrink,
            :math:`\lambda_2`. One treats them like own lags; smaller is
            tighter.
        decay: Lag decay, :math:`\lambda_3`.
        exogenous: Looseness of the intercept and exogenous block,
            :math:`\lambda_4`. Large is flat.
        sum_of_coefficients: Confidence that the variables sit at their
            pre-sample means forever, :math:`\lambda_5`. ``None`` omits the
            restriction; larger imposes it harder.
        persistence: Prior mean of each variable's own first lag. One is the
            random-walk prior for levels; zero suits differenced data.
            Getting this wrong shrinks toward the wrong place, so tightening
            makes the estimate worse rather than better.

    Raises:
        SpecificationError: When a hyperparameter is used, not at
            construction: if ``tightness``, ``cross_equation``, or
            ``exogenous`` is not positive, if ``decay`` is negative, or if
            ``sum_of_coefficients`` is given and not positive. The dataclass
            is frozen and does no validation of its own, so a bad value
            surfaces at the first :meth:`coefficient_mean` or
            :meth:`coefficient_variance` call, which is the model's fit.
        DimensionError: If ``persistence`` is a sequence whose length is not
            the number of variables, at the same point.

    Note:
        Priors compose with ``+``: ``MinnesotaPrior() +
        SumOfCoefficientsPrior()`` stacks the moments of the first with the
        dummy rows of the second, and the label reads accordingly. The
        ``sum_of_coefficients`` field is the same restriction carried inside
        this prior for callers who prefer one object; a composition with
        :class:`SumOfCoefficientsPrior` is the same rows again, so use one
        or the other.

    See Also:
        * :class:`NormalInverseWishartPrior` -- the conjugate form with the
          cross-equation weight pinned at one, for the exact posterior.
        * :class:`SumOfCoefficientsPrior` and
          :class:`DummyInitialObservationPrior` -- the dummy-observation
          components, for stacking with ``+``.
        * :func:`~cultivars._core.minnesota_scales` -- the univariate
          residual scales :math:`s_i`, computed once per sample by the
          estimator and passed in through the context.

    References:
        Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and
        conditional projection using realistic prior distributions.
        *Econometric Reviews*, 3(1), 1-100.

        Litterman, R. B. (1986). Forecasting with Bayesian vector
        autoregressions: Five years of experience. *Journal of Business &
        Economic Statistics*, 4(1), 25-38.

        Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection
        for vector autoregressions. *Review of Economics and Statistics*,
        97(2), 436-451.

    Example:
        A tight Minnesota prior on a bivariate VAR(2) pulls the point
        estimate toward independent random walks; the second-lag block,
        which the prior shrinks hardest, nearly vanishes:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((120, 2))
        >>> for t in range(1, 120):
        ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
        >>> ols = VAR(y, order=2).fit()
        >>> shrunk = VAR(y, order=2, prior=MinnesotaPrior(tightness=0.05)).fit()
        >>> round(float(np.abs(ols.coefficients[1]).sum()), 3)
        0.176
        >>> round(float(np.abs(shrunk.coefficients[1]).sum()), 3)
        0.038

        The prior's own moments, on a context of two variables at two lags
        with residual scales one and four: own lags are dimensionless, cross
        lags carry the scale ratio, and the leading intercept row is loose:

        >>> from cultivars._internals import _PriorContext
        >>> ctx = _PriorContext(
        ...     k_endog=2, order=2, scales=np.array([1.0, 4.0]),
        ...     presample_mean=np.array([10.0, 20.0]),
        ... )
        >>> MinnesotaPrior().coefficient_mean(ctx)[1:3]
        array([[1., 0.],
               [0., 1.]])
        >>> MinnesotaPrior().coefficient_variance(ctx).round(4)
        array([[4.0e+02, 6.4e+03],
               [4.0e-02, 1.6e-01],
               [6.0e-04, 4.0e-02],
               [1.0e-02, 4.0e-02],
               [2.0e-04, 1.0e-02]])
    """

    tightness: float = 0.2
    r"""Overall confidence :math:`\lambda_1`; scales every prior variance.

    The standard deviation of a variable's own first-lag coefficient, and
    the hyperparameter to tune. Values near 0.1 to 0.3 are usual for
    macroeconomic data; a large value makes every variance large and the
    estimate approaches least squares. Must be positive.
    """
    cross_equation: float = 0.5
    r"""Relative tightness of cross-variable coefficients :math:`\lambda_2`.

    A cross lag's standard deviation is the own lag's times this, so one
    shrinks cross and own lags alike and smaller values encode more of the
    belief that a series is explained by its own past. Any value other than
    one breaks the Kronecker factorization that the conjugate
    :class:`~cultivars.multivariate.large_dim.bayesian.BVAR` requires, which
    refuses the prior at fit; the point-estimate and Gibbs paths accept it.
    Must be positive.
    """
    decay: float = 1.0
    r"""Lag decay :math:`\lambda_3`; lag :math:`l` is shrunk by :math:`l^{-\lambda_3}`.

    One is harmonic decay, Litterman's choice; zero treats every lag alike;
    two shrinks distant lags hard. Must be non-negative.
    """
    exogenous: float = 100.0
    r"""Looseness of the deterministic and exogenous block :math:`\lambda_4`.

    The intercept, any trend, and any exogenous regressor get standard
    deviation :math:`\lambda_1 \lambda_4 s_i`, so the default makes them
    four orders of magnitude looser than an own lag, that is, effectively
    unshrunk. Those coefficients carry levels and slopes, and shrinking
    them toward zero is rarely what anyone means. Must be positive.
    """
    sum_of_coefficients: float | None = None
    r"""Confidence in the sum-of-coefficients restriction :math:`\lambda_5`, or ``None``.

    When given, :meth:`dummy_observations` contributes one artificial row
    per variable stating that a series at its pre-sample mean stays there,
    scaled by this value; larger imposes it harder. ``None`` contributes no
    rows. The same restriction is available as a separate
    :class:`SumOfCoefficientsPrior` for composition with ``+``; do not use
    both. Must be positive when given.
    """
    persistence: float | Sequence[float] = 1.0
    """Prior mean of each variable's own first lag: a scalar, or one value per variable.

    One is the random-walk prior for series in levels; zero suits
    differenced or otherwise stationary data; a value between is a stated
    belief about mean reversion. A sequence gives each variable its own
    centre and must have one entry per variable. Everything else in the
    coefficient matrix is centred at zero regardless.
    """

    def _persistence(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Broadcast the prior mean of the own first lags to one value per variable.

        A scalar ``persistence`` is repeated ``k_endog`` times; a sequence is
        taken as given after its length is checked. Called by
        :meth:`coefficient_mean` and :meth:`dummy_observations`, the two
        places the centre enters, so the two cannot disagree.

        Args:
            context: The sample description; only ``k_endog`` is read.

        Returns:
            ``(k,)`` prior means, one per variable in column order.

        Raises:
            DimensionError: If ``persistence`` is a sequence whose length
                is not ``k_endog``.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> MinnesotaPrior(persistence=0.9)._persistence(ctx)
            array([0.9, 0.9])
            >>> MinnesotaPrior(persistence=(1.0, 0.0))._persistence(ctx)
            array([1., 0.])
        """
        if isinstance(self.persistence, int | float):
            return np.full(context.k_endog, float(self.persistence), dtype=np.float64)
        values = np.asarray(self.persistence, dtype=np.float64).ravel()
        if values.shape != (context.k_endog,):
            raise DimensionError(
                f"persistence must be a scalar or have {context.k_endog} entries; "
                f"got shape {values.shape}."
            )
        return values

    def _check(self) -> None:
        """Reject hyperparameters that do not describe a prior.

        The dataclass is frozen and validates nothing at construction, so
        every public method calls this first; a prior built with a bad
        value fails at the fit that first reads it, with the field named.

        Raises:
            SpecificationError: If ``tightness``, ``cross_equation``, or
                ``exogenous`` is not positive, if ``decay`` is negative, or
                if ``sum_of_coefficients`` is given and not positive.

        Example:
            >>> MinnesotaPrior(decay=-1.0)._check()
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: decay must be non-negative; got -1.0.
        """
        for name, value in (
            ("tightness", self.tightness),
            ("cross_equation", self.cross_equation),
            ("exogenous", self.exogenous),
        ):
            if value <= 0.0:
                raise SpecificationError(f"{name} must be positive; got {value}.")
        if self.decay < 0.0:
            raise SpecificationError(f"decay must be non-negative; got {self.decay}.")
        if self.sum_of_coefficients is not None and self.sum_of_coefficients <= 0.0:
            raise SpecificationError(
                f"sum_of_coefficients must be positive when given; got {self.sum_of_coefficients}."
            )

    def coefficient_mean(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Each variable's own first lag at ``persistence``, everything else zero.

        The random-walk centre: row ``lag_offset + j`` of column ``j`` holds
        that variable's ``persistence``, and every other entry -- the
        deterministic block, the cross lags, the longer own lags, the
        exogenous block -- is zero.

        Args:
            context: The sample description; ``k_endog``, ``width``, and
                ``lag_offset`` are read.

        Returns:
            A ``(width, k)`` array in design-column order.

        Raises:
            SpecificationError: If a hyperparameter is out of range.
            DimensionError: If ``persistence`` has the wrong length.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> MinnesotaPrior(persistence=(1.0, 0.0)).coefficient_mean(ctx)
            array([[0., 0.],
                   [1., 0.],
                   [0., 0.]])
        """
        self._check()
        means = self._persistence(context)
        out = np.zeros((context.width, context.k_endog), dtype=np.float64)
        offset = context.lag_offset
        for index in range(context.k_endog):
            out[offset + index, index] = means[index]
        return out

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        r"""Litterman's variances, tighter for cross terms and for longer lags.

        Three kinds of column, three rules. A variable's own lag gets
        :math:`(\lambda_1 / l^{\lambda_3})^2`, the only rule with no scale
        in it, because a coefficient on a variable's own past is
        dimensionless. A cross lag gets the same thing multiplied by
        :math:`\lambda_2` and by :math:`s_i / s_j`, the ratio that makes a
        coefficient linking two variables measured in different units shrink
        by the same amount it would if they were measured in the same ones.
        Deterministic terms and exogenous regressors get
        :math:`(\lambda_1 \lambda_4 s_i)^2`, which with the default
        ``exogenous`` is four orders of magnitude looser than an own lag and
        is meant to be.

        The deterministic block is a loop rather than a single row because
        its width is a property of the family, not a constant: none for a
        trendless specification, one for a constant, two for a constant and
        trend, and one per unit for a fixed-effects panel. Writing it as
        ``out[0]`` was silently one column short for every ``"ct"`` model
        and off by ``N - 1`` for every panel -- a prior whose columns do not
        line up with the design shrinks the wrong coefficients and reports a
        number rather than raising.

        Args:
            context: The sample description; ``k_endog``, ``order``,
                ``scales``, ``width``, ``lag_offset``, ``n_deterministic``,
                and ``k_exog`` are read.

        Returns:
            A ``(width, k)`` array of variances in design-column order,
            every entry finite and positive.

        Raises:
            SpecificationError: If a hyperparameter is out of range.

        Example:
            Two variables, one lag, residual scales one and four; the
            cross-lag variance in the first equation is scaled down by
            :math:`(1/4)^2` and in the second scaled up by :math:`4^2`:

            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.array([1.0, 4.0]),
            ...     presample_mean=np.zeros(2),
            ... )
            >>> prior = MinnesotaPrior(tightness=0.2, cross_equation=0.5)
            >>> prior.coefficient_variance(ctx).round(4)
            array([[4.0e+02, 6.4e+03],
                   [4.0e-02, 1.6e-01],
                   [6.0e-04, 4.0e-02]])
        """
        self._check()
        size, order = context.k_endog, context.order
        scales = context.scales
        out = np.empty((context.width, size), dtype=np.float64)
        offset = context.lag_offset
        loose = (self.tightness * self.exogenous * scales) ** 2
        for term in range(context.n_deterministic):
            out[term] = loose
        for lag in range(1, order + 1):
            damped = float(lag) ** self.decay
            for source in range(size):
                column = offset + (lag - 1) * size + source
                own = (self.tightness / damped) ** 2
                cross = (
                    self.tightness * self.cross_equation * scales / (damped * scales[source])
                ) ** 2
                out[column] = np.where(np.arange(size) == source, own, cross)
        for extra in range(context.k_exog):
            out[offset + size * order + extra] = loose
        return out

    def dummy_observations(
        self, context: _PriorContext
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""The sum-of-coefficients rows, when that restriction is asked for.

        One artificial observation per variable, saying that a series
        sitting at its pre-sample average forever should stay there. For
        variable :math:`j` with pre-sample mean :math:`\bar y_j` and
        centre :math:`\delta_j` (its ``persistence``), the row has target
        :math:`\lambda_5 \delta_j \bar y_j` in column :math:`j` and the same
        value in that variable's column at every lag of the design, with
        zeros in the deterministic and exogenous columns. Regressing target
        on design then says the lag coefficients of variable :math:`j` sum
        to :math:`\delta_j`, with weight :math:`\lambda_5`. That is a
        statement about a *sum* of coefficients, which no diagonal variance
        can make, which is why this hyperparameter alone enters as data
        (Doan, Litterman & Sims 1984; Sims & Zha 1998).

        Args:
            context: The sample description; ``k_endog``, ``order``,
                ``presample_mean``, ``lag_offset``, and ``k_exog`` are read.

        Returns:
            A ``(k, k)`` target block and a ``(k, width)`` design block when
            ``sum_of_coefficients`` is set; the base class's empty
            ``(0, k)`` and ``(0, width)`` blocks otherwise.

        Raises:
            SpecificationError: If a hyperparameter is out of range.
            DimensionError: If ``persistence`` has the wrong length.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=2, scales=np.ones(2),
            ...     presample_mean=np.array([10.0, 20.0]),
            ... )
            >>> target, design = MinnesotaPrior(sum_of_coefficients=1.0).dummy_observations(ctx)
            >>> target
            array([[10.,  0.],
                   [ 0., 20.]])
            >>> design
            array([[ 0., 10.,  0., 10.,  0.],
                   [ 0.,  0., 20.,  0., 20.]])
            >>> MinnesotaPrior().dummy_observations(ctx)[0].shape
            (0, 2)
        """
        self._check()
        if self.sum_of_coefficients is None:
            return super(MinnesotaPrior, self).dummy_observations(context)
        size = context.k_endog
        centre = (
            np.diag(self._persistence(context) * context.presample_mean) * self.sum_of_coefficients
        )
        block = np.hstack(
            [
                np.zeros((size, context.lag_offset), dtype=np.float64),
                np.kron(np.ones((1, context.order)), centre),
                np.zeros((size, context.k_exog), dtype=np.float64),
            ]
        )
        return centre, block

    def _label(self) -> str:
        """Short description for summary tables and comparison rows.

        ``minnesota(l1=..., l2=..., l3=..., l4=...)`` with ``l5`` appended
        when the sum-of-coefficients restriction is set, each value in
        ``%g`` format. It is the ``prior_label`` of every result fitted
        under this prior and the ``source`` of its marginal-likelihood
        record, so two fits differing only in a hyperparameter are told
        apart in a table.

        Returns:
            The label.

        Example:
            >>> MinnesotaPrior()._label()
            'minnesota(l1=0.2, l2=0.5, l3=1, l4=100)'
            >>> MinnesotaPrior(tightness=0.05, sum_of_coefficients=1.0)._label()
            'minnesota(l1=0.05, l2=0.5, l3=1, l4=100, l5=1)'
        """
        tail = "" if self.sum_of_coefficients is None else f", l5={self.sum_of_coefficients:g}"
        return (
            f"minnesota(l1={self.tightness:g}, l2={self.cross_equation:g}, "
            f"l3={self.decay:g}, l4={self.exogenous:g}{tail})"
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class NormalInverseWishartPrior(_Prior):
    r"""The conjugate Minnesota prior: Litterman's variances with the one weight pinned.

    Exactly :class:`MinnesotaPrior` with ``cross_equation`` fixed at one, and
    that is not a simplification but a purchase. With :math:`\lambda_2 = 1`
    the prior variance of every coefficient in equation :math:`i` is the
    equation's residual variance :math:`s_i^2` times a factor that depends
    only on the column,

    .. math::

       \operatorname{Var}(A_{l,ij}) = s_i^2 \cdot
       \frac{\lambda_1^2}{l^{2\lambda_3}\, s_j^2},
       \qquad
       \operatorname{Var}(c_i) = s_i^2 \cdot (\lambda_1 \lambda_4)^2,

    so the coefficient prior factors as
    :math:`\operatorname{vec}(B) \mid \Sigma \sim \mathcal{N}(\operatorname{vec}(B_0),
    \Sigma \otimes \Omega)` with :math:`\Omega` diagonal. That Kronecker form
    is the condition for Normal-inverse-Wishart conjugacy (Kadiyala &
    Karlsson 1997; Banbura, Giannone & Reichlin 2010): the joint posterior
    over coefficients *and* covariance is then exact, its marginal
    likelihood has a closed form, and hierarchical shrinkage in the manner
    of Giannone, Lenza, and Primiceri (2015) becomes optimization of that
    closed form rather than simulation. What is given up is Litterman's
    claim that cross-variable coefficients deserve extra shrinkage; a caller
    who wants that claim back passes a :class:`MinnesotaPrior` to the
    point-estimate or Gibbs path instead, and the conjugate
    :class:`~cultivars.multivariate.large_dim.bayesian.BVAR` refuses it at
    fit by construction rather than approximating.

    The prior is proper, which is what a marginal likelihood requires: every
    variance is finite, so ``exogenous`` is large but not infinite and
    "flat" is not on offer here. Dummy-observation content -- the
    sum-of-coefficients and dummy-initial-observation rows -- is not a field
    of this prior but a composition: ``NormalInverseWishartPrior() +
    SumOfCoefficientsPrior()``.

    Attributes:
        tightness: Overall confidence, :math:`\lambda_1`.
        decay: Lag decay, :math:`\lambda_3`.
        exogenous: Looseness of the intercept and exogenous block,
            :math:`\lambda_4`. Large but finite: the marginal likelihood
            requires a proper prior, so "flat" is not on offer here.
        persistence: Prior mean of each variable's own first lag. One is the
            random-walk prior for levels; zero suits differenced data.

    Raises:
        SpecificationError: When a hyperparameter is used, not at
            construction: if ``tightness`` or ``exogenous`` is not positive
            or ``decay`` is negative, at the first :meth:`coefficient_mean`
            or :meth:`coefficient_variance` call, which is the model's fit.
        DimensionError: If ``persistence`` is a sequence whose length is not
            the number of variables, at the same point.

    See Also:
        * :class:`MinnesotaPrior` -- the general form with the cross-equation
          weight free, for the point-estimate and Gibbs paths.
        * :class:`SumOfCoefficientsPrior` and
          :class:`DummyInitialObservationPrior` -- the dummy-observation
          components this prior composes with.
        * :class:`~cultivars.multivariate.large_dim.bayesian.BVAR` -- the
          conjugate model this prior is the default for.
        * :class:`~cultivars.multivariate.large_dim.hierarchical.HierarchicalBVAR`
          -- treats ``tightness`` and ``decay`` as unknowns and maximizes
          the closed-form marginal likelihood over them.

    References:
        Kadiyala, K. R., & Karlsson, S. (1997). Numerical methods for
        estimation and inference in Bayesian VAR-models. *Journal of Applied
        Econometrics*, 12(2), 99-132.

        Banbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian
        vector auto regressions. *Journal of Applied Econometrics*, 25(1),
        71-92.

        Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection
        for vector autoregressions. *Review of Economics and Statistics*,
        97(2), 436-451.

    Example:
        The default prior of the conjugate BVAR, tightened; the fit is exact
        and reports a closed-form marginal likelihood:

        >>> import numpy as np
        >>> from cultivars.multivariate.large_dim.bayesian import BVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((120, 2))
        >>> for t in range(1, 120):
        ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
        >>> prior = NormalInverseWishartPrior(tightness=0.1)
        >>> res = BVAR(y, order=1, prior=prior).fit(n_draws=100, seed=0)
        >>> res.prior_label, round(res.log_marginal_likelihood, 1)
        ('niw(l1=0.1, l3=1, l4=100)', -357.8)

        Its variances are Litterman's with the cross-equation weight at one,
        so the cross-lag entries carry only the scale ratio:

        >>> from cultivars._internals import _PriorContext
        >>> ctx = _PriorContext(
        ...     k_endog=2, order=1, scales=np.array([1.0, 4.0]),
        ...     presample_mean=np.zeros(2),
        ... )
        >>> NormalInverseWishartPrior().coefficient_variance(ctx).round(4)
        array([[4.0e+02, 6.4e+03],
               [4.0e-02, 6.4e-01],
               [2.5e-03, 4.0e-02]])
        >>> minnesota = MinnesotaPrior(cross_equation=1.0).coefficient_variance(ctx)
        >>> np.array_equal(NormalInverseWishartPrior().coefficient_variance(ctx), minnesota)
        True

        A Minnesota prior with any other cross-equation weight is refused by
        the conjugate model at fit:

        >>> model = BVAR(y, order=1, prior=MinnesotaPrior(cross_equation=0.5))
        >>> model.fit(n_draws=10, seed=0)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: the prior variance does not factor ...
    """

    tightness: float = 0.2
    r"""Overall confidence :math:`\lambda_1`; the standard deviation of an own first lag.

    Scales every prior variance. Values near 0.1 to 0.3 are usual for
    macroeconomic data; a large value approaches least squares, and the
    marginal likelihood is the number to choose it by, either by hand or
    through
    :class:`~cultivars.multivariate.large_dim.hierarchical.HierarchicalBVAR`.
    Must be positive.
    """
    decay: float = 1.0
    r"""Lag decay :math:`\lambda_3`; lag :math:`l` is shrunk by :math:`l^{-\lambda_3}`.

    One is harmonic decay, Litterman's choice; zero treats every lag alike.
    Must be non-negative.
    """
    exogenous: float = 100.0
    r"""Looseness of the deterministic and exogenous block :math:`\lambda_4`.

    The intercept, any trend, and any exogenous regressor get standard
    deviation :math:`\lambda_1 \lambda_4 s_i`, four orders of magnitude
    looser than an own lag at the default. Large but finite by design: a
    marginal likelihood exists only under a proper prior, and this is the
    field an improper "flat" intercept prior would have to go through. Must
    be positive.
    """
    persistence: float | Sequence[float] = 1.0
    """Prior mean of each variable's own first lag: a scalar, or one value per variable.

    One is the random-walk prior for series in levels; zero suits
    differenced or otherwise stationary data. A sequence gives each
    variable its own centre and must have one entry per variable.
    """

    def _minnesota(self) -> MinnesotaPrior:
        """The equivalent :class:`MinnesotaPrior`, cross-equation weight pinned at one.

        Every moment this prior reports is computed by that object, so the
        two classes cannot drift apart in their variance rules; this class
        adds only the pinned weight, the absence of a sum-of-coefficients
        field, and its own label. Built on each call, which is cheap, since
        the prior is a frozen dataclass of four scalars.

        Returns:
            A :class:`MinnesotaPrior` with ``cross_equation=1.0`` and
            ``sum_of_coefficients=None``, the other fields copied.

        Example:
            >>> NormalInverseWishartPrior(tightness=0.1)._minnesota()._label()
            'minnesota(l1=0.1, l2=1, l3=1, l4=100)'
        """
        return MinnesotaPrior(
            tightness=self.tightness,
            cross_equation=1.0,
            decay=self.decay,
            exogenous=self.exogenous,
            sum_of_coefficients=None,
            persistence=self.persistence,
        )

    def coefficient_mean(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Each variable's own first lag at ``persistence``, everything else zero.

        Delegates to :meth:`MinnesotaPrior.coefficient_mean` through
        :meth:`_minnesota`.

        Args:
            context: The sample description; ``k_endog``, ``width``, and
                ``lag_offset`` are read.

        Returns:
            A ``(width, k)`` array in design-column order.

        Raises:
            SpecificationError: If a hyperparameter is out of range.
            DimensionError: If ``persistence`` has the wrong length.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> NormalInverseWishartPrior(persistence=0.0).coefficient_mean(ctx)
            array([[0., 0.],
                   [0., 0.],
                   [0., 0.]])
        """
        return self._minnesota().coefficient_mean(context)

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        r"""Litterman's variances with the cross-equation weight at one.

        Delegates to :meth:`MinnesotaPrior.coefficient_variance` through
        :meth:`_minnesota`, so an own lag at lag :math:`l` gets
        :math:`(\lambda_1 / l^{\lambda_3})^2`, a cross lag the same times
        :math:`(s_i / s_j)^2`, and the deterministic and exogenous block
        :math:`(\lambda_1 \lambda_4 s_i)^2`. Every column of the result is
        :math:`s_i^2` times a column-specific constant, which is the
        factorization the conjugate estimator checks before it proceeds.

        Args:
            context: The sample description; ``k_endog``, ``order``,
                ``scales``, ``width``, ``lag_offset``, ``n_deterministic``,
                and ``k_exog`` are read.

        Returns:
            A ``(width, k)`` array of finite positive variances in
            design-column order.

        Raises:
            SpecificationError: If a hyperparameter is out of range.

        Example:
            Dividing each column by :math:`s_i^2` leaves a column-wise
            constant, the factorization that makes the posterior exact:

            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.array([1.0, 4.0]),
            ...     presample_mean=np.zeros(2),
            ... )
            >>> variance = NormalInverseWishartPrior().coefficient_variance(ctx)
            >>> (variance / ctx.scales**2).round(4)
            array([[4.0e+02, 4.0e+02],
                   [4.0e-02, 4.0e-02],
                   [2.5e-03, 2.5e-03]])
        """
        return self._minnesota().coefficient_variance(context)

    def _label(self) -> str:
        """Short description for summary tables and comparison rows.

        ``niw(l1=..., l3=..., l4=...)`` in ``%g`` format; there is no ``l2``
        because it is pinned, and no ``l5`` because dummy content is a
        separate component whose own label is joined with ``+``. It is the
        ``prior_label`` of every result fitted under this prior and the
        ``source`` of its marginal-likelihood record.

        Returns:
            The label.

        Example:
            >>> NormalInverseWishartPrior()._label()
            'niw(l1=0.2, l3=1, l4=100)'
            >>> (NormalInverseWishartPrior() + SumOfCoefficientsPrior())._label()
            'niw(l1=0.2, l3=1, l4=100) + soc(1)'
        """
        return f"niw(l1={self.tightness:g}, l3={self.decay:g}, l4={self.exogenous:g})"


@dataclass(frozen=True, kw_only=True, slots=True)
class SumOfCoefficientsPrior(_Prior):
    r"""The no-cointegration dummies of Doan, Litterman & Sims.

    One artificial observation per variable, each saying that a series
    sitting at its pre-sample average forever should stay there. For
    variable :math:`j` with pre-sample mean :math:`\bar y_j` and tightness
    :math:`\mu`, the row has target :math:`\mu \bar y_j` in column
    :math:`j`, the same value in variable :math:`j`'s column at every lag
    of the design, and zeros in the deterministic and exogenous columns:

    .. math::

       \underbrace{\mu\, \bar y_j\, e_j^\top}_{\text{target}}
       \;=\;
       \underbrace{\bigl[\,0_{n_{\text{det}}},\;
       \mu\, \bar y_j\, e_j^\top,\; \ldots,\; \mu\, \bar y_j\, e_j^\top,\;
       0_{k_{\text{exog}}}\bigr]}_{\text{design row}}\; B + \text{error}.

    Regressing target on design then says :math:`\sum_l A_{l,jj} = 1` and
    :math:`\sum_l A_{l,ij} = 0` for :math:`i \ne j`, with weight :math:`\mu`
    relative to a data row. That is a restriction on the *sum* of a
    variable's lag coefficients, which no diagonal variance can state, and
    which therefore enters as rows rather than as moments. At the limit
    :math:`\mu \to \infty` it pushes every equation toward a unit root with
    no cross-variable error correction, which is why the literature reads
    it as a no-cointegration prior, and why it should be loosened when the
    system is believed to cointegrate.

    This is the standalone, composable form of what
    :attr:`MinnesotaPrior.sum_of_coefficients` embeds: written separately it
    can be stacked onto the conjugate prior with ``+``, given its own
    tightness, and carried into the hierarchical layer as its own
    hyperparameter, which is how Giannone, Lenza, and Primiceri (2015)
    treat it. Alone it is improper -- every coefficient variance is
    infinite -- so it is a component, not a prior: the conjugate model
    refuses it by itself and accepts ``NormalInverseWishartPrior() +
    SumOfCoefficientsPrior()``.

    Attributes:
        tightness: How hard the restriction binds; larger is tighter,
            matching the package's :math:`\lambda_5` convention. The
            Giannone-Lenza-Primiceri :math:`\mu` is its reciprocal.

    Raises:
        SpecificationError: If ``tightness`` is not positive, raised when
            :meth:`dummy_observations` is first called, which is the
            model's fit.

    Note:
        The rows are scaled by the pre-sample means, so on data with a
        mean near zero -- demeaned series, or growth rates -- the rows
        carry almost no weight whatever ``tightness`` is, and the
        restriction is effectively absent. That is correct behaviour
        (there is no level to hold), but it means the hyperparameter's
        effect depends on the data's units. The deterministic columns of
        the row are zero, so the restriction is on the lag sum alone and
        says nothing about the intercept; the
        :class:`DummyInitialObservationPrior` is the component that ties the
        intercept to the pre-sample mean.

    See Also:
        * :class:`DummyInitialObservationPrior` -- the companion row, one
          for the whole system, allowing cointegration.
        * :class:`NormalInverseWishartPrior` -- the proper base prior this
          component is stacked onto.
        * :attr:`MinnesotaPrior.sum_of_coefficients` -- the same rows
          carried inside the Minnesota prior for the non-conjugate paths.

    References:
        Doan, T., Litterman, R., & Sims, C. (1984). Forecasting and
        conditional projection using realistic prior distributions.
        *Econometric Reviews*, 3(1), 1-100.

        Sims, C. A., & Zha, T. (1998). Bayesian methods for dynamic
        multivariate models. *International Economic Review*, 39(4),
        949-968.

        Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection
        for vector autoregressions. *Review of Economics and Statistics*,
        97(2), 436-451.

    Example:
        Two variables at two lags, pre-sample means ten and twenty, at half
        weight; the design row repeats the target in the variable's own
        column at each lag and is zero in the leading intercept column:

        >>> import numpy as np
        >>> from cultivars._internals import _PriorContext
        >>> ctx = _PriorContext(
        ...     k_endog=2, order=2, scales=np.ones(2),
        ...     presample_mean=np.array([10.0, 20.0]),
        ... )
        >>> target, design = SumOfCoefficientsPrior(tightness=0.5).dummy_observations(ctx)
        >>> target
        array([[ 5.,  0.],
               [ 0., 10.]])
        >>> design
        array([[ 0.,  5.,  0.,  5.,  0.],
               [ 0.,  0., 10.,  0., 10.]])

        Stacked onto the conjugate prior, the rows enter the exact update
        and are counted as dummy observations on the result:

        >>> from cultivars.multivariate.large_dim.bayesian import BVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((120, 2))
        >>> for t in range(1, 120):
        ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
        >>> prior = NormalInverseWishartPrior() + SumOfCoefficientsPrior(tightness=2.0)
        >>> res = BVAR(y, order=1, prior=prior).fit(n_draws=100, seed=0)
        >>> res.prior_label, res.n_dummy
        ('niw(l1=0.2, l3=1, l4=100) + soc(2)', 2)

        Alone it is improper and the conjugate model refuses it:

        >>> BVAR(y, order=1, prior=SumOfCoefficientsPrior()).fit(n_draws=10)
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: the prior leaves some coefficient variances ...
    """

    tightness: float = 1.0
    r"""Weight of each restriction row relative to a data row, :math:`\lambda_5`.

    Larger binds the lag sums harder to one on the diagonal and zero off
    it; at the limit every equation is a unit root without error
    correction. Giannone, Lenza, and Primiceri parameterize the same rows
    by :math:`\mu = 1 / \lambda_5`, so their default of one coincides with
    this one and their "looser" is this field's "smaller". Must be
    positive.
    """

    def _check(self) -> None:
        """Reject a non-positive tightness.

        Called by :meth:`dummy_observations`, the only method whose output
        depends on the field; the moment methods return constants and do
        not call it.

        Raises:
            SpecificationError: If ``tightness`` is not positive.

        Example:
            >>> SumOfCoefficientsPrior(tightness=0.0)._check()
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: tightness must be positive; got 0.0.
        """
        if self.tightness <= 0.0:
            raise SpecificationError(f"tightness must be positive; got {self.tightness}.")

    def coefficient_mean(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Zero: this prior's content is entirely in its rows.

        A component with no moment content reports a zero mean so that
        composition, which averages means by precision, is unaffected by
        it; with infinite variance the precision is zero and the mean is
        never weighted.

        Args:
            context: The sample description; ``width`` and ``k_endog`` are
                read.

        Returns:
            A ``(width, k)`` array of zeros.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> SumOfCoefficientsPrior().coefficient_mean(ctx)
            array([[0., 0.],
                   [0., 0.],
                   [0., 0.]])
        """
        return np.zeros((context.width, context.k_endog), dtype=np.float64)

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Infinite: no diagonal opinion about any coefficient.

        Infinity is how a prior says it has no view of a coefficient, and
        in composition an infinite variance contributes zero precision, so
        stacking this component onto a proper prior leaves that prior's
        moments exactly as they were. It is also why the component cannot
        stand alone: a marginal likelihood needs every variance finite.

        Args:
            context: The sample description; ``width`` and ``k_endog`` are
                read.

        Returns:
            A ``(width, k)`` array of ``inf``.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> bool(np.isinf(SumOfCoefficientsPrior().coefficient_variance(ctx)).all())
            True
        """
        return np.full((context.width, context.k_endog), np.inf, dtype=np.float64)

    def dummy_observations(
        self, context: _PriorContext
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""One row per variable, centred on its pre-sample mean.

        The target block is :math:`\lambda_5 \operatorname{diag}(\bar y)`;
        the design block repeats it in every lag position, with the
        deterministic columns supplied by the context's ``_lead`` -- as many
        zero columns as the design has deterministic terms, so the block
        lines up under a constant, a constant and trend, or a panel's fixed
        effects alike -- and zeros under any exogenous regressors.

        Args:
            context: The sample description; ``k_endog``, ``order``,
                ``presample_mean``, ``n_deterministic``, and ``k_exog`` are
                read.

        Returns:
            A ``(k, k)`` target block and a ``(k, width)`` design block.

        Raises:
            SpecificationError: If ``tightness`` is not positive.

        Example:
            Under a constant and a trend the row carries two leading zeros:

            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2),
            ...     presample_mean=np.array([10.0, 20.0]), n_deterministic=2,
            ... )
            >>> SumOfCoefficientsPrior(tightness=0.5).dummy_observations(ctx)[1]
            array([[ 0.,  0.,  5.,  0.],
                   [ 0.,  0.,  0., 10.]])
        """
        self._check()
        size = context.k_endog
        centre = np.diag(context.presample_mean) * self.tightness
        block = np.hstack(
            [
                context._lead(size),
                np.kron(np.ones((1, context.order)), centre),
                np.zeros((size, context.k_exog), dtype=np.float64),
            ]
        )
        return centre, block

    def _label(self) -> str:
        """Short description for summary tables and comparison rows.

        ``soc(<tightness>)`` in ``%g`` format; in a composition it is joined
        to the other components' labels with ``+``.

        Returns:
            The label.

        Example:
            >>> SumOfCoefficientsPrior()._label()
            'soc(1)'
            >>> SumOfCoefficientsPrior(tightness=0.25)._label()
            'soc(0.25)'
        """
        return f"soc({self.tightness:g})"


@dataclass(frozen=True, kw_only=True, slots=True)
class DummyInitialObservationPrior(_Prior):
    r"""Sims' co-persistence dummy, the single-unit-root prior.

    One artificial observation in which every variable sits at its
    pre-sample mean and every deterministic term at one. With pre-sample
    means :math:`\bar y` and tightness :math:`\delta`, the row is

    .. math::

       \underbrace{\delta\, \bar y^\top}_{\text{target}}
       \;=\;
       \underbrace{\bigl[\,\delta\, 1_{n_{\text{det}}},\;
       \delta\, \bar y^\top,\; \ldots,\; \delta\, \bar y^\top,\;
       0_{k_{\text{exog}}}\bigr]}_{\text{design row}}\; B + \text{error},

    which says that a system starting at its pre-sample mean, with the
    deterministic terms switched on, should predict that same mean: either
    the lag coefficients sum to the identity and the constant is zero, or
    the constant absorbs whatever the lags leave. Where the
    sum-of-coefficients rows allow each variable its own unit root, this
    single row says that if the system is that persistent, it is persistent
    *jointly* -- a common stochastic trend rather than one per series -- and
    it is what keeps the sum-of-coefficients restriction from ruling out
    cointegration entirely (Sims 1993; Sims & Zha 1998). The pair is
    standard equipment in the Giannone, Lenza, and Primiceri (2015)
    hierarchy, each with its own tightness.

    Alone it is improper -- every coefficient variance is infinite -- so it
    is a component, not a prior; the conjugate model accepts it only
    stacked onto a proper base, ``NormalInverseWishartPrior() +
    SumOfCoefficientsPrior() + DummyInitialObservationPrior()``.

    Attributes:
        tightness: How hard the restriction binds; larger is tighter. The
            Giannone-Lenza-Primiceri :math:`\delta` is its reciprocal.

    Raises:
        SpecificationError: If ``tightness`` is not positive, raised when
            :meth:`dummy_observations` is first called, which is the
            model's fit.

    Note:
        Unlike the sum-of-coefficients rows, this row has non-zero
        deterministic columns, so it is the one component that ties the
        intercept (and trend, and panel fixed effects) to the pre-sample
        level. It is also the one that binds when the pre-sample means are
        near zero: with :math:`\bar y \approx 0` the row reduces to
        :math:`0 = \delta\, c + \text{error}`, a statement that the
        intercept is small.

    See Also:
        * :class:`SumOfCoefficientsPrior` -- the companion rows, one per
          variable, which this row softens toward a common trend.
        * :class:`NormalInverseWishartPrior` -- the proper base prior the
          two components are stacked onto.

    References:
        Sims, C. A. (1993). A nine-variable probabilistic macroeconomic
        forecasting model. In J. H. Stock & M. W. Watson (Eds.), *Business
        Cycles, Indicators, and Forecasting* (pp. 179-212). University of
        Chicago Press.

        Sims, C. A., & Zha, T. (1998). Bayesian methods for dynamic
        multivariate models. *International Economic Review*, 39(4),
        949-968.

        Giannone, D., Lenza, M., & Primiceri, G. E. (2015). Prior selection
        for vector autoregressions. *Review of Economics and Statistics*,
        97(2), 436-451.

    Example:
        Two variables at two lags, pre-sample means ten and twenty, at half
        weight; one row, with the intercept column at the tightness and the
        means repeated at each lag:

        >>> import numpy as np
        >>> from cultivars._internals import _PriorContext
        >>> ctx = _PriorContext(
        ...     k_endog=2, order=2, scales=np.ones(2),
        ...     presample_mean=np.array([10.0, 20.0]),
        ... )
        >>> target, design = DummyInitialObservationPrior(tightness=0.5).dummy_observations(ctx)
        >>> target
        array([[ 5., 10.]])
        >>> design
        array([[ 0.5,  5. , 10. ,  5. , 10. ]])

        The full Giannone-Lenza-Primiceri stack on the conjugate model, with
        the rows counted on the result:

        >>> from cultivars.multivariate.large_dim.bayesian import BVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((120, 2))
        >>> for t in range(1, 120):
        ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
        >>> base = NormalInverseWishartPrior() + SumOfCoefficientsPrior()
        >>> prior = base + DummyInitialObservationPrior()
        >>> res = BVAR(y, order=1, prior=prior).fit(n_draws=100, seed=0)
        >>> res.prior_label, res.n_dummy
        ('niw(l1=0.2, l3=1, l4=100) + soc(1) + dio(1)', 3)
    """

    tightness: float = 1.0
    r"""Weight of the restriction row relative to a data row.

    Larger binds the system harder to a common trend through its
    pre-sample mean. Giannone, Lenza, and Primiceri parameterize the same
    row by :math:`\delta = 1 / \text{tightness}`, so their default of one
    coincides with this one and their "looser" is this field's "smaller".
    Must be positive.
    """

    def _check(self) -> None:
        """Reject a non-positive tightness.

        Called by :meth:`dummy_observations`, the only method whose output
        depends on the field; the moment methods return constants and do
        not call it.

        Raises:
            SpecificationError: If ``tightness`` is not positive.

        Example:
            >>> DummyInitialObservationPrior(tightness=-1.0)._check()
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: tightness must be positive; got -1.0.
        """
        if self.tightness <= 0.0:
            raise SpecificationError(f"tightness must be positive; got {self.tightness}.")

    def coefficient_mean(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Zero: this prior's content is entirely in its row.

        With infinite variance the component has zero precision, so in a
        composition its mean is never weighted and the base prior's centre
        is untouched.

        Args:
            context: The sample description; ``width`` and ``k_endog`` are
                read.

        Returns:
            A ``(width, k)`` array of zeros.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> DummyInitialObservationPrior().coefficient_mean(ctx).shape
            (3, 2)
        """
        return np.zeros((context.width, context.k_endog), dtype=np.float64)

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Infinite: no diagonal opinion about any coefficient.

        Zero precision in a composition, so stacking this component onto a
        proper prior leaves that prior's moments exactly as they were; and
        the reason the component cannot stand alone, since a marginal
        likelihood needs every variance finite.

        Args:
            context: The sample description; ``width`` and ``k_endog`` are
                read.

        Returns:
            A ``(width, k)`` array of ``inf``.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> bool(np.isinf(DummyInitialObservationPrior().coefficient_variance(ctx)).all())
            True
        """
        return np.full((context.width, context.k_endog), np.inf, dtype=np.float64)

    def dummy_observations(
        self, context: _PriorContext
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""A single artificial observation at the pre-sample means.

        The target is :math:`\delta \bar y^\top`; the design row carries
        :math:`\delta` in every deterministic column, supplied by the
        context's ``_lead`` so that a constant, a constant and trend, and a
        panel's fixed effects are all switched on alike, then
        :math:`\delta \bar y^\top` at each lag, and zeros under any
        exogenous regressors. With no deterministic terms the row is the
        lag part alone.

        Args:
            context: The sample description; ``k_endog``, ``order``,
                ``presample_mean``, ``n_deterministic``, and ``k_exog`` are
                read.

        Returns:
            A ``(1, k)`` target block and a ``(1, width)`` design block.

        Raises:
            SpecificationError: If ``tightness`` is not positive.

        Example:
            Under a constant and a trend both deterministic columns are set
            to the tightness:

            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2),
            ...     presample_mean=np.array([10.0, 20.0]), n_deterministic=2,
            ... )
            >>> DummyInitialObservationPrior(tightness=0.5).dummy_observations(ctx)[1]
            array([[ 0.5,  0.5,  5. , 10. ]])
        """
        self._check()
        mean = context.presample_mean * self.tightness
        block = np.hstack(
            [
                context._lead(1, self.tightness),
                np.kron(np.ones((1, context.order)), mean[None, :]),
                np.zeros((1, context.k_exog), dtype=np.float64),
            ]
        )
        return mean[None, :], block

    def _label(self) -> str:
        """Short description for summary tables and comparison rows.

        ``dio(<tightness>)`` in ``%g`` format; in a composition it is joined
        to the other components' labels with ``+``.

        Returns:
            The label.

        Example:
            >>> DummyInitialObservationPrior()._label()
            'dio(1)'
            >>> DummyInitialObservationPrior(tightness=3.0)._label()
            'dio(3)'
        """
        return f"dio({self.tightness:g})"


@dataclass(frozen=True, kw_only=True, slots=True)
class IndependentNormalWishartPrior(_Prior):
    r"""Litterman's full prior under an independent inverse-Wishart pairing.

    The variances are exactly :class:`MinnesotaPrior`'s, cross-equation
    weight included, and that is the point. The conjugate
    Normal-inverse-Wishart pairing had to pin that weight to one because it
    ties the coefficient variance to :math:`\Sigma` through a Kronecker
    product; stating the prior on the coefficients *independently* of
    :math:`\Sigma`,

    .. math::

       \operatorname{vec}(B) \sim \mathcal{N}\bigl(\operatorname{vec}(B_0),\, V_0\bigr),
       \qquad
       \Sigma \sim \mathcal{IW}(S_0, \nu_0),
       \qquad V_0 \text{ diagonal, } B \perp \Sigma,

    drops the factorization requirement, so :math:`V_0` can carry
    Litterman's :math:`\lambda_2` and the sum-of-coefficients rows alike.
    The price is the closed form. With :math:`B` and :math:`\Sigma`
    independent a priori, the joint posterior is not a named distribution;
    it is reached by a two-block Gibbs sampler, coefficients given the
    covariance by generalized least squares,

    .. math::

       \operatorname{vec}(B) \mid \Sigma, y \sim \mathcal{N}\bigl(\bar\beta,\, \bar V\bigr),
       \quad
       \bar V = \bigl(V_0^{-1} + \Sigma^{-1} \otimes X^\top X\bigr)^{-1},

    and the covariance given the coefficients by an inverse-Wishart on the
    residuals, and the marginal likelihood is not available in closed form
    but is estimated from the Gibbs output by Chib's (1995) method. That
    trade -- Litterman's economics back, the evidence simulated rather than
    exact -- is the standard one (Koop & Korobilis 2010), and this class is
    its name. It is the default prior of
    :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR`, and the
    conjugate :class:`~cultivars.multivariate.large_dim.bayesian.BVAR`
    refuses it at fit.

    Attributes:
        tightness: Overall confidence, :math:`\lambda_1`.
        cross_equation: How much harder cross-variable coefficients shrink,
            :math:`\lambda_2`, the hyperparameter this pairing exists to
            keep.
        decay: Lag decay, :math:`\lambda_3`.
        exogenous: Looseness of the intercept and exogenous block,
            :math:`\lambda_4`.
        sum_of_coefficients: The no-cointegration restriction,
            :math:`\lambda_5`; ``None`` omits it.
        persistence: Prior mean of each variable's own first lag.

    Raises:
        SpecificationError: When a hyperparameter is used, not at
            construction: if ``tightness``, ``cross_equation``, or
            ``exogenous`` is not positive, if ``decay`` is negative, or if
            ``sum_of_coefficients`` is given and not positive, at the first
            moment or dummy call, which is the model's fit.
        DimensionError: If ``persistence`` is a sequence whose length is not
            the number of variables, at the same point.

    Note:
        The moments and rows are those of a :class:`MinnesotaPrior` with
        the same fields; the two classes differ only in the pairing with
        :math:`\Sigma` that the estimator infers from the type, and in the
        label. A :class:`MinnesotaPrior` handed to the Gibbs model is
        treated identically; this class exists so the pairing is stated in
        the prior's name rather than implied by the model it was passed to.

    See Also:
        * :class:`MinnesotaPrior` -- the same moments, for the point-estimate
          path.
        * :class:`NormalInverseWishartPrior` -- the conjugate pairing, with
          the cross-equation weight pinned and an exact posterior.
        * :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR` -- the
          model this prior is the default for.
        * :func:`~cultivars.bayes.evidence.marginal_likelihood` -- the Chib
          estimate the Gibbs result reports under this prior.

    References:
        Kadiyala, K. R., & Karlsson, S. (1997). Numerical methods for
        estimation and inference in Bayesian VAR-models. *Journal of Applied
        Econometrics*, 12(2), 99-132.

        Chib, S. (1995). Marginal likelihood from the Gibbs output. *Journal
        of the American Statistical Association*, 90(432), 1313-1321.

        Koop, G., & Korobilis, D. (2010). Bayesian multivariate time series
        methods for empirical macroeconomics. *Foundations and Trends in
        Econometrics*, 3(4), 267-358.

    Example:
        The Gibbs BVAR under this prior keeps the cross-equation weight and
        reports a Chib marginal likelihood:

        >>> import numpy as np
        >>> from cultivars.multivariate.large_dim.gibbs import GibbsBVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((120, 2))
        >>> for t in range(1, 120):
        ...     y[t] = 0.7 * y[t - 1] + rng.standard_normal(2)
        >>> prior = IndependentNormalWishartPrior(tightness=0.1, cross_equation=0.5)
        >>> res = GibbsBVAR(y, order=1, prior=prior).fit(n_draws=300, n_burn=100, seed=0)
        >>> res.prior_label, res.beta_draws.shape
        ('inw(l1=0.1, l2=0.5, l3=1, l4=100)', (200, 3, 2))
        >>> res.marginal_likelihood().method
        'chib'

        The conjugate model refuses it, since its variance does not factor:

        >>> from cultivars.multivariate.large_dim.bayesian import BVAR
        >>> BVAR(y, order=1, prior=prior).fit(n_draws=10, seed=0)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        cultivars.exceptions.SpecificationError: the prior variance does not factor ...
    """

    tightness: float = 0.2
    r"""Overall confidence :math:`\lambda_1`; scales every prior variance.

    The standard deviation of a variable's own first-lag coefficient.
    Values near 0.1 to 0.3 are usual for macroeconomic data; a large value
    approaches least squares. Must be positive.
    """
    cross_equation: float = 0.5
    r"""Relative tightness of cross-variable coefficients :math:`\lambda_2`.

    The hyperparameter this pairing exists to keep: a cross lag's standard
    deviation is the own lag's times this, and any value is admissible
    because the coefficient prior is stated independently of
    :math:`\Sigma`. Must be positive.
    """
    decay: float = 1.0
    r"""Lag decay :math:`\lambda_3`; lag :math:`l` is shrunk by :math:`l^{-\lambda_3}`.

    One is harmonic decay, Litterman's choice; zero treats every lag alike.
    Must be non-negative.
    """
    exogenous: float = 100.0
    r"""Looseness of the deterministic and exogenous block :math:`\lambda_4`.

    The intercept, any trend, and any exogenous regressor get standard
    deviation :math:`\lambda_1 \lambda_4 s_i`, four orders of magnitude
    looser than an own lag at the default. Finite, so the prior is proper
    and Chib's estimator applies. Must be positive.
    """
    sum_of_coefficients: float | None = None
    r"""Confidence in the sum-of-coefficients restriction :math:`\lambda_5`, or ``None``.

    When given, :meth:`dummy_observations` contributes one row per
    variable, exactly as :class:`MinnesotaPrior` does. Under the Gibbs
    model dummy rows are stacked under the sample before every sweep. Must
    be positive when given.
    """
    persistence: float | Sequence[float] = 1.0
    """Prior mean of each variable's own first lag: a scalar, or one value per variable.

    One is the random-walk prior for series in levels; zero suits
    differenced or otherwise stationary data. A sequence gives each
    variable its own centre and must have one entry per variable.
    """

    def _minnesota(self) -> MinnesotaPrior:
        """The :class:`MinnesotaPrior` whose moments and rows this prior states.

        Every field is copied across unchanged, so the two classes cannot
        drift apart in their variance rules or their dummy construction;
        this class contributes only its label and the pairing its type
        signals to the estimator. Built on each call, which is cheap.

        Returns:
            A :class:`MinnesotaPrior` with the same six fields.

        Example:
            >>> IndependentNormalWishartPrior(cross_equation=0.3)._minnesota()._label()
            'minnesota(l1=0.2, l2=0.3, l3=1, l4=100)'
        """
        return MinnesotaPrior(
            tightness=self.tightness,
            cross_equation=self.cross_equation,
            decay=self.decay,
            exogenous=self.exogenous,
            sum_of_coefficients=self.sum_of_coefficients,
            persistence=self.persistence,
        )

    def coefficient_mean(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Each variable's own first lag at ``persistence``, everything else zero.

        Delegates to :meth:`MinnesotaPrior.coefficient_mean` through
        :meth:`_minnesota`.

        Args:
            context: The sample description; ``k_endog``, ``width``, and
                ``lag_offset`` are read.

        Returns:
            A ``(width, k)`` array in design-column order.

        Raises:
            SpecificationError: If a hyperparameter is out of range.
            DimensionError: If ``persistence`` has the wrong length.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> IndependentNormalWishartPrior().coefficient_mean(ctx)
            array([[0., 0.],
                   [1., 0.],
                   [0., 1.]])
        """
        return self._minnesota().coefficient_mean(context)

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        r"""Litterman's variances, the cross-equation weight kept.

        Delegates to :meth:`MinnesotaPrior.coefficient_variance` through
        :meth:`_minnesota`: own lags :math:`(\lambda_1 / l^{\lambda_3})^2`,
        cross lags the same times :math:`(\lambda_2 s_i / s_j)^2`, the
        deterministic and exogenous block :math:`(\lambda_1 \lambda_4
        s_i)^2`. With :math:`\lambda_2 \ne 1` a column is no longer
        :math:`s_i^2` times a constant, which is what the conjugate
        estimator checks for and this pairing does not need.

        Args:
            context: The sample description; ``k_endog``, ``order``,
                ``scales``, ``width``, ``lag_offset``, ``n_deterministic``,
                and ``k_exog`` are read.

        Returns:
            A ``(width, k)`` array of finite positive variances in
            design-column order.

        Raises:
            SpecificationError: If a hyperparameter is out of range.

        Example:
            With residual scales one and four, dividing by :math:`s_i^2`
            no longer leaves a column-wise constant, because of the
            cross-equation weight:

            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.array([1.0, 4.0]),
            ...     presample_mean=np.zeros(2),
            ... )
            >>> variance = IndependentNormalWishartPrior().coefficient_variance(ctx)
            >>> (variance / ctx.scales**2).round(4)
            array([[4.0e+02, 4.0e+02],
                   [4.0e-02, 1.0e-02],
                   [6.0e-04, 2.5e-03]])
        """
        return self._minnesota().coefficient_variance(context)

    def dummy_observations(
        self, context: _PriorContext
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The sum-of-coefficients rows, when that restriction is asked for.

        Delegates to :meth:`MinnesotaPrior.dummy_observations` through
        :meth:`_minnesota`: one row per variable centred on
        ``persistence`` times its pre-sample mean, scaled by
        ``sum_of_coefficients``, or the empty blocks when that field is
        ``None``.

        Args:
            context: The sample description; ``k_endog``, ``order``,
                ``presample_mean``, ``lag_offset``, and ``k_exog`` are read.

        Returns:
            A ``(k, k)`` target block and a ``(k, width)`` design block when
            ``sum_of_coefficients`` is set; ``(0, k)`` and ``(0, width)``
            otherwise.

        Raises:
            SpecificationError: If a hyperparameter is out of range.
            DimensionError: If ``persistence`` has the wrong length.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2),
            ...     presample_mean=np.array([10.0, 20.0]),
            ... )
            >>> IndependentNormalWishartPrior(sum_of_coefficients=0.5).dummy_observations(ctx)[1]
            array([[ 0.,  5.,  0.],
                   [ 0.,  0., 10.]])
            >>> IndependentNormalWishartPrior().dummy_observations(ctx)[1].shape
            (0, 3)
        """
        return self._minnesota().dummy_observations(context)

    def _label(self) -> str:
        """Short description for summary tables and comparison rows.

        ``inw(l1=..., l2=..., l3=..., l4=...)`` with ``l5`` appended when
        the sum-of-coefficients restriction is set, each value in ``%g``
        format. It is the ``prior_label`` of every Gibbs result fitted
        under this prior and the ``source`` of its Chib record.

        Returns:
            The label.

        Example:
            >>> IndependentNormalWishartPrior()._label()
            'inw(l1=0.2, l2=0.5, l3=1, l4=100)'
            >>> IndependentNormalWishartPrior(sum_of_coefficients=2.0)._label()
            'inw(l1=0.2, l2=0.5, l3=1, l4=100, l5=2)'
        """
        tail = "" if self.sum_of_coefficients is None else f", l5={self.sum_of_coefficients:g}"
        return (
            f"inw(l1={self.tightness:g}, l2={self.cross_equation:g}, "
            f"l3={self.decay:g}, l4={self.exogenous:g}{tail})"
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class HorseshoePrior(_AdaptivePrior):
    r"""The horseshoe: aggressive shrinkage that lets large signals through.

    Each standardized lag coefficient gets a half-Cauchy local scale and the
    whole lag block shares a half-Cauchy global one (Carvalho, Polson &
    Scott 2010),

    .. math::

       \beta_{ij} \mid \lambda_{ij}, \tau \sim \mathcal{N}(0, \tau^2 \lambda_{ij}^2),
       \qquad
       \lambda_{ij} \sim \mathcal{C}^+(0, 1),
       \qquad
       \tau \sim \mathcal{C}^+(0, 1),

    where :math:`\beta_{ij}` is the coefficient divided by the Minnesota
    unit ratio :math:`s_i / s_j`, so one latent scale means the same thing
    whatever the variables' units. The Cauchy tails are the substance. The
    shrinkage weight :math:`\kappa_{ij} = 1 / (1 + \tau^2 \lambda_{ij}^2)`
    has a :math:`\mathrm{Beta}(1/2, 1/2)` prior, the horseshoe shape that
    names the estimator: mass piles up at :math:`\kappa = 1` hard enough to
    wipe out noise coefficients, and at :math:`\kappa = 0` so that a large
    coefficient escapes shrinkage almost entirely. A Gaussian variance --
    the Minnesota family -- shrinks everything proportionally and can do
    neither. There are no hyperparameters to tune, and that is real: the
    global scale learns the overall sparsity from the data, and the default
    half-Cauchy on :math:`\tau` is the one the literature settled on.

    Sampling uses the Makalic-Schmidt (2016) inverse-Gamma augmentation.
    A half-Cauchy is an inverse-Gamma mixture,

    .. math::

       \lambda^2 \mid \nu \sim \mathcal{IG}(1/2, 1/\nu),
       \qquad
       \nu \sim \mathcal{IG}(1/2, 1),

    and likewise :math:`\tau^2` through an auxiliary :math:`\xi`, so every
    conditional in the hierarchy is inverse-Gamma: exact draws, no tuning,
    no rejection, one block per sweep. The deterministic and exogenous
    columns are not shrunk; they sit at a fixed loose variance, exactly
    where the Minnesota prior leaves them.

    Attributes:
        deterministic_scale: Looseness of the unshrunk deterministic and
            exogenous rows, in units of each equation's residual scale.

    Raises:
        SpecificationError: If ``deterministic_scale`` is not positive, at
            the first moment call, which is the model's fit.

    Note:
        The prior mean is zero everywhere, without a persistence
        hyperparameter: centring a sparsity prior away from zero would
        change what shrinking a coefficient means. Difference the data, or
        use the Minnesota family, for random-walk centring. Adaptive priors
        do not compose, so ``HorseshoePrior() + MinnesotaPrior()`` is
        refused at fit, and the Chib marginal likelihood is not offered
        under one; compare by predictive score instead.

    Warning:
        Only :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR`
        samples the scale hierarchy. The static moments interface reports
        the variance *at the initial scales*, unit variance on the
        standardized lag block, which is what admissibility checks need and
        not the horseshoe; a model that reads only the static moments is
        fitting a fixed Gaussian prior under this class's label.

    See Also:
        * :meth:`~cultivars.multivariate.large_dim.gibbs.GibbsBVARResult.shrinkage_scales`
          -- the posterior mean :math:`\tau \lambda_{ij}` per lag
          coefficient, this prior's diagnostic.
        * :class:`DirichletLaplacePrior` -- a global-local prior with the
          same interface and a Dirichlet-tied local scale.
        * :class:`NormalGammaPrior` -- the same interface with a tunable
          shape governing the tails.
        * :class:`SpikeAndSlabPrior` -- selection rather than shrinkage,
          reporting inclusion probabilities.

    References:
        Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe
        estimator for sparse signals. *Biometrika*, 97(2), 465-480.

        Makalic, E., & Schmidt, D. F. (2016). A simple sampler for the
        horseshoe estimator. *IEEE Signal Processing Letters*, 23(1),
        179-182.

        Polson, N. G., & Scott, J. G. (2010). Shrink globally, act locally:
        Sparse Bayesian regularization and prediction. In *Bayesian
        Statistics 9* (pp. 501-538). Oxford University Press.

    Example:
        Three series of which only the first has an own lag. The posterior
        mean local-global scale is order one where the signal is and an
        order of magnitude smaller on every noise coefficient:

        >>> import numpy as np
        >>> from cultivars.multivariate.large_dim.gibbs import GibbsBVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((200, 3))
        >>> for t in range(1, 200):
        ...     y[t] = np.array([0.8, 0.0, 0.0]) * y[t - 1] + rng.standard_normal(3)
        >>> res = GibbsBVAR(y, order=1, prior=HorseshoePrior()).fit(
        ...     n_draws=1500, n_burn=500, seed=0
        ... )
        >>> res.prior_label
        'horseshoe'
        >>> scales = res.shrinkage_scales()[0]
        >>> bool(scales[0, 0] > 0.8), bool(np.delete(scales, 0).max() < 0.3)
        (True, True)
        >>> np.abs(res.coefficients[0]).round(1)
        array([[0.7, 0. , 0. ],
               [0. , 0. , 0.1],
               [0. , 0. , 0. ]])
    """

    deterministic_scale: float = 10.0
    """Prior standard deviation of the unshrunk rows, in residual-scale units.

    The intercept, any trend, and any exogenous regressor of equation
    :math:`i` get variance :math:`(\\text{deterministic\\_scale} \\cdot s_i)^2`,
    held fixed across sweeps. Ten is loose enough to be uninformative for
    standardized data and finite enough to keep the prior proper. Must be
    positive.
    """

    def _check(self) -> None:
        """Reject a non-positive deterministic scale.

        Called from :meth:`_initial_scales`, so the error surfaces at fit
        rather than at construction, matching the Minnesota family.

        Raises:
            SpecificationError: If ``deterministic_scale`` is not positive.

        Example:
            >>> HorseshoePrior(deterministic_scale=0.0)._check()
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: deterministic_scale must be positive; got 0.0.
        """
        if self.deterministic_scale <= 0.0:
            raise SpecificationError(
                f"deterministic_scale must be positive; got {self.deterministic_scale}."
            )

    def _initial_scales(
        self, context: _PriorContext, reference: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
        r"""Unit local scales, unit global scale.

        Starting every :math:`\lambda_{ij}^2` and :math:`\tau^2` at one
        puts the first sweep's coefficient draw under a unit-variance
        Gaussian on the standardized scale, a neutral start the burn-in
        leaves behind quickly. The auxiliaries :math:`\nu_{ij}` and
        :math:`\xi` start at one as well.

        Args:
            context: The sample description; ``k_endog`` and ``order`` are
                read.
            reference: Data-driven reference scales, ignored: the horseshoe
                does not anchor on least-squares standard errors.

        Returns:
            ``{"lam2", "nu"}`` of shape ``(k * p, k)`` and ``{"tau2", "xi"}``
            of shape ``(1,)``, all ones.

        Raises:
            SpecificationError: If ``deterministic_scale`` is not positive.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> prior = HorseshoePrior()
            >>> state = prior._initial_scales(ctx, prior._unit_ratio(ctx))
            >>> {name: value.shape for name, value in state.items()}
            {'lam2': (2, 2), 'nu': (2, 2), 'tau2': (1,), 'xi': (1,)}
        """
        self._check()
        shape = (context.k_endog * context.order, context.k_endog)
        return {
            "lam2": np.ones(shape, dtype=np.float64),
            "nu": np.ones(shape, dtype=np.float64),
            "tau2": np.ones(1, dtype=np.float64),
            "xi": np.ones(1, dtype=np.float64),
        }

    def _draw_scales(
        self,
        standardized: npt.NDArray[np.float64],
        context: _PriorContext,
        rng: np.random.Generator,
        scales: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.float64]]:
        r"""Makalic-Schmidt: four inverse-Gamma blocks, all exact.

        With :math:`m = kp \cdot k` lag coefficients :math:`\beta_{ij}` on
        the standardized scale, the sweep is

        .. math::

           \lambda_{ij}^2 \mid \cdot &\sim
             \mathcal{IG}\bigl(1,\; 1/\nu_{ij} + \beta_{ij}^2 / (2\tau^2)\bigr), \\
           \nu_{ij} \mid \cdot &\sim \mathcal{IG}\bigl(1,\; 1 + 1/\lambda_{ij}^2\bigr), \\
           \tau^2 \mid \cdot &\sim
             \mathcal{IG}\Bigl(\tfrac{m + 1}{2},\;
             1/\xi + \tfrac{1}{2} \sum_{ij} \beta_{ij}^2 / \lambda_{ij}^2\Bigr), \\
           \xi \mid \cdot &\sim \mathcal{IG}\bigl(1,\; 1 + 1/\tau^2\bigr),

        each inverse-Gamma drawn as rate over a Gamma variate. The local
        scales are drawn before the global one within a sweep, and the
        deterministic and exogenous rows of ``standardized`` are not read.

        Args:
            standardized: ``(width, k)`` current coefficients divided by the
                unit ratio.
            context: The sample description; ``k_endog``, ``order``, and
                ``lag_offset`` are read.
            rng: Random generator.
            scales: The current latent state, as returned by
                :meth:`_initial_scales` or the previous sweep.

        Returns:
            The refreshed state under the same four names.

        Example:
            Holding one large and three near-zero standardized coefficients
            fixed and sweeping the scales alone, the averaged
            local-global scale separates them by an order of magnitude:

            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> prior = HorseshoePrior()
            >>> rng = np.random.default_rng(0)
            >>> beta = np.array([[0.0, 0.0], [3.0, 0.05], [-0.05, 0.02]])
            >>> state = prior._initial_scales(ctx, prior._unit_ratio(ctx))
            >>> total = np.zeros((3, 2))
            >>> for _ in range(2000):
            ...     state = prior._draw_scales(beta, ctx, rng, state)
            ...     total += prior._tracked(state, ctx)
            >>> (total / 2000).round(1)
            array([[0. , 0. ],
                   [4.2, 0.3],
                   [0.3, 0.2]])
        """
        offset = context.lag_offset
        block = standardized[offset : offset + context.k_endog * context.order]
        squared = block**2
        count = block.size
        tau2 = float(scales["tau2"][0])
        lam2 = (1.0 / scales["nu"] + squared / (2.0 * tau2)) / rng.standard_gamma(
            1.0, size=squared.shape
        )
        nu = (1.0 + 1.0 / lam2) / rng.standard_gamma(1.0, size=squared.shape)
        rate = float(scales["xi"][0]) ** -1 + float(np.sum(squared / lam2)) / 2.0
        tau2 = rate / float(rng.standard_gamma((count + 1.0) / 2.0))
        xi = (1.0 + 1.0 / tau2) / float(rng.standard_gamma(1.0))
        return {
            "lam2": lam2,
            "nu": nu,
            "tau2": np.array([tau2]),
            "xi": np.array([xi]),
        }

    def _scale_variance(
        self, scales: dict[str, npt.NDArray[np.float64]], context: _PriorContext
    ) -> npt.NDArray[np.float64]:
        r"""``ratio**2 * tau**2 * lambda**2`` on the lag block, loose elsewhere.

        The raw-coefficient variance the current scales imply: the lag
        block carries :math:`(s_i / s_j)^2 \tau^2 \lambda_{ij}^2`, and every
        deterministic and exogenous row carries
        :math:`(\text{deterministic\_scale} \cdot s_i)^2`. At the initial
        scales this is what :meth:`coefficient_variance` reports.

        Args:
            scales: The current latent state.
            context: The sample description; ``scales``, ``width``,
                ``k_endog``, ``order``, and ``lag_offset`` are read.

        Returns:
            A ``(width, k)`` array of positive variances in design-column
            order.

        Example:
            At the initial scales, with residual scales one and four, the
            lag block is the squared unit ratio and the intercept row is a
            hundred times the squared residual scale:

            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.array([1.0, 4.0]),
            ...     presample_mean=np.zeros(2),
            ... )
            >>> prior = HorseshoePrior()
            >>> state = prior._initial_scales(ctx, prior._unit_ratio(ctx))
            >>> prior._scale_variance(state, ctx)
            array([[1.00e+02, 1.60e+03],
                   [1.00e+00, 1.60e+01],
                   [6.25e-02, 1.00e+00]])
        """
        ratio = self._unit_ratio(context)
        out = (self.deterministic_scale * ratio) ** 2
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        out[offset:stop] = ratio[offset:stop] ** 2 * float(scales["tau2"][0]) * scales["lam2"]
        return out

    def _tracked(
        self, scales: dict[str, npt.NDArray[np.float64]], context: _PriorContext
    ) -> npt.NDArray[np.float64]:
        r"""The local-global scale ``tau * lambda`` per lag coefficient.

        :math:`\tau \lambda_{ij}` is the prior standard deviation of the
        standardized coefficient under the current state: near zero means
        the coefficient is being shrunk away, order one or larger means it
        is left free. The sampler averages this over kept sweeps into the
        result's ``shrinkage``; the deterministic and exogenous rows are
        reported as zero because they are not shrunk.

        Args:
            scales: The current latent state.
            context: The sample description; ``width``, ``k_endog``,
                ``order``, and ``lag_offset`` are read.

        Returns:
            A ``(width, k)`` array, zero off the lag block.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> prior = HorseshoePrior()
            >>> state = prior._initial_scales(ctx, prior._unit_ratio(ctx))
            >>> state["tau2"][0] = 0.25
            >>> prior._tracked(state, ctx)
            array([[0. , 0. ],
                   [0.5, 0.5],
                   [0.5, 0.5]])
        """
        out = np.zeros((context.width, context.k_endog), dtype=np.float64)
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        out[offset:stop] = np.sqrt(float(scales["tau2"][0]) * scales["lam2"])
        return out

    def _tracked_label(self) -> str:
        """What the averaged diagnostic is.

        The result's ``shrinkage_label``; it contains ``"scale"``, which is
        how :meth:`~cultivars.multivariate.large_dim.gibbs.GibbsBVARResult.shrinkage_scales`
        knows the diagnostic is a scale rather than an inclusion
        probability.

        Returns:
            The label.

        Example:
            >>> HorseshoePrior()._tracked_label()
            'posterior mean local-global scale of the standardized coefficient'
        """
        return "posterior mean local-global scale of the standardized coefficient"

    def _label(self) -> str:
        """Short description for summary tables and comparison rows.

        The horseshoe has no hyperparameters worth printing, so the label
        is the bare name; ``deterministic_scale`` is not part of the
        shrinkage and is omitted.

        Returns:
            ``'horseshoe'``.

        Example:
            >>> HorseshoePrior()._label()
            'horseshoe'
        """
        return "horseshoe"


@dataclass(frozen=True, kw_only=True, slots=True)
class SpikeAndSlabPrior(_AdaptivePrior):
    r"""Stochastic search variable selection: exclusion as a latent state.

    George, Sun and Ni's (2008) BVAR prior. Each standardized lag
    coefficient carries a Bernoulli indicator choosing between a spike, a
    normal tight enough around zero that the coefficient is effectively
    excluded, and a slab loose enough that it is effectively free,

    .. math::

       \beta_{ij} \mid \gamma_{ij} \sim
       (1 - \gamma_{ij})\, \mathcal{N}(0, \tau_{0,ij}^2)
       + \gamma_{ij}\, \mathcal{N}(0, \tau_{1,ij}^2),
       \qquad
       \gamma_{ij} \sim \mathrm{Bernoulli}(p),

    where :math:`\beta_{ij}` is the coefficient divided by the Minnesota
    unit ratio :math:`s_i / s_j`. Given the coefficients the indicators
    are conditionally independent Bernoullis with the exact odds

    .. math::

       \frac{\Pr(\gamma_{ij} = 1 \mid \beta_{ij})}{\Pr(\gamma_{ij} = 0 \mid \beta_{ij})}
       = \frac{p}{1 - p} \cdot \frac{\tau_{0,ij}}{\tau_{1,ij}}
       \exp\!\Bigl\{\frac{\beta_{ij}^2}{2}
       \Bigl(\frac{1}{\tau_{0,ij}^2} - \frac{1}{\tau_{1,ij}^2}\Bigr)\Bigr\},

    so the Gibbs sweep redraws them without tuning, and their conditional
    probabilities averaged over kept sweeps are a Rao-Blackwellized
    posterior inclusion probability per coefficient. That is this prior's
    distinctive output and what
    :meth:`~cultivars.multivariate.large_dim.gibbs.GibbsBVARResult.inclusion_probabilities`
    reports; a global-local prior has no such quantity, because it never
    asks whether a coefficient is in. SSVS is this object's sampler, not a
    second prior; the two names in the literature name one thing.

    The spike and slab widths follow the semiautomatic default: each
    coefficient's least-squares standard error on the standardized scale,
    times ``spike`` and ``slab``, so the spike is narrower than the data
    can resolve and the slab wider than any coefficient the data would
    produce. The inclusion probability is a fixed hyperparameter rather
    than being given its own Beta layer, matching the reference treatment.
    The deterministic and exogenous columns are never candidates for
    exclusion; they sit at a fixed loose variance.

    Attributes:
        spike: Spike width as a multiple of the coefficient's reference
            standard error; small is a harder exclusion.
        slab: Slab width on the same scale; large is freer.
        inclusion: Prior inclusion probability of each lag coefficient.
        deterministic_scale: Looseness of the unshrunk deterministic and
            exogenous rows, in units of each equation's residual scale.

    Raises:
        SpecificationError: If ``0 < spike < slab`` fails, if ``inclusion``
            is not strictly between zero and one, or if
            ``deterministic_scale`` is not positive, at the first moment
            call, which is the model's fit.

    Note:
        The prior mean is zero everywhere and there is no persistence
        hyperparameter: an exclusion means the coefficient is zero, and
        centring the spike elsewhere would change what exclusion means.
        Difference the data, or use the Minnesota family, for random-walk
        centring. Adaptive priors do not compose, and the Chib marginal
        likelihood is not offered under one; compare by predictive score.
        Inclusion probabilities are posterior statements about *this*
        sample: a coefficient the sample estimates away from zero is
        included whether or not the generating process had it.

    Warning:
        Only :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR`
        samples the indicators. The static moments interface reports the
        variance with every indicator in and the unit ratio as reference,
        ``slab**2`` on the standardized lag block, which is what
        admissibility checks need and not the mixture; a model that reads
        only the static moments is fitting a fixed Gaussian prior under
        this class's label.

    See Also:
        * :meth:`~cultivars.multivariate.large_dim.gibbs.GibbsBVARResult.inclusion_probabilities`
          -- the averaged conditional probability per lag coefficient,
          this prior's diagnostic.
        * :class:`HorseshoePrior` -- continuous shrinkage with no
            exclusion state, when the question is *how much* rather than
            *whether*.
        * :class:`NormalGammaPrior` -- continuous shrinkage with a
          tunable tail.

    References:
        George, E. I., Sun, D., & Ni, S. (2008). Bayesian stochastic search
        for VAR model restrictions. *Journal of Econometrics*, 142(1),
        553-580.

        George, E. I., & McCulloch, R. E. (1993). Variable selection via
        Gibbs sampling. *Journal of the American Statistical Association*,
        88(423), 881-889.

        Koop, G., & Korobilis, D. (2010). Bayesian multivariate time series
        methods for empirical macroeconomics. *Foundations and Trends in
        Econometrics*, 3(4), 267-358.

    Example:
        Three series of which only the first has an own lag. The own lag
        is included with probability one and most noise coefficients sit
        near the floor the hard spike leaves them; the second inclusion is
        a coefficient this sample estimates at ``0.2``, which is the
        sample's doing, not the prior's:

        >>> import numpy as np
        >>> from cultivars.multivariate.large_dim.gibbs import GibbsBVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((200, 3))
        >>> for t in range(1, 200):
        ...     y[t] = np.array([0.8, 0.0, 0.0]) * y[t - 1] + rng.standard_normal(3)
        >>> res = GibbsBVAR(y, order=1, prior=SpikeAndSlabPrior()).fit(
        ...     n_draws=1500, n_burn=500, seed=0
        ... )
        >>> res.prior_label
        'ssvs(spike=0.1, slab=10, p=0.5)'
        >>> probability = res.inclusion_probabilities()[0]
        >>> probability[0, 0].round(2), bool(probability[1, 2] > 0.7)
        (np.float64(1.0), True)
        >>> probability > 0.5
        array([[ True, False, False],
               [False, False,  True],
               [False, False, False]])
        >>> np.abs(res.coefficients[0]).round(1)
        array([[0.7, 0. , 0. ],
               [0. , 0. , 0.2],
               [0. , 0. , 0. ]])
    """

    spike: float = 0.1
    r"""Spike width :math:`\tau_0` as a multiple of the reference standard error.

    A coefficient in the spike is shrunk to within a tenth of a standard
    error of zero at the default, which the data cannot distinguish from
    zero. Smaller is a harder exclusion and a sharper separation from the
    slab. Must be positive and less than ``slab``.
    """
    slab: float = 10.0
    r"""Slab width :math:`\tau_1` as a multiple of the reference standard error.

    A coefficient in the slab is essentially unshrunk at the default, ten
    standard errors wide. The ratio ``slab / spike`` sets how decisively
    the odds separate an included coefficient from an excluded one; the
    default hundred is George, Sun and Ni's. Must exceed ``spike``.
    """
    inclusion: float = 0.5
    r"""Prior inclusion probability :math:`p` of each lag coefficient.

    One half is indifferent: the odds are then decided by the likelihood
    alone. Lower values encode a belief that most coefficients are zero
    and raise the bar for inclusion. Fixed, not learned. Must be strictly
    between zero and one.
    """
    deterministic_scale: float = 10.0
    """Prior standard deviation of the unshrunk rows, in residual-scale units.

    The intercept, any trend, and any exogenous regressor of equation
    :math:`i` get variance :math:`(\\text{deterministic\\_scale} \\cdot s_i)^2`,
    held fixed across sweeps and never subject to exclusion. Must be
    positive.
    """

    def _check(self) -> None:
        """Reject widths and probabilities that do not describe a prior.

        Called from :meth:`_initial_scales`, so the error surfaces at fit
        rather than at construction, matching the Minnesota family.

        Raises:
            SpecificationError: If the widths are not ordered and positive,
                the inclusion probability is not interior, or the
                deterministic scale is not positive.

        Example:
            >>> SpikeAndSlabPrior(spike=1.0, slab=0.5)._check()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: spike and slab must satisfy 0 < spike < ...
            >>> SpikeAndSlabPrior(inclusion=1.0)._check()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: inclusion must be strictly between 0 and 1; ...
        """
        if not 0.0 < self.spike < self.slab:
            raise SpecificationError(
                f"spike and slab must satisfy 0 < spike < slab; got {self.spike}, {self.slab}."
            )
        if not 0.0 < self.inclusion < 1.0:
            raise SpecificationError(
                f"inclusion must be strictly between 0 and 1; got {self.inclusion}."
            )
        if self.deterministic_scale <= 0.0:
            raise SpecificationError(
                f"deterministic_scale must be positive; got {self.deterministic_scale}."
            )

    def _initial_scales(
        self, context: _PriorContext, reference: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
        r"""Anchor both widths to the reference scale; start everything in.

        The semiautomatic default: ``reference`` holds each coefficient's
        least-squares standard error on the raw scale, and dividing by the
        unit ratio puts it on the standardized scale the indicators see.
        Spike and slab are ``spike`` and ``slab`` times that, floored at
        :math:`10^{-8}` so a degenerate column cannot produce a zero
        width. Every indicator starts in the slab, so the first coefficient
        draw is essentially least squares, and the first sweep decides
        which coefficients leave.

        Args:
            context: The sample description; ``k_endog``, ``order``,
                ``scales``, and ``lag_offset`` are read.
            reference: ``(width, k)`` reference scales, least-squares
                standard errors when the sampler can compute them and the
                unit ratio otherwise.

        Returns:
            ``{"tau0", "tau1", "gamma", "prob"}``, each of shape
            ``(k * p, k)``: the two widths, the indicators (all one), and
            the conditional inclusion probabilities (all ``inclusion``).

        Raises:
            SpecificationError: If a hyperparameter is out of range.

        Example:
            With a standard error of ``0.05`` everywhere, the spike is a
            two-hundredth and the slab a half:

            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> state = SpikeAndSlabPrior()._initial_scales(ctx, np.full((3, 2), 0.05))
            >>> state["tau0"].round(6)[0, 0], state["tau1"][0, 0], state["gamma"].sum()
            (np.float64(0.005), np.float64(0.5), np.float64(4.0))
        """
        self._check()
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        anchor = np.maximum(reference[offset:stop] / self._unit_ratio(context)[offset:stop], 1e-8)
        return {
            "tau0": self.spike * anchor,
            "tau1": self.slab * anchor,
            "gamma": np.ones_like(anchor),
            "prob": np.full_like(anchor, self.inclusion),
        }

    def _draw_scales(
        self,
        standardized: npt.NDArray[np.float64],
        context: _PriorContext,
        rng: np.random.Generator,
        scales: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.float64]]:
        r"""Exact Bernoulli conditionals for the indicators.

        For each lag coefficient the log odds of inclusion are

        .. math::

           \log\frac{p}{1 - p} + \log\frac{\tau_0}{\tau_1}
           + \frac{\beta^2}{2}\Bigl(\frac{1}{\tau_0^2} - \frac{1}{\tau_1^2}\Bigr),

        clipped to :math:`\pm 700` before the logistic so a coefficient
        far outside the spike cannot overflow, and the indicator is a
        Bernoulli draw at that probability. The widths do not change
        across sweeps; only ``gamma`` and ``prob`` are refreshed. The
        probability is kept alongside the draw because it, not the binary
        indicator, is what :meth:`_tracked` averages.

        Args:
            standardized: ``(width, k)`` current coefficients divided by the
                unit ratio.
            context: The sample description; ``k_endog``, ``order``, and
                ``lag_offset`` are read.
            rng: Random generator.
            scales: The current latent state.

        Returns:
            The state with ``gamma`` and ``prob`` refreshed.

        Example:
            A coefficient twelve spike widths from zero is in with
            certainty; one well inside the spike is in with roughly the
            prior odds discounted by the width ratio:

            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> prior = SpikeAndSlabPrior()
            >>> state = prior._initial_scales(ctx, np.full((3, 2), 0.05))
            >>> beta = np.array([[0.0, 0.0], [0.6, 0.001], [-0.03, 0.001]])
            >>> state = prior._draw_scales(beta, ctx, np.random.default_rng(0), state)
            >>> state["prob"].round(3)
            array([[1.  , 0.01],
                   [1.  , 0.01]])
            >>> state["gamma"]
            array([[1., 0.],
                   [1., 0.]])
        """
        offset = context.lag_offset
        block = standardized[offset : offset + context.k_endog * context.order]
        tau0, tau1 = scales["tau0"], scales["tau1"]
        gap = (
            np.log(self.inclusion / (1.0 - self.inclusion))
            + np.log(tau0 / tau1)
            + block**2 / 2.0 * (1.0 / tau0**2 - 1.0 / tau1**2)
        )
        prob = 1.0 / (1.0 + np.exp(-np.clip(gap, -700.0, 700.0)))
        gamma = (np.asarray(rng.random(block.shape), dtype=np.float64) < prob).astype(np.float64)
        return {"tau0": tau0, "tau1": tau1, "gamma": gamma, "prob": prob}

    def _scale_variance(
        self, scales: dict[str, npt.NDArray[np.float64]], context: _PriorContext
    ) -> npt.NDArray[np.float64]:
        r"""Spike or slab variance per the current indicators, loose elsewhere.

        The raw-coefficient variance the current state implies: each lag
        coefficient carries :math:`(s_i / s_j)^2 \tau_1^2` when its
        indicator is one and :math:`(s_i / s_j)^2 \tau_0^2` otherwise, and
        every deterministic and exogenous row carries
        :math:`(\text{deterministic\_scale} \cdot s_i)^2`. This is the
        variance the coefficient block is drawn under on the next sweep.

        Args:
            scales: The current latent state.
            context: The sample description; ``scales``, ``width``,
                ``k_endog``, ``order``, and ``lag_offset`` are read.

        Returns:
            A ``(width, k)`` array of positive variances in design-column
            order.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> prior = SpikeAndSlabPrior()
            >>> state = prior._initial_scales(ctx, np.full((3, 2), 0.05))
            >>> state["gamma"][0, 1] = 0.0
            >>> prior._scale_variance(state, ctx)
            array([[1.0e+02, 1.0e+02],
                   [2.5e-01, 2.5e-05],
                   [2.5e-01, 2.5e-01]])
        """
        ratio = self._unit_ratio(context)
        out = (self.deterministic_scale * ratio) ** 2
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        chosen = np.where(scales["gamma"] > 0.5, scales["tau1"] ** 2, scales["tau0"] ** 2)
        out[offset:stop] = ratio[offset:stop] ** 2 * chosen
        return out

    def _tracked(
        self, scales: dict[str, npt.NDArray[np.float64]], context: _PriorContext
    ) -> npt.NDArray[np.float64]:
        """The Rao-Blackwellized inclusion probability per lag coefficient.

        The conditional probability from the last sweep rather than the
        binary indicator: averaging the exact conditional over sweeps
        estimates the marginal inclusion probability with less Monte Carlo
        error than averaging zero-one draws. The sampler accumulates this
        into the result's ``shrinkage``; deterministic and exogenous rows
        are reported as zero because they are never candidates.

        Args:
            scales: The current latent state.
            context: The sample description; ``width``, ``k_endog``,
                ``order``, and ``lag_offset`` are read.

        Returns:
            A ``(width, k)`` array in ``[0, 1]``, zero off the lag block.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> prior = SpikeAndSlabPrior(inclusion=0.2)
            >>> state = prior._initial_scales(ctx, np.full((3, 2), 0.05))
            >>> prior._tracked(state, ctx)
            array([[0. , 0. ],
                   [0.2, 0.2],
                   [0.2, 0.2]])
        """
        out = np.zeros((context.width, context.k_endog), dtype=np.float64)
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        out[offset:stop] = scales["prob"]
        return out

    def _tracked_label(self) -> str:
        """What the averaged diagnostic is.

        The result's ``shrinkage_label``; it contains ``"inclusion"``,
        which is how
        :meth:`~cultivars.multivariate.large_dim.gibbs.GibbsBVARResult.inclusion_probabilities`
        knows the diagnostic is a probability rather than a scale.

        Returns:
            The label.

        Example:
            >>> SpikeAndSlabPrior()._tracked_label()
            'posterior inclusion probability'
        """
        return "posterior inclusion probability"

    def _label(self) -> str:
        """Short description for summary tables and comparison rows.

        ``ssvs(spike=..., slab=..., p=...)`` with each value in ``%g``
        format; ``deterministic_scale`` is not part of the selection and
        is omitted.

        Returns:
            The label.

        Example:
            >>> SpikeAndSlabPrior()._label()
            'ssvs(spike=0.1, slab=10, p=0.5)'
            >>> SpikeAndSlabPrior(spike=0.05, inclusion=0.2)._label()
            'ssvs(spike=0.05, slab=10, p=0.2)'
        """
        return f"ssvs(spike={self.spike:g}, slab={self.slab:g}, p={self.inclusion:g})"


@dataclass(frozen=True, kw_only=True, slots=True)
class DirichletLaplacePrior(_AdaptivePrior):
    r"""Dirichlet-Laplace shrinkage: a simplex rations the prior's attention.

    Bhattacharya, Pati, Pillai and Dunson's (2015) global-local prior. Each
    standardized lag coefficient is Laplace-distributed with its own scale,
    and the scales are a global magnitude times a point on the simplex,

    .. math::

       \beta_{ij} \mid \phi_{ij}, \tau \sim \mathrm{Laplace}(0,\, \phi_{ij} \tau),
       \qquad
       \phi \sim \mathrm{Dirichlet}(a, \ldots, a),
       \qquad
       \tau \sim \mathrm{Gamma}(m a,\, 1/2),

    with :math:`m` the number of lag coefficients and :math:`\beta_{ij}`
    the coefficient divided by the Minnesota unit ratio :math:`s_i / s_j`.
    The simplex is the substance. The local scales must sum to
    :math:`\tau`, so a small concentration :math:`a` makes the Dirichlet
    spiky and the prior can only pay attention to a few coefficients at
    once: a budget constraint the horseshoe, whose local scales are
    independent, does not impose. Among the continuous shrinkage priors it
    carries the strongest theoretical warrant, posterior contraction at the
    minimax rate for sparse means when the concentration is set near
    :math:`1/m`; the paper's working default :math:`a = 1/2` trades some
    of that sparsity for a gentler prior on dense signals.

    Sampling writes the Laplace as a scale mixture of normals,
    :math:`\beta_{ij} \mid \psi_{ij} \sim \mathcal{N}(0, \psi_{ij} \phi_{ij}^2 \tau^2)`
    with :math:`\psi_{ij} \sim \mathrm{Exp}(1/2)`, after which every
    conditional is exact: inverse-Gaussian for the mixing scales,
    generalized-inverse-Gaussian for the global magnitude and, after
    normalization, for the simplex. The deterministic and exogenous
    columns are not shrunk; they sit at a fixed loose variance.

    Attributes:
        concentration: The Dirichlet concentration :math:`a`; ``0.5`` is
            the paper's default, and values near ``1 / (k**2 * order)``
            are its theory's.
        deterministic_scale: Looseness of the unshrunk deterministic and
            exogenous rows, in units of each equation's residual scale.

    Raises:
        SpecificationError: If ``concentration`` or ``deterministic_scale``
            is not positive, at the first moment call, which is the
            model's fit.

    Note:
        The prior mean is zero everywhere, without a persistence
        hyperparameter: centring a sparsity prior away from zero would
        change what shrinking a coefficient means. Difference the data, or
        use the Minnesota family, for random-walk centring. Adaptive priors
        do not compose, and the Chib marginal likelihood is not offered
        under one; compare by predictive score. The local-global scale
        this prior reports is :math:`\sqrt{\psi_{ij}}\, \phi_{ij} \tau`,
        the prior standard deviation of the standardized coefficient, on
        the same footing as the horseshoe's :math:`\tau \lambda_{ij}`.

    Warning:
        Only :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR`
        samples the hierarchy. The static moments interface reports the
        variance at the initial scales, :math:`1/m^2` on the standardized
        lag block, which is what admissibility checks need and not the
        Dirichlet-Laplace; a model that reads only the static moments is
        fitting a fixed Gaussian prior under this class's label.

    See Also:
        * :meth:`~cultivars.multivariate.large_dim.gibbs.GibbsBVARResult.shrinkage_scales`
          -- the posterior mean local-global scale per lag coefficient,
          this prior's diagnostic.
        * :class:`HorseshoePrior` -- independent half-Cauchy local scales,
          no attention budget.
        * :class:`NormalGammaPrior` -- Gamma local variances with a tunable
          shape, the other GIG-conditional hierarchy.
        * :class:`SpikeAndSlabPrior` -- selection rather than shrinkage.

    References:
        Bhattacharya, A., Pati, D., Pillai, N. S., & Dunson, D. B. (2015).
        Dirichlet-Laplace priors for optimal shrinkage. *Journal of the
        American Statistical Association*, 110(512), 1479-1490.

        Polson, N. G., & Scott, J. G. (2010). Shrink globally, act locally:
        Sparse Bayesian regularization and prediction. In *Bayesian
        Statistics 9* (pp. 501-538). Oxford University Press.

    Example:
        Three series of which only the first has an own lag. The
        local-global scale is above one on the signal and a fraction of
        that on every noise coefficient:

        >>> import numpy as np
        >>> from cultivars.multivariate.large_dim.gibbs import GibbsBVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((200, 3))
        >>> for t in range(1, 200):
        ...     y[t] = np.array([0.8, 0.0, 0.0]) * y[t - 1] + rng.standard_normal(3)
        >>> res = GibbsBVAR(y, order=1, prior=DirichletLaplacePrior()).fit(
        ...     n_draws=1500, n_burn=500, seed=0
        ... )
        >>> res.prior_label
        'dl(a=0.5)'
        >>> scales = res.shrinkage_scales()[0]
        >>> bool(scales[0, 0] > 1.0), bool(np.delete(scales, 0).max() < 0.6)
        (True, True)
        >>> np.abs(res.coefficients[0]).round(1)
        array([[0.7, 0. , 0. ],
               [0.1, 0. , 0.1],
               [0. , 0. , 0. ]])
    """

    concentration: float = 0.5
    r"""The Dirichlet concentration :math:`a` shared by every simplex coordinate.

    Below one the Dirichlet piles mass on the faces of the simplex, so
    most local scales are near zero and a few carry the magnitude; the
    smaller the value the fewer. One half is the paper's default. The
    minimax-rate results hold for :math:`a` near :math:`1/m` with
    :math:`m = k^2 p` lag coefficients. Must be positive.
    """
    deterministic_scale: float = 10.0
    """Prior standard deviation of the unshrunk rows, in residual-scale units.

    The intercept, any trend, and any exogenous regressor of equation
    :math:`i` get variance :math:`(\\text{deterministic\\_scale} \\cdot s_i)^2`,
    held fixed across sweeps. Must be positive.
    """

    def _check(self) -> None:
        """Reject hyperparameters that do not describe a prior.

        Called from :meth:`_initial_scales`, so the error surfaces at fit
        rather than at construction, matching the Minnesota family.

        Raises:
            SpecificationError: If the concentration or the deterministic
                scale is not positive.

        Example:
            >>> DirichletLaplacePrior(concentration=0.0)._check()
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: concentration must be positive; got 0.0.
        """
        if self.concentration <= 0.0:
            raise SpecificationError(f"concentration must be positive; got {self.concentration}.")
        if self.deterministic_scale <= 0.0:
            raise SpecificationError(
                f"deterministic_scale must be positive; got {self.deterministic_scale}."
            )

    def _initial_scales(
        self, context: _PriorContext, reference: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
        r"""Uniform simplex, unit local scales, unit global magnitude.

        Every :math:`\phi_{ij}` starts at :math:`1/m`, the centre of the
        simplex, with :math:`\psi_{ij} = 1` and :math:`\tau = 1`, so the
        first coefficient draw is under a Gaussian of standard deviation
        :math:`1/m` on the standardized scale: tight, and the first sweeps
        redistribute the budget from there.

        Args:
            context: The sample description; ``k_endog`` and ``order`` are
                read.
            reference: Data-driven reference scales, ignored: the
                Dirichlet-Laplace does not anchor on least-squares standard
                errors.

        Returns:
            ``{"psi", "phi"}`` of shape ``(k * p, k)`` and ``{"tau"}`` of
            shape ``(1,)``; ``phi`` sums to one.

        Raises:
            SpecificationError: If a hyperparameter is out of range.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> prior = DirichletLaplacePrior()
            >>> state = prior._initial_scales(ctx, prior._unit_ratio(ctx))
            >>> state["phi"]
            array([[0.25, 0.25],
                   [0.25, 0.25]])
            >>> state["psi"].sum(), state["tau"]
            (np.float64(4.0), array([1.]))
        """
        self._check()
        shape = (context.k_endog * context.order, context.k_endog)
        count = shape[0] * shape[1]
        return {
            "psi": np.ones(shape, dtype=np.float64),
            "phi": np.full(shape, 1.0 / count, dtype=np.float64),
            "tau": np.ones(1, dtype=np.float64),
        }

    def _draw_scales(
        self,
        standardized: npt.NDArray[np.float64],
        context: _PriorContext,
        rng: np.random.Generator,
        scales: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.float64]]:
        r"""The exact conditionals of Bhattacharya et al. (2015).

        With :math:`m` lag coefficients :math:`\beta_{ij}` on the
        standardized scale and :math:`\mathrm{GIG}(p, a, b)` the density
        :math:`x^{p-1} \exp\{-(a x + b / x) / 2\}`, the sweep is

        .. math::

           \psi_{ij}^{-1} \mid \cdot &\sim
             \mathrm{InvGaussian}\bigl(\phi_{ij} \tau / |\beta_{ij}|,\; 1\bigr), \\
           \tau \mid \cdot &\sim
             \mathrm{GIG}\Bigl(m(a - 1),\; 1,\;
             2 \sum_{ij} |\beta_{ij}| / \phi_{ij}\Bigr), \\
           T_{ij} \mid \cdot &\sim \mathrm{GIG}(a - 1,\; 1,\; 2|\beta_{ij}|),
           \qquad \phi_{ij} = T_{ij} \Big/ \sum_{kl} T_{kl},

        the last line being the simplex conditional through independent
        GIG draws normalized to sum to one. The mixing scales are drawn
        under the *incoming* :math:`\phi` and :math:`\tau`, then the
        global magnitude, then the simplex. Three guards keep the
        conditionals finite: :math:`|\beta_{ij}|` is floored at
        :math:`10^{-10}`, the inverse-Gaussian mean is capped at
        :math:`10^8`, and the inverse-Gaussian draw is floored at
        :math:`10^{-300}` before inversion. The deterministic and
        exogenous rows of ``standardized`` are not read.

        Args:
            standardized: ``(width, k)`` current coefficients divided by the
                unit ratio.
            context: The sample description; ``k_endog``, ``order``, and
                ``lag_offset`` are read.
            rng: Random generator.
            scales: The current latent state.

        Returns:
            The refreshed state under the same three names.

        Example:
            Holding one large and three near-zero standardized coefficients
            fixed and sweeping the scales alone, the averaged simplex puts
            most of its budget on the large one and the averaged
            local-global scale separates it by an order of magnitude:

            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> prior = DirichletLaplacePrior()
            >>> rng = np.random.default_rng(0)
            >>> beta = np.array([[0.0, 0.0], [3.0, 0.05], [-0.05, 0.02]])
            >>> state = prior._initial_scales(ctx, prior._unit_ratio(ctx))
            >>> total, simplex = np.zeros((3, 2)), np.zeros((2, 2))
            >>> for _ in range(2000):
            ...     state = prior._draw_scales(beta, ctx, rng, state)
            ...     total += prior._tracked(state, ctx)
            ...     simplex += state["phi"]
            >>> (total / 2000).round(1)
            array([[0. , 0. ],
                   [3.8, 0.4],
                   [0.4, 0.2]])
            >>> bool(simplex[0, 0] / 2000 > 0.7), (simplex / 2000).sum().round(6)
            (True, np.float64(1.0))
        """
        offset = context.lag_offset
        block = standardized[offset : offset + context.k_endog * context.order]
        magnitude = np.maximum(np.abs(block), 1e-10)
        count = block.size
        a = self.concentration
        phi, tau = scales["phi"], float(scales["tau"][0])
        mean = np.minimum(phi * tau / magnitude, 1e8)
        psi = 1.0 / np.maximum(np.asarray(rng.wald(mean, 1.0), dtype=np.float64), 1e-300)
        tau = float(
            _draw_generalized_inverse_gaussian(
                count * (a - 1.0),
                1.0,
                2.0 * float(np.sum(magnitude / phi)),
                rng,
            )
        )
        raw = _draw_generalized_inverse_gaussian(a - 1.0, 1.0, 2.0 * magnitude, rng)
        phi = raw / float(np.sum(raw))
        return {"psi": psi, "phi": phi, "tau": np.array([tau])}

    def _scale_variance(
        self, scales: dict[str, npt.NDArray[np.float64]], context: _PriorContext
    ) -> npt.NDArray[np.float64]:
        r"""``ratio**2 * psi * (phi * tau)**2`` on the lag block, loose elsewhere.

        The raw-coefficient variance the current state implies: the lag
        block carries :math:`(s_i / s_j)^2 \psi_{ij} \phi_{ij}^2 \tau^2`,
        the normal-mixture representation of the Laplace at the current
        mixing scale, and every deterministic and exogenous row carries
        :math:`(\text{deterministic\_scale} \cdot s_i)^2`. At the initial
        scales this is what :meth:`coefficient_variance` reports.

        Args:
            scales: The current latent state.
            context: The sample description; ``scales``, ``width``,
                ``k_endog``, ``order``, and ``lag_offset`` are read.

        Returns:
            A ``(width, k)`` array of positive variances in design-column
            order.

        Example:
            At the initial scales with four lag coefficients, the lag
            block is the squared unit ratio over sixteen:

            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.array([1.0, 4.0]),
            ...     presample_mean=np.zeros(2),
            ... )
            >>> prior = DirichletLaplacePrior()
            >>> state = prior._initial_scales(ctx, prior._unit_ratio(ctx))
            >>> prior._scale_variance(state, ctx)
            array([[1.00000e+02, 1.60000e+03],
                   [6.25000e-02, 1.00000e+00],
                   [3.90625e-03, 6.25000e-02]])
        """
        ratio = self._unit_ratio(context)
        out = (self.deterministic_scale * ratio) ** 2
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        out[offset:stop] = (
            ratio[offset:stop] ** 2 * scales["psi"] * (scales["phi"] * float(scales["tau"][0])) ** 2
        )
        return out

    def _tracked(
        self, scales: dict[str, npt.NDArray[np.float64]], context: _PriorContext
    ) -> npt.NDArray[np.float64]:
        r"""The local-global scale per lag coefficient.

        :math:`\sqrt{\psi_{ij}}\, \phi_{ij} \tau`, the prior standard
        deviation of the standardized coefficient under the current state:
        near zero means the coefficient is being shrunk away, order one or
        larger means it is left free. The sampler averages this over kept
        sweeps into the result's ``shrinkage``; deterministic and exogenous
        rows are reported as zero because they are not shrunk.

        Args:
            scales: The current latent state.
            context: The sample description; ``width``, ``k_endog``,
                ``order``, and ``lag_offset`` are read.

        Returns:
            A ``(width, k)`` array, zero off the lag block.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> prior = DirichletLaplacePrior()
            >>> state = prior._initial_scales(ctx, prior._unit_ratio(ctx))
            >>> state["tau"][0] = 2.0
            >>> prior._tracked(state, ctx)
            array([[0. , 0. ],
                   [0.5, 0.5],
                   [0.5, 0.5]])
        """
        out = np.zeros((context.width, context.k_endog), dtype=np.float64)
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        out[offset:stop] = np.sqrt(scales["psi"]) * scales["phi"] * float(scales["tau"][0])
        return out

    def _tracked_label(self) -> str:
        """What the averaged diagnostic is.

        The result's ``shrinkage_label``; it contains ``"scale"``, which is
        how :meth:`~cultivars.multivariate.large_dim.gibbs.GibbsBVARResult.shrinkage_scales`
        knows the diagnostic is a scale rather than an inclusion
        probability. The same label as the horseshoe's, because the two
        report the same quantity.

        Returns:
            The label.

        Example:
            >>> DirichletLaplacePrior()._tracked_label()
            'posterior mean local-global scale of the standardized coefficient'
        """
        return "posterior mean local-global scale of the standardized coefficient"

    def _label(self) -> str:
        """Short description for summary tables and comparison rows.

        ``dl(a=...)`` with the concentration in ``%g`` format;
        ``deterministic_scale`` is not part of the shrinkage and is
        omitted.

        Returns:
            The label.

        Example:
            >>> DirichletLaplacePrior()._label()
            'dl(a=0.5)'
            >>> DirichletLaplacePrior(concentration=1 / 9)._label()
            'dl(a=0.111111)'
        """
        return f"dl(a={self.concentration:g})"


@dataclass(frozen=True, kw_only=True, slots=True)
class NormalGammaPrior(_AdaptivePrior):
    r"""Normal-Gamma shrinkage: the Bayesian lasso's adjustable-kurtosis parent.

    Griffin and Brown's (2010) prior. Each standardized lag coefficient is
    normal with its own variance, and the variances are Gamma,

    .. math::

       \beta_{ij} \mid \psi_{ij} \sim \mathcal{N}(0, \psi_{ij}),
       \qquad
       \psi_{ij} \sim \mathrm{Gamma}(\lambda,\, \gamma),
       \qquad
       \gamma \sim \mathrm{Gamma}(2,\, 1),

    with :math:`\lambda` the shape, :math:`\gamma` the rate, and
    :math:`\beta_{ij}` the coefficient divided by the Minnesota unit ratio
    :math:`s_i / s_j`. The marginal of :math:`\beta_{ij}` has variance
    :math:`\lambda / \gamma` and excess kurtosis :math:`3 / \lambda`, and
    that second fact is the point. At :math:`\lambda = 1` the marginal is
    exactly the Laplace of the Bayesian lasso; pushing the shape below one
    puts more mass at zero and fattens the tails at the same time, which is
    the knob the lasso lacks: the lasso can only trade sparsity against
    over-shrinking large coefficients, and a small shape buys both. The
    rate is the global tightness, and its own Gamma hyperprior means
    overall shrinkage is learned rather than tuned; the default shape of a
    tenth is well into the sparse regime.

    The variance conditionals are generalized-inverse-Gaussian and the
    rate's is Gamma, all drawn exactly. The deterministic and exogenous
    columns are not shrunk; they sit at a fixed loose variance.

    Attributes:
        shape: The Gamma shape :math:`\lambda` of the local variances;
            ``1`` is the Bayesian lasso, smaller is spikier with fatter
            tails.
        deterministic_scale: Looseness of the unshrunk deterministic and
            exogenous rows, in units of each equation's residual scale.

    Raises:
        SpecificationError: If ``shape`` or ``deterministic_scale`` is not
            positive, at the first moment call, which is the model's fit.

    Note:
        The prior mean is zero everywhere, without a persistence
        hyperparameter: centring a sparsity prior away from zero would
        change what shrinking a coefficient means. Difference the data, or
        use the Minnesota family, for random-walk centring. Adaptive priors
        do not compose, and the Chib marginal likelihood is not offered
        under one; compare by predictive score. The scale this prior
        reports is :math:`\sqrt{\psi_{ij}}`, the prior standard deviation
        of the standardized coefficient; the global rate is folded into it
        rather than reported separately, so it reads like the horseshoe's
        :math:`\tau \lambda_{ij}`.

    Warning:
        Only :class:`~cultivars.multivariate.large_dim.gibbs.GibbsBVAR`
        samples the hierarchy. The static moments interface reports the
        variance at the initial scales, ``0.04`` on the standardized lag
        block, which is what admissibility checks need and not the
        Normal-Gamma; a model that reads only the static moments is fitting
        a fixed Gaussian prior under this class's label.

    See Also:
        * :meth:`~cultivars.multivariate.large_dim.gibbs.GibbsBVARResult.shrinkage_scales`
          -- the posterior mean local scale per lag coefficient, this
          prior's diagnostic.
        * :class:`HorseshoePrior` -- half-Cauchy scales, no shape to set,
          heavier tails than any finite ``shape`` gives.
        * :class:`DirichletLaplacePrior` -- the other GIG-conditional
          hierarchy, with a simplex budget across coefficients.
        * :class:`SpikeAndSlabPrior` -- selection rather than shrinkage.

    References:
        Griffin, J. E., & Brown, P. J. (2010). Inference with normal-gamma
        prior distributions in regression problems. *Bayesian Analysis*,
        5(1), 171-188.

        Park, T., & Casella, G. (2008). The Bayesian lasso. *Journal of the
        American Statistical Association*, 103(482), 681-686.

        Huber, F., & Feldkircher, M. (2019). Adaptive shrinkage in Bayesian
        vector autoregressive models. *Journal of Business & Economic
        Statistics*, 37(1), 27-39.

    Example:
        Three series of which only the first has an own lag. The local
        scale is an order of magnitude larger on the signal than on any
        noise coefficient, and the noise coefficients are shrunk harder
        than the horseshoe shrinks them, the small shape at work:

        >>> import numpy as np
        >>> from cultivars.multivariate.large_dim.gibbs import GibbsBVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((200, 3))
        >>> for t in range(1, 200):
        ...     y[t] = np.array([0.8, 0.0, 0.0]) * y[t - 1] + rng.standard_normal(3)
        >>> res = GibbsBVAR(y, order=1, prior=NormalGammaPrior()).fit(
        ...     n_draws=1500, n_burn=500, seed=0
        ... )
        >>> res.prior_label
        'ng(theta=0.1)'
        >>> scales = res.shrinkage_scales()[0]
        >>> bool(scales[0, 0] > 0.5), bool(np.delete(scales, 0).max() < 0.2)
        (True, True)
        >>> np.abs(res.coefficients[0]).round(1)
        array([[0.7, 0. , 0. ],
               [0. , 0. , 0.1],
               [0. , 0. , 0. ]])
    """

    shape: float = 0.1
    r"""The Gamma shape :math:`\lambda` of every local variance.

    Sets the marginal's excess kurtosis, :math:`3 / \lambda`: one is the
    Laplace of the Bayesian lasso, and smaller values pile more mass at
    zero while fattening the tails. Below one half the variance
    conditional at an exact zero is improper, which :meth:`_draw_scales`
    guards with a floor. Written ``theta`` in the label. Must be
    positive.
    """
    deterministic_scale: float = 10.0
    """Prior standard deviation of the unshrunk rows, in residual-scale units.

    The intercept, any trend, and any exogenous regressor of equation
    :math:`i` get variance :math:`(\\text{deterministic\\_scale} \\cdot s_i)^2`,
    held fixed across sweeps. Must be positive.
    """

    def _check(self) -> None:
        """Reject hyperparameters that do not describe a prior.

        Called from :meth:`_initial_scales`, so the error surfaces at fit
        rather than at construction, matching the Minnesota family.

        Raises:
            SpecificationError: If the shape or the deterministic scale is
                not positive.

        Example:
            >>> NormalGammaPrior(shape=0.0)._check()
            Traceback (most recent call last):
                ...
            cultivars.exceptions.SpecificationError: shape must be positive; got 0.0.
        """
        if self.shape <= 0.0:
            raise SpecificationError(f"shape must be positive; got {self.shape}.")
        if self.deterministic_scale <= 0.0:
            raise SpecificationError(
                f"deterministic_scale must be positive; got {self.deterministic_scale}."
            )

    def _initial_scales(
        self, context: _PriorContext, reference: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
        r"""Minnesota-tightness local variances; the rate that implies them.

        Every :math:`\psi_{ij}` starts at :math:`0.04`, the square of the
        Minnesota default tightness :math:`\lambda_1 = 0.2`, so the first
        coefficient draw is under the standardized Minnesota prior, and
        the rate starts at :math:`\lambda / 0.04`, the value whose Gamma
        has that as its prior mean, so the first rate draw is not fighting
        the starting variances.

        Args:
            context: The sample description; ``k_endog`` and ``order`` are
                read.
            reference: Data-driven reference scales, ignored: the
                Normal-Gamma does not anchor on least-squares standard
                errors.

        Returns:
            ``{"psi"}`` of shape ``(k * p, k)`` and ``{"rate"}`` of shape
            ``(1,)``.

        Raises:
            SpecificationError: If a hyperparameter is out of range.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> prior = NormalGammaPrior()
            >>> state = prior._initial_scales(ctx, prior._unit_ratio(ctx))
            >>> state["psi"]
            array([[0.04, 0.04],
                   [0.04, 0.04]])
            >>> state["rate"]
            array([2.5])
        """
        self._check()
        shape = (context.k_endog * context.order, context.k_endog)
        start = 0.04
        return {
            "psi": np.full(shape, start, dtype=np.float64),
            "rate": np.array([self.shape / start]),
        }

    def _draw_scales(
        self,
        standardized: npt.NDArray[np.float64],
        context: _PriorContext,
        rng: np.random.Generator,
        scales: dict[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.float64]]:
        r"""GIG conditionals for the variances, Gamma for the global rate.

        With :math:`m` lag coefficients :math:`\beta_{ij}` on the
        standardized scale and :math:`\mathrm{GIG}(p, a, b)` the density
        :math:`x^{p-1} \exp\{-(a x + b / x) / 2\}`, the sweep is

        .. math::

           \psi_{ij} \mid \cdot &\sim
             \mathrm{GIG}\bigl(\lambda - \tfrac{1}{2},\; 2\gamma,\; \beta_{ij}^2\bigr), \\
           \gamma \mid \cdot &\sim
             \mathrm{Gamma}\Bigl(2 + m\lambda,\; 1 + \sum_{ij} \psi_{ij}\Bigr),

        the second in shape-rate form, the variances drawn under the
        incoming rate and the rate then refreshed from the new variances.
        Squared deviations are floored at :math:`10^{-11}`: at an exact
        zero the variance conditional is improper for
        :math:`\lambda < 1/2`, a known boundary of the hierarchy, and the
        floor, three parts per million on the standardized scale,
        restores it without moving anything statistically visible. The
        deterministic and exogenous rows of ``standardized`` are not read.

        Args:
            standardized: ``(width, k)`` current coefficients divided by the
                unit ratio.
            context: The sample description; ``k_endog``, ``order``, and
                ``lag_offset`` are read.
            rng: Random generator.
            scales: The current latent state.

        Returns:
            The refreshed state under the same two names.

        Example:
            Holding one large and three near-zero standardized coefficients
            fixed and sweeping the scales alone, the averaged local scale
            separates them by an order of magnitude:

            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> prior = NormalGammaPrior()
            >>> rng = np.random.default_rng(0)
            >>> beta = np.array([[0.0, 0.0], [3.0, 0.05], [-0.05, 0.02]])
            >>> state = prior._initial_scales(ctx, prior._unit_ratio(ctx))
            >>> total = np.zeros((3, 2))
            >>> for _ in range(2000):
            ...     state = prior._draw_scales(beta, ctx, rng, state)
            ...     total += prior._tracked(state, ctx)
            >>> (total / 2000).round(1)
            array([[0. , 0. ],
                   [1.8, 0.2],
                   [0.2, 0.1]])
        """
        offset = context.lag_offset
        block = standardized[offset : offset + context.k_endog * context.order]
        count = block.size
        rate = float(scales["rate"][0])
        squared = np.maximum(block**2, 1e-11)
        psi = _draw_generalized_inverse_gaussian(self.shape - 0.5, 2.0 * rate, squared, rng)
        rate = float(rng.gamma(2.0 + count * self.shape, 1.0 / (1.0 + float(np.sum(psi)))))
        return {"psi": psi, "rate": np.array([rate])}

    def _scale_variance(
        self, scales: dict[str, npt.NDArray[np.float64]], context: _PriorContext
    ) -> npt.NDArray[np.float64]:
        r"""``ratio**2 * psi`` on the lag block, loose elsewhere.

        The raw-coefficient variance the current state implies: the lag
        block carries :math:`(s_i / s_j)^2 \psi_{ij}`, and every
        deterministic and exogenous row carries
        :math:`(\text{deterministic\_scale} \cdot s_i)^2`. At the initial
        scales this is what :meth:`coefficient_variance` reports.

        Args:
            scales: The current latent state.
            context: The sample description; ``scales``, ``width``,
                ``k_endog``, ``order``, and ``lag_offset`` are read.

        Returns:
            A ``(width, k)`` array of positive variances in design-column
            order.

        Example:
            At the initial scales with residual scales one and four, the
            lag block is ``0.04`` times the squared unit ratio:

            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.array([1.0, 4.0]),
            ...     presample_mean=np.zeros(2),
            ... )
            >>> prior = NormalGammaPrior()
            >>> state = prior._initial_scales(ctx, prior._unit_ratio(ctx))
            >>> prior._scale_variance(state, ctx)
            array([[1.0e+02, 1.6e+03],
                   [4.0e-02, 6.4e-01],
                   [2.5e-03, 4.0e-02]])
        """
        ratio = self._unit_ratio(context)
        out = (self.deterministic_scale * ratio) ** 2
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        out[offset:stop] = ratio[offset:stop] ** 2 * scales["psi"]
        return out

    def _tracked(
        self, scales: dict[str, npt.NDArray[np.float64]], context: _PriorContext
    ) -> npt.NDArray[np.float64]:
        r"""The local scale per lag coefficient.

        :math:`\sqrt{\psi_{ij}}`, the prior standard deviation of the
        standardized coefficient under the current state, with the global
        rate already folded in through the variance conditional: near zero
        means the coefficient is being shrunk away, order one or larger
        means it is left free. The sampler averages this over kept sweeps
        into the result's ``shrinkage``; deterministic and exogenous rows
        are reported as zero because they are not shrunk.

        Args:
            scales: The current latent state.
            context: The sample description; ``width``, ``k_endog``,
                ``order``, and ``lag_offset`` are read.

        Returns:
            A ``(width, k)`` array, zero off the lag block.

        Example:
            >>> import numpy as np
            >>> from cultivars._internals import _PriorContext
            >>> ctx = _PriorContext(
            ...     k_endog=2, order=1, scales=np.ones(2), presample_mean=np.zeros(2)
            ... )
            >>> prior = NormalGammaPrior()
            >>> state = prior._initial_scales(ctx, prior._unit_ratio(ctx))
            >>> prior._tracked(state, ctx)
            array([[0. , 0. ],
                   [0.2, 0.2],
                   [0.2, 0.2]])
        """
        out = np.zeros((context.width, context.k_endog), dtype=np.float64)
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        out[offset:stop] = np.sqrt(scales["psi"])
        return out

    def _tracked_label(self) -> str:
        """What the averaged diagnostic is.

        The result's ``shrinkage_label``; it contains ``"scale"``, which is
        how :meth:`~cultivars.multivariate.large_dim.gibbs.GibbsBVARResult.shrinkage_scales`
        knows the diagnostic is a scale rather than an inclusion
        probability. The same label as the horseshoe's and the
        Dirichlet-Laplace's, because all three report the prior standard
        deviation of the standardized coefficient.

        Returns:
            The label.

        Example:
            >>> NormalGammaPrior()._tracked_label()
            'posterior mean local-global scale of the standardized coefficient'
        """
        return "posterior mean local-global scale of the standardized coefficient"

    def _label(self) -> str:
        """Short description for summary tables and comparison rows.

        ``ng(theta=...)`` with the shape in ``%g`` format;
        ``deterministic_scale`` is not part of the shrinkage and is
        omitted.

        Returns:
            The label.

        Example:
            >>> NormalGammaPrior()._label()
            'ng(theta=0.1)'
            >>> NormalGammaPrior(shape=1.0)._label()
            'ng(theta=1)'
        """
        return f"ng(theta={self.shape:g})"
