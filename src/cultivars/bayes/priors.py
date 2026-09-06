"""Cultivars Bayesian priors module."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import _draw_generalized_inverse_gaussian
from .._internals import _AdaptivePrior, _Prior, _PriorContext
from .._internals import _NoPrior as NoPrior
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
    "SpikeAndSlabPrior",
    "SumOfCoefficientsPrior",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class MinnesotaPrior(_Prior):
    """Litterman's prior: shrink toward independent random walks.

    Five hyperparameters, and each one answers a different question.

    ``tightness`` is how much the prior is believed at all; it scales every
    variance and is the one people tune. ``cross_equation`` says how much less
    a variable's dependence on *other* variables is believed than its
    dependence on itself, which is the prior's central economic claim -- most
    of what a series does is explained by its own past -- and is the
    hyperparameter that a dummy-observation implementation cannot express.
    ``decay`` tightens longer lags toward zero, encoding that distant history
    matters less. ``exogenous`` loosens the intercept and any exogenous block,
    and defaults large: those coefficients carry levels and slopes that nobody
    means to shrink, and a tight default there is a silent and expensive
    mistake. ``sum_of_coefficients`` is the only one that cannot be a variance
    -- it restricts a *sum* of coefficients, so it enters as artificial rows.

    Prior variances, for equation ``i`` and variable ``j`` at lag ``l``:

    ==================  ==================================================
    own lag             ``(tightness / l ** decay) ** 2``
    cross lag           ``(tightness * cross_equation * s_i /
                        (l ** decay * s_j)) ** 2``
    intercept, exog     ``(tightness * exogenous * s_i) ** 2``
    ==================  ==================================================

    The scale ratio is what makes variables measured in different units
    comparable, so that shrinkage is a statement about dynamics rather than
    about whether a series is quoted in percent or in levels.

    With ``cross_equation`` other than one the prior variance no longer factors
    as a Kronecker product, so there is no Normal-inverse-Wishart closed form
    and estimation runs equation by equation conditional on the scales. That is
    Litterman's original procedure and it is correct; it is also why this prior
    and a conjugate one are different estimation paths rather than the same
    object with different numbers.

    Attributes:
        tightness: Overall confidence, ``lambda_1``. Values near 0.1 to 0.3 are
            usual for macroeconomic data; large recovers least squares.
        cross_equation: How much harder cross-variable coefficients shrink,
            ``lambda_2``. One treats them like own lags; smaller is tighter.
        decay: Lag decay, ``lambda_3``.
        exogenous: Looseness of the intercept and exogenous block,
            ``lambda_4``. Large is flat.
        sum_of_coefficients: Confidence that the variables sit at their
            pre-sample means forever, ``lambda_5``. ``None`` omits the
            restriction; larger imposes it harder.
        persistence: Prior mean of each variable's own first lag. One is the
            random-walk prior for levels; zero suits differenced data. Getting
            this wrong shrinks toward the wrong place, so tightening makes the
            estimate worse rather than better.
    """

    tightness: float = 0.2
    cross_equation: float = 0.5
    decay: float = 1.0
    exogenous: float = 100.0
    sum_of_coefficients: float | None = None
    persistence: float | Sequence[float] = 1.0

    def _persistence(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Broadcast the prior mean to one value per variable."""
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

        Raises:
            SpecificationError: If a tightness is not positive or the decay is
                negative.
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
        """Each variable's own first lag at ``persistence``, everything else zero."""
        self._check()
        means = self._persistence(context)
        out = np.zeros((context.width, context.k_endog), dtype=np.float64)
        offset = context.lag_offset
        for index in range(context.k_endog):
            out[offset + index, index] = means[index]
        return out

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Litterman's variances, tighter for cross terms and for longer lags.

        Three kinds of column, three rules. A variable's own lag gets
        ``(tightness / lag ** decay) ** 2``, which is the only rule with no
        scale in it, because a coefficient on a variable's own past is
        dimensionless. A cross lag gets the same thing multiplied by
        ``cross_equation`` and by ``s_i / s_j``, the ratio that makes a
        coefficient linking two variables measured in different units shrink by
        the same amount it would if they were measured in the same ones.
        Deterministic terms and exogenous regressors get
        ``(tightness * exogenous * s_i) ** 2``, which with the default
        ``exogenous`` is four orders of magnitude looser than an own lag and is
        meant to be.

        The deterministic block is a loop rather than a single row because its
        width is a property of the family, not a constant: none for a trendless
        specification, one for a constant, two for a constant and trend, and
        one per unit for a fixed-effects panel. Writing it as ``out[0]`` was
        silently one column short for every ``"ct"`` model and off by ``N - 1``
        for every panel -- a prior whose columns do not line up with the design
        shrinks the wrong coefficients and reports a number rather than
        raising.

        Args:
            context: What the prior needs to know about the sample.

        Returns:
            A ``(width, k)`` array of variances in design-column order.

        Raises:
            SpecificationError: If a tightness is not positive or the decay is
                negative.
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
        """The sum-of-coefficients rows, when that restriction is asked for.

        One row per variable, saying that a series sitting at its pre-sample
        average forever should stay there. That is a statement about the sum of
        a variable's lag coefficients, which no diagonal variance can make,
        which is why this hyperparameter alone enters as data.
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
        """Short description for a summary table."""
        tail = "" if self.sum_of_coefficients is None else f", l5={self.sum_of_coefficients:g}"
        return (
            f"minnesota(l1={self.tightness:g}, l2={self.cross_equation:g}, "
            f"l3={self.decay:g}, l4={self.exogenous:g}{tail})"
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class NormalInverseWishartPrior(_Prior):
    """The conjugate Minnesota prior: Litterman's variances with the one weight pinned.

    Exactly :class:`MinnesotaPrior` with ``cross_equation`` fixed at one --
    and that is not a simplification but a purchase. Pinning the weight makes
    the prior variance factor as ``Sigma x Omega``, which is the condition
    for Normal-inverse-Wishart conjugacy (Banbura, Giannone & Reichlin 2010):
    the joint posterior over coefficients *and* covariance is then exact, its
    marginal likelihood has a closed form, and hierarchical shrinkage a la
    Giannone-Lenza-Primiceri becomes optimization of that closed form rather
    than simulation. What is given up is Litterman's claim that cross-variable
    coefficients deserve extra shrinkage; a caller who wants that claim back
    passes a :class:`MinnesotaPrior` to the point-estimate path instead, and
    the Bayesian model refuses it by construction rather than approximating.

    Attributes:
        tightness: Overall confidence, ``lambda_1``.
        decay: Lag decay, ``lambda_3``.
        exogenous: Looseness of the intercept and exogenous block,
            ``lambda_4``. Large but finite: the marginal likelihood requires
            a proper prior, so "flat" is not on offer here.
        persistence: Prior mean of each variable's own first lag. One is the
            random-walk prior for levels; zero suits differenced data.
    """

    tightness: float = 0.2
    decay: float = 1.0
    exogenous: float = 100.0
    persistence: float | Sequence[float] = 1.0

    def _minnesota(self) -> MinnesotaPrior:
        """The equivalent Minnesota prior with the cross-equation weight pinned."""
        return MinnesotaPrior(
            tightness=self.tightness,
            cross_equation=1.0,
            decay=self.decay,
            exogenous=self.exogenous,
            sum_of_coefficients=None,
            persistence=self.persistence,
        )

    def coefficient_mean(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Each variable's own first lag at ``persistence``, everything else zero."""
        return self._minnesota().coefficient_mean(context)

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Litterman's variances with the cross-equation weight at one."""
        return self._minnesota().coefficient_variance(context)

    def _label(self) -> str:
        """Short description for a summary table."""
        return f"niw(l1={self.tightness:g}, l3={self.decay:g}, l4={self.exogenous:g})"


@dataclass(frozen=True, kw_only=True, slots=True)
class SumOfCoefficientsPrior(_Prior):
    """The no-cointegration dummies of Doan, Litterman & Sims.

    One artificial observation per variable, each saying that a series
    sitting at its pre-sample average forever should stay there -- a
    restriction on the *sum* of a variable's lag coefficients, which no
    diagonal variance can state, and which therefore enters as rows rather
    than as moments. At the limit it pushes every equation toward a unit
    root with no cross-variable error correction, which is why the
    literature reads it as a no-cointegration prior.

    This is the standalone, composable form of what
    :attr:`MinnesotaPrior.sum_of_coefficients` embeds: written separately it
    can be stacked onto the conjugate prior with ``+``, given its own
    tightness, and later carried into the hierarchical layer as its own
    hyperparameter, which is exactly how Giannone-Lenza-Primiceri treat it.

    Attributes:
        tightness: How hard the restriction binds -- larger is tighter,
            matching the package's ``lambda_5`` convention; the
            Giannone-Lenza-Primiceri ``mu`` is its reciprocal.
    """

    tightness: float = 1.0

    def _check(self) -> None:
        """Reject a non-positive tightness.

        Raises:
            SpecificationError: If ``tightness`` is not positive.
        """
        if self.tightness <= 0.0:
            raise SpecificationError(f"tightness must be positive; got {self.tightness}.")

    def coefficient_mean(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Zero: this prior's content is entirely in its rows."""
        return np.zeros((context.width, context.k_endog), dtype=np.float64)

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Infinite: no diagonal opinion about any coefficient."""
        return np.full((context.width, context.k_endog), np.inf, dtype=np.float64)

    def dummy_observations(
        self, context: _PriorContext
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """One row per variable, centred on its pre-sample mean."""
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
        """Short description for a summary table."""
        return f"soc({self.tightness:g})"


@dataclass(frozen=True, kw_only=True, slots=True)
class DummyInitialObservationPrior(_Prior):
    """Sims' co-persistence dummy, the single-unit-root prior.

    One artificial observation in which every variable sits at its
    pre-sample mean and the deterministic terms at one. Where the
    sum-of-coefficients rows allow each variable its own unit root, this
    single row says that if the system is that persistent, it is persistent
    *jointly* -- a common stochastic trend rather than one per series -- and
    it is what keeps the sum-of-coefficients restriction from ruling out
    cointegration entirely. The pair is standard equipment in the
    Giannone-Lenza-Primiceri hierarchy, each with its own tightness.

    Attributes:
        tightness: How hard the restriction binds -- larger is tighter; the
            Giannone-Lenza-Primiceri ``delta`` is its reciprocal.
    """

    tightness: float = 1.0

    def _check(self) -> None:
        """Reject a non-positive tightness.

        Raises:
            SpecificationError: If ``tightness`` is not positive.
        """
        if self.tightness <= 0.0:
            raise SpecificationError(f"tightness must be positive; got {self.tightness}.")

    def coefficient_mean(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Zero: this prior's content is entirely in its row."""
        return np.zeros((context.width, context.k_endog), dtype=np.float64)

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Infinite: no diagonal opinion about any coefficient."""
        return np.full((context.width, context.k_endog), np.inf, dtype=np.float64)

    def dummy_observations(
        self, context: _PriorContext
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """A single artificial observation at the pre-sample means."""
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
        """Short description for a summary table."""
        return f"dio({self.tightness:g})"


@dataclass(frozen=True, kw_only=True, slots=True)
class IndependentNormalWishartPrior(_Prior):
    """Litterman's full prior under an independent inverse-Wishart pairing.

    The variances are exactly :class:`MinnesotaPrior`'s, cross-equation
    weight included -- and that is the point. The conjugate
    Normal-inverse-Wishart pairing had to pin that weight to one because it
    ties the coefficient variance to ``Sigma`` through a Kronecker product;
    stating the prior on the coefficients *independently* of ``Sigma``
    drops the factorization requirement, at the price of the closed form:
    the posterior is reached by Gibbs sampling (coefficients given
    covariance by generalized least squares, covariance given coefficients
    by inverse-Wishart) rather than exactly, and there is no closed-form
    marginal likelihood to report. That trade -- Litterman's economics
    back, the evidence gone -- is the standard one (Koop & Korobilis 2010),
    and this class is its name.

    Attributes:
        tightness: Overall confidence, ``lambda_1``.
        cross_equation: How much harder cross-variable coefficients shrink,
            ``lambda_2`` -- the hyperparameter this pairing exists to keep.
        decay: Lag decay, ``lambda_3``.
        exogenous: Looseness of the intercept and exogenous block,
            ``lambda_4``.
        sum_of_coefficients: The no-cointegration restriction, ``lambda_5``;
            ``None`` omits it.
        persistence: Prior mean of each variable's own first lag.
    """

    tightness: float = 0.2
    cross_equation: float = 0.5
    decay: float = 1.0
    exogenous: float = 100.0
    sum_of_coefficients: float | None = None
    persistence: float | Sequence[float] = 1.0

    def _minnesota(self) -> MinnesotaPrior:
        """The Minnesota prior whose moments this prior states."""
        return MinnesotaPrior(
            tightness=self.tightness,
            cross_equation=self.cross_equation,
            decay=self.decay,
            exogenous=self.exogenous,
            sum_of_coefficients=self.sum_of_coefficients,
            persistence=self.persistence,
        )

    def coefficient_mean(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Each variable's own first lag at ``persistence``, everything else zero."""
        return self._minnesota().coefficient_mean(context)

    def coefficient_variance(self, context: _PriorContext) -> npt.NDArray[np.float64]:
        """Litterman's variances, the cross-equation weight kept."""
        return self._minnesota().coefficient_variance(context)

    def dummy_observations(
        self, context: _PriorContext
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The sum-of-coefficients rows, when that restriction is asked for."""
        return self._minnesota().dummy_observations(context)

    def _label(self) -> str:
        """Short description for a summary table."""
        tail = "" if self.sum_of_coefficients is None else f", l5={self.sum_of_coefficients:g}"
        return (
            f"inw(l1={self.tightness:g}, l2={self.cross_equation:g}, "
            f"l3={self.decay:g}, l4={self.exogenous:g}{tail})"
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class HorseshoePrior(_AdaptivePrior):
    """The horseshoe: aggressive shrinkage that lets large signals through.

    Each standardized coefficient gets a half-Cauchy local scale and the
    whole lag block shares a half-Cauchy global one (Carvalho, Polson &
    Scott 2010). The Cauchy tails are the substance: mass piles up at zero
    hard enough to wipe out noise coefficients, while the poles let a
    genuinely large coefficient escape shrinkage almost entirely -- the
    behavior the Minnesota family cannot produce, because a Gaussian
    variance shrinks everything proportionally. No hyperparameters to tune
    is the other selling point, and it is real: the global scale learns the
    overall sparsity from the data.

    Sampling uses the Makalic-Schmidt inverse-Gamma augmentation, under
    which every conditional in the hierarchy is inverse-Gamma -- exact
    draws, no tuning, no rejection.

    Attributes:
        deterministic_scale: Looseness of the unshrunk deterministic and
            exogenous rows, in units of each equation's residual scale.

    References:
        Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). The horseshoe
            estimator for sparse signals. *Biometrika*, 97(2), 465-480.
        Makalic, E., & Schmidt, D. F. (2016). A simple sampler for the
            horseshoe estimator. *IEEE Signal Processing Letters*, 23(1),
            179-182.
    """

    deterministic_scale: float = 10.0

    def _check(self) -> None:
        """Reject a non-positive deterministic scale.

        Raises:
            SpecificationError: If ``deterministic_scale`` is not positive.
        """
        if self.deterministic_scale <= 0.0:
            raise SpecificationError(
                f"deterministic_scale must be positive; got {self.deterministic_scale}."
            )

    def _initial_scales(
        self, context: _PriorContext, reference: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Unit local scales, unit global scale."""
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
        """Makalic-Schmidt: four inverse-Gamma blocks, all exact."""
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
        """``ratio**2 * tau**2 * lambda**2`` on the lag block, loose elsewhere."""
        ratio = self._unit_ratio(context)
        out = (self.deterministic_scale * ratio) ** 2
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        out[offset:stop] = ratio[offset:stop] ** 2 * float(scales["tau2"][0]) * scales["lam2"]
        return out

    def _tracked(
        self, scales: dict[str, npt.NDArray[np.float64]], context: _PriorContext
    ) -> npt.NDArray[np.float64]:
        """The local-global scale ``tau * lambda`` per lag coefficient."""
        out = np.zeros((context.width, context.k_endog), dtype=np.float64)
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        out[offset:stop] = np.sqrt(float(scales["tau2"][0]) * scales["lam2"])
        return out

    def _tracked_label(self) -> str:
        """What the averaged diagnostic is."""
        return "posterior mean local-global scale of the standardized coefficient"

    def _label(self) -> str:
        """Short description for a summary table."""
        return "horseshoe"


@dataclass(frozen=True, kw_only=True, slots=True)
class SpikeAndSlabPrior(_AdaptivePrior):
    """Stochastic search variable selection: exclusion as a latent state.

    George, Sun and Ni's (2008) BVAR prior. Each lag coefficient carries a
    Bernoulli indicator choosing between a spike -- a normal tight enough
    around zero that the coefficient is effectively excluded -- and a slab
    loose enough that it is effectively free; the Gibbs sweep redraws the
    indicators from their exact conditionals, and their average over kept
    sweeps is a posterior inclusion probability per coefficient, which is
    this prior's distinctive output and what
    :meth:`~cultivars.multivariate.large_dim.GibbsBVARResult.inclusion_probabilities`
    reports. SSVS is this object's sampler, not a second prior; the two
    names in the literature name one thing.

    The spike and slab widths follow the semiautomatic default: each
    coefficient's least-squares standard error, times ``spike`` and
    ``slab``. The inclusion probability is a fixed hyperparameter rather
    than being given its own Beta layer, matching the reference treatment.

    Attributes:
        spike: Spike width as a multiple of the coefficient's reference
            standard error; small is a harder exclusion.
        slab: Slab width on the same scale; large is freer.
        inclusion: Prior inclusion probability of each lag coefficient.
        deterministic_scale: Looseness of the unshrunk deterministic and
            exogenous rows, in units of each equation's residual scale.

    References:
        George, E. I., Sun, D., & Ni, S. (2008). Bayesian stochastic search
            for VAR model restrictions. *Journal of Econometrics*, 142(1),
            553-580.
    """

    spike: float = 0.1
    slab: float = 10.0
    inclusion: float = 0.5
    deterministic_scale: float = 10.0

    def _check(self) -> None:
        """Reject widths and probabilities that do not describe a prior.

        Raises:
            SpecificationError: If the widths are not ordered and positive
                or the inclusion probability is not interior.
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
        """Anchor both widths to the reference scale; start everything in."""
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
        """Exact Bernoulli conditionals for the indicators."""
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
        """Spike or slab variance per the current indicators, loose elsewhere."""
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
        """The Rao-Blackwellized inclusion probability per lag coefficient."""
        out = np.zeros((context.width, context.k_endog), dtype=np.float64)
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        out[offset:stop] = scales["prob"]
        return out

    def _tracked_label(self) -> str:
        """What the averaged diagnostic is."""
        return "posterior inclusion probability"

    def _label(self) -> str:
        """Short description for a summary table."""
        return f"ssvs(spike={self.spike:g}, slab={self.slab:g}, p={self.inclusion:g})"


@dataclass(frozen=True, kw_only=True, slots=True)
class DirichletLaplacePrior(_AdaptivePrior):
    """Dirichlet-Laplace shrinkage: a simplex rations the prior's attention.

    Bhattacharya, Pati, Pillai and Dunson's (2015) global-local prior. Each
    standardized coefficient is Laplace-distributed with its own scale, the
    scales are a global magnitude times a point on the simplex drawn from a
    ``Dirichlet(concentration)``, and a small concentration makes the
    simplex spiky: the prior can only pay attention to a few coefficients
    at once, which is a budget constraint the horseshoe does not impose.
    Among the continuous shrinkage priors it carries the strongest
    theoretical warrant -- posterior contraction at the minimax rate for
    sparse means when the concentration is set near ``1/m``.

    Every conditional is exact: inverse-Gaussian for the local mixing
    scales, generalized-inverse-Gaussian for the global magnitude and the
    simplex.

    Attributes:
        concentration: The Dirichlet concentration; ``0.5`` is the paper's
            default, and values near ``1 / (k**2 * order)`` are its
            theory's.
        deterministic_scale: Looseness of the unshrunk deterministic and
            exogenous rows, in units of each equation's residual scale.

    References:
        Bhattacharya, A., Pati, D., Pillai, N. S., & Dunson, D. B. (2015).
            Dirichlet-Laplace priors for optimal shrinkage. *Journal of the
            American Statistical Association*, 110(512), 1479-1490.
    """

    concentration: float = 0.5
    deterministic_scale: float = 10.0

    def _check(self) -> None:
        """Reject hyperparameters that do not describe a prior.

        Raises:
            SpecificationError: If the concentration or the deterministic
                scale is not positive.
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
        """Uniform simplex, unit local scales, unit global magnitude."""
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
        """The exact conditionals of Bhattacharya et al. (2015)."""
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
        """``ratio**2 * psi * (phi * tau)**2`` on the lag block, loose elsewhere."""
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
        """The local-global scale per lag coefficient."""
        out = np.zeros((context.width, context.k_endog), dtype=np.float64)
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        out[offset:stop] = np.sqrt(scales["psi"]) * scales["phi"] * float(scales["tau"][0])
        return out

    def _tracked_label(self) -> str:
        """What the averaged diagnostic is."""
        return "posterior mean local-global scale of the standardized coefficient"

    def _label(self) -> str:
        """Short description for a summary table."""
        return f"dl(a={self.concentration:g})"


@dataclass(frozen=True, kw_only=True, slots=True)
class NormalGammaPrior(_AdaptivePrior):
    """Normal-Gamma shrinkage: the Bayesian lasso's adjustable-kurtosis parent.

    Griffin and Brown's (2010) prior: each standardized coefficient is
    normal with its own variance, and the variances are Gamma with shape
    ``shape``. At ``shape = 1`` the marginal is exactly the Laplace of the
    Bayesian lasso; pushing the shape below one puts more mass near zero
    and fattens the tails simultaneously, which is the knob the lasso
    lacks. The Gamma rate is the global tightness and gets its own Gamma
    hyperprior, so overall shrinkage is learned rather than tuned. The
    variance conditionals are generalized-inverse-Gaussian, drawn exactly.

    Attributes:
        shape: The Gamma shape of the local variances; ``1`` is the
            Bayesian lasso, smaller is spikier with fatter tails.
        deterministic_scale: Looseness of the unshrunk deterministic and
            exogenous rows, in units of each equation's residual scale.

    References:
        Griffin, J. E., & Brown, P. J. (2010). Inference with normal-gamma
            prior distributions in regression problems. *Bayesian
            Analysis*, 5(1), 171-188.
    """

    shape: float = 0.1
    deterministic_scale: float = 10.0

    def _check(self) -> None:
        """Reject hyperparameters that do not describe a prior.

        Raises:
            SpecificationError: If the shape or the deterministic scale is
                not positive.
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
        """Minnesota-tightness local variances; the rate that implies them."""
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
        """GIG conditionals for the variances, Gamma for the global rate.

        Squared deviations are floored at ``1e-11``: at an exact zero the
        variance conditional is improper for ``shape < 1/2`` (a known
        boundary of the hierarchy), and the floor -- three parts per
        million on the standardized scale -- restores it without moving
        anything statistically visible.
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
        """``ratio**2 * psi`` on the lag block, loose elsewhere."""
        ratio = self._unit_ratio(context)
        out = (self.deterministic_scale * ratio) ** 2
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        out[offset:stop] = ratio[offset:stop] ** 2 * scales["psi"]
        return out

    def _tracked(
        self, scales: dict[str, npt.NDArray[np.float64]], context: _PriorContext
    ) -> npt.NDArray[np.float64]:
        """The local scale per lag coefficient."""
        out = np.zeros((context.width, context.k_endog), dtype=np.float64)
        offset = context.lag_offset
        stop = offset + context.k_endog * context.order
        out[offset:stop] = np.sqrt(scales["psi"])
        return out

    def _tracked_label(self) -> str:
        """What the averaged diagnostic is."""
        return "posterior mean local-global scale of the standardized coefficient"

    def _label(self) -> str:
        """Short description for a summary table."""
        return f"ng(theta={self.shape:g})"
