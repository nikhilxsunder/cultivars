# filepath: /src/cultivars/multivariate/structural/stochastic_volatility.py
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
r"""Identification through stochastic volatility: the regimes nobody has to declare.

Rigobon's scheme buys identification from variance shifts, but it needs the
user to say *when* the shifts happen. This module drops that requirement.
Give each structural shock its own latent log-AR(1) variance,

.. math::

   u_t = B \varepsilon_t, \qquad
   \varepsilon_{jt} = e^{h_{jt}/2} \eta_{jt}, \qquad
   h_{jt} = \phi_j h_{j,t-1} + \sigma_j \xi_{jt},

so that the reduced-form covariance :math:`\Sigma_t = B\,
\operatorname{diag}(e^{h_t})\, B'` moves every period; as long as the
variance paths are not proportional to one another, the sequence of
covariances pins :math:`B` down up to column order and sign (Lewis 2021;
Bertsche and Braun 2022). No zero, no sign pattern, no instrument, and no
regime dates -- the volatility processes are estimated along with the
impact matrix. Estimation is Bayesian, by an exact Gibbs sampler on the
reduced-form innovations of a fitted closed system: the rows of
:math:`A = B^{-1}` by Waggoner and Zha's (2003) construction given the
paths, the paths by the Kim-Shephard-Chib mixture step given :math:`A`,
and each path's persistence and innovation variance from their
conditionals. What comes back is a posterior over impact matrices, not a
point, and :class:`StochasticVolatilitySVARResult` holds it: the quantile
surfaces of impulse responses summarize a set of complete structural
models, none of which traces any one surface.

The module also houses the univariate long-memory relative of the same
volatility law, :class:`LongMemorySV`: :math:`y_t = c + e^{h_t/2}
\epsilon_t` with :math:`(1 - \phi L)(1 - L)^d (h_t - \mu) = \eta_t`
(Breidt, Crato and de Lima 1998), whose fractional operator has no finite
state and is therefore estimated by Whittle in the frequency domain
rather than by any filter. It identifies nothing structural; it is here
because it is the one-series statement of what a volatility path is and
the natural diagnostic for whether an AR(1) log variance is the right
law for a shock at all.

Two commitments shape the surface. First, what the data cannot supply is
said, not hidden. The structural shocks are statistical objects, labelled
by which variable they load on most and signed to a positive diagonal --
a convention the summary names as such, with an economic name left as a
claim to be argued from outside the model; and the scale is the one the
likelihood can separate, since it cannot tell a path's level from its
column's scale, so every draw is expressed with zero sample-mean log
variance per shock and the level folded into :math:`B`, the impact of a
shock at its in-sample average variance. Second, identification is
tested, not assumed. With proportional variance paths the likelihood is
flat over rotations yet the smooth-AR(1) prior still returns a sharp
posterior, chain after chain, so the posterior cannot certify itself; the
identifying condition -- that the shocks' variance *ratios* move over
time -- is tested on the posterior-median shocks, pair by pair, and the
verdict is printed with the result alongside the relabelling rate, the
second symptom of the same failure.

Layout. :class:`StochasticVolatilitySVAR` is a
``_VolatilityIdentificationModel`` from ``_internals``, whose ``_sample``
runs the three-block Gibbs sampler and relabels the kept draws into a
``_VolatilityStructuralFit``; the prior is
:class:`~cultivars.bayes.priors.VolatilityPrior`; the identification test
is ``_variance_ratio_test`` in ``_core``, the quantile levels pass
through ``_validate_quantiles``, and the summary renders through the
shared ``_LABEL_NOTE`` and ``_SCALE_NOTE``. The posterior median lands on
:class:`~cultivars.multivariate.structural.zero_restrictions.SVARResult`
through ``to_point()``, and convergence is read through
``_ConvergenceMixin``. :class:`LongMemorySV` is a
``_LongMemoryVolatilityModel`` whose ``_fit_whittle`` maximizes the
criterion over ``_fractional_spectrum`` from ``_core`` and emits the
truncated AR(inf) law through ``_long_memory_quasi_state_space`` in
``_internals``. The declared-regime version of the identification is
:mod:`~cultivars.multivariate.structural.heteroskedacity`.

References:
    Lewis, D. J. (2021). Identifying shocks via time-varying volatility.
    *Review of Economic Studies*, 88(6), 3086-3124.

    Bertsche, D., & Braun, R. (2022). Identification of structural vector
    autoregressions by stochastic volatility. *Journal of Business &
    Economic Statistics*, 40(1), 328-341.

    Waggoner, D. F., & Zha, T. (2003). A Gibbs sampler for structural
    vector autoregressions. *Journal of Economic Dynamics and Control*,
    28(2), 349-366.

    Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
    Likelihood inference and comparison with ARCH models. *Review of
    Economic Studies*, 65(3), 361-393.

    Breidt, F. J., Crato, N., & de Lima, P. (1998). The detection and
    estimation of long memory in stochastic volatility. *Journal of
    Econometrics*, 83(1-2), 325-348.

Example:
    The same impact matrix under two volatility laws. With distinct paths
    the identification test passes and the column directions are
    recovered; with one path shared by both shocks the chain still returns
    a sharp matrix, and the test is what says it means nothing:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> rng = np.random.default_rng(0)
    >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
    >>> def simulate(h):
    ...     shocks = np.exp(h / 2) * rng.standard_normal((400, 2))
    ...     y = np.zeros((400, 2))
    ...     for t in range(1, 400):
    ...         y[t] = 0.4 * y[t - 1] + B @ shocks[t]
    ...     return VAR(y, order=1, names=("gdp", "infl")).fit()
    >>> distinct = np.zeros((400, 2))
    >>> for t in range(1, 400):
    ...     distinct[t] = [0.97, 0.6] * distinct[t - 1] + [0.2, 0.6] * rng.standard_normal(2)
    >>> shared = np.cumsum(0.2 * rng.standard_normal(400)) * 0.3
    >>> chain = dict(n_draws=300, n_burn=100, seed=0)
    >>> good = StochasticVolatilitySVAR(simulate(distinct)).identify(**chain)
    >>> same = simulate(np.column_stack([shared, shared]))
    >>> bad = StochasticVolatilitySVAR(same).identify(**chain)
    >>> dict(good.to_point().diagnostics)["Identification"]
    'variance ratios move over time for every pair'
    >>> dict(bad.to_point().diagnostics)["Identification"]
    "WEAK: some pair's variance ratio is not detectably time-varying"
    >>> unit = lambda v: v / np.linalg.norm(v)
    >>> bool(np.abs(unit(good.impact[:, 1]) - unit(B[:, 1])).max() < 0.1)
    True
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import (
    _LABEL_NOTE,
    _SCALE_NOTE,
    ClosedSystemResult,
    SummaryTable,
    _fractional_spectrum,
    _validate_quantiles,
    _variance_ratio_test,
)
from ..._internals import (
    _ComparisonMixin,
    _ConvergenceMixin,
    _long_memory_quasi_state_space,
    _LongMemoryVolatilityFit,
    _LongMemoryVolatilityModel,
    _LongMemoryVolatilityParameters,
    _SeriesMixin,
    _SummaryMixin,
    _VolatilityIdentificationModel,
    _VolatilityStructuralFit,
)
from ...bayes.priors import VolatilityPrior
from ...exceptions import SpecificationError
from ...state_space.linear_gaussian import LinearGaussianSSM
from .zero_restrictions import SVARResult

__all__ = [
    "LongMemorySV",
    "LongMemorySVResult",
    "StochasticVolatilitySVAR",
    "StochasticVolatilitySVARResult",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class StochasticVolatilitySVARResult(_SummaryMixin, _ConvergenceMixin):
    """The posterior of a stochastic-volatility identification.

    Attributes:
        source: The reduced-form result identified.
        impact_draws: ``(S, k, k)`` posterior draws of the impact matrix,
            already relabelled to one order and sign convention.
        h_draws: ``(S, T, k)`` posterior draws of the structural log
            variances, zero sample mean per shock.
        phi_draws: ``(S, k)`` draws of each shock's log-variance
            persistence.
        sigma2_draws: ``(S, k)`` draws of each shock's log-variance
            innovation variance.
        shock_names: One label per shock column.
        relabel_rate: Fraction of kept draws whose columns had to be
            permuted to match the reference labelling.
        restriction: The identifying assumption, stated.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.
    """

    source: ClosedSystemResult = field(repr=False)
    impact_draws: npt.NDArray[np.float64] = field(repr=False)
    h_draws: npt.NDArray[np.float64] = field(repr=False)
    phi_draws: npt.NDArray[np.float64] = field(repr=False)
    sigma2_draws: npt.NDArray[np.float64] = field(repr=False)
    shock_names: tuple[str, ...]
    relabel_rate: float
    restriction: str
    n_draws: int
    n_burn: int
    thin: int

    @classmethod
    def _from_fit(
        cls, fit: _VolatilityStructuralFit, model: StochasticVolatilitySVAR
    ) -> StochasticVolatilitySVARResult:
        """Assemble the public result from a raw fit and its specification."""
        return cls(
            source=model.source,
            impact_draws=fit.impact_draws,
            h_draws=fit.h_draws,
            phi_draws=fit.phi_draws,
            sigma2_draws=fit.sigma2_draws,
            shock_names=model.shock_names,
            relabel_rate=fit.relabel_rate,
            restriction=(
                "Each structural shock carries its own latent log-AR(1) variance, "
                "and the shocks' variance paths are assumed not proportional to one "
                "another; that non-proportionality is the entire identification "
                "(Lewis 2021; Bertsche and Braun 2022). The impact matrix is "
                "constant over the sample."
            ),
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Variable labels, from the reduced form."""
        return self.source.names

    @property
    def k_endog(self) -> int:
        """Number of variables."""
        return self.source.k_endog

    @property
    def n_kept(self) -> int:
        """Posterior draws retained."""
        return int(self.impact_draws.shape[0])

    @property
    def impact(self) -> npt.NDArray[np.float64]:
        """The pointwise posterior median impact matrix, ``(k, k)``.

        A summary, not a draw: no single structural model has exactly
        these entries. For a complete point-identified object at this
        matrix use :meth:`to_point`.
        """
        return np.asarray(np.median(self.impact_draws, axis=0), dtype=np.float64)

    def to_point(self) -> SVARResult:
        """The posterior-median impact as a point-identified structural result.

        Gives the whole point surface -- impulse responses, historical
        decompositions, structural shocks -- at one matrix. The result
        says on its face that it is a posterior summary.
        """
        return SVARResult(
            source=self.source,
            impact=self.impact,
            shock_names=self.shock_names,
            scheme="stochastic volatility (posterior median)",
            restriction=self.restriction,
            diagnostics=tuple(self._diagnostics()),
        )

    def irf_draws(self, horizon: int = 20, *, cumulative: bool = False) -> npt.NDArray[np.float64]:
        """Structural impulse responses for every posterior draw.

        Args:
            horizon: Largest lead to return.
            cumulative: Return running sums.

        Returns:
            An array of shape ``(S, horizon + 1, k, k)``.
        """
        psi = self.source.ma_representation(horizon)
        theta = np.einsum("hik,nkj->nhij", psi, self.impact_draws)
        return np.cumsum(theta, axis=1) if cumulative else theta

    def irf(
        self,
        horizon: int = 20,
        *,
        quantiles: Sequence[float] = (0.16, 0.5, 0.84),
        cumulative: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Pointwise posterior quantiles of the impulse responses.

        Args:
            horizon: Largest lead to return.
            quantiles: Probability levels, each in ``(0, 1)``.
            cumulative: Return running sums before taking quantiles.

        Returns:
            An array of shape ``(len(quantiles), horizon + 1, k, k)``.

        Raises:
            SpecificationError: If a quantile is outside the open unit
                interval.
        """
        levels = _validate_quantiles(quantiles)
        return np.quantile(self.irf_draws(horizon, cumulative=cumulative), levels, axis=0)

    def fevd(
        self, horizon: int = 20, *, quantiles: Sequence[float] = (0.16, 0.5, 0.84)
    ) -> npt.NDArray[np.float64]:
        """Pointwise posterior quantiles of the variance decomposition.

        Computed at each shock's in-sample average variance, the scale the
        impact columns carry; under stochastic volatility the true
        decomposition is time-varying, and this is its average-variance
        reading.

        Args:
            horizon: Largest lead to return.
            quantiles: Probability levels, each in ``(0, 1)``.

        Returns:
            An array of shape ``(len(quantiles), horizon + 1, k, k)``.

        Raises:
            SpecificationError: If a quantile is outside the open unit
                interval.
        """
        levels = _validate_quantiles(quantiles)
        theta = self.irf_draws(horizon)
        explained = np.cumsum(theta**2, axis=1)
        shares = explained / explained.sum(axis=3, keepdims=True)
        return np.quantile(shares, levels, axis=0)

    def structural_shocks(self) -> npt.NDArray[np.float64]:
        """Posterior-median structural shocks ``B**-1 u_t``, ``(T, k)``."""
        return np.asarray(
            np.asarray(self.source.resid) @ np.linalg.inv(self.impact).T, dtype=np.float64
        )

    def volatility_path(
        self, shock: str | int, *, quantiles: Sequence[float] = (0.16, 0.5, 0.84)
    ) -> npt.NDArray[np.float64]:
        """Posterior quantiles of one shock's standard deviation over the sample.

        The path is ``exp(h_t / 2)`` scaled by the shock's impact column
        norm, so it reads in the units of the reduced-form innovations
        rather than as a normalized ratio.

        Args:
            shock: A shock label or column index.
            quantiles: Probability levels, each in ``(0, 1)``.

        Returns:
            An array of shape ``(len(quantiles), T)``.

        Raises:
            SpecificationError: If the shock is unknown or a quantile is
                outside the open unit interval.
        """
        levels = _validate_quantiles(quantiles)
        column = self._column(shock)
        scale = np.linalg.norm(self.impact_draws[:, :, column], axis=1)
        paths = np.exp(0.5 * self.h_draws[:, :, column]) * scale[:, None]
        return np.quantile(paths, levels, axis=0)

    def identification_test(
        self, *, n_blocks: int = 10, window: int = 9
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Test the identifying condition pair by pair on the posterior-median shocks.

        Identification here rests on the shocks' variance paths not being
        proportional. That is not something the posterior can certify:
        with proportional paths the likelihood is flat over rotations, yet
        the volatility prior still hands back a sharp posterior at the
        rotation whose shocks look most like clustered volatility. So the
        condition is tested on the data instead. Under proportionality the
        *ratio* of two shocks' variances is constant over time whatever
        the common path does; the statistic is the precision-weighted
        dispersion of block-wise log variance ratios, chi-squared with
        ``n_blocks - 1`` degrees of freedom under the null. A large
        p-value for any pair means the data cannot tell those two shocks'
        rotation apart, and the summary says so.

        Args:
            n_blocks: Contiguous sample blocks, at least 3.
            window: Odd length of the moving average that proxies the
                pair's common variance when weighting blocks.

        Returns:
            ``(statistics, p_values)``, each ``(k, k)`` symmetric.

        Raises:
            SpecificationError: If the settings are malformed for the
                sample.
        """
        return _variance_ratio_test(self.structural_shocks(), n_blocks=n_blocks, window=window)

    def _column(self, shock: str | int) -> int:
        """Resolve a shock label or index to a column."""
        if isinstance(shock, str):
            if shock not in self.shock_names:
                raise SpecificationError(f"unknown shock {shock!r}; have {self.shock_names}.")
            return self.shock_names.index(shock)
        if not 0 <= int(shock) < self.k_endog:
            raise SpecificationError(f"shock index {shock} is out of range for {self.k_endog}.")
        return int(shock)

    def _diagnostics(self) -> list[tuple[str, str]]:
        """Identification diagnostics shared by the summary and the point result."""
        k = self.k_endog
        _, p_values = self.identification_test()
        largest = float(p_values[np.triu_indices(k, 1)].max()) if k > 1 else 0.0
        persistence = ", ".join(f"{value:.3f}" for value in self.phi_draws.mean(axis=0))
        verdict = (
            "WEAK: some pair's variance ratio is not detectably time-varying"
            if largest > 0.1
            else "variance ratios move over time for every pair"
        )
        return [
            ("Max ratio-test p-value", f"{largest:.4f}"),
            ("Relabelling rate", f"{self.relabel_rate:.3f}"),
            ("Persistence by shock", persistence),
            ("Identification", verdict),
        ]

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        med = self.impact
        lo, hi = np.quantile(self.impact_draws, [0.16, 0.84], axis=0)
        rows = tuple(
            (
                f"{self.names[i]} <- {self.shock_names[j]}",
                f"{med[i, j]:.4f}",
                f"[{lo[i, j]:.4f}, {hi[i, j]:.4f}]",
            )
            for j in range(self.k_endog)
            for i in range(self.k_endog)
        )
        return SummaryTable(
            title="Stochastic-Volatility SVAR (posterior)",
            metadata=(
                ("Scheme", "stochastic volatility"),
                ("Draws kept", f"{self.n_kept}"),
                ("Burn-in", f"{self.n_burn}"),
                ("Thin", f"{self.thin}"),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.source.nobs}"),
                *self._diagnostics(),
            ),
            columns=("impact", "median", "68% interval"),
            rows=rows,
            notes=(self.restriction, _LABEL_NOTE, _SCALE_NOTE),
        )


class StochasticVolatilitySVAR(_VolatilityIdentificationModel[StochasticVolatilitySVARResult]):
    """Identification through stochastic volatility, Lewis (2021) / Bertsche-Braun (2022).

    Every structural shock carries a latent log-AR(1) variance; the impact
    matrix is identified up to column order and sign because the shocks'
    variance paths are not proportional. Nothing is declared -- no regime
    dates, no restrictions -- and the price is that the shocks come back
    labelled by statistics, not economics.

    Args:
        result: A fitted closed reduced-form result (a VAR, VECM, ...).
        shock_names: One label per shock column, in the convention order
            (shock ``j`` loads most on variable ``j``). Defaults to
            ``shock_<variable>``.

    Raises:
        SpecificationError: If the result is not a closed system, or the
            labels do not match the number of variables.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> from cultivars.multivariate.reduced_form import VAR
        >>> n = 400
        >>> phi, sigma = np.array([0.97, 0.6]), np.array([0.2, 0.6])
        >>> h = np.zeros((n, 2))
        >>> for t in range(1, n):
        ...     h[t] = phi * h[t - 1] + sigma * rng.standard_normal(2)
        >>> shocks = np.exp(h / 2) * rng.standard_normal((n, 2))
        >>> y = np.zeros((n, 2))
        >>> for t in range(1, n):
        ...     y[t] = 0.4 * y[t - 1] + shocks[t] @ np.array([[1.0, 0.5], [0.3, 1.0]]).T
        >>> res = VAR(y, order=1).fit()
        >>> post = StochasticVolatilitySVAR(res).identify(n_draws=300, n_burn=100, seed=0)
        >>> post.impact.shape
        (2, 2)
        >>> post.to_point().is_complete
        True
    """

    __slots__ = ("_labels",)

    def __init__(
        self,
        result: ClosedSystemResult,
        *,
        shock_names: Sequence[str] | None = None,
    ) -> None:
        """Validate the source system and the labels."""
        super().__init__(result)
        k = self.k_endog
        if shock_names is None:
            self._labels = tuple(f"shock_{name}" for name in self.names)
        else:
            labels = tuple(str(name) for name in shock_names)
            if len(labels) != k:
                raise SpecificationError(
                    f"shock_names must have one entry per variable ({k}); got {len(labels)}."
                )
            self._labels = labels

    @property
    def shock_names(self) -> tuple[str, ...]:
        """One label per shock column."""
        return self._labels

    def identify(
        self,
        *,
        n_draws: int = 3000,
        n_burn: int = 1000,
        thin: int = 1,
        prior: VolatilityPrior | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> StochasticVolatilitySVARResult:
        """Sample the posterior over impact matrices and volatility paths.

        The prior on each shock's volatility law is the Kim-Shephard-Chib
        one: Beta ``(a, b)`` on ``(phi + 1) / 2`` and inverse-gamma
        ``(shape, rate)`` on the innovation variance. The impact matrix
        carries a weak Gaussian prior on the rows of ``B**-1`` scaled from
        the reduced-form covariance.

        Args:
            n_draws: Total sampler iterations.
            n_burn: Burn-in discarded.
            thin: Keep every ``thin``-th post-burn draw.
            prior: The prior on each shock's volatility law, an instance of
                :class:`VolatilityPrior` or ``None`` to use the default Kim-Shephard-Chib prior.
            seed: Seed or generator.

        Returns:
            The :class:`StochasticVolatilitySVARResult`.

        Raises:
            SpecificationError: If the draw bookkeeping is inconsistent.
            NumericalError: If the structural matrix degenerates.
        """
        fit = self._sample(
            n_draws=n_draws,
            n_burn=n_burn,
            thin=thin,
            prior=VolatilityPrior() if prior is None else prior,
            seed=seed,
        )
        return StochasticVolatilitySVARResult._from_fit(fit, self)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class LongMemorySVResult(_SummaryMixin, _SeriesMixin, _ComparisonMixin):
    r"""A Whittle-estimated long-memory stochastic-volatility model.

    The fitted law is

    .. math::

       y_t = c + e^{h_t/2} \epsilon_t, \qquad
       h_t = \mu + v_t, \qquad
       (1 - \phi L)(1 - L)^d v_t = \eta_t,

    estimated on the linearization :math:`\log (y_t - c)^2 = \mu +
    \mathbb{E}\log\epsilon_t^2 + v_t + \xi_t` by maximizing the Whittle
    criterion over the positive Fourier frequencies, where the spectrum of
    the log-squared data is the ARFIMA(1, d, 0) spectrum plus the
    :math:`\pi^2/2` white-noise floor of :math:`\xi_t`. The record carries
    the estimate, the criterion, and the log-variance path from the
    spectral smoother with its one stationary error band.

    Note:
        ``llf`` is a quasi-likelihood: comparable with other Whittle fits
        through :meth:`compare` and :meth:`likelihood_ratio_test`, not
        with a Kalman or particle likelihood of the same data. ``d`` and
        ``phi`` compete for the same persistence -- with ``short_memory``
        the AR(1) can absorb most of the memory and push ``d`` toward
        zero -- and a small ``d`` against the measurement floor is weakly
        identified; the summary says both.

    Attributes:
        endog: The observed series.
        mean_spec: ``"constant"`` or ``"zero"``.
        mean: The observation mean ``c``.
        mu: Level of the log variance.
        d: Fractional differencing order of the log variance.
        sigma2: Innovation variance of the fractional noise.
        phi: Short-memory AR(1) coefficient of the log variance; ``0.0``
            when the law is pure fractional noise.
        short_memory: Whether ``phi`` was estimated.
        llf: The Whittle criterion at the optimum, read as a
            log-likelihood: the frequency-domain quasi-likelihood of the
            linearized model. Comparable only with other Whittle fits.
        nobs: Observations.
        n_params: Free parameters.
        n_frequencies: Fourier ordinates the criterion summed over.
        log_variance: Smoothed log-variance path, ``(n,)``, from the
            spectral smoother.
        log_variance_std: The smoother's stationary root mean squared
            error, repeated ``(n,)`` times.

    See Also:
        * :class:`LongMemorySV` -- the model whose ``fit()`` returns this
          record.
        * :class:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM`
          -- the substrate :meth:`quasi_state_space` emits onto.

    References:
        Breidt, F. J., Crato, N., & de Lima, P. (1998). The detection and
        estimation of long memory in stochastic volatility. *Journal of
        Econometrics*, 83(1-2), 325-348.

        Harvey, A. C., Ruiz, E., & Shephard, N. (1994). Multivariate
        stochastic variance models. *Review of Economic Studies*, 61(2),
        247-264.

    Example:
        Fractional noise with ``d = 0.4`` in the log variance. The Whittle
        fit recovers the order and the innovation variance, the smoothed
        path tracks the truth, and the implied autocorrelation decays
        hyperbolically:

        >>> import numpy as np
        >>> from cultivars._core import fractional_difference_weights
        >>> rng = np.random.default_rng(0)
        >>> n, burn = 1500, 2000
        >>> eta = 0.5 * rng.standard_normal(n + burn)
        >>> weights = fractional_difference_weights(-0.4, n + burn)
        >>> v = np.convolve(eta, weights)[burn : n + burn]
        >>> y = np.exp((-1.0 + v) / 2) * rng.standard_normal(n)
        >>> res = LongMemorySV(y, mean="zero").fit()
        >>> round(res.d, 2), round(res.sigma2, 2), res.phi, res.short_memory
        (0.39, 0.1, 0.0, False)
        >>> res.nobs, res.n_params, res.n_frequencies, res.at_boundary
        (1500, 2.0, 750, False)
        >>> bool(np.corrcoef(res.log_variance, -1.0 + v)[0, 1] > 0.4)
        True
        >>> acf = res.autocorrelation(200)
        >>> bool(abs(acf[200] / acf[100] - 2 ** (2 * res.d - 1)) < 0.1)
        True
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    """The observed series, ``(n,)``. Kept out of the repr."""

    mean_spec: str
    """``"constant"`` when the sample mean was removed, ``"zero"`` when fixed at zero."""

    mean: float
    """The observation mean ``c``: the sample mean, or ``0.0``."""

    mu: float
    """Level of the log variance."""

    d: float
    """Fractional differencing order, searched over ``|d| < 0.5``."""

    sigma2: float
    """Innovation variance of the fractional noise."""

    phi: float
    """Short-memory AR(1) coefficient of the log variance; ``0.0`` unless estimated."""

    short_memory: bool
    """Whether ``phi`` was estimated."""

    llf: float
    """The Whittle quasi-likelihood at the optimum; comparable only with Whittle fits."""

    nobs: int
    """Observations."""

    n_params: float
    """Free parameters: ``mu``, ``d``, ``sigma2``, plus ``phi`` and ``mean`` when estimated."""

    n_frequencies: int
    """Positive Fourier ordinates the criterion summed over, ``n // 2``."""

    log_variance: npt.NDArray[np.float64] = field(repr=False)
    """The ``(n,)`` spectrally smoothed log-variance path. Kept out of the repr."""

    log_variance_std: npt.NDArray[np.float64] = field(repr=False)
    """The smoother's stationary root mean squared error, repeated ``(n,)`` times.

    Kept out of the repr.
    """

    @classmethod
    def _from_fit(cls, fit: _LongMemoryVolatilityFit, model: LongMemorySV) -> LongMemorySVResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The packed Whittle fit: the parameter record, criterion,
                counts and the smoothed path with its band.
            model: The model that produced it, supplying the data and
                the specification flags.

        Returns:
            A populated result.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> model = LongMemorySV(rng.standard_normal(300), mean="zero")
            >>> res = LongMemorySVResult._from_fit(model._fit_whittle(), model)
            >>> res.mean_spec, res.mean, res.short_memory, res.nobs, res.n_frequencies
            ('zero', 0.0, False, 300, 150)
        """
        return cls(
            endog=model.endog,
            mean_spec=model.mean_spec,
            mean=fit.params.mean,
            mu=fit.params.mu,
            d=fit.params.d,
            sigma2=fit.params.sigma2,
            phi=fit.params.phi,
            short_memory=model.short_memory,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=float(fit.n_params),
            n_frequencies=fit.n_frequencies,
            log_variance=fit.log_variance,
            log_variance_std=fit.log_variance_std,
        )

    @property
    def _params(self) -> _LongMemoryVolatilityParameters:
        """The parameter record, rebuilt for the emitter.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = LongMemorySV(rng.standard_normal(300)).fit()
            >>> params = res._params
            >>> (params.d, params.mu, params.mean) == (res.d, res.mu, res.mean)
            True
        """
        return _LongMemoryVolatilityParameters(
            mu=self.mu, d=self.d, sigma2=self.sigma2, phi=self.phi, mean=self.mean
        )

    @property
    def at_boundary(self) -> bool:
        """Whether ``d`` sits at the stationarity boundary of the search.

        The Whittle criterion is maximized over ``|d| < 0.5``; an estimate
        within ``0.01`` of that edge means the data prefer a nonstationary
        log variance, and ``d`` should be read as a lower bound rather than
        a point.

        Example:
            A random-walk log variance pushes ``d`` to the edge:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> h = np.cumsum(0.1 * rng.standard_normal(1500))
            >>> y = np.exp(h / 2) * rng.standard_normal(1500)
            >>> res = LongMemorySV(y, mean="zero").fit()
            >>> round(res.d, 3), res.at_boundary
            (0.499, True)
        """
        return abs(self.d) >= 0.49

    @property
    def volatility(self) -> npt.NDArray[np.float64]:
        """Smoothed volatility ``exp(h_t / 2)``, ``(n,)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = LongMemorySV(rng.standard_normal(300)).fit()
            >>> vol = res.volatility
            >>> vol.shape, bool(np.allclose(vol, np.exp(0.5 * res.log_variance)))
            ((300,), True)
        """
        return np.asarray(np.exp(0.5 * self.log_variance), dtype=np.float64)

    def volatility_bands(
        self, level: float = 0.95
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Smoother bands on the volatility at a Gaussian coverage level.

        The band is the smoother's stationary error, so its width in logs
        is the same at every observation.

        Args:
            level: Coverage probability in ``(0, 1)``.

        Returns:
            ``(lower, upper)`` bands, each ``(n,)``.

        Raises:
            SpecificationError: If ``level`` is outside ``(0, 1)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = LongMemorySV(rng.standard_normal(300)).fit()
            >>> lower, upper = res.volatility_bands()
            >>> bool(np.all(lower < res.volatility)), bool(np.all(res.volatility < upper))
            (True, True)
            >>> ratio = upper / lower
            >>> bool(np.allclose(ratio, ratio[0]))
            True
            >>> res.volatility_bands(1.0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: level must lie in (0, 1); got 1.0.
        """
        if not 0.0 < level < 1.0:
            raise SpecificationError(f"level must lie in (0, 1); got {level}.")
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + 0.5 * level))
        lower = np.exp(0.5 * (self.log_variance - z * self.log_variance_std))
        upper = np.exp(0.5 * (self.log_variance + z * self.log_variance_std))
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    def autocorrelation(self, lags: int = 50) -> npt.NDArray[np.float64]:
        """Implied autocorrelation of the log variance at lags ``0..lags``.

        The ARFIMA(1, d, 0) autocorrelation, computed from the spectrum by
        Fourier inversion on a fine grid; its hyperbolic tail is the
        model's signature and the thing to hold against the sample
        autocorrelation of ``log((y - c)**2)`` after the white-noise floor
        is netted out.

        Args:
            lags: Largest lag, at least 1.

        Returns:
            An array of shape ``(lags + 1,)`` with ``1.0`` first.

        Raises:
            SpecificationError: If ``lags`` is not positive.

        Example:
            >>> import numpy as np
            >>> from cultivars._core import fractional_difference_weights
            >>> rng = np.random.default_rng(0)
            >>> n, burn = 1500, 2000
            >>> eta = 0.5 * rng.standard_normal(n + burn)
            >>> weights = fractional_difference_weights(-0.4, n + burn)
            >>> v = np.convolve(eta, weights)[burn : n + burn]
            >>> y = np.exp((-1.0 + v) / 2) * rng.standard_normal(n)
            >>> res = LongMemorySV(y, mean="zero").fit()
            >>> acf = res.autocorrelation(10)
            >>> acf.shape, float(acf[0]), bool(np.all(np.diff(acf) < 0.0))
            ((11,), 1.0, True)
            >>> acf[[1, 5, 10]].round(2).tolist()
            [0.62, 0.42, 0.35]
            >>> res.autocorrelation(0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: lags must be at least 1; got 0.
        """
        if lags < 1:
            raise SpecificationError(f"lags must be at least 1; got {lags}.")
        grid = 2**14
        freqs = np.pi * (np.arange(grid) + 0.5) / grid
        spectrum = _fractional_spectrum(freqs, d=self.d, sigma2=self.sigma2, phi=self.phi)
        out = np.empty(lags + 1)
        for lag in range(lags + 1):
            out[lag] = float(np.mean(spectrum * np.cos(lag * freqs)))
        return np.asarray(out / out[0], dtype=np.float64)

    def quasi_state_space(self, truncation: int = 100) -> LinearGaussianSSM:
        """The linearized model as a truncated AR(inf) linear-Gaussian state space.

        The fractional operator has no finite state, so this is the
        autoregressive representation cut at ``truncation`` lags on the
        log-squared data: exact for the truncated law, an approximation of
        the long-memory one, and the ``pi**2 / 2`` measurement floor is the
        linearization's. What the Kalman filter returns on it is a
        truncated quasi-likelihood, not the model's likelihood.

        Args:
            truncation: Autoregressive lags retained, at least 1.

        Returns:
            The linear-Gaussian state-space model in the log-squared data.

        Raises:
            SpecificationError: If ``truncation`` is not positive.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = LongMemorySV(rng.standard_normal(300), mean="zero").fit()
            >>> system = res.quasi_state_space(20)
            >>> type(system).__name__, system.k_states, system.k_endog
            ('_LinearGaussianStateSpace', 20, 1)
            >>> res.quasi_state_space(0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: truncation must be at least 1; got 0.
        """
        return _long_memory_quasi_state_space(self._params, truncation=truncation)

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """Aligned per-observation output.

        Returns:
            The observed series, the smoothed log variance, the volatility
            and its 95% band, each ``(n,)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = LongMemorySV(rng.standard_normal(300)).fit()
            >>> list(res._series())
            ['observed', 'log_variance', 'volatility', 'volatility_lower', 'volatility_upper']
            >>> res.to_pandas().shape
            (300, 5)
        """
        lower, upper = self.volatility_bands()
        return {
            "observed": self.endog,
            "log_variance": self.log_variance,
            "volatility": self.volatility,
            "volatility_lower": lower,
            "volatility_upper": upper,
        }

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(300)
            >>> LongMemorySV(y).fit()._comparison_label()
            'LongMemorySV[constant, ARFIMA(0,d,0)]'
            >>> LongMemorySV(y, mean="zero", short_memory=True).fit()._comparison_label()
            'LongMemorySV[zero, ARFIMA(1,d,0)]'
        """
        law = "ARFIMA(1,d,0)" if self.short_memory else "ARFIMA(0,d,0)"
        return f"LongMemorySV[{self.mean_spec}, {law}]"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        One row per estimated parameter; the Whittle criterion, counts
        and information criteria in the metadata; four standing notes on
        the quasi-likelihood, the hyperbolic decay, the smoother and the
        weak identification of small ``d``, plus a fifth when ``d`` is at
        the boundary.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(300)
            >>> table = LongMemorySV(y, short_memory=True).fit()._summary_table()
            >>> table.title, len(table.notes)
            ('LongMemorySV[constant, ARFIMA(1,d,0)] Results', 4)
            >>> [row[0] for row in table.rows]
            ['mean', 'mu', 'd', 'sigma2', 'phi']
            >>> dict(table.metadata)["Method"], dict(table.metadata)["Frequencies"]
            ('whittle', '150')
        """
        ic = self.information_criteria
        rows: list[tuple[str, str]] = [
            ("mu", f"{self.mu:.4f}"),
            ("d", f"{self.d:.4f}"),
            ("sigma2", f"{self.sigma2:.6g}"),
        ]
        if self.short_memory:
            rows.append(("phi", f"{self.phi:.4f}"))
        if self.mean_spec == "constant":
            rows.insert(0, ("mean", f"{self.mean:.6g}"))
        notes = [
            "The criterion is the Whittle quasi-likelihood of the Harvey-Ruiz-Shephard "
            "linearization, summed over the positive Fourier frequencies; compare only "
            "against other Whittle fits, and read the information criteria in that light.",
            "Long memory means volatility shocks decay hyperbolically, at rate lag**(2d - 1) "
            "in the autocorrelation; autocorrelation() has the implied shape. There is no "
            "half-life.",
            "The log-variance path is the spectral (Wiener-Kolmogorov) smoother under a "
            "circular approximation, with one stationary error band; the truncated "
            "quasi_state_space() offers a Kalman reading of the same law, labelled as a "
            "truncation.",
            "Small d against the pi**2 / 2 measurement floor is weakly identified in the "
            "frequency domain: dispersion of d across samples is large below about 0.3 "
            "unless the volatility signal is strong.",
        ]
        if self.at_boundary:
            notes.append(
                "d is at the stationarity boundary of the search (|d| < 0.5): the data "
                "prefer a nonstationary log variance, and d is a lower bound, not a point. "
                "UCSV is the random-walk (d = 1) alternative."
            )
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=(
                ("Method", "whittle"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("AIC", f"{ic.aic:.3f}"),
                ("Frequencies", f"{self.n_frequencies}"),
                ("BIC", f"{ic.bic:.3f}"),
            ),
            columns=("", "estimate"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class LongMemorySV(_LongMemoryVolatilityModel[LongMemorySVResult]):
    r"""Long-memory stochastic volatility, Breidt, Crato, and de Lima (1998).

    .. math::

       y_t = c + e^{h_t/2} \epsilon_t, \qquad h_t = \mu + v_t, \qquad
       (1 - \phi L)(1 - L)^d v_t = \eta_t,

    the log variance is fractionally integrated, so its shocks decay
    hyperbolically -- the persistence that squared and absolute returns
    show at long lags and that no finite-order AR(1) log variance can
    produce. Estimation is Whittle in the frequency domain, the natural
    home of a process with no finite state: the periodogram of
    :math:`\log (y_t - c)^2` against the ARFIMA spectrum plus the
    :math:`\pi^2/2` floor of :math:`\log \epsilon_t^2`, consistent for
    :math:`(d, \sigma^2, \phi)` and computed in one FFT. The price is a
    quasi-likelihood on the linearized model, and the summary says so.
    The constructor is inherited from ``_LongMemoryVolatilityModel``.

    Args:
        endog: The observed series (returns, typically), at least 100
            observations.
        mean: ``"constant"`` to remove the sample mean, ``"zero"`` to fix
            the observation mean at zero.
        short_memory: Whether the log variance carries an AR(1) factor as
            well as the fractional one.

    Raises:
        SpecificationError: If ``mean`` is unrecognized.
        DimensionError: If the series is shorter than 100 observations.

    Attributes:
        _endog: The validated series.
        _mean_spec: ``"constant"`` or ``"zero"``.
        _short_memory: Whether ``phi`` is estimated.

    See Also:
        * :class:`LongMemorySVResult` -- the record ``fit()`` returns.
        * :class:`StochasticVolatilitySVAR` -- short-memory stochastic
          volatility as a multivariate identification device, by Gibbs
          sampling rather than Whittle.

    References:
        Breidt, F. J., Crato, N., & de Lima, P. (1998). The detection and
        estimation of long memory in stochastic volatility. *Journal of
        Econometrics*, 83(1-2), 325-348.

        Harvey, A. C., Ruiz, E., & Shephard, N. (1994). Multivariate
        stochastic variance models. *Review of Economic Studies*, 61(2),
        247-264.

    Example:
        Fractional noise with ``d = 0.4`` in the log variance; the pure
        fractional law recovers it, and adding the short-memory factor
        lets the AR(1) absorb most of the persistence -- ``d`` collapses
        toward zero while ``phi`` rises -- with the likelihood-ratio test
        unable to separate the two:

        >>> import numpy as np
        >>> from cultivars._core import fractional_difference_weights
        >>> rng = np.random.default_rng(0)
        >>> n, burn = 1500, 2000
        >>> eta = 0.5 * rng.standard_normal(n + burn)
        >>> weights = fractional_difference_weights(-0.4, n + burn)
        >>> v = np.convolve(eta, weights)[burn : n + burn]
        >>> y = np.exp((-1.0 + v) / 2) * rng.standard_normal(n)
        >>> pure = LongMemorySV(y, mean="zero").fit()
        >>> bool(0.2 < pure.d <= 0.5)
        True
        >>> mixed = LongMemorySV(y, mean="zero", short_memory=True).fit()
        >>> round(mixed.d, 2), round(mixed.phi, 2), mixed.n_params
        (0.01, 0.86, 3.0)
        >>> bool(pure.likelihood_ratio_test(mixed).pvalue > 0.05)
        True
        >>> LongMemorySV(y, mean="median")
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: mean must be 'constant' or 'zero'; got 'median'.
    """

    def fit(self) -> LongMemorySVResult:
        """Estimate by Whittle quasi-likelihood and smooth the log variance.

        Returns:
            The fitted :class:`LongMemorySVResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = LongMemorySV(rng.standard_normal(300)).fit()
            >>> res.mean_spec, res.nobs, bool(abs(res.d) < 0.5)
            ('constant', 300, True)
        """
        return LongMemorySVResult._from_fit(self._fit_whittle(), self)
