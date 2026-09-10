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

"""Identification through stochastic volatility: the regimes nobody has to declare.

Rigobon's scheme buys identification from variance shifts, but it needs the
user to say *when* the shifts happen. This scheme drops that requirement.
Give each structural shock its own latent log-AR(1) variance and the
reduced-form covariance ``Sigma_t = B diag(exp(h_t)) B'`` moves every
period; as long as the variance paths are not proportional to one another,
the sequence of covariances pins ``B`` down up to column order and sign
(Lewis 2021; Bertsche and Braun 2022). No zero, no sign pattern, no
instrument, and no regime dates -- the volatility processes are estimated
along with the impact matrix.

Estimation is Bayesian, by an exact Gibbs sampler on the reduced-form
innovations of a fitted closed system: the rows of ``A = B**-1`` by
Waggoner and Zha's (2003) construction given the paths, the paths by the
Kim-Shephard-Chib mixture step given ``A``, and each path's persistence and
innovation variance from their conditionals. What comes back is a posterior
over impact matrices, not a point, and the object holds it: the quantile
surfaces of impulse responses summarize a set of complete structural models,
none of which traces any one surface.

Three things are said rather than hidden. The shocks are statistical
objects, labelled by which variable they load on most, not by economics;
an economic name is a claim to be argued from outside the model. The
posterior cannot certify its own identification -- with proportional
variance paths the likelihood is flat over rotations and the volatility
prior still returns a sharp answer -- so the identifying condition is
tested on the data, pair by pair, and the verdict is printed with the
result. And the scale: the likelihood cannot tell a path's level from its
column's scale, so every draw is expressed with zero sample-mean log
variance per shock and the level folded into ``B`` -- the impact of a
shock at its in-sample average variance.

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
    _validate_quantiles,
    _variance_ratio_test,
)
from ..._internals import _SummaryMixin, _VolatilityIdentificationModel, _VolatilityStructuralFit
from ...exceptions import SpecificationError
from .zero_restrictions import SVARResult

__all__ = ["StochasticVolatilitySVAR", "StochasticVolatilitySVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class StochasticVolatilitySVARResult(_SummaryMixin):
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
        prior_phi: tuple[float, float] = (20.0, 1.5),
        prior_sigma2: tuple[float, float] = (2.5, 0.025),
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
            prior_phi: ``(a, b)`` of the Beta prior on ``(phi + 1) / 2``.
            prior_sigma2: ``(shape, rate)`` of the inverse-gamma prior.
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
            prior_phi=prior_phi,
            prior_sigma2=prior_sigma2,
            seed=seed,
        )
        return StochasticVolatilitySVARResult._from_fit(fit, self)
