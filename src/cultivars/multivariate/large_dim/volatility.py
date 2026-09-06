# filepath: /src/cultivars/multivariate/large_dim/volatility.py
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

"""The BVAR with stochastic volatility: constant transmission, drifting shocks.

Half of what changed in postwar macroeconomic data is the size of the shocks
-- the Great Moderation, its end, the pandemic -- and a homoskedastic VAR
answers questions about that period with a covariance averaged across
regimes nobody believes were the same. This model (Carriero, Clark &
Marcellino; Clark 2011) is the workhorse fix: coefficients constant,
covariance drifting every period through the triangular factorization
``Sigma_t = A^{-1} H_t A^{-T}`` with a constant unit-lower ``A`` and
random-walk log variances -- the Primiceri machinery with the coefficient
drift switched off, which is also exactly how it is assembled here, from the
same Kim-Shephard-Chib block and simulation smoother the TVP family runs on.

Two implementation commitments matter. The coefficient draw is the *joint*
generalized-least-squares conditional across all equations -- exact by
construction, sidestepping the equation-at-a-time factorization whose
original ordering required the 2022 corrigendum -- and because that draw
handles any diagonal prior variance, Litterman's cross-equation weight is
admissible again: conjugacy is not needed where nothing is being solved in
closed form. Dummy-observation priors are refused, with the reason stated:
an artificial row has no date, so under time-varying volatility it has no
covariance to be weighted by.

The covariance the shared surface consumes -- orthogonalized responses, the
predictive -- is the *end-of-sample* one, the covariance relevant for what
comes next; the predictive then propagates the log-volatility random walk
forward, so forecast bands widen with horizon the way drifting volatility
says they must. The full paths live in ``h_draws`` and
:meth:`BVARSVResult.volatility_path`.

References:
    Carriero, A., Clark, T. E., & Marcellino, M. (2019). Large Bayesian
        vector autoregressions with stochastic volatility and non-conjugate
        priors. *Journal of Econometrics*, 212(1), 137-154 (and corrigendum,
        2022).
    Clark, T. E. (2011). Real-time density forecasts from Bayesian vector
        autoregressions with stochastic volatility. *Journal of Business &
        Economic Statistics*, 29(3), 327-341.
    Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
        Likelihood inference and comparison with ARCH models. *Review of
        Economic Studies*, 65(3), 361-393.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable, Trend
from ..._internals import (
    _Prior,
    _VectorPosteriorDrawsResult,
    _VectorVolatilityFit,
    _VolatilityBayesianVectorAutoRegressionModel,
)
from ...bayes.priors import MinnesotaPrior
from ...exceptions import SpecificationError

__all__ = ["BVARSV", "BVARSVResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class BVARSVResult(_VectorPosteriorDrawsResult):
    """A fitted stochastic-volatility BVAR: one system, a path of covariances.

    The family's shared propagation surface, with the covariance read at the
    end of the sample: ``sigma_u`` and the covariance draws are
    ``Sigma_T = A^{-1} diag(exp h_T) A^{-T}``, the covariance relevant for
    what comes next, and the predictive propagates each draw's log-variance
    random walk forward from there. The full history lives in
    :attr:`h_draws` and :meth:`volatility_path`.

    The triangular factorization is taken in the order of ``names`` at
    estimation time, so -- as for the TVP-SV model -- the recursive ordering
    is a structural declaration already made; orthogonalized responses
    inherit it.

    Attributes:
        prior_label: Short description of the prior estimated under.
        h_draws: ``(S, nobs, k)`` kept log-variance path draws.
        impact_draws: ``(S, k, k)`` kept draws of the unit-lower ``A^{-1}``.
        vol_of_vol: ``(k,)`` posterior mean random-walk variances of the
            log volatilities.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    names: tuple[str, ...]
    order: int
    trend: str
    prior_label: str
    coefficients: npt.NDArray[np.float64] = field(repr=False)
    deterministic: npt.NDArray[np.float64] = field(repr=False)
    beta_mean: npt.NDArray[np.float64] = field(repr=False)
    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    beta_draws: npt.NDArray[np.float64] = field(repr=False)
    sigma_draws: npt.NDArray[np.float64] = field(repr=False)
    h_draws: npt.NDArray[np.float64] = field(repr=False)
    impact_draws: npt.NDArray[np.float64] = field(repr=False)
    vol_of_vol: npt.NDArray[np.float64] = field(repr=False)
    resid: npt.NDArray[np.float64] = field(repr=False)
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    nobs: int
    n_draws: int
    n_burn: int
    thin: int

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorVolatilityFit,
        model: _VolatilityBayesianVectorAutoRegressionModel[BVARSVResult],
    ) -> BVARSVResult:
        """Assemble the public result from a raw fit and its specification."""
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
            h_draws=fit.h_draws,
            impact_draws=fit.impact_draws,
            vol_of_vol=fit.vol_of_vol,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            nobs=fit.nobs,
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
        )

    def volatility_path(self, name: str) -> npt.NDArray[np.float64]:
        """One equation's posterior innovation standard deviation path.

        The standard deviation of the *orthogonalized* innovation
        ``exp(h_t / 2)`` -- the object the model actually drifts; the
        reduced-form variances mix these through ``A^{-1}``.

        Args:
            name: An endogenous variable.

        Returns:
            An ``(nobs, 3)`` array with columns ``(low, mean, high)``.

        Raises:
            SpecificationError: If the variable is unknown.
        """
        if name not in self.names:
            raise SpecificationError(f"unknown variable {name!r}; expected one of {self.names}.")
        i = self.names.index(name)
        sd = np.exp(0.5 * self.h_draws[:, :, i])
        return np.column_stack(
            [
                np.quantile(sd, 0.16, axis=0),
                sd.mean(axis=0),
                np.quantile(sd, 0.84, axis=0),
            ]
        )

    def _predictive_noise(
        self, draw: int, steps: int, rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        """Innovations under the draw's own forward-simulated volatility.

        The log variances continue their random walk from the end of the
        sample with the draw's vol-of-vol, so predictive bands widen with
        horizon the way drifting volatility says they must.
        """
        k = self.k_endog
        level = self.h_draws[draw, -1].copy()
        impact = self.impact_draws[draw]
        step_sd = np.sqrt(self.vol_of_vol)
        out = np.empty((steps, k))
        for h in range(steps):
            level = level + step_sd * rng.standard_normal(k)
            out[h] = impact @ (np.exp(0.5 * level) * rng.standard_normal(k))
        return out

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        rows = []
        for name in self.names:
            path = self.volatility_path(name)
            rows.append((name, f"{path[0, 1]:.4f}", f"{path[-1, 1]:.4f}"))
        notes = [
            "The covariance surfaced to irf() and the covariance draws is "
            "the end-of-sample Sigma_T; h_draws and volatility_path() carry "
            "the full history.",
            "The recursive ordering was declared at estimation time through "
            "the triangular factorization, exactly as for TVPVARSV: a "
            "different ordering is a different fitted model.",
            "The predictive propagates each draw's log-variance random walk "
            "forward, so forecast bands widen with horizon.",
            "The coefficient draw is the joint GLS conditional across "
            "equations -- exact, with no equation-ordering artefacts -- so "
            "any diagonal prior variance is admissible, Litterman's "
            "cross-equation weight included; dummy-observation priors are "
            "refused because an artificial row has no date to be "
            "volatility-weighted by.",
            "No likelihood, parameter count, or information criteria are "
            "reported: this is a Gibbs posterior and has none.",
            f"Posterior probability of stability: {self.stable_share:.2f}.",
        ]
        return SummaryTable(
            title=f"BVAR-SV({self.order}) Results",
            metadata=(
                ("Model", f"BVAR-SV({self.order})"),
                ("Prior", self.prior_label),
                ("Draws", f"{self.n_draws} ({self.n_burn} burn, thin {self.thin})"),
                ("Kept", f"{self.n_kept}"),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.nobs}"),
                ("Trend", self.trend),
            ),
            columns=("equation", "innovation sd at start", "at end"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class BVARSV(_VolatilityBayesianVectorAutoRegressionModel[BVARSVResult]):
    """Bayesian VAR with stochastic volatility, Carriero-Clark-Marcellino.

    Constant coefficients under a Minnesota prior -- the full Litterman
    prior, cross-equation weight included, since no conjugacy is needed --
    with per-equation random-walk log volatilities through the triangular
    factorization. Order ``names`` in the recursive ordering you intend:
    the factorization makes it a structural declaration.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order.
        prior: Any finite-variance moment prior; dummy-observation content
            is refused. Defaults to ``MinnesotaPrior()``.
        trend: Deterministic terms.
        names: One label per variable, in the intended recursive order.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((150, 2))
        >>> for t in range(1, 150):
        ...     scale = 0.5 + 1.0 * t / 150
        ...     y[t] = 0.5 * y[t - 1] + scale * rng.standard_normal(2)
        >>> res = BVARSV(y, order=1).fit(n_draws=60, n_burn=20, seed=0)
        >>> res.volatility_path("y1").shape
        (149, 3)
        >>> res.h_draws.shape[1:]
        (149, 2)
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
        """Default the prior to the full Minnesota, then validate."""
        super().__init__(
            endog,
            order=order,
            trend=trend,
            names=names,
            prior=prior if prior is not None else MinnesotaPrior(),
        )

    def fit(
        self,
        *,
        n_draws: int = 2000,
        n_burn: int = 1000,
        thin: int = 2,
        k_vol: float = 0.01,
        seed: int | np.random.Generator | None = None,
    ) -> BVARSVResult:
        """Estimate by Gibbs with the KSC volatility block.

        Args:
            n_draws: Total sampler iterations.
            n_burn: Burn-in iterations discarded.
            thin: Keep every ``thin``-th post-burn draw.
            k_vol: Prior scale of the log-volatility random walk; raise
                toward 0.1 when volatility drifts substantially.
            seed: Seed or generator, for reproducibility.

        Returns:
            The fitted :class:`BVARSVResult`.

        Raises:
            SpecificationError: If the draw bookkeeping is inconsistent or
                the prior is unusable.
            NumericalError: If a conditional draw collapses.
        """
        return BVARSVResult._from_fit(
            self._fit_volatility(n_draws=n_draws, n_burn=n_burn, thin=thin, k_vol=k_vol, seed=seed),
            self,
        )
