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
r"""The Student-t Bayesian VAR: fat tails as a model, not a casualty.

Macroeconomic samples carry a handful of dates -- 2008, 2020 -- that a
Gaussian VAR can only accommodate by inflating the covariance for every
other date, which is how a few observations end up owning the error bands.
Student-t innovations (Chiu, Mumtaz and Pinter, 2017) treat those dates as
what they are: draws from the same system's fatter tail. The model is

.. math::

   y_t = c + \sum_{i=1}^{p} A_i y_{t-i} + u_t,
   \qquad u_t \sim t_\nu(0, \Sigma),
   \qquad
   u_t \mid w_t \sim N(0, \Sigma / w_t),\quad
   w_t \sim \mathrm{Gamma}(\nu/2, \nu/2),

and the second line is how it is estimated (Geweke, 1993): each date
carries a latent precision weight, and conditional on the weights the
model is exactly the conjugate Normal-inverse-Wishart VAR on rows rescaled
by :math:`\sqrt{w_t}`. The Gibbs sampler therefore alternates three exact
conditionals -- :math:`(B, \Sigma)` from the weighted conjugate update,
the weights from :math:`\mathrm{Gamma}\bigl((\nu + k)/2,
(\nu + u_t' \Sigma^{-1} u_t)/2\bigr)`, and :math:`\nu` from its
conditional on a fixed grid -- and inherits the conjugate model's prior
machinery unchanged, dummy observations included. The reported
:math:`\Sigma` is the t *scale*; the innovation covariance is
:math:`\Sigma\,\nu/(\nu - 2)`.

Two commitments shape the surface. First, the tail index can be stated or
learned, and when learned it is learned exactly: ``nu`` is drawn from its
conditional evaluated on the grid ``2 .. 29, 30, 35, 40, 50, 60, 80, 100``
under a prior uniform over grid points, a step with no tuning parameter
and no acceptance rate. The grid is coarse above thirty because the t and
the Gaussian are indistinguishable there; on data whose innovations are
Gaussian the posterior settles in the teens and twenties at macro sample
sizes, which is the model's way of saying the tails were not needed, and
only a long Gaussian sample pushes it to the grid's top. Second, the
latent weights are reported, as :attr:`StudentBVARResult.weight_mean` -- a
per-date outlier map, and often the most readable diagnostic the fit
produces: the dates the t distribution absorbed are exactly the dates with
small weights, and :meth:`StudentBVARResult.outlier_dates` lists them.

Deliberately absent: the marginal likelihood. The t likelihood breaks the
conjugacy that made the Gaussian model's evidence a closed form, and a
simulated stand-in would not deserve the name; rank tail specifications
by predictive performance instead.

Layout. :class:`StudentBVAR` defaults the prior to
:class:`~cultivars.bayes.priors.NormalInverseWishartPrior` and validates
on ``_StudentBayesianVectorAutoRegressionModel`` in ``_internals``, which
extends the conjugate ``_BayesianVectorAutoRegressionModel``; its
``_fit_student`` runs the three-block sampler, drawing ``(B, Sigma)``
through ``_draw_conjugate`` on the reweighted rows and ``nu`` through
``_draw_degrees`` on ``_STUDENT_DF_GRID`` from ``_core``, and packs a
``_VectorStudentFit``. :class:`StudentBVARResult` extends
``_VectorPosteriorDrawsResult`` -- credible intervals, impulse responses
with posterior bands, the predictive, the stability share, the chain
diagnostics -- and overrides its three noise hooks so that the
predictive, :meth:`~StudentBVARResult.simulate` and
:meth:`~StudentBVARResult.posterior_replications` all draw t innovations.
The Gaussian conjugate model is
:mod:`~cultivars.multivariate.large_dim.bayesian`; the other account of
fat marginal tails, through time-varying volatility, is
:mod:`~cultivars.multivariate.large_dim.volatility`.

References:
    Chiu, C.-W. J., Mumtaz, H., & Pinter, G. (2017). Forecasting with VAR
    models: Fat tails and stochastic volatility. *International Journal
    of Forecasting*, 33(4), 1124-1143.

    Geweke, J. (1993). Bayesian treatment of the independent Student-t
    linear model. *Journal of Applied Econometrics*, 8(S1), S19-S40.

Example:
    A bivariate VAR(1) with :math:`t_4` innovations next to one with
    Gaussian innovations: the tail index separates them, and the
    weight map on the fat-tailed sample points at its largest residuals:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> fat, normal = np.zeros((200, 2)), np.zeros((200, 2))
    >>> for t in range(1, 200):
    ...     w = rng.gamma(2.0, 0.5)
    ...     fat[t] = 0.5 * fat[t - 1] + rng.standard_normal(2) / np.sqrt(w)
    ...     normal[t] = 0.5 * normal[t - 1] + rng.standard_normal(2)
    >>> heavy = StudentBVAR(fat, order=1).fit(n_draws=400, n_burn=200, seed=0)
    >>> light = StudentBVAR(normal, order=1).fit(n_draws=400, n_burn=200, seed=0)
    >>> bool(heavy.df < 10.0 < light.df)
    True
    >>> bool(heavy.outlier_dates().size > light.outlier_dates().size)
    True
    >>> worst = int(np.argmax(np.abs(heavy.resid).max(axis=1)))
    >>> bool(worst in heavy.outlier_dates())
    True
    >>> ratio = heavy.innovation_covariance / heavy.sigma_u
    >>> bool(np.allclose(ratio, heavy.df / (heavy.df - 2.0)))
    True
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ...bayes.priors import MinnesotaPrior
from ...engine._core import SummaryTable, Trend
from ...engine._internals import (
    _Prior,
    _VectorPosteriorDrawsResult,
    _VectorVolatilityFit,
    _VolatilityBayesianVectorAutoRegressionModel,
)
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

    def _replication_noise(self, draw: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
        """Innovations under the draw's own in-sample volatility path.

        A replication conditions on the draw's log-variance path ``h_t``
        rather than simulating a fresh random walk: the walk has no
        stationary distribution to start from, and what the check asks
        is whether the dynamics reproduce the data *given* a volatility
        history the posterior finds plausible.
        """
        scale = np.exp(0.5 * self.h_draws[draw])
        shocks = scale * rng.standard_normal(scale.shape)
        return np.asarray(shocks @ self.impact_draws[draw].T, dtype=np.float64)

    def _replication_notes(self) -> tuple[str, ...]:
        return (
            "Replications condition on each draw's in-sample volatility path: the check "
            "reads the dynamics given the volatility history, not the volatility law itself.",
        )

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
