# filepath: /src/cultivars/multivariate/large_dim/student.py
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

"""The Student-t Bayesian VAR: fat tails as a model, not a casualty.

Macroeconomic samples carry a handful of dates -- 2008, 2020 -- that a
Gaussian VAR can only accommodate by inflating the covariance for every
other date, which is how a few observations end up owning the error bands.
Student-t innovations (Chiu, Mumtaz & Pinter) treat those dates as what they
are: draws from the same system's fatter tail. Estimation works through the
scale-mixture representation -- each observation carries a latent precision
weight, and conditional on the weights the model is exactly the conjugate
Normal-inverse-Wishart VAR on rescaled rows -- so the Gibbs sampler
alternates exact conditionals and inherits the conjugate model's prior
machinery unchanged, dummy observations included.

The tail index can be stated or learned. Left unstated, ``nu`` gets a
posterior of its own, drawn exactly on a fixed grid; on Gaussian data that
posterior piles up at the top of the grid, which is the model's way of
saying the tails were not needed. The latent weights are reported as
:attr:`StudentBVARResult.weight_mean` -- a per-date outlier map, and often
the most readable diagnostic the fit produces: the dates the t distribution
absorbed are exactly the dates with small weights.

Deliberately absent: the marginal likelihood. The t likelihood breaks the
conjugacy that made the Gaussian model's evidence a closed form, and a
simulated stand-in would not deserve the name; rank tail specifications by
predictive performance instead.

References:
    Chiu, C.-W. J., Mumtaz, H., & Pinter, G. (2017). Forecasting with VAR
        models: Fat tails and stochastic volatility. *International Journal
        of Forecasting*, 33(4), 1124-1143.
    Geweke, J. (1993). Bayesian treatment of the independent Student-t
        linear model. *Journal of Applied Econometrics*, 8(S1), S19-S40.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable, Trend
from ..._internals import (
    _Prior,
    _StudentBayesianVectorAutoRegressionModel,
    _VectorPosteriorDrawsResult,
    _VectorStudentFit,
)
from ...bayes.priors import NormalInverseWishartPrior
from ...exceptions import SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class StudentBVARResult(_VectorPosteriorDrawsResult):
    """A fitted Student-t Bayesian VAR: the tails carry a posterior too.

    The family's shared propagation surface over Gibbs draws, with two
    readings specific to the t model. :attr:`sigma_u` and the covariance
    draws are the t distribution's *scale* matrix -- the innovation
    covariance is ``scale * df / (df - 2)`` -- and the predictive draws its
    innovations from the t, so forecast bands carry the fat tails forward.

    Deliberately absent: a marginal likelihood; the module docstring states
    why.

    Attributes:
        prior_label: Short description of the prior estimated under.
        df: Posterior mean degrees of freedom -- the stated value when the
            caller fixed it.
        df_draws: ``(S,)`` kept degrees-of-freedom draws; empty when fixed.
        weight_mean: ``(n,)`` posterior mean latent precision weights,
            aligned with ``resid``. Small values mark the dates the t
            distribution treats as tail events.
        n_dummy: Artificial rows the prior contributed.
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
    df: float
    df_draws: npt.NDArray[np.float64] = field(repr=False)
    weight_mean: npt.NDArray[np.float64] = field(repr=False)
    resid: npt.NDArray[np.float64] = field(repr=False)
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    nobs: int
    n_dummy: int
    n_draws: int
    n_burn: int
    thin: int

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorStudentFit,
        model: _StudentBayesianVectorAutoRegressionModel[StudentBVARResult],
    ) -> StudentBVARResult:
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
            df=fit.df,
            df_draws=fit.df_draws,
            weight_mean=fit.weight_mean,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            nobs=fit.nobs,
            n_dummy=fit.n_dummy,
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
        )

    @property
    def df_estimated(self) -> bool:
        """Whether the degrees of freedom carry a posterior."""
        return bool(self.df_draws.shape[0])

    def df_interval(self) -> npt.NDArray[np.float64]:
        """The tail index's posterior summary: 16th percentile, mean, 84th.

        Returns:
            A ``(3,)`` array ``(low, mean, high)``.

        Raises:
            SpecificationError: If the degrees of freedom were stated rather
                than estimated.
        """
        if not self.df_estimated:
            raise SpecificationError(
                f"df was fixed at {self.df} by the caller and has no "
                "posterior; refit with df=None to estimate the tail index."
            )
        return np.array(
            [
                float(np.quantile(self.df_draws, 0.16)),
                float(self.df_draws.mean()),
                float(np.quantile(self.df_draws, 0.84)),
            ]
        )

    def outlier_dates(self, threshold: float = 0.5) -> npt.NDArray[np.intp]:
        """Effective-sample rows the t distribution treats as tail events.

        Args:
            threshold: Weight below which a date counts -- ``1`` is a
                typical date, and the 2008-shaped dates sit far below.

        Returns:
            Row indices into ``resid``, ascending.
        """
        return np.flatnonzero(self.weight_mean < threshold)

    @property
    def innovation_covariance(self) -> npt.NDArray[np.float64]:
        """The innovation covariance ``scale * df / (df - 2)``.

        Reported alongside the scale rather than instead of it, because the
        scale is what the structural factorization and the predictive
        consume. Requires ``df > 2``, which the estimator guarantees.
        """
        return self.sigma_u * self.df / (self.df - 2.0)

    def _predictive_noise(
        self, draw: int, steps: int, rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        """Student-t innovations: the predictive keeps the fat tails."""
        nu = float(self.df_draws[draw]) if self.df_estimated else self.df
        chol = np.linalg.cholesky(self.sigma_draws[draw])
        mixing = np.asarray(rng.gamma(0.5 * nu, 2.0 / nu, size=steps), dtype=np.float64)
        shocks = rng.standard_normal((steps, self.k_endog)) / np.sqrt(mixing)[:, None]
        return np.asarray(shocks @ chol.T, dtype=np.float64)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        rows = []
        for name in self.names:
            if self.order:
                low, mid, high = self.credible_interval(name, f"{name}.L1")
                rows.append((name, f"{mid:.4f}", f"[{low:.4f}, {high:.4f}]"))
            else:
                rows.append((name, "-", "-"))
        if self.df_estimated:
            low, mid, high = self.df_interval()
            df_note = (
                f"Degrees of freedom estimated on a grid: posterior mean "
                f"{mid:.1f}, 68% interval [{low:.1f}, {high:.1f}]. Mass at "
                "the grid's top says the tails were not needed."
            )
        else:
            df_note = f"Degrees of freedom fixed at {self.df:g} by the caller."
        outliers = self.outlier_dates()
        notes = [
            df_note,
            f"{outliers.size} of {self.nobs} dates carry posterior mean "
            "precision weight below 0.5 -- the dates the t distribution "
            "absorbed as tail events; weight_mean is the per-date map.",
            "sigma_u is the t scale matrix; the innovation covariance is "
            "scale * df / (df - 2), reported as innovation_covariance.",
            "The predictive draws t innovations, so forecast bands carry the fat tails forward.",
            "No marginal likelihood is reported: the t likelihood has no "
            "closed-form evidence, and a simulated stand-in would not "
            "deserve the name.",
            f"Posterior probability of stability: {self.stable_share:.2f}.",
        ]
        return SummaryTable(
            title=f"Student-t BVAR({self.order}) Results",
            metadata=(
                ("Model", f"BVAR-t({self.order})"),
                ("Prior", self.prior_label),
                ("Draws", f"{self.n_draws} ({self.n_burn} burn, thin {self.thin})"),
                ("Kept", f"{self.n_kept}"),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.nobs}"),
                ("Trend", self.trend),
                ("df", f"{self.df:.1f}" + (" (estimated)" if self.df_estimated else "")),
            ),
            columns=("equation", "own L1 mean", "68% interval"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class StudentBVAR(_StudentBayesianVectorAutoRegressionModel[StudentBVARResult]):
    """Bayesian VAR with Student-t innovations, Chiu-Mumtaz-Pinter.

    The same conjugacy-compatible prior surface as :class:`BVAR` -- the
    weighted update is still Normal-inverse-Wishart -- with the Gaussian
    likelihood swapped for the t through its scale-mixture form.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order.
        prior: A conjugacy-compatible prior, possibly a composition.
            Defaults to ``NormalInverseWishartPrior()``.
        trend: Deterministic terms.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((200, 2))
        >>> for t in range(1, 200):
        ...     w = rng.gamma(2.0, 0.5)
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2) / np.sqrt(w)
        >>> res = StudentBVAR(y, order=1).fit(n_draws=120, n_burn=40, seed=0)
        >>> res.beta_draws.shape[1:]
        (3, 2)
        >>> res.weight_mean.shape
        (199,)
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
        """Default the prior to the conjugate Minnesota, then validate."""
        super().__init__(
            endog,
            order=order,
            trend=trend,
            names=names,
            prior=prior if prior is not None else NormalInverseWishartPrior(),
        )

    def fit(
        self,
        *,
        df: float | None = None,
        n_draws: int = 2000,
        n_burn: int = 1000,
        thin: int = 2,
        seed: int | np.random.Generator | None = None,
    ) -> StudentBVARResult:
        """Estimate by Gibbs on the scale-mixture representation.

        Args:
            df: Degrees of freedom, above two -- or ``None`` (the default)
                to give the tail index a posterior of its own.
            n_draws: Total sampler iterations.
            n_burn: Burn-in iterations discarded.
            thin: Keep every ``thin``-th post-burn draw.
            seed: Seed or generator, for reproducibility.

        Returns:
            The fitted :class:`StudentBVARResult`.

        Raises:
            SpecificationError: If the draw bookkeeping is inconsistent or
                the prior is unusable.
            NumericalError: If a conditional draw collapses.
        """
        return StudentBVARResult._from_fit(
            self._fit_student(df=df, n_draws=n_draws, n_burn=n_burn, thin=thin, seed=seed),
            self,
        )
