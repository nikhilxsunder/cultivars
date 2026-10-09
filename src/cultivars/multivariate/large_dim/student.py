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

from ...bayes.priors import NormalInverseWishartPrior
from ...engine._core import SummaryTable, Trend
from ...engine._internals import (
    _Prior,
    _StudentBayesianVectorAutoRegressionModel,
    _VectorPosteriorDrawsResult,
    _VectorStudentFit,
)
from ...exceptions import SpecificationError

__all__ = ["StudentBVAR", "StudentBVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class StudentBVARResult(_VectorPosteriorDrawsResult):
    r"""A fitted Student-t Bayesian VAR: the tails carry a posterior too.

    The posterior of

    .. math::

       y_t = c + \sum_{i=1}^{p} A_i y_{t-i} + u_t,
       \qquad u_t \sim t_\nu(0, \Sigma),

    reached through the scale-mixture representation
    :math:`u_t \mid w_t \sim N(0, \Sigma / w_t)`,
    :math:`w_t \sim \mathrm{Gamma}(\nu/2, \nu/2)`: conditional on the
    weights the model is the conjugate Normal-inverse-Wishart VAR on rows
    rescaled by :math:`\sqrt{w_t}`, so the Gibbs sampler alternates exact
    conditionals for :math:`(B, \Sigma)`, for the weights, and -- when
    :math:`\nu` is not stated -- for :math:`\nu` on a fixed grid. The
    propagation surface is the family's shared one from
    ``_VectorPosteriorDrawsResult``; what this result adds is the tail
    index's posterior, the per-date weights, and three noise hooks that
    make every propagated object -- predictive, simulation, replication
    -- draw its innovations from the t rather than the Gaussian.

    Note:
        :attr:`sigma_u` and :attr:`sigma_draws` are the t distribution's
        *scale* matrix :math:`\Sigma`, not its covariance; the innovation
        covariance is :math:`\Sigma\,\nu / (\nu - 2)` and is reported as
        :attr:`innovation_covariance`. The scale is what the structural
        factorization and the predictive consume, which is why it keeps
        the family's name. ``df`` is the posterior mean of the kept
        :math:`\nu` draws when estimated and the stated value when fixed;
        the grid runs over the integers ``2 .. 29`` and then
        ``30, 35, 40, 50, 60, 80, 100``, so an estimated ``df`` is a mean
        of grid points and ``df_interval`` reports grid quantiles.
        ``weight_mean`` is the posterior mean of :math:`w_t`, aligned with
        ``resid``: a typical date sits near one, and a date the t absorbed
        as a tail event sits well below it. No marginal likelihood is
        reported; the module docstring states why.

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

    See Also:
        * :class:`StudentBVAR` -- the model that produces this record.
        * :class:`~cultivars.multivariate.large_dim.bayesian.BVARResult`
          -- the Gaussian conjugate posterior this one relaxes.
        * :class:`~cultivars.multivariate.large_dim.volatility.BVARSVResult`
          -- the other route to fat marginal tails, through time-varying
          volatility.

    References:
        Chiu, C.-W. J., Mumtaz, H., & Pinter, G. (2017). Forecasting with
        VAR models: Fat tails and stochastic volatility. *International
        Journal of Forecasting*, 33(4), 1124-1143.

        Geweke, J. (1993). Bayesian treatment of the independent Student-t
        linear model. *Journal of Applied Econometrics*, 8(S1), S19-S40.

    Example:
        A bivariate VAR(1) whose innovations are :math:`t_4`, generated
        through the same scale mixture the sampler uses. The tail index
        is recovered in the single digits, the dates with the largest
        residuals carry the smallest weights, and the scale sits below the
        covariance by the factor :math:`\nu / (\nu - 2)`:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((200, 2))
        >>> for t in range(1, 200):
        ...     w = rng.gamma(2.0, 0.5)
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2) / np.sqrt(w)
        >>> res = StudentBVAR(y, order=1).fit(n_draws=400, n_burn=200, seed=0)
        >>> res.prior_label, res.n_kept, res.nobs, res.n_dummy
        ('niw(l1=0.2, l3=1, l4=100)', 100, 199, 0)
        >>> res.df_estimated, bool(3.0 < res.df < 10.0), res.df_draws.shape
        (True, True, (100,))
        >>> res.beta_draws.shape, res.sigma_draws.shape, res.weight_mean.shape
        ((100, 3, 2), (100, 2, 2), (199,))
        >>> largest = np.argsort(np.abs(res.resid).max(axis=1))[-5:]
        >>> bool(res.weight_mean[largest].max() < 0.5), bool(0.9 < res.weight_mean.mean() < 1.1)
        (True, True)
        >>> ratio = np.diag(res.innovation_covariance) / np.diag(res.sigma_u)
        >>> bool(np.allclose(ratio, res.df / (res.df - 2.0)))
        True
        >>> low, mid, high = res.credible_interval("y1", "y1.L1")
        >>> bool(0.4 < low < mid < high < 0.65)
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
    """``(k, k)`` posterior mean t *scale* matrix, not the covariance. Kept out of the repr."""
    beta_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, w, k)`` coefficient draws. Kept out of the repr."""
    sigma_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S, k, k)`` scale-matrix draws. Kept out of the repr."""
    df: float
    """Degrees of freedom: posterior mean of the kept draws, or the stated value."""
    df_draws: npt.NDArray[np.float64] = field(repr=False)
    """``(S,)`` kept degrees-of-freedom draws; shape ``(0,)`` when fixed. Kept out of the repr."""
    weight_mean: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs,)`` posterior mean latent precision weights, aligned with ``resid``."""
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

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorStudentFit,
        model: _StudentBayesianVectorAutoRegressionModel[StudentBVARResult],
    ) -> StudentBVARResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The packed sampler output.
            model: The validated specification it was estimated on.

        Returns:
            The frozen :class:`StudentBVARResult`.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> model = StudentBVAR(y, order=1)
            >>> fit = model._fit_student(df=5.0, n_draws=30, n_burn=10, thin=1, seed=0)
            >>> res = StudentBVARResult._from_fit(fit, model)
            >>> res.n_kept, res.df, res.prior_label
            (20, 5.0, 'niw(l1=0.2, l3=1, l4=100)')
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
        """Whether the degrees of freedom carry a posterior.

        ``True`` when ``fit`` was called with ``df=None`` and ``df_draws``
        holds one value per kept draw; ``False`` when the caller stated
        the tail index.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> model = StudentBVAR(y, order=1)
            >>> model.fit(n_draws=30, n_burn=10, seed=0).df_estimated
            True
            >>> model.fit(df=5.0, n_draws=30, n_burn=10, seed=0).df_estimated
            False
        """
        return bool(self.df_draws.shape[0])

    def df_interval(self) -> npt.NDArray[np.float64]:
        """The tail index's posterior summary: 16th percentile, mean, 84th.

        The quantiles are of the kept grid draws, so they are grid points;
        the mean is ``df``.

        Returns:
            A ``(3,)`` array ``(low, mean, high)``.

        Raises:
            SpecificationError: If the degrees of freedom were stated rather
                than estimated.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((200, 2))
            >>> for t in range(1, 200):
            ...     w = rng.gamma(2.0, 0.5)
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2) / np.sqrt(w)
            >>> res = StudentBVAR(y, order=1).fit(n_draws=400, n_burn=200, seed=0)
            >>> low, mean, high = res.df_interval()
            >>> bool(low < mean < high), bool(mean == res.df), bool(high < 15.0)
            (True, True, True)
            >>> fixed = StudentBVAR(y, order=1).fit(df=5.0, n_draws=30, n_burn=10, seed=0)
            >>> fixed.df_interval()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: df was fixed at 5.0 by the caller and has ...
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
        r"""Effective-sample rows the t distribution treats as tail events.

        A weight :math:`w_t` scales the innovation covariance at date
        ``t`` to :math:`\Sigma / w_t`, so a weight of one half is a date
        the model reads as carrying twice the usual innovation variance.

        Args:
            threshold: Weight below which a date counts -- ``1`` is a
                typical date, and the 2008-shaped dates sit far below.

        Returns:
            Row indices into ``resid``, ascending.

        Example:
            The flagged dates are the large-residual dates, and the set
            shrinks as the threshold tightens:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((200, 2))
            >>> for t in range(1, 200):
            ...     w = rng.gamma(2.0, 0.5)
            ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2) / np.sqrt(w)
            >>> res = StudentBVAR(y, order=1).fit(n_draws=400, n_burn=200, seed=0)
            >>> flagged = res.outlier_dates()
            >>> bool(0 < flagged.size < 30), bool(np.all(np.diff(flagged) > 0))
            (True, True)
            >>> largest = int(np.argmax(np.abs(res.resid).max(axis=1)))
            >>> bool(largest in flagged), bool(res.outlier_dates(0.25).size <= flagged.size)
            (True, True)
        """
        return np.flatnonzero(self.weight_mean < threshold)

    @property
    def innovation_covariance(self) -> npt.NDArray[np.float64]:
        """The innovation covariance ``scale * df / (df - 2)``.

        Reported alongside the scale rather than instead of it, because the
        scale is what the structural factorization and the predictive
        consume. Requires ``df > 2``, which the estimator guarantees.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = StudentBVAR(y, order=1).fit(df=5.0, n_draws=30, n_burn=10, seed=0)
            >>> bool(np.allclose(res.innovation_covariance, res.sigma_u * 5.0 / 3.0))
            True
        """
        return self.sigma_u * self.df / (self.df - 2.0)

    def _predictive_noise(
        self, draw: int, steps: int, rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        """Student-t innovations: the predictive keeps the fat tails.

        Overrides the family's Gaussian hook. Each step draws its own
        mixing weight ``w ~ Gamma(nu/2, nu/2)`` and divides a Gaussian
        vector with the draw's scale matrix by ``sqrt(w)``, with ``nu`` the
        draw's own degrees of freedom when estimated and the stated value
        otherwise.

        Args:
            draw: Index of the kept draw.
            steps: Innovation rows to generate.
            rng: Random generator.

        Returns:
            An array of shape ``(steps, k)``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = StudentBVAR(y, order=1).fit(df=3.0, n_draws=30, n_burn=10, seed=0)
            >>> noise = res._predictive_noise(0, 5000, np.random.default_rng(0))
            >>> noise.shape
            (5000, 2)
            >>> scaled = noise / np.sqrt(np.diag(res.sigma_draws[0]))
            >>> bool(np.mean(np.abs(scaled) > 3.0) > 0.02)
            True
        """
        nu = float(self.df_draws[draw]) if self.df_estimated else self.df
        chol = np.linalg.cholesky(self.sigma_draws[draw])
        mixing = np.asarray(rng.gamma(0.5 * nu, 2.0 / nu, size=steps), dtype=np.float64)
        shocks = rng.standard_normal((steps, self.k_endog)) / np.sqrt(mixing)[:, None]
        return np.asarray(shocks @ chol.T, dtype=np.float64)

    def _simulation_noise(
        self, draw: int | None, n: int, rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        """Student-t innovations at one draw, or at the posterior mean when ``None``.

        At a draw this is :meth:`_predictive_noise`; at the posterior mean
        the scale is ``sigma_u`` and the degrees of freedom are ``df``.

        Args:
            draw: Index of a kept draw, or ``None`` for the posterior mean.
            n: Innovation rows to generate.
            rng: Random generator.

        Returns:
            An array of shape ``(n, k)``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = StudentBVAR(y, order=1).fit(df=5.0, n_draws=30, n_burn=10, seed=0)
            >>> res._simulation_noise(None, 7, np.random.default_rng(0)).shape
            (7, 2)
            >>> res._simulation_noise(3, 7, np.random.default_rng(0)).shape
            (7, 2)
        """
        if draw is not None:
            return self._predictive_noise(draw, n, rng)
        chol = np.linalg.cholesky(self.sigma_u)
        mixing = np.asarray(rng.gamma(0.5 * self.df, 2.0 / self.df, size=n), dtype=np.float64)
        shocks = rng.standard_normal((n, self.k_endog)) / np.sqrt(mixing)[:, None]
        return np.asarray(shocks @ chol.T, dtype=np.float64)

    def _replication_noise(self, draw: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
        """Student-t in-sample innovations: a replication keeps the fat tails.

        Args:
            draw: Index of the kept draw.
            rng: Random generator.

        Returns:
            An array of shape ``(nobs, k)`` -- one row per effective-sample
            date, the shape :meth:`posterior_replications` expects.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = StudentBVAR(y, order=1).fit(df=5.0, n_draws=30, n_burn=10, seed=0)
            >>> res._replication_noise(0, np.random.default_rng(0)).shape
            (119, 2)
        """
        n = self.endog.shape[0] - self.order
        return self._predictive_noise(draw, n, rng)

    def _replication_notes(self) -> tuple[str, ...]:
        """The predictive check's remark on how this result replicates.

        Returns:
            One sentence naming the innovation law and whether its degrees
            of freedom are drawn or stated.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = StudentBVAR(y, order=1).fit(df=5.0, n_draws=30, n_burn=10, seed=0)
            >>> res._replication_notes()
            ('Replications draw Student-t innovations with the stated 5 degrees of freedom.',)
        """
        tails = "its own drawn" if self.df_estimated else f"the stated {self.df:g}"
        return (f"Replications draw Student-t innovations with {tails} degrees of freedom.",)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Returns:
            One row per equation with its own first-lag posterior mean and
            68% credible interval; the metadata carries the draw
            bookkeeping and the degrees of freedom, marked ``(estimated)``
            when they carry a posterior; the notes state the tail index's
            interval or fixed value, the count of tail-event dates, the
            scale-versus-covariance reading, the t predictive, the absent
            marginal likelihood, and the stability share.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> res = StudentBVAR(y, order=1).fit(df=5.0, n_draws=30, n_burn=10, seed=0)
            >>> table = res._summary_table()
            >>> table.title, table.columns, len(table.rows), len(table.notes)
            ('Student-t BVAR(1) Results', ('equation', 'own L1 mean', '68% interval'), 2, 6)
            >>> table.metadata[-1], table.notes[0]
            (('df', '5.0'), 'Degrees of freedom fixed at 5 by the caller.')
        """
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
    r"""Bayesian VAR with Student-t innovations, after Chiu, Mumtaz and Pinter.

    The same conjugacy-compatible prior surface as
    :class:`~cultivars.multivariate.large_dim.bayesian.BVAR` -- Minnesota
    moments, Normal-inverse-Wishart, dummy observations and their
    compositions -- with the Gaussian likelihood replaced by the t through
    its scale mixture of normals. Each date carries a latent precision
    weight :math:`w_t \sim \mathrm{Gamma}(\nu/2, \nu/2)`; conditional on
    the weights the model is exactly the conjugate VAR on rows rescaled by
    :math:`\sqrt{w_t}`, so the weighted update is still Normal-inverse-
    Wishart and the prior machinery is inherited unchanged. Dummy rows
    enter unweighted: they are prior content, and the tails belong to the
    data. The tail index :math:`\nu` is either stated above two or given a
    posterior of its own, drawn exactly on a fixed grid.

    Attributes:
        _endog: The observed panel, from the base model.
        _order: The autoregressive order ``p``, from the base model.
        _trend: The deterministic specification, from the base model.
        _names: The variable labels, from the base model.
        _prior: The composed prior, from the base model.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order.
        prior: A conjugacy-compatible prior, possibly a composition.
            Defaults to ``NormalInverseWishartPrior()``.
        trend: Deterministic terms.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Raises:
        SpecificationError: If the panel, order, trend or names are
            malformed, or the prior is not conjugacy-compatible.

    See Also:
        * :class:`StudentBVARResult` -- the record ``fit`` returns.
        * :class:`~cultivars.multivariate.large_dim.bayesian.BVAR` -- the
          Gaussian conjugate model, with its closed-form evidence.
        * :class:`~cultivars.multivariate.large_dim.volatility.BVARSV`
          -- stochastic volatility, the other account of fat tails.
        * :class:`~cultivars.bayes.priors.NormalInverseWishartPrior` -- the
          default prior.

    References:
        Chiu, C.-W. J., Mumtaz, H., & Pinter, G. (2017). Forecasting with
        VAR models: Fat tails and stochastic volatility. *International
        Journal of Forecasting*, 33(4), 1124-1143.

        Geweke, J. (1993). Bayesian treatment of the independent Student-t
        linear model. *Journal of Applied Econometrics*, 8(S1), S19-S40.

    Example:
        Innovations drawn as a :math:`t_4` scale mixture; the sampler
        recovers single-digit degrees of freedom, and the Gaussian
        conjugate fit on the same data inflates its covariance to cover
        the tail dates while the t fit's scale sits below it:

        >>> import numpy as np
        >>> from cultivars.multivariate.large_dim.bayesian import BVAR
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((200, 2))
        >>> for t in range(1, 200):
        ...     w = rng.gamma(2.0, 0.5)
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2) / np.sqrt(w)
        >>> res = StudentBVAR(y, order=1).fit(n_draws=400, n_burn=200, seed=0)
        >>> res.beta_draws.shape[1:], res.weight_mean.shape
        ((3, 2), (199,))
        >>> bool(3.0 < res.df < 10.0)
        True
        >>> gaussian = BVAR(y, order=1).fit(n_draws=100, seed=0)
        >>> bool(np.all(np.diag(res.sigma_u) < np.diag(gaussian.sigma_u)))
        True
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
        """Default the prior to the conjugate Minnesota, then validate.

        Args:
            endog: The observed panel, shape ``(nobs, k)``.
            order: Autoregressive order.
            prior: A conjugacy-compatible prior, or ``None`` for
                ``NormalInverseWishartPrior()``.
            trend: Deterministic terms.
            names: One label per variable, or ``None`` for ``y1 ... yk``.

        Raises:
            SpecificationError: If the specification or the prior is
                unusable.

        Example:
            >>> import numpy as np
            >>> from cultivars.bayes.priors import NormalInverseWishartPrior
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> model = StudentBVAR(y, order=2, names=("gdp", "cpi"))
            >>> model.order, model.names, model.trend
            (2, ('gdp', 'cpi'), 'c')
            >>> tight = StudentBVAR(y, order=1, prior=NormalInverseWishartPrior(tightness=0.05))
            >>> tight.prior._label()
            'niw(l1=0.05, l3=1, l4=100)'
        """
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

        Each iteration draws ``(B, Sigma)`` from the weighted conjugate
        posterior, the weights from their Gamma conditionals given the
        residuals, and -- when ``df`` is ``None`` -- the degrees of freedom
        from their exact conditional on the grid. The retained draws
        number ``ceil((n_draws - n_burn) / thin)``.

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
            SpecificationError: If the draw bookkeeping is inconsistent,
                a stated ``df`` does not exceed two, or the prior is
                unusable.
            NumericalError: If a conditional draw collapses.

        Example:
            The tail index stated, then estimated; and the two refusals:

            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((120, 2))
            >>> model = StudentBVAR(y, order=1)
            >>> fixed = model.fit(df=5.0, n_draws=30, n_burn=10, thin=4, seed=0)
            >>> fixed.df, fixed.df_estimated, fixed.n_kept
            (5.0, False, 5)
            >>> free = model.fit(n_draws=30, n_burn=10, seed=0)
            >>> free.df_estimated, free.df_draws.shape
            (True, (10,))
            >>> model.fit(df=2.0, n_draws=30, n_burn=10)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: df must exceed 2 for the innovations to ...
            >>> model.fit(n_draws=10, n_burn=10)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: n_draws (10) must exceed n_burn (10).
        """
        return StudentBVARResult._from_fit(
            self._fit_student(df=df, n_draws=n_draws, n_burn=n_burn, thin=thin, seed=seed),
            self,
        )
