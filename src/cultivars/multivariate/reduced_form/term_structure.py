# filepath: /src/cultivars/multivariate/reduced_form/term_structure.py
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
r"""The dynamic Nelson-Siegel model: the yield curve as three moving factors.

Nelson and Siegel (1987) fit one date's curve with three loadings in a single
decay parameter,

.. math::

   y(\tau) = L + S\,\frac{1 - e^{-\lambda\tau}}{\lambda\tau}
   + C\left(\frac{1 - e^{-\lambda\tau}}{\lambda\tau} - e^{-\lambda\tau}\right),

whose coefficients read as level, slope and curvature. Diebold and Li (2006)
turned the per-date fit into a time-series model by letting the three
coefficients evolve, and Diebold, Rudebusch and Aruoba (2006) wrote that as
one state space: diagonal autoregressions for the factors, curves observing
them through the loadings with per-maturity measurement noise. That is the
form estimated here, by exact maximum likelihood through the Kalman filter
in a single step, with the Diebold-Li two-step regression surviving as the
optimizer's warm start. Koopman, Mallee and van der Wel (2010) then let the
decay move as well, :math:`\log\lambda_t` a fourth autoregressive state, so
the loadings themselves become time-varying and the measurement nonlinear;
that model is estimated through the extended or unscented filter on the
nonlinear substrate.

Two commitments shape the surface. First, exactness is tracked, not assumed.
The constant-decay likelihood is exact and one number, so nested
specifications -- decay estimated against decay fixed at a published value
-- compare by the usual criteria; the time-varying model's likelihood is
whichever filter's Gaussian approximation was maximized, and its result
says so in its summary, in its comparison label and in the note that
travels with the criteria, because a reader comparing the two by AIC is
comparing an exact number with an approximate one. Second, the measurement
is per-maturity and tolerant of gaps. Each point on the curve carries its
own fitted noise variance, so an illiquid tenor is down-weighted by its own
scatter instead of contaminating the factors, and the linear model's filter
handles element-wise missingness, so ragged panels -- maturities that enter
and leave the sample -- estimate without imputation. The nonlinear model
inherits only whole-row missingness from its substrate and refuses partial
rows rather than imputing them.

Layout. :class:`DynamicNelsonSiegel` validates the panel and maturities
inline, builds ``_NelsonSiegelObjective`` from ``_internals`` -- whose
``starts()`` is the Diebold-Li two-step warm start -- maximizes it with
``_maximize_likelihood``, and reads the smoother through
:class:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM`.
:class:`TimeVaryingNelsonSiegel` is a thin ``fit()`` over
``_DecayNelsonSiegelModel`` in ``_internals``, which runs the constant-decay
fit first, appends the three decay parameters, maximizes
``_DecayNelsonSiegelObjective`` through the chosen filter and smooths with
the matching smoother; ``_decay_nelson_siegel_state_space`` emits the fitted
:class:`~cultivars.state_space.nonlinear.NonlinearSSM`. Both results carry
the summary and comparison mixins; the loadings are
``_nelson_siegel_loadings`` in ``_core``, and the hump maturity
:math:`1.79 / \lambda` is where the curvature loading peaks. Each result
rebuilds its parameter record for the emitter from the public fields, so a
result constructed by hand is as usable as a fitted one.

References:
    Nelson, C. R., & Siegel, A. F. (1987). Parsimonious modeling of yield
    curves. *Journal of Business*, 60(4), 473-489.

    Diebold, F. X., & Li, C. (2006). Forecasting the term structure of
    government bond yields. *Journal of Econometrics*, 130(2), 337-364.

    Diebold, F. X., Rudebusch, G. D., & Aruoba, S. B. (2006). The
    macroeconomy and the yield curve: A dynamic latent factor approach.
    *Journal of Econometrics*, 131(1-2), 309-338.

    Koopman, S. J., Mallee, M. I. P., & van der Wel, M. (2010). Analyzing
    the term structure of interest rates using the dynamic Nelson-Siegel
    model with time-varying parameters. *Journal of Business & Economic
    Statistics*, 28(3), 329-343.

Example:
    Sixty curves on eight maturities from a constant-decay model. The
    one-step fit recovers the decay, and the time-varying model, asked
    whether the decay moves, reports a smoothed path that barely does and
    loses the comparison on its two extra parameters -- the honest answer
    on data where the decay is constant:

    >>> import numpy as np
    >>> from cultivars._core import _nelson_siegel_loadings
    >>> rng = np.random.default_rng(0)
    >>> taus = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    >>> mu = np.array([5.0, -1.5, 1.0])
    >>> f = np.tile(mu, (60, 1))
    >>> for t in range(1, 60):
    ...     f[t] = mu + 0.9 * (f[t - 1] - mu) + 0.2 * rng.standard_normal(3)
    >>> curves = f @ _nelson_siegel_loadings(taus, 0.6).T
    >>> curves = curves + 0.02 * rng.standard_normal((60, 8))
    >>> dns = DynamicNelsonSiegel(curves, taus).fit()
    >>> round(dns.decay, 2), dns.n_params
    (0.62, 21.0)
    >>> tv = TimeVaryingNelsonSiegel(curves, taus).fit(filter="extended")
    >>> bool(tv.decay.max() - tv.decay.min() < 0.1), tv.n_params
    (True, 23.0)
    >>> table = dns.compare(tv)
    >>> table.rows[0][0], table.rows[1][0]
    ('DNS(lambda estimated)', 'DNS-TV[extended]')
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ...engine._core import SummaryTable, _nelson_siegel_loadings
from ...engine._internals import (
    _ComparisonMixin,
    _decay_nelson_siegel_state_space,
    _DecayNelsonSiegelFit,
    _DecayNelsonSiegelModel,
    _DecayNelsonSiegelParameters,
    _maximize_likelihood,
    _NelsonSiegelFit,
    _NelsonSiegelObjective,
    _NelsonSiegelParameters,
    _SummaryMixin,
)
from ...exceptions import DimensionError, NumericalError, SpecificationError
from ...state_space.linear_gaussian import LinearGaussianSSM
from ...state_space.nonlinear import NonlinearSSM

__all__ = [
    "DynamicNelsonSiegel",
    "DynamicNelsonSiegelResult",
    "TimeVaryingNelsonSiegel",
    "TimeVaryingNelsonSiegelResult",
]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class DynamicNelsonSiegelResult(_SummaryMixin, _ComparisonMixin):
    r"""A fitted dynamic Nelson-Siegel model: factors, decay, and noise.

    The yield at maturity :math:`\tau` on date :math:`t` is three factors
    through fixed exponential loadings,

    .. math::

       y_t(\tau) = L_t + S_t\,\frac{1 - e^{-\lambda\tau}}{\lambda\tau}
       + C_t\Bigl(\frac{1 - e^{-\lambda\tau}}{\lambda\tau} - e^{-\lambda\tau}\Bigr)
       + \varepsilon_t(\tau),
       \qquad
       f_t - \mu = \Phi\,(f_{t-1} - \mu) + \eta_t,

    with :math:`f_t = (L_t, S_t, C_t)'`, diagonal :math:`\Phi`, full
    innovation covariance :math:`Q` and per-maturity measurement variances
    :math:`\sigma^2_\tau`. Everything here was estimated in one step by
    exact maximum likelihood through the Kalman filter, and the factor paths
    are the smoothed states given the whole panel.

    Note:
        Three things to read correctly. The factors are *smoothed*, so
        :attr:`factors` at date :math:`t` uses curves after :math:`t`;
        :meth:`forecast` re-filters to the one-sided state before
        propagating, which is why its first step does not equal the last
        fitted curve pushed forward. Standard errors are reported for the
        forecast only, from factor uncertainty and measurement noise, never
        for the parameters -- there is no coefficient covariance on this
        record. And the decay :math:`\lambda` sets where the curvature
        loading peaks, at :math:`\tau \approx 1.79 / \lambda`; a fitted decay
        far from the Diebold-Li value for the panel's time unit (0.0609 per
        month, 0.73 per year) usually means the unit, not the curve, is
        unusual.

    Attributes:
        panel: The observed ``(nobs, p)`` yield panel (missing entries
            ``numpy.nan``).
        maturities: The ``(p,)`` maturities.
        decay: The fitted loading decay ``lambda``.
        decay_fixed: Whether the decay was held fixed rather than
            estimated.
        mu: ``(3,)`` factor means (level, slope, curvature).
        ar: ``(3,)`` diagonal factor persistences.
        state_innovation_cov: ``(3, 3)`` factor innovation covariance.
        measurement_var: ``(p,)`` per-maturity measurement variances.
        factors: ``(nobs, 3)`` smoothed factor paths.
        factor_cov: ``(nobs, 3, 3)`` smoothed factor covariances.
        llf: Exact Gaussian log-likelihood.
        nobs: Curve dates.
        n_params: Free parameters the likelihood was maximized over.

    See Also:
        * :class:`DynamicNelsonSiegel` -- the model whose ``fit()`` returns
          this record.
        * :class:`TimeVaryingNelsonSiegelResult` -- the extension in which
          the decay itself follows a random walk.
        * :class:`~cultivars.multivariate.reduced_form.functional.FunctionalVARResult`
          -- the same loadings used as a fixed basis for an unrestricted
          factor VAR, with the decay profiled rather than estimated jointly.
        * :class:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM`
          -- the type of :attr:`state_space`.

    References:
        Nelson, C. R., & Siegel, A. F. (1987). Parsimonious modeling of
        yield curves. *Journal of Business*, 60(4), 473-489.

        Diebold, F. X., & Li, C. (2006). Forecasting the term structure of
        government bond yields. *Journal of Econometrics*, 130(2), 337-364.

        Diebold, F. X., Rudebusch, G. D., & Aruoba, S. B. (2006). The
        macroeconomy and the yield curve: A dynamic latent factor approach.
        *Journal of Econometrics*, 131(1-2), 309-338.

    Example:
        Curves on four maturities generated from persistent level, slope
        and curvature factors at decay 0.6 and read with 5 basis points of
        noise. The one-step fit tracks each factor, reproduces the panel to
        within the noise, prices a maturity it never saw, and forecasts with
        standard errors that grow with the horizon:

        >>> import numpy as np
        >>> from cultivars._core import _nelson_siegel_loadings
        >>> rng = np.random.default_rng(0)
        >>> taus = np.array([0.5, 2.0, 5.0, 10.0])
        >>> basis = _nelson_siegel_loadings(taus, 0.6)
        >>> mu = np.array([5.0, -1.5, 0.5])
        >>> f = np.tile(mu, (40, 1))
        >>> for t in range(1, 40):
        ...     f[t] = mu + 0.9 * (f[t - 1] - mu) + 0.2 * rng.standard_normal(3)
        >>> curves = f @ basis.T + 0.05 * rng.standard_normal((40, 4))
        >>> res = DynamicNelsonSiegel(curves, taus, decay=0.6).fit()
        >>> res.nobs, res.n_maturities, res.n_params, res.decay_fixed
        (40, 4, 16.0, True)
        >>> bool(np.corrcoef(res.level, f[:, 0])[0, 1] > 0.95)
        True
        >>> bool(np.sqrt(np.mean((res.fitted_curves() - curves) ** 2)) < 0.1)
        True
        >>> res.curve(-1, maturities=[7.0]).shape
        (1,)
        >>> mean, std = res.forecast(6)
        >>> mean.shape, bool(np.all(std[-1] > std[0]))
        ((6, 4), True)
    """

    panel: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs, p)`` yield panel as supplied, ``nan`` where missing. Kept out of the repr."""

    maturities: npt.NDArray[np.float64]
    """The ``(p,)`` strictly positive maturities, in the panel's time unit."""

    decay: float
    r"""The loading decay :math:`\lambda`, fitted or held fixed."""

    decay_fixed: bool
    """Whether :attr:`decay` was held at a supplied value rather than estimated."""

    mu: npt.NDArray[np.float64]
    """The ``(3,)`` unconditional factor means, in the order level, slope, curvature."""

    ar: npt.NDArray[np.float64]
    r"""The ``(3,)`` diagonal of :math:`\Phi`, one persistence per factor."""

    state_innovation_cov: npt.NDArray[np.float64] = field(repr=False)
    r"""The ``(3, 3)`` factor innovation covariance :math:`Q`. Kept out of the repr."""

    measurement_var: npt.NDArray[np.float64] = field(repr=False)
    """The ``(p,)`` per-maturity measurement variances. Kept out of the repr."""

    factors: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs, 3)`` smoothed factor paths. Kept out of the repr."""

    factor_cov: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs, 3, 3)`` smoothed factor covariances. Kept out of the repr."""

    llf: float
    """The exact Gaussian log-likelihood at the estimate."""

    nobs: int
    """Curve dates in the panel."""

    n_params: float
    """Free parameters: ``3 + 3 + 6 + p``, plus one when the decay was estimated."""

    @property
    def n_maturities(self) -> int:
        """Points on the curve.

        Example:
            >>> import numpy as np
            >>> res = DynamicNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     decay=0.6, decay_fixed=True, mu=np.zeros(3), ar=np.full(3, 0.9),
            ...     state_innovation_cov=np.eye(3), measurement_var=np.full(4, 0.01),
            ...     factors=np.zeros((30, 3)), factor_cov=np.tile(np.eye(3), (30, 1, 1)),
            ...     llf=0.0, nobs=30, n_params=16.0,
            ... )
            >>> res.n_maturities
            4
        """
        return int(self.maturities.shape[0])

    @property
    def level(self) -> npt.NDArray[np.float64]:
        """The smoothed level factor (the long end), ``(nobs,)``.

        Example:
            >>> import numpy as np
            >>> paths = np.column_stack([np.linspace(4.0, 5.0, 30), np.zeros(30), np.zeros(30)])
            >>> res = DynamicNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     decay=0.6, decay_fixed=True, mu=np.zeros(3), ar=np.full(3, 0.9),
            ...     state_innovation_cov=np.eye(3), measurement_var=np.full(4, 0.01),
            ...     factors=paths, factor_cov=np.tile(np.eye(3), (30, 1, 1)),
            ...     llf=0.0, nobs=30, n_params=16.0,
            ... )
            >>> res.level.shape, float(res.level[0]), float(res.level[-1])
            ((30,), 4.0, 5.0)
        """
        return np.asarray(self.factors[:, 0], dtype=np.float64)

    @property
    def slope(self) -> npt.NDArray[np.float64]:
        """The smoothed slope factor, ``(nobs,)``.

        In this parameterization the slope loading is one at the short end
        and zero at the long end, so a negative slope is an upward-sloping
        curve.

        Example:
            >>> import numpy as np
            >>> paths = np.column_stack([np.zeros(30), np.full(30, -1.5), np.zeros(30)])
            >>> res = DynamicNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     decay=0.6, decay_fixed=True, mu=np.zeros(3), ar=np.full(3, 0.9),
            ...     state_innovation_cov=np.eye(3), measurement_var=np.full(4, 0.01),
            ...     factors=paths, factor_cov=np.tile(np.eye(3), (30, 1, 1)),
            ...     llf=0.0, nobs=30, n_params=16.0,
            ... )
            >>> res.slope.shape, float(res.slope[0])
            ((30,), -1.5)
        """
        return np.asarray(self.factors[:, 1], dtype=np.float64)

    @property
    def curvature(self) -> npt.NDArray[np.float64]:
        """The smoothed curvature factor, ``(nobs,)``.

        Example:
            >>> import numpy as np
            >>> paths = np.column_stack([np.zeros(30), np.zeros(30), np.full(30, 0.5)])
            >>> res = DynamicNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     decay=0.6, decay_fixed=True, mu=np.zeros(3), ar=np.full(3, 0.9),
            ...     state_innovation_cov=np.eye(3), measurement_var=np.full(4, 0.01),
            ...     factors=paths, factor_cov=np.tile(np.eye(3), (30, 1, 1)),
            ...     llf=0.0, nobs=30, n_params=16.0,
            ... )
            >>> res.curvature.shape, float(res.curvature[0])
            ((30,), 0.5)
        """
        return np.asarray(self.factors[:, 2], dtype=np.float64)

    @property
    def loadings(self) -> npt.NDArray[np.float64]:
        """The ``(p, 3)`` Nelson-Siegel loadings at the fitted decay.

        Example:
            >>> import numpy as np
            >>> res = DynamicNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     decay=0.6, decay_fixed=True, mu=np.zeros(3), ar=np.full(3, 0.9),
            ...     state_innovation_cov=np.eye(3), measurement_var=np.full(4, 0.01),
            ...     factors=np.zeros((30, 3)), factor_cov=np.tile(np.eye(3), (30, 1, 1)),
            ...     llf=0.0, nobs=30, n_params=16.0,
            ... )
            >>> res.loadings.shape, bool(np.allclose(res.loadings[:, 0], 1.0))
            ((4, 3), True)
            >>> bool(np.all(np.diff(res.loadings[:, 1]) < 0))
            True
        """
        return _nelson_siegel_loadings(self.maturities, self.decay)

    @property
    def _params(self) -> _NelsonSiegelParameters:
        """The parameter record, rebuilt for the system builder.

        Returns:
            A ``_NelsonSiegelParameters`` with the Cholesky factor of
            :attr:`state_innovation_cov` in place of the covariance, which is
            the form the state-space builder and the objective share.

        Example:
            >>> import numpy as np
            >>> res = DynamicNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     decay=0.6, decay_fixed=True, mu=np.zeros(3), ar=np.full(3, 0.9),
            ...     state_innovation_cov=np.diag([0.04, 0.09, 0.16]),
            ...     measurement_var=np.full(4, 0.01), factors=np.zeros((30, 3)),
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)), llf=0.0, nobs=30, n_params=16.0,
            ... )
            >>> params = res._params
            >>> params.decay, np.diag(params.state_chol).round(2)
            (0.6, array([0.2, 0.3, 0.4]))
        """
        return _NelsonSiegelParameters(
            decay=self.decay,
            mu=self.mu,
            ar=self.ar,
            state_chol=np.linalg.cholesky(self.state_innovation_cov),
            obs_var=self.measurement_var,
        )

    @property
    def state_space(self) -> LinearGaussianSSM:
        """The fitted system, re-applicable to data it was not estimated on.

        The exact linear-Gaussian emitter: its ``loglikelihood`` on the
        estimation panel reproduces ``llf``, and filtering a different
        panel (or the same maturities over new dates) reads it with this
        fit's factor dynamics and noise.

        Example:
            >>> import numpy as np
            >>> from cultivars._core import _nelson_siegel_loadings
            >>> rng = np.random.default_rng(0)
            >>> taus = np.array([0.5, 2.0, 5.0, 10.0])
            >>> basis = _nelson_siegel_loadings(taus, 0.6)
            >>> mu = np.array([5.0, -1.5, 0.5])
            >>> f = np.tile(mu, (40, 1))
            >>> for t in range(1, 40):
            ...     f[t] = mu + 0.9 * (f[t - 1] - mu) + 0.2 * rng.standard_normal(3)
            >>> curves = f @ basis.T + 0.05 * rng.standard_normal((40, 4))
            >>> res = DynamicNelsonSiegel(curves, taus, decay=0.6).fit()
            >>> system = res.state_space
            >>> bool(abs(system.loglikelihood(curves) - res.llf) < 1e-6)
            True
            >>> system.filter(curves[:10]).filtered_state.shape
            (10, 3)
        """
        return LinearGaussianSSM._from_nelson_siegel_system(self._params, self.maturities)

    def fitted_curves(self) -> npt.NDArray[np.float64]:
        """Smoothed fitted yields, ``(nobs, p)``.

        Returns:
            :attr:`factors` pushed through :attr:`loadings`; the gap to
            :attr:`panel` is the measurement error the fit attributes to
            each maturity.

        Example:
            >>> import numpy as np
            >>> res = DynamicNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     decay=0.6, decay_fixed=True, mu=np.zeros(3), ar=np.full(3, 0.9),
            ...     state_innovation_cov=np.eye(3), measurement_var=np.full(4, 0.01),
            ...     factors=np.tile([5.0, 0.0, 0.0], (30, 1)),
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)), llf=0.0, nobs=30, n_params=16.0,
            ... )
            >>> fitted = res.fitted_curves()
            >>> fitted.shape, bool(np.allclose(fitted, 5.0))
            ((30, 4), True)
        """
        return np.asarray(self.factors @ self.loadings.T, dtype=np.float64)

    def curve(
        self, date_index: int, maturities: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """The fitted curve at one date, on any maturity grid.

        Args:
            date_index: Row of the panel (negative indexing allowed).
            maturities: Strictly positive maturities to evaluate on;
                defaults to the estimation grid.

        Returns:
            Fitted yields at those maturities.

        Raises:
            SpecificationError: If a requested maturity is not strictly
                positive.

        Example:
            A pure-level date prices every maturity at the level; a
            negative slope makes the curve rise with maturity:

            >>> import numpy as np
            >>> paths = np.tile([5.0, -1.5, 0.0], (30, 1))
            >>> res = DynamicNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     decay=0.6, decay_fixed=True, mu=np.zeros(3), ar=np.full(3, 0.9),
            ...     state_innovation_cov=np.eye(3), measurement_var=np.full(4, 0.01),
            ...     factors=paths, factor_cov=np.tile(np.eye(3), (30, 1, 1)),
            ...     llf=0.0, nobs=30, n_params=16.0,
            ... )
            >>> fine = res.curve(-1, maturities=np.linspace(0.25, 30.0, 50))
            >>> fine.shape, bool(np.all(np.diff(fine) > 0)), bool(fine[-1] < 5.0)
            ((50,), True, True)
            >>> res.curve(0, maturities=[0.0, 1.0])
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: maturities must be strictly positive.
        """
        grid = (
            self.maturities
            if maturities is None
            else np.asarray(maturities, dtype=np.float64).ravel()
        )
        if np.any(grid <= 0.0):
            raise SpecificationError("maturities must be strictly positive.")
        basis = _nelson_siegel_loadings(grid, self.decay)
        return np.asarray(basis @ self.factors[date_index], dtype=np.float64)

    def forecast(self, steps: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Curve forecasts with honest standard errors.

        Filters the estimation panel to the last factor state, then
        propagates mean and covariance through the fitted dynamics,

        .. math::

           \hat f_{T+h} = \mu + \Phi\,(\hat f_{T+h-1} - \mu), \qquad
           P_{T+h} = \Phi P_{T+h-1} \Phi' + Q,

        and maps both through the loadings, adding the measurement variance
        to the yield-level spread.

        Args:
            steps: Horizons ahead, at least 1.

        Returns:
            ``(mean, std)`` arrays of shape ``(steps, p)`` on the
            estimation maturities; the standard errors include factor
            uncertainty and measurement noise, but not parameter
            uncertainty.

        Raises:
            SpecificationError: If ``steps`` is not positive.

        Example:
            >>> import numpy as np
            >>> from cultivars._core import _nelson_siegel_loadings
            >>> rng = np.random.default_rng(0)
            >>> taus = np.array([0.5, 2.0, 5.0, 10.0])
            >>> basis = _nelson_siegel_loadings(taus, 0.6)
            >>> mu = np.array([5.0, -1.5, 0.5])
            >>> f = np.tile(mu, (40, 1))
            >>> for t in range(1, 40):
            ...     f[t] = mu + 0.9 * (f[t - 1] - mu) + 0.2 * rng.standard_normal(3)
            >>> curves = f @ basis.T + 0.05 * rng.standard_normal((40, 4))
            >>> res = DynamicNelsonSiegel(curves, taus, decay=0.6).fit()
            >>> mean, std = res.forecast(12)
            >>> mean.shape, std.shape, bool(np.all(std > 0.0))
            ((12, 4), (12, 4), True)
            >>> bool(np.all(np.diff(std, axis=0) >= -1e-12))
            True
            >>> bool(np.abs(mean[-1] - res.loadings @ res.mu).max() < 0.5)
            True
            >>> res.forecast(0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: steps must be at least 1; got 0.
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        forward = self.state_space.filter(self.panel)
        mean = forward.filtered_state[-1].copy()
        cov = forward.filtered_state_cov[-1].copy()
        transition = np.diag(self.ar)
        basis = self.loadings
        out_mean = np.empty((steps, self.n_maturities))
        out_std = np.empty((steps, self.n_maturities))
        for step in range(steps):
            mean = self.mu + transition @ (mean - self.mu)
            cov = transition @ cov @ transition.T + self.state_innovation_cov
            out_mean[step] = basis @ mean
            spread = np.einsum("pi,ij,pj->p", basis, cov, basis)
            out_std[step] = np.sqrt(np.maximum(spread + self.measurement_var, 0.0))
        return out_mean, out_std

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Example:
            >>> import numpy as np
            >>> res = DynamicNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     decay=0.6, decay_fixed=False, mu=np.zeros(3), ar=np.full(3, 0.9),
            ...     state_innovation_cov=np.eye(3), measurement_var=np.full(4, 0.01),
            ...     factors=np.zeros((30, 3)), factor_cov=np.tile(np.eye(3), (30, 1, 1)),
            ...     llf=0.0, nobs=30, n_params=17.0,
            ... )
            >>> res._comparison_label()
            'DNS(lambda estimated)'
        """
        tag = "fixed" if self.decay_fixed else "estimated"
        return f"DNS(lambda {tag})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        One row per factor with its mean, persistence and innovation
        standard deviation; metadata pairing the sample dimensions with the
        likelihood, the parameter count and the criteria; notes on the decay
        and where the curvature loading peaks, the one-step estimator, and
        the per-maturity measurement noise.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> res = DynamicNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     decay=0.6, decay_fixed=True, mu=np.array([5.0, -1.5, 0.5]),
            ...     ar=np.array([0.95, 0.9, 0.8]), state_innovation_cov=np.diag([0.04, 0.09, 0.16]),
            ...     measurement_var=np.full(4, 0.01), factors=np.zeros((30, 3)),
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)), llf=100.0, nobs=30, n_params=16.0,
            ... )
            >>> table = res._summary_table()
            >>> table.title, table.columns
            ('Dynamic Nelson-Siegel Results', ('factor', 'mean', 'persistence', 'innovation sd'))
            >>> table.rows[1]
            ('slope', '-1.5000', '0.9000', '0.3000')
            >>> table.notes[0][:45]
            'Loading decay lambda = 0.6000 (held fixed); t'
        """
        ic = self.information_criteria
        rows = (
            (
                "level",
                f"{self.mu[0]:.4f}",
                f"{self.ar[0]:.4f}",
                f"{np.sqrt(self.state_innovation_cov[0, 0]):.4f}",
            ),
            (
                "slope",
                f"{self.mu[1]:.4f}",
                f"{self.ar[1]:.4f}",
                f"{np.sqrt(self.state_innovation_cov[1, 1]):.4f}",
            ),
            (
                "curvature",
                f"{self.mu[2]:.4f}",
                f"{self.ar[2]:.4f}",
                f"{np.sqrt(self.state_innovation_cov[2, 2]):.4f}",
            ),
        )
        notes = [
            f"Loading decay lambda = {self.decay:.4f} "
            f"({'held fixed' if self.decay_fixed else 'estimated'}); the "
            "curvature loading peaks near maturity "
            f"{1.79 / self.decay:.2f}.",
            "Estimated in one step by exact maximum likelihood through the "
            "Kalman filter; the Diebold-Li two-step estimator is the warm "
            "start, not the answer.",
            "Measurement noise is per-maturity, so noisy points on the "
            "curve are down-weighted rather than contaminating the "
            "factors; missing entries are handled element-wise.",
        ]
        return SummaryTable(
            title="Dynamic Nelson-Siegel Results",
            metadata=(
                ("Dates", f"{self.nobs}"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Maturities", f"{self.n_maturities}"),
                ("AIC", f"{ic.aic:.3f}"),
                ("Parameters", f"{self.n_params:.0f}"),
                ("BIC", f"{ic.bic:.3f}"),
            ),
            columns=("factor", "mean", "persistence", "innovation sd"),
            rows=rows,
            notes=tuple(notes),
        )


class DynamicNelsonSiegel:
    r"""One-step maximum-likelihood dynamic Nelson-Siegel estimation.

    The Diebold-Rudebusch-Aruoba state space: level, slope and curvature
    follow diagonal autoregressions around their means with a full innovation
    covariance, curves observe them through the Nelson-Siegel loadings at the
    panel's maturities with per-maturity measurement noise, and the decay
    :math:`\lambda` is either estimated with everything else or held at a
    supplied value. The parameter vector -- :math:`\log\lambda` when free,
    the three means, the three tanh-bounded persistences, the six entries of
    a log-diagonal Cholesky factor and the :math:`p` log measurement
    variances -- is maximized from the Diebold-Li two-step estimate as the
    warm start, with the decay seeded where the curvature loading peaks at
    the median maturity.

    Attributes:
        _panel: The validated ``(nobs, p)`` yield panel, ``nan`` where
            missing.
        _maturities: The validated ``(p,)`` maturities.
        _fixed_decay: The decay to hold fixed, or ``None`` to estimate it.

    See Also:
        * :class:`DynamicNelsonSiegelResult` -- the record ``fit()`` returns.
        * :class:`TimeVaryingNelsonSiegel` -- the extension with a
          random-walk decay.
        * :class:`~cultivars.multivariate.reduced_form.functional.FunctionalVAR`
          -- the two-step route, with ``basis="nelson-siegel"``, when the
          factor VAR should be unrestricted rather than diagonal.

    References:
        Diebold, F. X., & Li, C. (2006). Forecasting the term structure of
        government bond yields. *Journal of Econometrics*, 130(2), 337-364.

        Diebold, F. X., Rudebusch, G. D., & Aruoba, S. B. (2006). The
        macroeconomy and the yield curve: A dynamic latent factor approach.
        *Journal of Econometrics*, 131(1-2), 309-338.

    Example:
        >>> import numpy as np
        >>> from cultivars._core import _nelson_siegel_loadings
        >>> rng = np.random.default_rng(0)
        >>> taus = np.array([0.25, 1.0, 2.0, 5.0, 10.0])
        >>> basis = _nelson_siegel_loadings(taus, 0.6)
        >>> f = np.zeros((120, 3))
        >>> mu = np.array([5.0, -1.5, 0.5])
        >>> f[0] = mu
        >>> for t in range(1, 120):
        ...     f[t] = mu + 0.9 * (f[t - 1] - mu) + 0.2 * rng.standard_normal(3)
        >>> curves = f @ basis.T + 0.05 * rng.standard_normal((120, 5))
        >>> res = DynamicNelsonSiegel(curves, taus, decay=0.6).fit()
        >>> bool(np.corrcoef(res.level, f[:, 0])[0, 1] > 0.95)
        True
    """

    __slots__ = ("_fixed_decay", "_maturities", "_panel")

    def __init__(
        self,
        panel: npt.ArrayLike,
        maturities: npt.ArrayLike,
        *,
        decay: float | None = None,
    ) -> None:
        """Validate the panel, the maturity grid, and any fixed decay.

        Args:
            panel: The ``(nobs, p)`` yield panel, one row per date and one
                column per maturity; ``numpy.nan`` entries are missing and
                handled element-wise, so ragged panels estimate without
                imputation.
            maturities: The ``(p,)`` strictly positive maturities, in
                whatever time unit the decay should be quoted in.
            decay: A loading decay to hold fixed, or ``None`` (default) to
                estimate it by maximum likelihood.

        Raises:
            DimensionError: If the panel and maturities disagree, fewer than
                four maturities are given, or fewer than thirty dates.
            SpecificationError: If a maturity or a fixed decay is not
                strictly positive.
            NumericalError: If the panel has infinite entries, or a date
                with no finite entry at all.

        Example:
            >>> import numpy as np
            >>> curves = np.random.default_rng(0).standard_normal((40, 4)) + 5.0
            >>> taus = np.array([1.0, 2.0, 5.0, 10.0])
            >>> model = DynamicNelsonSiegel(curves, taus, decay=0.6)
            >>> DynamicNelsonSiegel(curves[:, :3], taus[:3])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.DimensionError: the three factors need at least 4 maturities ...
            >>> DynamicNelsonSiegel(curves[:20], taus)
            Traceback (most recent call last):
            cultivars.exceptions.DimensionError: the factor dynamics need at least 30 dates; got 20.
            >>> DynamicNelsonSiegel(curves, taus, decay=0.0)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: a fixed decay must be strictly positive; ...
            >>> gap = curves.copy()
            >>> gap[3] = np.nan
            >>> DynamicNelsonSiegel(gap, taus)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.NumericalError: every date must observe at least one maturity; ...
        """
        block = np.asarray(panel, dtype=np.float64)
        if block.ndim != 2:
            raise DimensionError(f"panel must be (nobs, p); got shape {block.shape}.")
        grid = np.asarray(maturities, dtype=np.float64).ravel()
        if grid.shape[0] != block.shape[1]:
            raise DimensionError(
                f"maturities must have one entry per panel column "
                f"({block.shape[1]}); got {grid.shape[0]}."
            )
        if grid.shape[0] < 4:
            raise DimensionError(
                "the three factors need at least 4 maturities to be "
                f"identified with measurement noise; got {grid.shape[0]}."
            )
        if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
            raise SpecificationError("maturities must be finite and strictly positive.")
        if block.shape[0] < 30:
            raise DimensionError(
                f"the factor dynamics need at least 30 dates; got {block.shape[0]}."
            )
        finite = np.isfinite(block)
        if np.any(np.isinf(block)):
            raise NumericalError("panel entries must be finite or NaN.")
        if not np.all(finite.any(axis=1)):
            raise NumericalError(
                "every date must observe at least one maturity; drop all-missing rows."
            )
        if decay is not None and not decay > 0.0:
            raise SpecificationError(f"a fixed decay must be strictly positive; got {decay}.")
        self._panel = block
        self._maturities = grid
        self._fixed_decay = None if decay is None else float(decay)

    def fit(self) -> DynamicNelsonSiegelResult:
        """Maximize the exact likelihood from the two-step warm start.

        Builds the ``_NelsonSiegelObjective``, maximizes it with the
        package's likelihood driver, rebuilds the linear-Gaussian system at
        the optimum, smooths the panel for the factor paths, and packs the
        result; ``n_params`` is the length of the flat parameter vector,
        so it counts the decay only when it was free.

        Returns:
            A :class:`DynamicNelsonSiegelResult`.

        Example:
            Held at 0.6 the decay costs no parameter; left free it is
            recovered near the truth and costs one:

            >>> import numpy as np
            >>> from cultivars._core import _nelson_siegel_loadings
            >>> rng = np.random.default_rng(0)
            >>> taus = np.array([0.5, 2.0, 5.0, 10.0])
            >>> basis = _nelson_siegel_loadings(taus, 0.6)
            >>> mu = np.array([5.0, -1.5, 0.5])
            >>> f = np.tile(mu, (40, 1))
            >>> for t in range(1, 40):
            ...     f[t] = mu + 0.9 * (f[t - 1] - mu) + 0.2 * rng.standard_normal(3)
            >>> curves = f @ basis.T + 0.05 * rng.standard_normal((40, 4))
            >>> fixed = DynamicNelsonSiegel(curves, taus, decay=0.6).fit()
            >>> free = DynamicNelsonSiegel(curves, taus).fit()
            >>> fixed.n_params, free.n_params, bool(free.llf >= fixed.llf - 1e-6)
            (16.0, 17.0, True)
            >>> bool(0.3 < free.decay < 1.2), free.decay_fixed
            (True, False)
        """
        objective = _NelsonSiegelObjective(
            panel=self._panel,
            maturities=self._maturities,
            fixed_decay=self._fixed_decay,
        )
        params, llf = _maximize_likelihood(objective)
        model = LinearGaussianSSM._from_nelson_siegel_system(params, self._maturities)
        smoothed = model.smooth(self._panel)
        fit = _NelsonSiegelFit(
            params=params,
            llf=llf,
            n_params=objective.starts()[0].shape[0],
            nobs=int(self._panel.shape[0]),
            factors=smoothed.smoothed_state,
            factor_cov=smoothed.smoothed_state_cov,
        )
        return DynamicNelsonSiegelResult(
            panel=self._panel,
            maturities=self._maturities,
            decay=fit.params.decay,
            decay_fixed=self._fixed_decay is not None,
            mu=fit.params.mu,
            ar=fit.params.ar,
            state_innovation_cov=fit.params.state_chol @ fit.params.state_chol.T,
            measurement_var=fit.params.obs_var,
            factors=fit.factors,
            factor_cov=fit.factor_cov,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=float(fit.n_params),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class TimeVaryingNelsonSiegelResult(_SummaryMixin, _ComparisonMixin):
    r"""A fitted Nelson-Siegel model with a time-varying loading decay.

    The Koopman-Mallee-van der Wel extension: the decay becomes a fourth
    latent factor with autoregressive dynamics in its log,

    .. math::

       y_t(\tau) = L_t + S_t\, s(\tau, \lambda_t) + C_t\, c(\tau, \lambda_t)
       + \varepsilon_t(\tau),
       \qquad
       \log\lambda_t - \bar\ell = \phi_\lambda\,(\log\lambda_{t-1} - \bar\ell)
       + \zeta_t,

    with the three factors keeping their diagonal autoregressions. The
    transition stays linear-Gaussian; the measurement is nonlinear only
    through the loadings' dependence on :math:`\lambda_t`, so the system is
    additive-Gaussian and the extended or unscented filter reads it. The
    factor and decay paths here are the matching smoother's output.

    Note:
        The likelihood is an approximation and the record says so wherever
        it appears. :attr:`llf` is the chosen filter's Gaussian
        approximation to the exact likelihood, so information criteria
        against the constant-decay model -- whose likelihood *is* exact --
        rank by a heuristic, and the two filters' values are not comparable
        with each other either. The decay path is weakly identified by
        construction: it moves the curve only through the shape of the
        slope and curvature loadings, so on a sparse maturity grid or a
        short panel the fit can collapse to a nearly constant decay with the
        curvature factor absorbing the slack, and a tiny :attr:`decay_sd`
        with a negative :attr:`decay_ar` is that collapse, not a finding.
        Many maturities across the hump region are what identify it.

    Attributes:
        panel: The observed ``(nobs, p)`` yield panel.
        maturities: The ``(p,)`` maturities.
        mu: ``(3,)`` factor means (level, slope, curvature).
        ar: ``(3,)`` diagonal factor persistences.
        state_innovation_cov: ``(3, 3)`` factor innovation covariance.
        measurement_var: ``(p,)`` per-maturity measurement variances.
        log_decay_mean: Unconditional mean of the log decay.
        decay_ar: Persistence of the log decay.
        decay_sd: Innovation standard deviation of the log decay.
        factors: ``(nobs, 3)`` smoothed factor paths.
        factor_cov: ``(nobs, 3, 3)`` smoothed factor covariances.
        log_decay: ``(nobs,)`` smoothed log-decay path.
        log_decay_std: ``(nobs,)`` smoothed log-decay standard deviations.
        llf: The *approximate* Gaussian log-likelihood of the chosen
            filter. Not the exact likelihood: information criteria are
            comparable against the constant-decay model only as a
            heuristic, and the summary says so.
        filter: ``"extended"`` or ``"unscented"``.
        nobs: Curve dates.
        n_params: Free parameters the likelihood was maximized over.

    See Also:
        * :class:`TimeVaryingNelsonSiegel` -- the model whose ``fit()``
          returns this record.
        * :class:`DynamicNelsonSiegelResult` -- the constant-decay model with
          the exact likelihood, and this fit's warm start.
        * :class:`~cultivars.state_space.nonlinear.NonlinearSSM` -- the type
          of :attr:`state_space`.

    References:
        Koopman, S. J., Mallee, M. I. P., & van der Wel, M. (2010).
        Analyzing the term structure of interest rates using the dynamic
        Nelson-Siegel model with time-varying parameters. *Journal of
        Business & Economic Statistics*, 28(3), 329-343.

        Diebold, F. X., & Li, C. (2006). Forecasting the term structure of
        government bond yields. *Journal of Econometrics*, 130(2), 337-364.

    Example:
        Eight maturities across the hump region, sixty dates, and a decay
        that wanders between 0.44 and 1.5 around 0.6. The unscented fit
        tracks the level almost exactly and the log-decay path well enough
        to move the fitted hump with it:

        >>> import numpy as np
        >>> from cultivars._core import _nelson_siegel_loadings
        >>> rng = np.random.default_rng(0)
        >>> taus = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        >>> mu, bar = np.array([5.0, -1.5, 1.0]), np.log(0.6)
        >>> f, log_lam = np.tile(mu, (60, 1)), np.full(60, bar)
        >>> for t in range(1, 60):
        ...     f[t] = mu + 0.9 * (f[t - 1] - mu) + 0.2 * rng.standard_normal(3)
        ...     log_lam[t] = bar + 0.95 * (log_lam[t - 1] - bar) + 0.15 * rng.standard_normal()
        >>> curves = np.stack(
        ...     [_nelson_siegel_loadings(taus, np.exp(log_lam[t])) @ f[t] for t in range(60)]
        ... )
        >>> curves = curves + 0.02 * rng.standard_normal((60, 8))
        >>> res = TimeVaryingNelsonSiegel(curves, taus).fit()
        >>> res.filter, res.nobs, res.n_maturities, res.n_params
        ('unscented', 60, 8, 23.0)
        >>> bool(np.corrcoef(res.level, f[:, 0])[0, 1] > 0.98)
        True
        >>> bool(np.corrcoef(res.log_decay, log_lam)[0, 1] > 0.7)
        True
        >>> bool(res.hump_maturity.max() - res.hump_maturity.min() > 1.0)
        True
    """

    panel: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs, p)`` yield panel as supplied, whole rows ``nan`` where missing.

    Kept out of the repr.
    """

    maturities: npt.NDArray[np.float64]
    """The ``(p,)`` strictly positive maturities, in the panel's time unit."""

    mu: npt.NDArray[np.float64]
    """The ``(3,)`` unconditional factor means, in the order level, slope, curvature."""

    ar: npt.NDArray[np.float64]
    """The ``(3,)`` diagonal factor persistences."""

    state_innovation_cov: npt.NDArray[np.float64] = field(repr=False)
    """The ``(3, 3)`` factor innovation covariance. Kept out of the repr."""

    measurement_var: npt.NDArray[np.float64] = field(repr=False)
    """The ``(p,)`` per-maturity measurement variances. Kept out of the repr."""

    log_decay_mean: float
    r"""The unconditional mean :math:`\bar\ell` of :math:`\log\lambda_t`."""

    decay_ar: float
    r"""The persistence :math:`\phi_\lambda` of the log decay."""

    decay_sd: float
    r"""The innovation standard deviation of the log decay."""

    factors: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs, 3)`` smoothed factor paths. Kept out of the repr."""

    factor_cov: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs, 3, 3)`` smoothed factor covariances. Kept out of the repr."""

    log_decay: npt.NDArray[np.float64] = field(repr=False)
    r"""The ``(nobs,)`` smoothed path of :math:`\log\lambda_t`. Kept out of the repr."""

    log_decay_std: npt.NDArray[np.float64] = field(repr=False)
    """The ``(nobs,)`` smoothed standard deviations of the log decay. Kept out of the repr."""

    llf: float
    """The chosen filter's Gaussian approximation to the log-likelihood."""

    filter: str
    """``"extended"`` or ``"unscented"``: the filter that produced :attr:`llf` and the paths."""

    nobs: int
    """Curve dates in the panel."""

    n_params: float
    """Free parameters: the constant-decay count plus the decay mean, persistence and sd."""

    @classmethod
    def _from_fit(
        cls, fit: _DecayNelsonSiegelFit, model: TimeVaryingNelsonSiegel
    ) -> TimeVaryingNelsonSiegelResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The packed fit from ``_fit_decay``, with the parameter
                record, the smoothed paths and the filter name.
            model: The model that produced it, read for the panel and the
                maturities.

        Returns:
            A populated result, with the innovation covariance rebuilt from
            the fit's Cholesky factor.
        """
        p = fit.params
        return cls(
            panel=model.panel,
            maturities=model.maturities,
            mu=p.mu,
            ar=p.ar,
            state_innovation_cov=p.state_chol @ p.state_chol.T,
            measurement_var=p.obs_var,
            log_decay_mean=p.log_decay_mean,
            decay_ar=p.decay_ar,
            decay_sd=p.decay_sd,
            factors=fit.factors,
            factor_cov=fit.factor_cov,
            log_decay=fit.log_decay,
            log_decay_std=fit.log_decay_std,
            llf=fit.llf,
            filter=fit.filter,
            nobs=fit.nobs,
            n_params=float(fit.n_params),
        )

    @property
    def n_maturities(self) -> int:
        """Points on the curve.

        Example:
            >>> import numpy as np
            >>> res = TimeVaryingNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     mu=np.zeros(3), ar=np.full(3, 0.9), state_innovation_cov=np.eye(3),
            ...     measurement_var=np.full(4, 0.01), log_decay_mean=np.log(0.6),
            ...     decay_ar=0.9, decay_sd=0.1, factors=np.zeros((30, 3)),
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)), log_decay=np.full(30, np.log(0.6)),
            ...     log_decay_std=np.full(30, 0.1), llf=0.0, filter="unscented", nobs=30,
            ...     n_params=19.0,
            ... )
            >>> res.n_maturities
            4
        """
        return int(self.maturities.shape[0])

    @property
    def decay(self) -> npt.NDArray[np.float64]:
        """The smoothed decay path ``exp(log lambda_t)``, ``(nobs,)``.

        Example:
            >>> import numpy as np
            >>> res = TimeVaryingNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     mu=np.zeros(3), ar=np.full(3, 0.9), state_innovation_cov=np.eye(3),
            ...     measurement_var=np.full(4, 0.01), log_decay_mean=np.log(0.6),
            ...     decay_ar=0.9, decay_sd=0.1, factors=np.zeros((30, 3)),
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)),
            ...     log_decay=np.linspace(np.log(0.4), np.log(1.0), 30),
            ...     log_decay_std=np.full(30, 0.1), llf=0.0, filter="unscented", nobs=30,
            ...     n_params=19.0,
            ... )
            >>> res.decay.shape, res.decay[[0, -1]].round(2)
            ((30,), array([0.4, 1. ]))
        """
        return np.asarray(np.exp(self.log_decay), dtype=np.float64)

    @property
    def decay_mean(self) -> float:
        """The unconditional decay ``exp(log_decay_mean)``.

        Example:
            >>> import numpy as np
            >>> res = TimeVaryingNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     mu=np.zeros(3), ar=np.full(3, 0.9), state_innovation_cov=np.eye(3),
            ...     measurement_var=np.full(4, 0.01), log_decay_mean=np.log(0.6),
            ...     decay_ar=0.9, decay_sd=0.1, factors=np.zeros((30, 3)),
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)), log_decay=np.full(30, np.log(0.6)),
            ...     log_decay_std=np.full(30, 0.1), llf=0.0, filter="unscented", nobs=30,
            ...     n_params=19.0,
            ... )
            >>> round(res.decay_mean, 6)
            0.6
        """
        return float(np.exp(self.log_decay_mean))

    @property
    def hump_maturity(self) -> npt.NDArray[np.float64]:
        """Where the curvature loading peaks each date, ``1.79 / lambda_t``.

        Example:
            >>> import numpy as np
            >>> res = TimeVaryingNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     mu=np.zeros(3), ar=np.full(3, 0.9), state_innovation_cov=np.eye(3),
            ...     measurement_var=np.full(4, 0.01), log_decay_mean=np.log(0.6),
            ...     decay_ar=0.9, decay_sd=0.1, factors=np.zeros((30, 3)),
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)), log_decay=np.full(30, np.log(0.6)),
            ...     log_decay_std=np.full(30, 0.1), llf=0.0, filter="unscented", nobs=30,
            ...     n_params=19.0,
            ... )
            >>> res.hump_maturity.shape, round(float(res.hump_maturity[0]), 2)
            ((30,), 2.99)
        """
        return np.asarray(1.7916 / self.decay, dtype=np.float64)

    @property
    def level(self) -> npt.NDArray[np.float64]:
        """The smoothed level factor, ``(nobs,)``.

        Example:
            >>> import numpy as np
            >>> paths = np.column_stack([np.linspace(4.0, 5.0, 30), np.zeros(30), np.zeros(30)])
            >>> res = TimeVaryingNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     mu=np.zeros(3), ar=np.full(3, 0.9), state_innovation_cov=np.eye(3),
            ...     measurement_var=np.full(4, 0.01), log_decay_mean=np.log(0.6),
            ...     decay_ar=0.9, decay_sd=0.1, factors=paths,
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)), log_decay=np.full(30, np.log(0.6)),
            ...     log_decay_std=np.full(30, 0.1), llf=0.0, filter="unscented", nobs=30,
            ...     n_params=19.0,
            ... )
            >>> res.level.shape, float(res.level[-1])
            ((30,), 5.0)
        """
        return np.asarray(self.factors[:, 0], dtype=np.float64)

    @property
    def slope(self) -> npt.NDArray[np.float64]:
        """The smoothed slope factor, ``(nobs,)``.

        Example:
            >>> import numpy as np
            >>> paths = np.column_stack([np.zeros(30), np.full(30, -1.5), np.zeros(30)])
            >>> res = TimeVaryingNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     mu=np.zeros(3), ar=np.full(3, 0.9), state_innovation_cov=np.eye(3),
            ...     measurement_var=np.full(4, 0.01), log_decay_mean=np.log(0.6),
            ...     decay_ar=0.9, decay_sd=0.1, factors=paths,
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)), log_decay=np.full(30, np.log(0.6)),
            ...     log_decay_std=np.full(30, 0.1), llf=0.0, filter="unscented", nobs=30,
            ...     n_params=19.0,
            ... )
            >>> res.slope.shape, float(res.slope[0])
            ((30,), -1.5)
        """
        return np.asarray(self.factors[:, 1], dtype=np.float64)

    @property
    def curvature(self) -> npt.NDArray[np.float64]:
        """The smoothed curvature factor, ``(nobs,)``.

        Example:
            >>> import numpy as np
            >>> paths = np.column_stack([np.zeros(30), np.zeros(30), np.full(30, 0.5)])
            >>> res = TimeVaryingNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     mu=np.zeros(3), ar=np.full(3, 0.9), state_innovation_cov=np.eye(3),
            ...     measurement_var=np.full(4, 0.01), log_decay_mean=np.log(0.6),
            ...     decay_ar=0.9, decay_sd=0.1, factors=paths,
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)), log_decay=np.full(30, np.log(0.6)),
            ...     log_decay_std=np.full(30, 0.1), llf=0.0, filter="unscented", nobs=30,
            ...     n_params=19.0,
            ... )
            >>> res.curvature.shape, float(res.curvature[0])
            ((30,), 0.5)
        """
        return np.asarray(self.factors[:, 2], dtype=np.float64)

    def loadings(self, t: int) -> npt.NDArray[np.float64]:
        """The ``(p, 3)`` Nelson-Siegel loadings at date ``t``'s smoothed decay.

        Args:
            t: Row of the panel (negative indexing allowed).

        Returns:
            The loading matrix at ``decay[t]``.

        Example:
            A faster decay at the last date pulls the slope loading down
            sooner across the same maturities:

            >>> import numpy as np
            >>> res = TimeVaryingNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     mu=np.zeros(3), ar=np.full(3, 0.9), state_innovation_cov=np.eye(3),
            ...     measurement_var=np.full(4, 0.01), log_decay_mean=np.log(0.6),
            ...     decay_ar=0.9, decay_sd=0.1, factors=np.zeros((30, 3)),
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)),
            ...     log_decay=np.linspace(np.log(0.4), np.log(1.0), 30),
            ...     log_decay_std=np.full(30, 0.1), llf=0.0, filter="unscented", nobs=30,
            ...     n_params=19.0,
            ... )
            >>> res.loadings(0).shape, bool(np.all(res.loadings(-1)[:, 1] < res.loadings(0)[:, 1]))
            ((4, 3), True)
        """
        return _nelson_siegel_loadings(self.maturities, float(self.decay[t]))

    @property
    def fitted(self) -> npt.NDArray[np.float64]:
        """Smoothed fitted curves, ``(nobs, p)``.

        Each date's factors through that date's own loadings.

        Example:
            >>> import numpy as np
            >>> res = TimeVaryingNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     mu=np.zeros(3), ar=np.full(3, 0.9), state_innovation_cov=np.eye(3),
            ...     measurement_var=np.full(4, 0.01), log_decay_mean=np.log(0.6),
            ...     decay_ar=0.9, decay_sd=0.1, factors=np.tile([5.0, 0.0, 0.0], (30, 1)),
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)), log_decay=np.full(30, np.log(0.6)),
            ...     log_decay_std=np.full(30, 0.1), llf=0.0, filter="unscented", nobs=30,
            ...     n_params=19.0,
            ... )
            >>> res.fitted.shape, bool(np.allclose(res.fitted, 5.0))
            ((30, 4), True)
        """
        return np.asarray(
            np.stack([self.loadings(t) @ self.factors[t] for t in range(self.nobs)]),
            dtype=np.float64,
        )

    @property
    def _params(self) -> _DecayNelsonSiegelParameters:
        """The parameter record, rebuilt for the emitter.

        Returns:
            A ``_DecayNelsonSiegelParameters`` carrying the Cholesky factor
            of :attr:`state_innovation_cov` and the three decay parameters.

        Example:
            >>> import numpy as np
            >>> res = TimeVaryingNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     mu=np.zeros(3), ar=np.full(3, 0.9),
            ...     state_innovation_cov=np.diag([0.04, 0.09, 0.16]),
            ...     measurement_var=np.full(4, 0.01), log_decay_mean=np.log(0.6),
            ...     decay_ar=0.9, decay_sd=0.1, factors=np.zeros((30, 3)),
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)), log_decay=np.full(30, np.log(0.6)),
            ...     log_decay_std=np.full(30, 0.1), llf=0.0, filter="unscented", nobs=30,
            ...     n_params=19.0,
            ... )
            >>> params = res._params
            >>> np.diag(params.state_chol).round(2), params.decay_ar, params.decay_sd
            (array([0.2, 0.3, 0.4]), 0.9, 0.1)
        """
        return _DecayNelsonSiegelParameters(
            mu=self.mu,
            ar=self.ar,
            state_chol=np.linalg.cholesky(self.state_innovation_cov),
            obs_var=self.measurement_var,
            log_decay_mean=self.log_decay_mean,
            decay_ar=self.decay_ar,
            decay_sd=self.decay_sd,
        )

    @property
    def state_space(self) -> NonlinearSSM:
        """The fitted system on the nonlinear substrate.

        Additive-Gaussian, so the extended, unscented, and particle
        filters all read it; the unscented filter on the estimation panel
        reproduces ``llf`` when the fit used it.

        Example:
            >>> import numpy as np
            >>> res = TimeVaryingNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     mu=np.zeros(3), ar=np.full(3, 0.9), state_innovation_cov=np.eye(3),
            ...     measurement_var=np.full(4, 0.01), log_decay_mean=np.log(0.6),
            ...     decay_ar=0.9, decay_sd=0.1, factors=np.zeros((30, 3)),
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)), log_decay=np.full(30, np.log(0.6)),
            ...     log_decay_std=np.full(30, 0.1), llf=0.0, filter="unscented", nobs=30,
            ...     n_params=19.0,
            ... )
            >>> system = res.state_space
            >>> system.k_states, system.k_endog, system.is_additive_gaussian
            (4, 4, True)
        """
        return _decay_nelson_siegel_state_space(self._params, self.maturities)

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Example:
            >>> import numpy as np
            >>> res = TimeVaryingNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     mu=np.zeros(3), ar=np.full(3, 0.9), state_innovation_cov=np.eye(3),
            ...     measurement_var=np.full(4, 0.01), log_decay_mean=np.log(0.6),
            ...     decay_ar=0.9, decay_sd=0.1, factors=np.zeros((30, 3)),
            ...     factor_cov=np.tile(np.eye(3), (30, 1, 1)), log_decay=np.full(30, np.log(0.6)),
            ...     log_decay_std=np.full(30, 0.1), llf=0.0, filter="extended", nobs=30,
            ...     n_params=19.0,
            ... )
            >>> res._comparison_label()
            'DNS-TV[extended]'
        """
        return f"DNS-TV[{self.filter}]"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        Four rows -- the three factors and the log decay -- with mean,
        persistence and innovation standard deviation; metadata naming the
        filter beside the approximate likelihood and criteria; notes stating
        that the likelihood is a filter approximation and where the
        unconditional and smoothed decays put the curvature hump.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> res = TimeVaryingNelsonSiegelResult(
            ...     panel=np.zeros((30, 4)), maturities=np.array([1.0, 2.0, 5.0, 10.0]),
            ...     mu=np.array([5.0, -1.5, 0.5]), ar=np.array([0.95, 0.9, 0.8]),
            ...     state_innovation_cov=np.eye(3), measurement_var=np.full(4, 0.01),
            ...     log_decay_mean=np.log(0.6), decay_ar=0.9, decay_sd=0.1,
            ...     factors=np.zeros((30, 3)), factor_cov=np.tile(np.eye(3), (30, 1, 1)),
            ...     log_decay=np.full(30, np.log(0.6)), log_decay_std=np.full(30, 0.1),
            ...     llf=100.0, filter="unscented", nobs=30, n_params=19.0,
            ... )
            >>> table = res._summary_table()
            >>> table.title
            'Dynamic Nelson-Siegel (time-varying decay) Results'
            >>> table.rows[3]
            ('log decay', '-0.5108', '0.9000', '0.1000')
            >>> dict(table.metadata)["Filter"], table.notes[0][:33]
            ('unscented', 'Likelihood is the unscented filte')
        """
        ic = self.information_criteria
        rows: list[tuple[str, str, str, str]] = []
        for j, name in enumerate(("level", "slope", "curvature")):
            rows.append(
                (
                    name,
                    f"{self.mu[j]:.4f}",
                    f"{self.ar[j]:.4f}",
                    f"{np.sqrt(self.state_innovation_cov[j, j]):.4f}",
                )
            )
        rows.append(
            (
                "log decay",
                f"{self.log_decay_mean:.4f}",
                f"{self.decay_ar:.4f}",
                f"{self.decay_sd:.4f}",
            )
        )
        notes = (
            f"Likelihood is the {self.filter} filter's Gaussian approximation, not "
            "the exact likelihood; treat information criteria against the "
            "constant-decay model as heuristic.",
            f"Unconditional decay {self.decay_mean:.4f} (curvature hump at maturity "
            f"{1.7916 / self.decay_mean:.2f}); smoothed decay ranges "
            f"{self.decay.min():.4f} to {self.decay.max():.4f}.",
        )
        return SummaryTable(
            title="Dynamic Nelson-Siegel (time-varying decay) Results",
            metadata=(
                ("Filter", self.filter),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Dates", f"{self.nobs}"),
                ("AIC", f"{ic.aic:.3f}"),
                ("Maturities", f"{self.n_maturities}"),
                ("BIC", f"{ic.bic:.3f}"),
            ),
            columns=("factor", "mean", "persistence", "innovation sd"),
            rows=tuple(rows),
            notes=notes,
        )


class TimeVaryingNelsonSiegel(_DecayNelsonSiegelModel[TimeVaryingNelsonSiegelResult]):
    r"""The dynamic Nelson-Siegel model with a time-varying loading decay.

    The state is ``(level, slope, curvature, log lambda)``, each a diagonal
    AR(1); the measurement is the Nelson-Siegel curve at the current
    :math:`\lambda_t`. Estimated by maximizing the unscented (default) or
    extended filter's likelihood from the constant-decay exact fit, which
    is run first as the warm start; the three decay parameters are
    appended to that fit's vector and the whole vector is re-optimized
    through the chosen filter. The unscented filter is the default because
    the loadings are smooth and bounded in :math:`\lambda`, so its sigma
    points capture the nonlinearity that a first-order Jacobian flattens;
    it costs two to three times the extended filter's running time.

    Attributes:
        _panel: The validated ``(nobs, p)`` yield panel, whole rows ``nan``
            where missing.
        _maturities: The validated ``(p,)`` maturities.

    See Also:
        * :class:`TimeVaryingNelsonSiegelResult` -- the record ``fit()``
          returns.
        * :class:`DynamicNelsonSiegel` -- the constant-decay model, exact
          likelihood, and the warm start of this one.
        * :mod:`~cultivars.state_space.nonlinear` -- the filters this model
          reads itself through.

    References:
        Koopman, S. J., Mallee, M. I. P., & van der Wel, M. (2010).
        Analyzing the term structure of interest rates using the dynamic
        Nelson-Siegel model with time-varying parameters. *Journal of
        Business & Economic Statistics*, 28(3), 329-343.

    Example:
        >>> import numpy as np
        >>> from cultivars._core import _nelson_siegel_loadings
        >>> rng = np.random.default_rng(0)
        >>> taus = np.array([0.25, 1.0, 2.0, 5.0, 10.0])
        >>> f = np.zeros((120, 3))
        >>> mu = np.array([5.0, -1.5, 0.5])
        >>> f[0] = mu
        >>> for t in range(1, 120):
        ...     f[t] = mu + 0.9 * (f[t - 1] - mu) + 0.2 * rng.standard_normal(3)
        >>> curves = f @ _nelson_siegel_loadings(taus, 0.6).T
        >>> curves = curves + 0.05 * rng.standard_normal((120, 5))
        >>> res = TimeVaryingNelsonSiegel(curves, taus).fit()
        >>> bool(np.corrcoef(res.level, f[:, 0])[0, 1] > 0.95)
        True
    """

    def fit(self, *, filter: str = "unscented") -> TimeVaryingNelsonSiegelResult:
        """Maximize the approximate likelihood.

        Args:
            filter: ``"unscented"`` (default; exact through the linear
                transition, third-order accurate through the loadings) or
                ``"extended"`` (central-difference Jacobians).

        Returns:
            A :class:`TimeVaryingNelsonSiegelResult` whose ``filter`` field
            records the choice.

        Raises:
            SpecificationError: If the filter is unknown.

        Example:
            The extended filter is the cheaper reader of the same system;
            its approximate likelihood is not comparable with the unscented
            one, and the parameter count is the same under both:

            >>> import numpy as np
            >>> from cultivars._core import _nelson_siegel_loadings
            >>> rng = np.random.default_rng(0)
            >>> taus = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
            >>> mu, bar = np.array([5.0, -1.5, 1.0]), np.log(0.6)
            >>> f, log_lam = np.tile(mu, (60, 1)), np.full(60, bar)
            >>> for t in range(1, 60):
            ...     f[t] = mu + 0.9 * (f[t - 1] - mu) + 0.2 * rng.standard_normal(3)
            ...     log_lam[t] = bar + 0.95 * (log_lam[t - 1] - bar) + 0.15 * rng.standard_normal()
            >>> curves = np.stack(
            ...     [_nelson_siegel_loadings(taus, np.exp(log_lam[t])) @ f[t] for t in range(60)]
            ... )
            >>> curves = curves + 0.02 * rng.standard_normal((60, 8))
            >>> res = TimeVaryingNelsonSiegel(curves, taus).fit(filter="extended")
            >>> res.filter, res.n_params, bool(np.corrcoef(res.log_decay, log_lam)[0, 1] > 0.5)
            ('extended', 23.0, True)
            >>> TimeVaryingNelsonSiegel(curves, taus).fit(filter="particle")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: filter must be 'extended' or 'unscented'; ...
        """
        return TimeVaryingNelsonSiegelResult._from_fit(self._fit_decay(filter_name=filter), self)
