# filepath: /src/cultivars/univariate/unobserved_components.py
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
r"""Structural time series: decomposition with a likelihood.

Harvey's unobserved-components family writes a series as the sum of
interpretable pieces, each with its own law of motion and its own
innovation,

.. math::

   y_t = \mu_t + \psi_t + \gamma_t + \varepsilon_t,
   \qquad
   \varepsilon_t \sim N(0, \sigma^2_\varepsilon),

with the trend :math:`\mu_t` one of three specifications in Harvey's
nomenclature -- the *local level* :math:`\mu_{t+1} = \mu_t + \eta_t`,
the *local linear trend* :math:`\mu_{t+1} = \mu_t + \beta_t + \eta_t`,
:math:`\beta_{t+1} = \beta_t + \zeta_t`, and the *smooth trend* that
sets :math:`\eta_t \equiv 0` so only the slope moves and the level is
an integrated random walk -- the cycle :math:`\psi_t` a damped bivariate
rotation with decay :math:`\rho` and frequency :math:`\lambda_c`, whose
period :math:`2\pi / \lambda_c` is estimated rather than imposed, and
the seasonal :math:`\gamma_t` the stochastic trigonometric form, all
harmonics sharing one innovation variance. The pieces are estimated
jointly by exact maximum likelihood through the Kalman filter and
reported by the fixed-interval smoother. That is the difference between
this and an ad-hoc filter: the components come with a likelihood,
information criteria and full-sample uncertainty bands, and the
end-of-sample estimates degrade honestly -- their variances grow --
instead of silently, which is where the Hodrick-Prescott filter is at
its worst. The connection is exact: the smooth trend's smoothed level
*is* the HP trend at :math:`\lambda = \sigma^2_\varepsilon /
\sigma^2_\zeta`, so this module estimates the penalty the filter asks
the user to assert.

Two commitments shape the surface. First, one initialization statement
rather than a hidden constant: the nonstationary states -- trend and
seasonal -- start from an approximate-diffuse prior with variance
``1e6``, the cycle from its exact stationary covariance. The likelihood
is exact Gaussian given that prior, the summary says so, and the
consequence is stated on the result: ``llf`` is comparable across
specifications on one sample, but differs from a strict-diffuse
likelihood by a constant in the number of diffuse states. Second, every
component is a smoothed path with a band, never a point estimate alone,
and the record carries nothing it cannot rebuild -- the system matrices
are re-derived from the public variances on demand, so ``state_space``,
``forecast`` and ``simulate`` cannot drift from the reported fit.

Layout. :class:`UnobservedComponents` validates ``trend``, ``cycle`` and
``seasonal`` on ``_UnobservedComponentsModel`` in ``_internals``, whose
``_fit_structural`` hands a ``_StructuralObjective`` -- log variances and
sigmoid-mapped cycle shape, so the surface is unconstrained -- to
``_maximize_likelihood`` under L-BFGS-B from data-scaled starts, then
smooths through ``_LinearGaussianStateSpace._from_structural_system``
and packs a ``_StructuralFit``. The system matrices and the state layout
come from ``_structural_matrices`` in ``_internals._systems``; the
parameter record is ``_StructuralParameters``.
:class:`UnobservedComponentsResult` rebuilds that record from its public
fields, exposes the fitted system as a
:class:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM`, and
takes the summary, the aligned series and the comparison machinery from
``_SummaryMixin``, ``_SeriesMixin`` and ``_ComparisonMixin``. The
constant-variance assumption is relaxed in
:class:`~cultivars.univariate.stochastic_volatility.UCSV`; the
penalized smoothers this family subsumes are in
:mod:`~cultivars.spectral.filters`.

References:
    Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and
    the Kalman Filter*. Cambridge University Press.

    Clark, P. K. (1987). The cyclical component of U.S. economic
    activity. *Quarterly Journal of Economics*, 102(4), 797-814.

    Harvey, A. C., & Jaeger, A. (1993). Detrending, stylized facts and
    the business cycle. *Journal of Applied Econometrics*, 8(3),
    231-247.

    Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State
    Space Methods* (2nd ed.). Oxford University Press.

Example:
    The smooth trend estimates the Hodrick-Prescott penalty instead of
    assuming it, and reproduces the HP trend at that penalty:

    >>> import numpy as np
    >>> from cultivars.spectral.filters import HodrickPrescottFilter
    >>> rng = np.random.default_rng(7)
    >>> n = 300
    >>> slope = np.cumsum(0.01 * rng.standard_normal(n))
    >>> y = np.cumsum(slope) + 0.5 * rng.standard_normal(n)
    >>> res = UnobservedComponents(y, trend="smooth").fit()
    >>> penalty = res.sigma2_irregular / res.sigma2_slope
    >>> bool(1000 < penalty < 2500)
    True
    >>> hp = HodrickPrescottFilter(penalty=penalty).filter(y).trend[:, 0]
    >>> bool(np.max(np.abs(hp - res.level)) < 1e-6)
    True
    >>> bool(res.level_std[0] > 1.5 * res.level_std[n // 2])
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..engine._core import SummaryTable
from ..engine._internals import (
    _ComparisonMixin,
    _SeriesMixin,
    _structural_matrices,
    _StructuralFit,
    _StructuralParameters,
    _SummaryMixin,
    _UnobservedComponentsModel,
)
from ..exceptions import SpecificationError
from ..state_space.linear_gaussian import LinearGaussianSSM

__all__ = ["UnobservedComponents", "UnobservedComponentsResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class UnobservedComponentsResult(_SummaryMixin, _SeriesMixin, _ComparisonMixin):
    r"""A fitted structural decomposition: components, variances and bands.

    The observed series is the sum of unobserved components, each with
    its own dynamics and its own innovation,

    .. math::

       y_t = \mu_t + \psi_t + \gamma_t + \varepsilon_t,
       \qquad
       \varepsilon_t \sim N(0, \sigma^2_\varepsilon),

    where the trend :math:`\mu_t` is a local level
    :math:`\mu_{t+1} = \mu_t + \eta_t`, a local linear trend
    :math:`\mu_{t+1} = \mu_t + \beta_t + \eta_t`,
    :math:`\beta_{t+1} = \beta_t + \zeta_t`, or the smooth trend that
    sets :math:`\eta_t \equiv 0` so the level is an integrated random
    walk; the cycle is the damped rotation

    .. math::

       \begin{pmatrix} \psi_{t+1} \\ \psi^*_{t+1} \end{pmatrix}
       = \rho \begin{pmatrix} \cos\lambda_c & \sin\lambda_c \\
       -\sin\lambda_c & \cos\lambda_c \end{pmatrix}
       \begin{pmatrix} \psi_t \\ \psi^*_t \end{pmatrix}
       + \begin{pmatrix} \kappa_t \\ \kappa^*_t \end{pmatrix},
       \qquad 0 < \rho < 1,

    with period :math:`2\pi / \lambda_c`; and the seasonal is the
    trigonometric form, one such rotation without damping at each
    harmonic :math:`\lambda_j = 2\pi j / s`, :math:`j = 1, \ldots,
    \lfloor s/2 \rfloor`, every harmonic sharing one innovation variance
    :math:`\sigma^2_\omega`. Everything is estimated by exact maximum
    likelihood through the Kalman filter, and every component path
    reported here is the full-sample smoothed estimate
    :math:`E[\cdot \mid y_1, \ldots, y_n]` with its smoothed standard
    deviation, so the bands widen at both ends of the sample -- the
    behaviour an ad-hoc filter hides.

    The record is immutable; it carries the smoothed state and its
    covariance privately and rebuilds the system matrices on demand, so
    the component properties, ``state_space``, ``forecast`` and
    ``simulate`` all read the same fitted variances. Information
    criteria, the summary and the likelihood-ratio test come from the
    mixins.

    Note:
        The likelihood is exact Gaussian under an *approximate*-diffuse
        initialization: the nonstationary states (level, slope, seasonal)
        start with variance ``1e6`` and the cycle starts at its stationary
        covariance. ``llf`` is therefore comparable across
        specifications on the same sample -- the ranking the
        ``information_criteria`` express -- but differs from an
        exact-diffuse likelihood by a constant that depends on the
        number of diffuse states, so it should not be compared with
        likelihoods from software that uses the exact-diffuse
        initialization. The cycle frequency is estimated on
        ``(0.05, pi - 0.05)``, so the cycle period lies between about two
        and about 126 observations; a slower swing is absorbed by the
        trend. ``n_params`` counts the variances and the two cycle shape
        parameters -- 2 for a local level, 3 for a local linear trend, 2
        for a smooth trend, plus 3 for the cycle and 1 for the seasonal.

    Attributes:
        endog: The observed series, shape ``(n,)``.
        trend: The trend specification (``"level"``, ``"lltrend"`` or
            ``"smooth"``).
        has_cycle: Whether a damped stochastic cycle was estimated.
        seasonal_period: The trigonometric seasonal period, or ``None``.
        sigma2_irregular: The irregular (observation noise) variance
            :math:`\sigma^2_\varepsilon`.
        sigma2_level: The level innovation variance :math:`\sigma^2_\eta`,
            or ``None`` for the smooth trend, whose level has no
            innovation of its own.
        sigma2_slope: The slope innovation variance :math:`\sigma^2_\zeta`,
            or ``None`` for the local level.
        cycle_rho: The cycle damping :math:`\rho`, or ``None``.
        cycle_freq: The cycle frequency :math:`\lambda_c` in radians per
            observation, or ``None``.
        sigma2_cycle: The cycle innovation variance :math:`\sigma^2_\kappa`,
            or ``None``.
        sigma2_seasonal: The seasonal innovation variance
            :math:`\sigma^2_\omega`, or ``None``.
        llf: The exact Gaussian log-likelihood at the optimum.
        nobs: The number of observations.
        n_params: The number of free parameters the likelihood was
            maximized over.

    See Also:
        * :class:`UnobservedComponents` -- the model that produces this
          record.
        * :class:`~cultivars.univariate.stochastic_volatility.UCSV` -- the
          local level whose two variances follow random walks.
        * :class:`~cultivars.state_space.linear_gaussian.LinearGaussianSSM` -- the
          substrate ``state_space`` exposes.

    References:
        Harvey, A. C. (1989). *Forecasting, Structural Time Series Models
        and the Kalman Filter*. Cambridge University Press, chs. 2 and 4.

        Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by
        State Space Methods* (2nd ed.). Oxford University Press, ch. 3.

    Example:
        A local level recovered from a noisy random walk, then a level
        plus a stochastic cycle whose period is read off the fit:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 300
        >>> level = np.cumsum(0.1 * rng.standard_normal(n))
        >>> y = level + 0.5 * rng.standard_normal(n)
        >>> res = UnobservedComponents(y, trend="level").fit()
        >>> res.trend, res.has_cycle, res.seasonal_period, res.nobs, res.n_params
        ('level', False, None, 300, 2.0)
        >>> bool(abs(res.sigma2_irregular - 0.25) < 0.1), bool(res.sigma2_level < 0.05)
        (True, True)
        >>> bool(np.mean((res.level - level) ** 2) < np.mean((y - level) ** 2))
        True
        >>> rng = np.random.default_rng(3)
        >>> n = 300
        >>> rho, freq = 0.95, 2 * np.pi / 20
        >>> rotation = rho * np.array([[np.cos(freq), np.sin(freq)], [-np.sin(freq), np.cos(freq)]])
        >>> cycle = np.zeros((n, 2))
        >>> for t in range(1, n):
        ...     cycle[t] = rotation @ cycle[t - 1] + 0.3 * rng.standard_normal(2)
        >>> walk = np.cumsum(0.05 * rng.standard_normal(n))
        >>> y = walk + cycle[:, 0] + 0.3 * rng.standard_normal(n)
        >>> res = UnobservedComponents(y, cycle=True).fit()
        >>> bool(abs(res.cycle_period - 20) < 2), bool(abs(res.cycle_rho - 0.95) < 0.05)
        (True, True)
        >>> bool(np.corrcoef(res.cycle, cycle[:, 0])[0, 1] > 0.9)
        True
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    """The observed series, shape ``(n,)``. Kept out of the repr."""
    trend: str
    """The trend specification: ``"level"``, ``"lltrend"`` or ``"smooth"``."""
    has_cycle: bool
    """Whether a damped stochastic cycle was estimated."""
    seasonal_period: int | None
    """The trigonometric seasonal period, or ``None``."""
    sigma2_irregular: float
    r"""The irregular variance :math:`\sigma^2_\varepsilon`."""
    sigma2_level: float | None
    """The level innovation variance, or ``None`` for the smooth trend."""
    sigma2_slope: float | None
    """The slope innovation variance, or ``None`` for the local level."""
    cycle_rho: float | None
    r"""The cycle damping :math:`\rho` in ``(0, 1)``, or ``None``."""
    cycle_freq: float | None
    r"""The cycle frequency :math:`\lambda_c` in radians, or ``None``."""
    sigma2_cycle: float | None
    """The cycle innovation variance, or ``None``."""
    sigma2_seasonal: float | None
    """The seasonal innovation variance shared by the harmonics, or ``None``."""
    llf: float
    """The exact Gaussian log-likelihood at the optimum."""
    nobs: int
    """The number of observations."""
    n_params: float
    """The number of free parameters the likelihood was maximized over."""
    _smoothed_state: npt.NDArray[np.float64] = field(repr=False)
    """The smoothed state path, shape ``(n, m)``. Kept out of the repr."""
    _smoothed_state_cov: npt.NDArray[np.float64] = field(repr=False)
    """The smoothed state covariances, shape ``(n, m, m)``. Kept out of the repr."""
    _slices: dict[str, slice] = field(repr=False)
    """Component name to state-index range. Kept out of the repr."""

    @classmethod
    def _from_fit(
        cls, fit: _StructuralFit, model: _UnobservedComponentsModel[UnobservedComponentsResult]
    ) -> UnobservedComponentsResult:
        """Assemble the public result from a raw fit and its specification.

        Copies the parameter record field by field, takes the sample
        facts and the smoothed paths from the fit, and the specification
        from the model, so the record can rebuild the system without
        either.

        Args:
            fit: The packed estimate from ``_fit_structural``.
            model: The specification the fit was produced for.

        Returns:
            The public result record.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> model = UnobservedComponents(y, trend="smooth")
            >>> res = UnobservedComponentsResult._from_fit(model._fit_structural(), model)
            >>> res.trend, res.sigma2_level, res.nobs
            ('smooth', None, 200)
        """
        p = fit.params
        return cls(
            endog=model.endog,
            trend=model.trend,
            has_cycle=model.cycle,
            seasonal_period=model.seasonal,
            sigma2_irregular=p.sigma2_irregular,
            sigma2_level=p.sigma2_level,
            sigma2_slope=p.sigma2_slope,
            cycle_rho=p.cycle_rho,
            cycle_freq=p.cycle_freq,
            sigma2_cycle=p.sigma2_cycle,
            sigma2_seasonal=p.sigma2_seasonal,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=float(fit.n_params),
            _smoothed_state=fit.smoothed_state,
            _smoothed_state_cov=fit.smoothed_state_cov,
            _slices=fit.slices,
        )

    @property
    def _params(self) -> _StructuralParameters:
        """The parameter record, rebuilt for the system builder.

        The record is not stored; it is re-assembled from the public
        fields each time a system matrix is needed, so the public fields
        are the single source of truth.

        Returns:
            The structural parameter record.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> res = UnobservedComponents(y, trend="level").fit()
            >>> res._params.sigma2_level == res.sigma2_level, res._params.cycle_rho
            (True, None)
        """
        return _StructuralParameters(
            sigma2_irregular=self.sigma2_irregular,
            sigma2_level=self.sigma2_level,
            sigma2_slope=self.sigma2_slope,
            cycle_rho=self.cycle_rho,
            cycle_freq=self.cycle_freq,
            sigma2_cycle=self.sigma2_cycle,
            sigma2_seasonal=self.sigma2_seasonal,
        )

    def _component(self, name: str) -> npt.NDArray[np.float64]:
        """One component's smoothed path, or a refusal naming what exists.

        The level, slope and cycle are single states and are read off
        the first index of their block; the seasonal is the sum of its
        harmonics, read off the block through the design row.

        Args:
            name: ``"level"``, ``"slope"``, ``"cycle"`` or ``"seasonal"``.

        Returns:
            The smoothed path, shape ``(n,)``.

        Raises:
            SpecificationError: If the specification lacks the component.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> res = UnobservedComponents(y, trend="level").fit()
            >>> res._component("level").shape
            (200,)
            >>> res._component("cycle")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: this specification has no cycle component; ...
        """
        if name not in self._slices:
            raise SpecificationError(
                f"this specification has no {name} component; it carries {sorted(self._slices)}."
            )
        block = self._slices[name]
        if name == "seasonal":
            design = _structural_matrices(
                self._params,
                trend=self.trend,
                cycle=self.has_cycle,
                seasonal=self.seasonal_period,
            )[0]
            pattern = design[0, block]
            return np.asarray(self._smoothed_state[:, block] @ pattern, dtype=np.float64)
        return np.asarray(self._smoothed_state[:, block.start], dtype=np.float64)

    def _component_std(self, name: str) -> npt.NDArray[np.float64]:
        """One component's smoothed standard deviation path.

        Single-state components read the diagonal of their block; the
        seasonal propagates the block covariance through the design row,
        so the harmonics' covariances are counted. The caller is
        responsible for the component existing.

        Args:
            name: ``"level"``, ``"slope"``, ``"cycle"`` or ``"seasonal"``.

        Returns:
            The smoothed standard deviations, shape ``(n,)``, floored at
            zero against rounding.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> res = UnobservedComponents(y, trend="level").fit()
            >>> std = res._component_std("level")
            >>> std.shape, bool(np.all(std > 0)), bool(std[0] > std[100])
            ((200,), True, True)
        """
        block = self._slices[name]
        if name == "seasonal":
            design = _structural_matrices(
                self._params,
                trend=self.trend,
                cycle=self.has_cycle,
                seasonal=self.seasonal_period,
            )[0]
            pattern = design[0, block]
            variances = np.einsum(
                "i,tij,j->t",
                pattern,
                self._smoothed_state_cov[:, block, block],
                pattern,
            )
            return np.asarray(np.sqrt(np.maximum(variances, 0.0)), dtype=np.float64)
        return np.asarray(
            np.sqrt(np.maximum(self._smoothed_state_cov[:, block.start, block.start], 0.0)),
            dtype=np.float64,
        )

    @property
    def level(self) -> npt.NDArray[np.float64]:
        r"""The smoothed trend level :math:`E[\mu_t \mid y]`, shape ``(n,)``.

        Present in every specification.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> walk = np.cumsum(0.2 * rng.standard_normal(200))
            >>> res = UnobservedComponents(walk + rng.standard_normal(200)).fit()
            >>> bool(np.corrcoef(res.level, walk)[0, 1] > 0.9)
            True
        """
        return self._component("level")

    @property
    def level_std(self) -> npt.NDArray[np.float64]:
        """The smoothed standard deviation of the level, shape ``(n,)``.

        Widest at the two ends of the sample, where fewer neighbours
        inform the estimate; the band ``level +/- 1.96 * level_std`` is
        the pointwise 95% band.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> res = UnobservedComponents(y).fit()
            >>> std = res.level_std
            >>> bool(std[0] > std[100]), bool(std[-1] > std[100])
            (True, True)
        """
        return self._component_std("level")

    @property
    def slope(self) -> npt.NDArray[np.float64]:
        r"""The smoothed trend slope :math:`E[\beta_t \mid y]`, shape ``(n,)``.

        Raises:
            SpecificationError: If the trend is a local level, which has
                no slope.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(7)
            >>> slope = np.cumsum(0.01 * rng.standard_normal(300))
            >>> y = np.cumsum(slope) + 0.5 * rng.standard_normal(300)
            >>> res = UnobservedComponents(y, trend="smooth").fit()
            >>> bool(np.corrcoef(res.slope, slope)[0, 1] > 0.95)
            True
            >>> UnobservedComponents(y, trend="level").fit().slope  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: this specification has no slope component; ...
        """
        return self._component("slope")

    @property
    def cycle(self) -> npt.NDArray[np.float64]:
        r"""The smoothed cycle :math:`E[\psi_t \mid y]`, shape ``(n,)``.

        Raises:
            SpecificationError: If no cycle was specified.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> UnobservedComponents(y).fit().cycle  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: this specification has no cycle component; ...
        """
        return self._component("cycle")

    @property
    def cycle_period(self) -> float:
        r"""The estimated cycle period :math:`2\pi / \lambda_c` in observations.

        Raises:
            SpecificationError: If no cycle was specified.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> UnobservedComponents(y).fit().cycle_period
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: this specification has no cycle component.
        """
        if self.cycle_freq is None:
            raise SpecificationError("this specification has no cycle component.")
        return float(2.0 * np.pi / self.cycle_freq)

    @property
    def seasonal(self) -> npt.NDArray[np.float64]:
        r"""The smoothed seasonal :math:`E[\gamma_t \mid y]`, shape ``(n,)``.

        The sum of the harmonics, so it is the seasonal effect on the
        observation and not the raw state block.

        Raises:
            SpecificationError: If no seasonal was specified.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(5)
            >>> n = 240
            >>> pattern = np.tile([1.5, -0.5, -1.5, 0.5], n // 4)
            >>> y = np.cumsum(0.1 * rng.standard_normal(n)) + pattern + 0.3 * rng.standard_normal(n)
            >>> res = UnobservedComponents(y, seasonal=4).fit()
            >>> bool(np.corrcoef(res.seasonal, pattern)[0, 1] > 0.99)
            True
        """
        return self._component("seasonal")

    @property
    def signal(self) -> npt.NDArray[np.float64]:
        """The systematic part: level plus cycle plus seasonal, shape ``(n,)``.

        Everything except the irregular, so ``signal + irregular`` is the
        observed series exactly.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> res = UnobservedComponents(y).fit()
            >>> bool(np.allclose(res.signal, res.level))
            True
        """
        total = self._component("level").copy()
        if "cycle" in self._slices:
            total += self._component("cycle")
        if "seasonal" in self._slices:
            total += self._component("seasonal")
        return total

    @property
    def irregular(self) -> npt.NDArray[np.float64]:
        r"""The observed series minus the smoothed signal, shape ``(n,)``.

        The smoothed estimate of :math:`\varepsilon_t`; its sample
        variance is below ``sigma2_irregular`` because smoothing shrinks
        the estimate toward zero.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> res = UnobservedComponents(y).fit()
            >>> bool(np.allclose(res.signal + res.irregular, y))
            True
            >>> bool(res.irregular.var() < res.sigma2_irregular)
            True
        """
        return np.asarray(self.endog - self.signal, dtype=np.float64)

    @property
    def state_space(self) -> LinearGaussianSSM:
        """The fitted system, re-applicable to data it was not estimated on.

        The exact linear-Gaussian emitter: filtering a new series through
        it reads that series with this decomposition's estimated
        variances, and its ``loglikelihood`` on the estimation sample
        reproduces ``llf``. The initial covariance is the
        approximate-diffuse one the likelihood was computed under.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> res = UnobservedComponents(y).fit()
            >>> system = res.state_space
            >>> system.k_states, bool(abs(system.loglikelihood(y) - res.llf) < 1e-8)
            (1, True)
        """
        model, _ = LinearGaussianSSM._from_structural_system(
            self._params,
            trend=self.trend,
            cycle=self.has_cycle,
            seasonal=self.seasonal_period,
        )
        return model

    def forecast(self, steps: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Out-of-sample forecasts with honest standard errors.

        Filters the estimation sample to the last state, then propagates
        mean and covariance through the fitted system,

        .. math::

           a_{n+h} = T a_{n+h-1}, \qquad
           P_{n+h} = T P_{n+h-1} T' + R Q R', \qquad
           \hat y_{n+h} = Z a_{n+h}, \quad
           \operatorname{Var}(y_{n+h}) = Z P_{n+h} Z' + \sigma^2_\varepsilon.

        A local level forecasts flat at the last filtered level; a trend
        with a slope extrapolates it; a cycle decays at :math:`\rho` per
        step; a seasonal repeats its last pattern. The standard errors
        grow without bound for the unit-root components.

        Args:
            steps: Horizons ahead, at least 1.

        Returns:
            ``(mean, std)`` arrays of length ``steps``; the standard
            errors include state uncertainty and the irregular, but not
            parameter uncertainty.

        Raises:
            SpecificationError: If ``steps`` is not positive.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> res = UnobservedComponents(y).fit()
            >>> mean, std = res.forecast(5)
            >>> mean.shape, bool(np.allclose(mean, mean[0])), bool(std[-1] > std[0])
            ((5,), True, True)
            >>> bool(std[0] ** 2 > res.sigma2_irregular)
            True
            >>> res.forecast(0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: steps must be at least 1; got 0.
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        design, transition, selection, state_cov, obs_cov, _, _ = _structural_matrices(
            self._params,
            trend=self.trend,
            cycle=self.has_cycle,
            seasonal=self.seasonal_period,
        )
        forward = self.state_space.filter(self.endog)
        mean = forward.filtered_state[-1].copy()
        cov = forward.filtered_state_cov[-1].copy()
        noise = selection @ state_cov @ selection.T
        out_mean = np.empty(steps)
        out_std = np.empty(steps)
        for step in range(steps):
            mean = transition @ mean
            cov = transition @ cov @ transition.T + noise
            out_mean[step] = float(design[0] @ mean)
            out_std[step] = float(np.sqrt(max(design[0] @ cov @ design[0] + obs_cov[0, 0], 0.0)))
        return out_mean, out_std

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """Aligned per-observation output: the decomposition itself.

        Returns:
            ``"observed"``, ``"level"``, then ``"slope"``, ``"cycle"`` and
            ``"seasonal"`` where present, then ``"irregular"``, each of
            shape ``(n,)``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> list(UnobservedComponents(y).fit()._series())
            ['observed', 'level', 'irregular']
            >>> list(UnobservedComponents(y, trend="smooth").fit()._series())
            ['observed', 'level', 'slope', 'irregular']
        """
        out: dict[str, npt.NDArray[np.float64]] = {"observed": self.endog}
        out["level"] = self.level
        if "slope" in self._slices:
            out["slope"] = self.slope
        if "cycle" in self._slices:
            out["cycle"] = self.cycle
        if "seasonal" in self._slices:
            out["seasonal"] = self.seasonal
        out["irregular"] = self.irregular
        return out

    def simulate(
        self,
        n: int = 200,
        *,
        seed: int | np.random.Generator | None = None,
        burn: int = 0,
    ) -> npt.NDArray[np.float64]:
        """A fresh sample path at the fitted variances.

        The unit-root components -- level, slope, seasonal -- start at
        zero rather than from the approximate-diffuse initial covariance
        the filter uses, so the sample is a random walk from the origin
        and not from a draw of standard deviation a thousand; the cycle
        starts from its stationary law. ``burn`` initial periods are
        discarded.

        Args:
            n: Observations kept.
            seed: Seed or generator.
            burn: Periods discarded from the start.

        Returns:
            An array of shape ``(n,)``.

        Raises:
            SpecificationError: If the counts are not usable.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> res = UnobservedComponents(y).fit()
            >>> path = res.simulate(100, seed=1)
            >>> path.shape, bool(abs(path[0]) < 3.0)
            ((100,), True)
            >>> bool(np.array_equal(path, res.simulate(100, seed=1)))
            True
            >>> res.simulate(0)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: n must be positive and burn non-negative; ...
        """
        if n < 1 or burn < 0:
            raise SpecificationError(f"n must be positive and burn non-negative; got {n}, {burn}.")
        design, transition, selection, state_cov, obs_cov, initial_cov, slices = (
            _structural_matrices(
                self._params, trend=self.trend, cycle=self.has_cycle, seasonal=self.seasonal_period
            )
        )
        start_cov = np.zeros_like(initial_cov)
        if "cycle" in slices:
            block = slices["cycle"]
            start_cov[block, block] = initial_cov[block, block]
        system = LinearGaussianSSM(
            design, obs_cov, transition, selection, state_cov, initial_state_cov=start_cov
        )
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        path = system.simulate(burn + n, seed=rng)
        return np.asarray(path[burn:, 0], dtype=np.float64)

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Returns:
            ``"UC[<trend>]"`` with ``"+cycle"`` and ``"+seasonal(<s>)"``
            appended where present.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> UnobservedComponents(y).fit()._comparison_label()
            'UC[level]'
        """
        pieces = [self.trend]
        if self.has_cycle:
            pieces.append("cycle")
        if self.seasonal_period is not None:
            pieces.append(f"seasonal({self.seasonal_period})")
        return "UC[" + "+".join(pieces) + "]"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        Returns:
            One row per estimated variance and cycle shape parameter;
            the metadata carries the trend, the log-likelihood, the
            sample size, the parameter count and both criteria; the
            notes state the initialization, the smoothed-band behaviour
            and, when a cycle is present, its period.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> table = UnobservedComponents(y).fit()._summary_table()
            >>> table.title, [row[0] for row in table.rows], len(table.notes)
            ('UC[level] Results', ['sigma2.irregular', 'sigma2.level'], 2)
        """
        ic = self.information_criteria
        rows: list[tuple[str, str]] = [("sigma2.irregular", f"{self.sigma2_irregular:.6g}")]
        if self.sigma2_level is not None:
            rows.append(("sigma2.level", f"{self.sigma2_level:.6g}"))
        if self.sigma2_slope is not None:
            rows.append(("sigma2.slope", f"{self.sigma2_slope:.6g}"))
        if self.cycle_rho is not None and self.cycle_freq is not None:
            rows.append(("cycle.rho", f"{self.cycle_rho:.4f}"))
            rows.append(("cycle.frequency", f"{self.cycle_freq:.4f}"))
            assert self.sigma2_cycle is not None
            rows.append(("sigma2.cycle", f"{self.sigma2_cycle:.6g}"))
        if self.sigma2_seasonal is not None:
            rows.append(("sigma2.seasonal", f"{self.sigma2_seasonal:.6g}"))
        notes = [
            "Exact Gaussian likelihood under an approximate-diffuse prior "
            "(variance 1e6) on the nonstationary states; the cycle starts "
            "at its stationary covariance.",
            "Component paths are full-sample (smoothed); their bands widen "
            "at the sample ends, which is the honest behavior an ad-hoc "
            "filter hides.",
        ]
        if self.cycle_freq is not None:
            notes.insert(
                0,
                f"The estimated cycle period is {self.cycle_period:.1f} observations.",
            )
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=(
                ("Trend", self.trend),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("AIC", f"{ic.aic:.3f}"),
                ("Parameters", f"{self.n_params:.0f}"),
                ("BIC", f"{ic.bic:.3f}"),
            ),
            columns=("", "estimate"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class UnobservedComponents(_UnobservedComponentsModel[UnobservedComponentsResult]):
    r"""Structural time-series decomposition by exact maximum likelihood.

    Harvey's basic structural model: a trend, optionally a damped
    stochastic cycle and a trigonometric seasonal, each with its own
    innovation variance, observed through an irregular. The
    specification is fixed at construction; ``fit`` maximizes the exact
    Gaussian likelihood over the log variances (and, for the cycle, the
    damping and frequency through sigmoid maps, so the search is
    unconstrained) by L-BFGS-B from data-scaled starts -- two of them
    when a cycle is present -- then runs the fixed-interval smoother so
    every component is reported with its band.

    The three trends answer different questions. ``"level"`` is the
    local level :math:`\mu_{t+1} = \mu_t + \eta_t`, an exponentially
    weighted moving average with its smoothing weight estimated.
    ``"lltrend"`` adds a random-walk slope, so both the level and its
    growth rate drift. ``"smooth"`` keeps the slope but removes the level
    innovation, an integrated random walk whose smoothed level is a
    maximum-likelihood cubic smoothing spline -- the model-based
    counterpart of the Hodrick-Prescott filter with its smoothing
    constant estimated rather than imposed.

    Attributes:
        _endog: The observed series as a float array, shape ``(n,)``.
        _trend: The trend specification.
        _cycle: Whether the damped stochastic cycle is present.
        _seasonal: The trigonometric seasonal period, or ``None``.

    Args:
        endog: The observed series.
        trend: ``"level"`` for a local level, ``"lltrend"`` for a local
            linear trend, ``"smooth"`` for an integrated random walk.
        cycle: Whether to estimate a damped stochastic cycle (damping,
            frequency and variance all estimated).
        seasonal: Trigonometric seasonal period, at least 2, or ``None``.

    Raises:
        SpecificationError: If ``trend`` is not one of the three names
            or ``seasonal`` is below 2.
        DimensionError: If the series is shorter than ``3 m + 10`` for
            the specification's ``m`` states, or is not one-dimensional.

    See Also:
        * :class:`UnobservedComponentsResult` -- the record ``fit``
          returns.
        * :class:`~cultivars.univariate.stochastic_volatility.UCSV` -- the
          local level with time-varying variances, estimated by Gibbs
          sampling.
        * :class:`~cultivars.univariate.autoregression.AR` -- the
          reduced-form alternative to a structural decomposition.

    References:
        Harvey, A. C. (1989). *Forecasting, Structural Time Series Models
        and the Kalman Filter*. Cambridge University Press, ch. 2.

        Harvey, A. C., & Jaeger, A. (1993). Detrending, stylized facts
        and the business cycle. *Journal of Applied Econometrics*, 8(3),
        231-247.

    Example:
        A local level recovers the random walk under the noise, and the
        three trends are ranked on BIC for a series whose slope drifts:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 300
        >>> level = np.cumsum(0.1 * rng.standard_normal(n))
        >>> y = level + 0.5 * rng.standard_normal(n)
        >>> res = UnobservedComponents(y, trend="level").fit()
        >>> bool(np.mean((res.level - level) ** 2) < np.mean((y - level) ** 2))
        True
        >>> rng = np.random.default_rng(7)
        >>> slope = np.cumsum(0.01 * rng.standard_normal(n))
        >>> y = np.cumsum(slope) + 0.5 * rng.standard_normal(n)
        >>> trends = ("level", "lltrend", "smooth")
        >>> fits = {t: UnobservedComponents(y, trend=t).fit() for t in trends}
        >>> bic = {t: fit.information_criteria.bic for t, fit in fits.items()}
        >>> bool(bic["smooth"] < bic["lltrend"] < bic["level"])
        True
        >>> UnobservedComponents(y, trend="quadratic")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        cultivars.exceptions.SpecificationError: trend must be 'level', 'lltrend', or 'smooth'; ...
    """

    __slots__ = ()

    def fit(self) -> UnobservedComponentsResult:
        """Maximize the exact likelihood and smooth every component.

        Returns:
            The fitted decomposition.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.cumsum(0.2 * rng.standard_normal(200)) + rng.standard_normal(200)
            >>> res = UnobservedComponents(y, seasonal=4).fit()
            >>> res.seasonal_period, res.n_params, bool(res.sigma2_seasonal < res.sigma2_level)
            (4, 3.0, True)
        """
        return UnobservedComponentsResult._from_fit(self._fit_structural(), self)
