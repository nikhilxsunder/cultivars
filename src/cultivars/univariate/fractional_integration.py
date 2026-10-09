# filepath: /src/cultivars/univariate/fractional_integration.py
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
r"""ARFIMA(p, d, q) -- fractionally integrated ARMA with long memory.

The model

.. math::

   \phi(L)\,(1 - L)^d\,(y_t - \mu) = \theta(L)\,\varepsilon_t,
   \qquad
   (1 - L)^d = \sum_{k=0}^{\infty} \frac{\Gamma(k - d)}{\Gamma(-d)\,\Gamma(k + 1)}\,L^k,

in which the fractional operator interpolates between a stationary ARMA
(:math:`d = 0`) and a unit-root process (:math:`d = 1`). For
:math:`0 < d < 0.5` the process is stationary with autocorrelations
decaying as :math:`k^{2d-1}` and a spectrum behaving as
:math:`\omega^{-2d}` at the origin -- hyperbolic rather than geometric
memory, which no finite ARMA reproduces without a near-unit root. The
likelihood is

.. math::

   \log L(\mu, d, \phi, \theta, \sigma^2)
   = \log L_{\text{ARMA}}\bigl(w_t(\mu, d);\, \phi, \theta, \sigma^2\bigr),
   \qquad w_t = (1 - L)^d (y_t - \mu),

with the inner term the exact Kalman likelihood of the short-memory
block on the differenced series; :math:`\mu`, :math:`d` and the block
are estimated jointly, the filter re-applied at every draw because
:math:`w_t` is itself a function of :math:`d`. The approximation
relative to Sowell's exact likelihood is the truncation of the filter,
not the ARMA step.

Two commitments shape the surface. First, stationarity is not the
short-memory test. It needs :math:`|d| < 0.5` *and* a stable
autoregressive block, so :class:`ARFIMAResult` overrides the inherited
property rather than reporting a process as stationary because its ARMA
block happens to be well behaved; mean reversion, :math:`d < 1`, is the
weaker and separately useful condition and has its own property. Second,
the estimator confines :math:`d` to :math:`(-0.499, 0.499)` through a
scaled ``tanh`` reparameterization, so the stationary-ARMA machinery is
always valid on the differenced series and an estimate at the boundary
is a sign the data want integer differencing rather than a long-memory
fit. The cost of that choice is that the stationarity verdict cannot
fail on :math:`d`; the summary prints the range so the boundary is
visible.

Layout. :class:`ARFIMA` validates ``order`` through
``validate_order_tuple``, ``trend`` through ``validate_choice`` against
:data:`~cultivars.typing.Trend`, and ``truncation`` through
``validate_order`` on the ``_FractionalIntegrationModel`` base in
``_internals``; ``_build_objective`` starts :math:`d` at
``local_whittle_d`` from ``_core`` (shared with
:mod:`~cultivars.diagnostics.long_memory`) and the ARMA block at its
conditional estimate, and hands a ``_FractionalIntegrationObjective`` to
``_maximize_likelihood``. The objective's ``differenced`` applies
``fractional_difference`` with the weights of
``fractional_difference_weights``, and its ``state_space`` builds the
short-memory block through ``_LinearGaussianStateSpace._from_arma``,
the same form :mod:`~cultivars.univariate.box_jenkins` uses. The packed
``_FractionalIntegrationFit`` is assembled by
:meth:`ARFIMAResult._from_fit`. Semiparametric estimates of :math:`d`
that need no short-memory specification are
:func:`~cultivars.diagnostics.long_memory.gph` and
:func:`~cultivars.diagnostics.long_memory.local_whittle`; long memory in
the variance is :class:`~cultivars.univariate.conditional_variance.FIGARCH`.

References:
    Granger, C. W. J., & Joyeux, R. (1980). An introduction to
    long-memory time series models and fractional differencing. *Journal
    of Time Series Analysis*, 1(1), 15-29.

    Hosking, J. R. M. (1981). Fractional differencing. *Biometrika*,
    68(1), 165-176.

    Sowell, F. (1992). Maximum likelihood estimation of stationary
    univariate fractionally integrated time series models. *Journal of
    Econometrics*, 53(1-3), 165-188.

    Beran, J. (1994). *Statistics for Long-Memory Processes*. Chapman &
    Hall.

Example:
    A long-memory series is not a short-memory one with a large root:

    >>> import numpy as np
    >>> from cultivars._core import fractional_difference_weights
    >>> from cultivars.univariate.box_jenkins import ARMA
    >>> rng = np.random.default_rng(0)
    >>> n = 600
    >>> w = fractional_difference_weights(-0.4, n)
    >>> y = np.convolve(rng.standard_normal(n + 300), w)[300 : 300 + n]
    >>> long = ARFIMA(y, order=(0, 0)).fit()
    >>> short = ARMA(y, order=(2, 0)).fit()
    >>> bool(abs(long.d - 0.4) < 0.1), long.is_stationary
    (True, True)
    >>> bool(long.information_criteria.bic < short.information_criteria.bic)
    True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..engine._core import _D_MAX, InformationCriteria, SummaryTable
from ..engine._internals import (
    _ComparisonMixin,
    _FractionalIntegrationFit,
    _FractionalIntegrationModel,
    _InvertibilityMixin,
    _SeriesMixin,
    _StationarityMixin,
    _SummaryMixin,
)

__all__ = ["ARFIMA", "ARFIMAResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ARFIMAResult(
    _SummaryMixin, _SeriesMixin, _ComparisonMixin, _StationarityMixin, _InvertibilityMixin
):
    r"""A fitted fractionally integrated ARMA.

    The model

    .. math::

        \phi(L)\,(1 - L)^d\,(y_t - \mu) = \theta(L)\,\varepsilon_t,
        \qquad \varepsilon_t \sim N(0, \sigma^2),

    with the fractional operator expanded as

    .. math::

        (1 - L)^d = \sum_{k=0}^{\infty} \pi_k L^k,
        \qquad \pi_k = \frac{\Gamma(k - d)}{\Gamma(-d)\,\Gamma(k + 1)},

    cut at ``truncation`` terms. The autocorrelations decay as
    :math:`k^{2d-1}` rather than geometrically, so ``d`` is the memory
    parameter: zero is a short-memory ARMA, :math:`0 < d < 0.5` a
    stationary long-memory process, :math:`d \ge 0.5` non-stationary,
    and :math:`d = 1` a unit root. The record carries the estimate of
    ``d`` with the short-memory block and the innovation variance; the
    fitted values and residuals are on the fractionally differenced
    scale, where the model is an ARMA, not on the levels.

    Attributes:
        endog: The full observed series.
        fittedvalues: One-step fitted values on the fractionally differenced series.
        resid: One-step residuals on the fractionally differenced series.
        llf: Maximized joint log-likelihood.
        nobs: Observations the likelihood was evaluated on.
        n_params: Free parameter count, including ``d`` and the variance.
        order: Short-memory ``(p, q)``.
        truncation: Length of the fractional-difference filter.
        d: Estimated fractional integration order.
        mean: Estimated mean, or ``None`` when ``trend == "n"``.
        ar_params: Short-memory AR coefficients.
        ma_params: Short-memory MA coefficients.
        sigma2: Innovation variance.

    Note:
        The estimator confines ``d`` to :math:`(-0.499, 0.499)` through
        a scaled ``tanh``, so every fit satisfies ``|d| < 0.5`` and the
        stationarity verdict reduces to the autoregressive block; a
        series with a unit root comes back with ``d`` pinned at the
        boundary and ``is_stationary`` still ``True``. Read ``d`` near
        :math:`\pm 0.499` as the data asking for integer differencing
        first, and :attr:`is_mean_reverting` as a condition the estimator
        cannot violate. ``ar_params`` and ``ma_params`` are searched
        through the partial autocorrelations, so the stability and
        invertibility verdicts verify the transform. Standard errors are
        not yet reported by this estimator.

    See Also:
        * :class:`ARFIMA` -- the specification that produces this.
        * :class:`~cultivars.univariate.box_jenkins.ARMAResult` -- the
          ``d = 0`` case with the exact likelihood.
        * :class:`~cultivars.univariate.conditional_variance.FIGARCHResult`
          -- long memory in the variance rather than the level.
        * :mod:`~cultivars.diagnostics.unit_roots` -- whether integer
          differencing is wanted before a fractional fit.

    References:
        Granger, C. W. J., & Joyeux, R. (1980). An introduction to
        long-memory time series models and fractional differencing.
        *Journal of Time Series Analysis*, 1(1), 15-29.

        Hosking, J. R. M. (1981). Fractional differencing. *Biometrika*,
        68(1), 165-176.

        Sowell, F. (1992). Maximum likelihood estimation of stationary
        univariate fractionally integrated time series models. *Journal
        of Econometrics*, 53(1-3), 165-188.

    Example:
        An ARFIMA(0, d, 0) with :math:`d = 0.3`, simulated by filtering
        white noise with the weights of :math:`(1 - L)^{-0.3}`:

        >>> import numpy as np
        >>> from cultivars._core import fractional_difference_weights
        >>> rng = np.random.default_rng(0)
        >>> n = 600
        >>> w = fractional_difference_weights(-0.3, n)
        >>> y = np.convolve(rng.standard_normal(n + 300), w)[300 : 300 + n]
        >>> res = ARFIMA(y, order=(0, 0)).fit()
        >>> list(res.params), res.nobs, res.n_params
        (['mean', 'd', 'sigma2'], 600, 3)
        >>> bool(abs(res.d - 0.3) < 0.1), res.is_stationary, res.has_long_memory
        (True, True, True)
        >>> res.fittedvalues.shape == res.endog.shape
        True
    """

    endog: npt.NDArray[np.float64]
    """``(n,)`` observed series, as given."""

    fittedvalues: npt.NDArray[np.float64]
    """``(nobs,)`` one-step fitted values of the fractionally differenced series."""

    resid: npt.NDArray[np.float64]
    """``(nobs,)`` one-step residuals of the fractionally differenced series."""

    llf: float
    """Maximized joint Gaussian log-likelihood."""

    nobs: int
    """Observations the likelihood was evaluated on."""

    n_params: float
    """Free parameters: the mean if estimated, ``d``, ``p + q``, and ``sigma2``."""

    order: tuple[int, int]
    """Short-memory ``(p, q)``."""

    truncation: int
    """Number of fractional-difference weights applied."""

    d: float
    """Fractional integration order, inside :math:`(-0.499, 0.499)`."""

    mean: float | None
    """Estimated level :math:`\\mu`, or ``None`` under ``trend="n"``."""

    ar_params: npt.NDArray[np.float64]
    """``(p,)`` short-memory AR coefficients."""

    ma_params: npt.NDArray[np.float64]
    """``(q,)`` short-memory MA coefficients, plus-sign convention."""

    sigma2: float
    """Innovation variance."""

    @classmethod
    def _from_fit(
        cls,
        fit: _FractionalIntegrationFit,
        model: _FractionalIntegrationModel[ARFIMAResult],
    ) -> ARFIMAResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The raw estimator output.
            model: The specification, read for the series, the order and
                the truncation.

        Returns:
            The assembled result.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> model = ARFIMA(rng.standard_normal(300), order=(1, 0), truncation=100)
            >>> res = ARFIMAResult._from_fit(model._fit_family(), model)
            >>> res.order, res.truncation, res.mean is not None
            ((1, 0), 100, True)
        """
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            order=model.order,
            truncation=model.truncation,
            d=fit.d,
            mean=fit.mean,
            ar_params=fit.ar_params,
            ma_params=fit.ma_params,
            sigma2=fit.sigma2,
        )

    @property
    def is_stationary(self) -> bool:
        """Whether the process is covariance stationary.

        Overrides the short-memory test because stationarity here has two
        conditions, not one: the fractional order must satisfy ``|d| < 0.5``
        *and* the short-memory autoregressive block must be stable. Inheriting
        the AR-only test would report a unit-root-like ``d = 0.9`` process as
        stationary purely because its ARMA block is well behaved.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = ARFIMA(rng.standard_normal(400), order=(1, 0)).fit()
            >>> res.is_stationary == (abs(res.d) < 0.5 and res.stability.is_stable)
            True
        """
        return abs(self.d) < 0.5 and self.stability.is_stable

    @property
    def has_long_memory(self) -> bool:
        """Whether ``d`` is far enough from zero to imply hyperbolic decay.

        A threshold on the point estimate at ``1e-3`` in either direction,
        not a test; white noise typically returns a ``d`` of a few
        hundredths and so reads as long memory here.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = ARFIMA(rng.standard_normal(400), order=(0, 0), trend="n").fit()
            >>> res.has_long_memory == (abs(res.d) > 1e-3)
            True
        """
        return abs(self.d) > 1e-3

    @property
    def is_mean_reverting(self) -> bool:
        """Whether shocks die out, which holds on the wider range ``d < 1``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> walk = ARFIMA(np.cumsum(rng.standard_normal(400)), order=(0, 0)).fit()
            >>> round(walk.d, 2), walk.is_mean_reverting
            (0.5, True)
        """
        return self.d < 1.0 and self.stability.is_stable

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order.

        ``mean`` when estimated, then ``d``, ``ar.L1`` .., ``ma.L1`` ..
        and ``sigma2``.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = rng.standard_normal(300)
            >>> list(ARFIMA(y, order=(1, 1), truncation=50).fit().params)
            ['mean', 'd', 'ar.L1', 'ma.L1', 'sigma2']
        """
        out: dict[str, float] = {}
        if self.mean is not None:
            out["mean"] = self.mean
        out["d"] = self.d
        for i, value in enumerate(self.ar_params, start=1):
            out[f"ar.L{i}"] = float(value)
        for i, value in enumerate(self.ma_params, start=1):
            out[f"ma.L{i}"] = float(value)
        out["sigma2"] = self.sigma2
        return out

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = ARFIMA(rng.standard_normal(300), order=(2, 1), truncation=50).fit()
            >>> res._comparison_label()
            'ARFIMA(2, d, 1)'
        """
        p, q = self.order
        return f"ARFIMA({p}, d, {q})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        Metadata carries the label, ``d``, the truncation, sample size,
        and the likelihood with its criteria; the coefficient table is
        :attr:`params` in order; the notes give the stationarity verdict
        with its two conditions, mean reversion and long memory, and the
        range the estimator confines ``d`` to.

        Returns:
            The :class:`~cultivars._core.SummaryTable`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> table = ARFIMA(rng.standard_normal(300), order=(0, 0)).fit()._summary_table()
            >>> table.title, table.metadata[2][0], table.notes[0][:18]
            ('ARFIMA(0, d, 0) Results', 'd', 'Stationary: True  ')
        """
        ic: InformationCriteria = self.information_criteria
        p, q = self.order
        return SummaryTable(
            title=f"ARFIMA({p}, d, {q}) Results",
            metadata=(
                ("Model", f"ARFIMA({p}, d, {q})"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("d", f"{self.d:.4f}"),
                ("AIC", f"{ic.aic:.3f}"),
                ("Truncation", f"{self.truncation}"),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=(
                f"Stationary: {self.is_stationary}   "
                f"(|d| < 0.5 and stable AR block; d = {self.d:.4f})",
                f"Mean reverting: {self.is_mean_reverting}   Long memory: {self.has_long_memory}",
                f"d is confined to (-{_D_MAX}, {_D_MAX}) by the estimator's reparameterization.",
                "Standard errors are not yet available for this estimator.",
            ),
        )


class ARFIMA(_FractionalIntegrationModel[ARFIMAResult]):
    r"""Fractionally integrated ARMA specification.

    The model :math:`\phi(L)(1 - L)^d (y_t - \mu) = \theta(L)\varepsilon_t`
    with :math:`\mu`, :math:`d`, the short-memory block and
    :math:`\sigma^2` estimated jointly. The filter is re-applied at every
    draw, because the differenced series is itself a function of
    :math:`d`; the search starts from a local Whittle estimate of
    :math:`d` and is confined to :math:`|d| < 0.499`, so a series that
    wants integer differencing shows up with :math:`d` at the boundary.
    The short-memory block is searched through the partial
    autocorrelations and so is stationary and invertible by
    construction. Mean reversion and the fractional order trade off
    against a near-unit autoregressive root -- :math:`d = 0.3` with no
    AR term and :math:`d = -0.5` with :math:`\phi = 0.93` fit a short
    sample about equally well -- so keep ``order`` small unless the
    short-memory structure is known.

    Attributes:
        _endog: The validated series.
        _p: Short-memory AR order.
        _q: Short-memory MA order.
        _const: Whether a mean is estimated.
        _truncation: Length of the fractional-difference filter.

    Args:
        endog: The series.
        order: Short-memory ``(p, q)``.
        trend: ``"c"`` to estimate a mean, ``"n"`` to omit it.
        truncation: Fractional-filter length; defaults to the sample size.

    Raises:
        SpecificationError: If an order is negative, the trend is
            unrecognized, or ``truncation`` is not positive.
        DimensionError: If ``endog`` is not one-dimensional or is shorter
            than ``2(p + q) + 8``.
        NumericalError: If ``endog`` contains non-finite values.

    See Also:
        * :class:`ARFIMAResult` -- what :meth:`fit` returns.
        * :class:`~cultivars.univariate.box_jenkins.ARIMA` -- integer
          differencing, the model for ``d`` at the boundary.
        * :class:`~cultivars.univariate.conditional_variance.FIGARCH` --
          the fractional filter on the variance.

    References:
        Granger, C. W. J., & Joyeux, R. (1980). An introduction to
        long-memory time series models and fractional differencing.
        *Journal of Time Series Analysis*, 1(1), 15-29.

        Hosking, J. R. M. (1981). Fractional differencing. *Biometrika*,
        68(1), 165-176.

    Example:
        Long memory preferred to a short-memory block on a fractionally
        integrated series:

        >>> import numpy as np
        >>> from cultivars._core import fractional_difference_weights
        >>> rng = np.random.default_rng(0)
        >>> n = 600
        >>> w = fractional_difference_weights(-0.3, n)
        >>> y = np.convolve(rng.standard_normal(n + 300), w)[300 : 300 + n]
        >>> pure = ARFIMA(y, order=(0, 0)).fit()
        >>> with_ar = ARFIMA(y, order=(1, 0)).fit()
        >>> bool(abs(with_ar.ar_params[0]) < 0.15), bool(abs(with_ar.d - pure.d) < 0.1)
        (True, True)
        >>> pure.compare(with_ar, criterion="bic").rows[0][0]
        'ARFIMA(0, d, 0)'
    """

    __slots__ = ()

    def fit(self) -> ARFIMAResult:
        """Estimate ``(mu, d, phi, theta, sigma2)`` by joint maximum likelihood.

        Returns:
            The fitted :class:`ARFIMAResult`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> res = ARFIMA(rng.standard_normal(400), order=(0, 0), trend="n").fit()
            >>> list(res.params), bool(abs(res.d) < 0.1)
            (['d', 'sigma2'], True)
        """
        return ARFIMAResult._from_fit(self._fit_family(), self)
