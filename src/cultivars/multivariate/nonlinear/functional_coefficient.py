# filepath: /src/cultivars/multivariate/nonlinear/functional_coefficient.py
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
r"""The functional-coefficient VAR: every state value gets its own system.

This is the nonparametric end of the observed-transition family. A threshold
VAR assigns each value of an observable state one of two systems; a
smooth-transition VAR blends the same two anchors; here the coefficient
matrices are unknown smooth *functions* of the state,

.. math::

   y_t = c(z_t) + \sum_{i=1}^{p} A_i(z_t)\, y_{t-i} + u_t,
   \qquad z_t = s_{t-d},

with :math:`s` one of the variables or an external series (Chen and Tsay,
1993; Cai, Fan and Yao, 2000), and nothing is assumed about the functions
but smoothness. Estimation is local-linear kernel regression: at each
evaluation point :math:`u` the level and the state-derivative of every
coefficient solve one Epanechnikov-weighted least squares on the augmented
design :math:`[x_t,\ x_t (z_t - u)]`,

.. math::

   \min_{a, b} \sum_t K_h(z_t - u)\,
   \bigl\lVert y_t - a' x_t - b' x_t (z_t - u) \bigr\rVert^2,

all equations sharing the weights, and :math:`a(u)` is the reported curve.
The payoff is diagnostic as much as predictive: the estimated curves show
whether transmission actually varies in the state, and whether its
variation looks like a threshold, a smooth transition, or something
neither parametric family can express -- which is why this model is the
natural specification check on both.

Two commitments shape the surface. First, the price of assuming nothing is
paid in two currencies the result reports honestly. Statistical: there is
no likelihood and no parameter count -- the estimator's complexity is the
trace of its smoother matrix, reported as ``effective_params``, and no
information criteria are derivable from a kernel fit, so none are
reported. Practical: everything is local, so the curves are trustworthy
only where the state has mass -- they are reported on a trimmed grid,
refuse to extrapolate beyond it, carry pointwise bands that are
conditional on the bandwidth rather than accounting for its selection,
and read coefficients between grid points by linear interpolation. Second,
the bandwidth is the model's one tuning constant and is chosen, when
unstated, by exact leave-one-out cross-validation -- the deletion residual
from the smoother diagonal, no refitting -- over a geometric grid around
the Silverman rule :math:`2.34\,\hat\sigma_z\, n^{-1/5}`; there is no
delay search, because searching the delay and the bandwidth together would
let the smoother trade one against the other invisibly.

One identification fact is handled by construction rather than by a
warning. When the state is one of the design's own lag columns -- the
self-exciting case with ``delay <= order`` -- the intercept function's
local slope :math:`1 \cdot (z_t - u)` is *exactly* collinear with the
design, so the intercept is fitted local-constant while every other
coefficient stays local-linear; the engine's ``_state_in_design`` note
states the algebra.

Layout. :class:`FunctionalCoefficientVAR` validates on
``_FunctionalCoefficientVectorAutoRegressionModel`` in ``_internals``,
which takes the named-or-external state, its delay and the aligned design
from ``_ObservedRegimeVectorModel`` -- the base the threshold and
smooth-transition VARs share -- and adds the kernel engine: ``_local_fit``
solves one weighted system, ``_cross_validation_score`` scores a candidate
bandwidth, and ``_fit_functional`` lays the trimmed grid, selects the
bandwidth, evaluates the curves and their sandwich standard errors, and
packs a ``_VectorFunctionalFit``. :class:`FunctionalCoefficientVARResult`
takes its summary from ``_SummaryMixin`` and builds the state-frozen
companion for ``irf`` and ``stability_curve`` through
:func:`~cultivars._core.companion_matrix`. The two-system and two-anchor
members of the family are :mod:`~cultivars.multivariate.nonlinear.threshold`
and :mod:`~cultivars.multivariate.nonlinear.smooth_transition`; the
model whose coefficients drift in time rather than in a state is
:mod:`~cultivars.multivariate.nonlinear.time_varying`.

References:
    Chen, R., & Tsay, R. S. (1993). Functional-coefficient autoregressive
    models. *Journal of the American Statistical Association*, 88(421),
    298-308.

    Cai, Z., Fan, J., & Yao, Q. (2000). Functional-coefficient regression
    models for nonlinear time series. *Journal of the American
    Statistical Association*, 95(451), 941-956.

    Hastie, T., & Tibshirani, R. (1993). Varying-coefficient models.
    *Journal of the Royal Statistical Society, Series B*, 55(4), 757-796.

    Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its
    Applications*. Chapman and Hall. Chapter 3.

Example:
    The specification check in action: a threshold truth, in which the
    own-lag coefficient of ``y1`` is 0.7 below zero and -0.3 above, read
    through the kernel. The curve is flat near 0.7 at the low end and
    descends through the threshold, the shape a threshold VAR would be
    the parsimonious account of:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((300, 2))
    >>> for t in range(1, 300):
    ...     a = 0.7 if y[t - 1, 0] <= 0.0 else -0.3
    ...     y[t, 0] = a * y[t - 1, 0] + 0.5 * rng.standard_normal()
    ...     y[t, 1] = 0.4 * y[t - 1, 1] + 0.3 * y[t - 1, 0] + 0.5 * rng.standard_normal()
    >>> res = FunctionalCoefficientVAR(y, order=1, transition_variable="y1").fit(n_grid=21)
    >>> res.bandwidth_searched, res.effective_params > 3.0
    (True, True)
    >>> curve = res.coefficient_curve("y1", "y1.L1")[:, 1]
    >>> bool(abs(curve[:5].mean() - 0.7) < 0.1), bool(curve[-5:].mean() < 0.3)
    (True, True)
    >>> bool(np.all(np.diff(curve[5:]) < 0.0))
    True
    >>> low, high = res.state_range
    >>> bool(res.coefficients_at(low)[0, 0, 0] > res.coefficients_at(high)[0, 0, 0])
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable, companion_matrix
from ..._internals import (
    _FunctionalCoefficientVectorAutoRegressionModel,
    _SummaryMixin,
    _VectorFunctionalFit,
)
from ...exceptions import SpecificationError

__all__ = ["FunctionalCoefficientVAR", "FunctionalCoefficientVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class FunctionalCoefficientVARResult(_SummaryMixin):
    r"""A fitted functional-coefficient VAR: coefficient curves over the state.

    The estimate of

    .. math::

       y_t = c(z_t) + \sum_{i=1}^{p} A_i(z_t)\, y_{t-i} + u_t,
       \qquad z_t = s_{t-d},

    where every coefficient is an unknown smooth function of the delayed
    state :math:`z_t` and :math:`s` is either one of the variables (the
    self-exciting case) or an external series. At each grid point
    :math:`u` the curves come from one Epanechnikov-weighted local-linear
    least squares on the augmented design :math:`[x_t,\ x_t (z_t - u)]`,

    .. math::

       \min_{a, b} \sum_t K_h(z_t - u)\,
       \bigl\lVert y_t - a' x_t - b' x_t (z_t - u) \bigr\rVert^2,

    all equations sharing the weights; the level :math:`a(u)` is reported
    and the slope :math:`b(u)` discarded. The curves are evaluated on a
    grid over the trimmed state range and the result reads coefficients at
    any state value by linear interpolation between grid points.

    Deliberately absent: ``llf``, ``n_params``, information criteria. A
    kernel estimator maximizes no likelihood; its complexity is
    :attr:`effective_params`, the trace of the smoother matrix.

    Note:
        Everything is local to the state's support. The curves live on
        ``[grid[0], grid[-1]]`` -- the state range after trimming ``trim``
        from each tail -- and every method that takes a state value refuses
        one outside it rather than extrapolating. The bands in
        :meth:`coefficient_curve` are one pointwise standard error,
        conditional on the bandwidth: they measure precision at this
        smoothing, not uncertainty over smoothings. ``sigma_u`` is a single
        covariance, corrected by ``nobs - effective_params``, under the
        model's assumption that the innovations are homoskedastic in the
        state. In the self-exciting case with ``delay <= order`` the
        intercept is fitted local-constant because its local slope is
        exactly collinear with the design.

    Attributes:
        endog: The observed panel.
        names: Variable labels, in column order.
        order: Autoregressive order.
        trend: Deterministic specification.
        delay: Delay of the state variable.
        state_name: The state variable's label -- a variable name when
            self-exciting, ``"external"`` otherwise.
        state_values: The delayed state ``z_t``, aligned with ``resid``.
        transition_series: The raw state series, aligned with ``endog``.
        grid: ``(G,)`` state values the curves are evaluated on.
        curves: ``(G, w, k)`` local-linear level coefficients, design-major:
            ``curves[g, :, i]`` is equation ``i``'s coefficient vector at
            ``grid[g]``, columns ordered deterministics first, then lags.
        curve_se: ``(G, w, k)`` pointwise standard errors, conditional on
            the bandwidth.
        bandwidth: The bandwidth the curves were estimated at.
        bandwidth_searched: Whether the bandwidth came from leave-one-out
            cross-validation rather than the caller.
        effective_params: Trace of the smoother matrix -- the kernel
            estimator's analogue of a parameter count.
        sigma_u: ``(k, k)`` innovation covariance, corrected by the
            effective degrees of freedom and assumed constant in the state.
        resid: Residuals of the local fits at the observed states.
        fittedvalues: One-step conditional means at the observed states.
        nobs: Effective sample size.

    See Also:
        * :class:`FunctionalCoefficientVAR` -- the model that produces this
          record.
        * :class:`~cultivars.multivariate.nonlinear.threshold.TVARResult` --
          the two-system limit this estimator relaxes.
        * :class:`~cultivars.multivariate.nonlinear.smooth_transition.STVARResult`
          -- the two-anchor blend between the threshold and this model.

    References:
        Cai, Z., Fan, J., & Yao, Q. (2000). Functional-coefficient
        regression models for nonlinear time series. *Journal of the
        American Statistical Association*, 95(451), 941-956.

        Chen, R., & Tsay, R. S. (1993). Functional-coefficient
        autoregressive models. *Journal of the American Statistical
        Association*, 88(421), 298-308.

    Example:
        A bivariate system whose first equation has the own-lag coefficient
        :math:`0.8 \tanh(y_{1,t-1})` and whose second is a linear AR(1).
        The estimated curve tracks the tanh and its band covers it, and the
        second equation's curve is flat:

        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((250, 2))
        >>> for t in range(1, 250):
        ...     a = 0.8 * np.tanh(y[t - 1, 0])
        ...     y[t, 0] = a * y[t - 1, 0] + 0.5 * rng.standard_normal()
        ...     y[t, 1] = 0.4 * y[t - 1, 1] + 0.5 * rng.standard_normal()
        >>> model = FunctionalCoefficientVAR(y, order=1, transition_variable="y1")
        >>> res = model.fit(bandwidth=1.2, n_grid=21)
        >>> res.names, res.state_name, res.delay, res.nobs, res.bandwidth_searched
        (('y1', 'y2'), 'y1', 1, 249, False)
        >>> res.grid.shape, res.curves.shape, res.curve_se.shape, res.state_values.shape
        ((21,), (21, 3, 2), (21, 3, 2), (249,))
        >>> low, point, high = res.coefficient_curve("y1", "y1.L1").T
        >>> truth = 0.8 * np.tanh(res.grid)
        >>> within_two = np.abs(point - truth) < 2.0 * (high - point)
        >>> bool(point[0] < 0.0 < point[-1]), bool(within_two.mean() > 0.8)
        (True, True)
        >>> flat = res.coefficient_curve("y2", "y2.L1")[:, 1]
        >>> bool(flat.max() - flat.min() < 0.5)
        True
        >>> bool(2.0 < res.effective_params < 8.0)
        True
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    """The observed panel, shape ``(nobs_total, k)``. Kept out of the repr."""
    names: tuple[str, ...]
    """Variable labels, in column order."""
    order: int
    """The autoregressive order ``p``, at least one."""
    trend: str
    """The deterministic specification: ``"n"``, ``"c"`` or ``"ct"``."""
    delay: int
    """Delay ``d`` of the state: ``z_t = s_{t-d}``."""
    state_name: str
    """The state's label: a name from ``names``, or ``"external"``."""
    state_values: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs,)`` delayed state ``z_t``, aligned with ``resid``. Kept out of the repr."""
    transition_series: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs_total,)`` raw state series, aligned with ``endog``. Kept out of the repr."""
    grid: npt.NDArray[np.float64] = field(repr=False)
    """``(G,)`` evaluation points over the trimmed state range. Kept out of the repr."""
    curves: npt.NDArray[np.float64] = field(repr=False)
    """``(G, w, k)`` level coefficients; ``[g, :, i]`` is equation ``i`` at ``grid[g]``."""
    curve_se: npt.NDArray[np.float64] = field(repr=False)
    """``(G, w, k)`` pointwise standard errors, conditional on the bandwidth."""
    bandwidth: float
    """Kernel bandwidth in the state's units."""
    bandwidth_searched: bool
    """``True`` when the bandwidth came from leave-one-out cross-validation."""
    effective_params: float
    """Trace of the smoother matrix over the scored observations."""
    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    """``(k, k)`` innovation covariance, divided by ``nobs - effective_params``."""
    resid: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, k)`` residuals at the observed states. Kept out of the repr."""
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    """``(nobs, k)`` one-step conditional means at the observed states. Kept out of the repr."""
    nobs: int
    """Effective sample size after the ``max(order, delay)`` presample rows."""

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorFunctionalFit,
        model: _FunctionalCoefficientVectorAutoRegressionModel[FunctionalCoefficientVARResult],
    ) -> FunctionalCoefficientVARResult:
        """Assemble the public result from a raw fit and its specification.

        Args:
            fit: The packed kernel fit.
            model: The validated specification it was estimated on.

        Returns:
            The frozen :class:`FunctionalCoefficientVARResult`.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((150, 2))
            >>> model = FunctionalCoefficientVAR(y, order=1, transition_variable="y2")
            >>> fit = model._fit_functional(bandwidth=1.0, n_grid=11, trim=0.05)
            >>> res = FunctionalCoefficientVARResult._from_fit(fit, model)
            >>> res.state_name, res.grid.shape, res.bandwidth
            ('y2', (11,), 1.0)
        """
        return cls(
            endog=model.endog,
            names=model.names,
            order=model.order,
            trend=model.trend,
            delay=fit.delay,
            state_name=model.transition_name,
            state_values=fit.state_values,
            transition_series=model.transition_series,
            grid=fit.grid,
            curves=fit.curves,
            curve_se=fit.curve_se,
            bandwidth=fit.bandwidth,
            bandwidth_searched=fit.bandwidth_searched,
            effective_params=fit.effective_params,
            sigma_u=fit.sigma_u,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            nobs=fit.nobs,
        )

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((150, 3))
            >>> model = FunctionalCoefficientVAR(y, order=1, transition_variable="y1")
            >>> model.fit(bandwidth=1.0, n_grid=11).k_endog
            3
        """
        return len(self.names)

    @property
    def _n_deterministic(self) -> int:
        """Deterministic columns per equation.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((150, 2))
            >>> model = FunctionalCoefficientVAR(y, order=1, transition_variable="y1", trend="ct")
            >>> model.fit(bandwidth=1.0, n_grid=11)._n_deterministic
            2
        """
        return {"n": 0, "c": 1, "ct": 2}[self.trend]

    @property
    def state_range(self) -> tuple[float, float]:
        """The state interval the curves are estimated on.

        The first and last grid points, i.e. the ``trim`` and ``1 - trim``
        quantiles of the observed state.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((150, 2))
            >>> model = FunctionalCoefficientVAR(y, order=1, transition_variable="y1")
            >>> res = model.fit(bandwidth=1.0, n_grid=11, trim=0.1)
            >>> low, high = res.state_range
            >>> bool(np.isclose(low, np.quantile(res.state_values, 0.1)))
            True
            >>> bool(np.isclose(high, np.quantile(res.state_values, 0.9)))
            True
        """
        return float(self.grid[0]), float(self.grid[-1])

    def _regressor_labels(self) -> tuple[str, ...]:
        """Per-equation regressor labels, in design order.

        Returns:
            Deterministic labels first (``"const"``, ``"trend"``), then
            ``"{name}.L{lag}"`` lag by lag, variable by variable within a
            lag -- the column order of ``curves``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((150, 2))
            >>> model = FunctionalCoefficientVAR(y, order=2, transition_variable="y1")
            >>> model.fit(bandwidth=1.0, n_grid=11)._regressor_labels()
            ('const', 'y1.L1', 'y2.L1', 'y1.L2', 'y2.L2')
        """
        det = ("const", "trend")[: self._n_deterministic]
        lags = tuple(f"{source}.L{lag + 1}" for lag in range(self.order) for source in self.names)
        return (*det, *lags)

    def coefficient_curve(self, equation: str, regressor: str) -> npt.NDArray[np.float64]:
        """One coefficient's estimated curve over the state grid.

        Args:
            equation: An endogenous variable.
            regressor: A per-equation regressor label -- ``"const"``,
                ``"trend"``, or ``"{name}.L{lag}"``.

        Returns:
            A ``(G, 3)`` array with columns ``(low, point, high)``, the band
            one pointwise standard error each side -- conditional on the
            bandwidth, so read it as precision at this smoothing, not as
            inference over smoothings.

        Raises:
            SpecificationError: If the equation or regressor is unknown.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((250, 2))
            >>> for t in range(1, 250):
            ...     a = 0.8 * np.tanh(y[t - 1, 0])
            ...     y[t, 0] = a * y[t - 1, 0] + 0.5 * rng.standard_normal()
            ...     y[t, 1] = 0.4 * y[t - 1, 1] + 0.5 * rng.standard_normal()
            >>> res = FunctionalCoefficientVAR(y, order=1, transition_variable="y1").fit(
            ...     bandwidth=1.2, n_grid=21
            ... )
            >>> curve = res.coefficient_curve("y1", "y1.L1")
            >>> curve.shape, bool(np.all(curve[:, 0] < curve[:, 1]))
            ((21, 3), True)
            >>> bool(np.all(curve[:, 1] < curve[:, 2]))
            True
            >>> bool(np.allclose(curve[:, 1], res.curves[:, 1, 0]))
            True
            >>> res.coefficient_curve("y1", "y1.L2")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: unknown regressor 'y1.L2'; expected one of ...
        """
        if equation not in self.names:
            raise SpecificationError(
                f"unknown variable {equation!r}; expected one of {self.names}."
            )
        labels = self._regressor_labels()
        if regressor not in labels:
            raise SpecificationError(f"unknown regressor {regressor!r}; expected one of {labels}.")
        column, i = labels.index(regressor), self.names.index(equation)
        point = self.curves[:, column, i]
        spread = self.curve_se[:, column, i]
        return np.column_stack([point - spread, point, point + spread])

    def _matrix_at(self, u: float) -> npt.NDArray[np.float64]:
        """The full ``(w, k)`` coefficient matrix at one state value.

        Linear interpolation between the two bracketing grid points --
        exact at grid points, and refusing to extrapolate: a kernel
        estimate outside the data's state range is a guess about a region
        the estimator never saw.

        Args:
            u: A state value inside :attr:`state_range`.

        Returns:
            The ``(w, k)`` matrix in design-row order.

        Raises:
            SpecificationError: If ``u`` lies outside the estimated range.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((150, 2))
            >>> res = FunctionalCoefficientVAR(y, order=1, transition_variable="y1").fit(
            ...     bandwidth=1.0, n_grid=11
            ... )
            >>> bool(np.allclose(res._matrix_at(float(res.grid[4])), res.curves[4]))
            True
            >>> midpoint = 0.5 * (res.grid[4] + res.grid[5])
            >>> blend = 0.5 * (res.curves[4] + res.curves[5])
            >>> bool(np.allclose(res._matrix_at(float(midpoint)), blend))
            True
        """
        low, high = self.state_range
        if not low <= u <= high:
            raise SpecificationError(
                f"state value {u} lies outside the estimated range "
                f"[{low:.6g}, {high:.6g}]; kernel estimates do not "
                "extrapolate."
            )
        position = int(np.searchsorted(self.grid, u, side="right"))
        if position >= len(self.grid):
            return np.asarray(self.curves[-1], dtype=np.float64)
        if position == 0:
            return np.asarray(self.curves[0], dtype=np.float64)
        left, right = self.grid[position - 1], self.grid[position]
        weight = float((u - left) / (right - left))
        blended = (1.0 - weight) * self.curves[position - 1] + weight * self.curves[position]
        return np.asarray(blended, dtype=np.float64)

    def coefficients_at(self, u: float) -> npt.NDArray[np.float64]:
        """The lag stack ``A_1(u), ..., A_p(u)`` prevailing at one state value.

        Args:
            u: A state value inside :attr:`state_range`.

        Returns:
            A ``(p, k, k)`` stack; ``[i - 1, r, c]`` is the response of
            variable ``r`` to lag ``i`` of variable ``c``.

        Raises:
            SpecificationError: If ``u`` lies outside the estimated range.

        Example:
            On the tanh system the own-lag coefficient of ``y1`` is negative
            at the low end of the state and positive at the high end:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((250, 2))
            >>> for t in range(1, 250):
            ...     a = 0.8 * np.tanh(y[t - 1, 0])
            ...     y[t, 0] = a * y[t - 1, 0] + 0.5 * rng.standard_normal()
            ...     y[t, 1] = 0.4 * y[t - 1, 1] + 0.5 * rng.standard_normal()
            >>> res = FunctionalCoefficientVAR(y, order=1, transition_variable="y1").fit(
            ...     bandwidth=1.2, n_grid=21
            ... )
            >>> low, high = res.state_range
            >>> res.coefficients_at(low).shape
            (1, 2, 2)
            >>> bool(res.coefficients_at(low)[0, 0, 0] < 0.0 < res.coefficients_at(high)[0, 0, 0])
            True
            >>> res.coefficients_at(10.0)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: state value 10.0 lies outside the estimated ...
        """
        matrix = self._matrix_at(u)
        k, offset = self.k_endog, self._n_deterministic
        if not self.order:
            return np.zeros((0, k, k))
        return np.stack(
            [matrix[offset + lag * k : offset + (lag + 1) * k].T for lag in range(self.order)]
        )

    def stability_curve(self) -> npt.NDArray[np.float64]:
        """Largest companion-root modulus of the local system along the grid.

        A functional-coefficient process can be locally explosive over part
        of the state space and globally stable -- excursions into the
        explosive region get pulled back -- so a modulus above one here is
        a prompt to look, not a verdict.

        Returns:
            An array of length ``G``, aligned with :attr:`grid`.

        Example:
            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((250, 2))
            >>> for t in range(1, 250):
            ...     a = 0.8 * np.tanh(y[t - 1, 0])
            ...     y[t, 0] = a * y[t - 1, 0] + 0.5 * rng.standard_normal()
            ...     y[t, 1] = 0.4 * y[t - 1, 1] + 0.5 * rng.standard_normal()
            >>> res = FunctionalCoefficientVAR(y, order=1, transition_variable="y1").fit(
            ...     bandwidth=1.2, n_grid=21
            ... )
            >>> roots = res.stability_curve()
            >>> roots.shape, bool(np.all(roots < 1.0)), bool(roots[-1] > roots[10])
            ((21,), True, True)
        """
        out = np.empty(len(self.grid))
        for g, u in enumerate(self.grid):
            eigs = np.linalg.eigvals(companion_matrix(self.coefficients_at(float(u))))
            out[g] = float(np.abs(eigs).max(initial=0.0))
        return out

    def irf(
        self,
        horizon: int = 20,
        *,
        u: float,
        orthogonalized: bool = True,
        cumulative: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Impulse responses at one state value, held frozen.

        The family's usual linearization, in its continuous form: the
        coefficients are frozen at ``A(u)``, so the response is exact only
        while the shock does not move the state -- read it as the
        transmission prevailing at ``u``. Orthogonalization uses the single
        estimated covariance under the model's constant-covariance
        assumption; the ordering of ``names`` is then an identifying
        assumption, as for a linear VAR.

        Args:
            horizon: Largest lead to return.
            u: A state value inside :attr:`state_range`.
            orthogonalized: Rotate by the Cholesky factor of ``sigma_u``.
            cumulative: Return running sums.

        Returns:
            An array of shape ``(horizon + 1, k, k)``; entry ``[h, i, j]``
            is the response of variable ``i`` at lead ``h`` to shock ``j``,
            as of state ``u``.

        Raises:
            SpecificationError: If ``horizon`` is negative or ``u`` is
                outside the estimated range.

        Example:
            The own response of ``y1`` dies faster where its local
            coefficient is near zero than where it is large:

            >>> import numpy as np
            >>> rng = np.random.default_rng(0)
            >>> y = np.zeros((250, 2))
            >>> for t in range(1, 250):
            ...     a = 0.8 * np.tanh(y[t - 1, 0])
            ...     y[t, 0] = a * y[t - 1, 0] + 0.5 * rng.standard_normal()
            ...     y[t, 1] = 0.4 * y[t - 1, 1] + 0.5 * rng.standard_normal()
            >>> res = FunctionalCoefficientVAR(y, order=1, transition_variable="y1").fit(
            ...     bandwidth=1.2, n_grid=21
            ... )
            >>> calm = res.irf(4, u=0.0, orthogonalized=False)
            >>> hot = res.irf(4, u=res.state_range[1], orthogonalized=False)
            >>> calm.shape, bool(np.allclose(calm[0], np.eye(2)))
            ((5, 2, 2), True)
            >>> bool(abs(calm[2, 0, 0]) < abs(hot[2, 0, 0]))
            True
            >>> bool(np.allclose(res.irf(4, u=0.0)[0], np.linalg.cholesky(res.sigma_u)))
            True
            >>> res.irf(-1, u=0.0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: horizon must be non-negative; got -1.
        """
        if horizon < 0:
            raise SpecificationError(f"horizon must be non-negative; got {horizon}.")
        stack = self.coefficients_at(u)
        k, p = self.k_endog, self.order
        psi = np.empty((horizon + 1, k, k))
        if p == 0:
            psi[:] = 0.0
            psi[0] = np.eye(k)
        else:
            selector = np.zeros((k, k * p))
            selector[:, :k] = np.eye(k)
            power = np.eye(k * p)
            companion = companion_matrix(stack)
            for h in range(horizon + 1):
                psi[h] = selector @ power @ selector.T
                power = power @ companion
        out = psi @ np.linalg.cholesky(self.sigma_u) if orthogonalized else psi
        return np.cumsum(out, axis=0) if cumulative else out

    def forecast(self) -> npt.NDArray[np.float64]:
        """The one-step-ahead conditional mean.

        One step because that is what the observed state licenses: with
        delay ``d >= 1`` the state that governs period ``T + 1`` is already
        observed at ``T``, so the forecast is an evaluation, not a
        simulation. Further steps would need the state's own future, which
        is the nonlinear simulation problem this result does not fake.

        Returns:
            A ``(k,)`` array.

        Raises:
            SpecificationError: If the next period's state lies outside the
                estimated range.

        Example:
            The forecast is the last design row times the coefficient
            matrix at the last observed state:

            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((150, 2))
            >>> res = FunctionalCoefficientVAR(y, order=1, transition_variable="y1").fit(
            ...     bandwidth=1.0, n_grid=11
            ... )
            >>> point = res.forecast()
            >>> point.shape
            (2,)
            >>> row = np.concatenate([[1.0], y[-1]])
            >>> bool(np.allclose(point, row @ res._matrix_at(float(y[-1, 0]))))
            True
        """
        n = self.endog.shape[0]
        next_state = float(self.transition_series[n - self.delay])
        matrix = self._matrix_at(next_state)
        det = {"n": [], "c": [1.0], "ct": [1.0, float(n + 1)]}[self.trend]
        regressors = np.concatenate(
            [np.asarray(det, dtype=np.float64)]
            + [self.endog[n - 1 - lag] for lag in range(self.order)]
        )
        return np.asarray(regressors @ matrix, dtype=np.float64)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary.

        Returns:
            One row per equation with its own first-lag coefficient at the
            low and high ends of the state range; the metadata names the
            state, its delay and range, the kernel and the grid; the notes
            state the bandwidth's provenance, the effective parameter count
            and the absent likelihood, the largest local companion root,
            the conditional nature of the bands, the no-extrapolation rule
            and the frozen-coefficient reading of ``irf``.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((150, 2))
            >>> res = FunctionalCoefficientVAR(y, order=1, transition_variable="y1").fit(
            ...     bandwidth=1.0, n_grid=11
            ... )
            >>> table = res._summary_table()
            >>> table.title, len(table.rows), len(table.notes)
            ('Functional-coefficient VAR(1) Results', 2, 6)
            >>> table.columns
            ('equation', 'own L1 at state low', 'own L1 at state high')
            >>> table.metadata[1], table.metadata[3]
            (('State', 'y1 (delay 1)'), ('Kernel', 'Epanechnikov, local linear'))
            >>> table.notes[0]
            'Bandwidth 1 stated by the caller.'
        """
        labels = self._regressor_labels()
        rows: list[tuple[str, ...]] = []
        for i, name in enumerate(self.names):
            own = f"{name}.L1"
            if self.order and own in labels:
                column = labels.index(own)
                rows.append(
                    (
                        name,
                        f"{self.curves[0, column, i]:.4f}",
                        f"{self.curves[-1, column, i]:.4f}",
                    )
                )
            else:
                rows.append((name, "-", "-"))
        roots = self.stability_curve()
        bandwidth_note = (
            f"Bandwidth {self.bandwidth:.4g} chosen by leave-one-out cross-validation."
            if self.bandwidth_searched
            else f"Bandwidth {self.bandwidth:.4g} stated by the caller."
        )
        notes = [
            bandwidth_note,
            f"Effective parameters (trace of the smoother): "
            f"{self.effective_params:.1f}. No likelihood, parameter count, "
            "or information criteria exist for a kernel fit, so none are "
            "reported.",
            f"Max companion-root modulus along the grid: "
            f"{float(roots.max()):.4f} (local explosiveness with global "
            "stability is possible; stability_curve() shows where).",
            "Curve bands are pointwise, one standard error, and conditional "
            "on the bandwidth; they do not account for its selection.",
            "Curves are reported on the trimmed state range only and "
            "refuse to extrapolate beyond it; observations whose state "
            "falls outside that range take the boundary system for their "
            "fitted value.",
            "irf(u=...) freezes the coefficients at one state value; read "
            "it as the transmission prevailing there, exact only while the "
            "shock does not move the state.",
        ]
        low, high = self.state_range
        return SummaryTable(
            title=f"Functional-coefficient VAR({self.order}) Results",
            metadata=(
                ("Model", f"FC-VAR({self.order})"),
                ("State", f"{self.state_name} (delay {self.delay})"),
                ("State range", f"[{low:.4g}, {high:.4g}]"),
                ("Kernel", "Epanechnikov, local linear"),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.nobs}"),
                ("Grid points", f"{len(self.grid)}"),
                ("Trend", self.trend),
            ),
            columns=("equation", "own L1 at state low", "own L1 at state high"),
            rows=tuple(rows),
            notes=tuple(notes),
        )


class FunctionalCoefficientVAR(
    _FunctionalCoefficientVectorAutoRegressionModel[FunctionalCoefficientVARResult]
):
    r"""Functional-coefficient VAR, after Cai, Fan and Yao: local-linear estimation.

    Every coefficient is an unknown smooth function of an observable state
    -- a named variable's own lag, or an external series -- estimated with
    no assumption on its shape beyond smoothness. At each evaluation point
    one Epanechnikov-weighted least squares on the augmented design
    :math:`[x_t,\ x_t (z_t - u)]` delivers the local level and slope of
    every coefficient, all equations sharing the weights. The bandwidth is
    the one tuning constant; left unstated, it is chosen by exact
    leave-one-out cross-validation over a geometric grid around the
    Silverman rule :math:`2.34\,\hat\sigma_z\, n^{-1/5}`. There is no delay
    search, because the bandwidth is this model's tuning axis and searching
    both would let the smoother trade one against the other invisibly.

    Attributes:
        _endog: The observed panel, from the base model.
        _order: The autoregressive order ``p``, from the base model.
        _trend: The deterministic specification, from the base model.
        _names: The variable labels, from the base model.
        _delays: The single delay, as a one-element tuple, from the
            observed-regime base.
        _threshold_name: The state's label, from the observed-regime base.
        _transition_series: The aligned state series, from the
            observed-regime base.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order, at least one.
        transition_variable: A variable name from ``names`` (the state is
            that variable's own lag) or an aligned external series.
        delay: Delay of the state variable. Defaults to one.
        trend: Deterministic terms.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Raises:
        SpecificationError: If the specification is malformed.
        DimensionError: If the effective sample is shorter than six times
            the per-window coefficient count, which local-linear estimation
            needs.

    See Also:
        * :class:`FunctionalCoefficientVARResult` -- the record ``fit``
          returns.
        * :class:`~cultivars.multivariate.nonlinear.threshold.TVAR` -- the
          two-system special case.
        * :class:`~cultivars.multivariate.nonlinear.smooth_transition.STVAR`
          -- the parametric smooth alternative.

    References:
        Cai, Z., Fan, J., & Yao, Q. (2000). Functional-coefficient
        regression models for nonlinear time series. *Journal of the
        American Statistical Association*, 95(451), 941-956.

        Chen, R., & Tsay, R. S. (1993). Functional-coefficient
        autoregressive models. *Journal of the American Statistical
        Association*, 88(421), 298-308.

        Hastie, T., & Tibshirani, R. (1993). Varying-coefficient models.
        *Journal of the Royal Statistical Society, Series B*, 55(4),
        757-796.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((250, 2))
        >>> for t in range(1, 250):
        ...     a = 0.8 * np.tanh(y[t - 1, 0])
        ...     y[t, 0] = a * y[t - 1, 0] + 0.5 * rng.standard_normal()
        ...     y[t, 1] = 0.4 * y[t - 1, 1] + 0.5 * rng.standard_normal()
        >>> model = FunctionalCoefficientVAR(y, order=1, transition_variable="y1")
        >>> res = model.fit(bandwidth=1.2, n_grid=21)
        >>> res.coefficient_curve("y1", "y1.L1").shape
        (21, 3)
        >>> res.irf(4, u=0.0).shape
        (5, 2, 2)
        >>> searched = model.fit(n_grid=21)
        >>> searched.bandwidth_searched, bool(0.5 < searched.bandwidth < 2.5)
        (True, True)
        >>> cycle = np.sin(np.arange(250) / 20)
        >>> external = FunctionalCoefficientVAR(y, order=1, transition_variable=cycle)
        >>> external.fit(bandwidth=0.5, n_grid=11).state_name
        'external'
    """

    __slots__ = ()

    def fit(
        self,
        *,
        bandwidth: float | None = None,
        n_grid: int = 51,
        trim: float = 0.05,
    ) -> FunctionalCoefficientVARResult:
        """Estimate the coefficient curves by local-linear kernel regression.

        The curves are evaluated on ``n_grid`` equally spaced points
        between the ``trim`` and ``1 - trim`` quantiles of the state; the
        cross-validation criterion, when it runs, is scored on the
        observations inside that range only.

        Args:
            bandwidth: Kernel bandwidth in the state variable's units, or
                ``None`` to select by leave-one-out cross-validation on a
                grid around the Silverman-scaled rule of thumb.
            n_grid: Evaluation points for the reported curves.
            trim: Quantile trimmed from each end of the state range before
                the curve grid is laid down.

        Returns:
            The fitted :class:`FunctionalCoefficientVARResult`.

        Raises:
            SpecificationError: If a stated bandwidth is not positive,
                ``n_grid`` is below 2, or ``trim`` is outside ``[0, 0.5)``.
            NumericalError: If no bandwidth identifies every local system.

        Example:
            >>> import numpy as np
            >>> y = np.random.default_rng(0).standard_normal((150, 2))
            >>> model = FunctionalCoefficientVAR(y, order=1, transition_variable="y1")
            >>> res = model.fit(bandwidth=1.0, n_grid=11, trim=0.1)
            >>> res.grid.shape, res.bandwidth, res.bandwidth_searched
            ((11,), 1.0, False)
            >>> model.fit(bandwidth=-1.0)
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: bandwidth must be positive; got -1.0.
            >>> model.fit(bandwidth=0.05)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.NumericalError: the local system at state value ... is ...
        """
        fit = self._fit_functional(bandwidth=bandwidth, n_grid=n_grid, trim=trim)
        return FunctionalCoefficientVARResult._from_fit(fit, self)
