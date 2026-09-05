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

"""The functional-coefficient VAR: every state value gets its own system.

This is the nonparametric end of the observed-transition family. A threshold
VAR assigns each value of an observable state one of two systems; a
smooth-transition VAR blends the same two anchors; here the coefficient
matrices are unknown smooth *functions* of the state (Chen & Tsay 1993; Cai,
Fan & Yao 2000), estimated by local-linear kernel regression with nothing
assumed about their shape but smoothness. The payoff is diagnostic as much as
predictive: the estimated curves show whether transmission actually varies in
the state, and whether its variation looks like a threshold, a smooth
transition, or something neither parametric family can express -- which is
why this model is the natural specification check on both.

The price of assuming nothing is paid in two currencies the result reports
honestly. Statistical: there is no likelihood and no parameter count -- the
estimator's complexity is the trace of its smoother matrix, reported as
``effective_params``, and no information criteria are derivable from a kernel
fit. Practical: everything is local, so the curves are trustworthy only
where the state has mass -- they are reported on a trimmed grid, refuse to
extrapolate beyond it, and carry pointwise bands that are conditional on the
bandwidth rather than accounting for its selection.

One identification fact is handled by construction rather than by a warning.
When the state is one of the design's own lag columns -- the usual
self-exciting case -- the intercept function's local slope is *exactly*
collinear with the design, so the intercept is fitted local-constant while
every other coefficient stays local-linear; the engine's
``_state_in_design`` note states the algebra.

References:
    Chen, R., & Tsay, R. S. (1993). Functional-coefficient autoregressive
        models. *Journal of the American Statistical Association*, 88(421),
        298-308.
    Cai, Z., Fan, J., & Yao, Q. (2000). Functional-coefficient regression
        models for nonlinear time series. *Journal of the American
        Statistical Association*, 95(451), 941-956.
    Hastie, T., & Tibshirani, R. (1993). Varying-coefficient models.
        *Journal of the Royal Statistical Society, Series B*, 55(4),
        757-796.
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


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class FunctionalCoefficientVARResult(_SummaryMixin):
    """A fitted functional-coefficient VAR: coefficient curves over the state.

    Deliberately absent: ``llf``, ``n_params``, information criteria. A
    kernel estimator maximizes no likelihood; its complexity is
    :attr:`effective_params`, the trace of the smoother matrix.

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
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    names: tuple[str, ...]
    order: int
    trend: str
    delay: int
    state_name: str
    state_values: npt.NDArray[np.float64] = field(repr=False)
    transition_series: npt.NDArray[np.float64] = field(repr=False)
    grid: npt.NDArray[np.float64] = field(repr=False)
    curves: npt.NDArray[np.float64] = field(repr=False)
    curve_se: npt.NDArray[np.float64] = field(repr=False)
    bandwidth: float
    bandwidth_searched: bool
    effective_params: float
    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    resid: npt.NDArray[np.float64] = field(repr=False)
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    nobs: int

    @classmethod
    def _from_fit(
        cls,
        fit: _VectorFunctionalFit,
        model: _FunctionalCoefficientVectorAutoRegressionModel[FunctionalCoefficientVARResult],
    ) -> FunctionalCoefficientVARResult:
        """Assemble the public result from a raw fit and its specification."""
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
        """Number of endogenous variables."""
        return len(self.names)

    @property
    def _n_deterministic(self) -> int:
        """Deterministic columns per equation."""
        return {"n": 0, "c": 1, "ct": 2}[self.trend]

    @property
    def state_range(self) -> tuple[float, float]:
        """The state interval the curves are estimated on."""
        return float(self.grid[0]), float(self.grid[-1])

    def _regressor_labels(self) -> tuple[str, ...]:
        """Per-equation regressor labels, in design order."""
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
            A ``(p, k, k)`` stack.

        Raises:
            SpecificationError: If ``u`` lies outside the estimated range.
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
        """Build the structured summary."""
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
    """Functional-coefficient VAR, Cai-Fan-Yao local-linear estimation.

    Every coefficient is an unknown smooth function of an observable state
    -- a named variable's own lag, or an external series -- estimated with
    no assumption on its shape. The bandwidth is the one tuning constant;
    left unstated, it is chosen by exact leave-one-out cross-validation.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order, at least one.
        transition_variable: A variable name from ``names`` (the state is
            that variable's own lag) or an aligned external series.
        delay: Delay of the state variable. Defaults to one.
        trend: Deterministic terms.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Example:
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
            SpecificationError: If a stated bandwidth is not positive.
            NumericalError: If no bandwidth identifies every local system.
        """
        fit = self._fit_functional(bandwidth=bandwidth, n_grid=n_grid, trim=trim)
        return FunctionalCoefficientVARResult._from_fit(fit, self)
