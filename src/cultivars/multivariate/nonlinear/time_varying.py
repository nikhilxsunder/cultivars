# filepath: /src/cultivars/multivariate/nonlinear/time_varying.py
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

"""Time-varying-parameter VARs: drift instead of switching, posteriors instead of dates.

Where the regime families let parameters jump between a few states, here
they drift: the stacked coefficient vector follows a random walk, and under
stochastic volatility so do the log variances, so every date has its own VAR
and its own covariance. This is the Cogley-Sargent / Primiceri (2005)
workhorse for questions like whether monetary policy transmission itself has
changed -- questions a constant-parameter model cannot even pose.

Estimation is Bayesian by necessity, not preference: an unrestricted
coefficient path has more parameters than observations, and what disciplines
it is the prior on how fast parameters may drift. Following Primiceri, a
training sample is split off the front, its OLS estimates calibrate the
coefficient prior and the drift scale ``k_drift`` (his ``k_Q``), and it is
then discarded rather than used twice. The sampler is Gibbs on the package's
own state-space substrate: every coefficient-path draw is one Durbin-Koopman
simulation-smoother call, the covariance blocks are conjugate
inverse-Wisharts, and the stochastic-volatility block is Kim-Shephard-Chib.
Under SV the innovation covariance is parameterized as ``Sigma_t = A^{-1}
H_t A^{-T}`` with a constant unit-lower-triangular ``A`` and random-walk log
variances -- which is why the SV model's structural interpretation is
settled at estimation time (see :class:`TimeVaryingSVAR`).

Being posteriors, the results report uncertainty rather than test
statistics: coefficient and volatility paths come with posterior bands, and
there is deliberately no likelihood, information criterion, or parameter
count on these results -- a Gibbs posterior has none to report, and numbers
shaped like them would be wrong.

References:
    Primiceri, G. E. (2005). Time varying structural vector autoregressions
        and monetary policy. *Review of Economic Studies*, 72(3), 821-852.
    Del Negro, M., & Primiceri, G. E. (2015). Time varying structural vector
        autoregressions and monetary policy: A corrigendum. *Review of
        Economic Studies*, 82(4), 1342-1345.
    Cogley, T., & Sargent, T. J. (2005). Drifts and volatilities: Monetary
        policies and outcomes in the post WWII US. *Review of Economic
        Dynamics*, 8(2), 262-302.
    Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
        Likelihood inference and comparison with ARCH models. *Review of
        Economic Studies*, 65(3), 361-393.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable, companion_matrix
from ..._internals import (
    _SummaryMixin,
    _TimeVaryingVectorAutoRegressionModel,
)
from ...exceptions import SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class TVPVARResult(_SummaryMixin):
    """A fitted time-varying-parameter VAR, homoskedastic innovations.

    Every coefficient is a path: the posterior over ``beta_t`` is summarized
    by its mean and a 68% band, with the kept draws retained so downstream
    consumers -- the structural wrapper, custom summaries -- can propagate
    posterior uncertainty rather than plugging in a point.

    Deliberately absent: ``llf``, ``n_params``, information criteria. A
    Gibbs posterior has none of them, and this result will not counterfeit
    the comparison surface of a likelihood fit.

    Attributes:
        endog: The observed panel.
        names: Variable labels, in column order.
        order: Autoregressive order.
        trend: Deterministic specification.
        training: Rows consumed by the training prior and then discarded.
        nobs: Estimation rows after the training split.
        beta_mean: ``(nobs, D)`` posterior mean coefficient path, stacked
            equation-major.
        beta_low: ``(nobs, D)`` posterior 16th percentile path.
        beta_high: ``(nobs, D)`` posterior 84th percentile path.
        beta_draws: ``(S, nobs, D)`` kept coefficient-path draws.
        state_cov: ``(D, D)`` posterior mean drift covariance ``Q``.
        sigma_u: ``(k, k)`` posterior mean innovation covariance.
        resid: Residuals at the posterior mean path.
        fittedvalues: One-step means at the posterior mean path.
        n_draws: Total sampler iterations.
        n_burn: Burn-in discarded.
        thin: Post-burn thinning.
    """

    endog: npt.NDArray[np.float64] = field(repr=False)
    names: tuple[str, ...]
    order: int
    trend: str
    training: int
    nobs: int
    beta_mean: npt.NDArray[np.float64] = field(repr=False)
    beta_low: npt.NDArray[np.float64] = field(repr=False)
    beta_high: npt.NDArray[np.float64] = field(repr=False)
    beta_draws: npt.NDArray[np.float64] = field(repr=False)
    state_cov: npt.NDArray[np.float64] = field(repr=False)
    sigma_u: npt.NDArray[np.float64] = field(repr=False)
    resid: npt.NDArray[np.float64] = field(repr=False)
    fittedvalues: npt.NDArray[np.float64] = field(repr=False)
    n_draws: int
    n_burn: int
    thin: int

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables."""
        return len(self.names)

    @property
    def n_kept(self) -> int:
        """Posterior draws retained after burn-in and thinning."""
        return int(self.beta_draws.shape[0])

    @property
    def _n_deterministic(self) -> int:
        """Deterministic columns per equation."""
        return {"n": 0, "c": 1, "ct": 2}[self.trend]

    @property
    def _width(self) -> int:
        """Regressors per equation."""
        return self._n_deterministic + self.k_endog * self.order

    def _regressor_labels(self) -> tuple[str, ...]:
        """Per-equation regressor labels, in design order."""
        det = ("const", "trend")[: self._n_deterministic]
        lags = tuple(f"{source}.L{lag + 1}" for lag in range(self.order) for source in self.names)
        return (*det, *lags)

    def coefficient_path(self, equation: str, regressor: str) -> npt.NDArray[np.float64]:
        """One coefficient's posterior path: 16th percentile, mean, 84th.

        Args:
            equation: An endogenous variable.
            regressor: A per-equation regressor label -- ``"const"``,
                ``"trend"``, or ``"{name}.L{lag}"``.

        Returns:
            An ``(nobs, 3)`` array with columns ``(low, mean, high)``.

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
        column = self.names.index(equation) * self._width + labels.index(regressor)
        return np.column_stack(
            [self.beta_low[:, column], self.beta_mean[:, column], self.beta_high[:, column]]
        )

    def coefficients_at(self, t: int) -> npt.NDArray[np.float64]:
        """The posterior mean lag stack ``A_1..A_p`` prevailing at date ``t``.

        Args:
            t: Row of the estimation sample, ``0..nobs-1``.

        Returns:
            A ``(p, k, k)`` stack.

        Raises:
            SpecificationError: If ``t`` is out of range.
        """
        if not 0 <= t < self.nobs:
            raise SpecificationError(f"t must be in 0..{self.nobs - 1}; got {t}.")
        k, w, d = self.k_endog, self._width, self._n_deterministic
        rows = self.beta_mean[t].reshape(k, w)
        return np.stack([rows[:, d + lag * k : d + (lag + 1) * k] for lag in range(self.order)])

    def stability_path(self) -> npt.NDArray[np.float64]:
        """Largest companion-root modulus of the posterior mean VAR at each date.

        The time-varying analogue of a stability check, and the honest form
        of it: a drifting model can wander through locally explosive
        parameter regions and back, and this path is where that shows.

        Returns:
            An array of length ``nobs``.
        """
        out = np.empty(self.nobs)
        for t in range(self.nobs):
            eigs = np.linalg.eigvals(companion_matrix(self.coefficients_at(t)))
            out[t] = float(np.abs(eigs).max(initial=0.0))
        return out

    def _drift_note(self) -> str:
        """One line on how much the coefficients actually moved."""
        moved = np.abs(self.beta_mean[-1] - self.beta_mean[0])
        spread = self.beta_high - self.beta_low
        typical = float(np.median(spread))
        return (
            f"Posterior mean drift over the sample: max |beta_T - beta_1| = "
            f"{float(moved.max()):.4f} against a median 68% band width of "
            f"{typical:.4f}; drift smaller than the band is not evidence of "
            "time variation."
        )

    def _family(self) -> str:
        """Specification stem for display."""
        return "TVP-VAR"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        roots = self.stability_path()
        rows = tuple(
            (
                name,
                f"{self.coefficient_path(name, f'{name}.L1')[0, 1]:.4f}" if self.order else "-",
                f"{self.coefficient_path(name, f'{name}.L1')[-1, 1]:.4f}" if self.order else "-",
            )
            for name in self.names
        )
        notes = [
            self._drift_note(),
            f"Max |companion root| of the posterior mean path: "
            f"{float(roots.max()):.4f} (a drifting model may pass through "
            "locally explosive dates; stability_path() shows where).",
            "The training sample calibrates the priors and is then discarded; "
            "all paths are indexed on the estimation sample that follows it.",
            "No likelihood, parameter count, or information criteria are "
            "reported: this is a Gibbs posterior and has none.",
        ]
        return SummaryTable(
            title=f"{self._family()}({self.order}) Results",
            metadata=(
                ("Model", f"{self._family()}({self.order})"),
                ("Draws", f"{self.n_draws} ({self.n_burn} burn, thin {self.thin})"),
                ("Variables", f"{self.k_endog}"),
                ("Kept", f"{self.n_kept}"),
                ("Observations", f"{self.nobs}"),
                ("Training", f"{self.training}"),
                ("Trend", self.trend),
                ("State dimension", f"{self.beta_mean.shape[1]}"),
            ),
            columns=("equation", "own L1 at start", "own L1 at end"),
            rows=rows,
            notes=tuple(notes),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class TVPVARSVResult(TVPVARResult):
    """A fitted TVP-VAR with stochastic volatility, Primiceri (2005).

    Everything the homoskedastic result reports, plus the volatility side:
    random-walk log variances per equation under the triangular
    factorization ``Sigma_t = A^{-1} H_t A^{-T}``, with the impact and
    log-variance draws retained because the structural wrapper needs their
    posterior, not a point.

    Attributes:
        h_draws: ``(S, nobs, k)`` kept log-variance path draws.
        impact_draws: ``(S, k, k)`` kept draws of ``A^{-1}``.
        vol_of_vol: ``(k,)`` posterior mean random-walk variances of the
            log volatilities.
    """

    h_draws: npt.NDArray[np.float64] = field(repr=False)
    impact_draws: npt.NDArray[np.float64] = field(repr=False)
    vol_of_vol: npt.NDArray[np.float64] = field(repr=False)

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

    def _family(self) -> str:
        """Specification stem for display."""
        return "TVP-VAR-SV"


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class TVPSVARResult(_SummaryMixin):
    """A structurally interpreted TVP-VAR: one recursive declaration, a path of answers.

    Attributes:
        source: The fitted reduced-form result, homoskedastic or SV.
        impacts: ``(nobs, k, k)`` posterior mean structural impact path --
            constant rows repeated for the homoskedastic model.
    """

    source: TVPVARResult = field(repr=False)
    impacts: npt.NDArray[np.float64] = field(repr=False)

    @property
    def names(self) -> tuple[str, ...]:
        """Variable labels, in column order."""
        return self.source.names

    @property
    def nobs(self) -> int:
        """Estimation rows."""
        return self.source.nobs

    def impact(self, t: int) -> npt.NDArray[np.float64]:
        """The posterior mean structural impact matrix at date ``t``.

        Args:
            t: Row of the estimation sample.

        Returns:
            A lower-triangular ``(k, k)`` matrix.

        Raises:
            SpecificationError: If ``t`` is out of range.
        """
        if not 0 <= t < self.nobs:
            raise SpecificationError(f"t must be in 0..{self.nobs - 1}; got {t}.")
        return self.impacts[t]

    def irf(
        self, horizon: int = 20, *, t: int, cumulative: bool = False
    ) -> npt.NDArray[np.float64]:
        """Structural impulse responses at date ``t``, held frozen.

        The same linearization every regime family declares, in its drifting
        form: coefficients and impact are frozen at date ``t``'s posterior
        mean, so the response reads "the transmission mechanism prevailing
        at t", exact only while the parameters do not drift over the horizon
        -- and a posterior-mean plug-in besides, with the path uncertainty
        available in the source's retained draws.

        Args:
            horizon: Largest lead to return.
            t: Date whose prevailing dynamics to propagate.
            cumulative: Return running sums.

        Returns:
            An array of shape ``(horizon + 1, k, k)``; entry ``[h, i, j]``
            is the response of variable ``i`` at lead ``h`` to structural
            shock ``j``, as of date ``t``.

        Raises:
            SpecificationError: If ``horizon`` is negative or ``t`` out of
                range.
        """
        if horizon < 0:
            raise SpecificationError(f"horizon must be non-negative; got {horizon}.")
        stack = self.source.coefficients_at(t)
        k, p = len(self.names), self.source.order
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
        out = psi @ self.impact(t)
        return np.cumsum(out, axis=0) if cumulative else out

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        first = np.abs(np.diagonal(self.impacts[0]))
        last = np.abs(np.diagonal(self.impacts[-1]))
        rows = tuple(
            (name, f"{first[i]:.4f}", f"{last[i]:.4f}") for i, name in enumerate(self.names)
        )
        sv = isinstance(self.source, TVPVARSVResult)
        notes = [
            "The identifying restriction is recursive in the order of names, "
            + (
                "declared at estimation time through the triangular "
                "factorization: a different ordering is a different fitted "
                "model, not a different rotation of this one."
                if sv
                else "applied to the posterior mean innovation covariance."
            ),
            "irf(t=...) freezes coefficients and impact at date t's posterior "
            "mean: read it as the transmission mechanism prevailing at t, "
            "exact only while parameters do not drift over the horizon.",
            "Point summaries are posterior-mean plug-ins; the source result "
            "retains the draws for full uncertainty propagation.",
        ]
        return SummaryTable(
            title=f"TVP-SVAR ({self.source._family()}) Results",
            metadata=(
                ("Scheme", "recursive"),
                ("Reduced form", f"{self.source._family()}({self.source.order})"),
                ("Observations", f"{self.nobs}"),
                ("Kept draws", f"{self.source.n_kept}"),
            ),
            columns=("equation", "|impact| at start", "|impact| at end"),
            rows=rows,
            notes=tuple(notes),
        )


class TVPVAR(_TimeVaryingVectorAutoRegressionModel[TVPVARResult]):
    """Time-varying-parameter VAR with constant innovation covariance.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order.
        training: Rows split off the front to calibrate the priors, then
            discarded. Defaults to roughly forty, floored by what the prior
            regression needs.
        trend: Deterministic terms.
        names: One label per variable. Defaults to ``y1 ... yk``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((140, 2))
        >>> for t in range(1, 140):
        ...     a = 0.4 + 0.3 * t / 140
        ...     y[t] = a * y[t - 1] + rng.standard_normal(2)
        >>> res = TVPVAR(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
        >>> res.coefficient_path("y1", "y1.L1").shape
        (99, 3)
    """

    __slots__ = ()

    def fit(
        self,
        *,
        n_draws: int = 2000,
        n_burn: int = 1000,
        thin: int = 2,
        k_drift: float = 0.01,
        seed: int | np.random.Generator | None = None,
    ) -> TVPVARResult:
        """Estimate by Gibbs sampling.

        Args:
            n_draws: Total sampler iterations.
            n_burn: Burn-in iterations discarded.
            thin: Keep every ``thin``-th post-burn draw.
            k_drift: Primiceri's ``k_Q``: prior scale of coefficient drift
                relative to the training coefficient uncertainty. The single
                most consequential number in the model -- it is what
                disciplines a parameter path with more parameters than data.
            seed: Seed or generator, for reproducibility.

        Returns:
            The fitted :class:`TVPVARResult`.

        Raises:
            SpecificationError: If the draw bookkeeping is inconsistent.
            NumericalError: If a conditional draw collapses.
        """
        fit = self._fit_tvp(
            sv=False,
            n_draws=n_draws,
            n_burn=n_burn,
            thin=thin,
            k_drift=k_drift,
            k_vol=0.01,
            seed=seed,
        )
        return TVPVARResult(
            endog=self.endog,
            names=self.names,
            order=self.order,
            trend=self.trend,
            training=fit.training,
            nobs=fit.nobs,
            beta_mean=fit.beta_mean,
            beta_low=fit.beta_low,
            beta_high=fit.beta_high,
            beta_draws=fit.beta_draws,
            state_cov=fit.state_cov,
            sigma_u=fit.sigma_u,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
        )


class TVPVARSV(TVPVAR):
    """Time-varying-parameter VAR with stochastic volatility, Primiceri (2005).

    Coefficients drift as in :class:`TVPVAR`; in addition the innovation
    covariance moves every period through ``Sigma_t = A^{-1} H_t A^{-T}``
    with random-walk log variances -- the cited workhorse for "has policy
    changed, or have the shocks?", which is unanswerable with either
    ingredient alone.

    The triangular factorization is taken in the order of ``names``, and
    that ordering is a structural declaration made at estimation time; see
    :class:`TimeVaryingSVAR`.

    Args:
        endog: The observed panel, shape ``(nobs, k)``.
        order: Autoregressive order.
        training: Rows split off the front to calibrate the priors.
        trend: Deterministic terms.
        names: One label per variable, in the intended recursive order.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((140, 2))
        >>> for t in range(1, 140):
        ...     scale = 0.5 + 1.0 * t / 140
        ...     y[t] = 0.5 * y[t - 1] + scale * rng.standard_normal(2)
        >>> res = TVPVARSV(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
        >>> res.volatility_path("y1").shape
        (99, 3)
    """

    __slots__ = ()

    def fit(  # type: ignore[override]
        self,
        *,
        n_draws: int = 2000,
        n_burn: int = 1000,
        thin: int = 2,
        k_drift: float = 0.01,
        k_vol: float = 0.01,
        seed: int | np.random.Generator | None = None,
    ) -> TVPVARSVResult:
        """Estimate by Gibbs sampling with the KSC volatility block.

        Args:
            n_draws: Total sampler iterations.
            n_burn: Burn-in iterations discarded.
            thin: Keep every ``thin``-th post-burn draw.
            k_drift: Primiceri's ``k_Q``: prior scale of coefficient drift.
            k_vol: Primiceri's ``k_W``: prior scale of the log-volatility
                random walk.
            seed: Seed or generator.

        Returns:
            The fitted :class:`TVPVARSVResult`.

        Raises:
            SpecificationError: If the draw bookkeeping is inconsistent.
            NumericalError: If a conditional draw collapses.
        """
        fit = self._fit_tvp(
            sv=True,
            n_draws=n_draws,
            n_burn=n_burn,
            thin=thin,
            k_drift=k_drift,
            k_vol=k_vol,
            seed=seed,
        )
        assert fit.h_draws is not None
        assert fit.impact_draws is not None
        assert fit.vol_of_vol is not None
        return TVPVARSVResult(
            endog=self.endog,
            names=self.names,
            order=self.order,
            trend=self.trend,
            training=fit.training,
            nobs=fit.nobs,
            beta_mean=fit.beta_mean,
            beta_low=fit.beta_low,
            beta_high=fit.beta_high,
            beta_draws=fit.beta_draws,
            state_cov=fit.state_cov,
            sigma_u=fit.sigma_u,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            n_draws=fit.n_draws,
            n_burn=fit.n_burn,
            thin=fit.thin,
            h_draws=fit.h_draws,
            impact_draws=fit.impact_draws,
            vol_of_vol=fit.vol_of_vol,
        )


class TimeVaryingSVAR:
    """Recursive structural interpretation of a fitted TVP-VAR, per date.

    Not an ``_IdentificationModel``: a drifting system is a continuum of
    closed systems, one per date, and what this class packages is the path
    of identifications under a single recursive declaration.

    The declaration works differently across the two reduced forms, and the
    difference is enforced rather than papered over. For the homoskedastic
    model the ordering is applied here, to the posterior mean covariance,
    and any permutation of ``names`` is admissible. For the SV model the
    triangular factorization *was the estimation*: ``A`` and the volatility
    paths were drawn in the order of ``names``, so the ordering was declared
    when the model was fitted, and asking for a different one here is asking
    for a different fitted model -- refit with the columns reordered.

    Args:
        source: A fitted :class:`TVPVARResult` or :class:`TVPVARSVResult`.
        order: Recursive ordering for the *homoskedastic* model; ``None``
            keeps the order of ``names``. Refused for an SV source.

    Raises:
        SpecificationError: If the source is not a fitted TVP result, an
            ordering is requested for an SV source, or the ordering is not a
            permutation of ``names``.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros((140, 2))
        >>> for t in range(1, 140):
        ...     y[t] = 0.5 * y[t - 1] + rng.standard_normal(2)
        >>> res = TVPVAR(y, order=1, training=40).fit(n_draws=40, n_burn=15, seed=0)
        >>> svar = TimeVaryingSVAR(res).identify()
        >>> svar.irf(4, t=50).shape
        (5, 2, 2)
    """

    __slots__ = ("_order", "_source")

    def __init__(
        self,
        source: TVPVARResult,
        *,
        order: Sequence[str] | None = None,
    ) -> None:
        """Validate the source and the ordering request."""
        if not isinstance(source, TVPVARResult):
            raise SpecificationError(
                "TimeVaryingSVAR constructs with a fitted TVPVAR or TVPVARSV "
                f"result; got {type(source).__name__}."
            )
        if order is not None and isinstance(source, TVPVARSVResult):
            raise SpecificationError(
                "the SV model's recursive ordering is declared at estimation "
                "time through the triangular factorization; a different "
                "ordering is a different fitted model. Refit TVPVARSV with "
                "the columns in the ordering you intend."
            )
        if order is not None:
            resolved = tuple(str(name) for name in order)
            if sorted(resolved) != sorted(source.names):
                raise SpecificationError(
                    f"order must be a permutation of {source.names}; got {resolved}."
                )
            self._order: tuple[str, ...] | None = resolved
        else:
            self._order = None
        self._source = source

    def identify(self) -> TVPSVARResult:
        """Assemble the impact path under the recursive declaration.

        Returns:
            The :class:`TVPSVARResult`. For an SV source, the impact at each
            date is the posterior mean of ``A^{-1} diag(exp(h_t / 2))``
            across the kept draws; for a homoskedastic source it is the
            Cholesky factor of the posterior mean covariance, repeated.
        """
        source = self._source
        n, k = source.nobs, source.k_endog
        if isinstance(source, TVPVARSVResult):
            scale = np.exp(0.5 * source.h_draws)
            impacts = np.einsum("sij,stj->stij", source.impact_draws, scale).mean(axis=0)
        else:
            sigma = source.sigma_u
            if self._order is not None:
                perm = [source.names.index(name) for name in self._order]
                inverse = np.argsort(perm)
                reordered = sigma[np.ix_(perm, perm)]
                chol = np.linalg.cholesky(reordered)
                chol = chol[np.ix_(inverse, inverse)]
            else:
                chol = np.linalg.cholesky(sigma)
            impacts = np.broadcast_to(chol, (n, k, k)).copy()
        return TVPSVARResult(source=source, impacts=impacts)
