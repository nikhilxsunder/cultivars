# filepath: /src/cultivars/univariate/threshold.py
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

"""Observed-regime autoregressions: SETAR, TAR, LSTAR, and ESTAR.

All four models here are two-regime autoregressions whose regime is a
*deterministic function of something you can see* -- a lag of the series, or an
external variable you supply. That is what separates them from the
Markov-switching family, where the regime is latent and only its filtered
probability is ever recovered. Here the regime weight is computed, not
inferred, which is why estimation is least squares rather than EM and why the
results carry the transition variable itself as an aligned series.

The group splits on how abruptly the regime changes, and the split runs all the
way down to the estimator:

1. **Hard threshold** (:class:`SETAR`, :class:`TAR`). The weight is an
   indicator, so the sum of squares is a step function of the threshold --
   piecewise constant, nowhere differentiable in it. No gradient method can
   find the minimum, so the estimator is an exhaustive grid search over
   trimmed quantiles of the transition variable, with regime-wise OLS at each
   candidate. The delay may be searched jointly.
2. **Smooth transition** (:class:`LSTAR`, :class:`ESTAR`). The weight is a
   logistic or exponential function taking values in ``[0, 1]``, so the surface
   is smooth in the two transition parameters. Conditional on those, the model
   is linear, so the regime coefficients are concentrated out and only
   ``(gamma, c)`` is searched -- derivative-free and multi-start, because the
   surface is flat in ``gamma`` far from the data and multimodal in ``c``.

Two properties of the group are reported carefully rather than conveniently.

Stationarity is reported *per regime*, never as a single verdict. A two-regime
nonlinear autoregression has no single companion polynomial, and regime-wise
stationarity is sufficient but not necessary for the process to be globally
stationary: a threshold model with an explosive inner regime and a contracting
outer one is perfectly well behaved, since excursions are pulled back. The
results therefore expose :attr:`ObservedRegimeResult.lower_stability` and
:attr:`ObservedRegimeResult.upper_stability` and refuse to collapse them.

Likelihood-ratio tests are blocked outright. The threshold is not identified
under the null of linearity -- when the model is linear, ``c`` and ``d`` simply
do not appear in the likelihood -- so the usual chi-squared limit does not hold
for any test that involves them (the Davies problem). Since the inherited
:meth:`compare` and :meth:`likelihood_ratio_test` would otherwise be perfectly
happy to hand back a well-formed and wrong p-value,
:meth:`ObservedRegimeResult.likelihood_ratio_test` raises instead.

References:
    Tong, H. (1990). *Non-linear Time Series: A Dynamical System Approach*.
    Tsay, R. S. (1989). Testing and modeling threshold autoregressive
    processes. *Journal of the American Statistical Association*, 84(405).
    Terasvirta, T. (1994). Specification, estimation, and evaluation of smooth
    transition autoregressive models. *Journal of the American Statistical
    Association*, 89(425).
    van Dijk, D., Terasvirta, T. & Franses, P. H. (2002). Smooth transition
    autoregressive models -- a survey of recent developments. *Econometric
    Reviews*, 21(1).
    Davies, R. B. (1987). Hypothesis testing when a nuisance parameter is
    present only under the alternative. *Biometrika*, 74(1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import InformationCriteria, SummaryTable, trailing_lag
from .._internals import (
    _ComparisonMixin,
    _LikelihoodRatioResult,
    _SeriesMixin,
    _SmoothTransitionFit,
    _SmoothTransitionModel,
    _StabilityResult,
    _SummaryMixin,
    _ThresholdFit,
    _ThresholdModel,
)
from ..exceptions import SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ObservedRegimeResult(_SummaryMixin, _SeriesMixin, _ComparisonMixin):
    """What every fitted observed-regime autoregression reports.

    Carries the two coefficient blocks, the transition variable they are
    indexed by, and the split point between them. Subclasses supply the
    regime weight, which is the one thing an abrupt and a smooth transition
    genuinely do differently.

    Attributes:
        endog: The full input series.
        fittedvalues: One-step fitted values over the effective sample.
        resid: Residuals over the effective sample.
        llf: Gaussian log-likelihood at the least-squares estimate.
        nobs: Effective sample size after trimming for lags and delay.
        n_params: Free parameters, including the innovation variance.
        order: Autoregressive order within each regime.
        delay: Delay of the transition variable.
        threshold: The regime split point, in the units of the transition
            variable.
        threshold_values: The transition variable, aligned with ``resid``.
        lower_params: Lower-regime coefficients, intercept first.
        upper_params: Upper-regime coefficients, intercept first.
        sigma2: Innovation variance, ``ssr / nobs``.
        ssr: Sum of squared residuals at the optimum.
    """

    endog: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int
    order: int
    delay: int
    threshold: float
    threshold_values: npt.NDArray[np.float64]
    lower_params: npt.NDArray[np.float64]
    upper_params: npt.NDArray[np.float64]
    sigma2: float
    ssr: float

    @property
    def regime_weight(self) -> npt.NDArray[np.float64]:
        """Weight on the upper regime at each observation, in ``[0, 1]``.

        Returns:
            An array of length ``nobs``.

        Raises:
            NotImplementedError: If the concrete result does not define one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define regime_weight to describe its transition."
        )

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """Aligned per-observation output, widened by the regime surface.

        Uses the two-argument ``super`` deliberately. ``@dataclass(slots=True)``
        builds a *new* class object and rebinds the name to it, so the
        ``__class__`` cell that zero-argument ``super()`` closes over still
        points at the original, pre-slots class -- which no subclass inherits
        from. ``super()._series()`` would therefore raise ``TypeError`` the
        moment a subclass called it, and only for subclasses, so it would
        survive any test that exercised the base alone.

        Returns:
            The observed/fitted/residual triple plus the transition variable
            and the upper-regime weight.
        """
        base = super(ObservedRegimeResult, self)._series()
        base["threshold_variable"] = self.threshold_values
        base["regime_weight"] = self.regime_weight
        return base

    @property
    def upper_fraction(self) -> float:
        """Share of the effective sample assigned to the upper regime.

        For a hard threshold this is the exact proportion of observations above
        the split; for a smooth transition it is the mean transition weight,
        which is the natural continuous analogue.
        """
        return float(np.mean(self.regime_weight))

    @property
    def lower_stability(self) -> _StabilityResult:
        """Companion-eigenvalue verdict for the lower-regime AR block."""
        return _StabilityResult.assess_stability(self.lower_params[1:])

    @property
    def upper_stability(self) -> _StabilityResult:
        """Companion-eigenvalue verdict for the upper-regime AR block."""
        return _StabilityResult.assess_stability(self.upper_params[1:])

    @property
    def is_regimewise_stationary(self) -> bool:
        """Whether *both* regimes are stationary read as linear autoregressions.

        Sufficient for global stationarity of the nonlinear process, but not
        necessary: a model whose inner regime is explosive and whose outer
        regime contracts is globally stationary, because excursions are pulled
        back. A ``False`` here is a prompt to check the regime blocks
        individually, not a verdict that the process explodes.
        """
        return self.lower_stability.is_stable and self.upper_stability.is_stable

    def _transition_params(self) -> dict[str, float]:
        """Parameters describing the transition itself, keyed for display."""
        return {}

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order."""
        out: dict[str, float] = {}
        for label, block in (("lower", self.lower_params), ("upper", self.upper_params)):
            out[f"{label}.const"] = float(block[0])
            for i, value in enumerate(block[1:], start=1):
                out[f"{label}.ar.L{i}"] = float(value)
        out["threshold"] = self.threshold
        out.update(self._transition_params())
        out["sigma2"] = self.sigma2
        return out

    def likelihood_ratio_test(self, unrestricted: _ComparisonMixin) -> _LikelihoodRatioResult:
        """Refuse the test: the threshold is unidentified under the null.

        Args:
            unrestricted: Ignored.

        Returns:
            Never returns.

        Raises:
            SpecificationError: Always. Under linearity the threshold and the
                delay drop out of the likelihood entirely, so the likelihood
                ratio has no chi-squared limit and the nuisance parameters are
                present only under the alternative -- the Davies problem. The
                correct procedure is a bootstrap or a supremum-type test
                (Hansen 1996), not a chi-squared tail. Returning a well-formed
                and wrong p-value here would be worse than declining.
        """
        raise SpecificationError(
            "a chi-squared likelihood-ratio test is not valid for observed-regime "
            "autoregressions: the threshold and delay are not identified under the "
            "null of linearity, so the statistic has a non-standard distribution. "
            "Use a bootstrap or a supremum-type test instead."
        )

    def _summary_metadata(self) -> tuple[tuple[str, str], ...]:
        """Left/right metadata pairs shared by both transition shapes."""
        ic: InformationCriteria = self.information_criteria
        return (
            ("Model", self._comparison_label()),
            ("Log-likelihood", f"{self.llf:.3f}"),
            ("Delay", f"{self.delay}"),
            ("AIC", f"{ic.aic:.3f}"),
            ("Threshold", f"{self.threshold:.4f}"),
            ("BIC", f"{ic.bic:.3f}"),
            ("Observations", f"{self.nobs}"),
            ("HQIC", f"{ic.hqic:.3f}"),
        )

    def _stationarity_note(self) -> str:
        """One line reporting each regime's largest companion root."""
        return (
            f"Regime-wise stationary: {self.is_regimewise_stationary}   "
            f"max |root| lower = {self.lower_stability.max_modulus:.4f}, "
            f"upper = {self.upper_stability.max_modulus:.4f}"
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SETARResult(ObservedRegimeResult):
    """A fitted hard-threshold autoregression.

    Attributes:
        self_exciting: Whether the transition variable is a lag of the series
            itself (:class:`SETAR`) or an external variable (:class:`TAR`).
        n_lower: Observations assigned to the lower regime.
        n_upper: Observations assigned to the upper regime.
        searched_delay: Whether the delay was chosen by the grid search rather
            than fixed by the caller.
    """

    self_exciting: bool
    n_lower: int
    n_upper: int
    searched_delay: bool

    @classmethod
    def _from_fit(cls, fit: _ThresholdFit, model: _ThresholdModel[SETARResult]) -> SETARResult:
        """Assemble the public result from a raw fit and its specification."""
        external = model._threshold_variable
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            order=model.order,
            delay=fit.delay,
            threshold=fit.threshold,
            threshold_values=trailing_lag(
                model.endog if external is None else external,
                delay=fit.delay,
                length=fit.nobs,
            ),
            lower_params=fit.lower_params,
            upper_params=fit.upper_params,
            sigma2=fit.sigma2,
            ssr=fit.ssr,
            self_exciting=model.self_exciting,
            n_lower=fit.n_lower,
            n_upper=fit.n_upper,
            searched_delay=model.delay is None,
        )

    @property
    def regime_weight(self) -> npt.NDArray[np.float64]:
        """Indicator of the upper regime: ``1.0`` above the threshold, else ``0.0``.

        The comparison is strict on the upper side, matching the estimator's
        ``z <= r`` assignment to the lower regime, so a transition value
        landing exactly on the threshold is counted the same way here as it was
        when the coefficients were solved for.
        """
        return (self.threshold_values > self.threshold).astype(np.float64)

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        family = "SETAR" if self.self_exciting else "TAR"
        return f"{family}({self.order}, d={self.delay})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        split = (
            f"Regime split: {self.n_lower} below, {self.n_upper} above "
            f"({100.0 * self.upper_fraction:.1f}% upper)."
        )
        notes = [self._stationarity_note(), split]
        if self.searched_delay:
            notes.append(
                f"The delay was selected by the same grid search as the threshold; "
                f"d = {self.delay} minimized the total sum of squares."
            )
        if not self.self_exciting:
            notes.append("The transition variable is external, not a lag of the series.")
        notes.append(
            "The threshold is not identified under linearity, so information criteria "
            "rank specifications but do not test for a threshold."
        )
        notes.append("Standard errors are not yet available for this estimator.")
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=self._summary_metadata(),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=tuple(notes),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class STARResult(ObservedRegimeResult):
    """A fitted smooth-transition autoregression.

    Attributes:
        transition: ``"logistic"`` or ``"exponential"``.
        gamma: Transition speed, in units of standard deviations of the
            transition variable.
        transition_scale: The standard deviation ``gamma`` is expressed
            against, retained so the weight path can be reproduced and so
            ``gamma`` can be read on the original scale.
    """

    transition: str
    gamma: float
    transition_scale: float

    @classmethod
    def _from_fit(
        cls, fit: _SmoothTransitionFit, model: _SmoothTransitionModel[STARResult]
    ) -> STARResult:
        """Assemble the public result from a raw fit and its specification."""
        values = trailing_lag(model.endog, delay=fit.delay, length=fit.nobs)
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            order=model.order,
            delay=fit.delay,
            threshold=fit.threshold,
            threshold_values=values,
            lower_params=fit.lower_params,
            upper_params=fit.upper_params,
            sigma2=fit.sigma2,
            ssr=fit.ssr,
            transition=model.transition,
            gamma=fit.gamma,
            transition_scale=float(np.std(values)),
        )

    @property
    def regime_weight(self) -> npt.NDArray[np.float64]:
        """The transition function evaluated over the sample.

        Reproduces the estimator's own weighting exactly, including the
        clipping of the exponent -- an extreme ``gamma`` saturates the weight
        rather than overflowing ``exp``. The logistic form is monotone in the
        standardized transition variable, so the two regimes are asymmetric;
        the exponential form is symmetric about the threshold, so both tails
        share a regime and the middle is the other.
        """
        u = (self.threshold_values - self.threshold) / self.transition_scale
        if self.transition == "logistic":
            return 1.0 / (1.0 + np.exp(-np.clip(self.gamma * u, -50.0, 50.0)))
        return 1.0 - np.exp(-np.clip(self.gamma * u**2, 0.0, 50.0))

    @property
    def is_effectively_abrupt(self) -> bool:
        """Whether the fitted transition is so fast it is a hard threshold.

        A large ``gamma`` drives the logistic weight to an indicator, at which
        point the smooth model is a :class:`SETAR` with two extra parameters
        and a flat likelihood in ``gamma``. Reported so the flatness is visible
        rather than mistaken for a converged estimate.
        """
        return self.transition == "logistic" and self.gamma > 100.0

    def _transition_params(self) -> dict[str, float]:
        """The transition speed, inserted between the threshold and the variance."""
        return {"gamma": self.gamma}

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        family = "LSTAR" if self.transition == "logistic" else "ESTAR"
        return f"{family}({self.order}, d={self.delay})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        shape = (
            "monotone in the transition variable"
            if self.transition == "logistic"
            else "symmetric about the threshold"
        )
        notes = [
            self._stationarity_note(),
            f"Transition: {self.transition}, {shape}. "
            f"gamma = {self.gamma:.4f} per s.d. of the transition variable "
            f"(s.d. = {self.transition_scale:.4f}); "
            f"mean upper weight = {self.upper_fraction:.3f}.",
        ]
        if self.is_effectively_abrupt:
            notes.append(
                "gamma is large enough that the transition is effectively a hard "
                "threshold; the likelihood is nearly flat in gamma here, so prefer "
                "SETAR unless the smooth form is required."
            )
        notes.append(
            "The threshold and delay are not identified under linearity, so information "
            "criteria rank specifications but do not test for nonlinearity."
        )
        notes.append("Standard errors are not yet available for this estimator.")
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=self._summary_metadata(),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=tuple(notes),
        )


class SETAR(_ThresholdModel[SETARResult]):
    """Self-exciting threshold autoregression with two regimes.

    The regime is decided by ``y_{t-d}`` against a threshold, both estimated by
    grid search. Leaving ``delay`` as ``None`` searches ``1..order`` jointly
    with the threshold, at the cost of ``order`` times the grid.

    Args:
        endog: The series.
        order: Autoregressive order within each regime.
        delay: Threshold delay; ``None`` searches ``1..order``.
        trim: Fraction trimmed from each tail of the quantile grid, so that
            neither regime can be estimated from a handful of extreme points.
        n_grid: Candidate thresholds per delay.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> y = np.zeros(400)
        >>> for t in range(1, 400):
        ...     phi = 0.6 if y[t - 1] <= 0.0 else -0.4
        ...     y[t] = phi * y[t - 1] + rng.standard_normal()
        >>> res = SETAR(y, order=1, delay=1).fit()
        >>> res.n_lower + res.n_upper == res.nobs
        True
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        delay: int | None = None,
        trim: float = 0.15,
        n_grid: int = 300,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(
            endog,
            order=order,
            delay=delay,
            trim=trim,
            n_grid=n_grid,
            threshold_variable=None,
        )

    def fit(self) -> SETARResult:
        """Estimate the threshold, the delay, and both regimes by grid search."""
        return SETARResult._from_fit(self._fit_family(), self)


class TAR(_ThresholdModel[SETARResult]):
    """Threshold autoregression driven by an external transition variable.

    Identical machinery to :class:`SETAR`, but the regime is decided by a
    variable you supply rather than by the series' own past. The delay is fixed
    rather than searched: with an exogenous driver the lag is usually a
    modelling choice with economic content, not a nuisance parameter.

    Args:
        endog: The series.
        order: Autoregressive order within each regime.
        threshold_variable: The transition variable, aligned with ``endog``.
        delay: Delay applied to the transition variable.
        trim: Fraction trimmed from each tail of the quantile grid.
        n_grid: Candidate thresholds.
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        order: int,
        threshold_variable: npt.ArrayLike,
        delay: int = 1,
        trim: float = 0.15,
        n_grid: int = 300,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(
            endog,
            order=order,
            delay=delay,
            trim=trim,
            n_grid=n_grid,
            threshold_variable=threshold_variable,
        )

    def fit(self) -> SETARResult:
        """Estimate the threshold and both regimes by grid search."""
        return SETARResult._from_fit(self._fit_family(), self)


class LSTAR(_SmoothTransitionModel[STARResult]):
    """Logistic smooth-transition autoregression.

    The weight rises monotonically through the threshold, so the two regimes
    are genuinely different states and the dynamics differ above and below.
    Use this when the asymmetry has a direction -- contractions behaving
    differently from expansions, for instance.

    Args:
        endog: The series.
        order: Autoregressive order within each regime.
        delay: Delay of the transition variable ``y_{t-d}``.
    """

    __slots__ = ()

    def __init__(self, endog: npt.ArrayLike, *, order: int, delay: int = 1) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, order=order, transition="logistic", delay=delay)

    def fit(self) -> STARResult:
        """Estimate the transition by concentrated least squares."""
        return STARResult._from_fit(self._fit_family(), self)


class ESTAR(_SmoothTransitionModel[STARResult]):
    """Exponential smooth-transition autoregression.

    The weight is symmetric about the threshold, so both tails share one regime
    and the middle is the other. This is the natural form for mean reversion
    that strengthens with distance from equilibrium -- the canonical
    application is a real exchange rate under transaction costs, where small
    deviations persist and large ones are arbitraged away.

    Args:
        endog: The series.
        order: Autoregressive order within each regime.
        delay: Delay of the transition variable ``y_{t-d}``.
    """

    __slots__ = ()

    def __init__(self, endog: npt.ArrayLike, *, order: int, delay: int = 1) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, order=order, transition="exponential", delay=delay)

    def fit(self) -> STARResult:
        """Estimate the transition by concentrated least squares."""
        return STARResult._from_fit(self._fit_family(), self)
