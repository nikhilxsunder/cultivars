# filepath: /src/cultivars/univariate/garch.py
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

"""Conditional-variance models: GARCH, GJR, EGARCH, and FIGARCH.

All four estimate the conditional mean and the conditional variance *jointly*,
so the reported likelihood is the true joint one rather than the two-step
approximation you get from fitting a mean model and then a variance model to
its residuals.

The group splits in two, and the split is not cosmetic. GARCH, GJR and EGARCH
are finite-order recursions whose shocks decay geometrically, so persistence is
a single number, a half-life exists, and -- for the level families -- so does an
unconditional variance. FIGARCH applies a fractional filter whose shocks decay
hyperbolically: no geometric rate describes them, no half-life is defined, and
the process is not covariance stationary for any ``d > 0``. Sharing one
``persistence`` implementation across both would produce a plausible number for
FIGARCH that means nothing, which is why :class:`ConditionalVarianceResult`
declares it and each concrete result answers it separately.

Two other places report ``None`` rather than a number that would mislead. The
log-variance family has no closed-form unconditional variance in the reported
parameters -- its stationary level is the mean of a lognormal -- so
:attr:`GARCHResult.unconditional_variance` withholds it rather than returning
``omega / (1 - beta)``. And stationarity here is variance stationarity,
``persistence < 1``; the conditional-mean autoregressive block plays no part,
so these results deliberately do not carry the mean-model stationarity mixin.

References:
    Bollerslev, T. (1986). Generalized autoregressive conditional
    heteroskedasticity. *Journal of Econometrics*, 31(3).
    Glosten, L., Jagannathan, R. & Runkle, D. (1993). On the relation between
    the expected value and the volatility of the nominal excess return on
    stocks. *Journal of Finance*, 48(5).
    Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns.
    *Econometrica*, 59(2).
    Baillie, R., Bollerslev, T. & Mikkelsen, H. (1996). Fractionally integrated
    generalized autoregressive conditional heteroskedasticity. *Journal of
    Econometrics*, 74(1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .._core import InformationCriteria, SummaryTable
from .._internals import (
    _ComparisonMixin,
    _ConditionalVarianceMixin,
    _FractionalVarianceFit,
    _FractionalVarianceModel,
    _SeriesMixin,
    _ShortMemoryVarianceFit,
    _ShortMemoryVarianceModel,
    _SummaryMixin,
)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ConditionalVarianceResult(
    _SummaryMixin, _SeriesMixin, _ComparisonMixin, _ConditionalVarianceMixin
):
    """What every fitted conditional-variance model reports.

    Abstract in intent: it carries the mean intercept, the variance intercept
    and the fitted variance path, and derives from the path alone everything
    that does not depend on how the path was produced. :attr:`persistence` is
    declared here and implemented by each subclass, because the two arms of the
    group measure it in incompatible ways.

    Attributes:
        endog: The full observed series.
        fittedvalues: Fitted conditional means over the effective sample.
        resid: Mean residuals over the effective sample.
        llf: Maximized joint Gaussian log-likelihood.
        nobs: Effective observations.
        n_params: Free parameter count.
        mean: ``"constant"`` or ``"zero"``.
        const: Mean intercept, or ``None`` when the mean is zero.
        omega: Variance intercept.
        conditional_variance: The fitted variance path.
    """

    endog: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int
    mean: str
    const: float | None
    omega: float
    conditional_variance: npt.NDArray[np.float64]

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """Aligned per-observation output, widened by the variance surface.

        Uses the two-argument ``super`` deliberately. ``@dataclass(slots=True)``
        builds a *new* class object and rebinds the name to it, so the
        ``__class__`` cell that zero-argument ``super()`` closes over still
        points at the original, pre-slots class -- which no subclass inherits
        from. ``super()._series()`` therefore raises ``TypeError`` the moment a
        subclass calls it, and only for subclasses, so it survives any test
        that exercises the base alone.
        """
        base = super(ConditionalVarianceResult, self)._series()
        base["conditional_variance"] = self.conditional_variance
        base["conditional_volatility"] = self.conditional_volatility
        base["standardized_resid"] = self.standardized_resid
        return base

    @property
    def standardized_resid(self) -> npt.NDArray[np.float64]:
        """Mean residuals scaled by the fitted conditional volatility."""
        return self.resid / self.conditional_volatility

    @property
    def persistence(self) -> float:
        """The decay rate of a shock to the conditional variance.

        Raises:
            NotImplementedError: If a concrete result does not define it.
        """
        raise NotImplementedError

    @property
    def is_covariance_stationary(self) -> bool:
        """Whether the variance process has a finite unconditional level."""
        return self.persistence < 1.0

    @property
    def half_life(self) -> float:
        """Periods for a variance shock to decay by half.

        Returns:
            ``inf`` when the process is not covariance stationary, since a
            shock to an integrated variance never halves.
        """
        p = self.persistence
        if not 0.0 < p < 1.0:
            return float("inf")
        return float(np.log(0.5) / np.log(p))


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class GARCHResult(ConditionalVarianceResult):
    """A fitted finite-order conditional-variance model.

    Attributes:
        vol: The family that produced the fit, needed because persistence and
            the unconditional variance are different functionals across the
            three.
        ar_params: Conditional-mean AR coefficients; empty when ``ar_lags == 0``.
        alpha: Coefficients on the shock magnitude.
        gamma: Asymmetry coefficients; empty for the symmetric family.
        beta: Persistence coefficients.
    """

    vol: str
    ar_params: npt.NDArray[np.float64]
    alpha: npt.NDArray[np.float64]
    gamma: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]

    @classmethod
    def _from_fit(
        cls, fit: _ShortMemoryVarianceFit, model: _ShortMemoryVarianceModel[GARCHResult]
    ) -> GARCHResult:
        """Assemble the public result from a raw fit and its specification."""
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            mean="constant" if model.has_constant_mean else "zero",
            const=fit.const,
            omega=fit.omega,
            conditional_variance=fit.conditional_variance,
            vol=fit.vol,
            ar_params=fit.ar_params,
            alpha=fit.alpha,
            gamma=fit.gamma,
            beta=fit.beta,
        )

    @property
    def order(self) -> tuple[int, int, int]:
        """The variance order ``(p, o, q)`` recovered from the coefficient blocks."""
        return (self.alpha.size, self.gamma.size, self.beta.size)

    @property
    def persistence(self) -> float:
        """The decay rate of a shock to the conditional variance.

        For the level families this sums the coefficients with the asymmetry
        block at half weight, its unconditional frequency under a symmetric
        innovation distribution. For the log-variance family it is the
        autoregressive root of the log variance alone: the magnitude and sign
        terms are mean-zero innovations, not persistence.
        """
        if self.vol == "EGARCH":
            return float(self.beta.sum())
        return float(self.alpha.sum() + 0.5 * self.gamma.sum() + self.beta.sum())

    @property
    def unconditional_variance(self) -> float | None:
        """The long-run variance ``omega / (1 - persistence)``.

        Returns:
            ``None`` for the log-variance family, whose stationary variance is
            the mean of a lognormal and has no closed form in the reported
            parameters, and ``None`` when the process is not covariance
            stationary. Returning a number in either case would invite it
            straight into a risk calculation.
        """
        if self.vol == "EGARCH" or not self.is_covariance_stationary:
            return None
        return self.omega / (1.0 - self.persistence)

    @property
    def has_leverage(self) -> bool:
        """Whether an asymmetry block was estimated."""
        return self.gamma.size > 0

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order."""
        out: dict[str, float] = {}
        if self.const is not None:
            out["const"] = self.const
        for i, value in enumerate(self.ar_params, start=1):
            out[f"ar.L{i}"] = float(value)
        out["omega"] = self.omega
        for i, value in enumerate(self.alpha, start=1):
            out[f"alpha[{i}]"] = float(value)
        for i, value in enumerate(self.gamma, start=1):
            out[f"gamma[{i}]"] = float(value)
        for i, value in enumerate(self.beta, start=1):
            out[f"beta[{i}]"] = float(value)
        return out

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        p, o, q = self.order
        return f"{self.vol}({p}, {o}, {q})" if o else f"{self.vol}({p}, {q})"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        ic: InformationCriteria = self.information_criteria
        uncond = self.unconditional_variance
        notes = [
            f"Persistence: {self.persistence:.4f}   "
            f"Covariance stationary: {self.is_covariance_stationary}   "
            f"Half-life: {self.half_life:.1f}",
        ]
        if uncond is not None:
            notes.append(f"Unconditional variance: {uncond:.4f}")
        elif self.vol == "EGARCH":
            notes.append(
                "Unconditional variance is not reported for a log-variance model; "
                "its stationary level is the mean of a lognormal."
            )
        notes.append("Standard errors are not yet available for this estimator.")
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=(
                ("Model", self._comparison_label()),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Mean", self.mean),
                ("AIC", f"{ic.aic:.3f}"),
                ("AR lags", f"{self.ar_params.size}"),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=tuple(notes),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class FIGARCHResult(ConditionalVarianceResult):
    """A fitted fractionally integrated conditional-variance model.

    Attributes:
        truncation: Number of infinite-order weights retained.
        phi: Short-memory numerator weight of the fractional polynomial.
        d: Fractional integration order.
        beta: Denominator weight.
    """

    truncation: int
    phi: float
    d: float
    beta: float

    @classmethod
    def _from_fit(
        cls, fit: _FractionalVarianceFit, model: _FractionalVarianceModel[FIGARCHResult]
    ) -> FIGARCHResult:
        """Assemble the public result from a raw fit and its specification."""
        return cls(
            endog=model.endog,
            fittedvalues=fit.fittedvalues,
            resid=fit.resid,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
            mean="constant" if model.has_constant_mean else "zero",
            const=fit.const,
            omega=fit.omega,
            conditional_variance=fit.conditional_variance,
            truncation=model.truncation,
            phi=fit.phi,
            d=fit.d,
            beta=fit.beta,
        )

    @property
    def persistence(self) -> float:
        """Unity, by construction.

        A fractionally integrated variance is not covariance stationary for any
        ``d > 0``: shocks decay hyperbolically rather than geometrically, so no
        geometric rate describes them. Reporting the finite-order formula
        ``phi + beta`` here would produce a plausible number that means nothing,
        which is why :class:`GARCHResult` and this class do not share one.
        """
        return 1.0

    @property
    def is_covariance_stationary(self) -> bool:
        """Always ``False``; see :attr:`persistence`."""
        return False

    @property
    def has_long_memory(self) -> bool:
        """Whether ``d`` is far enough from zero to imply hyperbolic decay."""
        return self.d > 1e-3

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order."""
        out: dict[str, float] = {}
        if self.const is not None:
            out["const"] = self.const
        out["omega"] = self.omega
        out["phi"] = self.phi
        out["d"] = self.d
        out["beta"] = self.beta
        return out

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        return "FIGARCH(1, d, 1)"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        ic: InformationCriteria = self.information_criteria
        return SummaryTable(
            title="FIGARCH(1, d, 1) Results",
            metadata=(
                ("Model", "FIGARCH(1, d, 1)"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Mean", self.mean),
                ("AIC", f"{ic.aic:.3f}"),
                ("Truncation", f"{self.truncation}"),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=(
                f"Long memory: {self.has_long_memory}   d = {self.d:.4f}",
                "Not covariance stationary for any d > 0: shocks to the variance "
                "decay hyperbolically, so no half-life is defined.",
                "Standard errors are not yet available for this estimator.",
            ),
        )


class GARCH(_ShortMemoryVarianceModel[GARCHResult]):
    """Symmetric GARCH(p, q) with an optional AR mean.

    Args:
        endog: The series, typically returns or residuals.
        p: ARCH order, the number of lagged squared residuals.
        q: GARCH order, the number of lagged variances.
        ar_lags: Conditional-mean AR order.
        mean: ``"constant"`` or ``"zero"``.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> e = np.zeros(2000)
        >>> s2 = np.full(2000, 2.5)
        >>> z = rng.standard_normal(2000)
        >>> for t in range(1, 2000):
        ...     s2[t] = 0.05 + 0.08 * e[t - 1] ** 2 + 0.90 * s2[t - 1]
        ...     e[t] = np.sqrt(s2[t]) * z[t]
        >>> res = GARCH(e).fit()
        >>> res.is_covariance_stationary
        True
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        p: int = 1,
        q: int = 1,
        ar_lags: int = 0,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, vol="GARCH", p=p, o=0, q=q, ar_lags=ar_lags, mean=mean)

    def fit(self) -> GARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood.

        Returns:
            The fitted :class:`GARCHResult`.
        """
        return GARCHResult._from_fit(self._fit_family(), self)


class GJR(_ShortMemoryVarianceModel[GARCHResult]):
    """GJR-GARCH(p, o, q): a level model with a sign-asymmetric ARCH term.

    The asymmetry block loads only on negative shocks, so ``gamma > 0`` is the
    leverage effect -- bad news raising volatility more than good news of the
    same size.

    Args:
        endog: The series, typically returns or residuals.
        p: ARCH order.
        o: Asymmetry order; must be at least 1.
        q: GARCH order.
        ar_lags: Conditional-mean AR order.
        mean: ``"constant"`` or ``"zero"``.
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        ar_lags: int = 0,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, vol="GJR", p=p, o=o, q=q, ar_lags=ar_lags, mean=mean)

    def fit(self) -> GARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood.

        Returns:
            The fitted :class:`GARCHResult`.
        """
        return GARCHResult._from_fit(self._fit_family(), self)


class EGARCH(_ShortMemoryVarianceModel[GARCHResult]):
    """EGARCH(p, o, q): the recursion runs in logs, so no positivity constraint binds.

    Because the variance is exponentiated, the coefficients are unconstrained
    and the optimizer searches freely -- but the reported parameters are on the
    log scale, which is why :attr:`GARCHResult.unconditional_variance` declines
    to translate them back.

    Args:
        endog: The series, typically returns or residuals.
        p: Magnitude order.
        o: Sign (leverage) order; must be at least 1.
        q: Log-variance persistence order.
        ar_lags: Conditional-mean AR order.
        mean: ``"constant"`` or ``"zero"``.
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        ar_lags: int = 0,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, vol="EGARCH", p=p, o=o, q=q, ar_lags=ar_lags, mean=mean)

    def fit(self) -> GARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood.

        Returns:
            The fitted :class:`GARCHResult`.
        """
        return GARCHResult._from_fit(self._fit_family(), self)


class FIGARCH(_FractionalVarianceModel[FIGARCHResult]):
    """FIGARCH(1, d, 1): long-memory volatility through a fractional filter.

    The order is fixed, so the only structural choice beyond the mean is how far
    the infinite-order representation is truncated. Weights beyond the available
    history are applied to the pre-sample variance rather than dropped, so the
    truncation tail contributes a constant instead of vanishing silently.

    Args:
        endog: The series, typically returns or residuals.
        mean: ``"constant"`` or ``"zero"``.
        truncation: Infinite-order truncation lag.
    """

    __slots__ = ()

    def fit(self) -> FIGARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood.

        Returns:
            The fitted :class:`FIGARCHResult`.
        """
        return FIGARCHResult._from_fit(self._fit_family(), self)
