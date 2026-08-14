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

from .._core import InformationCriteria, SummaryTable, _mean_label
from .._internals import (
    _ConditionalVarianceResult,
    _FractionalVarianceFit,
    _FractionalVarianceModel,
    _InvertibilityMixin,
    _ShortMemoryVarianceFit,
    _ShortMemoryVarianceModel,
    _StationarityMixin,
)


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class GARCHResult(_ConditionalVarianceResult):
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
class FIGARCHResult(_ConditionalVarianceResult):
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


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ARMAGARCHResult(GARCHResult, _StationarityMixin, _InvertibilityMixin):
    """A finite-order variance model under an ARMA conditional mean.

    Subclasses :class:`~cultivars.univariate.garch.GARCHResult` rather than
    reimplementing it: every variance property it exposes -- persistence, the
    unconditional variance, the leverage flag -- is a functional of the
    variance block alone and stays exactly as true when the mean grows a
    moving-average term. What is added is the mean side of the story.

    Attributes:
        mean_order: The conditional-mean order ``(ar_lags, ma_lags)``.
        ma_params: Moving-average coefficients of ``1 + theta_1 L + ...``;
            empty when ``ma_lags == 0``.
    """

    mean_order: tuple[int, int]
    ma_params: npt.NDArray[np.float64]

    @classmethod
    def _from_fit(
        cls,
        fit: _ShortMemoryVarianceFit,
        model: _ShortMemoryVarianceModel[GARCHResult],
    ) -> ARMAGARCHResult:
        """Assemble the public result from a raw fit and its specification.

        The model is annotated at the parent's result binding rather than this
        class's. Nothing read here depends on which result the model produces
        -- ``mean_order`` and ``has_constant_mean`` live on the shared
        specification base -- so narrowing the binding would break the override
        contract to express a dependency that does not exist.
        """
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
            mean_order=model.mean_order,
            ma_params=fit.ma_params,
        )

    @property
    def is_structurally_constrained(self) -> bool:
        """Whether the mean was searched inside the stationary-invertible region.

        ``True`` exactly when a moving-average block is present, because only
        then does the residual recursion feed on its own output and need the
        constraint to stay finite. When ``False``, the autoregressive lag
        weights were estimated unconstrained and :attr:`is_stationary` carries
        information about the data instead of about the transform.
        """
        return self.mean_order[1] > 0

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order."""
        out: dict[str, float] = {}
        if self.const is not None:
            out["const"] = self.const
        for i, value in enumerate(self.ar_params, start=1):
            out[f"ar.L{i}"] = float(value)
        for i, value in enumerate(self.ma_params, start=1):
            out[f"ma.L{i}"] = float(value)
        out["omega"] = self.omega
        for i, value in enumerate(self.alpha, start=1):
            out[f"alpha[{i}]"] = float(value)
        for i, value in enumerate(self.gamma, start=1):
            out[f"gamma[{i}]"] = float(value)
        for i, value in enumerate(self.beta, start=1):
            out[f"beta[{i}]"] = float(value)
        return out

    def _variance_label(self) -> str:
        """Name the variance family and its order."""
        p, o, q = self.order
        return f"{self.vol}({p}, {o}, {q})" if o else f"{self.vol}({p}, {q})"

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        mean = _mean_label(self.mean_order, has_const=self.const is not None)
        return f"{mean}-{self._variance_label()}"

    def _mean_notes(self) -> list[str]:
        """Diagnostics for the conditional-mean block."""
        notes = [
            f"Mean: stationary {self.is_stationary} "
            f"(max |AR root| = {self.stability.max_modulus:.4f})"
            + (
                f", invertible {self.is_invertible} "
                f"(max |MA root| = {self.invertibility.max_modulus:.4f})"
                if self.ma_params.size
                else ""
            )
            + "."
        ]
        if self.is_structurally_constrained:
            notes.append(
                "Both mean blocks were searched through the partial autocorrelations, "
                "so stationarity and invertibility hold by construction; the line above "
                "verifies the transform rather than describing the data."
            )
        elif self.ar_params.size:
            notes.append(
                "The mean is a regression on lagged levels with unconstrained weights, "
                "so the stationarity verdict above is a statement about the fit."
            )
        return notes

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        ic: InformationCriteria = self.information_criteria
        notes = self._mean_notes()
        notes.append(
            f"Variance: persistence {self.persistence:.4f}, covariance stationary "
            f"{self.is_covariance_stationary}"
            + (f", half-life {self.half_life:.2f} periods." if self.half_life else ".")
        )
        if self.unconditional_variance is not None:
            notes.append(f"Unconditional variance: {self.unconditional_variance:.6f}.")
        elif self.vol == "EGARCH":
            notes.append(
                "No unconditional variance is reported: the log-variance family's "
                "stationary level is the mean of a lognormal."
            )
        notes.append(
            "Mean and variance are estimated jointly, so the log-likelihood is the "
            "true joint one; the mean uses conditional sum of squares because the "
            "exact-ML filter assumes a constant innovation variance."
        )
        notes.append("Standard errors are not yet available for this estimator.")
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=(
                ("Model", self._comparison_label()),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Mean", _mean_label(self.mean_order, has_const=self.const is not None)),
                ("AIC", f"{ic.aic:.3f}"),
                ("Variance", self._variance_label()),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
                ("Persistence", f"{self.persistence:.4f}"),
                ("Parameters", f"{self.n_params}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=tuple(notes),
        )


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class ARMAFIGARCHResult(FIGARCHResult, _StationarityMixin, _InvertibilityMixin):
    """A fractionally integrated variance model under an ARMA conditional mean.

    Two long-memory-adjacent objects live in one model here and they are not
    the same object. The *mean* has an ARMA structure whose shocks decay
    geometrically; the *variance* is fractionally integrated and its shocks
    decay hyperbolically. :attr:`is_stationary` speaks only to the first,
    :attr:`is_covariance_stationary` only to the second, and the latter is
    ``False`` by construction for any ``d > 0``.

    Attributes:
        mean_order: The conditional-mean order ``(ar_lags, ma_lags)``.
        ar_params: Autoregressive coefficients of the mean; empty when
            ``ar_lags == 0``.
        ma_params: Moving-average coefficients of ``1 + theta_1 L + ...``;
            empty when ``ma_lags == 0``.
    """

    mean_order: tuple[int, int]
    ar_params: npt.NDArray[np.float64]
    ma_params: npt.NDArray[np.float64]

    @classmethod
    def _from_fit(
        cls,
        fit: _FractionalVarianceFit,
        model: _FractionalVarianceModel[FIGARCHResult],
    ) -> ARMAFIGARCHResult:
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
            mean_order=model.mean_order,
            ar_params=fit.ar_params,
            ma_params=fit.ma_params,
        )

    @property
    def is_structurally_constrained(self) -> bool:
        """Whether the mean was searched inside the stationary-invertible region."""
        return self.mean_order[1] > 0

    @property
    def params(self) -> dict[str, float]:
        """Estimated parameters keyed by display name, in table order."""
        out: dict[str, float] = {}
        if self.const is not None:
            out["const"] = self.const
        for i, value in enumerate(self.ar_params, start=1):
            out[f"ar.L{i}"] = float(value)
        for i, value in enumerate(self.ma_params, start=1):
            out[f"ma.L{i}"] = float(value)
        out["omega"] = self.omega
        out["phi"] = self.phi
        out["d"] = self.d
        out["beta"] = self.beta
        return out

    def _comparison_label(self) -> str:
        """Specification label used when this result appears in a ranking."""
        mean = _mean_label(self.mean_order, has_const=self.const is not None)
        return f"{mean}-FIGARCH(1, d, 1)"

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path."""
        ic: InformationCriteria = self.information_criteria
        notes = [
            f"Mean: stationary {self.is_stationary} "
            f"(max |AR root| = {self.stability.max_modulus:.4f})"
            + (
                f", invertible {self.is_invertible} "
                f"(max |MA root| = {self.invertibility.max_modulus:.4f})"
                if self.ma_params.size
                else ""
            )
            + ". The mean's shocks decay geometrically; the variance's do not.",
        ]
        if self.is_structurally_constrained:
            notes.append(
                "Both mean blocks were searched through the partial autocorrelations, "
                "so the verdict above verifies the transform rather than the data."
            )
        notes.extend(
            (
                f"Variance: long memory {self.has_long_memory}, d = {self.d:.4f}. Not "
                f"covariance stationary for any d > 0, so no half-life is defined and "
                f"persistence is reported as 1 by construction.",
                "Mean and variance are estimated jointly, so the log-likelihood is the "
                "true joint one; the mean uses conditional sum of squares because the "
                "exact-ML filter assumes a constant innovation variance.",
                "Standard errors are not yet available for this estimator.",
            )
        )
        return SummaryTable(
            title=f"{self._comparison_label()} Results",
            metadata=(
                ("Model", self._comparison_label()),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Mean", _mean_label(self.mean_order, has_const=self.const is not None)),
                ("AIC", f"{ic.aic:.3f}"),
                ("Truncation", f"{self.truncation}"),
                ("BIC", f"{ic.bic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("HQIC", f"{ic.hqic:.3f}"),
                ("d", f"{self.d:.4f}"),
                ("Parameters", f"{self.n_params}"),
            ),
            columns=("", "coef"),
            rows=tuple((name, f"{value:.4f}") for name, value in self.params.items()),
            notes=tuple(notes),
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


class ARGARCH(_ShortMemoryVarianceModel[ARMAGARCHResult]):
    """Autoregressive mean with a symmetric GARCH variance, estimated jointly.

    The same estimator as ``GARCH(y, ar_lags=...)``, under the name a reader
    expects when the mean is the point of the specification rather than an
    afterthought. The mean is a regression on lagged levels, so its weights are
    unconstrained and :attr:`ARMAGARCHResult.is_stationary` is informative.

    Args:
        endog: The series.
        ar_lags: Conditional-mean autoregressive order.
        p: Order of the shock-magnitude block.
        q: Order of the variance persistence block.
        mean: ``"constant"`` or ``"zero"``.
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        ar_lags: int = 1,
        p: int = 1,
        q: int = 1,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, vol="GARCH", p=p, o=0, q=q, ar_lags=ar_lags, ma_lags=0, mean=mean)

    def fit(self) -> ARMAGARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood."""
        return ARMAGARCHResult._from_fit(self._fit_family(), self)


class ARMAGARCH(_ShortMemoryVarianceModel[ARMAGARCHResult]):
    """ARMA mean with a symmetric GARCH variance, estimated jointly.

    Args:
        endog: The series.
        ar_lags: Conditional-mean autoregressive order.
        ma_lags: Conditional-mean moving-average order.
        p: Order of the shock-magnitude block.
        q: Order of the variance persistence block.
        mean: ``"constant"`` or ``"zero"``.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> n = 2000
        >>> s2 = np.ones(n)
        >>> e = np.zeros(n)
        >>> y = np.zeros(n)
        >>> for t in range(1, n):
        ...     s2[t] = 0.05 + 0.10 * e[t - 1] ** 2 + 0.85 * s2[t - 1]
        ...     e[t] = np.sqrt(s2[t]) * rng.standard_normal()
        ...     y[t] = 0.5 * y[t - 1] + e[t] + 0.4 * e[t - 1]
        >>> res = ARMAGARCH(y, ar_lags=1, ma_lags=1).fit()
        >>> bool(res.is_stationary and res.is_invertible)
        True
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        ar_lags: int = 1,
        ma_lags: int = 1,
        p: int = 1,
        q: int = 1,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(
            endog, vol="GARCH", p=p, o=0, q=q, ar_lags=ar_lags, ma_lags=ma_lags, mean=mean
        )

    def fit(self) -> ARMAGARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood."""
        return ARMAGARCHResult._from_fit(self._fit_family(), self)


class ARMAGJR(_ShortMemoryVarianceModel[ARMAGARCHResult]):
    """ARMA mean with a sign-asymmetric GJR variance, estimated jointly.

    Args:
        endog: The series.
        ar_lags: Conditional-mean autoregressive order.
        ma_lags: Conditional-mean moving-average order.
        p: Order of the shock-magnitude block.
        o: Order of the asymmetry block, at least one.
        q: Order of the variance persistence block.
        mean: ``"constant"`` or ``"zero"``.
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        ar_lags: int = 1,
        ma_lags: int = 1,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(
            endog, vol="GJR", p=p, o=o, q=q, ar_lags=ar_lags, ma_lags=ma_lags, mean=mean
        )

    def fit(self) -> ARMAGARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood."""
        return ARMAGARCHResult._from_fit(self._fit_family(), self)


class ARMAEGARCH(_ShortMemoryVarianceModel[ARMAGARCHResult]):
    """ARMA mean with a log-variance EGARCH process, estimated jointly.

    Args:
        endog: The series.
        ar_lags: Conditional-mean autoregressive order.
        ma_lags: Conditional-mean moving-average order.
        p: Order of the shock-magnitude block.
        o: Order of the asymmetry block, at least one.
        q: Order of the variance persistence block.
        mean: ``"constant"`` or ``"zero"``.
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        ar_lags: int = 1,
        ma_lags: int = 1,
        p: int = 1,
        o: int = 1,
        q: int = 1,
        mean: str = "constant",
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(
            endog, vol="EGARCH", p=p, o=o, q=q, ar_lags=ar_lags, ma_lags=ma_lags, mean=mean
        )

    def fit(self) -> ARMAGARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood."""
        return ARMAGARCHResult._from_fit(self._fit_family(), self)



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


class ARMAFIGARCH(_FractionalVarianceModel[ARMAFIGARCHResult]):
    """ARMA mean with a fractionally integrated variance, estimated jointly.

    The combination the constant-mean :class:`~cultivars.univariate.garch.FIGARCH`
    could not express: short-memory dynamics in the level alongside long-memory
    dynamics in the volatility, which is the usual empirical picture for a
    return series sampled finely enough to show mean reversion.

    Args:
        endog: The series.
        ar_lags: Conditional-mean autoregressive order.
        ma_lags: Conditional-mean moving-average order.
        mean: ``"constant"`` or ``"zero"``.
        truncation: Infinite-order truncation lag for the fractional filter.
    """

    __slots__ = ()

    def __init__(
        self,
        endog: npt.ArrayLike,
        *,
        ar_lags: int = 1,
        ma_lags: int = 1,
        mean: str = "constant",
        truncation: int = 1000,
    ) -> None:
        """Validate the specification and the data."""
        super().__init__(endog, mean=mean, ar_lags=ar_lags, ma_lags=ma_lags, truncation=truncation)

    def fit(self) -> ARMAFIGARCHResult:
        """Estimate mean and variance jointly by Gaussian maximum likelihood."""
        return ARMAFIGARCHResult._from_fit(self._fit_family(), self)
