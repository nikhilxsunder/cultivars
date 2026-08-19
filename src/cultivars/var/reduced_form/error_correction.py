"""Vector error-correction model (VECM) results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, cast

import numpy as np
import numpy.typing as npt

from ..._core import (
    _CHOLESKY_NOTE,
    _LEVELS_TREND,
    _UNSTABLE_NOTE,
    CointegrationTrend,
    SummaryTable,
    deterministic_columns,
)
from ..._internals import (
    _ComparisonMixin,
    _StabilityResult,
    _SummaryMixin,
    _VectorErrorCorrectionFit,
    _VectorErrorCorrectionModel,
    _VectorInferenceMixin,
    _WaldTestResult,
)
from ...exceptions import SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class VECMResult(_SummaryMixin, _ComparisonMixin, _VectorInferenceMixin):
    """A fitted vector error-correction model.

    The result carries two representations of one estimate, and each half of
    the inherited surface reads the one it needs.

    The *short-run* representation is what was actually regressed: differences
    on deterministic terms, lagged differences, and the error-correction terms.
    :attr:`design`, :attr:`resid`, :attr:`fittedvalues`, and everything built on
    the coefficient covariance -- ``params``, ``bse``, ``pvalues``,
    ``conf_int`` -- describe that regression, so a standard error here belongs
    to an ``alpha``, a ``Gamma``, or a deterministic term, conditional on
    ``beta``. The conditioning is what makes them exact rather than
    approximate: ``beta`` converges at rate ``T`` against the usual root-``T``,
    fast enough that everything estimated alongside it behaves as though the
    cointegrating space were known.

    The *levels* representation is the vector autoregression this model
    reparameterizes, recovered by :meth:`to_var` and exposed as
    :attr:`coefficients`. Impulse responses, the variance decomposition, and the
    historical decomposition read it, which is why they are inherited rather
    than rewritten: an error-correction model has no separate impulse-response
    theory, only different coordinates.

    Three members depart from the reduced-form surface deliberately.
    :meth:`stability_check` permits unit roots, because ``k - r`` of them are
    the specification rather than a failure. :meth:`forecast` folds any
    restricted deterministic term back out of the cointegrating space, since a
    constant that lives inside ``beta`` still shifts the level of a forecast.
    :meth:`granger_causality` restricts both transmission channels at once.

    Attributes:
        endog: The sample in levels.
        names: Variable labels, in Cholesky order.
        order: Lags of the levels system; ``order - 1`` lagged differences.
        rank: Cointegrating rank.
        cointegration_trend: The Johansen case the model was estimated under.
        trend: Deterministic terms of the *levels* representation, with any
            restricted term folded back out.
        alpha: ``(k, r)`` adjustment loadings.
        beta: ``(k or k + 1, r)`` cointegrating vectors.
        gamma: ``(p - 1, k, k)`` coefficients on lagged differences.
        coefficients: ``(p, k, k)`` implied levels autoregressive matrices.
        deterministic: ``(d, k)`` deterministic coefficients of the levels
            representation, restricted terms folded in.
        short_run_deterministic: ``(d_s, k)`` unrestricted deterministic terms,
            as they entered the short-run regression.
        eigenvalues: Squared canonical correlations, descending.
        sigma_u: Residual covariance with the degrees-of-freedom correction.
        sigma_ml: Residual covariance divided by the effective sample.
        resid: Residuals of the short-run equation.
        fittedvalues: Fitted differences.
        design: The short-run regressor matrix.
        llf: Gaussian log-likelihood.
        nobs: Effective sample size.
        n_params: Free parameters, covariance included.
    """

    endog: npt.NDArray[np.float64]
    names: tuple[str, ...]
    order: int
    rank: int
    cointegration_trend: str
    trend: str
    alpha: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    gamma: npt.NDArray[np.float64]
    coefficients: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    short_run_deterministic: npt.NDArray[np.float64]
    eigenvalues: npt.NDArray[np.float64]
    sigma_u: npt.NDArray[np.float64]
    sigma_ml: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    design: npt.NDArray[np.float64]
    llf: float
    nobs: int
    n_params: int

    @classmethod
    def _from_fit(
        cls, fit: _VectorErrorCorrectionFit, model: _VectorErrorCorrectionModel[Self]
    ) -> Self:
        """Assemble the public result from the internal fit and its model."""
        case = cast(CointegrationTrend, model.cointegration_trend)
        return cls(
            endog=model.endog,
            names=model.names,
            order=model.order,
            rank=model.rank,
            cointegration_trend=case,
            trend=_LEVELS_TREND[case],
            alpha=fit.alpha,
            beta=fit.beta,
            gamma=fit.gamma,
            coefficients=fit.coefficients,
            deterministic=fit.deterministic,
            short_run_deterministic=fit.short_run_deterministic,
            eigenvalues=fit.eigenvalues,
            sigma_u=fit.sigma_u,
            sigma_ml=fit.sigma_ml,
            resid=fit.resid,
            fittedvalues=fit.fittedvalues,
            design=fit.design,
            llf=fit.llf,
            nobs=fit.nobs,
            n_params=fit.n_params,
        )

    # ------------------------------------------------------------- long run

    @property
    def cointegrating_matrix(self) -> npt.NDArray[np.float64]:
        """The long-run impact matrix ``Pi = alpha beta'``, over the variables only."""
        return self.alpha @ self.beta[: self.k_endog].T

    @property
    def n_common_trends(self) -> int:
        """Unit roots the specification carries, ``k - r``."""
        return self.k_endog - self.rank

    def normalized_beta(self, *, on: int = 0) -> npt.NDArray[np.float64]:
        """Cointegrating vectors scaled so one variable's loading is one in each.

        The eigenvectors come out of the decomposition normalized to make
        ``beta' S11 beta`` the identity, which is convenient for the algebra and
        unreadable as economics. Without further restrictions only the
        cointegrating *space* is identified, so any basis for it is as valid as
        any other; this picks the basis someone would write on a blackboard.

        Args:
            on: Index of the variable whose coefficient is set to one.

        Returns:
            A ``(k or k + 1, r)`` array, empty when the rank is zero.

        Raises:
            SpecificationError: If the index is out of range, or its loading is
                numerically zero in some vector, which means the normalization
                would divide by nothing and another variable should carry it.
        """
        if not 0 <= on < self.k_endog:
            raise SpecificationError(f"on must index a variable, 0..{self.k_endog - 1}; got {on}.")
        if not self.rank:
            return np.zeros((self.beta.shape[0], 0), dtype=np.float64)
        pivots: npt.NDArray[np.float64] = self.beta[on]
        if np.any(np.abs(pivots) < 1e-10):
            raise SpecificationError(
                f"variable {self.names[on]!r} has a numerically zero loading in at least "
                "one cointegrating vector, so it cannot carry the normalization; choose a "
                "variable that enters every relation."
            )
        return np.asarray(self.beta / pivots, dtype=np.float64)

    def normalized_alpha(self, *, on: int = 0) -> npt.NDArray[np.float64]:
        """Adjustment loadings rescaled to pair with :meth:`normalized_beta`.

        Rescaling a cointegrating vector rescales its loading inversely, so the
        product and every test statistic are invariant and only the split
        between the two factors is convention. This returns the loadings that
        go with the readable basis.

        Args:
            on: Index of the variable whose coefficient normalizes each vector.

        Returns:
            A ``(k, r)`` array, empty when the rank is zero.

        Raises:
            SpecificationError: If the index is out of range or cannot carry
                the normalization.
        """
        if not self.rank:
            return np.zeros((self.k_endog, 0), dtype=np.float64)
        return np.asarray(self.alpha * self.beta[on], dtype=np.float64)

    def error_correction_terms(self) -> npt.NDArray[np.float64]:
        """The fitted disequilibria, one column per cointegrating relation."""
        return self.design[:, self.design.shape[1] - self.rank :]

    def to_var(self) -> npt.NDArray[np.float64]:
        """The levels autoregressive matrices this specification implies.

        Returns:
            A ``(p, k, k)`` stack from ``A_1 = I + Pi + Gamma_1``,
            ``A_i = Gamma_i - Gamma_{i-1}``, and ``A_p = -Gamma_{p-1}``. This is
            :attr:`coefficients`, named so that the conversion is discoverable
            from the econometrics rather than only from the attribute list.
        """
        return self.coefficients

    # -------------------------------------------------- short-run coordinates

    def _deterministic_labels(self) -> tuple[str, ...]:
        """Unrestricted deterministic column names of the short-run equation."""
        width = int(self.short_run_deterministic.shape[0])
        return ("const", "trend")[:width]

    def _lag_labels(self) -> tuple[str, ...]:
        """Lagged-difference column names."""
        return tuple(
            f"D.{source}.L{lag + 1}" for lag in range(self.order - 1) for source in self.names
        )

    def _trailing_labels(self) -> tuple[str, ...]:
        """Error-correction column names."""
        return tuple(f"ec{i + 1}" for i in range(self.rank))

    def _trailing_blocks(self) -> tuple[npt.NDArray[np.float64], ...]:
        """The adjustment loadings, laid out as design rows."""
        return (self.alpha.T,) if self.rank else ()

    @property
    def _lag_offset(self) -> int:
        """Short-run deterministic columns ahead of the lagged-difference block."""
        return int(self.short_run_deterministic.shape[0])

    def _coefficient_stack(self) -> npt.NDArray[np.float64]:
        """Short-run coefficients, laid out exactly as the design columns.

        Overridden because the inherited version reads :attr:`coefficients`,
        which here is the *levels* representation and was never regressed. The
        stack has to describe the regression that produced the standard errors,
        not the reparameterization of it.
        """
        blocks = [self.short_run_deterministic] if self.short_run_deterministic.shape[0] else []
        blocks += [self.gamma[i].T for i in range(self.order - 1)]
        blocks += list(self._trailing_blocks())
        return np.vstack(blocks) if blocks else np.zeros((0, self.k_endog), dtype=np.float64)

    # ------------------------------------------------------------- inference

    def stability_check(self) -> _StabilityResult:
        """Assess the levels companion, permitting the unit roots by design.

        A rank-``r`` system in ``k`` variables carries exactly ``k - r`` unit
        roots. Treating those as instability, which the reduced-form check does,
        would flag every correctly specified model in this family.
        """
        return _StabilityResult.assess_stability(self.coefficients, allow_unit_roots=True)

    def forecast(self, steps: int = 1) -> npt.NDArray[np.float64]:
        """Point forecasts in levels.

        Args:
            steps: Horizon.

        Returns:
            A ``(steps, k)`` array of conditional means for the levels.

        Raises:
            SpecificationError: If ``steps`` is not positive.
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        k, p = self.k_endog, self.order
        total = self.endog.shape[0]
        det = deterministic_columns(self.trend, steps, start=total + 1)
        blocks = self.coefficients
        history = [self.endog[total - i - 1] for i in range(p)]
        out = np.empty((steps, k), dtype=np.float64)
        for h in range(steps):
            point = det[h] @ self.deterministic if self.deterministic.shape[0] else np.zeros(k)
            for i in range(p):
                point = point + blocks[i] @ history[i]
            out[h] = point
            history = [point, *history[: p - 1]]
        return out

    def weak_exogeneity(self, variable: str) -> _WaldTestResult:
        """Test that a variable does not adjust to any disequilibrium.

        The null is that the variable's row of ``alpha`` is zero, so it drives
        the long-run relations without responding to them and can be treated as
        weakly exogenous for ``beta``.

        Args:
            variable: One of :attr:`names`.

        Returns:
            A :class:`_WaldTestResult` with ``rank`` degrees of freedom.

        Raises:
            SpecificationError: If the variable is unknown, or the rank is zero
                and there is no adjustment to test.
        """
        if variable not in self.names:
            raise SpecificationError(
                f"unknown variable {variable!r}; expected one of {self.names}."
            )
        if not self.rank:
            raise SpecificationError(
                "a rank-zero model has no cointegrating relations, so there is nothing "
                "for a variable to be weakly exogenous with respect to."
            )
        row = self.names.index(variable)
        first = self.design.shape[1] - self.rank
        cells = [(row, first + j) for j in range(self.rank)]
        return self.coefficient_covariance.wald(
            cells, null=f"{variable} is weakly exogenous for the cointegrating space"
        )

    def granger_causality(self, cause: str, effect: str) -> _WaldTestResult:
        """Test that one variable drives another through neither channel.

        An error-correction model transmits through two routes and a test that
        checks one of them is not a test of Granger causality. The lagged
        differences carry the short-run route; the error-correction term carries
        the long-run route, where ``cause`` moves ``effect`` by shifting a
        disequilibrium that ``effect`` adjusts to. The null restricts both:
        ``Gamma_l[effect, cause] = 0`` for every lag, and the ``(effect, cause)``
        entry of ``alpha beta'`` is zero.

        The second restriction is linear in ``alpha`` once ``beta`` is treated
        as known, which is exactly the conditioning the rest of this result's
        inference already rests on, so the pair goes into one Wald statistic
        rather than two that would have to be combined by hand.

        Args:
            cause: The variable whose influence is restricted.
            effect: The equation the restriction applies to.

        Returns:
            A :class:`_WaldTestResult` with ``order - 1 + (rank > 0)`` degrees
            of freedom.

        Raises:
            SpecificationError: If either name is unknown, they are the same
                variable, or the specification has neither channel to restrict.
        """
        names = self.names
        for label, value in (("cause", cause), ("effect", effect)):
            if value not in names:
                raise SpecificationError(f"{label} {value!r} is not one of {names}.")
        if cause == effect:
            raise SpecificationError("a variable cannot Granger-cause itself.")
        source, target = names.index(cause), names.index(effect)
        k, lags = self.k_endog, self.order - 1
        width = self.design.shape[1]
        if not lags and not self.rank:
            raise SpecificationError(
                "this specification has no lagged differences and no cointegrating "
                "relations, so there is no channel through which one variable could "
                "drive another."
            )
        rows: list[npt.NDArray[np.float64]] = []
        offset = self._lag_offset
        for lag in range(lags):
            restriction = np.zeros((k, width), dtype=np.float64)
            restriction[target, offset + lag * k + source] = 1.0
            rows.append(restriction.ravel())
        if self.rank:
            restriction = np.zeros((k, width), dtype=np.float64)
            first = width - self.rank
            restriction[target, first : first + self.rank] = self.beta[source]
            rows.append(restriction.ravel())
        return self.coefficient_covariance.wald_restriction(
            np.vstack(rows), null=f"{cause} does not Granger-cause {effect}"
        )

    # --------------------------------------------------------------- display

    def _comparison_label(self) -> str:
        """Short specification label for a ranking table."""
        return f"VECM({self.order}, r={self.rank}, {self.cointegration_trend})"

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        criteria = self.information_criteria
        stability = self.stability_check()
        notes = [
            f"Common trends: {self.n_common_trends} of {self.k_endog}   "
            f"max |companion root| = {stability.max_modulus:.4f}",
            "Coefficients and their standard errors describe the short-run regression of "
            "differences on deterministic terms, lagged differences, and the error-"
            "correction terms; they are conditional on beta, which converges fast enough "
            "to be treated as known.",
            "Only the cointegrating space is identified without further restrictions, so "
            "read beta through normalized_beta() and treat any single vector as one basis "
            "among many.",
            "The ec rows are the loadings that pair with beta as the decomposition "
            "normalizes it, not with the basis printed below; rescaling beta rescales "
            "alpha inversely, so the product and every z-statistic are unchanged. "
            "normalized_alpha() returns the loadings on the printed basis.",
            _CHOLESKY_NOTE,
        ]
        if stability.max_modulus > 1.0 + 1e-8:
            notes.insert(0, _UNSTABLE_NOTE)
        rows = self._coefficient_rows()
        if self.rank:
            beta_rows = self.normalized_beta()
            labels = (
                *self.names,
                "const" if self.cointegration_trend == "restricted_constant" else "trend",
            )[: beta_rows.shape[0]]
            rows = rows + tuple(
                (f"beta[{j + 1}]: {label}", f"{beta_rows[i, j]:.4f}", "", "", "", "", "")
                for j in range(self.rank)
                for i, label in enumerate(labels)
            )
        return SummaryTable(
            title=f"VECM({self.order}) Results",
            metadata=(
                ("Model", f"VECM({self.order}, r={self.rank})"),
                ("Log-likelihood", f"{self.llf:.3f}"),
                ("Variables", f"{self.k_endog}"),
                ("AIC", f"{criteria.aic:.3f}"),
                ("Rank", f"{self.rank}"),
                ("BIC", f"{criteria.bic:.3f}"),
                ("Deterministic", self.cointegration_trend),
                ("HQIC", f"{criteria.hqic:.3f}"),
                ("Observations", f"{self.nobs}"),
                ("Common trends", f"{self.n_common_trends}"),
            ),
            columns=self._coefficient_columns(),
            rows=rows,
            notes=tuple(notes),
        )


class VECM(_VectorErrorCorrectionModel[VECMResult]):
    """Vector error-correction model, estimated by Johansen's reduced-rank ML.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> common = np.cumsum(rng.standard_normal(300))
        >>> y = np.column_stack([common, 2 * common + rng.standard_normal(300)])
        >>> res = VECM(y, order=2, rank=1).fit()
        >>> res.n_common_trends
        1
    """

    __slots__ = ()

    def fit(self) -> VECMResult:
        """Estimate the system at the specified rank and return the result."""
        return VECMResult._from_fit(self._fit_family(), self)
