# filepath: /src/cultivars/_internals/_mixins.py
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

"""Behavior mixins: optional capabilities a public result opts into.

Every class here declares ``__slots__ = ()`` and no fields of its own. A mixin
annotates the attributes it reads and the concrete result declares them, which
is what lets several mixins compose onto one frozen slotted dataclass without
the ``multiple bases have instance lay-out conflict`` that field-carrying bases
would cause.

The split is by capability rather than by family, so a result declares what it
can do by what it inherits: stationarity assessment for anything with an
autoregressive polynomial, a volatility surface for anything with a variance
path, rendering for anything that can describe its own summary, frame interop
for anything with aligned per-observation output, and criterion-named
comparison for anything reporting a likelihood.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from .._core import (
    _NO_CLOSED_SYSTEM,
    InformationCriteria,
    SummaryTable,
    companion_matrix,
    deterministic_columns,
    to_pandas_frame,
    to_polars_frame,
)
from ..exceptions import DimensionError, NumericalError, SpecificationError
from ._covariances import _CoefficientCovariance
from ._inferences import _CoefficientInference
from ._results import _LikelihoodRatioResult, _StabilityResult, _WaldTestResult

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


class _StationarityMixin:
    """Stationarity assessment over an autoregressive polynomial.

    The concrete result declares ``ar_params``. Results whose stationarity is
    governed by a *composed* polynomial -- seasonal times non-seasonal AR in
    SARIMA -- override :meth:`_stationarity_ar` to return that expansion;
    assessing the raw non-seasonal block alone would silently pass a model with
    an explosive seasonal root.

    Not applied to Markov-switching results: their ``ar_params`` is ``(K, p)``
    and no single companion polynomial describes the process.
    """

    __slots__ = ()

    ar_params: npt.NDArray[np.float64]

    def _stationarity_ar(self) -> npt.NDArray[np.float64]:
        """The polynomial whose roots determine stationarity."""
        return self.ar_params

    @property
    def stability(self) -> _StabilityResult:
        """Full eigenvalue verdict for the autoregressive polynomial."""
        return _StabilityResult.assess_stability(self._stationarity_ar())

    @property
    def is_stationary(self) -> bool:
        """Whether every companion eigenvalue lies inside the unit circle."""
        return self.stability.is_stable


class _ConditionalVarianceMixin:
    """Volatility surface for results carrying a conditional-variance path.

    Reads ``conditional_variance``, which is deliberately a different name from
    the scalar ``sigma2`` on homoskedastic mean models: one is a path of shape
    ``(nobs,)``, the other a single float, and overloading one name across both
    shapes invites silent shape bugs downstream.
    """

    __slots__ = ()

    conditional_variance: npt.NDArray[np.float64]

    @property
    def conditional_volatility(self) -> npt.NDArray[np.float64]:
        """Conditional standard deviation, ``sqrt(conditional_variance)``."""
        return np.sqrt(self.conditional_variance)


class _SummaryMixin:
    """Text, notebook, and dataframe rendering for a fitted result.

    The concrete result supplies :meth:`_summary_table`; everything a user
    touches -- ``print(res)``, a bare ``res`` in a notebook cell,
    ``res.summary().to_pandas()`` -- is derived from that one method, so a
    family adds display by describing its own summary rather than by writing
    four renderers.

    Results that mix this in must be declared ``@dataclass(..., repr=False)``.
    A dataclass writes ``__repr__`` onto the class itself, and a method defined
    on the class always wins over one inherited from a base, so the generated
    field dump would silently shadow the summary.
    """

    __slots__ = ()

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary for this result.

        Returns:
            The :class:`SummaryTable` every renderer draws from.

        Raises:
            NotImplementedError: If the concrete result does not supply one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _summary_table() to be displayable."
        )

    def summary(self) -> SummaryTable:
        """The estimation summary, renderable as text, HTML, or a dataframe."""
        return self._summary_table()

    def __repr__(self) -> str:
        """Render the summary, so a bare result in a REPL is readable."""
        return self._summary_table().to_text()

    def __str__(self) -> str:
        """Render the summary."""
        return self._summary_table().to_text()

    def _repr_html_(self) -> str:
        """Render the summary as HTML for Jupyter."""
        return self._summary_table()._repr_html_()


class _SeriesMixin:
    """Dataframe interop for the per-observation outputs of a fit.

    Exposes the aligned series a user actually wants to plot or join --
    observed, fitted, residual -- as a dict of arrays, and converts that same
    mapping to pandas or polars on demand. Families with extra aligned series,
    such as a conditional-variance path, extend :meth:`_series` rather than
    reimplementing the converters.
    """

    __slots__ = ()

    endog: npt.NDArray[np.float64]
    fittedvalues: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]

    def _series(self) -> dict[str, npt.NDArray[np.float64]]:
        """The aligned per-observation outputs, keyed by column name.

        Returns:
            A mapping whose values all share a length. The observed series is
            trimmed to the effective sample so every column aligns.
        """
        fitted = self.fittedvalues
        return {
            "observed": self.endog[self.endog.shape[0] - fitted.shape[0] :],
            "fitted": fitted,
            "resid": self.resid,
        }

    def to_dict(self) -> dict[str, npt.NDArray[np.float64]]:
        """The aligned per-observation outputs as plain numpy arrays."""
        return self._series()

    def to_pandas(self, *, index: npt.ArrayLike | None = None) -> pd.DataFrame:
        """The aligned per-observation outputs as a :class:`pandas.DataFrame`.

        Args:
            index: Optional row index, typically the observation dates for the
                effective sample.

        Returns:
            One column per aligned series.

        Raises:
            ImportError: If pandas is not installed.
        """
        return to_pandas_frame(self._series(), index=index, index_name="obs")

    def to_polars(self) -> pl.DataFrame:
        """The aligned per-observation outputs as a :class:`polars.DataFrame`.

        Returns:
            One column per aligned series.

        Raises:
            ImportError: If polars is not installed.
        """
        return to_polars_frame(self._series())


class _ComparisonMixin:
    """Explicit, criterion-named comparison between fitted results.

    Deliberately not ordering dunders. ``a < b`` would have to silently pick a
    criterion, and AIC and BIC routinely disagree on the same pair of models,
    so an ordering operator would encode an arbitrary choice in syntax that
    reads like a fact. Comparison here always names its criterion.
    """

    __slots__ = ()

    llf: float
    nobs: int
    n_params: int

    def _comparison_label(self) -> str:
        """Short specification label for this result in a comparison table.

        Defaults to the class name, which is only distinguishing when the
        models being compared are of different families. Any result whose
        family spans several specifications -- an order, a lag count -- should
        override this so a ranking table is readable.
        """
        return type(self).__name__

    @property
    def information_criteria(self) -> InformationCriteria:
        """AIC, BIC, and HQIC derived from the likelihood and parameter count."""
        return InformationCriteria.from_likelihood(self.llf, self.nobs, self.n_params)

    def compare(self, *others: _ComparisonMixin, criterion: str = "aic") -> SummaryTable:
        """Rank this result against others by an information criterion.

        Args:
            others: Further fitted results, estimated on the same sample.
            criterion: ``"aic"``, ``"bic"``, or ``"hqic"``.

        Returns:
            A :class:`SummaryTable` ordered best-first, with the gap to the
            best model in the final column.

        Raises:
            SpecificationError: If ``criterion`` is not recognized.
            DimensionError: If the results were not fitted to the same number
                of observations, which would make the criteria incomparable.
        """
        if criterion not in ("aic", "bic", "hqic"):
            raise SpecificationError(
                f"criterion must be one of 'aic', 'bic', 'hqic'; got {criterion!r}."
            )
        models = (self, *others)
        sizes = {model.nobs for model in models}
        if len(sizes) > 1:
            raise DimensionError(
                f"information criteria are only comparable across a common sample; "
                f"got differing observation counts {sorted(sizes)}."
            )
        scored = sorted(
            ((getattr(model.information_criteria, criterion), model) for model in models),
            key=lambda pair: pair[0],
        )
        best = scored[0][0]
        return SummaryTable(
            title=f"Model comparison by {criterion.upper()}",
            metadata=(("Observations", f"{self.nobs}"), ("Models", f"{len(models)}")),
            columns=("model", "llf", "k", criterion.upper(), f"d{criterion.upper()}"),
            rows=tuple(
                (
                    model._comparison_label(),
                    f"{model.llf:.3f}",
                    f"{model.n_params}",
                    f"{value:.3f}",
                    f"{value - best:.3f}",
                )
                for value, model in scored
            ),
            notes=("Ranked best-first; the final column is the gap to the best model.",),
        )

    def likelihood_ratio_test(self, unrestricted: _ComparisonMixin) -> _LikelihoodRatioResult:
        """Test this result as the restricted model against a nesting one.

        Args:
            unrestricted: The nesting model, with at least as many parameters.

        Returns:
            The :class:`_LikelihoodRatioResult` verdict.

        Raises:
            SpecificationError: If ``unrestricted`` does not have strictly more
                parameters, or its likelihood is lower -- either indicates the
                two models are not nested the way the call assumes.
            DimensionError: If the two were fitted to different samples.
        """
        if self.nobs != unrestricted.nobs:
            raise DimensionError(
                f"a likelihood-ratio test requires a common sample; got {self.nobs} "
                f"and {unrestricted.nobs} observations."
            )
        df = unrestricted.n_params - self.n_params
        if df <= 0:
            raise SpecificationError(
                f"the unrestricted model must have more parameters; got {unrestricted.n_params} "
                f"against {self.n_params}."
            )
        statistic = 2.0 * (unrestricted.llf - self.llf)
        if statistic < 0.0:
            raise SpecificationError(
                "the restricted model attained a higher likelihood than the unrestricted "
                "one; the models are probably not nested, or one has not converged."
            )
        return _LikelihoodRatioResult(
            statistic=statistic,
            df=df,
            pvalue=float(chi2.sf(statistic, df)),
        )


class _InvertibilityMixin:
    """Invertibility assessment over a moving-average polynomial.

    The concrete result declares ``ma_params`` in the observation-equation sign
    convention, so the same companion-eigenvalue test that decides
    stationarity for an autoregressive polynomial decides invertibility here.
    Results whose invertibility is governed by a *composed* polynomial --
    seasonal times non-seasonal MA in SARIMA -- override
    :meth:`_invertibility_ma` to return that expansion.

    Note the sign. The fits store ``ma_params`` as the coefficients of
    ``theta(L) = 1 + theta_1 L + ...``, while
    :meth:`_StabilityResult.assess_stability` tests
    ``1 - c_1 z - c_2 z^2 - ...``. The argument is therefore negated. Getting
    that wrong is not a cosmetic error: across 500 random draws from the
    reparameterized region, passing ``ma_params`` unnegated reports only 181
    as invertible, so two thirds of correctly-invertible fits would be flagged.

    For any estimator that reparameterizes the MA block through the partial
    autocorrelations, invertibility then holds by construction -- 500 of 500 in
    the same check -- so this property verifies the transform rather than
    describing the data. It is still worth exposing: a ``False`` here means the
    reparameterization is broken.
    """

    __slots__ = ()

    ma_params: npt.NDArray[np.float64]

    def _invertibility_ma(self) -> npt.NDArray[np.float64]:
        """The MA block in the sign convention the companion test expects."""
        return -self.ma_params

    @property
    def invertibility(self) -> _StabilityResult:
        """Full eigenvalue verdict for the moving-average polynomial."""
        return _StabilityResult.assess_stability(self._invertibility_ma())

    @property
    def is_invertible(self) -> bool:
        """Whether every companion eigenvalue of the MA block is inside the unit circle."""
        return self.invertibility.is_stable


class _VectorInferenceMixin:
    """Post-estimation inference shared by every reduced-form vector model.

    Seven of the nine operations a usable VAR must expose -- forecasting,
    impulse responses, forecast-error variance decomposition, historical
    decomposition, Granger causality, stability, and residual diagnostics --
    are functions of the fitted coefficient stack, the innovation covariance,
    and the residuals. Not one of them asks how those were produced. A VARX
    differs from a VAR only in its regressor block, a VECM only in how the
    coefficients are restricted, a panel VAR only in how observations are
    pooled; all three land on the same three arrays and want the same seven
    answers. Writing them here once is what keeps the rest of the family from
    being seven reimplementations of Lutkepohl chapter 2.

    Concrete results declare the attributes below. ``deterministic`` and
    ``design`` are needed because two of the seven -- forecasting and Granger
    causality -- reach past the coefficient stack: a forecast must extrapolate
    the deterministic terms, and a Wald test needs the regressor cross-product.

    Attributes:
        endog: The full observed panel, shape ``(n, k)``.
        names: Variable names in column order; every method that reports per
            variable takes and returns these rather than integer positions.
        order: Autoregressive order ``p``.
        trend: Deterministic specification, ``"n"``, ``"c"`` or ``"ct"``.
        coefficients: Autoregressive matrices ``A_1, ..., A_p``, shape
            ``(p, k, k)``.
        deterministic: Coefficients on the deterministic block, shape
            ``(d, k)``; zero rows when ``trend == "n"``.
        sigma_u: Degrees-of-freedom-adjusted innovation covariance. Used for
            every *inferential* quantity -- Wald tests, the impact matrix --
            because the maximum-likelihood estimator is biased downward by the
            estimated coefficients.
        sigma_ml: The ``1/T`` estimator. Used for the likelihood and the
            information criteria, where the adjustment does not belong.
            Both are stored deliberately: substituting one for the other
            biases either the criteria or the standard errors, and nothing
            raises when it happens.
        resid: Residuals over the effective sample, shape ``(nobs, k)``.
        design: The regressor matrix the fit solved against, shape
            ``(nobs, d + k * p)``.
        nobs: Effective sample size.
    """

    __slots__ = ()

    endog: npt.NDArray[np.float64]
    names: tuple[str, ...]
    order: int
    trend: str
    coefficients: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    sigma_u: npt.NDArray[np.float64]
    sigma_ml: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    design: npt.NDArray[np.float64]
    nobs: int
    posterior: _CoefficientInference | None

    # -- structure ---------------------------------------------------------

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables."""
        return len(self.names)

    @property
    def _lag_offset(self) -> int:
        """Design columns that sit ahead of the endogenous lag block.

        The lag block always begins immediately after the deterministic block,
        so the deterministic coefficient matrix already knows this number and
        nothing needs to recount it. Any regressor a family appends -- an
        exogenous distributed lag, an instrument set -- goes behind the lags
        precisely so that this stays true and the unpacking offsets in
        ``_fit_family`` keep working unchanged.
        """
        return int(self.deterministic.shape[0])

    def _trailing_blocks(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Coefficient blocks a family appends after the endogenous lags.

        Returned in design order and shaped ``(rows, k_endog)``, matching the
        layout of :attr:`deterministic`. The default is empty, which is the
        statement that this family's design is deterministic terms and lags and
        nothing else.
        """
        return ()

    @property
    def _sample_blocks(self) -> tuple[tuple[int, int], ...]:
        """Half-open row spans of :attr:`resid`, one per independent series.

        A single vector autoregression has one span covering the whole residual
        matrix, and every method that reads residual history reduces to its
        previous form when that is so. A panel has one span per unit, and the
        distinction is not cosmetic: a lagged cross-product taken across a unit
        boundary multiplies one country's last observation by another country's
        first, which is not an autocovariance of anything.
        """
        return ((0, self.nobs),)

    def _impact(self) -> npt.NDArray[np.float64]:
        """Lower-triangular Cholesky factor of the innovation covariance.

        Returns:
            The matrix ``P`` with ``P P' = sigma_u``, which is the *recursive*
            structural impact matrix. Overridden by a structural result that
            identifies the impact some other way.

        Raises:
            NumericalError: If the innovation covariance is not positive
                definite, which happens when the system is singular -- a
                variable that is an exact linear combination of the others.
        """
        try:
            return np.linalg.cholesky(self.sigma_u)
        except np.linalg.LinAlgError as exc:
            raise NumericalError(
                "the innovation covariance is not positive definite, so no impact "
                "matrix exists; one endogenous variable is likely collinear with "
                "the others."
            ) from exc

    # -- dynamics ----------------------------------------------------------

    # -- shock accounting --------------------------------------------------

    def structural_shocks(self) -> npt.NDArray[np.float64]:
        """Orthogonalized innovations ``P^{-1} u_t``, shape ``(nobs, k)``.

        Unit variance and mutually uncorrelated by construction, under the same
        recursive ordering :meth:`irf` documents.
        """
        return np.linalg.solve(self._impact(), self.resid.T).T

    # -- tests -------------------------------------------------------------

    def _coefficient_stack(self) -> npt.NDArray[np.float64]:
        """Coefficients laid out as the fit solved them: ``(d + kp, k)``."""
        blocks: list[npt.NDArray[np.float64]] = []
        if self.deterministic.shape[0]:
            blocks.append(self.deterministic)
        blocks.extend(self.coefficients[i].T for i in range(self.order))
        return np.vstack(blocks) if blocks else np.zeros((0, self.k_endog), dtype=np.float64)

    def _deterministic_labels(self) -> tuple[str, ...]:
        """Names of the leading deterministic design columns, in design order."""
        return (("const",) if self.trend in ("c", "ct") else ()) + (
            ("trend",) if self.trend == "ct" else ()
        )

    def _lag_labels(self) -> tuple[str, ...]:
        """Names of the endogenous lag columns, in design order."""
        return tuple(f"{source}.L{lag + 1}" for lag in range(self.order) for source in self.names)

    def _trailing_labels(self) -> tuple[str, ...]:
        """Names of the columns a family appends after the endogenous lags."""
        return ()

    def _regressor_labels(self) -> tuple[str, ...]:
        """Every design column name, in design order.

        The label counterpart of :meth:`_coefficient_stack`, and the reason the
        two must be built from the same three pieces: every coefficient the user
        can name -- an estimate, a standard error, a p-value, a bound -- is
        produced by zipping these labels against that stack. Deriving them
        separately is how a table ends up with the right numbers under the wrong
        headings, which is worse than having no table.
        """
        return self._deterministic_labels() + self._lag_labels() + self._trailing_labels()

    @property
    def coefficient_covariance(self) -> _CoefficientInference:
        """Covariance of the coefficient matrix, sampling or posterior.

        Returns whichever object describes how the numbers were produced. A
        shrunk fit carries its posterior covariance from estimation, because it
        cannot be reconstructed here: under a prior the joint covariance stops
        factoring as ``kron(sigma_u, inv(X'X))`` and there is nothing to
        rebuild it from. Both objects answer the same questions, which is why
        everything downstream -- ``bse``, ``tvalues``, ``conf_int``, the
        coefficient table -- reads one interface and never asks which it has.
        """
        if self.posterior is not None:
            return self.posterior
        return _CoefficientCovariance(
            coefficients=self._coefficient_stack(),
            sigma_u=self.sigma_u,
            xtx_inv=np.linalg.inv(self.design.T @ self.design),
        )

    def _labelled(self, values: npt.NDArray[np.float64]) -> dict[str, float]:
        """Key a ``(width, k)`` array by ``"{equation}: {regressor}"``.

        Args:
            values: An array laid out like :meth:`_coefficient_stack`.

        Returns:
            One entry per coefficient, in equation-major order.

        Raises:
            DimensionError: If the array does not match the design.
        """
        labels = self._regressor_labels()
        if values.shape != (len(labels), self.k_endog):
            raise DimensionError(
                f"expected an array of shape {(len(labels), self.k_endog)} to label; "
                f"got {values.shape}."
            )
        return {
            f"{equation}: {label}": float(values[j, i])
            for i, equation in enumerate(self.names)
            for j, label in enumerate(labels)
        }

    def equation(self, name: str) -> dict[str, float]:
        """The coefficients of one equation, keyed by regressor.

        Args:
            name: One of :attr:`names`.

        Returns:
            Coefficients in design order: deterministic terms, then endogenous
            lags, then whatever the family appends.

        Raises:
            SpecificationError: If the variable is unknown.
        """
        if name not in self.names:
            raise SpecificationError(f"unknown variable {name!r}; expected one of {self.names}.")
        column = self.names.index(name)
        stack = self._coefficient_stack()
        return {label: float(stack[j, column]) for j, label in enumerate(self._regressor_labels())}

    @property
    def params(self) -> dict[str, float]:
        """Every coefficient, keyed ``"{equation}: {regressor}"``."""
        return self._labelled(self._coefficient_stack())

    @property
    def bse(self) -> dict[str, float]:
        """Standard errors, keyed identically to :attr:`params`."""
        return self._labelled(self.coefficient_covariance.stderr)

    @property
    def tvalues(self) -> dict[str, float]:
        """Normal test statistics against zero, keyed identically to :attr:`params`."""
        return self._labelled(self.coefficient_covariance.tstat)

    @property
    def pvalues(self) -> dict[str, float]:
        """Two-sided p-values against zero, keyed identically to :attr:`params`."""
        return self._labelled(self.coefficient_covariance.pvalue)

    def conf_int(self, *, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
        """Confidence bounds for every coefficient.

        Args:
            alpha: Two-sided level; ``0.05`` gives a 95 percent interval.

        Returns:
            One ``(lower, upper)`` pair per coefficient, keyed identically to
            :attr:`params`.
        """
        lower, upper = self.coefficient_covariance.conf_int(alpha=alpha)
        low, high = self._labelled(lower), self._labelled(upper)
        return {name: (value, high[name]) for name, value in low.items()}

    def _coefficient_rows(self, *, alpha: float = 0.05) -> tuple[tuple[str, ...], ...]:
        """Rows of the coefficient table, shaped by what produced the numbers.

        A shrunk fit drops the p-value column rather than renaming it. There is
        no honest name for a normal tail area centred on a posterior mean: it
        answers how far zero sits from that mean in posterior standard
        deviations, which the ratio column already says, and restating it in
        probability units invites the frequentist reading that shrinkage makes
        wrong.
        """
        inference = self.coefficient_covariance
        stack = self._coefficient_stack()
        stderr, tstat = inference.stderr, inference.tstat
        lower, upper = inference.conf_int(alpha=alpha)
        labels = self._regressor_labels()
        shrunk = inference._IS_POSTERIOR
        pvalue = None if shrunk else inference.pvalue
        return tuple(
            (
                f"{equation}: {label}",
                f"{stack[j, i]:.4f}",
                f"{stderr[j, i]:.4f}",
                f"{tstat[j, i]:.3f}",
                *(() if pvalue is None else (f"{pvalue[j, i]:.3f}",)),
                f"{lower[j, i]:.4f}",
                f"{upper[j, i]:.4f}",
            )
            for i, equation in enumerate(self.names)
            for j, label in enumerate(labels)
        )

    def _coefficient_columns(self, *, alpha: float = 0.05) -> tuple[str, ...]:
        """Headings for the coefficient table, with the bounds named honestly.

        The interval headings are derived from the same ``alpha`` that produced
        the bounds rather than written down beside them. A heading that says
        ``0.025`` above a 90 percent bound is not a cosmetic problem: it is the
        table asserting something false about its own contents.

        Under a prior the headings change with them. ``std err`` becomes a
        posterior standard deviation, ``z`` becomes a plain ratio because it is
        no longer a test statistic, the p-value column is absent, and the
        bounds are a credible interval rather than a confidence interval. A
        table that kept the frequentist headings would be asserting something
        false about its own contents, which is the same defect as printing
        ``0.025`` above a ninety percent bound.

        Args:
            alpha: Two-sided level used for the interval.

        Returns:
            One heading per column of :meth:`_coefficient_rows`.
        """
        edge = (f"[{alpha / 2:g}", f"{1 - alpha / 2:g}]")
        if self.coefficient_covariance._IS_POSTERIOR:
            return ("", "coef", "post. sd", "m/sd", *edge)
        return ("", "coef", "std err", "z", "P>|z|", *edge)

    def granger_causality(self, cause: str, effect: str) -> _WaldTestResult:
        """Wald test that one variable's lags are jointly zero in another equation.

        Args:
            cause: The variable whose lags are restricted.
            effect: The equation the restriction is applied to.

        Returns:
            A :class:`_WaldTestResult` naming the non-causality null.

        Raises:
            SpecificationError: If either name is unknown, they are the same
                variable, or the model has no lags to restrict.
        """
        names = self.names
        for label, value in (("cause", cause), ("effect", effect)):
            if value not in names:
                raise SpecificationError(f"{label} {value!r} is not one of {names}.")
        if cause == effect:
            raise SpecificationError("a variable cannot Granger-cause itself.")
        if self.order == 0:
            raise SpecificationError("a VAR(0) has no lags to test.")
        source, target = names.index(cause), names.index(effect)
        k, offset = self.k_endog, self._lag_offset
        cells = [(target, offset + lag * k + source) for lag in range(self.order)]
        return self.coefficient_covariance.wald(
            cells, null=f"{cause} does not Granger-cause {effect}"
        )

    def portmanteau_test(self, lags: int = 10) -> _WaldTestResult:
        """Multivariate Ljung-Box test for residual autocorrelation.

        Args:
            lags: Number of residual autocovariances to include. Must exceed
                the fitted order comfortably, or the fitted parameters consume
                every degree of freedom.

        Returns:
            The :class:`_WaldTestResult`. Rejection means the lag order is too
            short, so every other quantity here -- impulse responses included
            -- is computed from a misspecified system.

        Raises:
            SpecificationError: If ``lags`` is outside ``1..nobs-1``, or leaves
                no degrees of freedom after the fitted parameters.
        """
        residuals = self.resid
        n, k = residuals.shape
        if not 1 <= lags < n:
            raise SpecificationError(f"lags must be in 1..{n - 1}; got {lags}.")
        gamma0 = residuals.T @ residuals / n
        inverse = np.linalg.inv(gamma0)
        statistic = 0.0
        for i in range(1, lags + 1):
            gamma = residuals[i:].T @ residuals[:-i] / n
            statistic += float(np.trace(gamma.T @ inverse @ gamma @ inverse)) / (n - i)
        statistic *= n * n
        df = k * k * lags - k * self.design.shape[1]
        if df <= 0:
            raise SpecificationError(
                f"{lags} lags leaves {df} degrees of freedom against "
                f"{self.design.shape[1]} fitted parameters per equation; use more lags."
            )
        return _WaldTestResult(
            statistic=statistic,
            df=df,
            pvalue=float(chi2.sf(statistic, df)),
            null=f"residual autocorrelation is zero through lag {lags}",
        )

    def normality_test(self, variable: str) -> _WaldTestResult:
        """Jarque-Bera test on one equation's residuals.

        Args:
            variable: Name of the equation to test.

        Returns:
            The :class:`_WaldTestResult` with two degrees of freedom.

        Raises:
            SpecificationError: If ``variable`` is not in ``names``.

        Note:
            Non-normality does not invalidate the coefficient estimates, which
            are consistent regardless. It bites on anything that reads the
            Gaussian likelihood at face value -- the information criteria, and
            any bootstrap that resamples as though the innovations were
            symmetric.
        """
        if variable not in self.names:
            raise SpecificationError(
                f"unknown variable {variable!r}; expected one of {self.names}."
            )
        column = self.resid[:, self.names.index(variable)]
        n = column.shape[0]
        standardized = (column - column.mean()) / column.std()
        skew = float((standardized**3).mean())
        kurtosis = float((standardized**4).mean())
        statistic = n * (skew**2 / 6.0 + (kurtosis - 3.0) ** 2 / 24.0)
        return _WaldTestResult(
            statistic=float(statistic),
            df=2,
            pvalue=float(chi2.sf(statistic, 2)),
            null=f"{variable} residuals are Gaussian",
        )

    def arch_test(self, variable: str, lags: int = 5) -> _WaldTestResult:
        """Engle's LM test for conditional heteroskedasticity in one equation.

        Args:
            variable: Name of the equation to test.
            lags: Number of squared-residual lags in the auxiliary regression.

        Returns:
            The :class:`_WaldTestResult` with ``lags`` degrees of freedom.

        Raises:
            SpecificationError: If ``variable`` is unknown or ``lags`` is
                outside ``1..nobs-1``.

        Note:
            Rejection leaves the coefficients consistent but the reported
            covariance wrong, so Granger tests become unreliable while impulse
            responses stay interpretable as conditional means.
        """
        if variable not in self.names:
            raise SpecificationError(
                f"unknown variable {variable!r}; expected one of {self.names}."
            )
        squared = self.resid[:, self.names.index(variable)] ** 2
        n = squared.shape[0]
        if not 1 <= lags < n:
            raise SpecificationError(f"lags must be in 1..{n - 1}; got {lags}.")
        design = np.column_stack(
            [np.ones(n - lags), *(squared[lags - i : n - i] for i in range(1, lags + 1))]
        )
        target = squared[lags:]
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = target - design @ coefficients
        centred = target - target.mean()
        total = float(centred @ centred)
        r_squared = 0.0 if total <= 0.0 else 1.0 - float(residual @ residual) / total
        statistic = (n - lags) * r_squared
        return _WaldTestResult(
            statistic=float(statistic),
            df=lags,
            pvalue=float(chi2.sf(statistic, lags)),
            null=f"no ARCH effect in {variable} residuals through lag {lags}",
        )

    def residual_diagnostics(
        self, *, portmanteau_lags: int = 10, arch_lags: int = 5
    ) -> SummaryTable:
        """Run every residual test and collect them into one table.

        Args:
            portmanteau_lags: Lags for the multivariate autocorrelation test.
            arch_lags: Lags for the per-equation ARCH test.

        Returns:
            A :class:`SummaryTable` with one row per test.
        """
        tests: list[tuple[str, _WaldTestResult]] = [
            (f"portmanteau({portmanteau_lags})", self.portmanteau_test(portmanteau_lags)),
            *((f"jarque-bera[{n}]", self.normality_test(n)) for n in self.names),
            *((f"arch-lm({arch_lags})[{n}]", self.arch_test(n, arch_lags)) for n in self.names),
        ]
        return SummaryTable(
            title="Residual diagnostics",
            metadata=(("Observations", f"{self.nobs}"), ("Equations", f"{self.k_endog}")),
            columns=("test", "statistic", "df", "p-value"),
            rows=tuple(
                (label, f"{t.statistic:.3f}", f"{t.df}", f"{t.pvalue:.4f}") for label, t in tests
            ),
            notes=(
                "Small p-values reject the null named by each test. A rejected "
                "portmanteau means the lag order is too short, which invalidates "
                "every other quantity the result reports.",
            ),
        )


class _VectorPropagationMixin:
    """How a shock travels through a closed vector system.

    The half of the reduced-form surface that needs a law of motion for every
    variable in the model: the companion matrix and its roots, the
    moving-average representation, the two impulse responses, the variance
    decomposition, the historical decomposition, and the forecast. All of them
    read :attr:`coefficients`, and all of them are meaningless without it.

    Split from :class:`_VectorInferenceMixin` because that line is real rather
    than tidy. A conditional model -- a VECMX, a single unit of a global
    autoregression before the units are linked -- estimates its coefficients
    honestly and can report every standard error, p-value and residual
    diagnostic the inference mixin offers, while having no closed system at all.
    Before the split those families inherited nine methods they had to override
    one by one with refusals, which is a list that drifts the moment a tenth is
    added. Now they simply do not mix this in, and
    :class:`_ConditionalSystemMixin` keeps the error messages good without
    claiming the type.

    Attributes:
        coefficients: ``(p, k, k)`` autoregressive matrices of the closed
            system. A family that reparameterizes -- an error-correction model
            -- supplies the levels representation here, which is why its
            impulse responses need no separate implementation.
    """

    __slots__ = ()

    coefficients: npt.NDArray[np.float64]
    deterministic: npt.NDArray[np.float64]
    endog: npt.NDArray[np.float64]
    names: tuple[str, ...]
    order: int
    trend: str
    sigma_u: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    nobs: int

    _PROPAGATION_MEMBERS: ClassVar[frozenset[str]] = frozenset(
        {
            "companion",
            "fevd",
            "generalized_fevd",
            "forecast",
            "generalized_irf",
            "historical_decomposition",
            "irf",
            "is_stable",
            "ma_representation",
            "stability_check",
        }
    )

    @property
    def k_endog(self) -> int:
        """Number of endogenous variables."""
        return len(self.names)

    def _impact(self) -> npt.NDArray[np.float64]:
        """The Cholesky impact matrix."""
        return np.linalg.cholesky(self.sigma_u)

    def structural_shocks(self) -> npt.NDArray[np.float64]:
        """Residuals rotated by the inverse Cholesky impact matrix."""
        return np.linalg.solve(self._impact(), self.resid.T).T

    @property
    def _sample_blocks(self) -> tuple[tuple[int, int], ...]:
        """Half-open residual row spans, one per independent series."""
        return ((0, self.nobs),)

    @property
    def companion(self) -> npt.NDArray[np.float64]:
        """The ``(kp, kp)`` companion matrix of the autoregressive block."""
        return companion_matrix(self.coefficients)

    def stability_check(self) -> _StabilityResult:
        """Eigenvalue verdict for the companion matrix.

        Returns:
            The :class:`_StabilityResult`; the process is stable, and so has a
            convergent moving-average representation, exactly when every
            companion eigenvalue lies inside the unit circle. Every other
            method here presumes that: an impulse response computed from an
            explosive companion diverges rather than decays, and a forecast
            from one is meaningless at any horizon.
        """
        return _StabilityResult.assess_stability(self.coefficients)

    @property
    def is_stable(self) -> bool:
        """Whether every companion eigenvalue lies inside the unit circle."""
        return self.stability_check().is_stable

    def ma_representation(self, horizon: int = 20) -> npt.NDArray[np.float64]:
        """Moving-average matrices ``Psi_0, ..., Psi_horizon``.

        Computed as ``J C^h J'`` from the companion rather than by the
        recursion ``Psi_h = sum_i A_i Psi_{h-i}``. The two agree to machine
        precision, and the companion form generalizes without change to any
        member of the family that can produce a companion matrix.

        Args:
            horizon: Largest lead to return.

        Returns:
            An array of shape ``(horizon + 1, k, k)`` with ``Psi_0 = I``.

        Raises:
            SpecificationError: If ``horizon`` is negative.
        """
        if horizon < 0:
            raise SpecificationError(f"horizon must be non-negative; got {horizon}.")
        k, p = self.k_endog, self.order
        out = np.empty((horizon + 1, k, k), dtype=np.float64)
        if p == 0:
            out[:] = 0.0
            out[0] = np.eye(k)
            return out
        selector = np.zeros((k, k * p), dtype=np.float64)
        selector[:, :k] = np.eye(k)
        power = np.eye(k * p, dtype=np.float64)
        companion = self.companion
        for h in range(horizon + 1):
            out[h] = selector @ power @ selector.T
            power = power @ companion
        return out

    def irf(
        self, horizon: int = 20, *, orthogonalized: bool = True, cumulative: bool = False
    ) -> npt.NDArray[np.float64]:
        """Impulse responses to a one-standard-deviation shock.

        Args:
            horizon: Largest lead to return.
            orthogonalized: When ``True``, post-multiply by the Cholesky factor
                of ``sigma_u`` so the shocks are mutually uncorrelated. **This
                is a structural assumption, not a reduced-form fact.** The
                Cholesky factor is lower-triangular, so it imposes a recursive
                ordering: the first variable in ``names`` responds to no shock
                but its own on impact, the second to the first and its own, and
                so on. Permuting ``names`` changes the answer. An orthogonalized
                impulse response reported from a reduced-form VAR is a recursive
                SVAR whose identifying restriction happens to be undeclared;
                :mod:`cultivars.var.structural` is where that restriction gets
                stated rather than assumed. With ``False`` you get the raw
                ``Psi_h``, whose columns are responses to correlated
                innovations and are therefore not interpretable one at a time.
            cumulative: Return running sums, which is what you want when the
                data are differences and the question is about levels.

        Returns:
            An array of shape ``(horizon + 1, k, k)``; entry ``[h, i, j]`` is
            the response of variable ``i`` at lead ``h`` to shock ``j``.
        """
        psi = self.ma_representation(horizon)
        out = psi @ self._impact() if orthogonalized else psi
        return np.cumsum(out, axis=0) if cumulative else out

    def generalized_irf(
        self, horizon: int = 20, *, cumulative: bool = False
    ) -> npt.NDArray[np.float64]:
        """Pesaran-Shin impulse responses, which do not depend on variable order.

        The orthogonalized response asks what happens if shock ``j`` moves and
        the shocks ordered after it are held at zero. That question needs an
        order, and :meth:`irf` takes it from ``names``. The generalized response
        asks a different question -- what happens if shock ``j`` moves and the
        others take their conditional expectations given that move -- which
        needs no order at all, only the covariance.

        The two agree exactly for whichever variable is placed first, and
        disagree for every other. Neither is more correct in general: the
        orthogonalized one answers a structural question and requires a
        defensible recursive ordering, the generalized one answers a predictive
        question and requires none. Reach for this when no ordering is
        defensible, which is the usual situation once a system spans more
        variables than anyone has a theory about.

        Args:
            horizon: Periods to propagate.
            cumulative: Accumulate the responses over the horizon.

        Returns:
            An ``(horizon + 1, k, k)`` array whose ``[h, i, j]`` entry is the
            response of variable ``i`` at horizon ``h`` to a one-standard-
            deviation shock to variable ``j``.
        """
        psi = self.ma_representation(horizon)
        sigma = self.sigma_u
        scale = np.sqrt(np.diag(sigma))
        out = np.stack([psi @ sigma[:, j] / scale[j] for j in range(self.k_endog)], axis=-1)
        return np.cumsum(out, axis=0) if cumulative else out

    def fevd(self, horizon: int = 20) -> npt.NDArray[np.float64]:
        """Forecast-error variance decomposition.

        Args:
            horizon: Largest lead to return.

        Returns:
            An array of shape ``(horizon + 1, k, k)`` whose entry ``[h, i, j]``
            is the share of variable ``i``'s ``h + 1``-step forecast-error
            variance attributable to shock ``j``. Rows sum to one by
            construction, so a row that does not is a bug rather than a finding.

        Note:
            Inherits the ordering dependence of :meth:`irf` in full, and more
            visibly: at ``h = 0`` the decomposition is exactly triangular, so
            the first variable is always attributed 100% of its own impact
            variance purely because of where it sits in ``names``.
        """
        theta = self.irf(horizon, orthogonalized=True)
        contribution = np.cumsum(theta**2, axis=0)
        return contribution / contribution.sum(axis=2, keepdims=True)

    def generalized_fevd(self, horizon: int = 20) -> npt.NDArray[np.float64]:
        """Variance shares from the generalized responses, normalized to sum to one.

        The counterpart of :meth:`fevd` for systems with no defensible recursive
        ordering. It carries one caveat that the orthogonalized version does
        not, and the caveat is structural rather than cosmetic: generalized
        shocks are not orthogonal to each other, so the raw contributions do not
        add up to the forecast error variance. Dividing by their sum forces rows
        to one, which makes the table readable and makes the numbers a relative
        ranking rather than a decomposition.

        Read a cell as "how much of this variable's forecast error is associated
        with that shock, relative to the others", not as "how much is caused by
        it". When the shocks are close to orthogonal the two readings coincide;
        when they are strongly correlated the normalization is doing real work
        and the shares should be treated as indicative.

        Args:
            horizon: Periods to accumulate over.

        Returns:
            An ``(horizon + 1, k, k)`` array whose rows sum to one.
        """
        theta = self.generalized_irf(horizon)
        contrib = np.cumsum(theta**2, axis=0)
        return contrib / contrib.sum(axis=2, keepdims=True)

    def forecast(self, steps: int = 1) -> npt.NDArray[np.float64]:
        """Deterministic multi-step forecasts from the end of the sample.

        Iterates the estimated system forward with future innovations set to
        their zero mean, which is the conditional expectation. No interval is
        returned: the coefficient uncertainty that should widen it is exactly
        the standard-error machinery this package does not yet have, and a band
        computed as though the coefficients were known would be too narrow in a
        way nobody could see.

        Args:
            steps: Forecast horizon, at least one.

        Returns:
            An array of shape ``(steps, k)``.

        Raises:
            SpecificationError: If ``steps`` is less than one.
        """
        if steps < 1:
            raise SpecificationError(f"steps must be at least 1; got {steps}.")
        k, p, n = self.k_endog, self.order, self.endog.shape[0]
        future = deterministic_columns(self.trend, steps, start=n + 1)
        history = [self.endog[n - i - 1] for i in range(p)]
        out = np.empty((steps, k), dtype=np.float64)
        for h in range(steps):
            point = (
                future[h] @ self.deterministic
                if self.deterministic.shape[0]
                else np.zeros(k, dtype=np.float64)
            )
            for i in range(p):
                point = point + self.coefficients[i] @ history[i]
            out[h] = point
            history = [point, *history[: p - 1]] if p else []
        return out

    def historical_decomposition(self) -> npt.NDArray[np.float64]:
        """Attribute each observation to the shocks that produced it.

        Returns:
            An array of shape ``(nobs, k, k)`` whose entry ``[t, i, j]`` is
            shock ``j``'s cumulative contribution to variable ``i`` at time
            ``t``. Summing over ``j`` recovers the *stochastic* component of
            the path, not the observed series: the deterministic terms and the
            influence of pre-sample initial conditions are excluded by
            construction, since neither is attributable to any shock. Add the
            deterministic path back before comparing against ``endog``.

        Note:
            Costs ``O(nobs^2)`` moving-average terms, because the contribution
            at ``t`` sums over every shock up to ``t``. Fine for macro samples;
            noticeable past a few thousand observations.
        """
        weights = self.structural_shocks()
        theta = self.irf(self.nobs - 1, orthogonalized=True)
        out = np.zeros((self.nobs, self.k_endog, self.k_endog), dtype=np.float64)
        for t in range(self.nobs):
            out[t] = np.einsum("lij,lj->ij", theta[: t + 1], weights[t::-1])
        return out


class _ConditionalSystemMixin:
    """Good errors for the propagation surface a conditional family does not have.

    Deliberately *not* a subclass or sibling of :class:`_VectorPropagationMixin`.
    A conditional result is not substitutable for a closed one, and inheriting
    the propagation type only to raise from every method would assert a
    subtyping relationship that does not hold -- ``isinstance(result,
    _VectorPropagationMixin)`` is ``False`` here, which is the honest answer.
    What this mixin supplies is the courtesy of a message: without it the caller
    gets ``AttributeError: no attribute 'irf'``, which is correct and tells them
    nothing about why.

    :meth:`__init_subclass__` checks the coverage. Adding a method to the
    propagation mixin without a counterpart here fails at import, in the file
    where the omission is, rather than months later at a call site with an
    error naming whichever nested call happened to raise first.
    """

    __slots__ = ()

    names: tuple[str, ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Verify this family accounts for every propagation member.

        Raises:
            TypeError: If the propagation mixin has grown a member with no
                refusal here.
        """
        super().__init_subclass__(**kwargs)
        missing = sorted(
            name
            for name in _VectorPropagationMixin._PROPAGATION_MEMBERS
            if not any(name in vars(base) for base in cls.__mro__)
        )
        if missing:
            raise TypeError(
                f"{cls.__name__} mixes in _ConditionalSystemMixin but does not account "
                f"for {missing}; every member of _VectorPropagationMixin needs either an "
                "implementation or a refusal here."
            )

    def _no_closed_system(self, what: str) -> None:
        """Raise the shared explanation, naming the quantity that was asked for.

        Args:
            what: The quantity, phrased to read after "needs a law of motion".

        Raises:
            SpecificationError: Always.
        """
        raise SpecificationError(_NO_CLOSED_SYSTEM.format(model=type(self).__name__, what=what))

    @property
    def companion(self) -> npt.NDArray[np.float64]:
        """Unavailable: there is no closed system to take a companion of."""
        self._no_closed_system("a companion matrix")
        raise AssertionError  # pragma: no cover

    def stability_check(self) -> _StabilityResult:
        """Unavailable: stability is a property of the closed system."""
        self._no_closed_system("a stability check")
        raise AssertionError  # pragma: no cover

    @property
    def is_stable(self) -> bool:
        """Unavailable: stability is a property of the closed system."""
        self._no_closed_system("a stability verdict")
        raise AssertionError  # pragma: no cover

    def ma_representation(self, horizon: int) -> npt.NDArray[np.float64]:
        """Unavailable: propagation needs the exogenous block's law of motion."""
        self._no_closed_system("a moving-average representation")
        raise AssertionError  # pragma: no cover

    def irf(
        self, horizon: int = 20, *, orthogonalized: bool = True, cumulative: bool = False
    ) -> npt.NDArray[np.float64]:
        """Unavailable: propagation needs the exogenous block's law of motion."""
        self._no_closed_system("an impulse response")
        raise AssertionError  # pragma: no cover

    def generalized_irf(
        self, horizon: int = 20, *, cumulative: bool = False
    ) -> npt.NDArray[np.float64]:
        """Unavailable: propagation needs the exogenous block's law of motion."""
        self._no_closed_system("a generalized impulse response")
        raise AssertionError  # pragma: no cover

    def fevd(self, horizon: int = 20) -> npt.NDArray[np.float64]:
        """Unavailable: propagation needs the exogenous block's law of motion."""
        self._no_closed_system("a variance decomposition")
        raise AssertionError  # pragma: no cover

    def historical_decomposition(self) -> npt.NDArray[np.float64]:
        """Unavailable: propagation needs the exogenous block's law of motion."""
        self._no_closed_system("a historical decomposition")
        raise AssertionError  # pragma: no cover

    def forecast(self, steps: int = 1) -> npt.NDArray[np.float64]:
        """Conditional families override this with one that takes a path.

        Present so the coverage check passes, and so a family that has no
        conditional forecast either still fails with an explanation rather than
        an attribute error.
        """
        self._no_closed_system("an unconditional forecast")
        raise AssertionError  # pragma: no cover
