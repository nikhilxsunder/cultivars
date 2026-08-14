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

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from .._core import InformationCriteria, SummaryTable, to_pandas_frame, to_polars_frame
from ..exceptions import DimensionError, SpecificationError
from ._results import _LikelihoodRatioResult, _StabilityResult

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
