# filepath: /src/cultivars/multivariate/large_dim/dynamic_factor.py
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

"""The dynamic factor model: a large panel explained by a small VAR.

The premise (Stock-Watson; Forni-Reichlin) is that a hundred macroeconomic
series comove because a handful of common factors drive them: each series is
a loading on the factors plus an idiosyncratic remainder, and the factors
themselves follow a VAR. Estimation here is the Stock-Watson two-step,
deliberately: principal components estimate the factor space consistently as
both panel dimensions grow, a VAR on the estimated factors captures the
common dynamics, and the whole procedure is transparent, fast at any ``N``,
and free of the convergence pathologies an EM pass can hide. The factor VAR
is a full :class:`~cultivars.multivariate.reduced_form.VARResult`, so
everything a VAR result can do -- forecast, impulse responses, spillover
tables -- runs on the factor system directly.

How many factors is a modelling choice this class will make for you only by
a stated criterion: left unspecified, the count minimizes Bai-Ng's
``ICp2``, the standard consistent selector, and the criterion values are
kept on the result so the choice can be audited rather than trusted. Two
honesty notes carry through the surface. Factors are identified only up to
rotation -- principal components pin the space, not the basis -- so
individual factors carry no structural names, and anything worth
interpreting should be read from the common component or from loadings
patterns. And the two-step ignores estimation error in the factors when
fitting their VAR, which is asymptotically negligible for large panels
(Bai-Ng 2006) and is stated rather than hidden for small ones.

References:
    Stock, J. H., & Watson, M. W. (2002). Forecasting using principal
        components from a large number of predictors. *Journal of the
        American Statistical Association*, 97(460), 1167-1179.
    Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000). The
        generalized dynamic-factor model: Identification and estimation.
        *Review of Economics and Statistics*, 82(4), 540-554.
    Bai, J., & Ng, S. (2002). Determining the number of factors in
        approximate factor models. *Econometrica*, 70(1), 191-221.
    Bai, J., & Ng, S. (2006). Confidence intervals for diffusion index
        forecasts and inference for factor-augmented regressions.
        *Econometrica*, 74(4), 1133-1150.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import SummaryTable, principal_components, validate_endog_matrix
from ..._internals import _SummaryMixin
from ...exceptions import DimensionError, SpecificationError
from ..reduced_form import VAR
from ..reduced_form.vector_autoregression import VARResult


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class DFMResult(_SummaryMixin):
    """A fitted dynamic factor model: loadings, a factor VAR, and the split.

    The factor VAR is a complete fitted result of its own -- forecast it,
    decompose it, feed it to :class:`~cultivars.multivariate.large_dim.Spillover`
    -- while this result owns the mapping between the panel and the factor
    space.

    Attributes:
        panel: The observed ``(nobs, n_series)`` panel.
        series_names: One label per series.
        factors: The fitted VAR on the estimated factors, a full
            :class:`VARResult` with variables ``f1 ... fr``.
        loadings: ``(n_series, r)`` orthonormal loadings on the
            *standardized* panel; the factor space is identified, the basis
            is not.
        means: ``(n_series,)`` series means removed before extraction.
        scales: ``(n_series,)`` series standard deviations divided out.
        variance_shares: ``(r,)`` share of the standardized panel's variance
            each component explains.
        idiosyncratic: ``(nobs, n_series)`` standardized-panel remainders
            after the common component.
        criterion_values: Bai-Ng ``ICp2`` at each candidate count -- empty
            when the caller fixed the count, so the audit trail matches what
            was actually decided.
        n_factors: The factor count estimated with.
    """

    panel: npt.NDArray[np.float64] = field(repr=False)
    series_names: tuple[str, ...]
    factors: VARResult = field(repr=False)
    loadings: npt.NDArray[np.float64] = field(repr=False)
    means: npt.NDArray[np.float64] = field(repr=False)
    scales: npt.NDArray[np.float64] = field(repr=False)
    variance_shares: npt.NDArray[np.float64] = field(repr=False)
    idiosyncratic: npt.NDArray[np.float64] = field(repr=False)
    criterion_values: npt.NDArray[np.float64] = field(repr=False)
    n_factors: int

    @property
    def n_series(self) -> int:
        """Number of series in the panel."""
        return len(self.series_names)

    @property
    def nobs(self) -> int:
        """Panel length."""
        return int(self.panel.shape[0])

    @property
    def common_component(self) -> npt.NDArray[np.float64]:
        """The factors' reconstruction of the panel, in original units."""
        scores = np.asarray(self.factors.endog, dtype=np.float64)
        return scores @ self.loadings.T * self.scales + self.means

    def r_squared(self) -> dict[str, float]:
        """Share of each series' variance the common component explains.

        Returns:
            A mapping from series name to its common-component ``R^2`` on
            the standardized scale.
        """
        explained = 1.0 - self.idiosyncratic.var(axis=0)
        return {name: float(explained[i]) for i, name in enumerate(self.series_names)}

    def forecast(self, steps: int = 8) -> npt.NDArray[np.float64]:
        """Forecast the whole panel through the factor VAR.

        The factors are forecast by their own fitted system and mapped back
        through the loadings; the idiosyncratic components are forecast at
        their zero mean, which is the model's own claim about them.

        Args:
            steps: Horizons ahead, at least one.

        Returns:
            An array of shape ``(steps, n_series)`` in original units.
        """
        scores = self.factors.forecast(steps)
        return scores @ self.loadings.T * self.scales + self.means

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        fit = self.r_squared()
        ordered = sorted(fit.items(), key=lambda item: item[1], reverse=True)
        rows = tuple((name, f"{value:.3f}") for name, value in ordered[:10])
        notes = [
            f"{self.n_factors} factor(s) explain "
            f"{100.0 * float(self.variance_shares.sum()):.1f}% of the "
            "standardized panel's variance.",
            (
                "The factor count minimized Bai-Ng ICp2; criterion_values keeps the audited path."
                if self.criterion_values.size
                else "The factor count was fixed by the caller."
            ),
            "Factors are identified up to rotation: the common component "
            "and loading patterns are interpretable, individual factor "
            "labels are not.",
            "Two-step estimation ignores factor-estimation error in the "
            "factor VAR; negligible for large panels (Bai-Ng 2006), stated "
            "for small ones.",
            "The factor VAR is a full VARResult on `factors`; forecast, "
            "IRFs, and connectedness run on it directly.",
        ]
        return SummaryTable(
            title=f"DFM({self.n_factors}) Results",
            metadata=(
                ("Model", f"DFM(r={self.n_factors}, p={self.factors.order})"),
                ("Series", f"{self.n_series}"),
                ("Observations", f"{self.nobs}"),
                ("Common variance", f"{100.0 * float(self.variance_shares.sum()):.1f}%"),
            ),
            columns=("series (top 10 by fit)", "common R^2"),
            rows=rows,
            notes=tuple(notes),
        )


class DFM:
    """Dynamic factor model, Stock-Watson two-step.

    Args:
        panel: The observed ``(nobs, n_series)`` panel; wide is welcome.
        order: Autoregressive order of the factor VAR.
        n_factors: Factor count, or ``None`` to minimize Bai-Ng ``ICp2``
            over ``1 .. max_factors``.
        max_factors: Largest candidate count for the criterion search.
        series_names: One label per series. Defaults to ``x1 ... xN``.

    Raises:
        SpecificationError: If the counts are malformed.
        DimensionError: If the panel cannot support the specification.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> f = np.zeros((200, 2))
        >>> for t in range(1, 200):
        ...     f[t] = 0.7 * f[t - 1] + rng.standard_normal(2)
        >>> lam = rng.standard_normal((30, 2))
        >>> panel = f @ lam.T + 1.5 * rng.standard_normal((200, 30))
        >>> res = DFM(panel, order=1).fit()
        >>> res.n_factors
        2
        >>> res.factors.coefficients.shape
        (1, 2, 2)
    """

    __slots__ = ("_max_factors", "_n_factors", "_order", "_panel", "_series_names")

    def __init__(
        self,
        panel: npt.ArrayLike,
        *,
        order: int,
        n_factors: int | None = None,
        max_factors: int = 8,
        series_names: Sequence[str] | None = None,
    ) -> None:
        """Validate the panel and the factor-count request."""
        self._panel = validate_endog_matrix(panel)
        nobs, n_series = self._panel.shape
        ceiling = min(nobs - 1, n_series)
        if n_factors is not None:
            if int(n_factors) != n_factors or n_factors < 1:
                raise SpecificationError(f"n_factors must be an integer >= 1; got {n_factors!r}.")
            if n_factors > ceiling:
                raise DimensionError(
                    f"n_factors ({n_factors}) exceeds what a ({nobs}, "
                    f"{n_series}) panel can support."
                )
        if int(max_factors) != max_factors or max_factors < 1:
            raise SpecificationError(f"max_factors must be an integer >= 1; got {max_factors!r}.")
        self._n_factors = None if n_factors is None else int(n_factors)
        self._max_factors = min(int(max_factors), ceiling)
        self._order = int(order)
        if series_names is None:
            self._series_names = tuple(f"x{i + 1}" for i in range(n_series))
        else:
            resolved = tuple(str(name) for name in series_names)
            if len(resolved) != n_series:
                raise SpecificationError(
                    f"series_names must have one entry per series ({n_series}); "
                    f"got {len(resolved)}."
                )
            self._series_names = resolved

    @staticmethod
    def _bai_ng(
        standardized: npt.NDArray[np.float64], max_factors: int
    ) -> tuple[int, npt.NDArray[np.float64]]:
        """Bai-Ng ``ICp2`` over candidate counts, and its argmin.

        ``ICp2(r) = ln V(r) + r ((N + T) / (N T)) ln(min(N, T))`` with
        ``V(r)`` the mean squared residual of the ``r``-component
        approximation -- the paper's preferred penalty, consistent as both
        dimensions grow. One finite-sample honesty note: when the common
        component explains nearly all of the panel, the log criterion
        measures each spurious eigenvalue against a tiny remainder and
        overselects -- a known property of the criterion, not of this
        implementation. If the reported count looks generous next to the
        variance shares, state ``n_factors`` yourself.
        """
        nobs, n_series = standardized.shape
        scores, loadings, _ = principal_components(standardized, max_factors)
        penalty = (n_series + nobs) / (n_series * nobs) * np.log(min(n_series, nobs))
        values = np.empty(max_factors)
        for r in range(1, max_factors + 1):
            resid = standardized - scores[:, :r] @ loadings[:, :r].T
            values[r - 1] = float(np.log(np.mean(resid**2)) + r * penalty)
        return int(np.argmin(values)) + 1, values

    def fit(self) -> DFMResult:
        """Extract the factors, count them if unasked, and fit their VAR.

        Returns:
            The fitted :class:`DFMResult`.
        """
        means = self._panel.mean(axis=0)
        scales = self._panel.std(axis=0, ddof=0)
        scales = np.where(scales > 0.0, scales, 1.0)
        standardized = (self._panel - means) / scales
        if self._n_factors is None:
            count, criterion = self._bai_ng(standardized, self._max_factors)
        else:
            count, criterion = self._n_factors, np.zeros(0)
        scores, loadings, shares = principal_components(standardized, count)
        factor_result = VAR(
            scores,
            order=self._order,
            trend="n",
            names=tuple(f"f{i + 1}" for i in range(count)),
        ).fit()
        return DFMResult(
            panel=self._panel,
            series_names=self._series_names,
            factors=factor_result,
            loadings=loadings,
            means=means,
            scales=scales,
            variance_shares=shares,
            idiosyncratic=standardized - scores @ loadings.T,
            criterion_values=criterion,
            n_factors=count,
        )
