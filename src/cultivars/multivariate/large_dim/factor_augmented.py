# filepath: /src/cultivars/multivariate/large_dim/factor_augmented.py
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

"""The factor-augmented VAR: a few shocks, answered by hundreds of series.

A small VAR cannot say what a policy shock does to the hundreds of series a
policymaker actually watches, and a VAR on hundreds of series cannot be
estimated. The factor-augmented VAR threads the needle: a wide informational
panel is summarized by a handful of principal-component factors, a VAR runs on
those factors augmented with the observed variables of interest, and the
observation equation ``X_t = Lambda [F_t, Y_t] + e_t`` maps everything the
small system produces back onto every series in the panel -- Bernanke,
Boivin, and Eliasz (2005).

The design is composition, on the pattern the functional VAR set: the factor
dynamics *are* a :class:`~cultivars.multivariate.reduced_form.VAR` on the
augmented block, riding on the result as ``factors`` with its entire
reduced-form surface intact. And because that result is a closed system, the
entire structural layer applies to it unchanged -- recursive, sign,
narrative, proxy, heteroskedasticity, any of them -- with
:meth:`FAVARResult.panel_irf` translating whatever was identified into
responses of all ``N`` panel series. That translation is
rotation-invariant: principal components recover the factor space only up to
rotation, but loadings and factors rotate together, so panel responses to an
identified shock do not depend on the arbitrary factor basis.

One estimation subtlety is load-bearing for policy analysis and handled the
way the original paper handles it. Principal components of the *full* panel
absorb the observed variables' contemporaneous influence, which contaminates
a recursive identification that orders those variables last. Declaring the
``slow`` series -- those that do not react within the period -- triggers the
Bernanke-Boivin-Eliasz cleaning: the observed block's contemporaneous
component is regressed out of the factors against the slow-panel components,
and skipping the declaration is disclosed in the summary rather than papered
over.

References:
    Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the effects
        of monetary policy: A factor-augmented vector autoregressive (FAVAR)
        approach. *Quarterly Journal of Economics*, 120(1), 387-422.
    Stock, J. H., & Watson, M. W. (2002). Forecasting using principal
        components from a large number of predictors. *Journal of the
        American Statistical Association*, 97(460), 1167-1179.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import (
    SummaryTable,
    Trend,
    _validate_wide_panel,
    validate_exog_matrix,
)
from ..._internals import _SummaryMixin
from ...exceptions import SpecificationError
from ..reduced_form.vector_autoregression import VAR, VARResult


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class FAVARResult(_SummaryMixin):
    """A fitted factor-augmented vector autoregression.

    Composition, stated plainly: :attr:`factors` is a complete
    :class:`~cultivars.multivariate.reduced_form.VARResult` on the augmented
    block, and every factor-level question -- coefficients, diagnostics,
    stability, reduced-form impulse responses -- is answered there. It is
    also a closed system, so every identification model in
    :mod:`cultivars.multivariate.structural` accepts it directly; this object
    owns what the factor result cannot know: the panel, the loadings, and the
    map that turns an identified factor-space shock into responses of every
    series.

    Attributes:
        panel: The informational panel as supplied.
        observed: The observed block as supplied.
        factors: The fitted VAR on ``[F, Y]``, carrying the whole reduced-form
            surface and accepting every identification model.
        loadings: ``(n_series, r + m)`` observation-equation coefficients of
            the standardized panel on the augmented block.
        scores: The ``(nobs, r)`` estimated factors, cleaned when ``slow`` was
            declared.
        panel_names: One label per panel series.
        explained_variance: Share of standardized panel variance each factor
            carries, before cleaning.
        r2: Per-series fit of the observation equation, the common-component
            share of each series.
        cleaned: Whether the Bernanke-Boivin-Eliasz slow-variable cleaning
            ran.
    """

    panel: npt.NDArray[np.float64] = field(repr=False)
    observed: npt.NDArray[np.float64] = field(repr=False)
    factors: VARResult = field(repr=False)
    loadings: npt.NDArray[np.float64] = field(repr=False)
    scores: npt.NDArray[np.float64] = field(repr=False)
    panel_names: tuple[str, ...]
    explained_variance: npt.NDArray[np.float64] = field(repr=False)
    r2: npt.NDArray[np.float64] = field(repr=False)
    cleaned: bool
    _scales: npt.NDArray[np.float64] = field(repr=False)

    @property
    def n_series(self) -> int:
        """Number of panel series."""
        return len(self.panel_names)

    @property
    def n_factors(self) -> int:
        """Number of estimated factors."""
        return int(self.scores.shape[1])

    def common_component(self) -> npt.NDArray[np.float64]:
        """The panel as the factors and observed block see it, in original units.

        Returns:
            An ``(nobs, n_series)`` array; the gap between this and
            :attr:`panel` is the idiosyncratic component, and each series'
            :attr:`r2` says how much of it there is.
        """
        augmented = np.column_stack([self.scores, self.observed])
        centered = augmented @ self.loadings.T * self._scales
        return centered + self.panel.mean(axis=0)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        stability = self.factors.stability_check()
        rows = tuple(
            (
                name,
                f"{float(self.explained_variance[j]):.1%}",
            )
            for j, name in enumerate(self.factors.names[: self.n_factors])
        )
        notes = [
            f"Factor VAR stable: {self.factors.is_stable}   max |companion "
            f"root| = {stability.max_modulus:.4f}",
            "The factor VAR on .factors is a closed system: every "
            "identification model in cultivars.multivariate.structural "
            "accepts it, and FactorAugmentedSVAR maps an identification "
            f"onto all {self.n_series} panel series.",
            "Factors are principal components, identified up to rotation; "
            "loadings rotate with them, so panel responses to an identified "
            "shock do not depend on the factor basis.",
        ]
        if not self.cleaned:
            notes.append(
                "No slow series were declared, so the factors retain the "
                "observed block's contemporaneous influence; a recursive "
                "identification ordering the observed variables last is "
                "contaminated without the Bernanke-Boivin-Eliasz cleaning."
            )
        return SummaryTable(
            title=f"FAVAR({self.factors.order}) Results",
            metadata=(
                ("Factors", f"{self.n_factors}"),
                ("Observed variables", f"{self.observed.shape[1]}"),
                ("Panel series", f"{self.n_series}"),
                ("Observations", f"{self.panel.shape[0]}"),
                ("Order", f"{self.factors.order}"),
                ("Slow-variable cleaning", "yes" if self.cleaned else "no"),
                ("Mean observation R^2", f"{float(self.r2.mean()):.3f}"),
                ("Factor log-likelihood", f"{self.factors.llf:.3f}"),
            ),
            columns=("factor", "variance share"),
            rows=rows,
            notes=tuple(notes),
        )


class FAVAR:
    """Factor-augmented VAR over a wide informational panel, BBE (2005).

    Extract, clean, augment, fit. Principal components of the standardized
    panel estimate the factors; when ``slow`` series are declared, the
    observed block's contemporaneous influence is regressed out of them
    against the slow-panel components, which is what makes a recursive policy
    identification on the result defensible; the factors and the observed
    block then form a small VAR estimated by composition; and the observation
    equation is fitted by least squares, one panel series at a time.

    Args:
        panel: The ``(nobs, n_series)`` informational panel. Standardized
            internally; responses are reported back in original units.
        observed: The ``(nobs, m)`` observed block -- the variables whose
            shocks are of interest, a policy rate being the canonical case.
        n_factors: Number of principal-component factors to extract.
        order: Autoregressive order of the augmented VAR.
        slow: Names of the panel series that do not respond to the observed
            block within the period, for the Bernanke-Boivin-Eliasz cleaning.
            ``None`` skips the cleaning, and the summary says so.
        trend: Deterministic terms of the augmented VAR.
        observed_names: Labels for the observed block. Defaults to
            ``y1 ... ym``.
        panel_names: Labels for the panel series. Defaults to ``x1 ... xN``.

    Raises:
        SpecificationError: If the specification is malformed, a slow name is
            unknown, or ``n_factors`` exceeds what the panel supports.
        DimensionError: If the panel and observed block do not align.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> factor = np.zeros(300)
        >>> for t in range(1, 300):
        ...     factor[t] = 0.7 * factor[t - 1] + rng.standard_normal()
        >>> panel = np.outer(factor, rng.uniform(0.5, 1.5, 30))
        >>> panel += 0.3 * rng.standard_normal((300, 30))
        >>> policy = 0.4 * factor + rng.standard_normal(300)
        >>> res = FAVAR(panel, policy[:, None], n_factors=1, order=1).fit()
        >>> res.factors.k_endog
        2
    """

    __slots__ = (
        "_n_factors",
        "_observed",
        "_observed_names",
        "_order",
        "_panel",
        "_panel_names",
        "_slow",
        "_trend",
    )

    def __init__(
        self,
        panel: npt.ArrayLike,
        observed: npt.ArrayLike,
        *,
        n_factors: int,
        order: int,
        slow: Sequence[str] | None = None,
        trend: Trend = "c",
        observed_names: Sequence[str] | None = None,
        panel_names: Sequence[str] | None = None,
    ) -> None:
        """Validate the panel, the observed block, and the specification."""
        self._panel = _validate_wide_panel(panel, label="panel")
        nobs, n_series = self._panel.shape
        self._observed = validate_exog_matrix(observed, nobs=nobs, label="observed")
        if int(n_factors) != n_factors or n_factors < 1:
            raise SpecificationError(
                f"n_factors must be an integer >= 1; got {n_factors!r}."
            )
        if n_factors > min(nobs - 1, n_series):
            raise SpecificationError(
                f"n_factors ({n_factors}) exceeds what a ({nobs}, {n_series}) "
                "panel can support."
            )
        self._n_factors = int(n_factors)
        self._order = int(order)
        self._trend: Trend = trend
        if panel_names is None:
            self._panel_names = tuple(f"x{i + 1}" for i in range(n_series))
        else:
            resolved = tuple(str(name) for name in panel_names)
            if len(resolved) != n_series or len(set(resolved)) != n_series:
                raise SpecificationError(
                    f"panel_names must be {n_series} unique labels; got "
                    f"{len(resolved)}."
                )
            self._panel_names = resolved
        m = self._observed.shape[1]
        if observed_names is None:
            self._observed_names = tuple(f"y{i + 1}" for i in range(m))
        else:
            resolved = tuple(str(name) for name in observed_names)
            if len(resolved) != m or len(set(resolved)) != m:
                raise SpecificationError(
                    f"observed_names must be {m} unique labels; got "
                    f"{len(resolved)}."
                )
            self._observed_names = resolved
        if slow is None:
            self._slow: tuple[int, ...] | None = None
        else:
            positions: list[int] = []
            for name in slow:
                label = str(name)
                if label not in self._panel_names:
                    raise SpecificationError(
                        f"unknown slow series {label!r}; expected one of the "
                        "panel names."
                    )
                positions.append(self._panel_names.index(label))
            if len(positions) < self._n_factors + 1:
                raise SpecificationError(
                    f"the cleaning needs more slow series than factors; got "
                    f"{len(positions)} slow series for {self._n_factors} "
                    "factors."
                )
            self._slow = tuple(positions)

    @staticmethod
    def _components(
        standardized: npt.NDArray[np.float64], count: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Principal-component scores and variance shares of a panel."""
        _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
        shares = singular**2 / float(np.sum(singular**2))
        scores = standardized @ vt[:count].T
        return scores, shares[:count]

    def fit(self) -> FAVARResult:
        """Extract the factors, clean them, and estimate the augmented VAR.

        Returns:
            The fitted result, its ``factors`` ready for any identification
            model.
        """
        means = self._panel.mean(axis=0)
        scales = self._panel.std(axis=0, ddof=0)
        scales = np.where(scales > 0.0, scales, 1.0)
        standardized = (self._panel - means) / scales

        scores, shares = self._components(standardized, self._n_factors)
        cleaned = False
        if self._slow is not None:
            slow_scores, _ = self._components(
                standardized[:, list(self._slow)], self._n_factors
            )
            design = np.column_stack([slow_scores, self._observed])
            coef: npt.NDArray[np.float64] = np.linalg.lstsq(
                design, scores, rcond=None
            )[0]
            scores = scores - self._observed @ coef[self._n_factors :]
            cleaned = True

        factor_names = tuple(f"f{j + 1}" for j in range(self._n_factors))
        augmented = np.column_stack([scores, self._observed])
        factors = VAR(
            augmented,
            order=self._order,
            trend=self._trend,
            names=(*factor_names, *self._observed_names),
        ).fit()

        loadings: npt.NDArray[np.float64] = np.linalg.lstsq(
            augmented, standardized, rcond=None
        )[0].T
        fitted = augmented @ loadings.T
        residual = standardized - fitted
        total = np.sum(standardized**2, axis=0)
        r2 = 1.0 - np.sum(residual**2, axis=0) / np.where(total > 0.0, total, 1.0)

        return FAVARResult(
            panel=self._panel,
            observed=self._observed,
            factors=factors,
            loadings=loadings,
            scores=scores,
            panel_names=self._panel_names,
            explained_variance=shares,
            r2=r2,
            cleaned=cleaned,
            _scales=scales,
        )