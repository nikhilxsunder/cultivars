# filepath: /src/cultivars/multivariate/structural/factor_augmented.py
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

"""Structural identification on a factor block: one shock, hundreds of answers.

Identifying a factor-augmented VAR is identifying its small system -- the
factor VAR is a closed reduced form like any other, and any scheme in this
package factorizes it. What earns this row its own model is what happens
next: the observation equation maps the identified factor-space shocks onto
every series in the informational panel, which is the entire reason the FAVAR
exists -- one identification, answered by hundreds of responses -- and the
map is rotation-invariant, because principal components recover the factor
space only up to rotation while loadings and factors rotate together
(Bernanke, Boivin, and Eliasz 2005).

The canonical identification is the paper's own: recursive, with the observed
block -- the policy rate -- ordered last, so the policy shock moves no factor
within the period. That is this model's default, and it is exactly where the
slow-variable cleaning of the reduced form earns its keep: without it the
factors carry the observed block's contemporaneous influence and the
recursive ordering is contaminated, which the summary discloses. Any other
scheme rides in through the same door -- identify the factor VAR with
whichever model states the restriction you believe, and hand the result here
to be mapped.

References:
    Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the effects
        of monetary policy: A factor-augmented vector autoregressive (FAVAR)
        approach. *Quarterly Journal of Economics*, 120(1), 387-422.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import StructuralResult, SummaryTable
from ..._internals import _IdentificationModel, _SummaryMixin
from ...exceptions import SpecificationError
from ..large_dim import FAVARResult
from .zero_restrictions import RecursiveSVAR


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class FactorAugmentedSVARResult(_SummaryMixin):
    """An identified factor system, answered at the scale of the panel.

    Composition on both sides: :attr:`structural` is the factor-space
    identification with its whole surface -- impact matrix, factor impulse
    responses, variance shares, recovered shocks -- and :attr:`favar` is the
    reduced form with its loadings and fit. This object owns the join: the
    observation-equation map from identified factor shocks to responses of
    every panel series, in each series' own units.

    Attributes:
        favar: The fitted factor-augmented reduced form.
        structural: The identification of its factor VAR.
    """

    favar: FAVARResult = field(repr=False)
    structural: StructuralResult = field(repr=False)

    @property
    def panel_names(self) -> tuple[str, ...]:
        """One label per panel series."""
        return self.favar.panel_names

    @property
    def n_series(self) -> int:
        """Number of panel series."""
        return self.favar.n_series

    @property
    def shock_names(self) -> tuple[str, ...]:
        """Labels of the identified shocks, from the factor-space result."""
        names = getattr(self.structural, "shock_names", None)
        if names is None:
            return tuple(f"shock{j + 1}" for j in range(self.impact.shape[1]))
        return tuple(names)

    @property
    def impact(self) -> npt.NDArray[np.float64]:
        """Impact responses of every panel series, ``(n_series, s)``."""
        return self.irf(0)[0]

    def irf(self, horizon: int = 20, *, cumulative: bool = False) -> npt.NDArray[np.float64]:
        """Responses of every panel series to the identified shocks.

        The observation equation applied to the factor-space structural
        impulse responses, rescaled into each series' own units.

        Args:
            horizon: Largest lead to return.
            cumulative: Return running sums, for differenced data and level
                questions.

        Returns:
            An array of shape ``(horizon + 1, n_series, s)``; entry
            ``[h, i, j]`` is the response of panel series ``i`` at lead ``h``
            to one standard deviation of identified shock ``j``, in the
            series' original units.
        """
        responses = self.structural.irf(horizon, cumulative=cumulative)
        return (
            np.einsum("nk,hks->hns", self.favar.loadings, responses)
            * self.favar._scales[np.newaxis, :, np.newaxis]
        )

    def factor_irf(self, horizon: int = 20, *, cumulative: bool = False) -> npt.NDArray[np.float64]:
        """The factor-space structural impulse responses, undelegated.

        Args:
            horizon: Largest lead to return.
            cumulative: Return running sums.

        Returns:
            The ``(horizon + 1, r + m, s)`` factor-system responses; the full
            factor-level surface lives on :attr:`structural`.
        """
        return self.structural.irf(horizon, cumulative=cumulative)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        factor_impact = self.factor_irf(0)[0]
        factor_names = self.favar.factors.names
        shocks = self.shock_names
        rows = tuple(
            (name, *(f"{factor_impact[i, j]:.4f}" for j in range(len(shocks))))
            for i, name in enumerate(factor_names)
        )
        scheme = getattr(self.structural, "scheme", "supplied")
        restriction = getattr(self.structural, "restriction", "")
        notes = [
            restriction,
            "The factor-space identification above is answered at panel "
            f"scale: irf() maps it onto all {self.n_series} series through "
            "the observation equation, in each series' own units.",
            "The map is rotation-invariant: principal components recover the "
            "factor space only up to rotation, and loadings rotate with the "
            "factors, so panel responses do not depend on the factor basis.",
        ]
        if not self.favar.cleaned:
            notes.append(
                "The reduced form ran without the slow-variable cleaning, so "
                "the factors retain the observed block's contemporaneous "
                "influence and a recursive ordering placing the observed "
                "variables last is contaminated."
            )
        return SummaryTable(
            title=f"Structural FAVAR ({scheme})",
            metadata=(
                ("Scheme", str(scheme)),
                ("Identified shocks", f"{len(shocks)}"),
                ("Factor system", f"{len(factor_names)}"),
                ("Panel series", f"{self.n_series}"),
                ("Slow-variable cleaning", "yes" if self.favar.cleaned else "no"),
                ("Observations", f"{self.favar.panel.shape[0]}"),
            ),
            columns=("factor impact of", *shocks),
            rows=rows,
            notes=tuple(note for note in notes if note),
        )


class FactorAugmentedSVAR(_IdentificationModel[FactorAugmentedSVARResult]):
    """Structural identification on a factor block, Bernanke-Boivin-Eliasz (2005).

    Constructs with a fitted :class:`~cultivars.multivariate.large_dim.FAVAR`
    result; the closed system being identified is its factor VAR, and the
    model adds what no factor-space scheme can: the observation-equation map
    onto the panel. Two ways in, one result out. By default ``identify`` runs
    the paper's own scheme -- recursive, observed block last, so the policy
    shock moves no factor within the period. Alternatively, identify the
    factor VAR yourself with any point-identified model in this package and
    pass what it returns as ``structural``; the declaration then lives where
    you stated it, and this model only maps it.

    Args:
        favar: The fitted factor-augmented reduced form.
        structural: Optional factor-space identification to map, produced by
            any point-identified model applied to ``favar.factors``. When
            given, ``order`` must be omitted.
        order: The recursive ordering for the default scheme. ``None`` orders
            the system as it stands -- factors first, observed block last,
            the Bernanke-Boivin-Eliasz convention.

    Raises:
        SpecificationError: If ``favar`` is not a fitted FAVAR result, both
            ``structural`` and ``order`` are given, or ``structural`` was not
            identified from this FAVAR's own factor VAR.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> from cultivars.multivariate.large_dim import FAVAR
        >>> factor = np.zeros(300)
        >>> for t in range(1, 300):
        ...     factor[t] = 0.7 * factor[t - 1] + rng.standard_normal()
        >>> panel = np.outer(factor, rng.uniform(0.5, 1.5, 30))
        >>> panel += 0.3 * rng.standard_normal((300, 30))
        >>> policy = 0.4 * factor + rng.standard_normal(300)
        >>> res = FAVAR(panel, policy[:, None], n_factors=1, order=1).fit()
        >>> svar = FactorAugmentedSVAR(res).identify()
        >>> svar.impact.shape
        (30, 2)
    """

    __slots__ = ("_favar", "_order", "_structural")

    def __init__(
        self,
        favar: FAVARResult,
        structural: StructuralResult | None = None,
        *,
        order: Sequence[str] | None = None,
    ) -> None:
        """Validate the reduced form and the identification route."""
        if not isinstance(favar, FAVARResult):
            raise SpecificationError(
                "FactorAugmentedSVAR constructs with a fitted FAVAR result; "
                f"got {type(favar).__name__}. Fit "
                "cultivars.multivariate.large_dim.FAVAR first."
            )
        super().__init__(favar.factors)
        self._favar = favar
        if structural is not None:
            if order is not None:
                raise SpecificationError(
                    "pass either a factor-space identification to map or an "
                    "ordering for the default recursive scheme, not both."
                )
            if structural.source is not favar.factors:
                raise SpecificationError(
                    "the supplied identification was not built from this "
                    "FAVAR's own factor VAR; identify favar.factors and pass "
                    "what that returns."
                )
        self._structural = structural
        self._order = None if order is None else tuple(str(name) for name in order)

    def identify(self) -> FactorAugmentedSVARResult:
        """Identify the factor system and bind the panel map to it.

        Returns:
            The panel-scale structural result. When no identification was
            supplied, the factor VAR is factorized recursively with the
            observed block last -- the Bernanke-Boivin-Eliasz convention --
            or in the declared ordering.

        Raises:
            SpecificationError: If a declared ordering is not a permutation
                of the factor-system names.
            NumericalError: If the factor VAR's innovation covariance is not
                positive definite.
        """
        structural: StructuralResult = (
            self._structural
            if self._structural is not None
            else RecursiveSVAR(self._favar.factors, order=self._order).identify()
        )
        return FactorAugmentedSVARResult(favar=self._favar, structural=structural)
