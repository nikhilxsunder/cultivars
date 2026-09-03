# filepath: /src/cultivars/multivariate/structural/sign_restrictions.py
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

"""Sign-restriction identification: a set of models, reported as a set.

Declaring only the directions of responses -- a supply shock raises output
and lowers prices -- rules out rotations without pinning one down, so what
the data and the declaration jointly deliver is a *set* of structural models.
:class:`SignSVAR` collects that set by drawing Haar-distributed rotations of
the Cholesky factor and keeping those that satisfy every declared sign at
every declared horizon; :class:`SignSVARResult` holds the accepted rotations
themselves, and its quantile surfaces are labelled as summaries of the set
rather than as the impulse responses of any single model, which no rotation
traces.

The rotations are drawn at the reduced-form point estimate: the set reflects
identification uncertainty only, and says so in its summary. Reduced-form
parameter uncertainty belongs to the posterior-draw version that arrives with
the sampling backend.

References:
    Uhlig, H. (2005). What are the effects of monetary policy on output?
        *Journal of Monetary Economics*, 52(2), 381-419.
    Rubio-Ramirez, J. F., Waggoner, D. F., & Zha, T. (2010). Structural vector
        autoregressions: Theory of identification and algorithms for
        inference. *Review of Economic Studies*, 77(2), 665-696.
    Fry, R., & Pagan, A. (2011). Sign restrictions in structural vector
        autoregressions: A critical review. *Journal of Economic Literature*,
        49(4), 938-960.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import (
    _NARRATIVE_NOTE,
    _SIGN_QUANTILE_NOTE,
    _UNIT_SHOCK_NOTE,
    ClosedSystemResult,
    SummaryTable,
    _accepted_rotations,
    _lower_cholesky,
    _narrative_rotations,
    _validate_narrative_events,
    _validate_sign_patterns,
)
from ..._internals import _IdentificationModel, _SummaryMixin
from ...exceptions import SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SignRestrictedSVARResult(_SummaryMixin):
    """The accepted set of a sign-restricted identification.

    Not a point estimate wearing bands: every accepted rotation is a complete
    structural model consistent with both the reduced form and the declared
    signs, and the object holds all of them. The quantile surfaces are
    summaries of that set, computed pointwise, and the summary says what that
    means rather than letting the median band read as a model.

    Attributes:
        source: The reduced-form result the rotations factorize.
        impacts: ``(n_accepted, k, k)`` accepted impact matrices.
        shock_names: One label per column; restricted shocks first, in the
            order they were declared.
        horizons: The leads at which the sign restrictions were imposed.
        restriction: The declared signs, as a sentence.
        requested: Accepted draws asked for.
        attempts: Rotations actually drawn.
    """

    source: ClosedSystemResult = field(repr=False)
    impacts: npt.NDArray[np.float64] = field(repr=False)
    shock_names: tuple[str, ...]
    horizons: tuple[int, ...]
    restriction: str
    requested: int
    attempts: int

    @property
    def names(self) -> tuple[str, ...]:
        """Variable labels, from the reduced form."""
        return self.source.names

    @property
    def k_endog(self) -> int:
        """Number of variables."""
        return len(self.source.names)

    @property
    def n_accepted(self) -> int:
        """Accepted rotations in the set."""
        return int(self.impacts.shape[0])

    @property
    def acceptance_rate(self) -> float:
        """Accepted rotations per draw, a direct read on how binding the signs are."""
        return self.n_accepted / self.attempts

    def irf_draws(self, horizon: int = 20, *, cumulative: bool = False) -> npt.NDArray[np.float64]:
        """Structural impulse responses for every accepted rotation.

        Args:
            horizon: Largest lead to return.
            cumulative: Return running sums.

        Returns:
            An array of shape ``(n_accepted, horizon + 1, k, k)``.
        """
        psi = self.source.ma_representation(horizon)
        theta = np.einsum("hik,nkj->nhij", psi, self.impacts)
        return np.cumsum(theta, axis=1) if cumulative else theta

    def irf(
        self,
        horizon: int = 20,
        *,
        quantiles: Sequence[float] = (0.16, 0.5, 0.84),
        cumulative: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Pointwise quantiles of the impulse responses across the accepted set.

        Args:
            horizon: Largest lead to return.
            quantiles: Probability levels, each in ``(0, 1)``.
            cumulative: Return running sums before taking quantiles.

        Returns:
            An array of shape ``(len(quantiles), horizon + 1, k, k)``. Read it
            as a summary of the identified set: no single rotation traces any
            one of these surfaces.

        Raises:
            SpecificationError: If a quantile is outside the open unit
                interval.
        """
        levels = tuple(float(q) for q in quantiles)
        if any(not 0.0 < q < 1.0 for q in levels):
            raise SpecificationError(f"quantiles must lie in (0, 1); got {levels}.")
        draws = self.irf_draws(horizon, cumulative=cumulative)
        return np.quantile(draws, levels, axis=0)

    def fevd(
        self, horizon: int = 20, *, quantiles: Sequence[float] = (0.16, 0.5, 0.84)
    ) -> npt.NDArray[np.float64]:
        """Pointwise quantiles of the variance decomposition across the set.

        Each accepted rotation is complete, so within a rotation the shares
        sum to one; the quantile surfaces, being pointwise, need not.

        Args:
            horizon: Largest lead to return.
            quantiles: Probability levels, each in ``(0, 1)``.

        Returns:
            An array of shape ``(len(quantiles), horizon + 1, k, k)``.

        Raises:
            SpecificationError: If a quantile is outside the open unit
                interval.
        """
        levels = tuple(float(q) for q in quantiles)
        if any(not 0.0 < q < 1.0 for q in levels):
            raise SpecificationError(f"quantiles must lie in (0, 1); got {levels}.")
        theta = self.irf_draws(horizon)
        explained = np.cumsum(theta**2, axis=1)
        shares = explained / explained.sum(axis=3, keepdims=True)
        return np.quantile(shares, levels, axis=0)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        return SummaryTable(
            title="Sign-Restricted SVAR (identified set)",
            metadata=(
                ("Scheme", "sign restrictions"),
                ("Accepted", f"{self.n_accepted}"),
                ("Requested", f"{self.requested}"),
                ("Draws", f"{self.attempts}"),
                ("Acceptance rate", f"{self.acceptance_rate:.2%}"),
                ("Horizons", ", ".join(str(h) for h in self.horizons)),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.source.nobs}"),
            ),
            notes=(self.restriction, _UNIT_SHOCK_NOTE, _SIGN_QUANTILE_NOTE),
        )


class SignRestrictedSVAR(_IdentificationModel[SignRestrictedSVARResult]):
    """Set identification by declared response signs, Uhlig (2005) via RWZ draws.

    Rotations of the Cholesky factor are drawn uniformly and kept when every
    declared sign holds at every declared horizon. A restricted column may be
    flipped wholesale, since a shock and its negative are the same rotation;
    what is never done is relabeling columns to rescue a draw, so a shock's
    identity is its declared position.

    Args:
        result: The fitted closed reduced-form result to identify.
        restrictions: Mapping from shock label to its sign pattern, itself a
            mapping from variable name to ``"+"`` or ``"-"``. Declared shocks
            occupy the leading columns in declaration order; remaining columns
            are unrestricted.
        horizons: Leads at which every declared sign must hold.
        draws: Accepted rotations to collect.
        max_attempts: Rotations to try before giving up; defaults to one
            thousand per requested draw.
        seed: Seed for the rotation draws, so an identified set is
            reproducible.

    Raises:
        SpecificationError: If the result is not a closed system, or the
            declaration, horizons, or draw budget are malformed.
    """

    __slots__ = ("_budget", "_compiled", "_draws", "_labels", "_leads", "_restriction", "_seed")

    def __init__(
        self,
        result: ClosedSystemResult,
        restrictions: Mapping[str, Mapping[str, str]],
        *,
        horizons: Sequence[int] = (0,),
        draws: int = 1000,
        max_attempts: int | None = None,
        seed: int | None = 0,
    ) -> None:
        """Validate the source system and the full declaration."""
        super().__init__(result)
        if draws < 1:
            raise SpecificationError(f"draws must be at least 1; got {draws}.")
        self._draws = int(draws)
        leads = tuple(int(h) for h in horizons)
        if not leads or any(h < 0 for h in leads):
            raise SpecificationError(
                f"horizons must be a non-empty collection of non-negative "
                f"leads; got {tuple(horizons)}."
            )
        self._leads = leads
        self._compiled = _validate_sign_patterns(restrictions, self.names)
        budget = self._draws * 1000 if max_attempts is None else int(max_attempts)
        if budget < 1:
            raise SpecificationError(f"max_attempts must be at least 1; got {budget}.")
        self._budget = budget
        self._seed = seed
        self._labels = tuple(restrictions) + tuple(
            f"unrestricted{j + 1}" for j in range(self.k_endog - len(self._compiled))
        )
        self._restriction = (
            f"Declared signs, holding at horizons {leads}: "
            + "; ".join(
                f"{label}: " + ", ".join(f"{variable} {sign}" for variable, sign in pattern.items())
                for label, pattern in restrictions.items()
            )
            + ". Rotations were drawn at the reduced-form point estimate, so "
            "the set reflects identification uncertainty only."
        )

    def identify(self) -> SignRestrictedSVARResult:
        """Draw rotations and keep those satisfying every declared sign.

        Returns:
            The accepted set.

        Raises:
            SpecificationError: If no rotation is accepted within the attempt
                budget -- the signs are jointly unsatisfiable at this reduced
                form, or nearly so.
            NumericalError: If the innovation covariance is not positive
                definite.
        """
        sigma = np.asarray(self.source.sigma_u, dtype=np.float64)
        factor = _lower_cholesky(sigma, "sigma_u")
        psi = self.source.ma_representation(max(self._leads))[list(self._leads)]
        impacts, attempts = _accepted_rotations(
            factor,
            psi,
            self._compiled,
            draws=self._draws,
            budget=self._budget,
            rng=np.random.default_rng(self._seed),
        )
        return SignRestrictedSVARResult(
            source=self.source,
            impacts=impacts,
            shock_names=self._labels,
            horizons=self._leads,
            restriction=self._restriction,
            requested=self._draws,
            attempts=attempts,
        )


class NarrativeSignRestrictedSVAR(_IdentificationModel[SignRestrictedSVARResult]):
    """Sign restrictions sharpened by declared history, Antolin-Diaz and Rubio-Ramirez (2018).

    Traditional signs restrict what a shock *would do*; narrative events
    restrict what the shocks *did*. Two kinds are declared, both from the
    paper. A shock-sign event states the sign of a named shock in a named
    period -- the monetary shock was contractionary in the Volcker quarter. A
    contribution event states that a named shock was the ``"most"`` important
    contributor, or the ``"overwhelming"`` one -- larger than all others
    combined -- to the unexpected movement of a named variable in a named
    period. Each accepted rotation must reproduce the declared history through
    its own recovered shocks, which is what makes the resulting set sharper
    than the traditional one: history is a filter rotations rarely pass by
    accident.

    Periods index the effective sample -- one row per residual row of the
    result being identified -- exactly as a proxy's instrument rows do. The
    events are checked against the point-estimate residuals, and the summary
    carries the caveat that the full posterior treatment belongs to the
    sampling backend.

    Args:
        result: The fitted closed reduced-form result to identify.
        restrictions: Traditional sign patterns, mapping shock label to a
            mapping from variable to ``"+"`` or ``"-"``. Optional: narrative
            events alone can identify. Labels declared here occupy the leading
            columns, in declaration order; labels appearing only in narrative
            events follow, in order of first appearance.
        shock_signs: Shock-sign events, each ``(shock label, period, sign)``.
        contributions: Contribution events, each ``(shock label, variable,
            period, kind)`` with ``kind`` one of ``"most"`` or
            ``"overwhelming"``.
        horizons: Leads at which every traditional sign must hold.
        draws: Accepted rotations to collect.
        max_attempts: Rotations to try before giving up; defaults to one
            thousand per requested draw.
        seed: Seed for the rotation draws.

    Raises:
        SpecificationError: If the result is not a closed system, no narrative
            event is declared -- :class:`SignSVAR` is that model -- or any
            declaration is malformed.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> from cultivars.multivariate.reduced_form import VAR
        >>> y = np.diff(rng.standard_normal((121, 2)).cumsum(axis=0) * 0.1, axis=0)
        >>> res = VAR(y, order=1, names=("gdp", "infl")).fit()
        >>> sset = NarrativeSignSVAR(
        ...     res,
        ...     {"demand": {"gdp": "+", "infl": "+"}},
        ...     shock_signs=(("demand", 50, "+"),),
        ...     draws=50,
        ... ).identify()
        >>> sset.n_accepted
        50
    """

    __slots__ = (
        "_budget",
        "_compiled",
        "_contribution_events",
        "_draws",
        "_labels",
        "_leads",
        "_periods",
        "_restriction",
        "_seed",
        "_shock_events",
    )

    def __init__(
        self,
        result: ClosedSystemResult,
        restrictions: Mapping[str, Mapping[str, str]] | None = None,
        *,
        shock_signs: Sequence[tuple[str, int, str]] = (),
        contributions: Sequence[tuple[str, str, int, str]] = (),
        horizons: Sequence[int] = (0,),
        draws: int = 1000,
        max_attempts: int | None = None,
        seed: int | None = 0,
    ) -> None:
        """Validate the source system and every declaration."""
        super().__init__(result)
        k = self.k_endog
        if not shock_signs and not contributions:
            raise SpecificationError(
                "declare at least one narrative event; with traditional signs "
                "alone, SignSVAR is that model."
            )
        if draws < 1:
            raise SpecificationError(f"draws must be at least 1; got {draws}.")
        self._draws = int(draws)
        leads = tuple(int(h) for h in horizons)
        if not leads or any(h < 0 for h in leads):
            raise SpecificationError(
                f"horizons must be a non-empty collection of non-negative "
                f"leads; got {tuple(horizons)}."
            )
        self._leads = leads
        declared = dict(restrictions) if restrictions else {}
        traditional = _validate_sign_patterns(declared, self.names) if declared else ()
        labels: list[str] = list(declared)
        for event_label in (
            *(label for label, _, _ in shock_signs),
            *(label for label, _, _, _ in contributions),
        ):
            if event_label not in labels:
                labels.append(str(event_label))
        if len(labels) > k:
            raise SpecificationError(
                f"{len(labels)} labelled shocks exceed the {k} shocks the system has."
            )
        full_labels = tuple(labels) + tuple(f"unrestricted{j + 1}" for j in range(k - len(labels)))
        self._labels = full_labels
        self._compiled = tuple(
            traditional[position] if position < len(traditional) else () for position in range(k)
        )
        nobs_resid = int(np.asarray(result.resid).shape[0])
        self._shock_events, self._contribution_events, self._periods = _validate_narrative_events(
            tuple(shock_signs),
            tuple(contributions),
            labels=full_labels,
            names=self.names,
            nobs=nobs_resid,
        )
        budget = self._draws * 1000 if max_attempts is None else int(max_attempts)
        if budget < 1:
            raise SpecificationError(f"max_attempts must be at least 1; got {budget}.")
        self._budget = budget
        self._seed = seed
        described_signs = (
            "; ".join(
                f"{label}: " + ", ".join(f"{variable} {sign}" for variable, sign in pattern.items())
                for label, pattern in declared.items()
            )
            if declared
            else "none"
        )
        described_events = "; ".join(
            (
                *(
                    f"{label} shock {sign} in period {period}"
                    for label, period, sign in shock_signs
                ),
                *(
                    f"{label} the {kind} contributor to {variable} in period {period}"
                    for label, variable, period, kind in contributions
                ),
            )
        )
        self._restriction = (
            f"Traditional signs at horizons {leads}: {described_signs}. "
            f"Narrative events, indexed to the effective sample: "
            f"{described_events}."
        )

    def identify(self) -> SignRestrictedSVARResult:
        """Draw rotations and keep those reproducing signs and history alike.

        Returns:
            The accepted set.

        Raises:
            SpecificationError: If no rotation is accepted within the attempt
                budget -- the narrative events contradict the traditional
                signs, or the history at this reduced form.
            NumericalError: If the innovation covariance is not positive
                definite.
        """
        sigma = np.asarray(self.source.sigma_u, dtype=np.float64)
        factor = _lower_cholesky(sigma, "sigma_u")
        psi = self.source.ma_representation(max(self._leads))[list(self._leads)]
        residual_events = np.asarray(self.source.resid, dtype=np.float64)[list(self._periods)]
        impacts, attempts = _narrative_rotations(
            factor,
            psi,
            self._compiled,
            shock_events=self._shock_events,
            contribution_events=self._contribution_events,
            residual_events=residual_events,
            draws=self._draws,
            budget=self._budget,
            rng=np.random.default_rng(self._seed),
        )
        return SignRestrictedSVARResult(
            source=self.source,
            impacts=impacts,
            shock_names=self._labels,
            horizons=self._leads,
            restriction=self._restriction + " " + _NARRATIVE_NOTE,
            requested=self._draws,
            attempts=attempts,
        )
