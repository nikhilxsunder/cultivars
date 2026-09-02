# filepath: /src/cultivars/multivariate/structural/set_identification.py
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

"""Set identification: the bounds of the set, not a prior's summary of it.

A sign-restricted model is set-identified, and there are two honest ways to
report the set. :mod:`~cultivars.multivariate.structural.sign_restrictions`
samples it -- quantile bands over accepted rotations, which inherit the
uniform prior over rotations whether or not anyone meant to impose one, the
Baumeister-Hamilton critique. This module reports the set itself: for each
response of each variable at each horizon, the exact smallest and largest
value any admissible rotation delivers. No prior, no draws, no inner
approximation.

The exactness comes from the geometry. Restrictions on a single shock confine
its rotation column to the unit sphere intersected with a polyhedral cone,
every response to that shock is linear in the column, and a linear functional
on that region attains its extrema on the faces -- so the bound is found by
projecting the objective onto each face's span and keeping the feasible
candidates (Gafarov, Meier, and Montiel Olea 2018). Zero restrictions ride
along by projecting the whole problem onto their null space first. What does
not fit this geometry is a joint declaration across several shocks, whose
identified set has no face-enumeration form; that case belongs to the
sampling surface, and this model says so rather than approximating.

References:
    Giacomini, R., & Kitagawa, T. (2021). Robust Bayesian inference for
        set-identified models. *Econometrica*, 89(4), 1519-1556.
    Gafarov, B., Meier, M., & Montiel Olea, J. L. (2018). Delta-method
        inference for a class of set-identified SVARs. *Journal of
        Econometrics*, 203(2), 316-327.
    Baumeister, C., & Hamilton, J. D. (2015). Sign restrictions, structural
        vector autoregressions, and useful prior information. *Econometrica*,
        83(5), 1963-1999.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..._core import (
    _SET_BOUNDS_NOTE,
    _UNIT_SHOCK_NOTE,
    ClosedSystemResult,
    SummaryTable,
    _lower_cholesky,
    _validate_sign_patterns,
    _null_basis,
    _sphere_extrema,
    _face_projectors,
)
from ..._internals import (
    _IdentificationModel,
    _SummaryMixin,
)
from ...exceptions import SpecificationError


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SetIdentifiedSVARResult(_SummaryMixin):
    """The exact bounds of a set-identified shock's response surface.

    Holds the geometry of the identified set -- the whitening factor, the
    equality null-space basis, the sign cone, and its precomputed face
    projectors -- so that bounds at any horizon are cheap queries rather than
    fresh enumerations. Every number it reports is an endpoint of the set,
    pointwise per response and horizon, and the summary carries the two
    caveats that keep those endpoints honest: they are not a prior's
    quantiles, and the box they outline is not the set of paths.

    Attributes:
        source: The reduced-form result the set was computed from.
        shock: Label of the restricted shock.
        restriction: The declared restrictions, stated as a sentence.
        horizons: Leads at which the declared signs bind.
    """

    source: ClosedSystemResult = field(repr=False)
    shock: str
    restriction: str
    horizons: tuple[int, ...]
    _factor: npt.NDArray[np.float64] = field(repr=False)
    _basis: npt.NDArray[np.float64] = field(repr=False)
    _inequalities: npt.NDArray[np.float64] = field(repr=False)
    _projectors: tuple[npt.NDArray[np.float64], ...] = field(repr=False)

    @property
    def names(self) -> tuple[str, ...]:
        """Variable labels, from the reduced form."""
        return self.source.names

    @property
    def k_endog(self) -> int:
        """Number of variables."""
        return len(self.source.names)

    @property
    def impact_bounds(self) -> npt.NDArray[np.float64]:
        """Exact ``(k, 2)`` lower and upper impact responses to the shock."""
        return self.irf(0)[0]

    def irf(
        self, horizon: int = 20, *, cumulative: bool = False
    ) -> npt.NDArray[np.float64]:
        """Exact pointwise response bounds to the restricted shock.

        Args:
            horizon: Largest lead to return.
            cumulative: Bound the running sums instead, for differenced data
                and level questions -- the cumulated response is also linear
                in the rotation column, so the bounds stay exact.

        Returns:
            An array of shape ``(horizon + 1, k, 2)``; entry ``[h, i]`` is
            the ``(lower, upper)`` pair for variable ``i``'s response at lead
            ``h`` to one standard deviation of the restricted shock. Read the
            bounds pointwise: no single admissible rotation traces an edge of
            the box.
        """
        psi = self.source.ma_representation(horizon)
        loadings = np.einsum("hik,kl->hil", psi, self._factor)
        if cumulative:
            loadings = np.cumsum(loadings, axis=0)
        targets = loadings.reshape(-1, self.k_endog) @ self._basis
        bounds = _sphere_extrema(targets, self._inequalities, self._projectors)
        return bounds.reshape(horizon + 1, self.k_endog, 2)

    def _summary_table(self) -> SummaryTable:
        """Build the structured summary."""
        bounds = self.impact_bounds
        rows = tuple(
            (name, f"{bounds[i, 0]:.4f}", f"{bounds[i, 1]:.4f}")
            for i, name in enumerate(self.names)
        )
        return SummaryTable(
            title=f"Set-Identified SVAR ({self.shock!r} shock)",
            metadata=(
                ("Scheme", "set identification"),
                ("Restricted shock", self.shock),
                ("Sign horizons", ", ".join(str(h) for h in self.horizons)),
                ("Variables", f"{self.k_endog}"),
                ("Observations", f"{self.source.nobs}"),
            ),
            columns=("impact of " + self.shock, "lower", "upper"),
            rows=rows,
            notes=(self.restriction, _UNIT_SHOCK_NOTE, _SET_BOUNDS_NOTE),
        )


class SetIdentifiedSVAR(_IdentificationModel[SetIdentifiedSVARResult]):
    """Exact response bounds under restrictions on one shock, Giacomini-Kitagawa (2021).

    Declares what :class:`~cultivars.multivariate.structural.SignSVAR`
    declares -- signs of one shock's responses, at chosen horizons, plus
    optional exact zeros -- but answers a different question: not "what does a
    uniform prior over admissible rotations imply," but "what is the smallest
    and largest each response could be, over every admissible rotation." The
    identified set's endpoints are computed exactly, by enumerating the faces
    of the sphere-and-cone region the restrictions cut and projecting each
    objective onto them.

    One shock only, by the geometry: bounds for a single restricted column
    have a face-enumeration form; a joint declaration across several shocks
    does not, and belongs to the sampling surface.

    Args:
        result: The fitted closed reduced-form result.
        restrictions: The restricted shock's sign pattern, mapping variable to
            ``"+"`` or ``"-"``.
        shock: Label for the restricted shock.
        horizons: Leads at which every declared sign must hold.
        zeros: Optional exact-zero responses, mapping variable to the leads at
            which its response to this shock is zero. At most ``k - 1`` zero
            restrictions can bind before no direction remains.

    Raises:
        SpecificationError: If the result is not a closed system, the
            declaration is malformed, the zeros span the whole space, or the
            face count exceeds what exact enumeration can afford.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> from cultivars.multivariate.reduced_form import VAR
        >>> y = np.diff(rng.standard_normal((121, 2)).cumsum(axis=0) * 0.1, axis=0)
        >>> res = VAR(y, order=1, names=("gdp", "infl")).fit()
        >>> sset = SetIdentifiedSVAR(
        ...     res, {"gdp": "+", "infl": "+"}, shock="demand"
        ... ).identify()
        >>> bool(sset.impact_bounds[0, 0] >= 0.0)
        True
    """

    __slots__ = ("_leads", "_restriction", "_shock", "_sign_cells", "_zero_cells")

    def __init__(
        self,
        result: ClosedSystemResult,
        restrictions: Mapping[str, str],
        *,
        shock: str = "restricted",
        horizons: Sequence[int] = (0,),
        zeros: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        """Validate the source system and the single-shock declaration."""
        super().__init__(result)
        self._shock = str(shock)
        compiled = _validate_sign_patterns({self._shock: dict(restrictions)}, self.names)
        self._sign_cells = compiled[0]
        leads = tuple(int(h) for h in horizons)
        if not leads or any(h < 0 for h in leads):
            raise SpecificationError(
                f"horizons must be a non-empty collection of non-negative "
                f"leads; got {tuple(horizons)}."
            )
        self._leads = leads
        zero_cells: list[tuple[int, int]] = []
        for variable, moments in (zeros or {}).items():
            if variable not in self.names:
                raise SpecificationError(
                    f"unknown variable {variable!r} in zeros; expected one of "
                    f"{self.names}."
                )
            for moment in moments:
                lead = int(moment)
                if lead < 0:
                    raise SpecificationError(
                        f"zero restriction on {variable!r} names a negative "
                        f"lead {moment}."
                    )
                zero_cells.append((self.names.index(variable), lead))
        if len(zero_cells) >= self.k_endog:
            raise SpecificationError(
                f"{len(zero_cells)} zero restrictions in a {self.k_endog}-"
                "variable system leave no direction for the shock; at most "
                f"{self.k_endog - 1} can bind."
            )
        self._zero_cells = tuple(zero_cells)
        described_zeros = (
            "; zeros: "
            + ", ".join(
                f"{self.names[i]} at lead {h}" for i, h in self._zero_cells
            )
            if self._zero_cells
            else ""
        )
        self._restriction = (
            f"Declared signs on the {self._shock!r} shock, holding at "
            f"horizons {leads}: "
            + ", ".join(
                f"{self.names[i]} {'+' if sign > 0 else '-'}"
                for i, sign in self._sign_cells
            )
            + described_zeros
            + ". Every other shock is left entirely unrestricted."
        )

    def identify(self) -> SetIdentifiedSVARResult:
        """Build the identified set's geometry and precompute its faces.

        Returns:
            The bounds result, with impact bounds ready in its summary and
            arbitrary-horizon bounds available as queries.

        Raises:
            SpecificationError: If the zero restrictions leave no direction,
                or the face count exceeds the enumeration cap.
            NumericalError: If the innovation covariance is not positive
                definite, or the declared restrictions are infeasible -- the
                identified set is empty at this reduced form.
        """
        k = self.k_endog
        sigma = np.asarray(self.source.sigma_u, dtype=np.float64)
        factor = _lower_cholesky(sigma, "sigma_u")
        top = max(
            (*(h for _, h in self._zero_cells), *self._leads)
        )
        psi = self.source.ma_representation(top)
        loadings = np.einsum("hik,kl->hil", psi, factor)

        equality_rows = (
            np.stack([loadings[h, i] for i, h in self._zero_cells])
            if self._zero_cells
            else np.empty((0, k))
        )
        basis = _null_basis(equality_rows, k)
        inequality_rows = np.stack(
            [
                sign * loadings[h, i] @ basis
                for h in self._leads
                for i, sign in self._sign_cells
            ]
        )
        projectors = _face_projectors(inequality_rows)
        result = SetIdentifiedSVARResult(
            source=self.source,
            shock=self._shock,
            restriction=self._restriction,
            horizons=self._leads,
            _factor=factor,
            _basis=basis,
            _inequalities=inequality_rows,
            _projectors=projectors,
        )
        result.impact_bounds  # noqa: B018 -- feasibility surfaces here, at identify time
        return result