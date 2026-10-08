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
r"""Set identification: the bounds of the set, not a prior's summary of it.

A sign-restricted model is set-identified, and there are two honest ways to
report the set. :mod:`~cultivars.multivariate.structural.sign_restrictions`
samples it -- quantile bands over accepted rotations, which inherit the
uniform prior over rotations whether or not anyone meant to impose one, the
Baumeister-Hamilton critique. This module reports the set itself: for each
response of each variable at each horizon, the exact smallest and largest
value any admissible rotation delivers. No prior, no draws, no inner
approximation.

The exactness comes from the geometry. With :math:`\Sigma_u = L L'` and
the moving-average matrices :math:`\Psi_h`, restrictions on one shock
confine its rotation column :math:`q` to

.. math::

   \mathcal{Q} = \{\, q \in \mathbb{R}^k : \lVert q \rVert = 1,\;
   Z q = 0,\; S q \ge 0 \,\},

with each row of :math:`Z` and :math:`S` a row :math:`e_i' \Psi_h L` of a
response the user has pinned to zero or signed at lead :math:`h`. Every
response to the shock, :math:`e_i' \Psi_h L q`, is linear in :math:`q`, and
a linear functional on the sphere cut by a polyhedral cone attains its
extrema on the region's faces, so each bound is the largest or smallest
feasible value among the normalized projections of the objective onto every
face's span (Gafarov, Meier, and Montiel Olea 2018). Zero restrictions ride
along by projecting the whole problem onto their null space first, and
``k - 1`` of them on one column collapse the set to the recursive point.
What does not fit this geometry is a joint declaration across several
shocks, whose identified set has no face-enumeration form; that case
belongs to the sampling surface, and this model says so rather than
approximating.

Two commitments shape the surface. First, the bounds are the deliverable
and nothing is averaged over them: no posterior mean, no median path, no
band -- a result carries the endpoints and the geometry that produced them,
and the summary says in its notes that they depend on no prior and must be
read pointwise, since the box the bounds outline across horizons is not the
set of admissible paths. Second, exactness is not traded for reach. The
face count is combinatorial in the number of sign rows -- every declared
sign at every declared horizon is a row -- so the enumeration is capped,
and a declaration past the cap is refused with the count, not approximated
by sampling behind the user's back; an empty set, a declaration no rotation
satisfies, is likewise a refusal rather than a silent degenerate bound.

Layout. :class:`SetIdentifiedSVAR` is an ``_IdentificationModel`` from
``_internals``; the sign pattern is compiled by ``_validate_sign_patterns``
in ``_core``, the whitening factor comes from ``_lower_cholesky``, the zero
restrictions are reduced away by ``_null_basis``, the faces are enumerated
once by ``_face_projectors``, and every bound is a call to
``_sphere_extrema``, all in ``_core``. :class:`SetIdentifiedSVARResult`
holds that geometry so that bounds at any horizon are a query, and renders
through the shared ``_SET_BOUNDS_NOTE`` and ``_UNIT_SHOCK_NOTE``. The same
declaration sampled, and the only route for several restricted shocks, is
:mod:`~cultivars.multivariate.structural.sign_restrictions`.

References:
    Giacomini, R., & Kitagawa, T. (2021). Robust Bayesian inference for
    set-identified models. *Econometrica*, 89(4), 1519-1556.

    Gafarov, B., Meier, M., & Montiel Olea, J. L. (2018). Delta-method
    inference for a class of set-identified SVARs. *Journal of
    Econometrics*, 203(2), 316-327.

    Baumeister, C., & Hamilton, J. D. (2015). Sign restrictions, structural
    vector autoregressions, and useful prior information. *Econometrica*,
    83(5), 1963-1999.

Example:
    The same declaration on both surfaces. Every accepted rotation of the
    sampled set lies inside the exact bounds, and the 16-84 band sits
    strictly inside them: the band is a statement about the uniform prior,
    the bounds about the data:

    >>> import numpy as np
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> from cultivars.multivariate.structural.sign_restrictions import SignRestrictedSVAR
    >>> rng = np.random.default_rng(0)
    >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
    >>> A = np.array([[0.5, 0.1], [0.0, 0.4]])
    >>> eps = rng.standard_normal((402, 2))
    >>> y = np.zeros((402, 2))
    >>> for t in range(1, 402):
    ...     y[t] = A @ y[t - 1] + B @ eps[t]
    >>> res = VAR(y, order=1, names=("gdp", "infl")).fit()
    >>> signs = {"gdp": "+", "infl": "+"}
    >>> exact = SetIdentifiedSVAR(res, signs, shock="demand").identify().irf(4)
    >>> sampled = SignRestrictedSVAR(res, {"demand": signs}, draws=500, seed=0).identify()
    >>> draws = sampled.irf_draws(4)[:, :, :, 0]
    >>> bool(np.all((exact[..., 0] <= draws) & (draws <= exact[..., 1])))
    True
    >>> band = sampled.irf(4)[:, :, :, 0]
    >>> bool(np.all(exact[..., 0] < band[0]) and np.all(band[2] < exact[..., 1]))
    True
    >>> exact[0].round(3).tolist(), band[[0, 2], 0].round(3).tolist()
    ([[0.0, 1.123], [0.0, 1.042]], [[0.383, 0.413], [1.102, 1.024]])
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
    _face_projectors,
    _lower_cholesky,
    _null_basis,
    _sphere_extrema,
    _validate_sign_patterns,
)
from ..._internals import (
    _IdentificationModel,
    _SummaryMixin,
)
from ...exceptions import SpecificationError

__all__ = ["SetIdentifiedSVAR", "SetIdentifiedSVARResult"]


@dataclass(frozen=True, kw_only=True, slots=True, repr=False)
class SetIdentifiedSVARResult(_SummaryMixin):
    r"""The exact bounds of a set-identified shock's response surface.

    Restrictions on one shock confine its rotation column :math:`q` to the
    unit sphere cut by a polyhedral cone,

    .. math::

       \mathcal{Q} = \{\, q : \lVert q \rVert = 1,\;
       Z q = 0,\; S q \ge 0 \,\},

    where the rows of :math:`Z` are the zero-restricted responses and the
    rows of :math:`S` the sign-restricted ones, each written as the row of
    :math:`\Psi_h L` for the lead :math:`h` it binds at. Every response to
    the shock, :math:`e_i' \Psi_h L q`, is linear in :math:`q`, so its
    smallest and largest values over :math:`\mathcal{Q}` are attained on
    faces of the region and can be enumerated exactly. The record holds
    that geometry -- the whitening factor :math:`L`, an orthonormal basis of
    the null space of :math:`Z`, the sign rows projected onto it, and the
    precomputed face projectors -- so that bounds at any horizon are a
    cheap query rather than a fresh enumeration. Every number it reports is
    an endpoint of the set, pointwise per response and horizon.

    Note:
        Two caveats keep the endpoints honest, and the summary carries both.
        The bounds depend on no prior over rotations, which is what separates
        them from the quantile bands of
        :class:`~cultivars.multivariate.structural.sign_restrictions.SignRestrictedSVARResult`;
        and they are pointwise, so the box they outline across horizons is
        not the set of admissible paths -- no single rotation traces an edge
        of it. They are computed at the reduced-form point estimate.

    Attributes:
        source: The reduced-form result the set was computed from.
        shock: Label of the restricted shock.
        restriction: The declared restrictions, stated as a sentence.
        horizons: Leads at which the declared signs bind.

    See Also:
        * :class:`SetIdentifiedSVAR` -- the model whose ``identify()``
          returns this record.
        * :class:`~cultivars.multivariate.structural.sign_restrictions.SignRestrictedSVARResult`
          -- the sampled set, with quantile bands under the uniform prior
          over rotations.

    References:
        Giacomini, R., & Kitagawa, T. (2021). Robust Bayesian inference for
        set-identified models. *Econometrica*, 89(4), 1519-1556.

        Gafarov, B., Meier, M., & Montiel Olea, J. L. (2018). Delta-method
        inference for a class of set-identified SVARs. *Journal of
        Econometrics*, 203(2), 316-327.

    Example:
        Two variables, both restricted to rise on impact. With only sign
        restrictions in two dimensions the lower bounds are zero -- the
        column can lie on either edge of the cone -- and each upper bound
        is the variable's own innovation standard deviation, the largest
        response any unit column can produce:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> rng = np.random.default_rng(0)
        >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
        >>> A = np.array([[0.5, 0.1], [0.0, 0.4]])
        >>> eps = rng.standard_normal((402, 2))
        >>> y = np.zeros((402, 2))
        >>> for t in range(1, 402):
        ...     y[t] = A @ y[t - 1] + B @ eps[t]
        >>> res = VAR(y, order=1, names=("gdp", "infl")).fit()
        >>> sset = SetIdentifiedSVAR(res, {"gdp": "+", "infl": "+"}, shock="demand").identify()
        >>> sset.shock, sset.horizons, sset.names, sset.k_endog
        ('demand', (0,), ('gdp', 'infl'), 2)
        >>> sset.impact_bounds.round(3).tolist()
        [[0.0, 1.123], [0.0, 1.042]]
        >>> bool(np.allclose(sset.impact_bounds[:, 1], np.sqrt(np.diag(res.sigma_u))))
        True
        >>> b = sset.impact_bounds
        >>> bool(np.all((b[:, 0] <= B[:, 0]) & (B[:, 0] <= b[:, 1])))
        True
    """

    source: ClosedSystemResult = field(repr=False)
    """The closed reduced-form result the set was computed from. Kept out of the repr."""

    shock: str
    """Label of the restricted shock."""

    restriction: str
    """The declared signs, horizons and zeros, stated as a sentence."""

    horizons: tuple[int, ...]
    """Leads at which the declared signs bind."""

    _factor: npt.NDArray[np.float64] = field(repr=False)
    """The ``(k, k)`` lower Cholesky factor of ``sigma_u``. Kept out of the repr."""

    _basis: npt.NDArray[np.float64] = field(repr=False)
    """The ``(k, d)`` orthonormal null-space basis of the zero restrictions.

    The identity when there are none. Kept out of the repr.
    """

    _inequalities: npt.NDArray[np.float64] = field(repr=False)
    """The ``(m, d)`` sign rows in the reduced coordinates. Kept out of the repr."""

    _projectors: tuple[npt.NDArray[np.float64], ...] = field(repr=False)
    """Orthonormal bases of every face span of the cone. Kept out of the repr."""

    @property
    def names(self) -> tuple[str, ...]:
        """Variable labels, from the reduced form.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1, names=("gdp", "infl")).fit()
            >>> SetIdentifiedSVAR(res, {"gdp": "+"}).identify().names
            ('gdp', 'infl')
        """
        return self.source.names

    @property
    def k_endog(self) -> int:
        """Number of variables.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 3)), order=1).fit()
            >>> SetIdentifiedSVAR(res, {"y1": "+"}).identify().k_endog
            3
        """
        return len(self.source.names)

    @property
    def impact_bounds(self) -> npt.NDArray[np.float64]:
        """Exact ``(k, 2)`` lower and upper impact responses to the shock.

        Example:
            A single sign leaves the other variable's response free in sign:
            its upper bound is still its innovation standard deviation, and
            its lower bound is the most negative response compatible with
            the restricted variable rising:

            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
            >>> y = np.zeros((402, 2))
            >>> for t in range(1, 402):
            ...     y[t] = 0.4 * y[t - 1] + B @ rng.standard_normal(2)
            >>> res = VAR(y, order=1, names=("gdp", "infl")).fit()
            >>> one = SetIdentifiedSVAR(res, {"gdp": "+"}).identify()
            >>> one.impact_bounds.shape, one.impact_bounds.round(3).tolist()
            ((2, 2), [[0.0, 1.122], [-0.758, 1.041]])
        """
        return self.irf(0)[0]

    def irf(self, horizon: int = 20, *, cumulative: bool = False) -> npt.NDArray[np.float64]:
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

        Raises:
            NumericalError: If no rotation column satisfies the restrictions
                -- unreachable after a successful ``identify()``, which has
                already evaluated the impact bounds.

        Example:
            The true impact column lies inside the bounds at every lead, and
            the cumulative bound at lead one is at most the sum of the two
            pointwise upper bounds, because one rotation need not attain
            both:

            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
            >>> A = np.array([[0.5, 0.1], [0.0, 0.4]])
            >>> eps = rng.standard_normal((402, 2))
            >>> y = np.zeros((402, 2))
            >>> for t in range(1, 402):
            ...     y[t] = A @ y[t - 1] + B @ eps[t]
            >>> res = VAR(y, order=1, names=("gdp", "infl")).fit()
            >>> sset = SetIdentifiedSVAR(res, {"gdp": "+", "infl": "+"}, horizons=(0, 1)).identify()
            >>> bounds = sset.irf(4)
            >>> bounds.shape
            (5, 2, 2)
            >>> bounds[1].round(3).tolist()
            [[0.002, 0.636], [0.059, 0.361]]
            >>> true = np.einsum("hik,k->hi", res.ma_representation(4), B[:, 0])
            >>> bool(np.all((bounds[..., 0] <= true) & (true <= bounds[..., 1])))
            True
            >>> running = sset.irf(1, cumulative=True)
            >>> bool(np.all(running[1, :, 1] <= bounds[0, :, 1] + bounds[1, :, 1] + 1e-12))
            True
        """
        psi = self.source.ma_representation(horizon)
        loadings = np.einsum("hik,kl->hil", psi, self._factor)
        if cumulative:
            loadings = np.cumsum(loadings, axis=0)
        targets = loadings.reshape(-1, self.k_endog) @ self._basis
        bounds = _sphere_extrema(targets, self._inequalities, self._projectors)
        return bounds.reshape(horizon + 1, self.k_endog, 2)

    def _summary_table(self) -> SummaryTable:
        """Structured summary rendered by every display path.

        One row per variable with its impact bounds; the scheme, shock,
        sign horizons and dimensions in the metadata; the restriction
        sentence, the unit-shock convention and the set-bounds caveat as
        notes.

        Returns:
            The :class:`~cultivars.summary.SummaryTable` behind ``summary()``.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1, names=("gdp", "infl")).fit()
            >>> sset = SetIdentifiedSVAR(res, {"gdp": "+"}, shock="demand").identify()
            >>> table = sset._summary_table()
            >>> table.title, [row[0] for row in table.rows]
            ("Set-Identified SVAR ('demand' shock)", ['gdp', 'infl'])
            >>> table.columns
            ('impact of demand', 'lower', 'upper')
            >>> dict(table.metadata)["Sign horizons"], len(table.notes)
            ('0', 3)
        """
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
    r"""Exact response bounds under restrictions on one shock, Giacomini-Kitagawa (2021).

    Declares what
    :class:`~cultivars.multivariate.structural.sign_restrictions.SignRestrictedSVAR`
    declares -- signs of one shock's responses at chosen horizons, plus
    optional exact zeros -- but answers a different question: not "what does
    a uniform prior over admissible rotations imply," but "what is the
    smallest and largest each response could be, over every admissible
    rotation." The identified set's endpoints are computed exactly. The
    restrictions cut the unit sphere by a polyhedral cone; a linear
    functional on that region attains its extrema on the region's faces;
    so each bound is the best feasible candidate among the normalized
    projections of the objective onto every face span (Gafarov, Meier and
    Montiel Olea 2018), with the zero restrictions handled first by
    projecting the whole problem onto their null space.

    One shock only, by the geometry: bounds for a single restricted column
    have a face-enumeration form; a joint declaration across several shocks
    does not, and belongs to the sampling surface. The face count is
    combinatorial in the number of sign rows -- every declared sign at
    every declared horizon is a row -- so the enumeration is capped, and a
    large system with signs over many horizons is refused rather than
    approximated.

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

    Attributes:
        _source: The closed reduced-form result being identified.
        _shock: The restricted shock's label.
        _sign_cells: The compiled ``(variable index, +-1.0)`` sign pairs.
        _leads: The horizons at which the signs bind.
        _zero_cells: The ``(variable index, lead)`` zero restrictions.
        _restriction: The declaration, stated as a sentence.

    See Also:
        * :class:`SetIdentifiedSVARResult` -- the record ``identify()``
          returns.
        * :class:`~cultivars.multivariate.structural.sign_restrictions.SignRestrictedSVAR`
          -- the same declaration, sampled: quantile bands under the uniform
          prior over rotations, and the only route for several restricted
          shocks at once.
        * :class:`~cultivars.multivariate.structural.zero_restrictions.RecursiveSVAR`
          -- the point-identified limit, which ``k - 1`` zeros on one column
          reproduce.

    References:
        Giacomini, R., & Kitagawa, T. (2021). Robust Bayesian inference for
        set-identified models. *Econometrica*, 89(4), 1519-1556.

        Gafarov, B., Meier, M., & Montiel Olea, J. L. (2018). Delta-method
        inference for a class of set-identified SVARs. *Journal of
        Econometrics*, 203(2), 316-327.

        Baumeister, C., & Hamilton, J. D. (2015). Sign restrictions,
        structural vector autoregressions, and useful prior information.
        *Econometrica*, 83(5), 1963-1999.

    Example:
        A demand shock that raises both output and inflation. The sign
        declaration alone leaves a wide set; adding one exact zero in a
        two-variable system collapses it to a point, and that point is the
        recursive impact column with the zero-restricted variable ordered
        first:

        >>> import numpy as np
        >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
        >>> from cultivars.multivariate.structural.zero_restrictions import RecursiveSVAR
        >>> rng = np.random.default_rng(0)
        >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
        >>> A = np.array([[0.5, 0.1], [0.0, 0.4]])
        >>> eps = rng.standard_normal((402, 2))
        >>> y = np.zeros((402, 2))
        >>> for t in range(1, 402):
        ...     y[t] = A @ y[t - 1] + B @ eps[t]
        >>> res = VAR(y, order=1, names=("gdp", "infl")).fit()
        >>> wide = SetIdentifiedSVAR(res, {"gdp": "+", "infl": "+"}, shock="demand").identify()
        >>> wide.impact_bounds.round(3).tolist()
        [[0.0, 1.123], [0.0, 1.042]]
        >>> point = SetIdentifiedSVAR(res, {"gdp": "+"}, zeros={"infl": [0]}, shock="demand")
        >>> point.identify().impact_bounds.round(3).tolist()
        [[0.817, 0.817], [0.0, 0.0]]
        >>> RecursiveSVAR(res, order=("infl", "gdp")).identify().impact.round(3).tolist()
        [[0.771, 0.817], [1.042, 0.0]]
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
        """Validate the source system and the single-shock declaration.

        Args:
            result: The fitted closed reduced-form result to identify.
            restrictions: Mapping from variable name to ``"+"`` or ``"-"``.
            shock: Label for the restricted shock.
            horizons: Non-empty leads at which every declared sign holds.
            zeros: Mapping from variable name to the leads at which its
                response is exactly zero.

        Raises:
            SpecificationError: If the result is not a closed system, the
                sign pattern is empty or names an unknown variable or symbol,
                a horizon or lead is negative, the horizons are empty, or
                the zeros number ``k`` or more.

        Example:
            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> res = VAR(rng.standard_normal((121, 2)), order=1, names=("gdp", "infl")).fit()
            >>> model = SetIdentifiedSVAR(res, {"gdp": "+", "infl": "-"}, horizons=(0, 2))
            >>> model._sign_cells, model._leads, model._zero_cells, model._shock
            (((0, 1.0), (1, -1.0)), (0, 2), (), 'restricted')
            >>> SetIdentifiedSVAR(res, {"gdp": "+"}, zeros={"infl": [0]})._restriction[:76]
            "Declared signs on the 'restricted' shock, holding at horizons (0,): gdp +; z"
            >>> SetIdentifiedSVAR(res, {"gdp": "*"})  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: sign for 'gdp' in shock 'restricted' must ...
            >>> SetIdentifiedSVAR(res, {"gdp": "+"}, horizons=())  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: horizons must be a non-empty collection of ...
            >>> SetIdentifiedSVAR(res, {"gdp": "+"}, zeros={"q": [0]})  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: unknown variable 'q' in zeros; expected ...
            >>> SetIdentifiedSVAR(res, {"gdp": "+"}, zeros={"gdp": [0], "infl": [0]})
            ...     # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: 2 zero restrictions in a 2-variable system ...
        """
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
                    f"unknown variable {variable!r} in zeros; expected one of {self.names}."
                )
            for moment in moments:
                lead = int(moment)
                if lead < 0:
                    raise SpecificationError(
                        f"zero restriction on {variable!r} names a negative lead {moment}."
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
            "; zeros: " + ", ".join(f"{self.names[i]} at lead {h}" for i, h in self._zero_cells)
            if self._zero_cells
            else ""
        )
        self._restriction = (
            f"Declared signs on the {self._shock!r} shock, holding at "
            f"horizons {leads}: "
            + ", ".join(
                f"{self.names[i]} {'+' if sign > 0 else '-'}" for i, sign in self._sign_cells
            )
            + described_zeros
            + ". Every other shock is left entirely unrestricted."
        )

    def identify(self) -> SetIdentifiedSVARResult:
        """Build the identified set's geometry and precompute its faces.

        Takes the Cholesky factor of the innovation covariance, stacks the
        zero-restricted response rows and the sign-restricted ones out of
        the moving-average representation, reduces to the null space of the
        zeros, enumerates the face bases of the sign cone, and evaluates the
        impact bounds once so that an empty set surfaces here rather than
        at the first query.

        Returns:
            The bounds result, with impact bounds ready in its summary and
            arbitrary-horizon bounds available as queries.

        Raises:
            SpecificationError: If the zero restrictions leave no direction,
                or the face count exceeds the enumeration cap.
            NumericalError: If the innovation covariance is not positive
                definite, or the declared restrictions are infeasible -- the
                identified set is empty at this reduced form.

        Example:
            On a system whose responses alternate in sign, requiring one
            variable to rise at both leads zero and one, with the other
            variable's impact pinned to zero, leaves no admissible column;
            and signs on a six-variable system over many horizons exceed
            the face cap:

            >>> import numpy as np
            >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
            >>> rng = np.random.default_rng(0)
            >>> B = np.array([[1.0, 0.5], [0.3, 1.0]])
            >>> y = np.zeros((402, 2))
            >>> for t in range(1, 402):
            ...     y[t] = -0.5 * y[t - 1] + B @ rng.standard_normal(2)
            >>> res = VAR(y, order=1, names=("gdp", "infl")).fit()
            >>> feasible = SetIdentifiedSVAR(res, {"gdp": "+"}, zeros={"infl": [0]}).identify()
            >>> feasible.irf(1)[:, 0, :].round(3).tolist()
            [[0.816, 0.816], [-0.38, -0.38]]
            >>> SetIdentifiedSVAR(
            ...     res, {"gdp": "+"}, zeros={"infl": [0]}, horizons=(0, 1)
            ... ).identify()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.NumericalError: no rotation column satisfies the declared ...
            >>> wide = VAR(rng.standard_normal((300, 6)), order=1).fit()
            >>> SetIdentifiedSVAR(
            ...     wide, {name: "+" for name in wide.names}, horizons=tuple(range(8))
            ... ).identify()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            cultivars.exceptions.SpecificationError: 48 sign restrictions in dimension 6 ...
        """
        k = self.k_endog
        sigma = np.asarray(self.source.sigma_u, dtype=np.float64)
        factor = _lower_cholesky(sigma, "sigma_u")
        top = max((*(h for _, h in self._zero_cells), *self._leads))
        psi = self.source.ma_representation(top)
        loadings = np.einsum("hik,kl->hil", psi, factor)

        equality_rows = (
            np.stack([loadings[h, i] for i, h in self._zero_cells])
            if self._zero_cells
            else np.empty((0, k))
        )
        basis = _null_basis(equality_rows, k)
        inequality_rows = np.stack(
            [sign * loadings[h, i] @ basis for h in self._leads for i, sign in self._sign_cells]
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
